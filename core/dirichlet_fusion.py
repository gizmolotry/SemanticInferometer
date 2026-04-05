"""
dirichlet_fusion.py - Dirichlet Observer Fusion for CLS Embeddings

This module implements observer fusion for semantic manifold analysis.

MODE B ONLY (Map-then-Mix / "Born Aligned"):
All V-observers (bots) are projected through the same M-observer (RKS basis)
BEFORE mixing. This ensures all bots live in the same Hilbert space.

ARCHITECTURE:
    Input: [N, B, H] where B=8 bots, H=768 hidden dim
    Output: [N, D] fused representations + curvature metrics

OBSERVER TAXONOMY:
    V-observers: Bots/framings (B=8) - the input views
    M-observers: RKS basis (Î©, b) - measurement instrument, must be locked
    O-observers: Dirichlet Î± family - the probe distribution to sweep

CRN (Common Random Numbers):
    - Dirichlet weights can be locked across conditions
    - RKS basis must come from kernel_library.ensure_shared_basis()
    - This makes "difference" meaningful, not probe noise

CRITICAL: This module does NOT create RKS basis tensors.
All basis creation MUST go through kernel_library.ensure_shared_basis().
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

RKS_EPSILON = 1e-4 # Epsilon for RKS basis scaling to prevent division by zero


@dataclass
class DirichletFusionConfig:
    """
    Configuration for Dirichlet fusion.
    
    Supports V × M × O observer composition:
    - V-observers: 8 NLI bots (framings)
    - M-observers: Multiple kernel types (RKS bases)
    - O-observers: Dirichlet weight samples
    """
    n_bots: int = 8
    hidden_dim: int = 768
    rks_dim: int = 2048
    n_observers: int = 50
    alpha: float = 1.0
    
    # Kernel configuration - NOW SUPPORTS MULTIPLE
    kernel_type: str = 'rbf'  # Default single kernel (backward compat)
    kernel_types: List[str] = field(default_factory=lambda: ['rbf'])  # Multi-kernel list
    multi_kernel: bool = False  # If True, use kernel_types list
    
    # Basis locking (M-observer control)
    basis_seed: int = 42
    basis_path: Optional[str] = None  # Load pre-generated basis
    
    # CRN: Common Random Numbers
    crn_enabled: bool = True
    locked_dirichlet_weights: Optional[str] = None  # Path to locked weights
    crn_seed: int = 12345
    
    # Sigma policy
    sigma: Optional[float] = None  # If None, estimate from data
    sigma_policy: str = 'median'   # 'median', 'mean', 'fixed'
    nu: float = 1.5                # Matérn smoothness / Student-t df control
    roughness: int = 3             # IMQ Student-t degrees of freedom
    
    # Mode B support (map-then-mix)
    mix_in_rkhs: bool = False  # If True, project to RKHS before mixing
    
    # Sequential Cooling / Atmospheric Annealing settings
    # Production guardrail: keep legacy variance-reinforcement cooling disabled.
    use_sequential_cooling: bool = False
    cooling_iterations: int = 10
    cooling_decay: float = 0.1  # Path decay rate
    cooling_reinforce: float = 1.5  # Reinforcement factor

    # ASTER v3.2: Sequential Annealing (replaces parallel snapshots)
    # Hot stages stay close to the prior barycenter; cold stages release that constraint
    # and allow structured observer separation to emerge.
    use_annealing: bool = True  # If True, use sequential annealing instead of parallel
    annealing_schedule: List[float] = field(default_factory=lambda: [10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    annealing_kl_weight: float = 0.5  # How strongly to penalize deviation from prior state

    # Consensus removal
    # Production guardrail: keep observer-consensus residualization disabled unless
    # explicitly requested for diagnostic experiments.
    remove_consensus: bool = False
    n_consensus_components: int = 1  # How many PCs to remove
    
    # KernelContext integration (optional, for advanced usage)
    kernel_ctx: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Apply KernelContext overrides if provided."""
        # If multi_kernel is True but kernel_types not set, use defaults
        if self.multi_kernel and self.kernel_types == ['rbf']:
            self.kernel_types = ['rbf', 'laplacian', 'polynomial']
        
        # If not multi_kernel, ensure kernel_types matches kernel_type
        if not self.multi_kernel:
            self.kernel_types = [self.kernel_type]
        
        if self.kernel_ctx is not None:
            try:
                from .kernel_context import KernelContext
                if isinstance(self.kernel_ctx, KernelContext):
                    self.kernel_type = self.kernel_ctx.kernel_type
                    self.rks_dim = self.kernel_ctx.rks_dim or self.rks_dim
                    self.basis_seed = self.kernel_ctx.seed
                    if self.kernel_ctx.bandwidth is not None:
                        self.sigma = self.kernel_ctx.bandwidth
            except ImportError:
                pass
    
    def basis_hash(self) -> str:
        """Compute deterministic hash of basis configuration."""
        key = f"{self.hidden_dim}_{self.rks_dim}_{self.basis_seed}_{','.join(self.kernel_types)}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def weights_hash(self, weights: np.ndarray) -> str:
        """Compute hash of Dirichlet weights for provenance."""
        return hashlib.md5(weights.tobytes()).hexdigest()[:12]


class SharedRKSBasis(nn.Module):
    """
    Shared RKS basis that can be saved/loaded for exact reproducibility.
    
    This ensures all bots are projected with the SAME (Î©, b),
    which is required for the shared Hilbert space assumption.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        kernel_type: str = 'rbf',
        nu: float = 1.5,
        roughness: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.kernel_type = kernel_type
        self.nu = nu
        self.roughness = roughness

        # Generate deterministic basis with kernel-appropriate spectral measure
        omega, b = self._sample_basis(input_dim, output_dim, seed, kernel_type)

        self.register_buffer('omega', omega)
        self.register_buffer('b', b)

        # Initialize sigma lazily so the configured estimation policy actually runs.
        self._sigma = None
        self._sigma_diagnostics = {}

        # Compute hash for provenance tracking
        self._hash = self._compute_hash()

    def _sample_basis(self, input_dim, output_dim, seed, kernel_type):
        """
        Sample (Ω, b) from the correct spectral measure for each kernel type.

        Jagged Truth Protocol: Non-Gaussian kernels MUST use fat-tailed distributions.
        - RBF: Gaussian (smooth, legacy baseline)
        - Laplacian: Laplace distribution (sharp)
        - IMQ: Student-t with df=roughness (heavy-tailed, crack-sensitive)
        - Matern: Student-t with df=2*nu (roughness via smoothness parameter)
        - RQ/other: Gaussian fallback
        """
        # Deterministic RNG
        np.random.seed(seed)
        gen = torch.Generator().manual_seed(seed)

        kt = kernel_type.lower()

        if kt == 'laplacian':
            omega = np.random.laplace(0, 1.0, size=(input_dim, output_dim))
            omega = torch.FloatTensor(omega)
        elif kt == 'imq':
            df = max(1, self.roughness)
            omega = np.random.standard_t(df, size=(input_dim, output_dim))
            # Variance matching: normalize to unit variance (Holographic Axiom 1)
            if df > 2:
                omega = omega * np.sqrt((df - 2) / df)
            omega = torch.FloatTensor(omega)
        elif kt == 'matern':
            if self.nu == 0.5:
                omega = np.random.laplace(0, 1.0, size=(input_dim, output_dim))
            else:
                df = max(1, 2 * self.nu)
                omega = np.random.standard_t(df, size=(input_dim, output_dim))
                # Variance matching first (Holographic Axiom 1)
                if df > 2:
                    omega = omega * np.sqrt((df - 2) / df)
                # Then spectral scaling for Matérn
                omega = omega * np.sqrt(2 * self.nu)
            omega = torch.FloatTensor(omega)
        else:
            # RBF, RQ, and fallback: Gaussian
            omega = torch.randn(input_dim, output_dim, generator=gen)

        b = torch.rand(output_dim, generator=gen) * 2 * np.pi

        return omega, b

    def _compute_hash(self) -> str:
        """Compute hash of basis for provenance tracking."""
        data = torch.cat([self.omega.flatten(), self.b]).cpu().numpy()
        return hashlib.md5(data.tobytes()).hexdigest()[:12]
    
    @property
    def basis_hash(self) -> str:
        return self._hash
    
    def set_sigma(self, sigma: float):
        """Set the kernel bandwidth."""
        self._sigma = sigma
    
    def estimate_sigma(self, X: torch.Tensor, policy: str = 'median') -> float:
        """
        Estimate bandwidth from data using pairwise distances.
        
        Also saves diagnostic information to self._sigma_diagnostics for
        information preservation.
        """
        with torch.no_grad():
            # X is [N, B, H] - flatten to [N*B, H]
            if X.dim() == 3:
                N, B, H = X.shape
                flat = X.reshape(N * B, H)
            elif X.dim() == 2:
                flat = X
                H = X.shape[-1]
            else:
                print(f"[SharedRKSBasis] WARNING: Unexpected input shape {X.shape}")
                self._sigma_diagnostics = {'error': f'unexpected_shape_{X.shape}'}
                return 1.0
            
            # Subsample for efficiency
            n = min(1000, flat.shape[0])
            idx = torch.randperm(flat.shape[0], device=flat.device)[:n]
            X_sub = flat[idx].float()
            
            # Compute input statistics
            input_mean = float(X_sub.mean().item())
            input_std = float(X_sub.std().item())
            input_norm_mean = float(X_sub.norm(dim=-1).mean().item())
            
            # Check for degenerate input
            if input_std < 1e-10:
                print(f"[SharedRKSBasis] WARNING: Input has near-zero variance!")
                print(f"[SharedRKSBasis] Input stats: mean={input_mean:.6f}, std={input_std:.6f}")
                self._sigma_diagnostics = {
                    'error': 'near_zero_variance',
                    'input_mean': input_mean,
                    'input_std': input_std,
                    'input_shape': list(X.shape),
                }
                return 1.0
            
            # Compute pairwise distances
            dists = torch.cdist(X_sub, X_sub)
            mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            
            if dists[mask].numel() == 0:
                self._sigma_diagnostics = {'error': 'no_pairs'}
                return 1.0
            
            pairwise_dists = dists[mask]
            
            # Compute distance statistics
            dist_min = float(pairwise_dists.min().item())
            dist_median = float(torch.median(pairwise_dists).item())
            dist_mean = float(pairwise_dists.mean().item())
            dist_max = float(pairwise_dists.max().item())
            dist_std = float(pairwise_dists.std().item())
            
            if policy == 'median':
                raw_sigma = dist_median
            elif policy == 'mean':
                raw_sigma = dist_mean
            else:
                raw_sigma = 1.0
            
            # Debug info
            print(f"[SharedRKSBasis] Input shape: {X.shape}, subsample: {X_sub.shape}")
            print(f"[SharedRKSBasis] Input stats: mean={input_mean:.4f}, std={input_std:.4f}")
            print(f"[SharedRKSBasis] Distance stats: min={dist_min:.4f}, "
                  f"median={dist_median:.4f}, max={dist_max:.4f}")
            
            # Ensure reasonable sigma (between 0.1 and 1000)
            sigma_clamped = False
            original_sigma = raw_sigma
            sigma = max(raw_sigma, 0.1)
            sigma = min(sigma, 1000.0)
            if sigma != original_sigma:
                sigma_clamped = True
                print(f"[SharedRKSBasis] Sigma clamped from {original_sigma:.4f} to {sigma:.4f}")
            
            # Save comprehensive diagnostics for information preservation
            self._sigma_diagnostics = {
                'estimated_sigma': sigma,
                'original_sigma': original_sigma,
                'sigma_clamped': sigma_clamped,
                'policy': policy,
                'input_shape': list(X.shape),
                'subsample_size': n,
                'n_pairs': int(pairwise_dists.numel()),
                'input_stats': {
                    'mean': input_mean,
                    'std': input_std,
                    'norm_mean': input_norm_mean,
                },
                'distance_stats': {
                    'min': dist_min,
                    'median': dist_median,
                    'mean': dist_mean,
                    'max': dist_max,
                    'std': dist_std,
                },
            }
            
            return sigma
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project to RKHS using Random Kitchen Sinks.
        
        Args:
            X: [..., input_dim] input features
            
        Returns:
            [..., output_dim] RKS features
        """
        if self._sigma is None:
            raise ValueError("Sigma not set. Call set_sigma() or estimate_sigma() first.")
        
        # Scale omega by sigma
        omega_scaled = self.omega.to(X.device) / (self._sigma + RKS_EPSILON)
        b = self.b.to(X.device)
        
        # Project: Ï†(x) = sqrt(2/D) * cos(x @ Ï‰/Ïƒ + b)
        proj = X @ omega_scaled + b
        scale = np.sqrt(2.0 / self.output_dim)
        return scale * torch.cos(proj)
    
    def save(self, path: str):
        """Save basis for reuse."""
        torch.save({
            'omega': self.omega.cpu(),
            'b': self.b.cpu(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seed': self.seed,
            'kernel_type': self.kernel_type,
            'sigma': self._sigma,
            'hash': self._hash,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'SharedRKSBasis':
        """Load pre-generated basis."""
        data = torch.load(path, map_location='cpu')
        basis = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            seed=data['seed'],
            kernel_type=data.get('kernel_type', 'rbf'),
        )
        basis.omega.copy_(data['omega'])
        basis.b.copy_(data['b'])
        basis._sigma = data.get('sigma')
        basis._hash = data.get('hash', basis._compute_hash())
        return basis


class DirichletFusion(nn.Module):
    """
    Dirichlet observer fusion for CLS embeddings.
    
    Implements full V × M × O observer composition:
    - V-observers: 8 NLI bots (framings) - the input views
    - M-observers: Multiple kernel types (RKS bases) - measurement instruments
    - O-observers: Dirichlet weight samples - the probe distribution
    
    KEY FEATURES:
    1. Multi-kernel: Project through RBF, Laplacian, Polynomial simultaneously
    2. Basis locking: RKS basis (Ω, b) is deterministic and logged
    3. CRN support: Dirichlet weights can be pre-generated and shared
    4. Curvature metrics: per-item "framing hull complexity"
    5. Variance decomposition: V, M, O variance contributions
    6. Consensus removal: Optional PCA-based consensus subtraction
    7. Sequential Cooling: Optional iterative Atmospheric Annealing algorithm
    8. Provenance: basis_hash, sigma, weights_hash all tracked
    """
    
    def __init__(self, config: DirichletFusionConfig = None):
        super().__init__()
        
        if config is None:
            config = DirichletFusionConfig()
        self.config = config
        
        # Initialize provenance FIRST
        self._provenance = {
            'basis_seed': config.basis_seed,
            'rks_dim': config.rks_dim,
            'hidden_dim': config.hidden_dim,
            'kernel_types': config.kernel_types,
            'multi_kernel': config.multi_kernel,
            'crn_enabled': config.crn_enabled,
            'alpha': config.alpha,
            'n_observers': config.n_observers,
            'use_sequential_cooling': config.use_sequential_cooling,
            'remove_consensus': config.remove_consensus,
            'thermodynamic_objective': {
                'hot_stage': 'strong prior regularization, weak separation pressure',
                'cold_stage': 'weak prior regularization, strong separation pressure',
                'prior_reference': 'simplex weights',
            },
        }
        
        # Initialize multiple RKS bases (M-observers)
        self.bases = nn.ModuleDict()
        self._sigma_estimated = {}
        
        for kernel_type in config.kernel_types:
            basis = SharedRKSBasis(
                input_dim=config.hidden_dim,
                output_dim=config.rks_dim,
                seed=config.basis_seed,
                kernel_type=kernel_type,
                nu=config.nu,
                roughness=config.roughness,
            )
            self.bases[kernel_type] = basis
            self._sigma_estimated[kernel_type] = False
            print(f"[DirichletFusion] Created {kernel_type} basis, hash: {basis.basis_hash}")
        
        # For backward compatibility, keep single .basis reference
        self.basis = self.bases[config.kernel_types[0]]
        
        self._provenance['basis_hashes'] = {k: b.basis_hash for k, b in self.bases.items()}
        
        # Load or generate Dirichlet weights (CRN)
        self._locked_weights: Optional[torch.Tensor] = None
        self._init_crn_weights(config)
        
        # Sequential cooling network weights (initialized if used)
        self._cooling_weights: Optional[torch.Tensor] = None
    
    def _init_basis(self, config: DirichletFusionConfig) -> SharedRKSBasis:
        """Initialize or load the shared RKS basis."""
        if config.basis_path and Path(config.basis_path).exists():
            basis = SharedRKSBasis.load(config.basis_path)
            print(f"[DirichletFusion] Loaded basis from {config.basis_path}")
        else:
            basis = SharedRKSBasis(
                input_dim=config.hidden_dim,
                output_dim=config.rks_dim,
                seed=config.basis_seed,
                kernel_type=config.kernel_type,
                nu=config.nu,
                roughness=config.roughness,
            )
            print(f"[DirichletFusion] Created basis with seed {config.basis_seed}")
        
        print(f"[DirichletFusion] Basis hash: {basis.basis_hash}")
        return basis
    
    def _init_crn_weights(self, config: DirichletFusionConfig):
        """Initialize Dirichlet weights for CRN."""
        if config.locked_dirichlet_weights and Path(config.locked_dirichlet_weights).exists():
            # Load pre-generated weights
            data = np.load(config.locked_dirichlet_weights)
            self._locked_weights = torch.from_numpy(data['weights']).float()
            weights_hash = config.weights_hash(data['weights'])
            self._provenance['weights_hash'] = weights_hash
            print(f"[DirichletFusion] Loaded CRN weights from {config.locked_dirichlet_weights}")
            print(f"[DirichletFusion] Weights hash: {weights_hash}")
        elif config.crn_enabled:
            # Generate and cache
            torch.manual_seed(config.crn_seed)
            alpha_vec = torch.full((config.n_bots,), config.alpha)
            dirichlet = torch.distributions.Dirichlet(alpha_vec)
            self._locked_weights = dirichlet.sample((config.n_observers,))
            weights_hash = config.weights_hash(self._locked_weights.numpy())
            self._provenance['weights_hash'] = weights_hash
            self._provenance['crn_seed'] = config.crn_seed
            print(f"[DirichletFusion] Generated CRN weights with seed {config.crn_seed}")
    
    def save_basis(self, path: str):
        """Save the RKS basis for reuse across conditions."""
        self.basis.save(path)
        print(f"[DirichletFusion] Saved basis to {path}")
    
    def save_weights(self, path: str):
        """Save Dirichlet weights for exact reproducibility."""
        if self._locked_weights is not None:
            np.savez(path, 
                weights=self._locked_weights.numpy(),
                alpha=self.config.alpha,
                n_observers=self.config.n_observers,
                n_bots=self.config.n_bots,
                crn_seed=self.config.crn_seed,
            )
            print(f"[DirichletFusion] Saved CRN weights to {path}")
    
    def forward(
        self,
        cls_per_bot: torch.Tensor,
        return_samples: bool = False,
        compute_curvature: bool = True,
        diagnostics: Optional[Any] = None,  # PipelineDiagnostics instance
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse bot CLS embeddings via Dirichlet-weighted mixtures.
        
        MODE B ONLY: Map to RKHS first, then mix ("born aligned")
        
        Args:
            cls_per_bot: [N, B, H] V-observer embeddings
            return_samples: whether to return all K observer samples
            compute_curvature: whether to compute curvature metrics
            diagnostics: optional PipelineDiagnostics for recording
            
        Returns:
            - 'fused': [N, D] mean fused representation
            - 'fused_std': [N, D] std across O-observers
            - 'curvature': dict with per-item metrics (if compute_curvature)
            - 'observer_samples': [N, K, D] all samples (if return_samples)
            - 'bot_rkhs': [N, B, D] per-bot RKHS features
            - 'rkhs_views': [N, B, D] same as bot_rkhs (for artifact saving)
            - 'provenance': metadata dict
        """
        N, B, H = cls_per_bot.shape
        device = cls_per_bot.device
        
        # Record raw CLS for diagnostics
        if diagnostics is not None:
            diagnostics.record_stage('raw_cls', cls_per_bot)
        
        # =====================================================================
        # STEP 1: Estimate sigma for ALL kernels
        # =====================================================================
        for k_name, basis in self.bases.items():
            if basis._sigma is None:
                sigma = basis.estimate_sigma(cls_per_bot, self.config.sigma_policy)
                basis.set_sigma(sigma)
                self._sigma_estimated[k_name] = True
                print(f"[DirichletFusion] {k_name} sigma = {sigma:.4f}")
        
        self._provenance['sigmas'] = {k: b._sigma for k, b in self.bases.items()}
        
        # =====================================================================
        # STEP 2: Get Dirichlet weights (CRN or fresh sample)
        # =====================================================================
        if self._locked_weights is not None:
            weights = self._locked_weights.to(device)
        else:
            alpha_vec = torch.full((B,), self.config.alpha, device=device)
            dirichlet = torch.distributions.Dirichlet(alpha_vec)
            weights = dirichlet.sample((self.config.n_observers,))
        
        K_o = weights.shape[0]  # O-observers (Dirichlet samples)
        K_m = len(self.bases)   # M-observers (kernels)
        K_v = B                  # V-observers (bots)
        D = self.config.rks_dim
        
        # =====================================================================
        # STEP 3: Project through ALL kernels (M-observers)
        # =====================================================================
        batch_size = 5000
        n_batches = (N + batch_size - 1) // batch_size
        
        print(f"[DirichletFusion] V×M×O composition: {K_v} bots × {K_m} kernels × {K_o} observers")
        print(f"[DirichletFusion] Processing {N} articles in {n_batches} batches...")
        
        # Storage for per-kernel results
        phi_per_kernel = {k: [] for k in self.bases.keys()}
        samples_per_kernel = {k: [] for k in self.bases.keys()}
        
        for batch_idx, start in enumerate(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            batch = cls_per_bot[start:end]  # [batch, B, H]
            batch_n = batch.shape[0]
            
            # Project through EACH kernel
            for k_name, basis in self.bases.items():
                flat = batch.reshape(batch_n * B, H)
                phi_flat = basis(flat)
                phi = phi_flat.reshape(batch_n, B, -1)  # [batch, B, D]
                
                phi_per_kernel[k_name].append(phi.cpu())
                
                # Fuse with Dirichlet weights for this kernel
                # phi: [batch, B, D], weights: [K_o, B] -> samples: [batch, K_o, D]
                batch_samples = torch.einsum('nbd,kb->nkd', phi, weights)
                samples_per_kernel[k_name].append(batch_samples.cpu())
                
                del phi_flat, flat, phi
            
            del batch
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                print(f"  Batch {batch_idx + 1}/{n_batches} done")
        
        # Concatenate per-kernel results
        phi_per_kernel = {k: torch.cat(v, dim=0) for k, v in phi_per_kernel.items()}
        samples_per_kernel = {k: torch.cat(v, dim=0) for k, v in samples_per_kernel.items()}
        
        # =====================================================================
        # STEP 4: Optional consensus residualization (diagnostic only; disabled in production)
        # =====================================================================
        if self.config.remove_consensus:
            print("[DirichletFusion] Diagnostic consensus residualization enabled (robust median field)...")
            phi_per_kernel = self._remove_consensus(phi_per_kernel)
            # Re-fuse after consensus removal
            for k_name in samples_per_kernel.keys():
                phi = phi_per_kernel[k_name].to(device)
                samples_per_kernel[k_name] = torch.einsum('nbd,kb->nkd', phi, weights).cpu()
        
        # =====================================================================
        # STEP 5: Optional sequential cooling adaptive weighting (legacy diagnostic path)
        # =====================================================================
        if self.config.use_sequential_cooling:
            print(f"[DirichletFusion] Running Sequential Cooling ({self.config.cooling_iterations} iterations)...")
            self._cooling_weights = self._sequential_cooling(
                phi_per_kernel, 
                n_iterations=self.config.cooling_iterations
            )
            # Re-fuse with sequential cooling weights
            samples_per_kernel = self._apply_cooling_weights(phi_per_kernel, samples_per_kernel)
        
        # =====================================================================
        # STEP 6: Combine across kernels for final V×M×O representation
        # =====================================================================
        # Stack all kernel samples: [N, K_m, K_o, D]
        kernel_names = list(samples_per_kernel.keys())
        all_samples = torch.stack([samples_per_kernel[k] for k in kernel_names], dim=1)
        # all_samples shape: [N, K_m, K_o, D]
        
        # Reshape to [N, K_m * K_o, D] for compatibility
        N_out = all_samples.shape[0]
        all_samples_flat = all_samples.reshape(N_out, K_m * K_o, -1)
        
        # Final fused representation: mean across all V×M×O observers
        fused_mean = all_samples_flat.mean(dim=1)  # [N, D]
        fused_std = all_samples_flat.std(dim=1)    # [N, D]
        
        # =====================================================================
        # STEP 7: Compute V×M×O variance decomposition
        # =====================================================================
        variance_decomp = self._decompose_vmo_variance(
            phi_per_kernel=phi_per_kernel,
            samples_per_kernel=samples_per_kernel,
            weights=weights,
        )
        print(f"[DirichletFusion] Variance decomposition:")
        print(f"  V-variance (bots): {variance_decomp['v_fraction']:.1%}")
        print(f"  M-variance (kernels): {variance_decomp['m_fraction']:.1%}")
        print(f"  O-variance (Dirichlet): {variance_decomp['o_fraction']:.1%}")
        print(f"  Interaction: {variance_decomp['interaction_fraction']:.1%}")
        
        # =====================================================================
        # STEP 8: Build result dict
        # =====================================================================
        # Use first kernel's phi for backward compatibility
        bot_rkhs = phi_per_kernel[kernel_names[0]]
        
        result = {
            'fused': fused_mean,
            'fused_std': fused_std,
            'bot_rkhs': bot_rkhs,
            'rkhs_views': bot_rkhs,
            
            # Multi-kernel outputs
            'phi_per_kernel': phi_per_kernel,
            'samples_per_kernel': samples_per_kernel,
            'all_samples': all_samples_flat,
            
            # Variance decomposition
            'variance_decomposition': variance_decomp,
            
            # Provenance
            'provenance': self._provenance.copy(),
            'sigma_diagnostics': {k: getattr(b, '_sigma_diagnostics', {}) for k, b in self.bases.items()},
        }
        
        # Add sequential cooling weights if used
        if self._cooling_weights is not None:
            result['cooling_weights'] = self._cooling_weights
        
        if return_samples:
            result['observer_samples_shape'] = (N, K_m, K_o, D)
        
        # =====================================================================
        # STEP 9: Compute curvature if requested
        # =====================================================================
        if compute_curvature:
            curvature_pr_list = []
            curvature_er_list = []
            curvature_lr_list = []
            
            for batch_idx, start in enumerate(range(0, N, batch_size)):
                end = min(start + batch_size, N)
                batch_samples = all_samples_flat[start:end].to(device)
                pr, er, lr = self._compute_curvature_batch(batch_samples)
                curvature_pr_list.append(pr)
                curvature_er_list.append(er)
                curvature_lr_list.append(lr)
            
            result['curvature'] = {
                'participation_ratio': torch.cat(curvature_pr_list, dim=0),
                'effective_rank_90': torch.cat(curvature_er_list, dim=0),
                'lambda1_lambda2': torch.cat(curvature_lr_list, dim=0),
            }
            print(f"[DirichletFusion] Curvature computed for {N} articles")
        
        # Record for diagnostics
        if diagnostics is not None:
            diagnostics.record_stage('fused', fused_mean)
            diagnostics.record_stage('fused_std', fused_std)
            diagnostics.record_metadata('variance_decomposition', variance_decomp)
            diagnostics.record_metadata('provenance', self._provenance)
        
        return result
    
    def _remove_consensus(self, phi_per_kernel: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Diagnostic-only consensus residualization using a robust shared field.

        The production path keeps this disabled. When explicitly enabled for audits,
        it subtracts a geometric-median shared field rather than a PCA disagreement axis.
        """
        return self._remove_consensus_pca(
            phi_per_kernel,
            n_components=self.config.n_consensus_components,
        )
    
    def _sequential_cooling(
        self, 
        phi_per_kernel: Dict[str, torch.Tensor],
        n_iterations: int = 10,
    ) -> torch.Tensor:
        """
        Atmospheric Annealing-inspired adaptive network optimization.
        
        Iteratively adjusts weights for (kernel, bot) paths based on
        how much unique variance each path contributes.
        
        Paths with high unique variance get reinforced.
        Paths with low contribution get pruned.
        
        Args:
            phi_per_kernel: {kernel_name: [N, B, D]} RKHS projections
            n_iterations: Number of refinement iterations
            
        Returns:
            weights: [K_m, B] adaptive weights for (kernel, bot) combinations
        """
        kernel_names = list(phi_per_kernel.keys())
        K_m = len(kernel_names)
        B = next(iter(phi_per_kernel.values())).shape[1]
        
        # Initialize uniform weights across all (kernel, bot) paths
        weights = torch.ones(K_m, B) / (K_m * B)
        
        decay = self.config.cooling_decay
        reinforce = self.config.cooling_reinforce
        
        for iteration in range(n_iterations):
            # Compute "flow" through each (kernel, bot) path
            flow = torch.zeros(K_m, B)
            
            for k_idx, k_name in enumerate(kernel_names):
                phi = phi_per_kernel[k_name]  # [N, B, D]
                N, _, D = phi.shape
                
                for b in range(B):
                    # Unique variance: how different is this bot from mean of others?
                    bot_repr = phi[:, b, :]  # [N, D]
                    other_mean = torch.cat([phi[:, :b, :], phi[:, b+1:, :]], dim=1).mean(dim=1)  # [N, D]
                    
                    # Difference from consensus
                    diff = bot_repr - other_mean
                    unique_var = diff.var().item()
                    
                    # Also consider absolute variance (information content)
                    abs_var = bot_repr.var().item()
                    
                    # Flow = unique variance * absolute variance
                    flow[k_idx, b] = np.sqrt(unique_var * abs_var)
            
            # Normalize flow
            flow = flow / (flow.sum() + 1e-10)
            
            # Update weights: reinforce high-flow paths, decay low-flow
            weights = weights * (1 - decay) + flow * reinforce
            
            # Prune near-zero paths
            weights[weights < 0.001] = 0
            
            # Renormalize
            weights = weights / (weights.sum() + 1e-10)
            
            if (iteration + 1) % 3 == 0:
                active_paths = (weights > 0.001).sum().item()
                print(f"  Iteration {iteration + 1}: {active_paths}/{K_m * B} active paths")
        
        return weights
    
    def _apply_cooling_weights(
        self,
        phi_per_kernel: Dict[str, torch.Tensor],
        samples_per_kernel: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply sequential cooling weights to re-weight bot contributions per kernel.

        Instead of uniform Dirichlet, uses adaptive weights from Atmospheric Annealing optimization.
        """
        if self._cooling_weights is None:
            return samples_per_kernel
        
        kernel_names = list(phi_per_kernel.keys())
        result = {}
        
        for k_idx, k_name in enumerate(kernel_names):
            phi = phi_per_kernel[k_name]  # [N, B, D]
            N, B, D = phi.shape
            
            # Get cooling weights for this kernel
            kernel_weights = self._cooling_weights[k_idx]  # [B]
            
            # Normalize to sum to 1
            kernel_weights = kernel_weights / (kernel_weights.sum() + 1e-10)
            
            # Apply as single "optimal" observer view
            # fused = Σ_b w_b * φ_b
            fused = torch.einsum('nbd,b->nd', phi, kernel_weights)
            
            # Expand to match expected shape [N, K_o, D] by repeating
            K_o = samples_per_kernel[k_name].shape[1]
            result[k_name] = fused.unsqueeze(1).expand(-1, K_o, -1)
        
        return result
    
    def _decompose_vmo_variance(
        self,
        phi_per_kernel: Dict[str, torch.Tensor],
        samples_per_kernel: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Decompose total variance into V, M, O components.
        
        Uses ANOVA-style decomposition:
        - V-variance: variance across bots (holding kernel, weights fixed)
        - M-variance: variance across kernels (holding bots, weights fixed)
        - O-variance: variance across Dirichlet samples (holding bots, kernels fixed)
        - Interaction: remaining variance
        
        Returns:
            Dict with variance components and fractions
        """
        kernel_names = list(samples_per_kernel.keys())
        K_m = len(kernel_names)
        
        # Stack: [N, K_m, K_o, D]
        all_samples = torch.stack([samples_per_kernel[k] for k in kernel_names], dim=1)
        N, _, K_o, D = all_samples.shape
        
        # Total variance
        total_var = all_samples.var().item()
        
        if total_var < 1e-10:
            return {
                'total_variance': 0.0,
                'v_variance': 0.0, 'v_fraction': 0.0,
                'm_variance': 0.0, 'm_fraction': 0.0,
                'o_variance': 0.0, 'o_fraction': 0.0,
                'interaction_variance': 0.0, 'interaction_fraction': 0.0,
            }
        
        # V-variance: across bots
        # For this, we need phi_per_kernel which has bot dimension
        # Average across kernels to isolate bot effect
        phi_stacked = torch.stack([phi_per_kernel[k] for k in kernel_names], dim=1)  # [N, K_m, B, D]
        v_mean = phi_stacked.mean(dim=1)  # Average over kernels: [N, B, D]
        v_var = v_mean.var().item()
        
        # M-variance: across kernels
        # Average over bots and O-observers: [N, K_m, D]
        m_mean = all_samples.mean(dim=2)  # Average over O-observers
        m_var = m_mean.var(dim=1).mean().item()  # Variance across kernels
        
        # O-variance: across Dirichlet samples
        # Average over kernels: [N, K_o, D]
        o_mean = all_samples.mean(dim=1)  # Average over kernels
        o_var = o_mean.var(dim=1).mean().item()  # Variance across O-observers
        
        # Interaction (residual)
        interaction_var = max(0, total_var - (v_var + m_var + o_var))
        
        return {
            'total_variance': total_var,
            'v_variance': v_var,
            'v_fraction': v_var / total_var,
            'm_variance': m_var,
            'm_fraction': m_var / total_var,
            'o_variance': o_var,
            'o_fraction': o_var / total_var,
            'interaction_variance': interaction_var,
            'interaction_fraction': interaction_var / total_var,
        }
    
    def _compute_curvature_batch(self, observer_samples: torch.Tensor):
        """
        Compute curvature metrics for a batch on GPU.
        
        Args:
            observer_samples: [batch, K, D]
            
        Returns:
            participation_ratio, effective_rank_90, lambda1_lambda2 (all [batch])
        """
        batch_n, K, D = observer_samples.shape
        
        # Center samples
        centered = observer_samples - observer_samples.mean(dim=1, keepdim=True)
        
        # Compute covariance: [batch, D, D] - too big, use SVD on samples directly
        # samples: [batch, K, D], do SVD per item
        # Use batched SVD approximation via eigenvalues of Gram matrix
        # Gram: [batch, K, K] = samples @ samples.T
        gram = torch.bmm(centered, centered.transpose(1, 2))  # [batch, K, K]
        
        # Eigenvalues of Gram matrix (sorted descending)
        eigenvalues = torch.linalg.eigvalsh(gram)  # [batch, K]
        eigenvalues = eigenvalues.flip(dims=[-1])  # Descending
        eigenvalues = eigenvalues.clamp(min=0)  # Numerical safety
        
        # Participation ratio: (sum Î»)Â² / sum(Î»Â²)
        sum_eig = eigenvalues.sum(dim=-1)
        sum_eig_sq = (eigenvalues ** 2).sum(dim=-1)
        participation_ratio = (sum_eig ** 2) / (sum_eig_sq + 1e-10)
        
        # Effective rank at 90% variance
        total_var = sum_eig
        cumsum = eigenvalues.cumsum(dim=-1)
        threshold = 0.9 * total_var.unsqueeze(-1)
        effective_rank_90 = (cumsum < threshold).sum(dim=-1).float() + 1
        
        # Lambda1 / Lambda2 ratio (dominance)
        lambda1 = eigenvalues[:, 0]
        lambda2 = eigenvalues[:, 1].clamp(min=1e-10)
        lambda_ratio = lambda1 / lambda2
        
        return participation_ratio, effective_rank_90, lambda_ratio
    
    def get_provenance(self) -> Dict[str, Any]:
        """Get full provenance for artifact logging."""
        return {
            **self._provenance,
            'sigma': self.basis._sigma,
        }
    
    # =========================================================================
    # MULTI-KERNEL V × M × O FORWARD (FULL OBSERVER COMPOSITION)
    # =========================================================================
    
    def forward_vmo(
        self,
        cls_per_bot: torch.Tensor,
        return_samples: bool = False,
        compute_curvature: bool = True,
        compute_variance_decomposition: bool = True,
        remove_consensus: bool = None,
        use_sequential_cooling: bool = None,
    ) -> Dict[str, Any]:
        """
        Full V × M × O observer composition with multiple kernels.
        
        This is the complete implementation:
        - V-observers: 8 NLI bots (framings)
        - M-observers: Multiple kernel types (RBF, Laplacian, etc.)
        - O-observers: Dirichlet weight samples
        
        Args:
            cls_per_bot: [N, B, H] V-observer embeddings
            return_samples: whether to return all samples
            compute_curvature: whether to compute curvature metrics
            compute_variance_decomposition: whether to decompose V/M/O variance
            remove_consensus: override config.remove_consensus
            use_sequential_cooling: override config.use_sequential_cooling
            
        Returns:
            - 'fused': [N, D] mean fused representation
            - 'fused_std': [N, D] std across all observers
            - 'phi_per_kernel': dict of {kernel: [N, B, D]} RKHS projections
            - 'all_samples': [N, K_m * K_o, D] all observer samples (if return_samples)
            - 'variance_decomposition': V/M/O variance fractions
            - 'cooling_weights': [K_m, B] adaptive weights (if use_sequential_cooling)
            - 'provenance': metadata dict
        """
        N, B, H = cls_per_bot.shape
        device = cls_per_bot.device
        
        # Use config defaults if not overridden
        if remove_consensus is None:
            remove_consensus = self.config.remove_consensus
        if use_sequential_cooling is None:
            use_sequential_cooling = self.config.use_sequential_cooling
        
        K_m = len(self.bases)  # Number of M-observers (kernels)
        K_o = self.config.n_observers  # Number of O-observers (Dirichlet)
        D = self.config.rks_dim
        
        print(f"\n[VMO] Full V × M × O composition:")
        print(f"  V-observers (bots): {B}")
        print(f"  M-observers (kernels): {K_m} ({list(self.bases.keys())})")
        print(f"  O-observers (Dirichlet): {K_o}")
        print(f"  Total observer combinations: {B} × {K_m} × {K_o} = {B * K_m * K_o}")
        
        # Step 1: Project through ALL kernels (M-observers)
        phi_per_kernel = {}
        for kernel_name, basis in self.bases.items():
            # Estimate sigma if not set
            if basis._sigma is None:
                sigma = basis.estimate_sigma(cls_per_bot, self.config.sigma_policy)
                basis.set_sigma(sigma)
                print(f"  [{kernel_name}] sigma = {sigma:.4f}")
            
            # Project
            flat = cls_per_bot.reshape(N * B, H)
            phi_flat = basis(flat)
            phi = phi_flat.reshape(N, B, D)
            phi_per_kernel[kernel_name] = phi
        
        # Step 2: Optional consensus removal (subtract what observers agree on)
        if remove_consensus:
            print(f"  Removing {self.config.n_consensus_components} consensus component(s)...")
            phi_per_kernel = self._remove_consensus_pca(
                phi_per_kernel, 
                n_components=self.config.n_consensus_components
            )
        
        # Step 3: Get Dirichlet weights
        if self._locked_weights is not None:
            dirichlet_weights = self._locked_weights.to(device)
        else:
            alpha_vec = torch.full((B,), self.config.alpha, device=device)
            dirichlet = torch.distributions.Dirichlet(alpha_vec)
            dirichlet_weights = dirichlet.sample((K_o,))  # [K_o, B]
        
        # Step 4: Optionally run sequential cooling to get adaptive kernel weights
        if use_sequential_cooling:
            print(f"  Running sequential cooling ({self.config.cooling_iterations} iterations)...")
            kernel_weights = self._sequential_cooling_adapt(
                phi_per_kernel, 
                n_iterations=self.config.cooling_iterations
            )
            self._cooling_weights = kernel_weights
        else:
            # Uniform kernel weights
            kernel_weights = torch.ones(K_m, device=device) / K_m
            self._cooling_weights = kernel_weights
        
        # Step 5: Fuse across V × M × O
        # For each kernel m, for each Dirichlet sample o:
        #   sample_{m,o}(n) = Σ_b w_{o,b} * φ_m(bot_b(n))
        all_samples = []  # Will be [N, K_m * K_o, D]
        
        for k_idx, (kernel_name, phi) in enumerate(phi_per_kernel.items()):
            # phi: [N, B, D]
            # dirichlet_weights: [K_o, B]
            # samples: [N, K_o, D]
            samples_k = torch.einsum('nbd,kb->nkd', phi.to(device), dirichlet_weights)
            
            # Weight by sequential cooling kernel weights
            samples_k = samples_k * kernel_weights[k_idx]
            
            all_samples.append(samples_k)
        
        # Concatenate: [N, K_m * K_o, D]
        all_samples = torch.cat(all_samples, dim=1)
        
        # Step 6: Compute fused output
        fused_mean = all_samples.mean(dim=1)  # [N, D]
        fused_std = all_samples.std(dim=1)    # [N, D]
        
        result = {
            'fused': fused_mean,
            'fused_std': fused_std,
            'phi_per_kernel': {k: v.cpu() for k, v in phi_per_kernel.items()},
            'provenance': self._provenance.copy(),
            'kernel_weights': kernel_weights.cpu() if torch.is_tensor(kernel_weights) else kernel_weights,
        }
        
        if return_samples:
            result['all_samples'] = all_samples.cpu()
            result['all_samples_shape'] = f"[N={N}, K_m*K_o={K_m}*{K_o}={K_m*K_o}, D={D}]"
        
        # Step 7: Variance decomposition
        if compute_variance_decomposition:
            var_decomp = self._decompose_vmo_variance(
                phi_per_kernel, dirichlet_weights, kernel_weights, device
            )
            result['variance_decomposition'] = var_decomp
            print(f"\n  Variance decomposition:")
            print(f"    V (bots):     {var_decomp['v_fraction']:.1%}")
            print(f"    M (kernels):  {var_decomp['m_fraction']:.1%}")
            print(f"    O (Dirichlet):{var_decomp['o_fraction']:.1%}")
            print(f"    Interaction:  {var_decomp['interaction_fraction']:.1%}")
        
        # Step 8: Curvature metrics
        if compute_curvature:
            pr, er, lr = self._compute_curvature_batch(all_samples)
            result['curvature'] = {
                'participation_ratio': pr.cpu(),
                'effective_rank_90': er.cpu(),
                'lambda1_lambda2': lr.cpu(),
            }
        
        return result

    # =========================================================================
    # ASTER v3.2: SEQUENTIAL ANNEALING (Hysteretic Fusion)
    # =========================================================================

    def forward_annealing(
        self,
        cls_per_bot: torch.Tensor,
        compute_curvature: bool = True,
    ) -> Dict[str, Any]:
        """
        Sequential Annealing: Hot-to-Cold Hysteretic Fusion.

        Instead of parallel snapshots at different alphas, we:
        1. Start HOT (high alpha = consensus/liquid state)
        2. Cool sequentially (low alpha = partisan/solid state)
        3. Pass weights between stages (hysteresis/memory)
        4. Measure stability: did the bonds hold during cooling?

        The "Atmosphere" (Fog) is the stability of this cooling process.
        High stability = Crystal (real consensus), Low stability = Fog (fake consensus).

        Args:
            cls_per_bot: [N, B, H] V-observer embeddings
            compute_curvature: whether to compute curvature metrics

        Returns:
            - 'fused': [N, D] final cold-state embedding
            - 'fused_std': [N, D] uncertainty from annealing
            - 'consensus_embedding': [N, D] hot-state embedding (initial consensus)
            - 'stability_mask': [N] per-article stability (1.0=Crystal, 0.0=Fog)
            - 'atmospheric_alpha': scalar mean stability
            - 'adaptive_work': total work done during annealing
            - 'weight_trajectory': list of [N, B] weights at each stage
        """
        N, B, H = cls_per_bot.shape
        device = cls_per_bot.device

        # Get annealing schedule (sorted high to low = hot to cold)
        schedule = sorted(self.config.annealing_schedule, reverse=True)
        print(f"\n[Track 3] Sequential Annealing: {schedule[0]:.1f} -> {schedule[-1]:.1f}")
        print("  Hot stages: strong prior lock, weak separation pressure")
        print("  Cold stages: weak prior lock, strong separation pressure")

        # Project through kernel (use first/primary kernel)
        kernel_name = list(self.bases.keys())[0]
        basis = self.bases[kernel_name]

        if basis._sigma is None:
            sigma = basis.estimate_sigma(cls_per_bot, self.config.sigma_policy)
            basis.set_sigma(sigma)

        flat = cls_per_bot.reshape(N * B, H)
        phi_flat = basis(flat)
        D = phi_flat.shape[-1]
        phi = phi_flat.reshape(N, B, D)  # [N, B, D]

        # Initialize: uniform barycenter prior
        current_weights = torch.ones(N, B, device=device) / B  # [N, B]

        # Tracking
        weight_trajectory = [current_weights.clone()]
        consensus_weights = None
        adaptive_work_accum = 0.0
        stage_embeddings = []

        # Annealing loop: Hot → Cold
        for stage_idx, alpha in enumerate(schedule):
            # Hot stages remain close to the prior barycenter; cold stages relax it.
            new_weights, stage_work = self._anneal_step(
                phi=phi,
                alpha=alpha,
                prior_weights=current_weights,
                kl_weight=self.config.annealing_kl_weight,
            )

            # Record work (distance from prior state)
            dist = torch.norm(new_weights - current_weights, p=2, dim=-1).mean().item()
            adaptive_work_accum += dist

            # Update state (THE BRIDGE: memory passes to next stage)
            current_weights = new_weights
            weight_trajectory.append(current_weights.clone())

            # Fuse at this stage
            stage_emb = torch.einsum('nbd,nb->nd', phi, current_weights)
            stage_embeddings.append(stage_emb)

            # Capture the hot-stage barycenter before the system is released.
            if stage_idx == 0:
                consensus_weights = current_weights.clone()

            print(f"  a={alpha:.2f}: work={dist:.4f}, weight_entropy={self._entropy(current_weights).mean():.3f}")

        # Final (cold) state
        final_weights = current_weights
        fused_cold = stage_embeddings[-1]
        fused_hot = stage_embeddings[0]

        # Compute stability: cosine similarity between hot-stage barycenter and cold-state weights.
        # High similarity means the observer mixture stayed near the hot prior;
        # low similarity means colder stages separated meaningfully.
        stability_mask = F.cosine_similarity(consensus_weights, final_weights, dim=1)
        stability_mask = torch.clamp(stability_mask, 0.0, 1.0)

        # Compute uncertainty from trajectory variance
        stage_stack = torch.stack(stage_embeddings, dim=0)  # [n_stages, N, D]
        fused_std = stage_stack.std(dim=0)  # [N, D]

        result = {
            'fused': fused_cold,
            'fused_std': fused_std,
            'consensus_embedding': fused_hot,
            'stability_mask': stability_mask,
            'atmospheric_alpha': float(stability_mask.mean().item()),
            'adaptive_work': adaptive_work_accum,
            'weight_trajectory': [w.cpu() for w in weight_trajectory],
            'provenance': {
                'method': 'sequential_annealing',
                'schedule': schedule,
                'n_stages': len(schedule),
                'thermodynamic_objective': {
                    'hot_stage': 'strong prior regularization, weak separation pressure',
                    'cold_stage': 'weak prior regularization, strong separation pressure',
                    'prior_reference': 'previous-stage simplex weights',
                },
            },
        }

        # Curvature across annealing trajectory (n_stages samples per article)
        if compute_curvature and len(stage_embeddings) >= 2:
            # stage_stack is [n_stages, N, D], permute to [N, n_stages, D]
            trajectory_samples = stage_stack.permute(1, 0, 2)
            pr, er, lr = self._compute_curvature_batch(trajectory_samples)
            result['curvature'] = {
                'participation_ratio': pr.cpu(),
                'effective_rank_90': er.cpu(),
                'lambda1_lambda2': lr.cpu(),
            }

        print(f"  Atmospheric alpha (stability): {result['atmospheric_alpha']:.3f}")
        print(f"  Total adaptive work: {adaptive_work_accum:.3f}")

        return result

    def _anneal_step(
        self,
        phi: torch.Tensor,  # [N, B, D]
        alpha: float,
        prior_weights: torch.Tensor,  # [N, B]
        kl_weight: float = 0.5,
        n_iter: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        """
        Single annealing step: optimize weights at given temperature.

        Hot stage:
            - strong prior regularization
            - weak separation pressure
        Cold stage:
            - weak prior regularization
            - strong separation pressure

        The prior_weights input is already a simplex distribution, so KL is
        computed directly against that prior rather than softmaxing it again.

        Returns:
            new_weights: [N, B] optimized weights
            work: scalar work done (distance from prior)
        """
        N, B, D = phi.shape

        eps = 1e-8
        prior_prob = prior_weights.detach().clamp_min(eps)
        prior_prob = prior_prob / prior_prob.sum(dim=-1, keepdim=True).clamp_min(eps)

        # Optimize logits, but anchor them to the true simplex prior.
        logits = prior_prob.log().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=0.1)

        separation_scale = 1.0 / max(float(alpha), eps)
        prior_scale = float(kl_weight) * float(alpha)

        for _ in range(n_iter):
            optimizer.zero_grad()

            # Softmax to ensure simplex
            w_soft = F.softmax(logits, dim=-1)

            # Fused embedding: [N, D]
            fused = torch.einsum('nbd,nb->nd', phi, w_soft)

            # Structured separation should emerge gradually as the temperature cools.
            separation_gain = fused.var(dim=0).mean()

            # KL divergence from the previous stage's simplex weights.
            kl_div = F.kl_div(w_soft.log(), prior_prob, reduction='batchmean')

            # Hot alpha enforces conformity; cold alpha releases the constraint.
            loss = -(separation_scale * separation_gain) + prior_scale * kl_div

            loss.backward()
            optimizer.step()

        # Final weights
        final_weights = F.softmax(logits.detach(), dim=-1)
        work = torch.norm(final_weights - prior_weights, p=2).item()

        return final_weights, work

    def _entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of a simplex weight distribution per sample."""
        w = weights
        if (w < 0).any() or not torch.allclose(
            w.sum(dim=-1),
            torch.ones_like(w.sum(dim=-1)),
            atol=1e-4,
            rtol=1e-4,
        ):
            w = F.softmax(weights, dim=-1)
        log_w = torch.log(w + 1e-9)
        return -(w * log_w).sum(dim=-1)

    def _geometric_median(self, points: torch.Tensor, max_iter: int = 32, tol: float = 1e-5) -> torch.Tensor:
        """Weiszfeld geometric median for a small set of observer vectors."""
        guess = points.mean(dim=0)
        eps = 1e-8
        for _ in range(max_iter):
            distances = torch.norm(points - guess, dim=1).clamp_min(eps)
            if torch.any(distances <= tol):
                return points[torch.argmin(distances)]
            inv_dist = 1.0 / distances
            next_guess = (points * inv_dist.unsqueeze(1)).sum(dim=0) / inv_dist.sum()
            if torch.norm(next_guess - guess).item() <= tol:
                return next_guess
            guess = next_guess
        return guess

    def _remove_consensus_pca(
        self,
        phi_per_kernel: Dict[str, torch.Tensor],
        n_components: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Diagnostic-only consensus residualization using a robust geometric median.

        This intentionally avoids PCA: the leading PCA direction across observers is
        a disagreement axis, not a principled consensus field.

        Args:
            phi_per_kernel: dict of {kernel_name: [N, B, D]}
            n_components: unused, retained for API compatibility

        Returns:
            residuals: dict of {kernel_name: [N, B, D]} with robust shared field removed
        """
        residuals = {}

        for k_name, phi in phi_per_kernel.items():
            N, B, D = phi.shape
            phi_residual = torch.zeros_like(phi)

            for i in range(N):
                bots_i = phi[i]
                shared_field = self._geometric_median(bots_i)
                phi_residual[i] = bots_i - shared_field.unsqueeze(0)

            residuals[k_name] = phi_residual

        return residuals

    def _decompose_vmo_variance_legacy(
        self,
        phi_per_kernel: Dict[str, torch.Tensor],
        dirichlet_weights: torch.Tensor,
        kernel_weights: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        LEGACY: Decompose variance into V, M, O components.

        Note: This version is superseded by the method at line 890 which takes
        samples_per_kernel directly. Kept for backward compatibility.

        Uses ANOVA-style decomposition:
        - V-variance: across bots (holding kernel, weights fixed)
        - M-variance: across kernels (holding bots, weights fixed)
        - O-variance: across Dirichlet samples (holding bots, kernels fixed)
        - Interaction: remaining variance

        Returns:
            Dict with variance fractions
        """
        K_m = len(phi_per_kernel)
        K_o = dirichlet_weights.shape[0]
        
        # Get first kernel's shape
        first_phi = next(iter(phi_per_kernel.values()))
        N, B, D = first_phi.shape
        
        # Build full sample tensor: [N, B, K_m, K_o, D]
        # This is expensive but gives exact decomposition
        
        # Subsample for efficiency
        n_sample = min(N, 500)
        idx = torch.randperm(N)[:n_sample]
        
        # Collect all combinations
        all_views = []
        for k_idx, (k_name, phi) in enumerate(phi_per_kernel.items()):
            phi_sub = phi[idx].to(device)  # [n_sample, B, D]
            
            # For each Dirichlet weight
            for o_idx in range(K_o):
                w = dirichlet_weights[o_idx]  # [B]
                # Weighted combination: [n_sample, D]
                view = torch.einsum('nbd,b->nd', phi_sub, w)
                all_views.append(view)
        
        # Stack: [n_sample, K_m * K_o, D]
        all_views = torch.stack(all_views, dim=1)
        
        # Reshape to [n_sample, K_m, K_o, D] for variance decomposition
        all_views_4d = all_views.reshape(n_sample, K_m, K_o, D)
        
        # Total variance
        total_var = all_views.var().item()
        
        if total_var < 1e-10:
            return {
                'total_variance': 0.0,
                'v_variance': 0.0, 'v_fraction': 0.0,
                'm_variance': 0.0, 'm_fraction': 0.0,
                'o_variance': 0.0, 'o_fraction': 0.0,
                'interaction_variance': 0.0, 'interaction_fraction': 0.0,
            }
        
        # M-variance: variance across kernels (averaging over O)
        m_mean = all_views_4d.mean(dim=2)  # [n_sample, K_m, D]
        m_var = m_mean.var().item()
        
        # O-variance: variance across Dirichlet (averaging over M)
        o_mean = all_views_4d.mean(dim=1)  # [n_sample, K_o, D]
        o_var = o_mean.var().item()
        
        # V-variance: we need per-bot views
        # Compute bot contributions separately
        v_views = []
        for k_idx, (k_name, phi) in enumerate(phi_per_kernel.items()):
            phi_sub = phi[idx].to(device)  # [n_sample, B, D]
            v_views.append(phi_sub)
        v_stacked = torch.stack(v_views, dim=2)  # [n_sample, B, K_m, D]
        v_mean = v_stacked.mean(dim=2)  # [n_sample, B, D] - average over kernels
        v_var = v_mean.var().item()
        
        # Interaction variance (what's left)
        interaction_var = max(0, total_var - (v_var + m_var + o_var))
        
        return {
            'total_variance': total_var,
            'v_variance': v_var,
            'v_fraction': v_var / total_var,
            'm_variance': m_var,
            'm_fraction': m_var / total_var,
            'o_variance': o_var,
            'o_fraction': o_var / total_var,
            'interaction_variance': interaction_var,
            'interaction_fraction': interaction_var / total_var,
        }
    
    def _sequential_cooling_adapt(
        self,
        phi_per_kernel: Dict[str, torch.Tensor],
        n_iterations: int = 10,
    ) -> torch.Tensor:
        """
        Iterative sequential cooling (Atmospheric Annealing) adaptation.
        
        Instead of static weights, adaptively weight kernel × bot
        combinations based on "flow" (variance contribution).
        
        The algorithm:
        1. Initialize uniform weights
        2. For each iteration:
           a. Compute "flow" through each path (variance captured)
           b. Reinforce high-flow paths
           c. Decay low-flow paths
           d. Prune paths below threshold
        3. Return final weights
        
        Args:
            phi_per_kernel: dict of {kernel_name: [N, B, D]}
            n_iterations: number of adaptation iterations
            
        Returns:
            weights: [K_m] kernel weights after adaptation
        """
        K_m = len(phi_per_kernel)
        kernel_names = list(phi_per_kernel.keys())
        
        # Get shape from first kernel
        first_phi = next(iter(phi_per_kernel.values()))
        N, B, D = first_phi.shape
        device = first_phi.device
        
        # Initialize uniform weights over (kernel, bot) paths
        # Shape: [K_m, B]
        path_weights = torch.ones(K_m, B, device=device) / (K_m * B)
        
        # Precompute per-bot variance for each kernel
        # This measures "how much unique info does this bot contribute in this kernel"
        bot_variances = torch.zeros(K_m, B, device=device)
        for k_idx, (k_name, phi) in enumerate(phi_per_kernel.items()):
            for b in range(B):
                # Variance of bot b's features across articles
                bot_var = phi[:, b, :].var().item()
                bot_variances[k_idx, b] = bot_var
        
        # Normalize to get "flow" (relative contribution)
        flow = bot_variances / (bot_variances.sum() + 1e-10)
        
        # Iterative refinement
        decay = self.config.cooling_decay
        reinforce = self.config.cooling_reinforce
        prune_threshold = 0.01 / (K_m * B)  # Minimum path weight
        
        for iteration in range(n_iterations):
            # Reinforce paths proportional to flow
            path_weights = path_weights * (1 + reinforce * flow)
            
            # Apply decay
            path_weights = path_weights * (1 - decay)
            
            # Prune low-weight paths
            path_weights[path_weights < prune_threshold] = 0
            
            # Renormalize
            if path_weights.sum() > 0:
                path_weights = path_weights / path_weights.sum()
            else:
                # If everything pruned, reset to uniform
                path_weights = torch.ones(K_m, B, device=device) / (K_m * B)
        
        # Aggregate to kernel weights (sum over bots)
        kernel_weights = path_weights.sum(dim=1)  # [K_m]
        
        # Log which paths survived
        active_paths = (path_weights > prune_threshold).sum().item()
        print(f"    Sequential cooling: {active_paths}/{K_m * B} paths active")
        for k_idx, k_name in enumerate(kernel_names):
            print(f"      {k_name}: weight={kernel_weights[k_idx]:.3f}")
        
        return kernel_weights


# =============================================================================
# CRN UTILITIES
# =============================================================================

def generate_locked_weights(
    n_bots: int,
    n_observers: int,
    alphas: List[float],
    seed: int,
    output_dir: Path,
) -> Dict[str, str]:
    """
    Pre-generate Dirichlet weights for all alpha values.
    
    Returns paths to saved weight files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    torch.manual_seed(seed)
    
    for alpha in alphas:
        alpha_vec = torch.full((n_bots,), alpha)
        dirichlet = torch.distributions.Dirichlet(alpha_vec)
        weights = dirichlet.sample((n_observers,)).numpy()
        
        path = output_dir / f"dirichlet_weights_alpha_{alpha}.npz"
        np.savez(path, 
            weights=weights,
            alpha=alpha,
            n_observers=n_observers,
            n_bots=n_bots,
            seed=seed,
        )
        paths[f"alpha_{alpha}"] = str(path)
    
    # Save index
    index_path = output_dir / "weights_index.json"
    with open(index_path, 'w') as f:
        json.dump({
            'seed': seed,
            'alphas': alphas,
            'n_bots': n_bots,
            'n_observers': n_observers,
            'paths': paths,
        }, f, indent=2)
    
    return paths


def generate_shared_basis(
    hidden_dim: int,
    rks_dim: int,
    seed: int,
    output_path: Path,
) -> SharedRKSBasis:
    """
    Generate and save a shared RKS basis.
    """
    basis = SharedRKSBasis(
        input_dim=hidden_dim,
        output_dim=rks_dim,
        seed=seed,
    )
    basis.save(str(output_path))
    return basis


# =============================================================================
# ALPHA SWEEP
# =============================================================================

def run_alpha_sweep(
    cls_per_bot: torch.Tensor,
    alphas: List[float],
    config: DirichletFusionConfig = None,
    shared_basis: Optional[SharedRKSBasis] = None,
    locked_weights_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run Dirichlet alpha sweep with proper CRN.
    
    Args:
        cls_per_bot: [N, B, H] input
        alphas: list of alpha values to sweep
        config: base config (alpha will be overridden)
        shared_basis: pre-generated basis (for M-observer control)
        locked_weights_dir: directory with pre-generated weights
        
    Returns:
        Results for each alpha with curvature, Gram metrics, etc.
    """
    if config is None:
        config = DirichletFusionConfig()
    
    results = {'alphas': {}, 'wavelength': {}}
    grams = {}
    
    for alpha in alphas:
        # Create config for this alpha
        alpha_config = DirichletFusionConfig(
            n_bots=config.n_bots,
            hidden_dim=config.hidden_dim,
            rks_dim=config.rks_dim,
            n_observers=config.n_observers,
            alpha=alpha,
            basis_seed=config.basis_seed,
            crn_enabled=config.crn_enabled,
            crn_seed=config.crn_seed,
        )
        
        # Use locked weights if available
        if locked_weights_dir:
            weights_path = Path(locked_weights_dir) / f"dirichlet_weights_alpha_{alpha}.npz"
            if weights_path.exists():
                alpha_config.locked_dirichlet_weights = str(weights_path)
        
        # Create fusion module
        fusion = DirichletFusion(alpha_config)
        
        # Use shared basis if provided
        if shared_basis is not None:
            fusion.basis = shared_basis
        
        # Run fusion
        output = fusion(cls_per_bot, compute_curvature=True)
        
        # Compute Gram matrix
        fused = output['fused']
        fused_norm = F.normalize(fused, dim=-1)
        gram = fused_norm @ fused_norm.T
        grams[alpha] = gram
        
        results['alphas'][f"alpha_{alpha}"] = {
            'curvature_mean': {
                'participation_ratio': float(output['curvature']['participation_ratio'].mean().item()),
                'effective_rank_90': float(output['curvature']['effective_rank_90'].float().mean().item()),
                'lambda1_lambda2': float(output['curvature']['lambda1_lambda2'].mean().item()),
            },
            'fused_stats': {
                'mean_norm': float(fused.norm(dim=-1).mean().item()),
                'std_mean': float(output['fused_std'].mean().item()),
            },
            'provenance': output['provenance'],
        }
    
    # Compute wavelength metrics between adjacent alphas
    sorted_alphas = sorted(alphas)
    for i in range(len(sorted_alphas) - 1):
        a1, a2 = sorted_alphas[i], sorted_alphas[i+1]
        g1, g2 = grams[a1], grams[a2]
        
        # Frobenius distance
        fro_dist = float(torch.norm(g1 - g2, p='fro').item())
        
        # kNN flip rate
        k = min(10, g1.shape[0] - 1)
        idx1 = torch.argsort(-g1, dim=1)[:, 1:k+1]
        idx2 = torch.argsort(-g2, dim=1)[:, 1:k+1]
        flips = sum(len(set(idx1[i].tolist()) - set(idx2[i].tolist())) for i in range(g1.shape[0]))
        flip_rate = flips / (g1.shape[0] * k)
        
        results['wavelength'][f"{a1}_to_{a2}"] = {
            'frobenius_distance': fro_dist,
            'knn_flip_rate': flip_rate,
        }
    
    return results


# =============================================================================
# PHYSARUM POLYCEPHALUM: SLIME MOLD SEMANTIC NETWORK ANALYSIS
# =============================================================================
"""
Sequential Cooling (Atmospheric Annealing) finds optimal paths by:
1. Spreading uniformly (exploration)
2. Reinforcing paths that connect sources efficiently
3. Pruning paths that don't contribute

In our system:
- α → 0: Concentrated weights (exploitation, commit to specific bots)
- α → ∞: Uniform weights (exploration collapses to consensus)
- The α sweep PROBES the semantic manifold like atmospheric annealing probing a phase space

CRACKS: Article pairs where similarity is observer-dependent (high variance across α)
BONDS: Article pairs where similarity is robust (low variance, stable across α)

Controls validate: Real text has meaningful crack/bond topology; random text doesn't.
"""


@dataclass
class CrackBondMetrics:
    """Metrics for a single article pair across the α sweep."""
    article_i: int
    article_j: int
    
    # Core metrics
    crack_score: float      # Variance of similarity across α values
    bond_score: float       # Mean similarity * inverse variance (stable strong connections)
    
    # Per-α similarities
    alpha_similarities: Dict[float, float]  # α -> mean similarity across observers
    alpha_variances: Dict[float, float]     # α -> variance across observers
    
    # Derived
    max_similarity: float
    min_similarity: float
    similarity_range: float  # max - min (how much it swings)
    
    # Classification
    is_crack: bool          # High variance, unstable relationship
    is_bond: bool           # Low variance, strong stable relationship
    is_contested: bool      # High variance AND high mean (strong but unstable)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class AnnealingResult:
    """Full atmospheric annealing analysis results for a corpus."""
    corpus_name: str
    n_articles: int
    alphas_tested: List[float]
    
    # Global metrics
    mean_crack_score: float
    mean_bond_score: float
    n_cracks: int
    n_bonds: int
    n_contested: int
    crack_fraction: float   # What fraction of pairs are cracks
    bond_fraction: float
    
    # Per-pair metrics (sparse - only store interesting pairs)
    top_cracks: List[CrackBondMetrics]      # Highest crack scores
    top_bonds: List[CrackBondMetrics]       # Highest bond scores
    top_contested: List[CrackBondMetrics]   # High variance + high mean
    
    # Full matrices for visualization
    crack_matrix: Optional[torch.Tensor]    # [N, N] crack scores
    bond_matrix: Optional[torch.Tensor]     # [N, N] bond scores
    mean_similarity_matrix: Optional[torch.Tensor]  # [N, N] mean across all α
    
    # Alpha response data (for plotting α curves)
    alpha_response_samples: Dict[str, Dict[float, Tuple[float, float]]]  # pair_id -> α -> (mean, std)
    
    # Provenance
    timestamp: str
    config: Dict
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert tensors to lists
        for key in ['crack_matrix', 'bond_matrix', 'mean_similarity_matrix']:
            if d[key] is not None:
                d[key] = d[key].tolist() if torch.is_tensor(d[key]) else d[key]
        return d
    
    def save(self, path: Path):
        """Save to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'AnnealingResult':
        """Load from JSON."""
        with open(path) as f:
            d = json.load(f)
        # Convert lists back to tensors
        for key in ['crack_matrix', 'bond_matrix', 'mean_similarity_matrix']:
            if d[key] is not None:
                d[key] = torch.tensor(d[key])
        # Convert top_* back to dataclasses
        for key in ['top_cracks', 'top_bonds', 'top_contested']:
            d[key] = [CrackBondMetrics(**item) for item in d[key]]
        return cls(**d)


class AtmosphericAnnealer:
    """
    Atmospheric semantic network analyzer.
    
    Runs α sweeps and computes crack/bond topology to reveal
    where semantic structure is contested vs agreed upon.
    """
    
    def __init__(
        self,
        fusion: DirichletFusion,
        alphas: List[float] = None,
        n_observers_per_alpha: int = 50,
        crack_threshold: float = 0.01,    # Variance above this = crack
        bond_threshold: float = 0.5,       # Mean similarity above this for bond consideration
        device: str = 'cuda',
    ):
        """
        Args:
            fusion: DirichletFusion instance (with locked M-observer basis)
            alphas: α values to sweep (default: log-spaced from 0.1 to 20)
            n_observers_per_alpha: How many O-observers to sample per α
            crack_threshold: Variance threshold for crack detection
            bond_threshold: Mean similarity threshold for bond detection
            device: Compute device
        """
        self.fusion = fusion
        self.alphas = alphas or [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        self.n_observers = n_observers_per_alpha
        self.crack_threshold = crack_threshold
        self.bond_threshold = bond_threshold
        self.device = device
    
    def _compute_gram_for_alpha(
        self,
        bot_rkhs: torch.Tensor,  # [N, B, D] - already in RKHS
        alpha: float,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gram matrix for a specific α value.
        
        Returns:
            mean_gram: [N, N] mean similarity across observers
            var_gram: [N, N] variance of similarity across observers
        """
        N, B, D = bot_rkhs.shape
        
        # Sample Dirichlet weights
        if seed is not None:
            torch.manual_seed(seed)
        
        # Dirichlet(α, α, ..., α) for B bots
        concentration = torch.full((B,), alpha)
        dirichlet = torch.distributions.Dirichlet(concentration)
        weights = dirichlet.sample((self.n_observers,))  # [K, B]
        
        # Fuse: [N, K, D]
        bot_rkhs_gpu = bot_rkhs.to(self.device)
        weights_gpu = weights.to(self.device)
        
        # observer_k(article_n) = Σ_b w_{k,b} * φ_b(n)
        fused = torch.einsum('nbd,kb->nkd', bot_rkhs_gpu, weights_gpu)  # [N, K, D]
        
        # Normalize for cosine similarity
        fused_norm = F.normalize(fused, p=2, dim=-1)  # [N, K, D]
        
        # Compute per-observer Gram matrices and aggregate
        # gram_k[i,j] = <fused_k(i), fused_k(j)>
        # We want mean and variance across k
        
        # Efficient: compute all K Gram matrices at once
        # fused_norm: [N, K, D]
        # gram: [K, N, N] where gram[k] = fused_norm[:, k, :] @ fused_norm[:, k, :].T
        gram_all = torch.bmm(
            fused_norm.permute(1, 0, 2),  # [K, N, D]
            fused_norm.permute(1, 2, 0)   # [K, D, N]
        )  # [K, N, N]
        
        mean_gram = gram_all.mean(dim=0)  # [N, N]
        var_gram = gram_all.var(dim=0)    # [N, N]
        
        return mean_gram.cpu(), var_gram.cpu()
    
    def analyze(
        self,
        cls_per_bot: torch.Tensor,  # [N, B, H] raw CLS embeddings
        corpus_name: str = 'unknown',
        article_ids: List[str] = None,
        top_k: int = 50,  # How many top cracks/bonds to store
        store_full_matrices: bool = True,
        verbose: bool = True,
    ) -> AnnealingResult:
        """
        Run full Atmospheric Annealing analysis on a corpus.
        
        Args:
            cls_per_bot: [N, B, H] CLS embeddings per bot
            corpus_name: Name for logging/saving
            article_ids: Optional canonical IDs for article identification
            top_k: Number of top cracks/bonds to store in detail
            store_full_matrices: Whether to store full N×N matrices
            verbose: Print progress
            
        Returns:
            AnnealingResult with full crack/bond topology
        """
        import datetime
        
        N, B, H = cls_per_bot.shape
        if verbose:
            print(f"\n{'='*60}")
            print(f"PHYSARUM ANALYSIS: {corpus_name}")
            print(f"{'='*60}")
            print(f"Articles: {N}, Bots: {B}, Hidden: {H}")
            print(f"Alpha sweep: {self.alphas}")
            print(f"Observers per α: {self.n_observers}")
        
        # Step 1: Project to RKHS (shared basis, done once)
        if verbose:
            print(f"\n[1/4] Projecting to shared RKHS...")
        
        with torch.no_grad():
            cls_flat = cls_per_bot.reshape(N * B, H).to(self.device)
            
            # Ensure sigma is estimated
            if not self.fusion._sigma_estimated:
                self.fusion._ensure_sigma(cls_per_bot)
            
            phi_flat = self.fusion.basis(cls_flat)
            bot_rkhs = phi_flat.reshape(N, B, -1)  # [N, B, D]
        
        D = bot_rkhs.shape[-1]
        if verbose:
            print(f"  RKHS dim: {D}")
        
        # Step 2: Sweep α and compute Gram matrices
        if verbose:
            print(f"\n[2/4] Running α sweep...")
        
        alpha_grams_mean = {}  # α -> [N, N]
        alpha_grams_var = {}   # α -> [N, N]
        
        for alpha in self.alphas:
            mean_g, var_g = self._compute_gram_for_alpha(
                bot_rkhs, alpha, seed=int(alpha * 1000)  # Deterministic per α
            )
            alpha_grams_mean[alpha] = mean_g
            alpha_grams_var[alpha] = var_g
            
            if verbose:
                print(f"  α={alpha:5.1f}: mean_sim={mean_g.mean():.4f}, mean_var={var_g.mean():.6f}")
        
        # Step 3: Compute crack and bond scores
        if verbose:
            print(f"\n[3/4] Computing crack/bond topology...")
        
        # Stack mean grams: [n_alphas, N, N]
        mean_stack = torch.stack([alpha_grams_mean[a] for a in self.alphas], dim=0)
        var_stack = torch.stack([alpha_grams_var[a] for a in self.alphas], dim=0)
        
        # Crack score: variance of mean similarity ACROSS α values
        # High crack = relationship changes depending on α (observer-dependent)
        crack_matrix = mean_stack.var(dim=0)  # [N, N]
        
        # Also consider within-α variance (summed)
        # This captures: even for a fixed α, do observers disagree?
        within_alpha_var = var_stack.mean(dim=0)  # [N, N]
        
        # Combined crack score: between-α variance + within-α variance
        crack_matrix = crack_matrix + within_alpha_var
        
        # Mean similarity across all α (for bond detection)
        mean_similarity_matrix = mean_stack.mean(dim=0)  # [N, N]
        
        # Bond score: high mean similarity AND low variance = stable strong connection
        # bond = mean_sim / (crack + ε)
        eps = 1e-6
        bond_matrix = mean_similarity_matrix / (crack_matrix + eps)
        
        # Step 4: Classify pairs and extract top examples
        if verbose:
            print(f"\n[4/4] Classifying article pairs...")
        
        # Get upper triangle indices (avoid self-similarity and duplicates)
        triu_idx = torch.triu_indices(N, N, offset=1)
        
        crack_scores = crack_matrix[triu_idx[0], triu_idx[1]]
        bond_scores = bond_matrix[triu_idx[0], triu_idx[1]]
        mean_sims = mean_similarity_matrix[triu_idx[0], triu_idx[1]]
        
        # Classifications
        is_crack = crack_scores > self.crack_threshold
        is_strong = mean_sims > self.bond_threshold
        is_bond = (~is_crack) & is_strong  # Low variance + strong similarity
        is_contested = is_crack & is_strong  # High variance + strong similarity
        
        n_pairs = len(crack_scores)
        n_cracks = int(is_crack.sum().item())
        n_bonds = int(is_bond.sum().item())
        n_contested = int(is_contested.sum().item())
        
        if verbose:
            print(f"  Total pairs: {n_pairs}")
            print(f"  Cracks (high var): {n_cracks} ({100*n_cracks/n_pairs:.1f}%)")
            print(f"  Bonds (low var, strong): {n_bonds} ({100*n_bonds/n_pairs:.1f}%)")
            print(f"  Contested (high var, strong): {n_contested} ({100*n_contested/n_pairs:.1f}%)")
        
        # Build detailed metrics for top pairs
        def build_pair_metrics(i: int, j: int) -> CrackBondMetrics:
            alpha_sims = {a: float(alpha_grams_mean[a][i, j].item()) for a in self.alphas}
            alpha_vars = {a: float(alpha_grams_var[a][i, j].item()) for a in self.alphas}
            sims = list(alpha_sims.values())
            
            crack_sc = float(crack_matrix[i, j].item())
            bond_sc = float(bond_matrix[i, j].item())
            
            return CrackBondMetrics(
                article_i=i,
                article_j=j,
                crack_score=crack_sc,
                bond_score=bond_sc,
                alpha_similarities=alpha_sims,
                alpha_variances=alpha_vars,
                max_similarity=max(sims),
                min_similarity=min(sims),
                similarity_range=max(sims) - min(sims),
                is_crack=crack_sc > self.crack_threshold,
                is_bond=crack_sc <= self.crack_threshold and np.mean(sims) > self.bond_threshold,
                is_contested=crack_sc > self.crack_threshold and np.mean(sims) > self.bond_threshold,
            )
        
        # Top cracks (highest crack score)
        top_crack_idx = torch.argsort(crack_scores, descending=True)[:top_k]
        top_cracks = [
            build_pair_metrics(
                int(triu_idx[0][idx].item()),
                int(triu_idx[1][idx].item())
            )
            for idx in top_crack_idx
        ]
        
        # Top bonds (highest bond score among non-cracks)
        bond_scores_masked = bond_scores.clone()
        bond_scores_masked[is_crack] = -float('inf')
        top_bond_idx = torch.argsort(bond_scores_masked, descending=True)[:top_k]
        top_bonds = [
            build_pair_metrics(
                int(triu_idx[0][idx].item()),
                int(triu_idx[1][idx].item())
            )
            for idx in top_bond_idx
            if bond_scores_masked[idx] > -float('inf')
        ]
        
        # Top contested (high variance AND high mean)
        contested_score = crack_scores * mean_sims  # Both high = high score
        contested_score[~is_contested] = -float('inf')
        top_contested_idx = torch.argsort(contested_score, descending=True)[:top_k]
        top_contested = [
            build_pair_metrics(
                int(triu_idx[0][idx].item()),
                int(triu_idx[1][idx].item())
            )
            for idx in top_contested_idx
            if contested_score[idx] > -float('inf')
        ]
        
        # Alpha response samples (for plotting)
        # Store α curves for a sample of interesting pairs
        alpha_response_samples = {}
        sample_pairs = (top_cracks[:10] + top_bonds[:10] + top_contested[:10])
        for metrics in sample_pairs:
            pair_id = f"{metrics.article_i}_{metrics.article_j}"
            alpha_response_samples[pair_id] = {
                a: (metrics.alpha_similarities[a], metrics.alpha_variances.get(a, 0))
                for a in self.alphas
            }
        
        result = AnnealingResult(
            corpus_name=corpus_name,
            n_articles=N,
            alphas_tested=self.alphas,
            mean_crack_score=float(crack_scores.mean().item()),
            mean_bond_score=float(bond_scores[~torch.isinf(bond_scores)].mean().item()) if not torch.all(torch.isinf(bond_scores)) else 0.0,
            n_cracks=n_cracks,
            n_bonds=n_bonds,
            n_contested=n_contested,
            crack_fraction=n_cracks / n_pairs if n_pairs > 0 else 0,
            bond_fraction=n_bonds / n_pairs if n_pairs > 0 else 0,
            top_cracks=top_cracks,
            top_bonds=top_bonds,
            top_contested=top_contested,
            crack_matrix=crack_matrix if store_full_matrices else None,
            bond_matrix=bond_matrix if store_full_matrices else None,
            mean_similarity_matrix=mean_similarity_matrix if store_full_matrices else None,
            alpha_response_samples=alpha_response_samples,
            timestamp=datetime.datetime.now().isoformat(),
            config={
                'alphas': self.alphas,
                'n_observers_per_alpha': self.n_observers,
                'crack_threshold': self.crack_threshold,
                'bond_threshold': self.bond_threshold,
                'fusion_config': asdict(self.fusion.config) if hasattr(self.fusion, 'config') else {},
            },
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PHYSARUM ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Mean crack score: {result.mean_crack_score:.6f}")
            print(f"Mean bond score: {result.mean_bond_score:.4f}")
            print(f"Crack fraction: {result.crack_fraction:.2%}")
            print(f"Bond fraction: {result.bond_fraction:.2%}")
        
        return result


def compare_annealing_results(
    real: AnnealingResult,
    shuffled: AnnealingResult = None,
    constant: AnnealingResult = None,
    random: AnnealingResult = None,
) -> Dict:
    """
    Compare Atmospheric Annealing analyses across control conditions.
    
    Validates: Real >> Shuffled >> Constant ≈ Random
    
    Returns comparison metrics and verdict.
    """
    results = {
        'crack_scores': {'real': real.mean_crack_score},
        'bond_scores': {'real': real.mean_bond_score},
        'crack_fractions': {'real': real.crack_fraction},
        'n_contested': {'real': real.n_contested},
    }
    
    corpora = {'real': real}
    if shuffled:
        corpora['shuffled'] = shuffled
        results['crack_scores']['shuffled'] = shuffled.mean_crack_score
        results['bond_scores']['shuffled'] = shuffled.mean_bond_score
        results['crack_fractions']['shuffled'] = shuffled.crack_fraction
        results['n_contested']['shuffled'] = shuffled.n_contested
    if constant:
        corpora['constant'] = constant
        results['crack_scores']['constant'] = constant.mean_crack_score
        results['bond_scores']['constant'] = constant.mean_bond_score
        results['crack_fractions']['constant'] = constant.crack_fraction
        results['n_contested']['constant'] = constant.n_contested
    if random:
        corpora['random'] = random
        results['crack_scores']['random'] = random.mean_crack_score
        results['bond_scores']['random'] = random.mean_bond_score
        results['crack_fractions']['random'] = random.crack_fraction
        results['n_contested']['random'] = random.n_contested
    
    # Check ordering
    cs = results['crack_scores']
    ordering_satisfied = True
    ordering_details = []
    
    if 'shuffled' in cs:
        if cs['real'] > cs['shuffled']:
            ordering_details.append("✓ Real > Shuffled")
        else:
            ordering_details.append("✗ Real <= Shuffled (UNEXPECTED)")
            ordering_satisfied = False
    
    if 'shuffled' in cs and 'constant' in cs:
        if cs['shuffled'] > cs['constant']:
            ordering_details.append("✓ Shuffled > Constant")
        else:
            ordering_details.append("✗ Shuffled <= Constant")
            ordering_satisfied = False
    
    if 'constant' in cs and 'random' in cs:
        ratio = cs['constant'] / (cs['random'] + 1e-10)
        if 0.5 < ratio < 2.0:
            ordering_details.append(f"✓ Constant ≈ Random (ratio={ratio:.2f})")
        else:
            ordering_details.append(f"? Constant vs Random ratio={ratio:.2f}")
    
    results['ordering_satisfied'] = ordering_satisfied
    results['ordering_details'] = ordering_details
    
    # Verdict
    if ordering_satisfied:
        results['verdict'] = "PASS: Sequential cooling found real semantic structure"
    else:
        results['verdict'] = "FAIL: Control ordering not satisfied - check for artifacts"
    
    return results


# =============================================================================
# PLOTLY VISUALIZATION HELPERS (for later use)
# =============================================================================

def prepare_crack_heatmap_data(analysis: AnnealingResult) -> Dict:
    """
    Prepare data for Plotly heatmap of crack scores.
    
    Returns dict ready for plotly.graph_objects.Heatmap
    """
    if analysis.crack_matrix is None:
        return {'error': 'No crack matrix stored'}
    
    return {
        'z': analysis.crack_matrix.numpy(),
        'colorscale': 'RdBu_r',  # Red = high crack, Blue = low
        'title': f'Crack Topology: {analysis.corpus_name}',
        'xlabel': 'Article Index',
        'ylabel': 'Article Index',
        'colorbar_title': 'Crack Score',
    }


def prepare_alpha_response_curves(analysis: AnnealingResult) -> Dict:
    """
    Prepare data for Plotly line charts showing α response.
    
    Returns dict with traces for each sampled pair.
    """
    traces = []
    
    for pair_id, alpha_data in analysis.alpha_response_samples.items():
        alphas = sorted(alpha_data.keys())
        means = [alpha_data[a][0] for a in alphas]
        stds = [np.sqrt(alpha_data[a][1]) for a in alphas]
        
        # Determine if this is a crack, bond, or contested
        pair_metrics = None
        for m in analysis.top_cracks + analysis.top_bonds + analysis.top_contested:
            if f"{m.article_i}_{m.article_j}" == pair_id:
                pair_metrics = m
                break
        
        pair_type = 'unknown'
        if pair_metrics:
            if pair_metrics.is_contested:
                pair_type = 'contested'
            elif pair_metrics.is_crack:
                pair_type = 'crack'
            elif pair_metrics.is_bond:
                pair_type = 'bond'
        
        traces.append({
            'pair_id': pair_id,
            'pair_type': pair_type,
            'alphas': alphas,
            'means': means,
            'stds': stds,
            'upper': [m + s for m, s in zip(means, stds)],
            'lower': [m - s for m, s in zip(means, stds)],
        })
    
    return {
        'traces': traces,
        'title': f'α Response Curves: {analysis.corpus_name}',
        'xlabel': 'α (Dirichlet concentration)',
        'ylabel': 'Cosine Similarity',
    }


def prepare_network_graph_data(
    analysis: AnnealingResult,
    similarity_threshold: float = 0.3,
    max_edges: int = 500,
) -> Dict:
    """
    Prepare data for Plotly network graph visualization.
    
    Returns dict with nodes and edges for networkx/plotly.
    """
    if analysis.mean_similarity_matrix is None or analysis.crack_matrix is None:
        return {'error': 'No matrices stored'}
    
    N = analysis.n_articles
    sim = analysis.mean_similarity_matrix.numpy()
    crack = analysis.crack_matrix.numpy()
    
    # Get edges above threshold
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if sim[i, j] > similarity_threshold:
                edges.append({
                    'source': i,
                    'target': j,
                    'similarity': float(sim[i, j]),
                    'crack_score': float(crack[i, j]),
                })
    
    # Sort by similarity and take top max_edges
    edges = sorted(edges, key=lambda e: e['similarity'], reverse=True)[:max_edges]
    
    return {
        'n_nodes': N,
        'edges': edges,
        'title': f'Semantic Network: {analysis.corpus_name}',
        'note': f'{len(edges)} edges (similarity > {similarity_threshold})',
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'DirichletFusionConfig',
    'SharedRKSBasis',
    'DirichletFusion',
    'generate_locked_weights',
    'generate_shared_basis',
    'run_alpha_sweep',
    # Atmospheric Annealing / Sequential Cooling
    'CrackBondMetrics',
    'AnnealingResult',
    'AtmosphericAnnealer',
    'compare_annealing_results',
    # Visualization helpers
    'prepare_crack_heatmap_data',
    'prepare_alpha_response_curves',
    'prepare_network_graph_data',
]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Dirichlet Fusion + Multi-Kernel + Sequential Cooling")
    print("=" * 60)
    
    # Create dummy CLS embeddings [N=10, B=8, H=768]
    torch.manual_seed(42)
    cls_per_bot = torch.randn(10, 8, 768)
    
    # Test 1: Basic single-kernel fusion (backward compat)
    print("\n--- Test 1: Single Kernel Fusion ---")
    config = DirichletFusionConfig(
        rks_dim=512,
        n_observers=20,
        alpha=1.0,
    )
    fusion = DirichletFusion(config)
    
    output = fusion(cls_per_bot, compute_curvature=True)
    
    print(f"Input shape: {cls_per_bot.shape}")
    print(f"Fused shape: {output['fused'].shape}")
    print(f"Curvature PR mean: {output['curvature']['participation_ratio'].mean():.3f}")
    print("[OK] Single kernel test passed")
    
    # Test 2: Multi-kernel V × M × O fusion
    print("\n--- Test 2: Multi-Kernel V × M × O Fusion ---")
    config_multi = DirichletFusionConfig(
        rks_dim=256,
        n_observers=10,
        alpha=1.0,
        multi_kernel=True,
        kernel_types=['rbf', 'laplacian', 'polynomial'],
    )
    fusion_multi = DirichletFusion(config_multi)
    
    output_vmo = fusion_multi.forward_vmo(
        cls_per_bot,
        compute_variance_decomposition=True,
        return_samples=True,
    )
    
    print(f"Fused shape: {output_vmo['fused'].shape}")
    print(f"Samples shape: {output_vmo['all_samples_shape']}")
    print(f"Variance decomposition:")
    vd = output_vmo['variance_decomposition']
    print(f"  V (bots): {vd['v_fraction']:.1%}")
    print(f"  M (kernels): {vd['m_fraction']:.1%}")
    print(f"  O (Dirichlet): {vd['o_fraction']:.1%}")
    print("[OK] Multi-kernel test passed")
    
    # Test 3: Consensus removal
    print("\n--- Test 3: Consensus Removal ---")
    config_consensus = DirichletFusionConfig(
        rks_dim=256,
        n_observers=10,
        alpha=1.0,
        multi_kernel=True,
        kernel_types=['rbf', 'laplacian'],
        remove_consensus=True,
        n_consensus_components=1,
    )
    fusion_consensus = DirichletFusion(config_consensus)
    
    output_consensus = fusion_consensus.forward_vmo(
        cls_per_bot,
        compute_variance_decomposition=True,
    )
    print(f"Fused shape (after consensus removal): {output_consensus['fused'].shape}")
    print("[OK] Consensus removal test passed")
    
    # Test 4: Sequential Cooling adaptive weighting
    print("\n--- Test 4: Sequential Cooling Adaptation ---")
    config_cooling = DirichletFusionConfig(
        rks_dim=256,
        n_observers=10,
        alpha=1.0,
        multi_kernel=True,
        kernel_types=['rbf', 'laplacian', 'polynomial'],
        use_sequential_cooling=True,
        cooling_iterations=5,
    )
    fusion_cooling = DirichletFusion(config_cooling)
    
    output_cooling = fusion_cooling.forward_vmo(
        cls_per_bot,
        compute_variance_decomposition=True,
    )
    print(f"Kernel weights after sequential cooling: {output_cooling['kernel_weights']}")
    print("[OK] Sequential cooling test passed")
    
    # Test 5: Atmospheric Annealer
    print("\n--- Test 5: Atmospheric Annealer ---")
    
    analyzer = AtmosphericAnnealer(
        fusion=fusion,
        alphas=[0.1, 1.0, 10.0],
        n_observers_per_alpha=10,
        device='cpu',
    )
    
    result = analyzer.analyze(
        cls_per_bot,
        corpus_name='test_corpus',
        top_k=5,
        store_full_matrices=True,
        verbose=True,
    )
    
    print(f"\nTop crack: pair ({result.top_cracks[0].article_i}, {result.top_cracks[0].article_j})")
    print(f"  Crack score: {result.top_cracks[0].crack_score:.6f}")
    
    heatmap_data = prepare_crack_heatmap_data(result)
    print(f"Heatmap data shape: {heatmap_data['z'].shape}")
    print("[OK] Atmospheric annealer test passed")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
