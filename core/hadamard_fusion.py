"""
hadamard_fusion.py — Track 5: Hadamard Kernel Fusion (ASTER v3.2)

THE METRIC TENSOR ASSEMBLY:
===========================
This module is Track 5 of the ASTER architecture — the true fusion layer that
assembles the metric tensor g_μν from its constituent tracks:

    g_μν = (1/ρ)·δ_μν + ∇_μΦ·∇_νΦ

Where:
    - δ_μν (Track 2: Hologram) = Base RKS kernel geometry
    - ∇Φ (Track 1.5: Spectral Polarity) = Gradient/stress tensor
    - 1/ρ (Track 3: Blinker) = Conformal density factor

THE TWO FIXES:
==============
Fix 1: Hadamard Kernel Product (Track 2 ∘ Track 1.5)
    K_final = K_rks * K_spectral  (element-wise)
    Connection exists IFF nodes share BOTH Topic AND Stance.
    P(A ∩ B) = P(A) × P(B) — zero parameters, purely structural.

Fix 2: Conformal Metric (Track 3 as distance scaler, NOT concatenated)
    Distance_final(A, B) = Distance(A, B) / sqrt(ρ(A) × ρ(B))
    High variance → low ρ → stretched distances (hard to connect)
    Low variance → high ρ → compressed distances (easy to connect)

THE PROBLEM SOLVED:
===================
Old pipeline concatenated tracks: [T2 || T1.5 || T3 || ...]
This diluted NMI from 0.89 to 0.58 by drowning semantic signal in dimensions.

New pipeline (this module):
    - Track 2 ∘ Track 1.5: Hadamard product in kernel space (multiplicative)
    - Track 3: Conformal factor that scales distances (not concatenated)
    - Result: Spectral embedding recovers vector space for Walker compatibility

TRACK ARCHITECTURE (ASTER v3.2):
================================
Track 1 (Logits):       Surface Claim — explicit NLI verdict
Track 1.5 (Spectral):   Internal Stress — phase-gate gradient (∇Φ)
Track 2 (Hologram):     Base Terrain — RKS-projected embeddings (δ_μν)
Track 3 (Blinker):      Conformal Factor — Dirichlet variance (1/ρ)
Track 4 (Walker):       Kinetic Probe — MCMC path conductivity
Track 5 (THIS MODULE):  Metric Assembly — K_final = K_T2 ∘ K_T1.5 / sqrt(ρ⊗ρ)
Track 6 (HoTT):         Formal Proofs — type-theoretic verdicts (✓/✗/?)

EXIT GATE: Phase Space Integrator — final normalization to unit hypersphere

Author: Belief Transformer Project (ASTER v3.2)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from .thermo_config import ThermodynamicConfig


@dataclass
class HadamardFusionConfig:
    """Configuration for Hadamard kernel product fusion."""

    # Dimensions
    rks_dim: int = 2048          # Track 2 (hologram) dimension
    spectral_dim: int = 2048     # Track 1.5 (antagonism) dimension
    output_dim: int = 2048       # Spectral embedding output dimension

    # Kernel bandwidth parameters (None = auto-estimate via median heuristic)
    sigma_rks: Optional[float] = None
    sigma_spectral: Optional[float] = None
    sigma_conformal: Optional[float] = None  # For final kernel after conformal scaling

    # Dark manifold protection
    dark_manifold_threshold: float = 0.01  # Min row-sum used only to flag isolated nodes
    dark_manifold_rescue: bool = False     # Deprecated/no-op under the strict contract
    hadamard_softening: float = 0.0        # Deprecated/no-op under the strict contract

    # Conformal metric parameters
    temperature_scale: float = 1.0         # Scale factor for blinker variance

    # Numerical stability
    eigenvalue_floor: float = 1e-10        # Min eigenvalue for spectral embedding
    kernel_floor: float = field(
        default_factory=lambda: ThermodynamicConfig().hadamard_floor
    )  # Min kernel value

    # Performance
    use_nystrom: bool = False              # Use Nystrom approximation for large N
    nystrom_threshold: int = 5000          # N above which to use Nystrom


@dataclass
class HadamardFusionResult:
    """Result container for Hadamard fusion computation."""

    # Final kernel matrix after Hadamard product and conformal scaling
    K_final: torch.Tensor                  # [N, N]

    # Spectral embedding (vector space for Walker)
    embeddings: torch.Tensor               # [N, output_dim]

    # Intermediate kernels (for diagnostics)
    K_rks: torch.Tensor                    # [N, N] - Track 2 kernel
    K_spectral: torch.Tensor               # [N, N] - Track 1.5 kernel
    K_hadamard: torch.Tensor               # [N, N] - Before conformal scaling

    # Conformal factors
    rho: torch.Tensor                      # [N] - Density from Track 3

    # Dark manifold diagnostics
    n_isolated_rescued: int = 0
    isolated_indices: Optional[torch.Tensor] = None

    # Eigenvalue spectrum (for diagnostics)
    eigenvalues: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            'n_samples': self.K_final.shape[0],
            'output_dim': self.embeddings.shape[1],
            'n_isolated_rescued': self.n_isolated_rescued,
            'rho_mean': float(self.rho.mean()),
            'rho_std': float(self.rho.std()),
            'K_hadamard_sparsity': float((self.K_hadamard < 0.01).sum() / self.K_hadamard.numel()),
            'top_eigenvalues': self.eigenvalues[:5].tolist() if self.eigenvalues is not None else None,
        }


class HadamardFusion:
    """
    Hadamard kernel product fusion for Track 2 + Track 1.5.

    Instead of concatenating features (which dilutes signal), we:
    1. Compute kernel matrices in each space
    2. Multiply element-wise (Hadamard product)
    3. Apply conformal scaling from Track 3
    4. Spectral embed to recover vector space

    Mathematical justification:
        K_final(i,j) = K_rks(i,j) * K_spectral(i,j)

    This ensures connection IFF both kernels agree - nodes must share
    BOTH topic (Track 2) AND stance (Track 1.5) to be connected.
    """

    def __init__(self, config: Optional[HadamardFusionConfig] = None):
        self.config = config or HadamardFusionConfig()

    def compute_kernel_matrices(
        self,
        track2: torch.Tensor,   # [N, D_rks] hologram
        track15: torch.Tensor,  # [N, D_spectral] antagonism
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RBF kernel matrices for Track 2 and Track 1.5.

        Uses median heuristic for bandwidth if not specified:
            sigma = median(pairwise_distances) / sqrt(2)

        Args:
            track2: [N, D] hologram features (RKS-projected)
            track15: [N, D] antagonism features (spectral polarity)

        Returns:
            K_rks: [N, N] kernel matrix for Track 2
            K_spectral: [N, N] kernel matrix for Track 1.5
        """
        # Compute pairwise squared distances
        dist_rks_sq = self._pairwise_squared_distances(track2)
        dist_spectral_sq = self._pairwise_squared_distances(track15)

        # Estimate bandwidth via median heuristic
        sigma_rks = self.config.sigma_rks
        if sigma_rks is None:
            sigma_rks = self._median_heuristic(dist_rks_sq)

        sigma_spectral = self.config.sigma_spectral
        if sigma_spectral is None:
            sigma_spectral = self._median_heuristic(dist_spectral_sq)

        # Compute RBF kernels: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        K_rks = torch.exp(-dist_rks_sq / (2 * sigma_rks**2 + self.config.kernel_floor))
        K_spectral = torch.exp(-dist_spectral_sq / (2 * sigma_spectral**2 + self.config.kernel_floor))

        return K_rks, K_spectral

    def hadamard_product(
        self,
        K_rks: torch.Tensor,      # [N, N]
        K_spectral: torch.Tensor, # [N, N]
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Compute Hadamard (element-wise) product with dark manifold protection.

        Risk: Multiplication is destructive - it can legitimately create isolated
        nodes where both kernels have low values. We preserve those voids.

        Args:
            K_rks: [N, N] kernel from Track 2
            K_spectral: [N, N] kernel from Track 1.5

        Returns:
            K_final: [N, N] strict Hadamard product kernel
            n_rescued: Always 0 under the strict contract
            isolated_indices: Indices of isolated nodes (or None)
        """
        # Strict logical AND in kernel space. No softening and no dark-manifold
        # rescue: if cross-track support vanishes, the node remains isolated.
        K_hadamard = K_rks * K_spectral
        K_hadamard = K_hadamard.clamp(min=0.0, max=1.0)

        row_sums = K_hadamard.sum(dim=1)
        threshold = self.config.dark_manifold_threshold * K_hadamard.shape[0]
        isolated_mask = row_sums < threshold
        # Preserve the historical return contract while making the strict behavior
        # explicit: no rows/cols are bridged back into the manifold.
        n_rescued = 0
        isolated_indices = torch.where(isolated_mask)[0] if isolated_mask.any() else None

        return K_hadamard, n_rescued, isolated_indices

    def spectral_embedding(
        self,
        K_final: torch.Tensor,  # [N, N]
        output_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recover vector embeddings from kernel matrix via eigendecomposition.

        Given K = X @ X.T, we recover X by:
            K = V @ Lambda @ V.T  (eigendecomposition)
            X = V @ sqrt(Lambda)  (embedding)

        This is O(N^3) - for large N, use Nystrom approximation.

        Args:
            K_final: [N, N] positive semi-definite kernel matrix
            output_dim: Number of dimensions to keep (default: config.output_dim)

        Returns:
            embeddings: [N, output_dim] vector embeddings
            eigenvalues: [output_dim] eigenvalues used
        """
        output_dim = output_dim or self.config.output_dim
        N = K_final.shape[0]

        # Use Nystrom for large matrices
        if self.config.use_nystrom and N > self.config.nystrom_threshold:
            return self._nystrom_embedding(K_final, output_dim)

        # Ensure symmetry (numerical stability)
        K_sym = (K_final + K_final.T) / 2

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(K_sym)

        # Sort by descending eigenvalue (eigh returns ascending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top-k positive eigenvalues
        k = min(output_dim, N)
        top_eigenvalues = eigenvalues[:k].clamp(min=self.config.eigenvalue_floor)
        top_eigenvectors = eigenvectors[:, :k]

        # Embedding: X = V @ sqrt(Lambda)
        embeddings = top_eigenvectors * torch.sqrt(top_eigenvalues).unsqueeze(0)

        return embeddings, top_eigenvalues

    def _nystrom_embedding(
        self,
        K: torch.Tensor,
        output_dim: int,
        n_landmarks: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nystrom approximation for large kernel matrices.

        O(N * m^2) where m << N is the number of landmarks.
        """
        N = K.shape[0]
        n_landmarks = n_landmarks or min(1000, N // 5)

        # Random landmark selection
        landmark_idx = torch.randperm(N)[:n_landmarks]

        # Extract submatrices
        K_mm = K[landmark_idx][:, landmark_idx]  # [m, m]
        K_nm = K[:, landmark_idx]                 # [N, m]

        # Eigendecomposition of landmark kernel
        eigenvalues_m, eigenvectors_m = torch.linalg.eigh(K_mm)
        idx = torch.argsort(eigenvalues_m, descending=True)
        eigenvalues_m = eigenvalues_m[idx].clamp(min=self.config.eigenvalue_floor)
        eigenvectors_m = eigenvectors_m[:, idx]

        # Nystrom extension
        k = min(output_dim, n_landmarks)
        sqrt_lambda_inv = 1.0 / torch.sqrt(eigenvalues_m[:k])

        # Approximate eigenvectors: V_n = K_nm @ V_m @ Lambda_m^{-1/2}
        embeddings = K_nm @ eigenvectors_m[:, :k] @ torch.diag(sqrt_lambda_inv)

        return embeddings, eigenvalues_m[:k]

    def _pairwise_squared_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)
        dist_sq = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
        return dist_sq.clamp(min=0)  # Numerical stability

    def _median_heuristic(self, dist_sq: torch.Tensor) -> float:
        """Estimate kernel bandwidth via median heuristic."""
        # Take upper triangle (exclude diagonal)
        triu_idx = torch.triu_indices(dist_sq.shape[0], dist_sq.shape[1], offset=1)
        distances = torch.sqrt(dist_sq[triu_idx[0], triu_idx[1]] + self.config.kernel_floor)

        if distances.numel() == 0:
            return 1.0

        median_dist = float(distances.median())
        # sigma = median / sqrt(2) is standard heuristic
        return max(median_dist / 1.414, 0.1)


class ConformalMetric:
    """
    Track 3 (Blinker variance) as conformal factor for distance computation.

    Interpretation:
        - Track 2 = Coordinates (where in semantic space)
        - Track 1.5 = Orientation (which direction of tension)
        - Track 3 = Temperature (stability/variance)

    High variance (hot) -> low density -> stretched distances (hard to connect)
    Low variance (cold) -> high density -> compressed distances (easy to connect)

    Conformal metric:
        g_ij = (1/rho_i * 1/rho_j) * delta_ij
        D_conformal(i,j) = D(i,j) / sqrt(rho_i * rho_j)
    """

    def __init__(self, temperature_scale: float = 1.0):
        self.temperature_scale = temperature_scale

    def compute_density(self, blinker_variance: torch.Tensor) -> torch.Tensor:
        """
        Convert Track 3 variance to density factor rho.

        rho = 1 / (1 + temperature_scale * ||blinker||)

        Args:
            blinker_variance: [N, D] blinker features (fused_std from Dirichlet)

        Returns:
            rho: [N] density factors in (0, 1]
        """
        # Compute variance magnitude per sample
        if blinker_variance.dim() == 1:
            variance_magnitude = blinker_variance.abs()
        else:
            variance_magnitude = blinker_variance.norm(dim=-1)  # [N]

        # Convert to density: high variance -> low density
        rho = 1.0 / (1.0 + self.temperature_scale * variance_magnitude)

        return rho

    def apply_conformal_scaling(
        self,
        distance_matrix: torch.Tensor,  # [N, N]
        rho: torch.Tensor,               # [N]
    ) -> torch.Tensor:
        """
        Scale distances by conformal factor.

        D_conformal(i,j) = D(i,j) / sqrt(rho_i * rho_j)

        High variance nodes have low rho, so their distances stretch.
        Low variance nodes have high rho, so their distances compress.

        Args:
            distance_matrix: [N, N] pairwise distances
            rho: [N] density factors

        Returns:
            D_conformal: [N, N] conformally scaled distances
        """
        # Compute outer product: sqrt(rho_i * rho_j)
        rho_outer = torch.sqrt(torch.outer(rho, rho))

        # Scale distances (avoid division by zero)
        D_conformal = distance_matrix / rho_outer.clamp(min=1e-6)

        return D_conformal


def compute_kernel_with_conformal(
    track2: torch.Tensor,       # [N, D] hologram
    track15: torch.Tensor,      # [N, D] antagonism
    track3: torch.Tensor,       # [N, D] blinker variance
    config: Optional[HadamardFusionConfig] = None,
) -> HadamardFusionResult:
    """
    Full Hadamard fusion pipeline with conformal scaling.

    Steps:
    1. Compute Hadamard kernel: K = K_rks * K_spectral
    2. Convert to distance: D = sqrt(2 - 2*K)
    3. Apply conformal scaling: D_conf = D / sqrt(rho_i * rho_j)
    4. Convert back to kernel: K_conf = exp(-D_conf^2 / 2*sigma^2)
    5. Spectral embed: X = V @ sqrt(Lambda)

    Args:
        track2: [N, D_rks] hologram features
        track15: [N, D_spectral] antagonism features
        track3: [N, D_blinker] blinker variance features
        config: Optional configuration

    Returns:
        HadamardFusionResult with all intermediate values
    """
    config = config or HadamardFusionConfig()
    hf = HadamardFusion(config)
    conformal = ConformalMetric(temperature_scale=config.temperature_scale)

    # Step 1: Compute kernel matrices
    K_rks, K_spectral = hf.compute_kernel_matrices(track2, track15)

    # Step 2: Hadamard product with dark manifold protection
    K_hadamard, n_rescued, isolated_idx = hf.hadamard_product(K_rks, K_spectral)

    # Step 3: Kernel to distance
    # For normalized RBF kernel: D^2 = 2 - 2*K (when ||x|| = ||y|| = 1)
    # For general RBF: D = sqrt(-2 * sigma^2 * log(K)) but this can be unstable
    # Use the relationship: K = exp(-D^2 / 2*sigma^2) -> D = sigma * sqrt(-2*log(K))
    # Simpler: use kernel-induced distance D = sqrt(K(x,x) + K(y,y) - 2*K(x,y))
    # Since diagonal is 1: D = sqrt(2 - 2*K)
    D_hadamard = torch.sqrt((2 - 2 * K_hadamard).clamp(min=0))

    # Step 4: Conformal scaling from Track 3
    rho = conformal.compute_density(track3)
    D_conformal = conformal.apply_conformal_scaling(D_hadamard, rho)

    # Step 5: Distance back to kernel
    sigma_conf = config.sigma_conformal
    if sigma_conf is None:
        # Median heuristic on conformal distances
        triu_idx = torch.triu_indices(D_conformal.shape[0], D_conformal.shape[1], offset=1)
        distances = D_conformal[triu_idx[0], triu_idx[1]]
        sigma_conf = float(distances.median()) / 1.414 if distances.numel() > 0 else 1.0
        sigma_conf = max(sigma_conf, 0.1)

    K_final = torch.exp(-D_conformal**2 / (2 * sigma_conf**2))

    # Step 6: Spectral embedding
    embeddings, eigenvalues = hf.spectral_embedding(K_final, config.output_dim)

    return HadamardFusionResult(
        K_final=K_final,
        embeddings=embeddings,
        K_rks=K_rks,
        K_spectral=K_spectral,
        K_hadamard=K_hadamard,
        rho=rho,
        n_isolated_rescued=n_rescued,
        isolated_indices=isolated_idx,
        eigenvalues=eigenvalues,
    )


def compute_hadamard_fusion_simple(
    track2: torch.Tensor,       # [N, D] hologram
    track15: torch.Tensor,      # [N, D] antagonism
    config: Optional[HadamardFusionConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Simplified Hadamard fusion without conformal scaling.

    Use this when Track 3 is not available or you want just the
    Hadamard kernel product.

    Args:
        track2: [N, D_rks] hologram features
        track15: [N, D_spectral] antagonism features
        config: Optional configuration

    Returns:
        embeddings: [N, output_dim] spectral embeddings
        diagnostics: Dict with fusion metrics
    """
    config = config or HadamardFusionConfig()
    hf = HadamardFusion(config)

    # Compute kernels
    K_rks, K_spectral = hf.compute_kernel_matrices(track2, track15)

    # Hadamard product
    K_hadamard, n_rescued, isolated_idx = hf.hadamard_product(K_rks, K_spectral)

    # Spectral embedding
    embeddings, eigenvalues = hf.spectral_embedding(K_hadamard, config.output_dim)

    diagnostics = {
        'n_samples': track2.shape[0],
        'n_isolated_rescued': n_rescued,
        'K_hadamard_sparsity': float((K_hadamard < 0.01).sum() / K_hadamard.numel()),
        'top_eigenvalues': eigenvalues[:5].tolist() if eigenvalues is not None else None,
        'eigenvalue_ratio': float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues is not None else None,
    }

    return embeddings, diagnostics
