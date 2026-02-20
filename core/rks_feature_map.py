"""
rks_feature_map.py

Reusable kernel feature map utility supporting:
- Injected shared basis (for CRN/reproducibility)
- Both 2D [N, H] and 3D [N, B, H] inputs
- Legacy multi-framing mode (via MultiFramingRKS)
- Direct RKS mode (no framing structure)

USAGE MODES:
1. Legacy framing mode (default): Uses MultiFramingRKS for structured logits
2. Direct mode: Simple RKS projection for CLS embeddings
3. Injected basis mode: Accepts external (omega, b) for CRN

This file provides the kernel mapping used by:
- complete_pipeline.py (legacy path)
- dirichlet_fusion.py (new CLS path)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .multi_framing_rks import MultiFramingRKS


@dataclass
class RKSConfig:
    input_dim: int
    output_dim: int
    n_framings: int = 8
    dims_per_framing: Optional[int] = None
    output_per_framing: Optional[int] = None

    kernel_type: str = "rbf"
    sigma: Optional[float] = None
    gamma: Optional[float] = None  # forwarded via kernel_params for kernels that use it
    kernel_params: Optional[Dict[str, Any]] = None

    random_seed: int = 42
    device: str = "cuda"

    auto_sigma: bool = True
    verbose: bool = False
    
    # NEW: Direct mode (no framing structure)
    direct_mode: bool = False
    
    # NEW: Injected basis for CRN
    injected_omega: Optional[torch.Tensor] = None
    injected_bias: Optional[torch.Tensor] = None
    
    # NEW: KernelContext integration
    kernel_ctx: Optional[Any] = None  # KernelContext for provenance
    
    def __post_init__(self):
        """Apply KernelContext overrides if provided."""
        if self.kernel_ctx is not None:
            try:
                self.kernel_type = self.kernel_ctx.kernel_type
                if self.kernel_ctx.bandwidth is not None:
                    self.sigma = self.kernel_ctx.bandwidth
                if self.kernel_ctx.rks_dim is not None:
                    self.output_dim = self.kernel_ctx.rks_dim
                self.random_seed = self.kernel_ctx.seed
            except AttributeError:
                pass  # kernel_ctx doesn't have expected attributes


class SharedBasis(nn.Module):
    """
    Shared RKS basis that can be saved/loaded for CRN (Common Random Numbers).
    
    This ensures reproducibility across conditions by using the same (omega, b).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        omega: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        
        if omega is not None and bias is not None:
            # Use injected basis
            assert omega.shape == (input_dim, output_dim), f"omega shape mismatch: {omega.shape}"
            assert bias.shape == (output_dim,), f"bias shape mismatch: {bias.shape}"
            self.register_buffer('omega', omega.clone())
            self.register_buffer('b', bias.clone())
        else:
            # Generate deterministic basis
            gen = torch.Generator().manual_seed(seed)
            self.register_buffer('omega', torch.randn(input_dim, output_dim, generator=gen))
            self.register_buffer('b', torch.rand(output_dim, generator=gen) * 2 * np.pi)
        
        self._sigma: Optional[float] = None
        self._hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for provenance tracking."""
        data = torch.cat([self.omega.flatten(), self.b]).cpu().numpy()
        return hashlib.md5(data.tobytes()).hexdigest()[:12]
    
    @property
    def basis_hash(self) -> str:
        return self._hash
    
    def set_sigma(self, sigma: float):
        self._sigma = sigma
    
    def estimate_sigma(self, X: torch.Tensor, policy: str = 'median') -> float:
        """Estimate bandwidth from data."""
        with torch.no_grad():
            flat = X.reshape(-1, self.input_dim)
            n = min(500, flat.shape[0])
            idx = torch.randperm(flat.shape[0])[:n]
            X_sub = flat[idx]
            
            dists = torch.cdist(X_sub, X_sub)
            mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            
            if dists[mask].numel() == 0:
                return 1.0
            
            if policy == 'median':
                sigma = float(torch.median(dists[mask]).item())
            else:
                sigma = float(torch.mean(dists[mask]).item())
            
            return max(sigma, 1e-6)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project to RKHS: phi(x) = sqrt(2/D) * cos(x @ omega/sigma + b)
        
        Supports both 2D [N, H] and 3D [N, B, H] inputs.
        """
        if self._sigma is None:
            raise ValueError("Sigma not set. Call set_sigma() first.")
        
        omega = self.omega.to(X.device) / self._sigma
        b = self.b.to(X.device)
        
        # Handle both 2D and 3D inputs
        original_shape = X.shape
        if X.dim() == 3:
            N, B, H = X.shape
            X = X.reshape(N * B, H)
        
        proj = X @ omega + b
        scale = np.sqrt(2.0 / self.output_dim)
        result = scale * torch.cos(proj)
        
        if len(original_shape) == 3:
            result = result.reshape(N, B, -1)
        
        return result
    
    def save(self, path: str):
        """Save basis for reuse."""
        torch.save({
            'omega': self.omega.cpu(),
            'b': self.b.cpu(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seed': self.seed,
            'sigma': self._sigma,
            'hash': self._hash,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'SharedBasis':
        """Load pre-generated basis."""
        data = torch.load(path, map_location='cpu')
        basis = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            seed=data['seed'],
            omega=data['omega'],
            bias=data['b'],
        )
        basis._sigma = data.get('sigma')
        basis._hash = data.get('hash', basis._compute_hash())
        return basis


class RKSFeatureMap:
    """
    Deterministic multi-framing Random Kitchen Sinks feature map.

    MODES:
    1. Legacy framing mode (default): Uses MultiFramingRKS for structured inputs
    2. Direct mode (direct_mode=True): Simple RKS for flat/block inputs
    3. Injected basis mode: Use external basis for CRN reproducibility

    Important invariants:
    - Never mixes CLS and logits channels (that's upstream in NLI extraction).
    - Preserves framing block structure in legacy mode.
    - Direct mode supports both [N, H] and [N, B, H] inputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_type: str = "rbf",
        gamma: Optional[float] = None,
        sigma: Optional[float] = None,
        n_framings: int = 8,
        dims_per_framing: Optional[int] = None,
        output_per_framing: Optional[int] = None,
        random_seed: int = 42,
        device: str = "cuda",
        kernel_params: Optional[Dict[str, Any]] = None,
        auto_sigma: bool = True,
        verbose: bool = False,
        # NEW: Direct mode (bypass MultiFramingRKS)
        direct_mode: bool = False,
        # NEW: Injected basis for CRN
        shared_basis: Optional[SharedBasis] = None,
        injected_omega: Optional[torch.Tensor] = None,
        injected_bias: Optional[torch.Tensor] = None,
        **_ignored: Any,
    ):
        self.cfg = RKSConfig(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            n_framings=int(n_framings),
            dims_per_framing=(int(dims_per_framing) if dims_per_framing is not None else None),
            output_per_framing=(int(output_per_framing) if output_per_framing is not None else None),
            kernel_type=str(kernel_type),
            sigma=(float(sigma) if sigma is not None else None),
            gamma=(float(gamma) if gamma is not None else None),
            kernel_params=(dict(kernel_params) if kernel_params is not None else None),
            random_seed=int(random_seed),
            device=str(device),
            auto_sigma=bool(auto_sigma),
            verbose=bool(verbose),
            direct_mode=bool(direct_mode),
            injected_omega=injected_omega,
            injected_bias=injected_bias,
        )
        
        # Store shared basis if provided
        self._shared_basis = shared_basis
        
        # Choose mode based on config
        if direct_mode or shared_basis is not None or injected_omega is not None:
            self._mode = 'direct'
            self._build_direct()
        else:
            self._mode = 'framing'
            self._infer_dims()
            self._build()
    
    def _build_direct(self) -> None:
        """Build direct RKS (no framing structure)."""
        if self._shared_basis is not None:
            self.basis = self._shared_basis
        elif self.cfg.injected_omega is not None and self.cfg.injected_bias is not None:
            self.basis = SharedBasis(
                input_dim=self.cfg.input_dim,
                output_dim=self.cfg.output_dim,
                seed=self.cfg.random_seed,
                omega=self.cfg.injected_omega,
                bias=self.cfg.injected_bias,
            )
        else:
            self.basis = SharedBasis(
                input_dim=self.cfg.input_dim,
                output_dim=self.cfg.output_dim,
                seed=self.cfg.random_seed,
            )
        
        if self.cfg.sigma is not None:
            self.basis.set_sigma(self.cfg.sigma)
        
        self.rks = None  # Not using MultiFramingRKS in direct mode

    def _infer_dims(self) -> None:
        # Infer dims_per_framing from input_dim if not given
        if self.cfg.dims_per_framing is None:
            if self.cfg.input_dim % self.cfg.n_framings != 0:
                raise ValueError(
                    f"input_dim={self.cfg.input_dim} must be divisible by n_framings={self.cfg.n_framings} "
                    "to infer dims_per_framing safely."
                )
            self.cfg.dims_per_framing = self.cfg.input_dim // self.cfg.n_framings

        # Allocate output evenly per framing unless explicitly overridden
        if self.cfg.output_per_framing is None:
            if self.cfg.output_dim % self.cfg.n_framings != 0:
                raise ValueError(
                    f"output_dim={self.cfg.output_dim} must be divisible by n_framings={self.cfg.n_framings} "
                    "or set output_per_framing explicitly."
                )
            self.cfg.output_per_framing = self.cfg.output_dim // self.cfg.n_framings

        # Compose kernel_params, forwarding gamma if provided
        kp = dict(self.cfg.kernel_params or {})
        if self.cfg.gamma is not None and "gamma" not in kp:
            kp["gamma"] = self.cfg.gamma
        self.cfg.kernel_params = kp

    def _build(self) -> None:
        self.rks = MultiFramingRKS(
            n_framings=self.cfg.n_framings,
            dims_per_framing=int(self.cfg.dims_per_framing),
            output_per_framing=int(self.cfg.output_per_framing),
            kernel_types=self.cfg.kernel_type,
            sigma=self.cfg.sigma,
            kernel_params=self.cfg.kernel_params,
            seed=self.cfg.random_seed,
            device=self.cfg.device,
        )

    @property
    def kernel_type(self) -> str:
        return self.cfg.kernel_type

    @kernel_type.setter
    def kernel_type(self, value: str) -> None:
        # Do NOT allow silent mutation without rebuild; callers should call rebuild().
        self.cfg.kernel_type = str(value)

    def rebuild(
        self,
        kernel_type: Optional[str] = None,
        sigma: Optional[float] = None,
        kernel_params: Optional[Dict[str, Any]] = None,
        gamma: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Reinitialize the RKS kernels (needed when you change kernel_type / sigma).
        """
        if kernel_type is not None:
            self.cfg.kernel_type = str(kernel_type)
        if sigma is not None:
            self.cfg.sigma = float(sigma)
        if gamma is not None:
            self.cfg.gamma = float(gamma)
        if kernel_params is not None:
            self.cfg.kernel_params = dict(kernel_params)
        if device is not None:
            self.cfg.device = str(device)

        self._infer_dims()
        self._build()

    def transform(self, X: torch.Tensor, rep_kind: Optional[str] = None) -> torch.Tensor:
        """
        Apply the RKS map.
        
        For direct mode: handles both [N, H] and [N, B, H] inputs.
        For framing mode: returns [N, output_dim] combined output.
        
        Args:
            X: Input tensor
            rep_kind: Optional rep_kind string for contract enforcement.
                      If 'logits_raw', raises ValueError (defense-in-depth).
        """
        # DEFENSE-IN-DEPTH: Refuse to transform logits
        if rep_kind is not None and rep_kind.lower() == 'logits_raw':
            raise ValueError(
                "[CONSTITUTIONAL CONTRACT] RKSFeatureMap.transform() refuses rep_kind='logits_raw'. "
                "Logits are verdict coordinates and must not be kernelized."
            )
        
        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if self._mode == 'direct':
            return self._transform_direct(X)
        else:
            return self._transform_framing(X)
    
    def _transform_direct(self, X: torch.Tensor) -> torch.Tensor:
        """Direct mode transform using SharedBasis."""
        # Estimate sigma if needed
        if self.basis._sigma is None:
            if self.cfg.auto_sigma:
                sigma = self.basis.estimate_sigma(X)
                self.basis.set_sigma(sigma)
                if self.cfg.verbose:
                    print(f"[RKSFeatureMap] Auto-estimated sigma = {sigma:.4f}")
            else:
                raise ValueError("Sigma not set and auto_sigma=False")
        
        # Move basis to device
        if X.device != self.basis.omega.device:
            self.basis = self.basis.to(X.device)
        
        return self.basis(X)
    
    def _transform_framing(self, X: torch.Tensor) -> torch.Tensor:
        """Legacy framing mode transform using MultiFramingRKS."""
        # Ensure module is on the same device
        try:
            if next(self.rks.parameters()).device != X.device:
                self.rks = self.rks.to(X.device)
        except StopIteration:
            self.rks = self.rks.to(X.device)

        # Ensure sigma is set before forward
        if hasattr(self.rks, "estimate_and_set_sigma"):
            need_sigma = getattr(self.rks, "sigma", None) is None
            if self.cfg.auto_sigma or need_sigma:
                try:
                    _ = self.rks.estimate_and_set_sigma(X, sample_size=1000, percentile=50.0, per_framing=True)
                except Exception as e:
                    if getattr(self.rks, "sigma", None) is None:
                        raise RuntimeError(f"RKS sigma estimation failed: {e}") from e

        out = self.rks(X)
        if isinstance(out, dict):
            if "combined" in out:
                return out["combined"]
            for v in out.values():
                if torch.is_tensor(v):
                    return v
            raise RuntimeError("MultiFramingRKS returned dict without tensor outputs.")
        return out
    
    def get_basis(self) -> Optional[SharedBasis]:
        """Get the shared basis (direct mode only)."""
        if self._mode == 'direct':
            return self.basis
        return None
    
    def get_provenance(self) -> Dict[str, Any]:
        """Get provenance metadata for artifact logging."""
        prov = {
            'mode': self._mode,
            'input_dim': self.cfg.input_dim,
            'output_dim': self.cfg.output_dim,
            'kernel_type': self.cfg.kernel_type,
            'seed': self.cfg.random_seed,
        }
        
        if self._mode == 'direct' and self.basis is not None:
            prov['basis_hash'] = self.basis.basis_hash
            prov['sigma'] = self.basis._sigma
        elif self._mode == 'framing' and self.rks is not None:
            prov['sigma'] = getattr(self.rks, 'sigma', None)
            prov['n_framings'] = self.cfg.n_framings
        
        return prov

    def get_full_output(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Return the full output dict (for debugging / research).
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if self._mode == 'direct':
            features = self._transform_direct(X)
            return {
                'combined': features,
                'mode': 'direct',
                'basis_hash': self.basis.basis_hash,
                'sigma': self.basis._sigma,
            }

        # Legacy framing mode
        try:
            if next(self.rks.parameters()).device != X.device:
                self.rks = self.rks.to(X.device)
        except StopIteration:
            self.rks = self.rks.to(X.device)

        if hasattr(self.rks, "estimate_and_set_sigma"):
            need_sigma = getattr(self.rks, "sigma", None) is None
            if self.cfg.auto_sigma or need_sigma:
                try:
                    _ = self.rks.estimate_and_set_sigma(X, sample_size=1000, percentile=50.0, per_framing=True)
                except Exception as e:
                    if getattr(self.rks, "sigma", None) is None:
                        raise RuntimeError(f"RKS sigma estimation failed: {e}") from e

        out = self.rks(X)
        if isinstance(out, dict):
            out['mode'] = 'framing'
            return out
        return {"combined": out, "mode": "framing"}


# =============================================================================
# CONVENIENCE FACTORIES
# =============================================================================

def create_shared_basis(
    input_dim: int,
    output_dim: int,
    seed: int = 42,
    sigma: Optional[float] = None,
) -> SharedBasis:
    """Create a SharedBasis for CRN use."""
    basis = SharedBasis(input_dim, output_dim, seed)
    if sigma is not None:
        basis.set_sigma(sigma)
    return basis


def create_direct_rks(
    input_dim: int,
    output_dim: int,
    seed: int = 42,
    sigma: Optional[float] = None,
    shared_basis: Optional[SharedBasis] = None,
) -> RKSFeatureMap:
    """Create an RKSFeatureMap in direct mode (for CLS embeddings)."""
    return RKSFeatureMap(
        input_dim=input_dim,
        output_dim=output_dim,
        random_seed=seed,
        sigma=sigma,
        direct_mode=True,
        shared_basis=shared_basis,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'RKSConfig',
    'RKSFeatureMap',
    'SharedBasis',
    'create_shared_basis',
    'create_direct_rks',
]