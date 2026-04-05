"""
kernel_context.py - The Epistemic Anchor

Defines KernelContext: the "Laws of Physics" for a semantic manifold run.

CORE PRINCIPLE:
--------------
Kernels are NOT observers. They are geometry regimes ("universes").
If the kernel changes, you are in a different universe.

OBSERVER TAXONOMY:
-----------------
- V-Observers: Views (the 8 framings/bots) - generate points in R^H
- M-Observers: Measurement instrument (RKS basis Ω,b) - must be deterministic/lockable
- O-Observers: Operators (Dirichlet α, weights, probes) - move through fixed geometry

Every artifact is conditioned on:
    (text, V-Observers, KernelContext, M-Observer basis, O-Observer settings)

USAGE:
------
    ctx = KernelContext(
        kernel_type='rbf',
        input_dim=768,
        rks_dim=2048,
        seed=42,
        bandwidth=1.0,
    )
    ctx.validate()
    
    # Use canonical_id for output paths
    output_dir = f"results/kernel={ctx.canonical_id()}/..."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


# =============================================================================
# Supported Kernel Types (Phase 1)
# =============================================================================

SUPPORTED_KERNELS_PHASE1 = frozenset({'cosine', 'linear', 'rbf'})
PLACEHOLDER_KERNELS = frozenset({'matern', 'poly', 'rq', 'imq', 'laplacian'})


@dataclass
class KernelContext:
    """
    The 'Laws of Physics' for a semantic manifold run.
    
    This is the ONLY way kernels should be configured. All kernel-related
    code must accept a KernelContext rather than scattered parameters.
    
    Attributes:
        kernel_type: One of 'cosine', 'linear', 'rbf' (Phase 1)
        input_dim: Input dimension (768 for CLS embeddings)
        rks_dim: Output dimension for RKS projection (required for rbf)
        seed: Deterministic seed for basis generation
        bandwidth: Kernel bandwidth σ (required for rbf)
        nu: Matérn smoothness parameter (placeholder, Phase 2)
        degree: Polynomial degree (placeholder, Phase 2)
        notes: Optional notes for provenance
    """
    kernel_type: str
    input_dim: int
    seed: int = 42
    
    # RKS parameters (required for rbf)
    rks_dim: Optional[int] = None
    bandwidth: Optional[float] = None
    
    # Placeholders for future kernels (Phase 2+)
    nu: Optional[float] = None          # Matérn smoothness
    degree: Optional[int] = None        # Polynomial degree
    
    # Provenance
    notes: str = ""
    
    def validate(self) -> None:
        """
        Validate the kernel configuration.
        
        Raises:
            ValueError: If configuration is invalid
            NotImplementedError: If kernel type is placeholder (Phase 2+)
        """
        kt = self.kernel_type.lower()
        
        # Check Phase 1 support
        if kt in PLACEHOLDER_KERNELS:
            raise NotImplementedError(
                f"Kernel type '{kt}' is planned for Phase 2+. "
                f"Phase 1 supports: {sorted(SUPPORTED_KERNELS_PHASE1)}"
            )
        
        if kt not in SUPPORTED_KERNELS_PHASE1:
            raise ValueError(
                f"Unknown kernel type '{kt}'. "
                f"Supported: {sorted(SUPPORTED_KERNELS_PHASE1)}"
            )
        
        # Validate input_dim
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        
        # Kernel-specific validation
        if kt == 'rbf':
            if self.rks_dim is None:
                raise ValueError("rbf kernel requires rks_dim")
            if self.bandwidth is None:
                raise ValueError("rbf kernel requires bandwidth (sigma)")
            if self.rks_dim <= 0:
                raise ValueError(f"rks_dim must be positive, got {self.rks_dim}")
            if self.bandwidth <= 0:
                raise ValueError(f"bandwidth must be positive, got {self.bandwidth}")
        
        # cosine/linear: rks_dim and bandwidth may be None (no projection needed)
    
    def context_hash(self) -> str:
        """
        Compute hash of configuration fields ONLY (not basis Ω,b).
        
        This identifies the "universe" - two contexts with same hash
        are in the same geometry regime.
        """
        data = {
            'kernel_type': self.kernel_type.lower(),
            'input_dim': self.input_dim,
            'seed': self.seed,
            'rks_dim': self.rks_dim,
            'bandwidth': self.bandwidth,
            'nu': self.nu,
            'degree': self.degree,
        }
        dump = json.dumps(data, sort_keys=True)
        return hashlib.sha256(dump.encode()).hexdigest()[:16]
    
    def canonical_id(self) -> str:
        """
        Deterministic string for output directory naming.
        
        Format: k_{type}_d{rks_dim}_s{seed}_b{bandwidth}_h{hash[:8]}
        
        Examples:
            - k_cosine_s42_h1a2b3c4d
            - k_rbf_d2048_s42_b1.0_h5e6f7g8h
        """
        kt = self.kernel_type.lower()
        parts = [f"k_{kt}"]
        
        if self.rks_dim is not None:
            parts.append(f"d{self.rks_dim}")
        
        parts.append(f"s{self.seed}")
        
        if self.bandwidth is not None:
            # Format bandwidth nicely (avoid floating point weirdness)
            bw_str = f"{self.bandwidth:.4g}".replace('.', 'p')
            parts.append(f"b{bw_str}")
        
        parts.append(f"h{self.context_hash()[:8]}")
        
        return "_".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'kernel_type': self.kernel_type,
            'input_dim': self.input_dim,
            'seed': self.seed,
            'rks_dim': self.rks_dim,
            'bandwidth': self.bandwidth,
            'nu': self.nu,
            'degree': self.degree,
            'notes': self.notes,
            'context_hash': self.context_hash(),
            'canonical_id': self.canonical_id(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelContext':
        """Create from dictionary."""
        # Filter to only constructor args
        valid_keys = {'kernel_type', 'input_dim', 'seed', 'rks_dim', 
                      'bandwidth', 'nu', 'degree', 'notes'}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
    
    def requires_basis(self) -> bool:
        """Check if this kernel type requires an RKS basis (Ω, b)."""
        return self.kernel_type.lower() == 'rbf'
    
    def get_output_dim(self) -> int:
        """Get the output dimension after kernel mapping."""
        if self.kernel_type.lower() in ('cosine', 'linear'):
            return self.input_dim  # No projection
        elif self.rks_dim is not None:
            return self.rks_dim
        else:
            raise ValueError(f"Cannot determine output_dim for {self.kernel_type}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_rbf_context(
    input_dim: int = 768,
    rks_dim: int = 2048,
    bandwidth: float = 1.0,
    seed: int = 42,
    notes: str = "",
) -> KernelContext:
    """Create a validated RBF kernel context."""
    ctx = KernelContext(
        kernel_type='rbf',
        input_dim=input_dim,
        rks_dim=rks_dim,
        bandwidth=bandwidth,
        seed=seed,
        notes=notes,
    )
    ctx.validate()
    return ctx


def create_cosine_context(
    input_dim: int = 768,
    seed: int = 42,
    notes: str = "",
) -> KernelContext:
    """Create a validated cosine kernel context (no RKS projection)."""
    ctx = KernelContext(
        kernel_type='cosine',
        input_dim=input_dim,
        seed=seed,
        notes=notes,
    )
    ctx.validate()
    return ctx


def create_linear_context(
    input_dim: int = 768,
    seed: int = 42,
    notes: str = "",
) -> KernelContext:
    """Create a validated linear kernel context (no RKS projection)."""
    ctx = KernelContext(
        kernel_type='linear',
        input_dim=input_dim,
        seed=seed,
        notes=notes,
    )
    ctx.validate()
    return ctx


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'KernelContext',
    'SUPPORTED_KERNELS_PHASE1',
    'PLACEHOLDER_KERNELS',
    'create_rbf_context',
    'create_cosine_context',
    'create_linear_context',
]
