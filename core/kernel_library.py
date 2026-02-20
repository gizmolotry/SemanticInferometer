"""
Multiple kernel implementations for observer comparison

Different kernels = different observers with genuinely different similarity metrics

PHASE 2 UPDATE:
--------------
This file is now the ONLY place where RKS basis (Ω, b) is created/loaded/hashed.
All other modules (dirichlet_fusion, multi_framing_rks, rks_feature_map) must
call ensure_shared_basis() or accept an injected SharedBasis.

OBSERVER TAXONOMY:
- Kernels are NOT observers - they are geometry regimes
- M-Observers = measurement instrument (the basis Ω, b)
- This file is the M-Observer factory
"""

import torch
import numpy as np
import math
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from scipy.spatial.distance import pdist

# Import KernelContext (circular import safe - only used for type hints initially)
try:
    from .kernel_context import KernelContext
except ImportError:
    KernelContext = None  # Will be imported at runtime


# =============================================================================
# Shared Basis Management (M-Observer Factory)
# =============================================================================

def compute_basis_hash(omega: torch.Tensor, bias: torch.Tensor) -> str:
    """
    Compute deterministic hash from actual basis tensors (Ω, b).
    
    This is the M-Observer fingerprint - if this matches, the bases are identical.
    """
    # Ensure deterministic byte representation
    omega_bytes = omega.detach().cpu().contiguous().float().numpy().tobytes()
    bias_bytes = bias.detach().cpu().contiguous().float().numpy().tobytes()
    
    # Include shape info to catch dimension mismatches
    shape_info = f"{omega.shape}_{bias.shape}".encode()
    
    combined = omega_bytes + bias_bytes + shape_info
    return hashlib.sha256(combined).hexdigest()[:16]


def create_rbf_basis(
    input_dim: int,
    rks_dim: int,
    bandwidth: float,
    seed: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create RBF basis tensors (Ω, b) deterministically.
    
    This is the canonical way to create basis tensors. Do NOT sample
    Ω and b anywhere else in the codebase.
    
    Args:
        input_dim: Input dimension (e.g., 768 for CLS)
        rks_dim: Output dimension (D)
        bandwidth: Kernel bandwidth σ
        seed: Random seed for reproducibility
        device: Torch device
        
    Returns:
        (omega, bias) tuple of tensors
    """
    # Use local generator to avoid polluting global RNG
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # Sample Ω from N(0, 1/σ²) - spectral measure of RBF
    omega = torch.randn(rks_dim, input_dim, generator=generator, device=device)
    omega = omega / bandwidth
    
    # Sample b uniformly from [0, 2π]
    bias = 2.0 * math.pi * torch.rand(rks_dim, generator=generator, device=device)
    
    return omega, bias


def ensure_shared_basis(
    ctx: 'KernelContext',
    basis_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[Optional['SharedBasis'], str]:
    """
    The M-Observer factory. Creates or loads a shared basis for the given context.
    
    This is the ONLY function that should create/load basis tensors.
    
    Args:
        ctx: KernelContext defining the geometry regime
        basis_path: Optional path to load pre-existing basis
        device: Torch device
        
    Returns:
        (basis, basis_hash) tuple. basis is None for cosine/linear kernels.
    """
    # Import here to avoid circular imports
    from .rks_feature_map import SharedBasis
    
    kt = ctx.kernel_type.lower()
    
    # Cosine/linear kernels don't need a basis
    if kt in ('cosine', 'linear'):
        return (None, "")
    
    # RBF kernel needs basis
    if kt == 'rbf':
        if basis_path is not None and Path(basis_path).exists():
            # Load existing basis
            basis = SharedBasis.load(basis_path)
            basis_hash = compute_basis_hash(basis.omega, basis.b)
            print(f"[M-OBSERVER] Loaded basis from {basis_path}, hash={basis_hash}")
        else:
            # Create new basis deterministically
            omega, bias = create_rbf_basis(
                input_dim=ctx.input_dim,
                rks_dim=ctx.rks_dim,
                bandwidth=ctx.bandwidth,
                seed=ctx.seed,
                device=device,
            )
            basis = SharedBasis(
                input_dim=ctx.input_dim,
                output_dim=ctx.rks_dim,
                seed=ctx.seed,
                omega=omega.t(),  # SharedBasis expects [input_dim, output_dim]
                bias=bias,
            )
            basis.set_sigma(ctx.bandwidth)
            basis_hash = compute_basis_hash(omega, bias)
            print(f"[M-OBSERVER] Created RBF basis: D={ctx.rks_dim}, σ={ctx.bandwidth}, hash={basis_hash}")
        
        return (basis, basis_hash)
    
    # Placeholder kernels
    raise NotImplementedError(f"Kernel type '{kt}' not supported in Phase 1")


# =============================================================================
# Bandwidth Estimation (Migrated from rks_expansion.py)
# =============================================================================

def estimate_rbf_sigma(
    features: torch.Tensor,
    sample_size: int = 1000,
    percentile: float = 50.0
) -> float:
    """
    Estimate RBF kernel bandwidth from data using median heuristic.
    
    This replaces the version in rks_expansion.py to break dependency chains.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature matrix [N, D]
    sample_size : int
        Number of samples to use (for efficiency with large N)
    percentile : float
        Percentile to use (50.0 = median)
        
    Returns
    -------
    sigma : float
        Estimated bandwidth
    """
    # Detach and move to CPU for scipy
    if torch.is_tensor(features):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.array(features)
    
    # Sample if dataset is large
    if len(features_np) > sample_size:
        idx = np.random.choice(len(features_np), sample_size, replace=False)
        features_np = features_np[idx]
        
    # Compute pairwise distances
    distances = pdist(features_np, metric='euclidean')
    
    # Handle degenerate data (e.g. constant control)
    if len(distances) == 0:
        return 1.0
        
    if distances.max() < 1e-8:
        print("  [Kernel] Warning: Degenerate data (distances ≈ 0). Using sigma=1.0")
        return 1.0
        
    sigma = np.percentile(distances, percentile)
    return float(max(sigma, 1e-6))


# =============================================================================
# Kernel Implementations
# =============================================================================

class KernelBase:
    """Base class for all kernels"""
    
    def __init__(self, input_dim, output_dim, sigma, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Random frequencies (kernel-specific distribution)
        self.W, self.b = self._sample_frequencies()
    
    def _sample_frequencies(self):
        """Sample random frequencies - override in subclass"""
        raise NotImplementedError
    
    def transform(self, X):
        """Transform features using random features"""
        # X: [batch, input_dim]
        # W: [output_dim, input_dim]
        # Result: [batch, output_dim]
        
        if torch.is_tensor(X):
            device = X.device
            W = self.W.to(device)
            b = self.b.to(device)
        else:
            X = torch.FloatTensor(X)
            W = self.W
            b = self.b
        
        # z = cos(W @ x + b)
        projection = X @ W.T + b  # [batch, output_dim]
        features = torch.cos(projection)
        
        # Normalize
        features = features * math.sqrt(2.0 / self.output_dim)
        
        return features

    def approximation_diagnostic(self, X: torch.Tensor, n_samples: int = 1000) -> Dict[str, float]:
        """
        Compare RKS approximation against exact kernel values.
        
        CRITICAL for validating that the observer is not hallucinating geometry.
        Computes correlation between Exact Kernel Matrix (K) and Approximate (ZZ^T).
        """
        # Sample random pairs to keep memory check in check
        n = min(len(X), n_samples)
        if n < 2: 
            return {'correlation': 0.0, 'mae': 0.0, 'rmse': 0.0}
            
        indices = torch.randperm(len(X))[:n]
        X_sample = X[indices]
        
        if torch.is_tensor(X_sample):
            X_sample = X_sample.to(self.W.device)
            
        # 1. Compute Exact Kernel (Ground Truth)
        # Note: This assumes RBF/translation-invariant structure. 
        # For complex kernels, we rely on the specific formula.
        dists_sq = torch.cdist(X_sample, X_sample, p=2)**2
        
        if isinstance(self, RBFKernel):
            K_true = torch.exp(-dists_sq / (2 * self.sigma**2))
        elif isinstance(self, LaplacianKernel):
            dists_l1 = torch.cdist(X_sample, X_sample, p=1)
            K_true = torch.exp(-dists_l1 / self.sigma)
        else:
            # Fallback for other kernels (RQ, IMQ, etc)
            # We treat RBF as the "reference truth" for sanity if specific formula not implemented
            K_true = torch.exp(-dists_sq / (2 * self.sigma**2))

        # 2. Compute Observer Approximation
        Z = self.transform(X_sample)
        K_approx = Z @ Z.T
        
        # 3. Compare
        diff = (K_true - K_approx).abs()
        mae = float(diff.mean().item())
        rmse = float(torch.sqrt((diff**2).mean()).item())
        
        # Compute Correlation (The most important metric)
        # Flatten upper triangles to compare unique pairs
        mask = torch.triu(torch.ones_like(K_true), diagonal=1).bool()
        true_vals = K_true[mask].cpu().numpy()
        approx_vals = K_approx[mask].detach().cpu().numpy()
        
        if len(true_vals) > 1:
            correlation = float(np.corrcoef(true_vals, approx_vals)[0, 1])
        else:
            correlation = 0.0
            
        return {
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'sigma': self.sigma
        }


class RBFKernel(KernelBase):
    """
    RBF (Gaussian) Kernel: K(x,y) = exp(-||x-y||²/(2σ²))
    
    Features: Sample from Gaussian distribution
    Properties: Smooth, global kernel
    """
    
    def _sample_frequencies(self):
        # Sample from N(0, 1/sigma²)
        W = torch.randn(self.output_dim, self.input_dim) / self.sigma
        b = torch.rand(self.output_dim) * 2 * math.pi
        return W, b


class LaplacianKernel(KernelBase):
    """
    Laplacian Kernel: K(x,y) = exp(-||x-y||/σ)
    
    Features: Sample from Laplace distribution
    Properties: Sharper, more local than RBF
    """
    
    def _sample_frequencies(self):
        # Sample from Laplace(0, 1/sigma)
        W = np.random.laplace(0, 1.0/self.sigma, 
                              size=(self.output_dim, self.input_dim))
        W = torch.FloatTensor(W)
        b = torch.rand(self.output_dim) * 2 * math.pi
        return W, b


class RationalQuadraticKernel(KernelBase):
    """
    Rational Quadratic Kernel: K(x,y) = (1 + ||x-y||²/(2ασ²))^(-α)
    
    Features: Mixture of RBF kernels with different scales
    Properties: α controls smoothness
      - α→∞: approaches RBF
      - α=1: Cauchy kernel (very sharp)
    """
    
    def __init__(self, input_dim, output_dim, sigma, alpha=1.0, seed=42):
        self.alpha = alpha
        super().__init__(input_dim, output_dim, sigma, seed)
    
    def _sample_frequencies(self):
        # Sample mixture of Gaussians with gamma-distributed scales
        scales = np.random.gamma(self.alpha, 1.0/self.alpha, 
                                 size=self.output_dim)
        
        # Sample frequencies with varying scales
        W_list = []
        for scale in scales:
            w = torch.randn(self.input_dim) / (self.sigma * np.sqrt(scale))
            W_list.append(w)
        
        W = torch.stack(W_list, dim=0)
        b = torch.rand(self.output_dim) * 2 * math.pi
        
        return W, b


class IMQKernel(KernelBase):
    """
    Inverse Multiquadratic Kernel: K(x,y) = 1 / sqrt(1 + ||x-y||²/σ²)

    Properties: Heavy-tailed, good for outliers
    Polynomial decay (vs exponential for RBF)

    Spectral measure: Student-t distribution (fat tails capture structural breaks).
    The `roughness` parameter controls degrees of freedom:
      - roughness=1 (df=1): Cauchy-level fat tails (maximum jaggedness)
      - roughness=3 (df=3): Moderate fat tails (default)
      - roughness→∞: approaches Gaussian (smooth, loses crack detection)
    """

    def __init__(self, input_dim, output_dim, sigma, roughness=3, seed=42):
        self.roughness = roughness  # degrees of freedom for Student-t
        super().__init__(input_dim, output_dim, sigma, seed)

    def _sample_frequencies(self):
        # Sample from Student-t distribution (fat tails for crack detection)
        df = max(1, self.roughness)

        W = np.random.standard_t(df, size=(self.output_dim, self.input_dim))

        # VARIANCE MATCHING (Holographic Truth Protocol, Axiom 1):
        # Raw Student-t(df) has variance df/(df-2) > 1 for finite df.
        # Without correction, this amplifies scale (Zoom Illusion) rather
        # than just changing texture (roughness). We normalize to unit variance
        # so that roughness is isolated from magnitude.
        if df > 2:
            correction = np.sqrt((df - 2) / df)
            W = W * correction

        W = torch.FloatTensor(W) / self.sigma
        b = torch.rand(self.output_dim) * 2 * math.pi

        return W, b


class MaternKernel(KernelBase):
    """
    Matérn Kernel: Parameterized by smoothness ν

    ν = 1/2: Laplacian (non-differentiable, maximum roughness)
    ν = 3/2: Once differentiable
    ν = 5/2: Twice differentiable
    ν → ∞: RBF (infinitely differentiable)

    Spectral density of Matérn-ν is Student-t with df=2ν.
    This naturally produces fat tails for low ν (rough geometries),
    approaching Gaussian only as ν→∞.

    The `nu` parameter IS the roughness control:
      - nu=0.5: Maximum roughness (Laplace/Cauchy tails)
      - nu=1.5: Default, captures cracks while remaining stable
      - nu=2.5: Moderate smoothness
      - nu→∞: Gaussian (too smooth for crack detection)
    """

    def __init__(self, input_dim, output_dim, sigma, nu=1.5, seed=42):
        self.nu = nu
        super().__init__(input_dim, output_dim, sigma, seed)

    def _sample_frequencies(self):
        if self.nu == 0.5:
            # Exact Laplacian case (variance = 2/σ², already correct)
            W = np.random.laplace(0, 1.0 / self.sigma,
                                  size=(self.output_dim, self.input_dim))
        else:
            # Correct spectral density: Student-t with df=2*nu
            # This is the EXACT Fourier dual of the Matérn covariance.
            # Low nu → fat tails → jagged geometry (crack-sensitive)
            # High nu → thin tails → smooth geometry (approaches RBF)
            df = max(1, 2 * self.nu)
            W = np.random.standard_t(df, size=(self.output_dim, self.input_dim))

            # VARIANCE MATCHING (Holographic Truth Protocol, Axiom 1):
            # Normalize Student-t to unit variance before applying spectral scaling.
            # This isolates roughness (texture) from magnitude (zoom).
            if df > 2:
                correction = np.sqrt((df - 2) / df)
                W = W * correction

            # Spectral scaling for Matérn
            scale = math.sqrt(2 * self.nu) / self.sigma
            W = W * scale

        W = torch.FloatTensor(W)
        b = torch.rand(self.output_dim) * 2 * math.pi

        return W, b


def create_kernel(kernel_type, input_dim, output_dim, sigma, seed=42, **kwargs):
    """
    Factory function to create kernels
    
    Parameters:
    -----------
    kernel_type : str
        'rbf', 'laplacian', 'rq', 'imq', or 'matern'
    sigma : float
        Kernel bandwidth
    seed : int
        Random seed
    **kwargs : dict
        Kernel-specific parameters:
          - alpha (for RQ)
          - nu (for Matérn, controls smoothness/roughness)
          - roughness (for IMQ, Student-t degrees of freedom)
    """
    # Accept sklearn-style `gamma` for RBF-like kernels by converting (or ignoring) it.
    gamma = kwargs.pop('gamma', None)
    if gamma is not None and sigma is None:
        try:
            gamma_f = float(gamma)
            if gamma_f > 0:
                sigma = (1.0 / (2.0 * gamma_f)) ** 0.5
        except Exception:
            # If gamma is malformed, just ignore it and rely on provided sigma.
            pass

    
    kernel_map = {
        'rbf': RBFKernel,
        'laplacian': LaplacianKernel,
        'rq': RationalQuadraticKernel,
        'rational_quadratic': RationalQuadraticKernel,
        'imq': IMQKernel,
        'matern': MaternKernel
    }
    
    if kernel_type not in kernel_map:
        raise ValueError(f"Unknown kernel: {kernel_type}. "
                        f"Choose from {list(kernel_map.keys())}")
    
    kernel_class = kernel_map[kernel_type]
    return kernel_class(input_dim, output_dim, sigma, seed=seed, **kwargs)


# Kernel descriptions for user
KERNEL_INFO = {
    'rbf': {
        'name': 'RBF (Gaussian)',
        'formula': 'exp(-||x-y||²/(2σ²))',
        'properties': 'Smooth, global, standard choice',
        'params': ['sigma']
    },
    'laplacian': {
        'name': 'Laplacian',
        'formula': 'exp(-||x-y||/σ)',
        'properties': 'Sharper, more local than RBF',
        'params': ['sigma']
    },
    'rq': {
        'name': 'Rational Quadratic',
        'formula': '(1 + ||x-y||²/(2ασ²))^(-α)',
        'properties': 'Tunable smoothness via α',
        'params': ['sigma', 'alpha']
    },
    'imq': {
        'name': 'Inverse Multiquadratic',
        'formula': '1 / sqrt(1 + ||x-y||²/σ²)',
        'properties': 'Heavy-tailed, captures outliers',
        'params': ['sigma']
    },
    'matern': {
        'name': 'Matérn',
        'formula': 'Complex (Bessel function)',
        'properties': 'Smoothness controlled by ν',
        'params': ['sigma', 'nu']
    }
}

# Export list of supported kernels for diagnostics
SUPPORTED_KERNELS = list(KERNEL_INFO.keys())


def print_kernel_info():
    """Print information about available kernels"""
    print("="*70)
    print("AVAILABLE KERNELS")
    print("="*70)
    for ktype, info in KERNEL_INFO.items():
        print(f"\n{ktype.upper()}: {info['name']}")
        print(f"  Formula: {info['formula']}")
        print(f"  Properties: {info['properties']}")
        print(f"  Parameters: {', '.join(info['params'])}")
    print("="*70)


if __name__ == "__main__":
    # Demo
    print_kernel_info()
    
    # Test each kernel
    print("\nTesting kernels on sample data...")
    X = torch.randn(100, 8)  # 100 samples, 8D
    
    for ktype in ['rbf', 'laplacian', 'rq', 'imq']:
        kernel = create_kernel(ktype, input_dim=8, output_dim=512, sigma=0.6)
        features = kernel.transform(X)
        print(f"{ktype:15s}: {features.shape}, norm={features.norm(dim=-1).mean():.4f}")
        
        # Run diagnostic
        diag = kernel.approximation_diagnostic(X, n_samples=100)
        print(f"  Diagnostic: Corr={diag['correlation']:.4f}, MAE={diag['mae']:.4f}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Basis management (M-Observer factory)
    'compute_basis_hash',
    'create_rbf_basis',
    'ensure_shared_basis',
    'estimate_rbf_sigma',
    # Legacy kernel classes
    'KernelBase',
    'RBFKernel',
    'LaplacianKernel',
    'RationalQuadraticKernel',
    'IMQKernel',
    'MaternKernel',
    'create_kernel',
    'KERNEL_INFO',
    'SUPPORTED_KERNELS',
    'print_kernel_info',
]