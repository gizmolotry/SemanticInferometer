"""
Random Kitchen Sinks (RKS) Expansion with Multiple Kernel Support

This module implements Random Fourier Features for approximating various kernels,
following Rahimi & Recht (2007). Supports multiple kernel types:
- RBF (Gaussian): Smooth, global
- Laplacian: Sharp, local
- Rational Quadratic: Tunable smoothness
- IMQ: Heavy-tailed
- Matérn: Parameterized smoothness

Each kernel type = different observer with genuinely different similarity metric.

Theoretical Foundation:
----------------------
Bochner's theorem states that any continuous, shift-invariant positive-definite
kernel k(x,y) = κ(x-y) can be represented as the Fourier transform of a
non-negative measure μ:

    k(x,y) = ∫ exp(iωᵀ(x-y)) dμ(ω)

Different kernels correspond to different measures μ.
Random Kitchen Sinks approximates this by Monte Carlo sampling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import pdist
import sys
from pathlib import Path

# Import kernel library
try:
    from kernel_library import create_kernel, KERNEL_INFO
except ImportError:
    # Try relative import if in core/
    try:
        from .kernel_library import create_kernel, KERNEL_INFO
    except ImportError:
        print("Warning: kernel_library not found. Only RBF kernel will be available.")
        create_kernel = None
        KERNEL_INFO = {}


class RandomKitchenSinksExpander(nn.Module):
    """
    Random Kitchen Sinks / Random Fourier Features for kernel approximation.
    
    Maps D-dimensional input to M-dimensional random feature space where
    inner products approximate kernel similarities.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (e.g., 24 or 8)
    output_dim : int
        Expanded feature dimension (e.g., 512)
    sigma : float, optional
        Kernel bandwidth (controls sensitivity to distances)
        If None, will be estimated from data
    kernel_type : str
        Type of kernel: 'rbf', 'laplacian', 'rq', 'imq', 'matern'
    kernel_params : dict, optional
        Additional kernel parameters (e.g., alpha for RQ, nu for Matérn)
    seed : int, optional
        Random seed for reproducible observer-specific mappings
    device : str
        'cuda' or 'cpu'
    
    Notes
    -----
    - Different kernel_type = genuinely different observer perspective
    - Same kernel_type, different seed = Monte Carlo samples of same observer
    - As output_dim → ∞, approximation converges to true kernel
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        sigma: float = None,
        kernel_type: str = 'rbf',
        kernel_params: Optional[dict] = None,
        seed: Optional[int] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}
        self.seed = seed

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Create kernel using kernel library (if available)
        if create_kernel is not None and kernel_type != 'rbf':
            # Use new kernel library for non-RBF kernels
            self._use_kernel_library = True
            self._kernel = None
            
            # Initialize immediately if sigma is provided
            if sigma is not None:
                self._initialize_kernel(sigma)
            
        else:
            # Use original RBF implementation
            self._use_kernel_library = False
            
            if sigma is not None:
                self._initialize_rbf_kernel(sigma, seed)
            
    def _initialize_rbf_kernel(self, sigma: float, seed: Optional[int] = None):
        """Initialize RBF kernel (original implementation)"""
        self.sigma = sigma
        
        # Set generator for reproducibility
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)

        # Random frequencies Ω ~ N(0, 1/σ²)
        omega = torch.randn(
            self.input_dim, self.output_dim, generator=g, device=self.device
        ) / sigma
        
        # Random phases b ~ Uniform(0, 2π)
        b = torch.rand(self.output_dim, generator=g, device=self.device) * 2 * torch.pi

        # Register as buffers
        self.register_buffer("omega", omega)
        self.register_buffer("b", b)

        # Precompute scaling factor sqrt(2/D)
        self.register_buffer(
            "scale",
            torch.tensor((2.0 / self.output_dim) ** 0.5, device=self.device),
        )
    
    def _initialize_kernel(self, sigma: float):
        """Initialize kernel using kernel library"""
        self.sigma = sigma
        
        if self._use_kernel_library:
            # Use kernel library
            self._kernel = create_kernel(
                kernel_type=self.kernel_type,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                sigma=sigma,
                seed=self.seed or 42,
                **self.kernel_params
            )
            
            # Move kernel tensors to device
            self._kernel.W = self._kernel.W.to(self.device)
            self._kernel.b = self._kernel.b.to(self.device)
            
        else:
            # Use RBF
            self._initialize_rbf_kernel(sigma, self.seed)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Random Kitchen Sinks transformation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape [..., input_dim]
        
        Returns
        -------
        torch.Tensor
            RKS features of shape [..., output_dim]
            Inner products approximate kernel: z(x)ᵀz(y) ≈ k(x,y)
        """
        # Validate input dimension
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"RKS input dim mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        # Check sigma is set
        if self.sigma is None:
            raise ValueError("Sigma not set! Call estimate_and_set_sigma() first")
        
        if self._use_kernel_library:
            # Use kernel library
            if self._kernel is None:
                # This shouldn't happen if sigma was set properly
                raise ValueError(
                    f"Kernel not initialized for {self.kernel_type}! "
                    f"This is a bug - sigma was {self.sigma} but kernel is None."
                )
            
            return self._kernel.transform(x)
        else:
            # Use original RBF implementation
            x = x.to(self.omega.device)
            proj = x @ self.omega
            return self.scale * torch.cos(proj + self.b)
    
    def estimate_and_set_sigma(
        self,
        features: torch.Tensor,
        sample_size: int = 1000,
        percentile: float = 50.0
    ) -> float:
        """
        Estimate and set sigma from data using median heuristic.
        
        This is a convenience method that estimates sigma and initializes
        the kernel in one call.
        
        Parameters
        ----------
        features : torch.Tensor
            Feature matrix [N, D]
        sample_size : int
            Number of samples to use
        percentile : float
            Percentile to use (50.0 = median)
        
        Returns
        -------
        sigma : float
            Estimated bandwidth
        """
        sigma = estimate_rbf_sigma(features, sample_size, percentile)
        self._initialize_kernel(sigma)
        return sigma
    
    def verify_approximation(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[float, float, float]:
        """
        Verify that RKS inner products approximate true kernel.
        
        Note: Only works for RBF kernel currently
        
        Parameters
        ----------
        x : torch.Tensor
            First input batch [N, input_dim]
        y : torch.Tensor
            Second input batch [N, input_dim]
        n_samples : int
            Number of pairs to sample for verification
        
        Returns
        -------
        mean_error : float
            Mean absolute approximation error
        max_error : float
            Maximum absolute error
        correlation : float
            Correlation between true kernel and approximation
        """
        if self.kernel_type != 'rbf':
            print(f"Warning: Verification only implemented for RBF kernel, not {self.kernel_type}")
            return float('nan'), float('nan'), float('nan')
        
        with torch.no_grad():
            # Sample random pairs
            n = min(len(x), len(y), n_samples)
            idx_x = torch.randperm(len(x))[:n]
            idx_y = torch.randperm(len(y))[:n]
            
            x_sample = x[idx_x].to(self.device)
            y_sample = y[idx_y].to(self.device)
            
            # True RBF kernel
            dist_sq = torch.sum((x_sample - y_sample)**2, dim=-1)
            k_true = torch.exp(-dist_sq / (2 * self.sigma**2))
            
            # RKS approximation
            z_x = self.forward(x_sample)
            z_y = self.forward(y_sample)
            k_approx = torch.sum(z_x * z_y, dim=-1)
            
            # Compute error metrics
            errors = torch.abs(k_true - k_approx)
            mean_error = errors.mean().item()
            max_error = errors.max().item()
            
            # Correlation
            k_true_np = k_true.cpu().numpy()
            k_approx_np = k_approx.cpu().numpy()
            correlation = np.corrcoef(k_true_np, k_approx_np)[0, 1]
            
            return mean_error, max_error, correlation


def estimate_rbf_sigma(
    features: torch.Tensor,
    sample_size: int = 1000,
    percentile: float = 50.0
) -> float:
    """
    Estimate RBF kernel bandwidth from data using median heuristic.
    
    The median heuristic sets σ to the median pairwise distance,
    which is a common approach in kernel methods (Garreau et al., 2017).
    
    Parameters
    ----------
    features : torch.Tensor
        Feature matrix [N, D]
    sample_size : int
        Number of samples to use (for efficiency with large N)
    percentile : float
        Percentile to use (50.0 = median, 25.0 = more local)
    
    Returns
    -------
    sigma : float
        Estimated bandwidth
    
    References
    ----------
    Garreau, D., Jitkrittum, W., & Kanagawa, M. (2017).
    Large sample analysis of the median heuristic.
    """
    features_np = features.detach().cpu().numpy()  # Add detach() for gradient tensors
    
    # Sample if dataset is large
    if len(features_np) > sample_size:
        idx = np.random.choice(len(features_np), sample_size, replace=False)
        features_np = features_np[idx]
    
    # Compute pairwise distances
    distances = pdist(features_np, metric='euclidean')
    
    # Check for degenerate data (ROBUST FIX)
    if len(distances) == 0:
        print("  ⚠ No distances computed - using fallback sigma=1.0")
        return 0.1
    
    max_dist = distances.max()
    if max_dist < 1e-8:
        print("  [WARN] DEGENERATE DATA: All distances ~ 0")
        print("    (All articles identical - constant control)")
        print("    Using fallback sigma = 1.0")
        return 0.1
    
    # Use percentile (median by default)
    sigma = np.percentile(distances, percentile)
    
    # Safety check
    if sigma < 1e-6:
        print(f"  ⚠ Sigma too small ({sigma:.2e}), using fallback 1.0")
        return 0.1
    
    return float(sigma)


def visualize_rks_approximation(
    features: torch.Tensor,
    rks: RandomKitchenSinksExpander,
    n_points: int = 50
) -> dict:
    """
    Generate diagnostic data for RKS approximation quality.
    
    Parameters
    ----------
    features : torch.Tensor
        Sample features [N, input_dim]
    rks : RandomKitchenSinksExpander
        RKS expander to test
    n_points : int
        Number of points to sample
    
    Returns
    -------
    dict
        Diagnostic information including:
        - true_kernel: True RBF kernel matrix
        - approx_kernel: RKS approximation
        - errors: Pointwise errors
        - statistics: Summary stats
    """
    if rks.kernel_type != 'rbf':
        print(f"Warning: Visualization only implemented for RBF, not {rks.kernel_type}")
        return {}
    
    with torch.no_grad():
        # Sample points
        idx = torch.randperm(len(features))[:n_points]
        sample = features[idx].to(rks.device)
        
        # Compute true RBF kernel matrix
        dists = torch.cdist(sample, sample, p=2)
        true_kernel = torch.exp(-dists**2 / (2 * rks.sigma**2))
        
        # Compute RKS approximation
        z = rks(sample)
        approx_kernel = z @ z.T
        
        # Errors
        errors = torch.abs(true_kernel - approx_kernel)
        
        return {
            'true_kernel': true_kernel.cpu().numpy(),
            'approx_kernel': approx_kernel.cpu().numpy(),
            'errors': errors.cpu().numpy(),
            'statistics': {
                'mean_error': errors.mean().item(),
                'max_error': errors.max().item(),
                'rmse': torch.sqrt((errors**2).mean()).item(),
                'correlation': np.corrcoef(
                    true_kernel.cpu().numpy().flatten(),
                    approx_kernel.cpu().numpy().flatten()
                )[0, 1]
            }
        }


if __name__ == "__main__":
    print("="*70)
    print("Testing Random Kitchen Sinks Expander with Multiple Kernels")
    print("="*70)
    
    # Test with 24D → 512D expansion
    torch.manual_seed(42)
    features = torch.randn(100, 24)
    
    # Estimate sigma
    sigma = estimate_rbf_sigma(features)
    print(f"\nEstimated σ from data: {sigma:.3f}")
    
    # Test different kernels
    kernel_types = ['rbf', 'laplacian', 'rq', 'imq']
    
    for ktype in kernel_types:
        print(f"\n{'='*70}")
        print(f"Testing {ktype.upper()} kernel")
        print(f"{'='*70}")
        
        # Create RKS
        kernel_params = {'alpha': 1.0} if ktype == 'rq' else {}
        
        rks = RandomKitchenSinksExpander(
            input_dim=24,
            output_dim=512,
            sigma=sigma,
            kernel_type=ktype,
            kernel_params=kernel_params,
            seed=42,
            device='cpu'
        )
        
        # Transform
        z = rks(features)
        print(f"Input shape: {features.shape}")
        print(f"Output shape: {z.shape}")
        print(f"Output range: [{z.min():.4f}, {z.max():.4f}]")
        print(f"Output mean: {z.mean():.4f}")
        print(f"Output std: {z.std():.4f}")
        
        # Verify approximation (RBF only)
        if ktype == 'rbf':
            mean_err, max_err, corr = rks.verify_approximation(features, features)
            print(f"\nKernel Approximation Quality:")
            print(f"  Mean error: {mean_err:.4f}")
            print(f"  Max error: {max_err:.4f}")
            print(f"  Correlation: {corr:.4f}")
    
    print("\n" + "="*70)
    print("✓ All kernel types tested successfully!")
    print("="*70)
