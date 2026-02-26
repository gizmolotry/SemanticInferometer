"""
Multiple kernel implementations for observer comparison

Different kernels = different observers with genuinely different similarity metrics
"""

import torch
import numpy as np
import math


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
    """
    
    def _sample_frequencies(self):
        # Sample from Student-t distribution (heavy tails)
        df = 3  # Moderate heavy tails
        
        W = np.random.standard_t(df, size=(self.output_dim, self.input_dim))
        W = torch.FloatTensor(W) / self.sigma
        b = torch.rand(self.output_dim) * 2 * math.pi
        
        return W, b


class MaternKernel(KernelBase):
    """
    Matérn Kernel: Parameterized by smoothness ν
    
    ν = 1/2: Laplacian (non-differentiable)
    ν = 3/2: Once differentiable  
    ν = 5/2: Twice differentiable
    ν → ∞: RBF (infinitely differentiable)
    """
    
    def __init__(self, input_dim, output_dim, sigma, nu=1.5, seed=42):
        self.nu = nu
        super().__init__(input_dim, output_dim, sigma, seed)
    
    def _sample_frequencies(self):
        if self.nu == 0.5:
            # Laplacian case
            W = np.random.laplace(0, 1.0/self.sigma,
                                  size=(self.output_dim, self.input_dim))
        else:
            # General case: mixture approximation
            W = torch.randn(self.output_dim, self.input_dim) / self.sigma
            # Scale by nu-dependent factor
            W = W * (2 * self.nu / self.sigma)
        
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
          - nu (for Matérn)
    """
    
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
