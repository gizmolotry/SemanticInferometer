"""
RKSFeatureMap - Wrapper for MultiFramingRKS
============================================

Makes MultiFramingRKS compatible with complete_pipeline.py
"""

import torch
import torch.nn as nn
from .multi_framing_rks import MultiFramingRKS


class RKSFeatureMap(nn.Module):
    """Wrapper that makes MultiFramingRKS compatible with complete_pipeline.py"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        random_seed: int = 42,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Convert parameters for MultiFramingRKS
        n_framings = 8
        # Infer per-framing dimensionality when possible (e.g., 24D -> 3, 8192D -> 1024).
        # This keeps the multi-framing structure consistent across feature regimes.
        if input_dim % n_framings == 0:
            dims_per_framing = input_dim // n_framings
        else:
            # Legacy fallback: treat as 8×3 framing layout
            dims_per_framing = 3
        if output_dim % n_framings != 0:
            raise ValueError(f"output_dim must be divisible by {n_framings}; got {output_dim}")
        output_per_framing = output_dim // n_framings
        
        # gamma to sigma conversion
        sigma = 1.0 / (2.0 * gamma) ** 0.5 if gamma > 0 else 1.0
        
        print(f"\nRKSFeatureMap: {input_dim}D → {output_dim}D")
        print(f"  Kernel: {kernel_type}, Sigma: {sigma:.4f}, Seed: {random_seed}")
        
        # Create MultiFramingRKS
        self.multi_rks = MultiFramingRKS(
            n_framings=n_framings,
            dims_per_framing=dims_per_framing,
            output_per_framing=output_per_framing,
            kernel_types=kernel_type,
            sigma=sigma,
            seed=random_seed,
            device=device
        )
        
        self.kernel_type = kernel_type
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform features - returns ONLY combined for backward compatibility.
        Individual kernels still computed internally.
        """
        result = self.multi_rks(features)
        return result['combined']
    
    def forward(self, features: torch.Tensor):
        return self.transform(features)
    
    def get_full_output(self, features: torch.Tensor):
        """Get BOTH combined and individual outputs."""
        return self.multi_rks(features)
