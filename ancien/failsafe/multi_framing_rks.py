"""
Multi-Framing Random Kitchen Sinks

Processes each of 8 framings independently through separate RKS expansions.
Each framing (3D) gets its own kernel, maintaining framing structure through
random transformations.

Key Design:
-----------
- Input: [N, 24] = [N, 8 framings × 3D]
- Output: [N, 512] = [N, 8 framings × 64D]
- Each framing stays in separate subspace
- Different kernel per framing possible
- Prevents cross-framing contamination

Architecture:
------------
Framing 0 (dims 0-2)   → RKS_0 (kernel_0) → dims 0-63
Framing 1 (dims 3-5)   → RKS_1 (kernel_1) → dims 64-127
Framing 2 (dims 6-8)   → RKS_2 (kernel_2) → dims 128-191
...
Framing 7 (dims 21-23) → RKS_7 (kernel_7) → dims 448-511

This maintains framing independence while allowing observer-dependent
kernel sampling.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from .kernel_library import create_kernel
from .rks_expansion import estimate_rbf_sigma


class MultiFramingRKS(nn.Module):
    """
    Separate RKS expansion for each framing dimension.
    
    Parameters
    ----------
    n_framings : int
        Number of framings (default: 8)
    dims_per_framing : int
        Dimensions per framing (default: 3 for contrastive)
    output_per_framing : int
        RKS output dimensions per framing (default: 64)
    kernel_types : list of str or str
        Kernel type(s): 'rbf', 'laplacian', 'rq', 'imq', 'matern'
        If list: different kernel for each framing
        If str: same kernel for all framings
    sigma : float, optional
        Kernel bandwidth (if None, will be estimated)
    kernel_params : dict or list of dict, optional
        Kernel-specific parameters
    seed : int
        Random seed for reproducible observer
    device : str
        'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        n_framings: int = 8,
        dims_per_framing: int = 3,
        output_per_framing: int = 64,
        kernel_types = 'rbf',
        sigma: Optional[float] = None,
        kernel_params: Optional[Dict] = None,
        seed: int = 42,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.n_framings = n_framings
        self.dims_per_framing = dims_per_framing
        self.output_per_framing = output_per_framing
        self.sigma = sigma
        self.seed = seed
        
        # Handle device
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
        
        # Prepare kernel types (one per framing)
        if isinstance(kernel_types, str):
            self.kernel_types = [kernel_types] * n_framings
        elif isinstance(kernel_types, list):
            if len(kernel_types) != n_framings:
                raise ValueError(
                    f"kernel_types list must have {n_framings} elements, "
                    f"got {len(kernel_types)}"
                )
            self.kernel_types = kernel_types
        else:
            raise ValueError("kernel_types must be str or list of str")
        
        # Prepare kernel params (one per framing)
        if kernel_params is None:
            self.kernel_params_list = [{}] * n_framings
        elif isinstance(kernel_params, dict):
            # Same params for all kernels
            self.kernel_params_list = [kernel_params.copy()] * n_framings
        elif isinstance(kernel_params, list):
            if len(kernel_params) != n_framings:
                raise ValueError(
                    f"kernel_params list must have {n_framings} elements"
                )
            self.kernel_params_list = kernel_params
        else:
            raise ValueError("kernel_params must be dict or list of dict")
        
        # Storage for kernel objects (will be initialized when sigma is set)
        self._kernels = [None] * n_framings
        
        print(f"\nMultiFramingRKS initialized:")
        print(f"  - Framings: {n_framings}")
        print(f"  - Input per framing: {dims_per_framing}D")
        print(f"  - Output per framing: {output_per_framing}D")
        print(f"  - Total input: {n_framings * dims_per_framing}D")
        print(f"  - Total output: {n_framings * output_per_framing}D")
        print(f"  - Kernel types: {self.kernel_types}")
        print(f"  - Sigma: {sigma if sigma else 'will be estimated'}")
        print(f"  - Device: {self.device}")
        
        # Initialize kernels if sigma is provided
        if sigma is not None:
            self._initialize_kernels(sigma)
    
    
    def _maybe_infer_dims_per_framing(self, features: torch.Tensor) -> None:
        """Infer dims_per_framing from incoming feature dimension when it differs.

        Supports legacy 24D (8×3) and high-D block features (e.g. 8×1024).
        """
        d = int(features.shape[-1])
        if d % self.n_framings != 0:
            raise ValueError(
                f"Input dim {d} is not divisible by n_framings={self.n_framings}; "
                "cannot infer dims_per_framing."
            )
        inferred = d // self.n_framings
        if inferred != self.dims_per_framing:
            old = self.dims_per_framing
            self.dims_per_framing = inferred
            # Rebuild kernels if they were initialized
            if getattr(self, "_kernels", None) is not None and any(k is not None for k in self._kernels):
                sigma = self.sigma
                self._kernels = [None] * self.n_framings
                if sigma is not None:
                    self._initialize_kernels(sigma)
            print(f"[MultiFramingRKS] dims_per_framing inferred as {inferred} (was {old})")
def _initialize_kernels(self, sigma: float):
        """Initialize all kernel objects."""
        self.sigma = sigma
        
        for i in range(self.n_framings):
            kernel = create_kernel(
                kernel_type=self.kernel_types[i],
                input_dim=self.dims_per_framing,
                output_dim=self.output_per_framing,
                sigma=sigma,
                seed=self.seed + i,  # Slightly different seed per framing
                **self.kernel_params_list[i]
            )
            
            # Move kernel tensors to device
            kernel.W = kernel.W.to(self.device)
            kernel.b = kernel.b.to(self.device)
            
            self._kernels[i] = kernel
        
        print(f"  ✓ All {self.n_framings} kernels initialized with σ={sigma:.4f}")
    
    def estimate_and_set_sigma(
        self,
        features: torch.Tensor,
        sample_size: int = 1000,
        percentile: float = 50.0
    ) -> float:
        """
        Estimate sigma from data using median heuristic.
        
        Parameters
        ----------
        features : torch.Tensor
            Feature matrix [N, n_framings * dims_per_framing]
        sample_size : int
            Number of samples for estimation
        percentile : float
            Percentile to use (50.0 = median)
        
        Returns
        -------
        sigma : float
            Estimated bandwidth
        """
        # Estimate from all features together
        sigma = estimate_rbf_sigma(features, sample_size, percentile)
        self._initialize_kernels(sigma)
        return sigma
    
    def forward(self, features: torch.Tensor):
        """
        Apply multi-framing RKS transformation.
        
        ALWAYS returns both individual kernel outputs AND combined output!
        
        Parameters
        ----------
        features : torch.Tensor
            Shape: [N, n_framings * dims_per_framing]
            e.g., [N, 24] for 8 framings × 3D
        
        Returns
        -------
        dict
            'combined': torch.Tensor [N, 512]
                Concatenated output from all 8 kernels
            
            'individual': list of 8 dicts, each containing:
                'features': torch.Tensor [N, 64]
                    This kernel's output in isolation
                'framing_idx': int (0-7)
                    Which framing this kernel processed
                'kernel_type': str
                    Kernel name ('rbf', 'laplacian', 'rq', etc.)
                'input_slice': tuple (start, end)
                    Where this kernel's input came from in the 24D vector
                'output_slice': tuple (start, end)
                    Where this kernel's output goes in the 512D vector
            
            'kernel_types': list of str
                All 8 kernel type names
            
            'n_framings': int
                Number of framings (8)
            
        Notes
        -----
        Each framing occupies a separate subspace in combined output:
        - Framing i: combined[:, i*64:(i+1)*64]
        
        This lets you analyze:
        - How each kernel sees the data differently
        - Which kernels capture what structure
        - Observer variance per kernel type
        """
        # Validate / infer input
        self._maybe_infer_dims_per_framing(features)
        expected_dim = self.n_framings * self.dims_per_framing
        if features.shape[-1] != expected_dim:
            raise ValueError(f"Expected input dim {expected_dim}, got {features.shape[-1]}")
        
        # Check sigma is set
        if self.sigma is None:
            raise ValueError(
                "Sigma not set! Call estimate_and_set_sigma() first"
            )
        
        # Check kernels initialized
        if any(k is None for k in self._kernels):
            raise ValueError(
                "Kernels not initialized! This is a bug."
            )
        
        # Process each framing independently
        outputs = []
        individual_outputs = []
        
        for i in range(self.n_framings):
            # Extract framing features
            start_idx = i * self.dims_per_framing
            end_idx = (i + 1) * self.dims_per_framing
            framing_features = features[:, start_idx:end_idx]  # [N, 3]
            
            # Expand using framing-specific kernel
            expanded = self._kernels[i].transform(framing_features)  # [N, 64]
            
            outputs.append(expanded)
            
            # ALWAYS store individual kernel output
            individual_outputs.append({
                'features': expanded,
                'framing_idx': i,
                'kernel_type': self.kernel_types[i],
                'input_slice': (start_idx, end_idx),
                'output_slice': (i * self.output_per_framing, (i + 1) * self.output_per_framing)
            })
        
        # Concatenate: [N, 512] = 8 × 64
        combined = torch.cat(outputs, dim=1)
        
        # ALWAYS return both individual and combined
        return {
            'combined': combined,
            'individual': individual_outputs,
            'kernel_types': self.kernel_types,
            'n_framings': self.n_framings
        }
    
    def get_framing_subspace(
        self,
        features: torch.Tensor,
        framing_idx: int
    ) -> torch.Tensor:
        """
        Extract features for a specific framing.
        
        Parameters
        ----------
        features : torch.Tensor
            Output from forward(), shape [N, n_framings * output_per_framing]
        framing_idx : int
            Framing index (0-7)
        
        Returns
        -------
        torch.Tensor
            Framing-specific features, shape [N, output_per_framing]
        """
        start_idx = framing_idx * self.output_per_framing
        end_idx = (framing_idx + 1) * self.output_per_framing
        return features[:, start_idx:end_idx]


if __name__ == "__main__":
    print("="*70)
    print("Testing MultiFramingRKS")
    print("="*70)
    
    # Test data: 100 articles, 8 framings × 3D = 24D
    torch.manual_seed(42)
    features = torch.randn(100, 24)
    
    print("\n--- Test 1: All framings same kernel (RBF) ---")
    rks1 = MultiFramingRKS(
        n_framings=8,
        dims_per_framing=3,
        output_per_framing=64,
        kernel_types='rbf',
        seed=42,
        device='cpu'
    )
    
    # Estimate sigma
    sigma = rks1.estimate_and_set_sigma(features)
    print(f"Estimated σ: {sigma:.4f}")
    
    # Transform
    output1 = rks1(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Output range: [{output1.min():.4f}, {output1.max():.4f}]")
    
    # Check subspace separation
    print("\nFraming subspaces:")
    for i in range(8):
        subspace = rks1.get_framing_subspace(output1, i)
        print(f"  Framing {i}: {subspace.shape}, mean={subspace.mean():.4f}")
    
    print("\n--- Test 2: Different kernel per framing ---")
    kernel_types = ['rbf', 'laplacian', 'rq', 'imq', 'matern', 'rbf', 'laplacian', 'rq']
    
    rks2 = MultiFramingRKS(
        n_framings=8,
        dims_per_framing=3,
        output_per_framing=64,
        kernel_types=kernel_types,
        sigma=sigma,  # Use same sigma
        kernel_params={'alpha': 1.0},  # For RQ kernels
        seed=42,
        device='cpu'
    )
    
    output2 = rks2(features)
    print(f"Output shape: {output2.shape}")
    
    # Compare outputs
    diff = (output1 - output2).abs().mean()
    print(f"\nDifference between same vs mixed kernels: {diff:.4f}")
    
    print("\n" + "="*70)
    print("✓ MultiFramingRKS tests complete!")
    print("="*70)