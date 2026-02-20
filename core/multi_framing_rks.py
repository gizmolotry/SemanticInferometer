"""
Multi-Framing Random Kitchen Sinks

Processes each of 8 framings independently through separate RKS expansions.
Each framing (3D) gets its own kernel, maintaining framing structure through
random transformations.

Key Design:
-----------
- Input: [N, 24] = [N, 8 framings Ã— 3D]
- Output: [N, 512] = [N, 8 framings Ã— 64D]
- Each framing stays in separate subspace
- Different kernel per framing possible
- Prevents cross-framing contamination

Architecture:
------------
Framing 0 (dims 0-2)   â†’ RKS_0 (kernel_0) â†’ dims 0-63
Framing 1 (dims 3-5)   â†’ RKS_1 (kernel_1) â†’ dims 64-127
Framing 2 (dims 6-8)   â†’ RKS_2 (kernel_2) â†’ dims 128-191
...
Framing 7 (dims 21-23) â†’ RKS_7 (kernel_7) â†’ dims 448-511

This maintains framing independence while allowing observer-dependent
kernel sampling.
"""

import torch
import torch.nn as nn
from typing import List, Any, Optional, Dict
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
        device: str = 'cuda',
        # NEW: KernelContext integration
        kernel_ctx: Optional[Any] = None,
    ):
        super().__init__()
        
        self.n_framings = n_framings
        self.dims_per_framing = dims_per_framing
        self.output_per_framing = output_per_framing
        self.sigma = sigma
        self.seed = seed
        
        # NEW: Store kernel context for provenance
        self.kernel_ctx = kernel_ctx
        
        # NEW: If KernelContext provided, use its settings
        if kernel_ctx is not None:
            try:
                kernel_types = kernel_ctx.kernel_type
                if kernel_ctx.bandwidth is not None:
                    sigma = kernel_ctx.bandwidth
                    self.sigma = sigma
                seed = kernel_ctx.seed
                self.seed = seed
                if kernel_ctx.rks_dim is not None:
                    # Distribute RKS dim across framings
                    output_per_framing = kernel_ctx.rks_dim // n_framings
                    self.output_per_framing = output_per_framing
                print(f"[MultiFramingRKS] Using KernelContext: {kernel_ctx.canonical_id()}")
            except AttributeError:
                pass  # kernel_ctx doesn't have expected attributes
        
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
    
    def _initialize_kernels(self, sigma: float):
        """Initialize all kernel objects with the same sigma (legacy mode)."""
        self.sigma = sigma
        self._sigmas_per_framing = [sigma] * self.n_framings

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

        print(f"  [OK] All {self.n_framings} kernels initialized with sigma={sigma:.4f}")

    def _initialize_kernels_per_framing(self, sigmas: List[float]):
        """Initialize each kernel with its own sigma (per-framing mode)."""
        self._sigmas_per_framing = sigmas
        self.sigma = sum(sigmas) / len(sigmas)  # Store mean for backward compat

        for i in range(self.n_framings):
            kernel = create_kernel(
                kernel_type=self.kernel_types[i],
                input_dim=self.dims_per_framing,
                output_dim=self.output_per_framing,
                sigma=sigmas[i],
                seed=self.seed + i,
                **self.kernel_params_list[i]
            )

            kernel.W = kernel.W.to(self.device)
            kernel.b = kernel.b.to(self.device)

            self._kernels[i] = kernel

        sigma_str = ", ".join([f"{s:.3f}" for s in sigmas])
        print(f"  [OK] All {self.n_framings} kernels initialized with per-framing sigma=[{sigma_str}]")
    
    def estimate_and_set_sigma(
        self,
        features: torch.Tensor,
        sample_size: int = 1000,
        percentile: float = 50.0,
        per_framing: bool = True
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
        per_framing : bool
            If True, estimate sigma separately for each framing (recommended).
            If False, use global sigma (legacy behavior, causes kernel collapse).

        Returns
        -------
        sigma : float
            Mean estimated bandwidth (for backward compat)
        """
        if per_framing:
            # FIXED: Estimate sigma per-framing to avoid kernel collapse
            # Each framing (e.g., 1536D) has different distance statistics
            # Using global sigma causes severe mismatch and cosine sim → 1.0
            import torch.nn.functional as F

            sigmas = []
            for i in range(self.n_framings):
                start = i * self.dims_per_framing
                end = (i + 1) * self.dims_per_framing
                framing = features[:, start:end]

                # L2 normalize the framing before sigma estimation
                # (kernel expects unit vectors after main pipeline normalizes)
                framing_norm = F.normalize(framing, dim=1)

                sigma_i = estimate_rbf_sigma(framing_norm, sample_size, percentile)
                sigmas.append(sigma_i)

            self._initialize_kernels_per_framing(sigmas)
            return sum(sigmas) / len(sigmas)
        else:
            # Legacy: global sigma (not recommended)
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
            e.g., [N, 24] for 8 framings Ã— 3D
        
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
        # Validate input
        expected_dim = self.n_framings * self.dims_per_framing
        if features.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected input dim {expected_dim}, got {features.shape[-1]}"
            )
        
        # Ensure features live on the same device as kernels
        if torch.is_tensor(features) and features.device != self.device:
            features = features.to(self.device)

        # Check sigma is set
        if self.sigma is None:
            raise ValueError(
                "Sigma not set! Call estimate_and_set_sigma() first"
            )
        # Check kernels initialized
        if any(k is None for k in self._kernels):
            # Defensive lazy-init: sigma can be set externally (or after rebuild)
            # without calling _initialize_kernels().
            try:
                self._initialize_kernels(float(self.sigma))
            except Exception as e:
                raise ValueError(
                    f"Kernels not initialized and lazy init failed: {e}"
                )
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
            framing_features = features[:, start_idx:end_idx]  # [N, dims_per_framing]

            # CRITICAL FIX: L2 normalize each framing before RKS transform
            # When the full 12288D vector is globally normalized, each 1536D slice
            # does NOT have unit norm. RKS sigma was estimated on normalized framings,
            # so we must normalize here to match.
            framing_features = torch.nn.functional.normalize(framing_features, dim=1)

            # Expand using framing-specific kernel
            expanded = self._kernels[i].transform(framing_features)  # [N, output_per_framing]
            
            outputs.append(expanded)
            
            # ALWAYS store individual kernel output
            individual_outputs.append({
                'features': expanded,
                'framing_idx': i,
                'kernel_type': self.kernel_types[i],
                'input_slice': (start_idx, end_idx),
                'output_slice': (i * self.output_per_framing, (i + 1) * self.output_per_framing)
            })
        
        # Concatenate: [N, 512] = 8 Ã— 64
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


# =============================================================================
# MultiViewRKS: Shared-Basis Projection for CLS Views (Mode B backbone)
# =============================================================================

class MultiViewRKS(nn.Module):
    """
    Generic multiview projector using a SHARED basis for all views.
    
    This is the "born aligned" projection: all 8 V-observers (bots/framings)
    are projected through the same M-observer (RKS basis), ensuring they
    live in the same Hilbert space.
    
    CRITICAL: This class does NOT create basis tensors. It must receive
    an injected SharedBasis from kernel_library.ensure_shared_basis().
    
    Input: [N, B, H] where B=8 views, H=768 hidden
    Output: [N, B, D] where D is the RKS output dimension
    
    Usage:
        from kernel_library import ensure_shared_basis
        from kernel_context import create_rbf_context
        
        ctx = create_rbf_context(input_dim=768, rks_dim=2048, bandwidth=1.0)
        basis, basis_hash = ensure_shared_basis(ctx, device='cuda')
        
        projector = MultiViewRKS(shared_basis=basis)
        rkhs_views = projector(cls_per_bot)  # [N, 8, 2048]
        
    Alternative (with KernelContext):
        projector = MultiViewRKS.from_context(ctx, device='cuda')
        rkhs_views = projector(cls_per_bot)
    """
    
    def __init__(self, shared_basis: 'SharedBasis', kernel_ctx: Optional[Any] = None):
        """
        Args:
            shared_basis: SharedBasis instance from kernel_library.ensure_shared_basis()
            kernel_ctx: Optional KernelContext for provenance tracking
        """
        super().__init__()
        
        if shared_basis is None:
            raise ValueError(
                "MultiViewRKS requires a SharedBasis. "
                "Use kernel_library.ensure_shared_basis() to create one."
            )
        
        self.shared_basis = shared_basis
        self.input_dim = shared_basis.input_dim
        self.output_dim = shared_basis.output_dim
        self.kernel_ctx = kernel_ctx
    
    @classmethod
    def from_context(cls, kernel_ctx: 'KernelContext', device: str = 'cpu') -> 'MultiViewRKS':
        """
        Create MultiViewRKS from a KernelContext.
        
        This is the preferred way to create a MultiViewRKS for Mode B,
        as it ensures the basis is created through the canonical factory.
        
        Args:
            kernel_ctx: KernelContext defining the geometry regime
            device: Torch device
            
        Returns:
            MultiViewRKS instance with shared basis
        """
        from .kernel_library import ensure_shared_basis
        
        basis, basis_hash = ensure_shared_basis(kernel_ctx, device=device)
        if basis is None:
            raise ValueError(
                f"KernelContext with type '{kernel_ctx.kernel_type}' doesn't produce a basis. "
                "Use 'rbf' for RKHS projection."
            )
        
        return cls(shared_basis=basis, kernel_ctx=kernel_ctx)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project all views through the shared basis.
        
        Args:
            X: Input tensor [N, B, H] where B=views, H=hidden_dim
            
        Returns:
            Projected tensor [N, B, D] where D=rks_dim
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D tensor [N, B, H], got shape {X.shape}")
        
        N, B, H = X.shape
        
        if H != self.input_dim:
            raise ValueError(
                f"Input hidden dim {H} doesn't match basis input_dim {self.input_dim}"
            )
        
        # Flatten views for batch projection
        X_flat = X.view(N * B, H)
        
        # Apply shared basis (same Ω, b for all views)
        phi_flat = self.shared_basis(X_flat)  # [N*B, D]
        
        # Reshape back to [N, B, D]
        return phi_flat.view(N, B, self.output_dim)
    
    @property
    def basis_hash(self) -> str:
        """Get the M-observer fingerprint."""
        return self.shared_basis.basis_hash


def project_views(
    cls_per_bot: torch.Tensor,
    shared_basis: 'SharedBasis',
) -> torch.Tensor:
    """
    Helper function to project CLS views through a shared basis.
    
    This is the canonical "born-aligned multiview projection" used by
    Dirichlet Mode B and sheaf diagnostics.
    
    Args:
        cls_per_bot: [N, B, H] tensor of CLS embeddings per view
        shared_basis: SharedBasis from kernel_library.ensure_shared_basis()
        
    Returns:
        [N, B, D] tensor of RKHS-projected views
    """
    projector = MultiViewRKS(shared_basis)
    return projector(cls_per_bot)


if __name__ == "__main__":
    print("="*70)
    print("Testing MultiFramingRKS")
    print("="*70)
    
    # Test data: 100 articles, 8 framings Ã— 3D = 24D
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
    print(f"Estimated Ïƒ: {sigma:.4f}")
    
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
    print("âœ“ MultiFramingRKS tests complete!")
    print("="*70) 