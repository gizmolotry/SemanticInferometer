"""
PCA Topic Removal (Option 1)

Implements the "vertical" approach to removing emergent topic components
from feature representations without defining what the topic is.

Theoretical Foundation:
----------------------
Option 1 from the theoretical framework: Remove the first principal component
of the corpus features. This is the classic "common component removal" trick.

The idea: In a topic-filtered corpus (e.g., all articles about Gaza/Israel),
the biggest variance direction often corresponds to "topic/background" rather
than framing differences. By removing it, we isolate the orthogonal bias axes.

Algorithm:
---------
1. Center features: X_c = X - Î¼
2. Compute PCA â†’ extract top component uâ‚
3. Remove it: X' = X_c - (X_c uâ‚)uâ‚áµ€

Why it's vertical: We don't declare what "topic" is; we remove whatever
is empirically shared across all samples.

Risk: Sometimes the top PC is not purely topic; it can include framing
if the corpus is imbalanced. Use with caution and validate on controls.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# =============================================================================
# Constitutional Contract Enforcement
# =============================================================================

def _check_pca_contract(rep_kind, operation="pca_removal"):
    """
    Check if PCA operation is allowed for given rep_kind.
    
    Returns True if allowed, raises ValueError if forbidden.
    """
    if rep_kind is not None and str(rep_kind).lower() == 'logits_raw':
        raise ValueError(
            f"[CONSTITUTIONAL CONTRACT] {operation} is FORBIDDEN for rep_kind='logits_raw'. "
            "Logits are verdict coordinates and must not have PCs removed."
        )
    return True



def remove_first_pc(
    features: torch.Tensor,
    return_removed_component: bool = False,
    rep_kind: Optional[str] = None,  # NEW: contract enforcement
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Remove the first principal component from features (Option 1).
    
    This eliminates the dominant variance direction, which in topic-filtered
    corpora often corresponds to topic similarity rather than framing.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature matrix [N, D]
    return_removed_component : bool
        If True, also return the removed component and variance explained
    rep_kind : str, optional
        If 'logits_raw', raises ValueError (constitutional contract)
    
    Returns
    -------
    cleaned_features : torch.Tensor
        Features with first PC removed [N, D]
    u1 : torch.Tensor (optional)
        The removed principal component [D]
    variance_explained : float (optional)
        Fraction of variance explained by removed component
    
    Examples
    --------
    >>> features = torch.randn(1000, 24)
    >>> cleaned = remove_first_pc(features)
    >>> print(cleaned.shape)
    torch.Size([1000, 24])
    
    >>> cleaned, u1, var_exp = remove_first_pc(features, return_removed_component=True)
    >>> print(f"Removed {var_exp:.2%} of variance")
    """
    # CONSTITUTIONAL CONTRACT CHECK
    _check_pca_contract(rep_kind, "remove_first_pc")
    
    if features.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {features.shape}")
    
    N, D = features.shape
    device = features.device  # Preserve device (CUDA/CPU)
    
    # 1. Center features
    centered = features - features.mean(dim=0, keepdim=True)
    
    # 2. Compute PCA (extract top component only)
    # Using torch.pca_lowrank for efficiency
    U, S, V = torch.pca_lowrank(centered, q=1, center=False)  # Already centered
    
    # V[:, 0] is the first principal component (eigenvector)
    u1 = V[:, 0].to(device)  # [D] - Ensure same device
    
    # 3. Project onto u1 and remove
    projection = (centered @ u1.unsqueeze(1)) @ u1.unsqueeze(0)
    cleaned = centered - projection
    cleaned = cleaned.to(device)  # Ensure same device
    
    if return_removed_component:
        # Compute variance explained
        total_var = torch.var(centered, dim=0).sum()
        removed_var = S[0]**2 / (N - 1)  # Eigenvalue = variance along PC
        var_explained = (removed_var / total_var).item()
        
        return cleaned, u1, var_explained
    else:
        return cleaned


def remove_top_k_pcs(
    features: torch.Tensor,
    k: int = 1,
    return_components: bool = False
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove top k principal components (generalization of Option 1).
    
    Useful if topic structure spans multiple dimensions.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature matrix [N, D]
    k : int
        Number of top components to remove
    return_components : bool
        If True, return removed components and singular values
    
    Returns
    -------
    cleaned_features : torch.Tensor
        Features with top k PCs removed [N, D]
    components : torch.Tensor (optional)
        Removed principal components [D, k]
    singular_values : torch.Tensor (optional)
        Singular values of removed components [k]
    """
    if features.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {features.shape}")
    
    N, D = features.shape
    device = features.device  # Preserve device (CUDA/CPU)
    
    if k > min(N, D):
        raise ValueError(f"k={k} too large for features shape {features.shape}")
    
    # Center
    centered = features - features.mean(dim=0, keepdim=True)
    
    # Compute top k components
    U, S, V = torch.pca_lowrank(centered, q=k, center=False)
    
    # V[:, :k] are the top k principal components
    components = V[:, :k].to(device)  # [D, k] - Ensure same device
    
    # Project and remove
    projection = centered @ components @ components.T
    cleaned = centered - projection
    cleaned = cleaned.to(device)  # Ensure same device
    
    if return_components:
        return cleaned, components, S
    else:
        return cleaned


def visualize_pc_removal_effect(
    features: torch.Tensor,
    labels: Optional[np.ndarray] = None
) -> dict:
    """
    Analyze the effect of PC removal for diagnostics.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature matrix [N, D]
    labels : np.ndarray, optional
        Class labels for computing separability metrics
    
    Returns
    -------
    dict
        Diagnostic information:
        - variance_explained: Variance explained by each PC
        - separability_before: Metric before removal (if labels provided)
        - separability_after: Metric after removal (if labels provided)
    """
    N, D = features.shape
    
    # Compute full PCA for analysis
    centered = features - features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(centered, q=min(10, D))
    
    # Variance explained by each component
    total_var = torch.var(centered, dim=0).sum().item()
    variance_explained = [(s**2 / (N-1) / total_var).item() for s in S]
    
    diagnostics = {
        'variance_explained': variance_explained,
        'cumulative_variance': np.cumsum(variance_explained).tolist(),
        'first_pc_dominance': variance_explained[0] if variance_explained else 0.0
    }
    
    # If labels provided, compute separability
    if labels is not None:
        # Simple metric: ratio of between-class to within-class variance
        def compute_separability(feats):
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Between-class variance
            overall_mean = feats.mean(dim=0)
            between_var = 0.0
            for label in unique_labels:
                mask = labels == label
                class_mean = feats[mask].mean(dim=0)
                n_class = mask.sum()
                between_var += n_class * ((class_mean - overall_mean)**2).sum()
            
            # Within-class variance
            within_var = 0.0
            for label in unique_labels:
                mask = labels == label
                class_feats = feats[mask]
                class_mean = class_feats.mean(dim=0)
                within_var += ((class_feats - class_mean)**2).sum()
            
            return (between_var / within_var).item() if within_var > 0 else 0.0
        
        diagnostics['separability_before'] = compute_separability(features)
        
        # After PC removal
        cleaned = remove_first_pc(features)
        diagnostics['separability_after'] = compute_separability(cleaned)
        diagnostics['separability_ratio'] = (
            diagnostics['separability_after'] / diagnostics['separability_before']
            if diagnostics['separability_before'] > 0 else 0.0
        )
    
    return diagnostics


def save_pca_components(
    features: torch.Tensor,
    output_path: str | Path,
    k: int = 5
):
    """
    Save PCA components for later analysis or visualization.
    
    Parameters
    ----------
    features : torch.Tensor
        Feature matrix [N, D]
    output_path : str or Path
        Where to save components
    k : int
        Number of components to save
    """
    centered = features - features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(centered, q=k, center=False)
    
    N = features.shape[0]
    variance_explained = [(s**2 / (N-1)).item() for s in S]
    
    torch.save({
        'components': V[:, :k],  # [D, k]
        'singular_values': S,  # [k]
        'variance_explained': variance_explained,
        'mean': features.mean(dim=0),  # [D]
        'feature_dim': features.shape[1],
        'num_samples': N
    }, output_path)
    
    print(f"Saved {k} PCA components to {output_path}")
    print(f"Total variance explained: {sum(variance_explained):.2%}")


def load_and_apply_pca_removal(
    features: torch.Tensor,
    pca_path: str | Path,
    k: int = 1
) -> torch.Tensor:
    """
    Load pre-computed PCA components and apply removal.
    
    Useful for applying the same transformation to new data.
    
    Parameters
    ----------
    features : torch.Tensor
        New features to transform [N, D]
    pca_path : str or Path
        Path to saved PCA components
    k : int
        Number of components to remove
    
    Returns
    -------
    torch.Tensor
        Features with k components removed
    """
    pca_data = torch.load(pca_path)
    
    if features.shape[1] != pca_data['feature_dim']:
        raise ValueError(
            f"Feature dimension mismatch: {features.shape[1]} vs {pca_data['feature_dim']}"
        )
    
    # Center using saved mean
    centered = features - pca_data['mean']
    
    # Get top k components
    components = pca_data['components'][:, :k]  # [D, k]
    
    # Remove projection
    projection = centered @ components @ components.T
    cleaned = centered - projection
    
    return cleaned


if __name__ == "__main__":
    print("="*70)
    print("Testing PCA Topic Removal (Option 1)")
    print("="*70)
    
    # Create synthetic data with known structure
    # Topic component + orthogonal bias components
    torch.manual_seed(42)
    
    N = 1000  # Articles
    D = 24    # Features
    
    # Simulate: Topic (strong PC1) + Bias (weaker PC2, PC3)
    topic_component = torch.randn(N, 1) @ torch.randn(1, D)  # Rank-1 topic
    topic_component = topic_component * 3  # Make it dominant
    
    bias_component = torch.randn(N, D) * 0.5  # Orthogonal bias structure
    
    features = topic_component + bias_component
    
    print(f"\nSynthetic data: {features.shape}")
    print(f"Feature norms: mean={features.norm(dim=-1).mean():.2f}, std={features.norm(dim=-1).std():.2f}")
    
    # Test 1: Basic removal
    print("\n" + "="*70)
    print("TEST 1: Basic PC Removal")
    print("="*70)
    
    cleaned, u1, var_exp = remove_first_pc(features, return_removed_component=True)
    
    print(f"\nVariance explained by PC1: {var_exp:.2%}")
    print(f"Cleaned feature norms: mean={cleaned.norm(dim=-1).mean():.2f}, std={cleaned.norm(dim=-1).std():.2f}")
    print(f"Expected: Lower variance after removing dominant topic component")
    
    # Test 2: Multiple components
    print("\n" + "="*70)
    print("TEST 2: Remove Top 3 Components")
    print("="*70)
    
    cleaned_3, components, S = remove_top_k_pcs(features, k=3, return_components=True)
    
    print(f"\nRemoved {len(S)} components")
    print(f"Singular values: {S.numpy()}")
    print(f"Cleaned shape: {cleaned_3.shape}")
    
    # Test 3: Diagnostics
    print("\n" + "="*70)
    print("TEST 3: Diagnostic Analysis")
    print("="*70)
    
    # Create fake labels (pro/anti/neutral)
    labels = np.random.choice(['pro', 'anti', 'neutral'], size=N)
    
    diagnostics = visualize_pc_removal_effect(features, labels)
    
    print(f"\nFirst 5 PCs variance explained:")
    for i, var in enumerate(diagnostics['variance_explained'][:5]):
        print(f"  PC{i+1}: {var:.2%}")
    
    print(f"\nFirst PC dominance: {diagnostics['first_pc_dominance']:.2%}")
    print("  (If >30%, topic is likely dominant component)")
    
    if 'separability_ratio' in diagnostics:
        print(f"\nClass separability after removal: {diagnostics['separability_ratio']:.2f}x")
        print("  (>1.0 means removal improved class separation)")
    
    # Test 4: Save/load
    print("\n" + "="*70)
    print("TEST 4: Save and Apply to New Data")
    print("="*70)
    
    save_pca_components(features, "test_pca.pt", k=3)
    
    # New data
    new_features = torch.randn(100, D)
    cleaned_new = load_and_apply_pca_removal(new_features, "test_pca.pt", k=1)
    
    print(f"\nNew features: {new_features.shape}")
    print(f"Cleaned new features: {cleaned_new.shape}")
    
    # Cleanup
    Path("test_pca.pt").unlink()
    
    print("\n" + "="*70)
    print("âœ“ PCA removal module tests complete!")
    print("="*70)

# =============================================================================
# ENDOGENOUS ZCA WHITENING (Holographic Truth Protocol, Axiom 2)
# =============================================================================
#
# Unlike PC removal (which deletes a direction), ZCA whitening ROTATES the
# data so that its covariance becomes the Identity matrix. This maximizes
# the entropy of the input relative to RKS random projections, ensuring
# that high-frequency details (framing cracks) are not masked by the
# dominant variance direction (the "Narrow Cone" problem).
#
# CRITICAL CONSTRAINTS:
# - ENDOGENOUS ONLY: Fit from a sample of the target corpus itself.
#   Do NOT use Wikipedia or external data (that would dilute the geometry).
# - Do NOT fit per-article (that would delete the signal).
# - Fit ONCE from a random sample (~5000 articles), then apply as fixed transform.
#
# The whitening matrix W_zca is computed via SVD:
#   X_white = (X - μ) @ U @ diag(S^{-1/2}) @ U^T
# where U, S, _ = SVD(X_centered)


def fit_whitening_matrix(
    sample: torch.Tensor,
    regularization: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """
    Compute the ZCA whitening matrix from a corpus sample.

    This should be called ONCE on a random ~5000-article sample of the
    target corpus. The resulting matrix is then applied as a fixed transform
    to all articles (fit_whitening_matrix → apply_whitening_fixed).

    Args:
        sample: [N, D] feature matrix (e.g., mean-pooled embeddings)
        regularization: Small epsilon added to singular values for stability

    Returns:
        Dict with:
        - 'mu': [D] corpus mean
        - 'whitening_matrix': [D, D] the ZCA transform W
        - 'dewhitening_matrix': [D, D] inverse transform (for reconstruction)
        - 'singular_values': [D] singular values before regularization
        - 'n_samples': int
        - 'regularization': float
    """
    if sample.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {sample.shape}")

    N, D = sample.shape
    device = sample.device

    # 1. Center
    mu = sample.mean(dim=0)
    centered = sample - mu

    # 2. SVD (more stable than eigendecomposition for whitening)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    # S contains singular values; eigenvalues of covariance = S^2 / (N-1)

    # 3. Compute whitening matrix via ZCA
    # ZCA: W = V @ diag(1/sqrt(eigenvalues + eps)) @ V^T
    # where eigenvalues = S^2 / (N-1)
    eig_vals = S ** 2 / (N - 1)
    inv_sqrt_eig = 1.0 / torch.sqrt(eig_vals + regularization)

    # V from SVD(X_centered): columns of Vt.T = right singular vectors
    V = Vt.T  # [D, min(N,D)]

    # ZCA whitening matrix: W = V @ diag(inv_sqrt_eig) @ V^T
    whitening_matrix = V @ torch.diag(inv_sqrt_eig) @ V.T  # [D, D]

    # Dewhitening (inverse): W_inv = V @ diag(sqrt(eig)) @ V^T
    sqrt_eig = torch.sqrt(eig_vals + regularization)
    dewhitening_matrix = V @ torch.diag(sqrt_eig) @ V.T  # [D, D]

    print(f"[ZCA] Fit whitening from {N} samples, D={D}")
    print(f"[ZCA] Top 5 singular values: {S[:5].tolist()}")
    print(f"[ZCA] Condition number: {S[0] / S[-1]:.2f}")

    return {
        'mu': mu.cpu(),
        'whitening_matrix': whitening_matrix.cpu(),
        'dewhitening_matrix': dewhitening_matrix.cpu(),
        'singular_values': S.cpu(),
        'n_samples': N,
        'regularization': regularization,
    }


def apply_whitening_fixed(
    features: torch.Tensor,
    mu: torch.Tensor,
    whitening_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a pre-computed ZCA whitening transform.

    Args:
        features: [N, D] or [N, B, D] features to whiten
        mu: [D] corpus mean (from fit_whitening_matrix)
        whitening_matrix: [D, D] ZCA matrix (from fit_whitening_matrix)

    Returns:
        Whitened features, same shape as input
    """
    original_shape = features.shape
    D = whitening_matrix.shape[0]

    # Handle 3D input [N, B, D]
    if features.dim() == 3:
        N, B, H = features.shape
        assert H == D, f"Dimension mismatch: features {H} vs whitening {D}"
        features = features.reshape(N * B, D)

    # Move to same device
    mu = mu.to(features.device)
    W = whitening_matrix.to(features.device)

    # Apply: X_white = (X - μ) @ W
    whitened = (features - mu) @ W

    # Restore shape
    if len(original_shape) == 3:
        whitened = whitened.reshape(original_shape)

    return whitened


def save_whitening_matrix(
    whitening_data: Dict[str, Any],
    output_path: str,
):
    """Save pre-computed whitening matrix for reuse."""
    torch.save(whitening_data, output_path)
    print(f"[ZCA] Saved whitening matrix to {output_path}")


def load_whitening_matrix(path: str) -> Dict[str, torch.Tensor]:
    """Load pre-computed whitening matrix."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"[ZCA] Loaded whitening from {path} (N={data['n_samples']}, reg={data['regularization']})")
    return data


# Alias for backward compatibility with complete_pipeline.py
remove_top_pca_component = remove_first_pc