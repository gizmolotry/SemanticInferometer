"""
Observer Agreement Residuals (Option 2)

Implements consensus-based subspace removal to extract observer-dependent
interpretations while factoring out shared semantic structure.

Theoretical Foundation:
----------------------
Option 2 from the theoretical framework: Use multiple observers to define
"ground truth" as the consensus subspace, then extract residuals that
represent observer-specific interpretations.

This mirrors Inter-Rater Reliability (IRR) but uses it as a feature
extractor rather than a validation metric.

Algorithm:
---------
Given embeddings Z_o from different observers (different models, prompts, or seeds):

1. Stack observer embeddings: Z = [Z_1, Z_2, ..., Z_k]  [N, k*D]
2. Compute consensus subspace U via SVD or PCA
3. For each observer: Z'_o = Z_o - Z_o U Uᵀ

The consensus U captures "what happened" (e.g., building destroyed).
The residuals Z'_o capture "how it's framed" (strike vs civilian infrastructure).

Math Advantage:
--------------
By removing U (the consensus), we perform a high-pass filter on semantic space,
isolating discordance/framing from shared factual content.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def torch_load_trusted(path, map_location="cpu"):
    """Load a .pt experiment artifact produced by this repo.

    Why this exists:
      - In PyTorch 2.6+, torch.load() defaults to weights_only=True (safer unpickling).
      - Our observer_*.pt files are NOT just model weights; they include data (often numpy arrays),
        which requires full pickle support and will fail under weights_only=True.
      - These artifacts are generated locally, so we intentionally load with weights_only=False
        when the runtime supports it.

    SECURITY NOTE: Do not use this to load untrusted .pt files.
    """
    import inspect
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        # If signature inspection fails, fall back to the old behavior.
        pass
    return torch.load(path, map_location=map_location)
def compute_consensus_subspace(
    observer_embeddings: List[torch.Tensor],
    k_components: int = 5,
    method: str = 'svd'
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Compute consensus subspace from multiple observer embeddings.
    
    The consensus subspace U represents semantic structure that all observers
    agree on (e.g., "this article is about Gaza", "building was destroyed").
    
    Parameters
    ----------
    observer_embeddings : List[torch.Tensor]
        List of embedding matrices, each [N, D]
        All must have same shape
    k_components : int
        Dimensionality of consensus subspace
    method : str
        'svd' (recommended) or 'mean_pca'
    
    Returns
    -------
    U : torch.Tensor
        Consensus subspace basis [D, k]
    S : torch.Tensor
        Singular values / importance of each component [k]
    variance_explained : List[float]
        Variance explained by each component
    
    Notes
    -----
    Different methods:
    - 'svd': Concatenate all observers and compute SVD (captures shared structure)
    - 'mean_pca': Average embeddings then compute PCA (simpler but less principled)
    """
    if not observer_embeddings:
        raise ValueError("Need at least one observer")
    
    # Validate shapes
    N, D = observer_embeddings[0].shape
    for i, emb in enumerate(observer_embeddings):
        if emb.shape != (N, D):
            raise ValueError(f"Observer {i} shape mismatch: {emb.shape} vs {(N, D)}")
    
    if method == 'svd':
        # Concatenate all observers: [N, k_obs * D]
        concatenated = torch.cat(observer_embeddings, dim=-1)
        
        # Center
        centered = concatenated - concatenated.mean(dim=0, keepdim=True)
        
        # SVD to find shared structure
        # U [N, k], S [k], V [k_obs*D, k]
        _, S, V = torch.pca_lowrank(centered, q=k_components, center=False)
        
        # Project back to original D-dimensional space
        # Take mean projection across observers
        k_obs = len(observer_embeddings)
        V_reshaped = V.view(k_obs, D, k_components)  # [k_obs, D, k]
        U = V_reshaped.mean(dim=0)  # [D, k]
        
        # Re-orthogonalize (mean might break orthogonality)
        U, _ = torch.linalg.qr(U)
        
    elif method == 'mean_pca':
        # Average all observers first
        mean_embedding = torch.stack(observer_embeddings).mean(dim=0)  # [N, D]
        
        # PCA on mean
        centered = mean_embedding - mean_embedding.mean(dim=0, keepdim=True)
        _, S, V = torch.pca_lowrank(centered, q=k_components, center=False)
        
        U = V[:, :k_components]  # [D, k]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute variance explained
    total_var = sum([torch.var(emb, dim=0).sum() for emb in observer_embeddings])
    variance_explained = [(s**2 / (N-1) / total_var).item() for s in S]
    
    return U, S, variance_explained


def extract_observer_residuals(
    observer_embeddings: List[torch.Tensor],
    consensus_subspace: torch.Tensor,
    normalize: bool = False
) -> List[torch.Tensor]:
    """
    Extract observer-specific residuals after removing consensus.
    
    For each observer: Z'_o = Z_o - Z_o U Uᵀ
    
    This removes the shared semantic structure, leaving only the
    observer-dependent framing/interpretation.
    
    Parameters
    ----------
    observer_embeddings : List[torch.Tensor]
        List of embedding matrices, each [N, D]
    consensus_subspace : torch.Tensor
        Consensus basis [D, k]
    normalize : bool
        If True, L2-normalize residuals
    
    Returns
    -------
    List[torch.Tensor]
        Residual embeddings for each observer, each [N, D]
    
    Examples
    --------
    >>> embs = [torch.randn(1000, 512) for _ in range(5)]
    >>> U, S, var_exp = compute_consensus_subspace(embs, k_components=10)
    >>> residuals = extract_observer_residuals(embs, U)
    >>> print(residuals[0].shape)
    torch.Size([1000, 512])
    """
    residuals = []
    
    for emb in observer_embeddings:
        # Project onto consensus subspace
        # projection = Z @ U @ Uᵀ
        projection = emb @ consensus_subspace @ consensus_subspace.T
        
        # Residual = original - projection
        residual = emb - projection
        
        if normalize:
            residual = torch.nn.functional.normalize(residual, p=2, dim=-1)
        
        residuals.append(residual)
    
    return residuals


def analyze_consensus_vs_residuals(
    observer_embeddings: List[torch.Tensor],
    consensus_subspace: torch.Tensor,
    labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze how much variance is consensus vs observer-specific.
    
    Parameters
    ----------
    observer_embeddings : List[torch.Tensor]
        List of embedding matrices
    consensus_subspace : torch.Tensor
        Consensus basis [D, k]
    labels : np.ndarray, optional
        Article labels for computing separability
    
    Returns
    -------
    dict
        Analysis including:
        - consensus_variance_fraction: How much variance is shared
        - residual_variance_fraction: How much is observer-specific
        - inter_observer_agreement: Correlation between observers
        - separability_consensus: Class separability in consensus space
        - separability_residuals: Class separability in residual space
    """
    residuals = extract_observer_residuals(observer_embeddings, consensus_subspace)
    
    # Compute variance fractions
    total_var = sum([torch.var(emb).item() for emb in observer_embeddings])
    residual_var = sum([torch.var(res).item() for res in residuals])
    consensus_var = total_var - residual_var
    
    analysis = {
        'consensus_variance_fraction': consensus_var / total_var,
        'residual_variance_fraction': residual_var / total_var,
        'num_observers': len(observer_embeddings),
        'consensus_dim': consensus_subspace.shape[1]
    }
    
    # Inter-observer agreement (average correlation)
    n_obs = len(observer_embeddings)
    correlations = []
    for i in range(n_obs):
        for j in range(i+1, n_obs):
            # Flatten and correlate
            emb_i = observer_embeddings[i].flatten().numpy()
            emb_j = observer_embeddings[j].flatten().numpy()
            corr = np.corrcoef(emb_i, emb_j)[0, 1]
            correlations.append(corr)
    
    analysis['inter_observer_agreement'] = np.mean(correlations)
    analysis['agreement_std'] = np.std(correlations)
    
    # If labels provided, compute separability
    if labels is not None:
        # Project onto consensus
        mean_emb = torch.stack(observer_embeddings).mean(dim=0)
        consensus_proj = mean_emb @ consensus_subspace @ consensus_subspace.T
        
        # Compute separability (simple between/within class variance ratio)
        def class_separability(feats, labels):
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            overall_mean = feats.mean(dim=0)
            between_var = 0.0
            within_var = 0.0
            
            for label in unique_labels:
                mask = labels == label
                class_feats = feats[mask]
                class_mean = class_feats.mean(dim=0)
                n = mask.sum()
                
                between_var += n * ((class_mean - overall_mean)**2).sum()
                within_var += ((class_feats - class_mean)**2).sum()
            
            return (between_var / within_var).item() if within_var > 0 else 0.0
        
        analysis['separability_consensus'] = class_separability(consensus_proj, labels)
        
        # Average separability across observer residuals
        residual_seps = [class_separability(res, labels) for res in residuals]
        analysis['separability_residuals'] = np.mean(residual_seps)
        analysis['separability_residuals_std'] = np.std(residual_seps)
    
    return analysis


def visualize_consensus_structure(
    observer_embeddings: List[torch.Tensor],
    consensus_subspace: torch.Tensor,
    article_metadata: Optional[List[Dict]] = None,
    output_dir: str | Path = "outputs/consensus_analysis"
) -> Dict:
    """
    Generate visualizations and diagnostics for consensus analysis.
    
    Parameters
    ----------
    observer_embeddings : List[torch.Tensor]
        Observer embeddings
    consensus_subspace : torch.Tensor
        Consensus basis
    article_metadata : List[Dict], optional
        Metadata for each article (titles, sources, etc.)
    output_dir : str or Path
        Where to save outputs
    
    Returns
    -------
    dict
        Paths to generated visualizations and summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    residuals = extract_observer_residuals(observer_embeddings, consensus_subspace)
    
    # Compute projection onto consensus for each observer
    n_obs = len(observer_embeddings)
    consensus_projs = []
    
    for emb in observer_embeddings:
        proj = emb @ consensus_subspace @ consensus_subspace.T
        consensus_projs.append(proj)
    
    # Save consensus subspace
    torch.save({
        'consensus_subspace': consensus_subspace,
        'num_observers': n_obs,
        'consensus_dim': consensus_subspace.shape[1],
        'feature_dim': consensus_subspace.shape[0]
    }, output_dir / 'consensus_subspace.pt')
    
    # Compute variance explained by consensus
    total_norms = [emb.norm(dim=-1) for emb in observer_embeddings]
    residual_norms = [res.norm(dim=-1) for res in residuals]
    
    mean_total_norm = torch.stack(total_norms).mean(dim=0)
    mean_residual_norm = torch.stack(residual_norms).mean(dim=0)
    
    consensus_strength = 1 - (mean_residual_norm / (mean_total_norm + 1e-8))
    
    # Save article-level statistics
    stats = {
        'consensus_strength': consensus_strength.numpy().tolist(),
        'total_norm': mean_total_norm.numpy().tolist(),
        'residual_norm': mean_residual_norm.numpy().tolist()
    }
    
    if article_metadata:
        stats['metadata'] = article_metadata
    
    import json
    with open(output_dir / 'article_consensus_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    summary = {
        'mean_consensus_strength': consensus_strength.mean().item(),
        'std_consensus_strength': consensus_strength.std().item(),
        'num_high_consensus': (consensus_strength > 0.7).sum().item(),
        'num_low_consensus': (consensus_strength < 0.3).sum().item(),
        'output_dir': str(output_dir)
    }
    
    return summary


class ObserverResidualExtractor:
    """
    Stateful extractor for computing observer residuals.
    
    Fit on a set of observers to learn consensus, then apply to new data.
    """
    
    def __init__(self, k_components: int = 10, method: str = 'svd'):
        self.k_components = k_components
        self.method = method
        self.consensus_subspace = None
        self.is_fitted = False
    
    def fit(self, observer_embeddings: List[torch.Tensor]) -> 'ObserverResidualExtractor':
        """Fit consensus subspace from observer embeddings."""
        U, S, var_exp = compute_consensus_subspace(
            observer_embeddings,
            k_components=self.k_components,
            method=self.method
        )
        
        self.consensus_subspace = U
        self.singular_values = S
        self.variance_explained = var_exp
        self.is_fitted = True
        
        print(f"Fitted consensus subspace:")
        print(f"  Components: {self.k_components}")
        print(f"  Total variance explained: {sum(var_exp):.2%}")
        
        return self
    
    def transform(
        self,
        embeddings: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        """Extract residuals from new embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        residuals = extract_observer_residuals(
            [embeddings],
            self.consensus_subspace,
            normalize=normalize
        )[0]
        
        return residuals
    
    def fit_transform(
        self,
        observer_embeddings: List[torch.Tensor],
        normalize: bool = False
    ) -> List[torch.Tensor]:
        """Fit and transform in one step."""
        self.fit(observer_embeddings)
        return extract_observer_residuals(
            observer_embeddings,
            self.consensus_subspace,
            normalize=normalize
        )
    
    def save(self, path: str | Path):
        """Save fitted consensus subspace."""
        if not self.is_fitted:
            raise RuntimeError("Must fit before saving")
        
        torch.save({
            'consensus_subspace': self.consensus_subspace,
            'singular_values': self.singular_values,
            'variance_explained': self.variance_explained,
            'k_components': self.k_components,
            'method': self.method
        }, path)
        
        print(f"Saved consensus subspace to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'ObserverResidualExtractor':
        """Load fitted consensus subspace."""
        data = torch_load_trusted(path)
        
        extractor = cls(
            k_components=data['k_components'],
            method=data['method']
        )
        
        extractor.consensus_subspace = data['consensus_subspace']
        extractor.singular_values = data['singular_values']
        extractor.variance_explained = data['variance_explained']
        extractor.is_fitted = True
        
        print(f"Loaded consensus subspace from {path}")
        print(f"  Variance explained: {sum(extractor.variance_explained):.2%}")
        
        return extractor


if __name__ == "__main__":
    print("="*70)
    print("Testing Observer Residuals (Option 2)")
    print("="*70)
    
    # Create synthetic observer data
    torch.manual_seed(42)
    
    N = 1000  # Articles
    D = 512   # Feature dim
    k_obs = 5  # Observers
    
    # Shared semantic structure (consensus)
    consensus_true = torch.randn(N, 20)  # True consensus is 20D
    consensus_embedding = consensus_true @ torch.randn(20, D)
    
    # Observer-specific noise/interpretations
    observer_embeddings = []
    for i in range(k_obs):
        observer_noise = torch.randn(N, D) * 0.3
        observer_emb = consensus_embedding + observer_noise
        observer_embeddings.append(observer_emb)
    
    print(f"\nSynthetic data:")
    print(f"  {k_obs} observers")
    print(f"  {N} articles")
    print(f"  {D}D features")
    print(f"  True consensus: 20D subspace")
    
    # Test 1: Compute consensus
    print("\n" + "="*70)
    print("TEST 1: Compute Consensus Subspace")
    print("="*70)
    
    U, S, var_exp = compute_consensus_subspace(
        observer_embeddings,
        k_components=25,
        method='svd'
    )
    
    print(f"\nConsensus subspace: {U.shape}")
    print(f"Variance explained by top 10 components:")
    for i, v in enumerate(var_exp[:10]):
        print(f"  Component {i+1}: {v:.2%}")
    print(f"Total: {sum(var_exp):.2%}")
    
    # Test 2: Extract residuals
    print("\n" + "="*70)
    print("TEST 2: Extract Observer Residuals")
    print("="*70)
    
    residuals = extract_observer_residuals(observer_embeddings, U)
    
    print(f"\nOriginal embedding norms:")
    for i, emb in enumerate(observer_embeddings):
        print(f"  Observer {i+1}: {emb.norm(dim=-1).mean():.2f}")
    
    print(f"\nResidual embedding norms:")
    for i, res in enumerate(residuals):
        print(f"  Observer {i+1}: {res.norm(dim=-1).mean():.2f}")
    
    # Test 3: Analysis
    print("\n" + "="*70)
    print("TEST 3: Consensus vs Residual Analysis")
    print("="*70)
    
    analysis = analyze_consensus_vs_residuals(observer_embeddings, U)
    
    print(f"\nConsensus variance: {analysis['consensus_variance_fraction']:.2%}")
    print(f"Residual variance: {analysis['residual_variance_fraction']:.2%}")
    print(f"Inter-observer agreement: {analysis['inter_observer_agreement']:.3f}")
    
    # Test 4: Extractor class
    print("\n" + "="*70)
    print("TEST 4: ObserverResidualExtractor Class")
    print("="*70)
    
    extractor = ObserverResidualExtractor(k_components=20)
    residuals_2 = extractor.fit_transform(observer_embeddings)
    
    print(f"\nFitted and transformed {len(residuals_2)} observers")
    print(f"Residual shapes: {[r.shape for r in residuals_2]}")
    
    # Save and load
    extractor.save("test_consensus.pt")
    loaded = ObserverResidualExtractor.load("test_consensus.pt")
    
    # Apply to new data
    new_emb = torch.randn(100, D)
    new_residual = loaded.transform(new_emb)
    print(f"\nApplied to new data: {new_residual.shape}")
    
    # Cleanup
    Path("test_consensus.pt").unlink()
    
    print("\n" + "="*70)
    print("✓ Observer residuals module tests complete!")
    print("="*70)