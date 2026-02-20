"""
UNIFIED CONTROL ANALYSIS - COMPLETE

Combines:
- Simple pairwise variance
- Advanced geometry metrics (Procrustes, distance correlation, kNN, clustering)
- Consensus/Residual decomposition (the 87% metric)
- Mixed shape reconciliation
- Multi-kernel support
- Statistical comparisons

Usage:
    # Basic comparison
    python compare_controls_UNIFIED.py --data-dir outputs
    
    # With specific seeds
    python compare_controls_UNIFIED.py --data-dir outputs --seeds 42 43 44 45 46
    
    # With residual analysis
    python compare_controls_UNIFIED.py --data-dir outputs --residual-analysis
    
    # Multi-kernel mode
    python compare_controls_UNIFIED.py --data-dir outputs --multi-kernel
"""

# ============================================================================
# WINDOWS UNICODE FIX - Must be before all other imports
# ============================================================================
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# ============================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind, spearmanr
from scipy.linalg import orthogonal_procrustes as scipy_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict


# =============================================================================
# CORE MATH UTILITIES
# =============================================================================

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
def center(X: np.ndarray) -> np.ndarray:
    """Center matrix (zero mean)."""
    return X - X.mean(axis=0, keepdims=True)


def fro_norm(X: np.ndarray) -> float:
    """Frobenius norm."""
    return float(np.linalg.norm(X, ord='fro'))


def normalize_fro(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize to unit Frobenius norm."""
    n = fro_norm(X)
    if n < eps:
        return X.copy()
    return X / n


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Find orthogonal matrix R that best maps A -> B.
    Returns R such that ||A @ R - B||_F is minimized.
    """
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return R


def median_absolute_deviation(x: List[float]) -> float:
    """Robust variance estimate."""
    if len(x) == 0:
        return 0.0
    arr = np.array(x)
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


# =============================================================================
# FILE LOADING
# =============================================================================

def parse_seed_from_filename(filename: str) -> Optional[int]:
    """Extract seed from filename like seed_42.pt or observer_42.pt."""
    import re
    # Try patterns: seed_42, observer_42, *_42
    patterns = [
        r'seed[_\-](\d+)',
        r'observer[_\-](\d+)',
        r'_(\d+)\.pt$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return None


def load_observer_files(file_list: List[Path], filter_seeds: Optional[List[int]] = None) -> Dict[int, Dict]:
    """Load observer .pt files and extract features."""
    observers = {}
    
    for fpath in file_list:
        try:
            # Extract seed
            seed = parse_seed_from_filename(fpath.name)
            if seed is None:
                print(f"  âš  Could not parse seed from {fpath.name}")
                continue
            
            # Filter seeds if requested
            if filter_seeds is not None and seed not in filter_seeds:
                continue
            
            # Load data
            data = torch_load_trusted(fpath, map_location='cpu')
            
            # Extract features
            if isinstance(data, dict):
                features = data.get('features', data.get('embeddings', data.get('article_tokens')))
            else:
                features = data
            
            if features is None:
                print(f"  âœ— No features in {fpath.name}")
                continue
            
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            
            observers[seed] = {
                'features': features,
                'file': fpath.name,
                'shape': features.shape
            }
            
        except Exception as e:
            print(f"  âœ— Failed to load {fpath.name}: {e}")
            continue
    
    return observers


def reconcile_shapes(observers: Dict[int, Dict], mode: str = 'most_common') -> Dict[int, Dict]:
    """
    Handle mixed shapes by filtering or truncating.
    
    Modes:
        - 'most_common': Keep only observers with most common shape
        - 'intersection': Truncate all to minimum size
        - 'strict': Raise error on mismatch
    """
    shapes = [obs['shape'] for obs in observers.values()]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) == 1:
        return observers  # All same shape, no action needed
    
    print(f"\n  âš  Mixed shapes detected: {unique_shapes}")
    
    if mode == 'strict':
        raise ValueError(f"Shape mismatch! {unique_shapes}")
    
    elif mode == 'most_common':
        shape_counter = Counter(shapes)
        most_common = shape_counter.most_common(1)[0][0]
        print(f"    Keeping most common shape: {most_common}")
        
        filtered = {seed: obs for seed, obs in observers.items() 
                   if obs['shape'] == most_common}
        print(f"    Kept {len(filtered)}/{len(observers)} observers")
        return filtered
    
    elif mode == 'intersection':
        # Find minimum dimensions
        min_n = min(s[0] for s in shapes)
        print(f"    Truncating all to {min_n} items")
        
        truncated = {}
        for seed, obs in observers.items():
            truncated[seed] = {
                'features': obs['features'][:min_n],
                'file': obs['file'],
                'shape': (min_n, obs['shape'][1])
            }
        return truncated
    
    else:
        raise ValueError(f"Unknown reconcile mode: {mode}")


# =============================================================================
# GEOMETRY METRICS
# =============================================================================

def metric_procrustes_residual(obs1: np.ndarray, obs2: np.ndarray) -> float:
    """
    Rotation-invariant distance between geometries.
    Lower = more similar.
    """
    # Center and normalize
    obs1_c = center(obs1)
    obs2_c = center(obs2)
    obs1_norm = normalize_fro(obs1_c)
    obs2_norm = normalize_fro(obs2_c)
    
    # Find optimal rotation
    R = orthogonal_procrustes(obs1_norm, obs2_norm)
    
    # Residual after alignment
    aligned = obs1_norm @ R
    residual = fro_norm(aligned - obs2_norm)
    
    return residual


def metric_distance_correlation(obs1: np.ndarray, obs2: np.ndarray) -> float:
    """
    Spearman correlation of pairwise distance matrices.
    Returns: 1 - Ï (so lower = more similar)
    """
    dist1 = pdist(obs1)
    dist2 = pdist(obs2)
    rho, _ = spearmanr(dist1, dist2)
    return 1 - rho


def metric_cluster_stability(obs1: np.ndarray, obs2: np.ndarray, n_clusters: int = 5) -> float:
    """
    Cluster agreement (Adjusted Rand Index).
    Returns: 1 - ARI (so lower = more stable)
    """
    kmeans1 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    labels1 = kmeans1.fit_predict(obs1)
    labels2 = kmeans2.fit_predict(obs2)
    
    ari = adjusted_rand_score(labels1, labels2)
    return 1 - ari


def metric_knn_overlap(obs1: np.ndarray, obs2: np.ndarray, k: int = 15) -> float:
    """
    Average Jaccard overlap of k-nearest-neighbor sets.
    Returns: 1 - mean(Jaccard) (so lower = better overlap)
    """
    n_articles = obs1.shape[0]
    
    nbrs1 = NearestNeighbors(n_neighbors=min(k+1, n_articles)).fit(obs1)
    nbrs2 = NearestNeighbors(n_neighbors=min(k+1, n_articles)).fit(obs2)
    
    _, indices1 = nbrs1.kneighbors(obs1)
    _, indices2 = nbrs2.kneighbors(obs2)
    
    overlaps = []
    for i in range(n_articles):
        set1 = set(indices1[i, 1:])  # Exclude self
        set2 = set(indices2[i, 1:])
        
        if len(set1 | set2) > 0:
            jaccard = len(set1 & set2) / len(set1 | set2)
            overlaps.append(jaccard)
    
    return 1 - np.mean(overlaps) if overlaps else 1.0


def metric_intrinsic_dimension(obs: np.ndarray, n_components: int = 50, threshold: float = 0.90) -> float:
    """
    Estimate intrinsic dimensionality via PCA.
    Returns: Number of components needed to explain threshold variance.
    """
    pca = PCA(n_components=min(n_components, obs.shape[1]))
    pca.fit(obs)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.searchsorted(cumsum, threshold) + 1
    
    return intrinsic_dim


# =============================================================================
# GRAM-FIRST ANALYSIS (per diagnostic mandate)
# =============================================================================
# Coordinate-space metrics (variance, Procrustes) are fragile under normalization.
# Gram-based metrics are the primary observables for wavelength analysis.

def compute_gram_matrix(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute normalized Gram matrix.
    
    This is the PRIMARY metric per the diagnostic mandate -
    coordinate-space metrics are fragile under normalization.
    """
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = X / norms
    
    return X @ X.T


def gram_frobenius_distance(G1: np.ndarray, G2: np.ndarray) -> float:
    """
    Compute ||G1 - G2||_F (wavelength energy).
    
    This measures how much the geometry changes between two conditions.
    """
    diff = G1 - G2
    return float(np.linalg.norm(diff, ord='fro'))


def gram_knn_flip_rate(G1: np.ndarray, G2: np.ndarray, k: int = 10) -> float:
    """
    Compute kNN neighbor flip rate from Gram matrices.
    
    Measures local structure stability - how many nearest neighbors change.
    """
    n = G1.shape[0]
    k = min(k, n - 1)
    
    # Get k-nearest neighbors from each Gram (higher similarity = nearer)
    idx1 = np.argsort(-G1, axis=1)[:, 1:k+1]  # Exclude self (diagonal)
    idx2 = np.argsort(-G2, axis=1)[:, 1:k+1]
    
    # Compute flip rate
    flips = 0
    for i in range(n):
        set1 = set(idx1[i].tolist())
        set2 = set(idx2[i].tolist())
        flips += len(set1 - set2)
    
    return flips / (n * k)


def gram_spectrum_stats(G: np.ndarray) -> Dict[str, float]:
    """
    Compute eigenvalue spectrum statistics from Gram matrix.
    
    These capture the intrinsic geometry of the embedding space.
    """
    # Eigendecompose
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]  # Descending
    eigvals = np.maximum(eigvals, 1e-12)
    
    # Participation ratio (effective rank)
    total = eigvals.sum()
    pr = (total ** 2) / (eigvals ** 2).sum() if (eigvals ** 2).sum() > 0 else 1.0
    
    # EV90: components for 90% variance
    cumsum = np.cumsum(eigvals) / total
    ev90 = int(np.searchsorted(cumsum, 0.9)) + 1
    
    # Lambda ratio
    lambda_ratio = float(eigvals[0] / eigvals[1]) if len(eigvals) > 1 else float('inf')
    
    return {
        'participation_ratio': float(pr),
        'ev90_components': ev90,
        'lambda1_lambda2_ratio': lambda_ratio,
        'top_eigenvalue': float(eigvals[0]),
        'trace': float(total),
    }


def compare_gram_metrics(
    obs1: np.ndarray,
    obs2: np.ndarray,
    label1: str = "A",
    label2: str = "B",
) -> Dict[str, Any]:
    """
    Compare two observation matrices using Gram-first metrics.
    
    This is the recommended comparison method per the diagnostic mandate.
    """
    # Handle shape mismatch
    n1, d1 = obs1.shape
    n2, d2 = obs2.shape
    
    if n1 != n2:
        min_n = min(n1, n2)
        obs1 = obs1[:min_n]
        obs2 = obs2[:min_n]
    
    if d1 != d2:
        min_d = min(d1, d2)
        obs1 = obs1[:, :min_d]
        obs2 = obs2[:, :min_d]
    
    # Compute Gram matrices
    G1 = compute_gram_matrix(obs1)
    G2 = compute_gram_matrix(obs2)
    
    # Wavelength metrics
    fro_dist = gram_frobenius_distance(G1, G2)
    flip_rate = gram_knn_flip_rate(G1, G2, k=10)
    
    # Spectrum stats for each
    spec1 = gram_spectrum_stats(G1)
    spec2 = gram_spectrum_stats(G2)
    
    # Spectrum divergence
    pr_diff = abs(spec1['participation_ratio'] - spec2['participation_ratio'])
    ev90_diff = abs(spec1['ev90_components'] - spec2['ev90_components'])
    
    return {
        'gram_frobenius_distance': fro_dist,
        'gram_knn_flip_rate': flip_rate,
        'spectrum_A': spec1,
        'spectrum_B': spec2,
        'participation_ratio_diff': pr_diff,
        'ev90_components_diff': ev90_diff,
        'labels': (label1, label2),
    }


# =============================================================================
# PROCRUSTES CONTROLS (SANITY + REAL-vs-CONTROL SUMMARY)
# =============================================================================

def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity in QR
    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    Q = Q * diag
    return Q


def run_procrustes_sanity_controls(rng_seed: int = 0,
                                  n_points: int = 300,
                                  dim: int = 64) -> Dict[str, Any]:
    """
    Synthetic controls to validate Procrustes behavior:

    1) Rigid transform invariance:
       X2 = X @ R + t  => Procrustes residual should be near 0 after centering/normalization.

    2) Scale sensitivity:
       X2 = s * X      => Residual should increase as s moves away from 1 (orthogonal Procrustes does NOT scale).

    3) Unrelated:
       X and Y random  => Residual should be much larger than rigid case.

    This does NOT validate semantics. It validates your Procrustes *machinery*.
    """
    rng = np.random.default_rng(rng_seed)

    X = rng.normal(size=(n_points, dim))

    # Rigid (rotation + translation)
    R = _random_orthogonal(dim, rng)
    t = rng.normal(size=(dim,)) * 2.0
    X_rigid = (X @ R) + t
    rigid_residual = metric_procrustes_residual(X, X_rigid)

    # Scaling (should not be invariant)
    scales = [0.5, 0.8, 1.2, 2.0]
    scale_residuals = {}
    for s in scales:
        scale_residuals[str(s)] = metric_procrustes_residual(X, X * s)

    # Unrelated
    Y = rng.normal(size=(n_points, dim))
    unrelated_residual = metric_procrustes_residual(X, Y)

    return {
        "rigid_residual": float(rigid_residual),
        "scale_residuals": {k: float(v) for k, v in scale_residuals.items()},
        "unrelated_residual": float(unrelated_residual),
        "config": {"rng_seed": rng_seed, "n_points": n_points, "dim": dim},
        "expected_behavior": {
            "rigid_residual": "near 0",
            "scale_residuals": "increase as scale deviates from 1",
            "unrelated_residual": ">> rigid_residual"
        }
    }


def summarize_real_vs_controls_procrustes(results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience summary: Real vs each control, for Procrustes mean + ratio.
    This is not a separate metric; it's just an explicit output so you can't miss it.
    """
    out = {"available": False, "rows": []}
    if "Real" not in results_dict:
        return out
    if "procrustes" not in results_dict["Real"]:
        return out

    real_mean = results_dict["Real"]["procrustes"]["mean"]
    out["available"] = True

    for ctrl in ["Constant", "Shuffled", "Random"]:
        if ctrl in results_dict and "procrustes" in results_dict[ctrl]:
            ctrl_mean = results_dict[ctrl]["procrustes"]["mean"]
            ratio = real_mean / ctrl_mean if ctrl_mean > 0 else float("inf")
            out["rows"].append({
                "control": ctrl,
                "real_mean": float(real_mean),
                "control_mean": float(ctrl_mean),
                "ratio_real_over_control": float(ratio)
            })
    return out


# =============================================================================
# CONSENSUS/RESIDUAL DECOMPOSITION (The 87% metric!)
# =============================================================================

def compute_consensus_residual(observers: Dict[int, Dict], k_components: int = 10) -> Dict[str, Any]:
    """
    Decompose variance into consensus (shared) vs residual (observer-specific).
    
    Returns:
        consensus_variance: Variance in shared structure
        residual_variance: Variance in observer-specific deviations
        consensus_pct: % of variance in consensus
        residual_pct: % of variance in residuals (the 87% metric!)
    """
    seeds = sorted(observers.keys())
    embeddings = [observers[s]['features'] for s in seeds]
    
    # Stack: [n_observers, n_items, n_dims]
    stacked = np.stack(embeddings, axis=0)
    n_obs, n_items, n_dims = stacked.shape
    
    print(f"\n  Computing consensus/residual decomposition...")
    print(f"    {n_obs} observers Ã— {n_items} items Ã— {n_dims} dims")
    
    # Method 1: Per-item mean (simple consensus)
    consensus = stacked.mean(axis=0)  # [n_items, n_dims]
    residuals = stacked - consensus[None, :, :]  # [n_obs, n_items, n_dims]
    
    # Variance decomposition
    total_var = np.var(stacked)
    consensus_var = np.var(consensus)
    residual_var = np.var(residuals)
    
    consensus_pct = (consensus_var / total_var * 100) if total_var > 0 else 0
    residual_pct = (residual_var / total_var * 100) if total_var > 0 else 0
    
    print(f"    Total variance: {total_var:.6f}")
    print(f"    Consensus: {consensus_pct:.2f}%")
    print(f"    Residual: {residual_pct:.2f}%")
    
    # Method 2: PCA-based (for comparison)
    # Reshape for PCA
    all_data = stacked.reshape(-1, n_dims)
    pca = PCA(n_components=min(k_components, n_dims))
    pca.fit(all_data)
    
    pca_consensus_var = pca.explained_variance_ratio_.sum()
    pca_residual_var = 1 - pca_consensus_var
    
    print(f"    PCA consensus (top-{k_components}): {pca_consensus_var*100:.2f}%")
    print(f"    PCA residual: {pca_residual_var*100:.2f}%")
    
    return {
        'total_variance': float(total_var),
        'consensus_variance': float(consensus_var),
        'residual_variance': float(residual_var),
        'consensus_pct': float(consensus_pct),
        'residual_pct': float(residual_pct),
        'pca_consensus_pct': float(pca_consensus_var * 100),
        'pca_residual_pct': float(pca_residual_var * 100),
        'n_observers': n_obs,
        'n_items': n_items,
        'n_dims': n_dims
    }


# =============================================================================
# COMPREHENSIVE METRICS COMPUTATION
# =============================================================================

def compute_all_metrics(observers: Dict[int, Dict], 
                       include_advanced: bool = True,
                       include_residual: bool = True,
                       include_gram: bool = True,
                       k_components: int = 10) -> Dict[str, Any]:
    """
    Compute ALL metrics for a set of observers.
    
    Returns dict with:
        - simple_variance: Mean pairwise absolute difference
        - procrustes: Procrustes residuals
        - distance_corr: Distance correlation
        - cluster_stability: Cluster agreement
        - knn_overlap: Neighborhood preservation
        - intrinsic_dim: Manifold dimensionality
        - consensus_residual: Consensus/residual decomposition
        - gram_analysis: Gram-first metrics (PRIMARY per diagnostic mandate)
    """
    seeds = sorted(observers.keys())
    n_obs = len(seeds)
    
    if n_obs < 2:
        return {'error': f'Need at least 2 observers (have {n_obs})'}
    
    results = {
        'n_observers': n_obs,
        'seeds': seeds
    }
    
    # Simple pairwise metrics
    simple_diffs = []
    procrustes_vals = []
    distcorr_vals = []
    cluster_vals = []
    knn_vals = []
    
    # Gram-first metrics (PRIMARY)
    gram_fro_vals = []
    gram_flip_vals = []
    
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            obs1 = observers[s1]['features']
            obs2 = observers[s2]['features']
            
            # Simple absolute difference
            simple = np.abs(obs1 - obs2).mean()
            simple_diffs.append(simple)
            
            if include_advanced:
                # Advanced metrics
                procrustes_vals.append(metric_procrustes_residual(obs1, obs2))
                distcorr_vals.append(metric_distance_correlation(obs1, obs2))
                cluster_vals.append(metric_cluster_stability(obs1, obs2))
                knn_vals.append(metric_knn_overlap(obs1, obs2))
            
            if include_gram:
                # Gram-first metrics (PRIMARY per diagnostic mandate)
                gram_result = compare_gram_metrics(obs1, obs2, f"seed_{s1}", f"seed_{s2}")
                gram_fro_vals.append(gram_result['gram_frobenius_distance'])
                gram_flip_vals.append(gram_result['gram_knn_flip_rate'])
    
    # Simple variance
    results['simple_variance'] = {
        'mean': float(np.mean(simple_diffs)),
        'std': float(np.std(simple_diffs)),
        'median': float(np.median(simple_diffs)),
        'mad': median_absolute_deviation(simple_diffs),
        'values': [float(v) for v in simple_diffs]
    }
    
    # Gram-first analysis (PRIMARY per diagnostic mandate)
    if include_gram and gram_fro_vals:
        results['gram_analysis'] = {
            'note': 'PRIMARY METRIC per diagnostic mandate - coordinate metrics are fragile',
            'gram_frobenius': {
                'mean': float(np.mean(gram_fro_vals)),
                'std': float(np.std(gram_fro_vals)),
                'median': float(np.median(gram_fro_vals)),
                'values': [float(v) for v in gram_fro_vals]
            },
            'gram_knn_flip_rate': {
                'mean': float(np.mean(gram_flip_vals)),
                'std': float(np.std(gram_flip_vals)),
                'median': float(np.median(gram_flip_vals)),
                'values': [float(v) for v in gram_flip_vals]
            },
        }
        
        # Add spectrum stats for first observer (representative)
        first_obs = observers[seeds[0]]['features']
        G = compute_gram_matrix(first_obs)
        results['gram_analysis']['spectrum'] = gram_spectrum_stats(G)
    
    if include_advanced:
        # Advanced metrics
        for name, vals in [
            ('procrustes', procrustes_vals),
            ('distance_corr', distcorr_vals),
            ('cluster_stability', cluster_vals),
            ('knn_overlap', knn_vals)
        ]:
            results[name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
                'mad': median_absolute_deviation(vals),
                'values': [float(v) for v in vals]
            }
        
        # Intrinsic dimension (per observer, then average)
        dims = [metric_intrinsic_dimension(observers[s]['features']) for s in seeds]
        results['intrinsic_dim'] = {
            'mean': float(np.mean(dims)),
            'std': float(np.std(dims)),
            'values': [float(d) for d in dims]
        }
    
    if include_residual:
        # Consensus/residual decomposition
        results['consensus_residual'] = compute_consensus_residual(observers, k_components)
    
    return results


# =============================================================================
# EXPERIMENT LOADING
# =============================================================================

def load_all_experiments(data_dir: Path, 
                        filter_seeds: Optional[List[int]] = None,
                        reconcile_mode: str = 'most_common') -> Dict[str, Dict[int, Dict]]:
    """Load all experiments from experiment directory with subdirectories."""
    print(f"\n{'='*70}")
    print("LOADING EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Directory: {data_dir}")
    if filter_seeds:
        print(f"Filtering seeds: {filter_seeds}")
    
    experiments = {}
    
    # Real corpus - check both locations
    print("\n[1/4] Real corpus:")
    real_files = []
    
    # NEW structure: real/observer_*.pt or real/seed_*.pt
    real_dir = data_dir / 'real'
    if real_dir.exists():
        real_files = sorted(list(real_dir.glob('observer_*.pt')) + list(real_dir.glob('seed_*.pt')))
    
    # OLD structure: seed_*.pt in root
    if not real_files:
        real_files = sorted(data_dir.glob('seed_*.pt'))
    
    if real_files:
        print(f"  Found {len(real_files)} files")
        real_obs = load_observer_files(real_files, filter_seeds)
        
        if real_obs:
            real_obs = reconcile_shapes(real_obs, reconcile_mode)
            experiments['Real'] = real_obs
            print(f"  âœ“ Loaded {len(real_obs)} observers")
            print(f"    Shape: {list(real_obs.values())[0]['shape']}")
    else:
        print(f"  âœ— No real corpus files found")
    
    # Control_Constant
    print("\n[2/4] Control_Constant:")
    const_files = []
    
    # NEW structure: control_constant/observer_*.pt
    const_dir = data_dir / 'control_constant'
    if const_dir.exists():
        const_files = sorted(list(const_dir.glob('observer_*.pt')) + list(const_dir.glob('seed_*.pt')))
    
    # OLD structure: control_constant_*_observer_*.pt in root
    if not const_files:
        const_files = sorted(data_dir.glob('control_constant_*_observer_*.pt'))
    
    if const_files:
        print(f"  Found {len(const_files)} files")
        const_obs = load_observer_files(const_files, filter_seeds)
        
        if const_obs:
            const_obs = reconcile_shapes(const_obs, reconcile_mode)
            experiments['Constant'] = const_obs
            print(f"  âœ“ Loaded {len(const_obs)} observers")
            print(f"    Shape: {list(const_obs.values())[0]['shape']}")
    else:
        print(f"  âœ— No constant control files found")
    
    # Control_Shuffled
    print("\n[3/4] Control_Shuffled:")
    shuf_files = []
    
    # NEW structure: control_shuffled/observer_*.pt
    shuf_dir = data_dir / 'control_shuffled'
    if shuf_dir.exists():
        shuf_files = sorted(list(shuf_dir.glob('observer_*.pt')) + list(shuf_dir.glob('seed_*.pt')))
    
    # OLD structure: control_shuffled_*_observer_*.pt in root
    if not shuf_files:
        shuf_files = sorted(data_dir.glob('control_shuffled_*_observer_*.pt'))
    
    if shuf_files:
        print(f"  Found {len(shuf_files)} files")
        shuf_obs = load_observer_files(shuf_files, filter_seeds)
        
        if shuf_obs:
            shuf_obs = reconcile_shapes(shuf_obs, reconcile_mode)
            experiments['Shuffled'] = shuf_obs
            print(f"  âœ“ Loaded {len(shuf_obs)} observers")
            print(f"    Shape: {list(shuf_obs.values())[0]['shape']}")
    else:
        print(f"  âœ— No shuffled control files found")
    
    # Control_Random
    print("\n[4/4] Control_Random:")
    rand_files = []
    
    # NEW structure: control_random/observer_*.pt
    rand_dir = data_dir / 'control_random'
    if rand_dir.exists():
        rand_files = sorted(list(rand_dir.glob('observer_*.pt')) + list(rand_dir.glob('seed_*.pt')))
    
    # OLD structure: control_random_*_observer_*.pt in root
    if not rand_files:
        rand_files = sorted(data_dir.glob('control_random_*_observer_*.pt'))
    
    if rand_files:
        print(f"  Found {len(rand_files)} files")
        rand_obs = load_observer_files(rand_files, filter_seeds)
        
        if rand_obs:
            rand_obs = reconcile_shapes(rand_obs, reconcile_mode)
            experiments['Random'] = rand_obs
            print(f"  âœ“ Loaded {len(rand_obs)} observers")
            print(f"    Shape: {list(rand_obs.values())[0]['shape']}")
    else:
        print(f"  âœ— No random control files found")
    
    return experiments


# =============================================================================
# MULTI-KERNEL SUPPORT
# =============================================================================

def discover_kernel_dirs(data_dir: Path) -> List[Path]:
    """
    Multi-kernel layout expected:
        data_dir/
            rbf/
                real/
                control_constant/
                control_shuffled/
                control_random/
            laplacian/
                ...
    This function returns subdirectories that look like kernels.
    """
    if not data_dir.exists():
        return []
    kernel_dirs = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir():
            # Heuristic: contains any of the expected corpus subdirs
            if (p / "real").exists() or (p / "control_constant").exists() or (p / "control_shuffled").exists() or (p / "control_random").exists():
                kernel_dirs.append(p)
    return kernel_dirs


# =============================================================================
# STATISTICAL COMPARISON
# =============================================================================

def compare_metrics(results_dict: Dict[str, Dict[str, Any]], 
                   metric_name: str = 'simple_variance') -> pd.DataFrame:
    """Compare a specific metric across corpora."""
    
    comparisons = []
    corpus_names = list(results_dict.keys())
    
    for i, name1 in enumerate(corpus_names):
        for name2 in corpus_names[i+1:]:
            if metric_name not in results_dict[name1] or metric_name not in results_dict[name2]:
                continue
            
            vals1 = results_dict[name1][metric_name]['values']
            vals2 = results_dict[name2][metric_name]['values']
            
            # T-test
            t_stat, p_val = ttest_ind(vals1, vals2)
            
            # Effect size
            mean1 = results_dict[name1][metric_name]['mean']
            mean2 = results_dict[name2][metric_name]['mean']
            ratio = mean1 / mean2 if mean2 > 0 else float('inf')
            
            comparisons.append({
                'corpus_1': name1,
                'corpus_2': name2,
                'metric': metric_name,
                'mean_1': mean1,
                'mean_2': mean2,
                'ratio': ratio,
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    return pd.DataFrame(comparisons)


def interpret_results(results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Provide comprehensive interpretation."""
    
    if 'Real' not in results_dict:
        return {'error': 'No real corpus found'}
    
    interpretation = {
        'has_real': True,
        'has_controls': len(results_dict) > 1,
        'metrics': {}
    }
    
    # For each metric, compare real vs controls
    metric_names = ['simple_variance', 'procrustes', 'distance_corr', 
                   'cluster_stability', 'knn_overlap']
    
    for metric_name in metric_names:
        if metric_name not in results_dict['Real']:
            continue
        
        real_val = results_dict['Real'][metric_name]['mean']
        control_vals = []
        
        for corpus_name, corpus_results in results_dict.items():
            if corpus_name != 'Real' and metric_name in corpus_results:
                control_vals.append(corpus_results[metric_name]['mean'])
        
        if control_vals:
            avg_control = np.mean(control_vals)
            ratio = real_val / avg_control if avg_control > 0 else float('inf')
            
            interpretation['metrics'][metric_name] = {
                'real_value': real_val,
                'control_avg': avg_control,
                'ratio': ratio,
                'separates': abs(ratio - 1.0) > 0.2  # More than 20% different
            }
    
    # Consensus/residual interpretation
    if 'consensus_residual' in results_dict['Real']:
        cr_real = results_dict['Real']['consensus_residual']
        
        interpretation['consensus_residual'] = {
            'real': cr_real,
            'thesis_claim': (
                f"{cr_real['residual_pct']:.1f}% of variance is observer-specific "
                f"({cr_real['consensus_pct']:.1f}% is shared)"
            )
        }

    # NEW: explicit Procrustes Real vs Controls summary
    interpretation['procrustes_real_vs_controls'] = summarize_real_vs_controls_procrustes(results_dict)
    
    return interpretation


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comprehensive_comparison(results_dict: Dict[str, Dict[str, Any]], 
                                  output_dir: Path):
    """Generate comprehensive comparison plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which metrics are available
    available_metrics = []
    for metric in ['simple_variance', 'procrustes', 'distance_corr', 
                   'cluster_stability', 'knn_overlap']:
        if any(metric in res for res in results_dict.values()):
            available_metrics.append(metric)
    
    n_metrics = len(available_metrics)
    if n_metrics == 0:
        print("  âš  No metrics to plot")
        return
    
    # Create grid
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    corpus_names = list(results_dict.keys())
    colors = ['#2E86AB' if name == 'Real' else '#A23B72' for name in corpus_names]
    
    for idx, metric_name in enumerate(available_metrics):
        ax = axes[idx]
        
        # Extract values
        means = []
        stds = []
        labels = []
        
        for corpus_name in corpus_names:
            if metric_name in results_dict[corpus_name]:
                means.append(results_dict[corpus_name][metric_name]['mean'])
                stds.append(results_dict[corpus_name][metric_name]['std'])
                labels.append(corpus_name)
        
        # Bar plot
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
               color=[colors[corpus_names.index(l)] for l in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'comprehensive_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  âœ“ Saved plot: {plot_path}")
    
    plt.close()


def plot_kernel_summary(kernel_summaries: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Multi-kernel summary plot: for each kernel, show Real-vs-controls Procrustes ratio (if available).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kname, summary in kernel_summaries.items():
        interp = summary.get("interpretation", {})
        pblock = interp.get("procrustes_real_vs_controls", {})
        if pblock.get("available") and pblock.get("rows"):
            for r in pblock["rows"]:
                rows.append({
                    "kernel": kname,
                    "control": r["control"],
                    "ratio_real_over_control": r["ratio_real_over_control"]
                })
    if not rows:
        return

    df = pd.DataFrame(rows)
    # Pivot for easier plotting
    piv = df.pivot_table(index="kernel", columns="control", values="ratio_real_over_control", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 5))
    piv.plot(kind="bar", ax=ax)
    ax.set_title("Procrustes Real vs Controls (ratio) across kernels")
    ax.set_ylabel("Real / Control (mean Procrustes)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    outpath = output_dir / "multi_kernel_procrustes_ratios.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SAVING RESULTS
# =============================================================================

def save_comprehensive_results(results_dict: Dict[str, Dict[str, Any]],
                               comparisons: Dict[str, pd.DataFrame],
                               interpretation: Dict[str, Any],
                               output_dir: Path):
    """Save all results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # JSON with everything
    json_data = {
        'results': results_dict,
        'interpretation': interpretation,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    json_path = output_dir / 'comprehensive_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)
    print(f"  âœ“ Saved JSON: {json_path}")
    
    # CSV tables for each metric
    for metric_name, df in comparisons.items():
        csv_path = output_dir / f'{metric_name}_comparisons.csv'
        df.to_csv(csv_path, index=False)
        print(f"  âœ“ Saved CSV: {csv_path}")
    
    # Summary table
    summary_rows = []
    for corpus_name, corpus_results in results_dict.items():
        row = {'corpus': corpus_name}
        
        for metric in ['simple_variance', 'procrustes', 'distance_corr', 
                      'cluster_stability', 'knn_overlap', 'intrinsic_dim']:
            if metric in corpus_results:
                row[f'{metric}_mean'] = corpus_results[metric]['mean']
                row[f'{metric}_std'] = corpus_results[metric]['std']
        
        if 'consensus_residual' in corpus_results:
            cr = corpus_results['consensus_residual']
            row['consensus_pct'] = cr['consensus_pct']
            row['residual_pct'] = cr['residual_pct']
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  âœ“ Saved summary: {summary_path}")


# =============================================================================
# SINGLE-RUN ANALYSIS (one data_dir containing real + controls)
# =============================================================================

def run_single_analysis(data_dir: Path,
                        output_dir: Path,
                        seeds: Optional[List[int]],
                        reconcile: str,
                        include_advanced: bool,
                        include_residual: bool,
                        k_components: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame], Dict[str, Any]]:
    # Load experiments
    experiments = load_all_experiments(data_dir, seeds, reconcile)
    if not experiments:
        return {}, {}, {'error': 'No experiments loaded!'}

    # Compute all metrics
    print(f"\n{'='*70}")
    print("COMPUTING COMPREHENSIVE METRICS")
    print(f"{'='*70}")
    
    results_dict = {}
    
    for corpus_name, observers in experiments.items():
        print(f"\n{corpus_name}:")
        print(f"  Observers: {len(observers)}")
        
        results = compute_all_metrics(
            observers,
            include_advanced=include_advanced,
            include_residual=include_residual,
            k_components=k_components
        )
        
        if 'error' in results:
            print(f"  âœ— {results['error']}")
            continue
        
        results_dict[corpus_name] = results
        
        # Print summary
        print(f"\n  Simple variance: {results['simple_variance']['mean']:.6f}")
        
        if 'procrustes' in results:
            print(f"  Procrustes: {results['procrustes']['mean']:.6f}")
        
        if 'consensus_residual' in results:
            cr = results['consensus_residual']
            print(f"  Consensus: {cr['consensus_pct']:.1f}%")
            print(f"  Residual: {cr['residual_pct']:.1f}%")

    # Statistical comparisons
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISONS")
    print(f"{'='*70}")
    
    comparisons: Dict[str, pd.DataFrame] = {}
    # Only compare metrics that exist for ALL corpora in results_dict
    for metric_name in ['simple_variance', 'procrustes', 'distance_corr', 
                        'cluster_stability', 'knn_overlap']:
        if len(results_dict) > 1 and all(metric_name in res for res in results_dict.values()):
            df = compare_metrics(results_dict, metric_name)
            comparisons[metric_name] = df
            
            print(f"\n{metric_name.upper()}:")
            for _, row in df.iterrows():
                sig = "âœ“" if row['significant'] else "âœ—"
                print(f"  {row['corpus_1']} vs {row['corpus_2']}: "
                      f"{row['ratio']:.2f}x (p={row['p_value']:.4f}) {sig}")
        elif len(results_dict) > 1:
            # Still compute partial comparisons if metric exists for Real and any control
            # but skip printing tables to avoid confusion
            pass

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    interpretation = interpret_results(results_dict)
    
    if 'metrics' in interpretation:
        print("\nMetric separation (Real vs Controls):")
        for metric_name, metric_data in interpretation['metrics'].items():
            if metric_data['separates']:
                print(f"  âœ“ {metric_name}: {metric_data['ratio']:.2f}x")
            else:
                print(f"  âœ— {metric_name}: {metric_data['ratio']:.2f}x (weak)")
    
    if 'consensus_residual' in interpretation:
        print(f"\nConsensus/Residual:")
        print(f"  {interpretation['consensus_residual']['thesis_claim']}")

    # Explicit Procrustes Real-vs-controls line item
    pblock = interpretation.get("procrustes_real_vs_controls", {})
    if pblock.get("available") and pblock.get("rows"):
        print("\nProcrustes (Real vs Controls):")
        for r in pblock["rows"]:
            print(f"  Real vs {r['control']}: ratio={r['ratio_real_over_control']:.2f} "
                  f"(Real={r['real_mean']:.6f}, Control={r['control_mean']:.6f})")
    
    # Visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_comprehensive_comparison(results_dict, output_dir)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    save_comprehensive_results(results_dict, comparisons, interpretation, output_dir)

    return results_dict, comparisons, interpretation


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Control Analysis - Complete'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='outputs',
        help='Directory with experimental results'
    )
    
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        help='Filter to specific seeds (e.g., 42 43 44 45 46)'
    )
    
    parser.add_argument(
        '--reconcile',
        type=str,
        choices=['most_common', 'intersection', 'strict'],
        default='most_common',
        help='How to handle mixed shapes'
    )
    
    parser.add_argument(
        '--no-advanced',
        action='store_true',
        help='Skip advanced metrics (faster)'
    )

    # Alias requested in your docstring
    parser.add_argument(
        '--residual-analysis',
        action='store_true',
        help='Enable consensus/residual decomposition explicitly (alias for NOT setting --no-residual)'
    )
    
    parser.add_argument(
        '--no-residual',
        action='store_true',
        help='Skip consensus/residual decomposition'
    )

    parser.add_argument(
        '--run-procrustes-controls',
        action='store_true',
        help='Run synthetic Procrustes sanity controls before corpus analysis'
    )

    parser.add_argument(
        '--procrustes-control-seed',
        type=int,
        default=0,
        help='RNG seed for Procrustes sanity controls'
    )

    parser.add_argument(
        '--multi-kernel',
        action='store_true',
        help='Treat immediate subdirectories under --data-dir as kernels and run analysis per-kernel'
    )
    
    parser.add_argument(
        '--k-components',
        type=int,
        default=10,
        help='Number of PCA components for consensus'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/comprehensive_analysis',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Resolve residual-analysis alias
    # If user passes --residual-analysis, we force include_residual True regardless of --no-residual
    include_residual = (not args.no_residual) or args.residual_analysis
    include_advanced = (not args.no_advanced)

    print(f"{'='*70}")
    print("UNIFIED CONTROL ANALYSIS")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Seeds filter: {args.seeds if args.seeds else 'All'}")
    print(f"Shape reconciliation: {args.reconcile}")
    print(f"Advanced metrics: {include_advanced}")
    print(f"Residual analysis: {include_residual}")
    print(f"Multi-kernel: {args.multi_kernel}")

    # Optional: Procrustes sanity controls
    if args.run_procrustes_controls:
        print(f"\n{'='*70}")
        print("PROCRUSTES SANITY CONTROLS")
        print(f"{'='*70}")
        ctrl = run_procrustes_sanity_controls(rng_seed=args.procrustes_control_seed)
        print(f"Rigid residual: {ctrl['rigid_residual']:.6f}  (expected near 0)")
        print("Scale residuals (orthogonal Procrustes is NOT scale-invariant):")
        for s, v in ctrl["scale_residuals"].items():
            print(f"  scale={s:>4s} residual={v:.6f}")
        print(f"Unrelated residual: {ctrl['unrelated_residual']:.6f}  (expected >> rigid residual)")

    # Multi-kernel mode
    if args.multi_kernel:
        kernel_dirs = discover_kernel_dirs(data_dir)
        if not kernel_dirs:
            print("\nâœ— No kernel directories found under data-dir. Expected subfolders like 'rbf/', 'laplacian/', etc.")
            return

        kernel_summaries: Dict[str, Dict[str, Any]] = {}
        print(f"\nDiscovered {len(kernel_dirs)} kernels:")
        for kd in kernel_dirs:
            print(f"  - {kd.name}")

        for kd in kernel_dirs:
            kname = kd.name
            print(f"\n{'='*70}")
            print(f"KERNEL: {kname}")
            print(f"{'='*70}")

            out_k = output_dir / kname
            results_dict, comparisons, interpretation = run_single_analysis(
                data_dir=kd,
                output_dir=out_k,
                seeds=args.seeds,
                reconcile=args.reconcile,
                include_advanced=include_advanced,
                include_residual=include_residual,
                k_components=args.k_components
            )

            kernel_summaries[kname] = {
                "results": results_dict,
                "interpretation": interpretation
            }

        # Save a multi-kernel master summary
        output_dir.mkdir(parents=True, exist_ok=True)

        master = {
            "kernels": list(kernel_summaries.keys()),
            "timestamp": pd.Timestamp.now().isoformat(),
            "procrustes_real_vs_controls": {
                k: kernel_summaries[k].get("interpretation", {}).get("procrustes_real_vs_controls", {})
                for k in kernel_summaries.keys()
            }
        }
        master_path = output_dir / "multi_kernel_summary.json"
        with open(master_path, "w") as f:
            json.dump(master, f, indent=2)
        print(f"\n  âœ“ Saved multi-kernel summary: {master_path}")

        # Optional plot across kernels
        plot_kernel_summary(kernel_summaries, output_dir)
        print(f"\n{'='*70}")
        print("âœ“ COMPREHENSIVE MULTI-KERNEL ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"\nResults saved to: {output_dir}")
        return

    # Single-kernel (normal) mode
    experiments = load_all_experiments(data_dir, args.seeds, args.reconcile)
    
    if not experiments:
        print("\nâœ— No experiments loaded!")
        return
    
    # Compute all metrics
    print(f"\n{'='*70}")
    print("COMPUTING COMPREHENSIVE METRICS")
    print(f"{'='*70}")
    
    results_dict = {}
    
    for corpus_name, observers in experiments.items():
        print(f"\n{corpus_name}:")
        print(f"  Observers: {len(observers)}")
        
        results = compute_all_metrics(
            observers,
            include_advanced=include_advanced,
            include_residual=include_residual,
            k_components=args.k_components
        )
        
        if 'error' in results:
            print(f"  âœ— {results['error']}")
            continue
        
        results_dict[corpus_name] = results
        
        # Print summary
        print(f"\n  Simple variance: {results['simple_variance']['mean']:.6f}")
        
        if 'procrustes' in results:
            print(f"  Procrustes: {results['procrustes']['mean']:.6f}")
        
        if 'consensus_residual' in results:
            cr = results['consensus_residual']
            print(f"  Consensus: {cr['consensus_pct']:.1f}%")
            print(f"  Residual: {cr['residual_pct']:.1f}%")
    
    # Statistical comparisons
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISONS")
    print(f"{'='*70}")
    
    comparisons = {}
    for metric_name in ['simple_variance', 'procrustes', 'distance_corr', 
                       'cluster_stability', 'knn_overlap']:
        if len(results_dict) > 1 and all(metric_name in res for res in results_dict.values()):
            df = compare_metrics(results_dict, metric_name)
            comparisons[metric_name] = df
            
            print(f"\n{metric_name.upper()}:")
            for _, row in df.iterrows():
                sig = "âœ“" if row['significant'] else "âœ—"
                print(f"  {row['corpus_1']} vs {row['corpus_2']}: "
                      f"{row['ratio']:.2f}x (p={row['p_value']:.4f}) {sig}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    interpretation = interpret_results(results_dict)
    
    if 'metrics' in interpretation:
        print("\nMetric separation (Real vs Controls):")
        for metric_name, metric_data in interpretation['metrics'].items():
            if metric_data['separates']:
                print(f"  âœ“ {metric_name}: {metric_data['ratio']:.2f}x")
            else:
                print(f"  âœ— {metric_name}: {metric_data['ratio']:.2f}x (weak)")
    
    if 'consensus_residual' in interpretation:
        print(f"\nConsensus/Residual:")
        print(f"  {interpretation['consensus_residual']['thesis_claim']}")

    # NEW: explicit Procrustes Real-vs-controls output
    pblock = interpretation.get("procrustes_real_vs_controls", {})
    if pblock.get("available") and pblock.get("rows"):
        print("\nProcrustes (Real vs Controls):")
        for r in pblock["rows"]:
            print(f"  Real vs {r['control']}: ratio={r['ratio_real_over_control']:.2f} "
                  f"(Real={r['real_mean']:.6f}, Control={r['control_mean']:.6f})")
    
    # Visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_comprehensive_comparison(results_dict, output_dir)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    save_comprehensive_results(results_dict, comparisons, interpretation, output_dir)
    
    print(f"\n{'='*70}")
    print("âœ“ COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()