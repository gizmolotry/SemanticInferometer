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
                print(f"  ⚠ Could not parse seed from {fpath.name}")
                continue
            
            # Filter seeds if requested
            if filter_seeds is not None and seed not in filter_seeds:
                continue
            
            # Load data
            data = torch.load(fpath, map_location='cpu')
            
            # Extract features
            if isinstance(data, dict):
                features = data.get('features', data.get('embeddings', data.get('article_tokens')))
            else:
                features = data
            
            if features is None:
                print(f"  ✗ No features in {fpath.name}")
                continue
            
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            
            observers[seed] = {
                'features': features,
                'file': fpath.name,
                'shape': features.shape
            }
            
        except Exception as e:
            print(f"  ✗ Failed to load {fpath.name}: {e}")
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
    
    print(f"\n  ⚠ Mixed shapes detected: {unique_shapes}")
    
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
    Returns: 1 - ρ (so lower = more similar)
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
    print(f"    {n_obs} observers × {n_items} items × {n_dims} dims")
    
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
    
    # Simple variance
    results['simple_variance'] = {
        'mean': float(np.mean(simple_diffs)),
        'std': float(np.std(simple_diffs)),
        'median': float(np.median(simple_diffs)),
        'mad': median_absolute_deviation(simple_diffs),
        'values': [float(v) for v in simple_diffs]
    }
    
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
            print(f"  ✓ Loaded {len(real_obs)} observers")
            print(f"    Shape: {list(real_obs.values())[0]['shape']}")
    else:
        print(f"  ✗ No real corpus files found")
    
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
            print(f"  ✓ Loaded {len(const_obs)} observers")
            print(f"    Shape: {list(const_obs.values())[0]['shape']}")
    else:
        print(f"  ✗ No constant control files found")
    
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
            print(f"  ✓ Loaded {len(shuf_obs)} observers")
            print(f"    Shape: {list(shuf_obs.values())[0]['shape']}")
    else:
        print(f"  ✗ No shuffled control files found")
    
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
            print(f"  ✓ Loaded {len(rand_obs)} observers")
            print(f"    Shape: {list(rand_obs.values())[0]['shape']}")
    else:
        print(f"  ✗ No random control files found")
    
    return experiments


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
        print("  ⚠ No metrics to plot")
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
    print(f"  ✓ Saved plot: {plot_path}")
    
    plt.close()


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
    print(f"  ✓ Saved JSON: {json_path}")
    
    # CSV tables for each metric
    for metric_name, df in comparisons.items():
        csv_path = output_dir / f'{metric_name}_comparisons.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved CSV: {csv_path}")
    
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
    print(f"  ✓ Saved summary: {summary_path}")


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
    
    parser.add_argument(
        '--no-residual',
        action='store_true',
        help='Skip consensus/residual decomposition'
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
    
    print(f"{'='*70}")
    print("UNIFIED CONTROL ANALYSIS")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Seeds filter: {args.seeds if args.seeds else 'All'}")
    print(f"Shape reconciliation: {args.reconcile}")
    print(f"Advanced metrics: {not args.no_advanced}")
    print(f"Residual analysis: {not args.no_residual}")
    
    # Load experiments
    experiments = load_all_experiments(data_dir, args.seeds, args.reconcile)
    
    if not experiments:
        print("\n✗ No experiments loaded!")
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
            include_advanced=not args.no_advanced,
            include_residual=not args.no_residual,
            k_components=args.k_components
        )
        
        if 'error' in results:
            print(f"  ✗ {results['error']}")
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
        if all(metric_name in res for res in results_dict.values()):
            df = compare_metrics(results_dict, metric_name)
            comparisons[metric_name] = df
            
            print(f"\n{metric_name.upper()}:")
            for _, row in df.iterrows():
                sig = "✓" if row['significant'] else "✗"
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
                print(f"  ✓ {metric_name}: {metric_data['ratio']:.2f}x")
            else:
                print(f"  ✗ {metric_name}: {metric_data['ratio']:.2f}x (weak)")
    
    if 'consensus_residual' in interpretation:
        print(f"\nConsensus/Residual:")
        print(f"  {interpretation['consensus_residual']['thesis_claim']}")
    
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
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()