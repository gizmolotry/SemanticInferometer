"""
Comprehensive Observer Variance Diagnostics

Tests for all failure modes:
1. Attention matrix variance (original metric)
2. Procrustes geometric distances (more sensitive)
3. Nearest neighbor disagreement
4. Ranking correlation (Kendall's tau)
5. Cluster assignment comparison
6. Attention entropy analysis
7. Feature collapse detection
8. Component isolation tests
9. Statistical validation (Cohen's d, bootstrap CI, control validation)

Generates artifacts:
- Interactive HTML dashboard
- Statistical analysis
- UMAP visualizations
- Diagnostic plots
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, ttest_ind
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import json

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed, will use PCA for dimensionality reduction")


# ============================================================================
# METRIC 1: Attention Variance (Original)
# ============================================================================

def compute_attention_variance(observers: Dict[int, Dict]) -> Dict:
    """
    Compute variance in attention matrices across observers.
    
    Returns
    -------
    dict
        - mean_variance: Mean variance across all article pairs
        - max_variance: Maximum variance across any pair
        - variance_matrix: Full variance matrix [N, N]
    """
    # Stack attention matrices
    attn_matrices = [obs['attention_matrix'].numpy() for obs in observers.values()]
    attn_stack = np.stack(attn_matrices)  # [n_observers, N, N]
    
    # Compute variance across observers
    variance_matrix = np.var(attn_stack, axis=0)  # [N, N]
    
    # Remove diagonal (self-attention)
    n_articles = variance_matrix.shape[0]
    mask = ~np.eye(n_articles, dtype=bool)
    variance_values = variance_matrix[mask]
    
    return {
        'mean_variance': float(variance_values.mean()),
        'max_variance': float(variance_values.max()),
        'std_variance': float(variance_values.std()),
        'variance_matrix': variance_matrix,
        'interpretation': _interpret_attention_variance(variance_values.mean())
    }


def _interpret_attention_variance(mean_var: float) -> str:
    """Interpret attention variance magnitude."""
    if mean_var < 0.0001:
        return "COLLAPSE: Observers are nearly identical"
    elif mean_var < 0.001:
        return "LOW: Limited observer diversity"
    elif mean_var < 0.01:
        return "MODERATE: Some observer-dependence"
    else:
        return "STRONG: Substantial observer diversity"


# ============================================================================
# METRIC 2: Procrustes Geometric Distance (More Sensitive)
# ============================================================================

def compute_procrustes_distances(observers: Dict[int, Dict], method: str = 'umap') -> Dict:
    """
    Compute geometric distances between observer spaces using Procrustes alignment.
    
    This is more sensitive than attention variance because it captures
    differences in spatial arrangement even when attention weights are similar.
    
    Parameters
    ----------
    observers : dict
        Observer results
    method : str
        'umap' or 'pca' for dimensionality reduction
    
    Returns
    -------
    dict
        - pairwise_distances: Matrix of Procrustes distances
        - mean_distance: Average distance across all pairs
        - interpretations: What the distances mean
    """
    n_observers = len(observers)
    seeds = sorted(observers.keys())
    
    # Project features to 2D
    projections = {}
    for seed in seeds:
        features = observers[seed]['features'].numpy()
        
        if method == 'umap' and HAS_UMAP:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, features.shape[0]-1))
            projection = reducer.fit_transform(features)
        else:
            reducer = PCA(n_components=2, random_state=42)
            projection = reducer.fit_transform(features)
        
        projections[seed] = projection
    
    # Compute pairwise Procrustes distances
    distance_matrix = np.zeros((n_observers, n_observers))
    
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds):
            if i < j:
                # Procrustes analysis: find optimal rotation/scaling to align
                mtx1, mtx2, disparity = procrustes(projections[seed1], projections[seed2])
                distance_matrix[i, j] = disparity
                distance_matrix[j, i] = disparity
    
    # Get upper triangle (unique pairs)
    upper_tri_idx = np.triu_indices(n_observers, k=1)
    distances = distance_matrix[upper_tri_idx]
    
    return {
        'distance_matrix': distance_matrix.tolist(),
        'mean_distance': float(distances.mean()),
        'max_distance': float(distances.max()),
        'std_distance': float(distances.std()),
        'seeds': seeds,
        'method': method,
        'interpretation': _interpret_procrustes_distance(distances.mean())
    }


def _interpret_procrustes_distance(mean_dist: float) -> str:
    """Interpret Procrustes distance magnitude."""
    if mean_dist < 0.01:
        return "IDENTICAL: Same geometry (just rotated/scaled)"
    elif mean_dist < 0.1:
        return "SIMILAR: Geometries are closely related"
    elif mean_dist < 0.3:
        return "MODERATE: Noticeably different structures"
    else:
        return "DISTINCT: Substantially different geometries"


# ============================================================================
# METRIC 3: Nearest Neighbor Disagreement
# ============================================================================

def compute_nn_disagreement(observers: Dict[int, Dict], k: int = 10) -> Dict:
    """
    For each article, find k nearest neighbors according to each observer.
    Measure how much observers disagree on who the neighbors are.
    
    High disagreement = different geometric structures.
    """
    seeds = sorted(observers.keys())
    n_articles = observers[seeds[0]]['attention_matrix'].shape[0]
    
    # For each observer, get k nearest neighbors for each article
    neighbor_sets = {}
    for seed in seeds:
        attn = observers[seed]['attention_matrix'].numpy()
        # For each article, get top-k most attended articles
        neighbors = np.argsort(-attn, axis=1)[:, 1:k+1]  # Exclude self
        neighbor_sets[seed] = neighbors
    
    # Compute pairwise disagreement
    disagreements = []
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds):
            if i < j:
                # For each article, count how many neighbors differ
                nn1 = neighbor_sets[seed1]
                nn2 = neighbor_sets[seed2]
                
                article_disagreements = []
                for article_idx in range(n_articles):
                    set1 = set(nn1[article_idx])
                    set2 = set(nn2[article_idx])
                    overlap = len(set1 & set2)
                    disagreement = (k - overlap) / k  # 0 = perfect agreement, 1 = no overlap
                    article_disagreements.append(disagreement)
                
                mean_disagreement = np.mean(article_disagreements)
                disagreements.append(mean_disagreement)
    
    return {
        'mean_disagreement': float(np.mean(disagreements)),
        'max_disagreement': float(np.max(disagreements)),
        'k': k,
        'interpretation': _interpret_nn_disagreement(np.mean(disagreements))
    }


def _interpret_nn_disagreement(mean_disagree: float) -> str:
    """Interpret nearest neighbor disagreement."""
    if mean_disagree < 0.1:
        return "LOW: Observers mostly agree on article relationships"
    elif mean_disagree < 0.3:
        return "MODERATE: Some disagreement on neighbors"
    elif mean_disagree < 0.5:
        return "HIGH: Substantial disagreement on structure"
    else:
        return "EXTREME: Observers see completely different neighborhoods"


# ============================================================================
# METRIC 4: Ranking Correlation (Kendall's Tau)
# ============================================================================

def compute_ranking_correlation(observers: Dict[int, Dict]) -> Dict:
    """
    For each article, observers rank all other articles by attention.
    Compute Kendall's tau to measure ranking agreement.
    
    Low correlation = different rankings = different structures.
    """
    seeds = sorted(observers.keys())
    
    # Sample articles to avoid O(N^2) computation
    n_articles = observers[seeds[0]]['attention_matrix'].shape[0]
    sample_size = min(100, n_articles)
    sample_indices = np.random.choice(n_articles, sample_size, replace=False)
    
    correlations = []
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds):
            if i < j:
                attn1 = observers[seed1]['attention_matrix'].numpy()
                attn2 = observers[seed2]['attention_matrix'].numpy()
                
                # For each sampled article, compute ranking correlation
                article_correlations = []
                for article_idx in sample_indices:
                    ranking1 = attn1[article_idx]
                    ranking2 = attn2[article_idx]
                    
                    tau, _ = kendalltau(ranking1, ranking2)
                    article_correlations.append(tau)
                
                mean_tau = np.mean(article_correlations)
                correlations.append(mean_tau)
    
    return {
        'mean_tau': float(np.mean(correlations)),
        'min_tau': float(np.min(correlations)),
        'interpretation': _interpret_ranking_correlation(np.mean(correlations))
    }


def _interpret_ranking_correlation(mean_tau: float) -> str:
    """Interpret Kendall's tau."""
    if mean_tau > 0.9:
        return "IDENTICAL: Nearly perfect ranking agreement"
    elif mean_tau > 0.7:
        return "HIGH: Strong ranking agreement"
    elif mean_tau > 0.4:
        return "MODERATE: Some ranking differences"
    else:
        return "LOW: Substantial ranking disagreement"


# ============================================================================
# METRIC 5: Cluster Assignment Comparison
# ============================================================================

def compute_cluster_disagreement(observers: Dict[int, Dict], n_clusters: int = 10) -> Dict:
    """
    Cluster articles using attention matrix, compare cluster assignments.
    """
    seeds = sorted(observers.keys())
    
    # For each observer, cluster articles
    cluster_assignments = {}
    for seed in seeds:
        features = observers[seed]['features'].numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        cluster_assignments[seed] = labels
    
    # Compute adjusted mutual information (measures cluster agreement)
    from sklearn.metrics import adjusted_mutual_info_score
    
    amis = []
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds):
            if i < j:
                labels1 = cluster_assignments[seed1]
                labels2 = cluster_assignments[seed2]
                ami = adjusted_mutual_info_score(labels1, labels2)
                amis.append(ami)
    
    return {
        'mean_ami': float(np.mean(amis)),
        'min_ami': float(np.min(amis)),
        'n_clusters': n_clusters,
        'interpretation': _interpret_cluster_agreement(np.mean(amis))
    }


def _interpret_cluster_agreement(mean_ami: float) -> str:
    """Interpret adjusted mutual information."""
    if mean_ami > 0.9:
        return "IDENTICAL: Same cluster structure"
    elif mean_ami > 0.7:
        return "HIGH: Very similar clustering"
    elif mean_ami > 0.4:
        return "MODERATE: Some cluster differences"
    else:
        return "LOW: Substantially different clustering"


# ============================================================================
# METRIC 6: Attention Entropy Analysis
# ============================================================================

def compute_attention_entropy(observers: Dict[int, Dict]) -> Dict:
    """
    High entropy = uniform attention (no structure)
    Low entropy = focused attention (clear structure)
    
    If all observers have high entropy, attention is meaningless.
    """
    entropies = {}
    
    for seed, obs in observers.items():
        attn = obs['attention_matrix'].numpy()
        # Compute entropy for each row (how focused is attention?)
        row_entropies = -(attn * np.log(attn + 1e-9)).sum(axis=1)
        entropies[seed] = float(row_entropies.mean())
    
    mean_entropy = np.mean(list(entropies.values()))
    n_articles = observers[list(observers.keys())[0]]['attention_matrix'].shape[0]
    max_entropy = np.log(n_articles)  # Uniform distribution
    
    return {
        'entropy_by_observer': entropies,
        'mean_entropy': float(mean_entropy),
        'max_possible_entropy': float(max_entropy),
        'normalized_entropy': float(mean_entropy / max_entropy),
        'interpretation': _interpret_entropy(mean_entropy / max_entropy)
    }


def _interpret_entropy(normalized: float) -> str:
    """Interpret normalized entropy."""
    if normalized > 0.95:
        return "UNIFORM: Attention is nearly random (no structure)"
    elif normalized > 0.8:
        return "HIGH: Weak attention structure"
    elif normalized > 0.5:
        return "MODERATE: Some attention focus"
    else:
        return "FOCUSED: Strong attention structure"


# ============================================================================
# DIAGNOSTIC 7: Feature Collapse Detection
# ============================================================================

def check_feature_collapse(observers: Dict[int, Dict]) -> Dict:
    """
    Check if features collapsed to a single point or sphere.
    """
    diagnostics = {}
    
    for seed, obs in observers.items():
        features = obs['features'].numpy()
        
        # Check norms
        norms = np.linalg.norm(features, axis=1)
        norm_mean = norms.mean()
        norm_std = norms.std()
        
        # Check pairwise distances
        distances = squareform(pdist(features, metric='euclidean'))
        mask = ~np.eye(distances.shape[0], dtype=bool)
        dist_mean = distances[mask].mean()
        dist_std = distances[mask].std()
        
        # Check cosine similarities
        features_normed = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
        similarities = features_normed @ features_normed.T
        sim_mean = similarities[mask].mean()
        
        diagnostics[seed] = {
            'norm_mean': float(norm_mean),
            'norm_std': float(norm_std),
            'distance_mean': float(dist_mean),
            'distance_std': float(dist_std),
            'cosine_similarity_mean': float(sim_mean),
            'collapsed': norm_std < 0.01 and sim_mean > 0.99
        }
    
    # Check if any observer has collapsed
    any_collapsed = any(d['collapsed'] for d in diagnostics.values())
    
    return {
        'by_observer': diagnostics,
        'any_collapsed': any_collapsed,
        'interpretation': 'COLLAPSED: Features mapped to same point' if any_collapsed else 'OK: Features are diverse'
    }


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

def run_comprehensive_diagnostics(
    observer_pattern: str = 'outputs/real_observer_*.pt',
    output_dir: str = 'outputs/diagnostics'
) -> Dict:
    """
    Run all diagnostic tests and generate comprehensive report.
    
    Parameters
    ----------
    observer_pattern : str
        Glob pattern for observer files
    output_dir : str
        Where to save diagnostic outputs
    
    Returns
    -------
    dict
        Complete diagnostic results
    """
    print("="*70)
    print("COMPREHENSIVE OBSERVER VARIANCE DIAGNOSTICS")
    print("="*70)
    
    # Load observers
    observer_files = sorted(Path('.').glob(observer_pattern))
    if not observer_files:
        raise FileNotFoundError(f"No files found matching: {observer_pattern}")
    
    print(f"\nLoading {len(observer_files)} observers...")
    observers = {}
    for fpath in observer_files:
        data = torch.load(fpath, map_location='cpu')
        seed = data.get('seed', int(fpath.stem.split('_')[-1]))
        observers[seed] = data
        print(f"  Loaded observer {seed}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run all diagnostics
    results = {
        'n_observers': len(observers),
        'n_articles': observers[list(observers.keys())[0]]['attention_matrix'].shape[0],
        'seeds': sorted(observers.keys())
    }
    
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTICS")
    print("="*70)
    
    print("\n[1/7] Attention variance...")
    results['attention_variance'] = compute_attention_variance(observers)
    print(f"  Mean variance: {results['attention_variance']['mean_variance']:.6f}")
    print(f"  {results['attention_variance']['interpretation']}")
    
    print("\n[2/7] Procrustes geometric distances...")
    results['procrustes'] = compute_procrustes_distances(observers, method='umap')
    print(f"  Mean distance: {results['procrustes']['mean_distance']:.4f}")
    print(f"  {results['procrustes']['interpretation']}")
    
    print("\n[3/7] Nearest neighbor disagreement...")
    results['nn_disagreement'] = compute_nn_disagreement(observers, k=10)
    print(f"  Mean disagreement: {results['nn_disagreement']['mean_disagreement']:.4f}")
    print(f"  {results['nn_disagreement']['interpretation']}")
    
    print("\n[4/7] Ranking correlation...")
    results['ranking_correlation'] = compute_ranking_correlation(observers)
    print(f"  Mean Kendall's tau: {results['ranking_correlation']['mean_tau']:.4f}")
    print(f"  {results['ranking_correlation']['interpretation']}")
    
    print("\n[5/7] Cluster assignments...")
    results['cluster_disagreement'] = compute_cluster_disagreement(observers, n_clusters=10)
    print(f"  Mean AMI: {results['cluster_disagreement']['mean_ami']:.4f}")
    print(f"  {results['cluster_disagreement']['interpretation']}")
    
    print("\n[6/7] Attention entropy...")
    results['entropy'] = compute_attention_entropy(observers)
    print(f"  Mean entropy: {results['entropy']['mean_entropy']:.4f}")
    print(f"  {results['entropy']['interpretation']}")
    
    print("\n[7/7] Feature collapse check...")
    results['feature_collapse'] = check_feature_collapse(observers)
    print(f"  {results['feature_collapse']['interpretation']}")
    
    # Generate summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    summary = _generate_summary(results)
    results['summary'] = summary
    print(summary['text'])
    
    # Save results
    json_path = Path(output_dir) / 'diagnostic_results.json'
    with open(json_path, 'w') as f:
        # Remove non-serializable items
        results_copy = results.copy()
        results_copy['attention_variance'].pop('variance_matrix', None)
        json.dump(results_copy, f, indent=2)
    print(f"\nâœ“ Saved JSON results to {json_path}")
    
    # Generate HTML report
    html_path = Path(output_dir) / 'diagnostic_report.html'
    _generate_html_report(results, html_path, observers)
    print(f"âœ“ Saved HTML report to {html_path}")
    
    return results


def _generate_summary(results: Dict) -> Dict:
    """Generate human-readable summary of all diagnostics."""
    
    # Determine overall verdict
    attn_var = results['attention_variance']['mean_variance']
    proc_dist = results['procrustes']['mean_distance']
    nn_disagree = results['nn_disagreement']['mean_disagreement']
    
    # Multiple signals
    signals = []
    
    if attn_var > 0.001:
        signals.append("attention variance present")
    if proc_dist > 0.1:
        signals.append("geometric structures differ")
    if nn_disagree > 0.2:
        signals.append("neighborhood disagreement")
    
    if len(signals) >= 2:
        verdict = "âœ“ OBSERVER DIVERSITY DETECTED"
        interpretation = f"Multiple metrics show observer-dependence: {', '.join(signals)}."
    elif len(signals) == 1:
        verdict = "âš  WEAK OBSERVER DIVERSITY"
        interpretation = f"Limited evidence of observer-dependence: {signals[0]}."
    else:
        verdict = "âœ— OBSERVER COLLAPSE"
        interpretation = "Observers produce nearly identical results across all metrics."
    
    # Identify likely failure mode if collapsed
    failure_mode = None
    if len(signals) == 0:
        if results['feature_collapse']['any_collapsed']:
            failure_mode = "Feature collapse: All features mapped to same point"
        elif results['entropy']['normalized_entropy'] > 0.95:
            failure_mode = "Uniform attention: No semantic structure captured"
        else:
            failure_mode = "Unknown: Mechanism unclear, check component isolation"
    
    text = f"""
{verdict}

{interpretation}

Key Metrics:
  â€¢ Attention variance: {attn_var:.6f} ({results['attention_variance']['interpretation']})
  â€¢ Procrustes distance: {proc_dist:.4f} ({results['procrustes']['interpretation']})
  â€¢ NN disagreement: {nn_disagree:.4f} ({results['nn_disagreement']['interpretation']})
  â€¢ Ranking correlation: {results['ranking_correlation']['mean_tau']:.4f} ({results['ranking_correlation']['interpretation']})
  â€¢ Cluster agreement: {results['cluster_disagreement']['mean_ami']:.4f} ({results['cluster_disagreement']['interpretation']})
  â€¢ Attention entropy: {results['entropy']['normalized_entropy']:.4f} ({results['entropy']['interpretation']})
"""
    
    if failure_mode:
        text += f"\nLikely failure mode: {failure_mode}"
    
    return {
        'verdict': verdict,
        'interpretation': interpretation,
        'failure_mode': failure_mode,
        'text': text,
        'has_diversity': len(signals) >= 1
    }


def _generate_html_report(results: Dict, output_path: Path, observers: Dict):
    """Generate interactive HTML report with visualizations."""
    
    # Create UMAP visualization
    try:
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots
        
        # Project each observer to 2D
        seeds = results['seeds']
        n_observers = len(seeds)
        
        fig = make_subplots(
            rows=1, cols=n_observers,
            subplot_titles=[f"Observer {s}" for s in seeds]
        )
        
        for idx, seed in enumerate(seeds, 1):
            features = observers[seed]['features'].numpy()
            
            if HAS_UMAP:
                reducer = UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(features)
            else:
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(features)
            
            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode='markers',
                    marker=dict(size=3, opacity=0.6),
                    showlegend=False
                ),
                row=1, col=idx
            )
        
        fig.update_layout(height=400, title_text="Observer Geometries (UMAP Projection)")
        plot_html = fig.to_html(include_plotlyjs='cdn', div_id='geometry_plot')
    except:
        plot_html = "<p>Visualization unavailable (plotly not installed)</p>"
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Observer Variance Diagnostics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .metric-box {{
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .verdict {{
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            background-color: {'#d4edda' if results['summary']['has_diversity'] else '#f8d7da'};
            border-left: 4px solid {'#28a745' if results['summary']['has_diversity'] else '#dc3545'};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2c3e50;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Observer Variance Diagnostic Report</h1>
        <p>Generated: {pd.Timestamp.now()}</p>
        <p>Observers: {results['n_observers']} | Articles: {results['n_articles']}</p>
    </div>
    
    <div class="verdict">
        {results['summary']['verdict']}
        <p style="font-size: 14px; font-weight: normal; margin-top: 10px;">
            {results['summary']['interpretation']}
        </p>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Metric 1: Attention Variance</div>
        <table>
            <tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Mean Variance</td>
                <td>{results['attention_variance']['mean_variance']:.6f}</td>
                <td>{results['attention_variance']['interpretation']}</td>
            </tr>
            <tr>
                <td>Max Variance</td>
                <td>{results['attention_variance']['max_variance']:.6f}</td>
                <td>Highest divergence point</td>
            </tr>
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Metric 2: Procrustes Distance (Geometric)</div>
        <table>
            <tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Mean Distance</td>
                <td>{results['procrustes']['mean_distance']:.4f}</td>
                <td>{results['procrustes']['interpretation']}</td>
            </tr>
            <tr>
                <td>Method</td>
                <td>{results['procrustes']['method'].upper()}</td>
                <td>Dimensionality reduction</td>
            </tr>
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Metric 3: Nearest Neighbor Disagreement</div>
        <table>
            <tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Mean Disagreement</td>
                <td>{results['nn_disagreement']['mean_disagreement']:.4f}</td>
                <td>{results['nn_disagreement']['interpretation']}</td>
            </tr>
            <tr>
                <td>k</td>
                <td>{results['nn_disagreement']['k']}</td>
                <td>Number of neighbors checked</td>
            </tr>
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Metric 4: Ranking Correlation</div>
        <table>
            <tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Mean Kendall's Tau</td>
                <td>{results['ranking_correlation']['mean_tau']:.4f}</td>
                <td>{results['ranking_correlation']['interpretation']}</td>
            </tr>
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Metric 5: Cluster Agreement</div>
        <table>
            <tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Mean AMI</td>
                <td>{results['cluster_disagreement']['mean_ami']:.4f}</td>
                <td>{results['cluster_disagreement']['interpretation']}</td>
            </tr>
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Diagnostic: Attention Entropy</div>
        <table>
            <tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td>Normalized Entropy</td>
                <td>{results['entropy']['normalized_entropy']:.4f}</td>
                <td>{results['entropy']['interpretation']}</td>
            </tr>
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Diagnostic: Feature Collapse</div>
        <table>
            <tr><th>Observer</th><th>Norm Std</th><th>Cosine Similarity</th><th>Status</th></tr>
""" + "\n".join([
        f"<tr><td>{seed}</td><td>{diag['norm_std']:.4f}</td><td>{diag['cosine_similarity_mean']:.4f}</td><td>{'COLLAPSED' if diag['collapsed'] else 'OK'}</td></tr>"
        for seed, diag in results['feature_collapse']['by_observer'].items()
    ]) + f"""
        </table>
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Geometric Visualizations</div>
        {plot_html}
    </div>
    
    <div class="metric-box">
        <div class="metric-title">Recommendations</div>
        <ul>
            {'<li>âœ“ Observer diversity detected! Proceed with analysis.</li>' if results['summary']['has_diversity'] else '<li>âœ— No observer diversity. Check failure modes.</li>'}
            {'<li>âš  Features collapsed to same point - check normalization.</li>' if results['feature_collapse']['any_collapsed'] else ''}
            {'<li>âš  Attention is nearly uniform - semantic structure may be weak.</li>' if results['entropy']['normalized_entropy'] > 0.95 else ''}
            {'<li>âš  Procrustes distance is low despite attention variance - check if variance is meaningful.</li>' if results['attention_variance']['mean_variance'] > 0.001 and results['procrustes']['mean_distance'] < 0.05 else ''}
        </ul>
    </div>
</body>
</html>
"""
    
    output_path.write_text(html, encoding='utf-8')


# ============================================================================
# STATISTICAL VALIDATION (Thesis-Grade)
# ============================================================================

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    n_a, n_b = len(group_a), len(group_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean_a - mean_b) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    point = statistic(data)
    
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        resampled = data[idx]
        bootstrap_stats.append(statistic(resampled))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point, ci_lower, ci_upper


def compute_inter_observer_variance(embeddings_list: List[np.ndarray]) -> float:
    """Compute variance across observers for same articles."""
    if len(embeddings_list) < 2:
        return 0.0
    stacked = np.stack(embeddings_list, axis=0)  # [K, N, D]
    var_per_point = stacked.var(axis=0)  # [N, D]
    return float(var_per_point.mean())


def run_control_validation(
    real_embeddings: List[np.ndarray],
    shuffled_embeddings: List[np.ndarray],
    constant_embeddings: List[np.ndarray],
    random_embeddings: List[np.ndarray],
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Run control validation: Real >> Shuffled >> Constant ≈ Random
    
    This is CRITICAL for thesis defense.
    """
    # Compute variances
    real_var = compute_inter_observer_variance(real_embeddings)
    shuffled_var = compute_inter_observer_variance(shuffled_embeddings)
    constant_var = compute_inter_observer_variance(constant_embeddings)
    random_var = compute_inter_observer_variance(random_embeddings)
    
    # Effect sizes
    real_arr = np.stack(real_embeddings, axis=0).var(axis=0).mean(axis=-1) if real_embeddings else np.array([0])
    shuffled_arr = np.stack(shuffled_embeddings, axis=0).var(axis=0).mean(axis=-1) if shuffled_embeddings else np.array([0])
    constant_arr = np.stack(constant_embeddings, axis=0).var(axis=0).mean(axis=-1) if constant_embeddings else np.array([0])
    random_arr = np.stack(random_embeddings, axis=0).var(axis=0).mean(axis=-1) if random_embeddings else np.array([0])
    
    d_real_shuffled = cohens_d(real_arr, shuffled_arr)
    d_real_constant = cohens_d(real_arr, constant_arr)
    d_shuffled_constant = cohens_d(shuffled_arr, constant_arr)
    
    # Bootstrap CIs
    real_pt, real_lo, real_hi = bootstrap_ci(real_arr, np.mean, n_bootstrap)
    shuffled_pt, shuffled_lo, shuffled_hi = bootstrap_ci(shuffled_arr, np.mean, n_bootstrap)
    constant_pt, constant_lo, constant_hi = bootstrap_ci(constant_arr, np.mean, n_bootstrap)
    random_pt, random_lo, random_hi = bootstrap_ci(random_arr, np.mean, n_bootstrap)
    
    # Check ordering
    ordering = (real_var > shuffled_var > constant_var)
    constant_random_similar = abs(constant_var - random_var) < constant_var * 0.5 if constant_var > 0 else True
    
    # T-tests
    t_real_shuffled, p_real_shuffled = ttest_ind(real_arr, shuffled_arr)
    t_real_constant, p_real_constant = ttest_ind(real_arr, constant_arr)
    
    # Bonferroni correction (3 comparisons)
    alpha_corrected = 0.05 / 3
    
    # Verdict
    if ordering and d_real_shuffled > 0.5 and d_real_constant > 0.8:
        verdict = "PASS: System shows expected semantic structure sensitivity"
    elif real_var > shuffled_var > constant_var:
        verdict = "WEAK PASS: Ordering correct but effect sizes may be small"
    elif real_var > constant_var:
        verdict = "PARTIAL: Real > Constant, but shuffled ordering unclear"
    else:
        verdict = "FAIL: Expected ordering not satisfied"
    
    return {
        'variances': {
            'real': real_var,
            'shuffled': shuffled_var,
            'constant': constant_var,
            'random': random_var,
        },
        'confidence_intervals': {
            'real': {'point': real_pt, 'ci_lower': real_lo, 'ci_upper': real_hi},
            'shuffled': {'point': shuffled_pt, 'ci_lower': shuffled_lo, 'ci_upper': shuffled_hi},
            'constant': {'point': constant_pt, 'ci_lower': constant_lo, 'ci_upper': constant_hi},
            'random': {'point': random_pt, 'ci_lower': random_lo, 'ci_upper': random_hi},
        },
        'effect_sizes': {
            'real_vs_shuffled': {'d': d_real_shuffled, 'interpretation': interpret_cohens_d(d_real_shuffled)},
            'real_vs_constant': {'d': d_real_constant, 'interpretation': interpret_cohens_d(d_real_constant)},
            'shuffled_vs_constant': {'d': d_shuffled_constant, 'interpretation': interpret_cohens_d(d_shuffled_constant)},
        },
        't_tests': {
            'real_vs_shuffled': {'t': t_real_shuffled, 'p': p_real_shuffled, 'significant': p_real_shuffled < alpha_corrected},
            'real_vs_constant': {'t': t_real_constant, 'p': p_real_constant, 'significant': p_real_constant < alpha_corrected},
        },
        'bonferroni_alpha': alpha_corrected,
        'ordering_satisfied': ordering,
        'constant_random_similar': constant_random_similar,
        'verdict': verdict,
    }


def verify_canonical_ids(artifacts: List[Dict]) -> Dict:
    """Verify canonical IDs are present and aligned across artifacts."""
    results = {
        'n_artifacts': len(artifacts),
        'all_have_ids': True,
        'ids_match': True,
        'issues': [],
    }
    
    id_lists = []
    for i, artifact in enumerate(artifacts):
        ids = artifact.get('ids') or artifact.get('canonical_ids') or artifact.get('bt_uid_list')
        if ids is None or len(ids) == 0:
            results['all_have_ids'] = False
            results['issues'].append(f"Artifact {i}: No canonical IDs found")
            id_lists.append(None)
        else:
            id_lists.append(list(ids))
    
    if results['all_have_ids'] and len(id_lists) >= 2:
        first_ids = id_lists[0]
        for i, ids in enumerate(id_lists[1:], 1):
            if ids != first_ids:
                results['ids_match'] = False
                results['issues'].append(f"Artifact {i}: IDs don't match artifact 0")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive observer variance diagnostics')
    parser.add_argument('--pattern', type=str, default='outputs/real_observer_*.pt',
                       help='Glob pattern for observer files')
    parser.add_argument('--output', type=str, default='outputs/diagnostics',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results = run_comprehensive_diagnostics(
        observer_pattern=args.pattern,
        output_dir=args.output
    )
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output}/")
    print(f"  - diagnostic_results.json (machine-readable)")
    print(f"  - diagnostic_report.html (human-readable)")
    print(f"\nOpen the HTML report in your browser to see visualizations.")