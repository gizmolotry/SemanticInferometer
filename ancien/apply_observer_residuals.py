"""
Apply Observer Residual Analysis - ROBUST VERSION

Computes observer-specific vs consensus variance for thesis validation.

This measures the 86% metric from your original successful run.

Usage:
    # Auto-detect all observers in outputs/
    python apply_observer_residuals.py
    
    # Specific pattern
    python apply_observer_residuals.py --pattern "control_random_*.pt"
    
    # Specific mode
    python apply_observer_residuals.py --mode enhanced_mode
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import json


def load_observers(pattern=None, data_dir='outputs', mode=None):
    """
    Load all observer files matching pattern.
    
    Returns dict: {seed: observer_data}
    """
    data_dir = Path(data_dir)
    
    # Build pattern
    if pattern:
        search_pattern = pattern
    elif mode:
        search_pattern = f"*{mode}_observer_*.pt"
    else:
        search_pattern = "*_observer_*.pt"
    
    print(f"\n{'='*70}")
    print("LOADING OBSERVERS")
    print(f"{'='*70}")
    print(f"Search pattern: {search_pattern}")
    print(f"Data directory: {data_dir.absolute()}")
    
    observer_files = list(data_dir.glob(search_pattern))
    
    if not observer_files:
        print(f"\n✗ No files found matching: {search_pattern}")
        print(f"  in directory: {data_dir}")
        return None
    
    print(f"Found {len(observer_files)} observer files")
    
    observers = {}
    for fpath in sorted(observer_files):
        try:
            # Extract seed from filename
            seed = int(fpath.stem.split('_')[-1])
            
            # Load data
            obs = torch.load(fpath, map_location='cpu')
            
            # Validate
            if 'features' not in obs:
                print(f"  ⚠ {fpath.name}: Missing 'features' - skipping")
                continue
            
            # Check for NaN/Inf
            if torch.isnan(obs['features']).any():
                print(f"  ⚠ {fpath.name}: Contains NaN - skipping")
                continue
            
            if torch.isinf(obs['features']).any():
                print(f"  ⚠ {fpath.name}: Contains Inf - skipping")
                continue
            
            observers[seed] = obs
            print(f"  ✓ Observer {seed}: {obs['features'].shape}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {fpath.name}: {e}")
            continue
    
    if len(observers) < 2:
        print(f"\n✗ Need at least 2 observers, found {len(observers)}")
        return None
    
    print(f"\n✓ Loaded {len(observers)} observers: {sorted(observers.keys())}")
    
    return observers


def compute_residual_variance(observers, k_components=10):
    """
    Compute consensus vs residual variance (the 86% metric).
    
    Method:
    1. Stack all observer embeddings
    2. Fit PCA to find consensus subspace (top k components)
    3. Project each observer onto consensus
    4. Measure variance in residuals vs total
    
    Returns dict with variance breakdown
    """
    seeds = sorted(observers.keys())
    n_obs = len(seeds)
    
    print(f"\n{'='*70}")
    print("COMPUTING RESIDUAL VARIANCE")
    print(f"{'='*70}")
    print(f"Observers: {n_obs}")
    print(f"Consensus components: {k_components}")
    
    # Extract embeddings
    embeddings = []
    seeds_list = []
    for seed in seeds:
        emb = observers[seed]['features'].cpu().numpy()
        embeddings.append(emb)
        seeds_list.append(seed)
        print(f"  Observer {seed}: {emb.shape}, norm={np.linalg.norm(emb, axis=1).mean():.4f}")
    
    # ROBUST FIX: Handle mixed shapes (different article counts)
    shapes = [emb.shape for emb in embeddings]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print(f"\n  ⚠ WARNING: Mixed shapes detected: {unique_shapes}")
        from collections import Counter
        most_common = Counter(shapes).most_common(1)[0][0]
        print(f"    Filtering to most common shape: {most_common}")
        
        filtered_embeddings = []
        filtered_seeds = []
        for seed, emb in zip(seeds_list, embeddings):
            if emb.shape == most_common:
                filtered_embeddings.append(emb)
                filtered_seeds.append(seed)
        
        embeddings = filtered_embeddings
        seeds = filtered_seeds
        n_obs = len(embeddings)
        print(f"    Kept {n_obs} observers with matching shape\n")
    
    embeddings = np.array(embeddings)  # [n_obs, n_articles, n_dims]
    n_obs, n_articles, n_dims = embeddings.shape
    
    print(f"\nStacked shape: {embeddings.shape}")
    
    # Reshape for PCA: [n_obs * n_articles, n_dims]
    all_data = embeddings.reshape(-1, n_dims)
    
    print(f"Reshaped for PCA: {all_data.shape}")
    
    # Fit PCA to find consensus subspace
    print(f"\nFitting PCA (k={k_components})...")
    pca = PCA(n_components=min(k_components, n_dims))
    pca.fit(all_data)
    
    print(f"✓ PCA fitted")
    print(f"  Total variance explained by top {k_components} PCs: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Compute total variance
    total_variance = embeddings.var()
    
    print(f"\nTotal variance: {total_variance:.6f}")
    
    # Project each observer onto consensus subspace
    consensus_parts = []
    residual_parts = []
    
    for i, seed in enumerate(seeds):
        # Project onto consensus
        obs_data = embeddings[i]  # [n_articles, n_dims]
        projected = pca.transform(obs_data)  # [n_articles, k_components]
        reconstructed = pca.inverse_transform(projected)  # [n_articles, n_dims]
        
        consensus_parts.append(reconstructed)
        residual_parts.append(obs_data - reconstructed)
    
    consensus_parts = np.array(consensus_parts)
    residual_parts = np.array(residual_parts)
    
    # Compute variances
    consensus_variance = consensus_parts.var()
    residual_variance = residual_parts.var()
    
    # As fractions
    consensus_fraction = consensus_variance / total_variance
    residual_fraction = residual_variance / total_variance
    
    # Inter-observer agreement (correlation of residuals)
    residual_flat = residual_parts.reshape(n_obs, -1)
    corr_matrix = np.corrcoef(residual_flat)
    
    # Mean off-diagonal correlation
    mask = ~np.eye(n_obs, dtype=bool)
    mean_agreement = corr_matrix[mask].mean()
    
    print(f"\n{'='*70}")
    print("VARIANCE BREAKDOWN")
    print(f"{'='*70}")
    print(f"Total variance:              {total_variance:.6f}")
    print(f"Consensus variance:          {consensus_variance:.6f} ({consensus_fraction:.2%})")
    print(f"Residual variance:           {residual_variance:.6f} ({residual_fraction:.2%})")
    print(f"Inter-observer agreement:    {mean_agreement:.4f}")
    
    # Per-observer residual norms
    print(f"\n{'='*70}")
    print("PER-OBSERVER RESIDUAL NORMS")
    print(f"{'='*70}")
    
    for i, seed in enumerate(seeds):
        total_norm = np.linalg.norm(embeddings[i], axis=1).mean()
        residual_norm = np.linalg.norm(residual_parts[i], axis=1).mean()
        consensus_strength = 1 - (residual_norm / total_norm)
        
        print(f"Observer {seed}:")
        print(f"  Total norm:     {total_norm:.4f}")
        print(f"  Residual norm:  {residual_norm:.4f}")
        print(f"  Consensus:      {consensus_strength:.2%}")
    
    return {
        'total_variance': float(total_variance),
        'consensus_variance': float(consensus_variance),
        'residual_variance': float(residual_variance),
        'consensus_fraction': float(consensus_fraction),
        'residual_fraction': float(residual_fraction),
        'inter_observer_agreement': float(mean_agreement),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'n_observers': n_obs,
        'n_articles': n_articles,
        'n_dims': n_dims,
        'k_components': k_components
    }


def save_results(results, output_path):
    """Save results to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to: {output_path}")


def print_interpretation(results):
    """Print thesis-relevant interpretation"""
    
    residual_pct = results['residual_fraction'] * 100
    consensus_pct = results['consensus_fraction'] * 100
    
    print(f"\n{'='*70}")
    print("INTERPRETATION FOR THESIS")
    print(f"{'='*70}")
    
    print(f"\nVariance Decomposition:")
    print(f"  Consensus (shared structure):     {consensus_pct:.2f}%")
    print(f"  Residual (observer-specific):     {residual_pct:.2f}%")
    
    print(f"\nInter-observer agreement: {results['inter_observer_agreement']:.4f}")
    print(f"  (0 = independent, 1 = identical)")
    
    # Interpretation
    print(f"\n{'='*70}")
    
    if residual_pct > 70:
        print("✓✓✓ HIGH OBSERVER-SPECIFIC VARIANCE")
        print(f"\n{residual_pct:.1f}% of semantic structure is observer-dependent.")
        print("This indicates that different observers produce substantially")
        print("different geometric arrangements of the same content.")
        
        if results['inter_observer_agreement'] < 0.1:
            print("\nLow inter-observer agreement confirms genuine divergence.")
            print("Observers are not merely adding random noise.")
        
    elif residual_pct > 40:
        print("✓ MODERATE OBSERVER-SPECIFIC VARIANCE")
        print(f"\n{residual_pct:.1f}% of variance is observer-specific.")
        print("Some observer-dependence exists, but substantial consensus remains.")
        
    else:
        print("⚠ LOW OBSERVER-SPECIFIC VARIANCE")
        print(f"\nOnly {residual_pct:.1f}% of variance is observer-specific.")
        print("Observers largely agree on semantic structure.")
        print("This suggests:")
        print("  - Strong consensus in representation")
        print("  - OR: Random seeds too similar")
        print("  - OR: Architecture too deterministic")
    
    print(f"{'='*70}")


def compare_experiments(results_dict):
    """Compare multiple experiments (e.g., real vs controls)"""
    
    print(f"\n{'='*70}")
    print("COMPARING EXPERIMENTS")
    print(f"{'='*70}")
    
    print(f"\n{'Experiment':<20} {'Consensus':<12} {'Residual':<12} {'Agreement':<12}")
    print("-" * 70)
    
    for name, results in sorted(results_dict.items()):
        cons = results['consensus_fraction'] * 100
        resid = results['residual_fraction'] * 100
        agree = results['inter_observer_agreement']
        
        print(f"{name:<20} {cons:>10.2f}%  {resid:>10.2f}%  {agree:>10.4f}")
    
    # Compute ratios
    if 'real' in results_dict or 'Real' in results_dict:
        real_key = 'real' if 'real' in results_dict else 'Real'
        real_resid = results_dict[real_key]['residual_fraction']
        
        print(f"\n{'='*70}")
        print("RESIDUAL VARIANCE RATIOS (Real / Control)")
        print(f"{'='*70}")
        
        for name, results in sorted(results_dict.items()):
            if name.lower() == 'real':
                continue
            
            ratio = real_resid / results['residual_fraction'] if results['residual_fraction'] > 0 else float('inf')
            print(f"  Real / {name}: {ratio:.2f}x")
        
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")
        
        # Check if real >> controls
        control_residuals = [r['residual_fraction'] for n, r in results_dict.items() if n.lower() != 'real']
        
        if control_residuals:
            max_control = max(control_residuals)
            ratio = real_resid / max_control
            
            if ratio > 2.0:
                print("✓✓✓ THESIS STRONGLY VALIDATED")
                print(f"\nReal corpus shows {ratio:.1f}x more observer variance than controls.")
                print("This demonstrates observer-dependence emerges from semantic interpretation,")
                print("not from architectural artifacts or random sampling.")
            elif ratio > 1.5:
                print("✓ THESIS MODERATELY VALIDATED")
                print(f"\nReal corpus shows {ratio:.1f}x more observer variance than controls.")
                print("Some semantic observer-dependence exists, though controls also show variance.")
            else:
                print("⚠ THESIS NOT CLEARLY VALIDATED")
                print(f"\nReal corpus shows only {ratio:.1f}x more variance than controls.")
                print("Observer variance may be primarily architectural/sampling artifact.")
                print("Consider:")
                print("  - Using different kernels (multi-kernel mode)")
                print("  - Increasing observer diversity (distant seeds)")
                print("  - Alternative metrics (Procrustes, UMAP divergence)")


def main():
    parser = argparse.ArgumentParser(
        description="Apply observer residual analysis"
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='File pattern to match (e.g., "control_random_*.pt")'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        help='Experiment mode (e.g., "enhanced_mode")'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='outputs',
        help='Directory containing observer files'
    )
    
    parser.add_argument(
        '--k-components',
        type=int,
        default=10,
        help='Number of consensus PCA components'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple experiments (detects real + controls)'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print("OBSERVER RESIDUAL ANALYSIS")
    print(f"{'='*70}")
    
    if args.compare:
        # Auto-detect real + control experiments
        experiments = {}
        
        # Try to find real corpus
        for pattern in ['real_enhanced_mode_*.pt', 'enhanced_mode_observer_*.pt', 'enhanced_observer_*.pt']:
            obs = load_observers(pattern=pattern, data_dir=args.data_dir)
            if obs and len(obs) >= 2:
                experiments['Real'] = obs
                break
        
        # Try to find controls
        for control_type in ['constant', 'shuffled', 'random']:
            pattern = f'control_{control_type}_*_observer_*.pt'
            obs = load_observers(pattern=pattern, data_dir=args.data_dir)
            if obs and len(obs) >= 2:
                experiments[f'Control_{control_type.capitalize()}'] = obs
        
        if len(experiments) == 0:
            print("\n✗ No experiments found!")
            print("  Make sure observer files are in outputs/")
            return
        
        print(f"\nFound {len(experiments)} experiments to compare:")
        for name in experiments.keys():
            print(f"  - {name}")
        
        # Compute residual variance for each
        results_dict = {}
        for name, obs in experiments.items():
            if len(obs) >= 3:  # Need ≥3 for residual analysis
                print(f"\n{'='*70}")
                print(f"ANALYZING: {name}")
                print(f"{'='*70}")
                results_dict[name] = compute_residual_variance(obs, args.k_components)
                print_interpretation(results_dict[name])
            else:
                print(f"\n⚠ {name}: Only {len(obs)} observers (need ≥3 for residual analysis)")
        
        # Compare
        if len(results_dict) > 1:
            compare_experiments(results_dict)
        
        # Save all results
        if args.output:
            save_results(results_dict, args.output)
        else:
            save_results(results_dict, f"{args.data_dir}/residual_analysis_comparison.json")
    
    else:
        # Single experiment
        observers = load_observers(
            pattern=args.pattern,
            mode=args.mode,
            data_dir=args.data_dir
        )
        
        if not observers:
            print("\n✗ No observers found!")
            return
        
        if len(observers) < 3:
            print(f"\n⚠ Only {len(observers)} observers found")
            print("  Residual analysis requires ≥3 observers")
            print("  Run more seeds:")
            print("    python run_experiments.py --mode enhanced --seeds 42 43 44 45 46")
            return
        
        # Compute
        results = compute_residual_variance(observers, args.k_components)
        
        # Interpret
        print_interpretation(results)
        
        # Save
        if args.output:
            save_results(results, args.output)
        else:
            output_name = args.pattern.replace('*.pt', 'residual_analysis.json') if args.pattern else 'residual_analysis.json'
            save_results(results, f"{args.data_dir}/{output_name}")


if __name__ == '__main__':
    main()