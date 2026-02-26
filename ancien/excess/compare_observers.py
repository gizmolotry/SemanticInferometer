"""
Compare Observers - Procrustes Analysis

Analyzes geometric differences between observers using Procrustes alignment.
Generates quantitative metrics + per-point divergence.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Import your Procrustes functions
import sys
sys.path.append('.')
from core.procrustes import procrustes_align_umap, procrustes_align_attention


def load_all_observers(output_dir='outputs'):
    """Load all observer files."""
    observers = {}
    for fpath in Path(output_dir).glob('observer_*.pt'):
        seed = int(fpath.stem.split('_')[1])
        observers[seed] = torch.load(fpath)
    return observers


def compute_umap_if_missing(observers, n_neighbors=15, min_dist=0.1):
    """Compute UMAP coords for all observers if not already present."""
    print("Computing UMAP coordinates...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    
    for seed, obs in observers.items():
        if 'umap_coords' not in obs:
            print(f"  Computing UMAP for observer {seed}...")
            tokens = obs['article_tokens'].cpu().numpy()
            coords = reducer.fit_transform(tokens)
            obs['umap_coords'] = torch.tensor(coords)
            
            # Save back to file
            torch.save(obs, f'outputs/observer_{seed}.pt')
            print(f"    → Saved UMAP coords to observer_{seed}.pt")
        else:
            print(f"  Observer {seed}: UMAP coords already exist")


def analyze_attention_variance(observers):
    """Analyze attention matrix properties."""
    print("\n" + "="*70)
    print("ATTENTION MATRIX ANALYSIS")
    print("="*70)
    
    results = []
    for seed, obs in observers.items():
        attn = obs['attention_matrix'].cpu()
        
        # Compute metrics
        variance = attn.var().item()
        entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
        
        # Off-diagonal mean (excluding self-attention)
        mask = ~torch.eye(len(attn), dtype=bool)
        off_diag_mean = attn[mask].mean().item()
        off_diag_std = attn[mask].std().item()
        
        results.append({
            'observer': seed,
            'variance': variance,
            'entropy': entropy,
            'off_diag_mean': off_diag_mean,
            'off_diag_std': off_diag_std
        })
        
        print(f"\nObserver {seed}:")
        print(f"  Variance:      {variance:.6f}")
        print(f"  Entropy:       {entropy:.4f}")
        print(f"  Off-diag mean: {off_diag_mean:.6f} ± {off_diag_std:.6f}")
    
    df = pd.DataFrame(results)
    df.to_csv('outputs/attention_analysis.csv', index=False)
    print(f"\n→ Saved to outputs/attention_analysis.csv")
    
    return df


def pairwise_procrustes_comparison(observers, mode='umap'):
    """Compare all pairs of observers using Procrustes."""
    print("\n" + "="*70)
    print(f"PAIRWISE PROCRUSTES COMPARISON ({mode.upper()})")
    print("="*70)
    
    results = []
    
    for (s1, obs1), (s2, obs2) in combinations(observers.items(), 2):
        print(f"\nComparing Observer {s1} vs {s2}...")
        
        if mode == 'umap':
            coords1 = obs1['umap_coords'].cpu().numpy()
            coords2 = obs2['umap_coords'].cpu().numpy()
            alignment = procrustes_align_umap(coords1, coords2)
        elif mode == 'attention':
            attn1 = obs1['attention_matrix'].cpu().numpy()
            attn2 = obs2['attention_matrix'].cpu().numpy()
            alignment = procrustes_align_attention(attn1, attn2)
        
        residuals = alignment['per_point_residuals']
        
        result = {
            'observer_1': s1,
            'observer_2': s2,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_median': np.median(residuals),
            'residual_max': residuals.max(),
            'residual_90pct': np.percentile(residuals, 90)
        }
        
        results.append(result)
        
        print(f"  Mean residual:   {result['residual_mean']:.6f}")
        print(f"  Median residual: {result['residual_median']:.6f}")
        print(f"  Max residual:    {result['residual_max']:.6f}")
    
    df = pd.DataFrame(results)
    df.to_csv(f'outputs/procrustes_{mode}.csv', index=False)
    print(f"\n→ Saved to outputs/procrustes_{mode}.csv")
    
    return df


def identify_high_divergence_articles(observers, top_k=50):
    """Find articles that move the most between observers."""
    print("\n" + "="*70)
    print("HIGH DIVERGENCE ARTICLES")
    print("="*70)
    
    # Compute mean residual per article across all observer pairs
    n_articles = len(observers[list(observers.keys())[0]]['metadata'])
    divergence_scores = np.zeros(n_articles)
    count = 0
    
    for (s1, obs1), (s2, obs2) in combinations(observers.items(), 2):
        coords1 = obs1['umap_coords'].cpu().numpy()
        coords2 = obs2['umap_coords'].cpu().numpy()
        alignment = procrustes_align_umap(coords1, coords2)
        divergence_scores += alignment['per_point_residuals']
        count += 1
    
    divergence_scores /= count  # Average across pairs
    
    # Get top-k
    top_indices = np.argsort(divergence_scores)[-top_k:][::-1]
    
    # Build report
    results = []
    reference_obs = observers[list(observers.keys())[0]]
    
    for idx in top_indices:
        meta = reference_obs['metadata'][idx]
        results.append({
            'index': idx,
            'divergence': divergence_scores[idx],
            'source': meta['source'],
            'timestamp': meta['timestamp'],
            'url': meta.get('url', ''),
            'preview': meta['text_preview'][:100]
        })
    
    df = pd.DataFrame(results)
    df.to_csv('outputs/high_divergence_articles.csv', index=False)
    
    print(f"\nTop {top_k} most divergent articles:")
    print(df[['index', 'divergence', 'source']].head(20).to_string())
    print(f"\n→ Saved to outputs/high_divergence_articles.csv")
    
    return df


def plot_residual_heatmap(procrustes_df, mode='umap'):
    """Plot heatmap of pairwise residuals."""
    seeds = sorted(set(procrustes_df['observer_1'].unique()) | 
                   set(procrustes_df['observer_2'].unique()))
    
    matrix = np.zeros((len(seeds), len(seeds)))
    
    for _, row in procrustes_df.iterrows():
        i = seeds.index(row['observer_1'])
        j = seeds.index(row['observer_2'])
        matrix[i, j] = row['residual_mean']
        matrix[j, i] = row['residual_mean']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, 
                xticklabels=seeds, 
                yticklabels=seeds,
                annot=True, 
                fmt='.4f',
                cmap='RdYlGn_r')
    plt.title(f'Pairwise Procrustes Residuals ({mode.upper()})')
    plt.tight_layout()
    plt.savefig(f'outputs/residual_heatmap_{mode}.png', dpi=150)
    print(f"→ Saved heatmap to outputs/residual_heatmap_{mode}.png")


def main():
    print("="*70)
    print("OBSERVER COMPARISON PIPELINE")
    print("="*70)
    
    # Load observers
    print("\nLoading observers...")
    observers = load_all_observers()
    print(f"Loaded {len(observers)} observers: {sorted(observers.keys())}")
    
    # Compute UMAP if missing
    compute_umap_if_missing(observers)
    
    # 1. Attention analysis
    attn_df = analyze_attention_variance(observers)
    
    # 2. UMAP Procrustes
    umap_df = pairwise_procrustes_comparison(observers, mode='umap')
    plot_residual_heatmap(umap_df, mode='umap')
    
    # 3. Attention Procrustes
    attn_proc_df = pairwise_procrustes_comparison(observers, mode='attention')
    plot_residual_heatmap(attn_proc_df, mode='attention')
    
    # 4. High divergence articles
    divergence_df = identify_high_divergence_articles(observers, top_k=50)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - outputs/attention_analysis.csv")
    print("  - outputs/procrustes_umap.csv")
    print("  - outputs/procrustes_attention.csv")
    print("  - outputs/high_divergence_articles.csv")
    print("  - outputs/residual_heatmap_umap.png")
    print("  - outputs/residual_heatmap_attention.png")


if __name__ == "__main__":
    main()
