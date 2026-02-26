"""
Compare Real vs Length-Matched Shuffle vs Random Control

This is THE critical test:

Expected outcomes:
- Real >> Shuffle >> Random → Semantics matter (GOOD)
- Real ≈ Shuffle >> Random → It's just length (BAD)
- Real ≈ Shuffle ≈ Random → Something very wrong (VERY BAD)
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind
import numpy as np


def load_observer_results(pattern):
    """Load all observer results matching pattern."""
    results = {}
    for fpath in Path('outputs').glob(f'{pattern}_*.pt'):
        try:
            seed = int(fpath.stem.split('_')[-1])
            results[seed] = torch.load(fpath)
        except:
            continue
    return results


def compute_pairwise_variance(observers):
    """
    Compute mean pairwise difference in attention across observers.
    """
    seeds = sorted(observers.keys())
    differences = []
    
    for i, seed1 in enumerate(seeds):
        for seed2 in seeds[i+1:]:
            attn1 = observers[seed1]['attention_matrix'].cpu()
            attn2 = observers[seed2]['attention_matrix'].cpu()
            
            # Pairwise difference
            diff = (attn1 - attn2).abs().mean().item()
            differences.append(diff)
    
    return np.array(differences)


def main():
    print("="*70)
    print("CRITICAL ABLATION: Real vs Shuffle vs Random")
    print("="*70)
    
    # Load results
    print("\nLoading observer results...")
    
    real_obs = load_observer_results('diverse_observer')
    if not real_obs:
        real_obs = load_observer_results('observer')
    
    shuffle_obs = load_observer_results('shuffle_observer')
    random_obs = load_observer_results('random_observer')  # From your earlier control
    
    if not real_obs:
        print("✗ No real corpus results found!")
        return
    
    print(f"  Real corpus: {len(real_obs)} observers")
    print(f"  Shuffle corpus: {len(shuffle_obs)} observers")
    print(f"  Random corpus: {len(random_obs)} observers")
    
    # Compute variances
    print("\n" + "="*70)
    print("COMPUTING PAIRWISE VARIANCES")
    print("="*70)
    
    results = {}
    
    if real_obs:
        real_var = compute_pairwise_variance(real_obs)
        results['Real'] = real_var
        print(f"\nReal corpus:")
        print(f"  Mean pairwise diff: {real_var.mean():.6f}")
        print(f"  Std: {real_var.std():.6f}")
    
    if shuffle_obs:
        shuffle_var = compute_pairwise_variance(shuffle_obs)
        results['Shuffle'] = shuffle_var
        print(f"\nShuffle corpus (length-matched):")
        print(f"  Mean pairwise diff: {shuffle_var.mean():.6f}")
        print(f"  Std: {shuffle_var.std():.6f}")
    
    if random_obs:
        random_var = compute_pairwise_variance(random_obs)
        results['Random'] = random_var
        print(f"\nRandom corpus:")
        print(f"  Mean pairwise diff: {random_var.mean():.6f}")
        print(f"  Std: {random_var.std():.6f}")
    
    # Statistical comparisons
    print("\n" + "="*70)
    print("STATISTICAL COMPARISONS")
    print("="*70)
    
    if 'Real' in results and 'Shuffle' in results:
        t_stat, p_val = ttest_ind(results['Real'], results['Shuffle'])
        ratio = results['Real'].mean() / results['Shuffle'].mean()
        
        print(f"\nReal vs Shuffle:")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  T-test: t={t_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            if ratio > 1.5:
                print(f"  ✓ Real >> Shuffle (semantics matter!)")
            else:
                print(f"  ⚠ Real slightly > Shuffle (weak effect)")
        else:
            print(f"  ✗ Real ≈ Shuffle (effect is just length!)")
    
    if 'Real' in results and 'Random' in results:
        t_stat, p_val = ttest_ind(results['Real'], results['Random'])
        ratio = results['Real'].mean() / results['Random'].mean()
        
        print(f"\nReal vs Random:")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  T-test: t={t_stat:.4f}, p={p_val:.4f}")
    
    if 'Shuffle' in results and 'Random' in results:
        t_stat, p_val = ttest_ind(results['Shuffle'], results['Random'])
        ratio = results['Shuffle'].mean() / results['Random'].mean()
        
        print(f"\nShuffle vs Random:")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  T-test: t={t_stat:.4f}, p={p_val:.4f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if 'Real' in results and 'Shuffle' in results:
        real_mean = results['Real'].mean()
        shuffle_mean = results['Shuffle'].mean()
        random_mean = results.get('Random', results['Shuffle']).mean()
        
        if real_mean > 1.5 * shuffle_mean and real_mean > 2 * random_mean:
            print("\n✓✓✓ STRONG EFFECT")
            print("Real >> Shuffle >> Random")
            print("Variance is driven by SEMANTICS, not just length.")
            print("Your thesis claims are VALIDATED.")
        
        elif real_mean > 1.2 * shuffle_mean:
            print("\n✓ MODERATE EFFECT")
            print("Real > Shuffle")
            print("Some semantic signal, but length also matters.")
            print("Thesis claims need to be more conservative.")
        
        elif real_mean > shuffle_mean and (real_mean / shuffle_mean) < 1.2:
            print("\n⚠ WEAK EFFECT")
            print("Real slightly > Shuffle")
            print("Effect exists but is small. Length is a major factor.")
            print("Thesis claims need significant hedging.")
        
        else:
            print("\n✗ NO EFFECT")
            print("Real ≈ Shuffle")
            print("Variance is primarily length-driven.")
            print("Need to pivot to conservative interpretation.")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    means = [results[k].mean() for k in results.keys()]
    stds = [results[k].std() for k in results.keys()]
    labels = list(results.keys())
    
    ax1.bar(labels, means, yerr=stds, capsize=5)
    ax1.set_ylabel('Mean Pairwise Attention Difference')
    ax1.set_title('Observer Variance by Corpus Type')
    ax1.grid(axis='y', alpha=0.3)
    
    # Violin plot
    data_for_violin = []
    labels_for_violin = []
    for label, data in results.items():
        data_for_violin.extend(data)
        labels_for_violin.extend([label] * len(data))
    
    df = pd.DataFrame({'Corpus': labels_for_violin, 'Variance': data_for_violin})
    sns.violinplot(data=df, x='Corpus', y='Variance', ax=ax2)
    ax2.set_title('Variance Distribution')
    
    plt.tight_layout()
    plt.savefig('outputs/critical_ablation_real_shuffle_random.png', dpi=150)
    print(f"\n→ Saved plot to outputs/critical_ablation_real_shuffle_random.png")
    
    # Save results
    summary = pd.DataFrame({
        'corpus': list(results.keys()),
        'mean_variance': [results[k].mean() for k in results.keys()],
        'std_variance': [results[k].std() for k in results.keys()],
        'n_comparisons': [len(results[k]) for k in results.keys()]
    })
    
    summary.to_csv('outputs/critical_ablation_summary.csv', index=False)
    print(f"→ Saved summary to outputs/critical_ablation_summary.csv")
    
    print("\n" + "="*70)
    print("✓ CRITICAL ABLATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
