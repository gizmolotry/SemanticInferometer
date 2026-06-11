"""
Component Isolation Tests

Tests each component in isolation to identify which contributes to variance:
1. RKS only (no GRU, no RoPE)
2. RKS + RoPE (no GRU)
3. RKS + GRU (no RoPE)
4. Full pipeline (RKS + RoPE + GRU)

Helps diagnose which component is failing or which is creating variance.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add repository root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.complete_pipeline import run_multi_observer_experiment


def load_articles(jsonl_path: str = 'data/scraped_articles.jsonl'):
    """Load articles from JSONL."""
    import json
    articles = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def quick_variance_check(observers: dict) -> dict:
    """Quick variance calculation."""
    attn_matrices = [obs['attention_matrix'].numpy() for obs in observers.values()]
    attn_stack = np.stack(attn_matrices)
    variance_matrix = np.var(attn_stack, axis=0)
    
    n_articles = variance_matrix.shape[0]
    mask = ~np.eye(n_articles, dtype=bool)
    variance_values = variance_matrix[mask]
    
    return {
        'mean': float(variance_values.mean()),
        'max': float(variance_values.max()),
        'std': float(variance_values.std())
    }


def run_isolation_tests(
    articles,
    seeds=[42, 43, 44],
    device='cuda',
    output_dir='outputs/isolation_tests'
):
    """
    Run component isolation tests.
    
    Tests:
    1. RKS only
    2. RKS + RoPE
    3. RKS + GRU  
    4. Full pipeline (RKS + RoPE + GRU)
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("="*70)
    print("COMPONENT ISOLATION TESTS")
    print("="*70)
    print(f"\nUsing {len(articles)} articles")
    print(f"Testing with {len(seeds)} observers: {seeds}")
    
    # Test 1: RKS only
    print("\n" + "="*70)
    print("TEST 1: RKS ONLY (no GRU, no RoPE)")
    print("="*70)
    
    observers_rks = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=False,
        use_framing_rope=False,
        device=device,
        use_rks=True,
        rks_dim=512,
        rks_sigma=None  # Auto-estimate
    )
    
    var_rks = quick_variance_check(observers_rks)
    results['rks_only'] = var_rks
    
    print(f"\nRKS ONLY VARIANCE:")
    print(f"  Mean: {var_rks['mean']:.6f}")
    print(f"  Max:  {var_rks['max']:.6f}")
    
    # Save
    for seed, obs in observers_rks.items():
        torch.save(obs, Path(output_dir) / f'rks_only_{seed}.pt')
    
    # Test 2: RKS + RoPE
    print("\n" + "="*70)
    print("TEST 2: RKS + RoPE (no GRU)")
    print("="*70)
    
    observers_rks_rope = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=False,
        use_framing_rope=True,
        device=device,
        use_rks=True,
        rks_dim=512,
        rks_sigma=None
    )
    
    var_rks_rope = quick_variance_check(observers_rks_rope)
    results['rks_rope'] = var_rks_rope
    
    print(f"\nRKS + RoPE VARIANCE:")
    print(f"  Mean: {var_rks_rope['mean']:.6f}")
    print(f"  Max:  {var_rks_rope['max']:.6f}")
    
    for seed, obs in observers_rks_rope.items():
        torch.save(obs, Path(output_dir) / f'rks_rope_{seed}.pt')
    
    # Test 3: RKS + GRU
    print("\n" + "="*70)
    print("TEST 3: RKS + GRU (no RoPE)")
    print("="*70)
    
    observers_rks_gru = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=True,
        use_framing_rope=False,
        device=device,
        use_rks=True,
        rks_dim=512,
        rks_sigma=None
    )
    
    var_rks_gru = quick_variance_check(observers_rks_gru)
    results['rks_gru'] = var_rks_gru
    
    print(f"\nRKS + GRU VARIANCE:")
    print(f"  Mean: {var_rks_gru['mean']:.6f}")
    print(f"  Max:  {var_rks_gru['max']:.6f}")
    
    for seed, obs in observers_rks_gru.items():
        torch.save(obs, Path(output_dir) / f'rks_gru_{seed}.pt')
    
    # Test 4: Full pipeline
    print("\n" + "="*70)
    print("TEST 4: FULL PIPELINE (RKS + RoPE + GRU)")
    print("="*70)
    
    observers_full = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=True,
        use_framing_rope=True,
        device=device,
        use_rks=True,
        rks_dim=512,
        rks_sigma=None
    )
    
    var_full = quick_variance_check(observers_full)
    results['full'] = var_full
    
    print(f"\nFULL PIPELINE VARIANCE:")
    print(f"  Mean: {var_full['mean']:.6f}")
    print(f"  Max:  {var_full['max']:.6f}")
    
    for seed, obs in observers_full.items():
        torch.save(obs, Path(output_dir) / f'full_{seed}.pt')
    
    # Summary
    print("\n" + "="*70)
    print("COMPONENT ISOLATION SUMMARY")
    print("="*70)
    
    print("\nVariance comparison:")
    print(f"  RKS only:      {results['rks_only']['mean']:.6f}")
    print(f"  RKS + RoPE:    {results['rks_rope']['mean']:.6f}")
    print(f"  RKS + GRU:     {results['rks_gru']['mean']:.6f}")
    print(f"  Full pipeline: {results['full']['mean']:.6f}")
    
    print("\nInterpretation:")
    
    # Which component adds most variance?
    base_var = results['rks_only']['mean']
    rope_contribution = results['rks_rope']['mean'] - base_var
    gru_contribution = results['rks_gru']['mean'] - base_var
    
    print(f"\nComponent contributions (relative to RKS baseline):")
    print(f"  RoPE: {rope_contribution:+.6f}")
    print(f"  GRU:  {gru_contribution:+.6f}")
    
    if base_var < 0.0001:
        print("\n⚠ RKS alone creates almost no variance!")
        print("  Possible causes:")
        print("  - σ is wrong (check auto-estimation)")
        print("  - RKS correlation is low (kernel approximation broken)")
        print("  - Features collapsed before RKS")
    elif rope_contribution < 0:
        print("\n⚠ RoPE is REDUCING variance (unexpected)")
        print("  RoPE might be normalizing features too aggressively")
    elif gru_contribution < 0:
        print("\n⚠ GRU is REDUCING variance (unexpected)")
        print("  GRU might be collapsing to same attractor")
    else:
        print("\n✓ Components are adding variance as expected")
    
    # Save summary
    summary_path = Path(output_dir) / 'isolation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_dir}/")
    print(f"  - isolation_summary.json")
    print(f"  - <config>_<seed>.pt files")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Component isolation tests')
    parser.add_argument('--articles', type=str, default='data/scraped_articles.jsonl',
                       help='Path to articles JSONL')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Observer seeds to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output', type=str, default='outputs/isolation_tests',
                       help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N articles (for quick testing)')
    
    args = parser.parse_args()
    
    # Load articles
    print("Loading articles...")
    articles = load_articles(args.articles)
    
    if args.sample:
        articles = articles[:args.sample]
        print(f"Using sample of {len(articles)} articles")
    
    # Run tests
    results = run_isolation_tests(
        articles=articles,
        seeds=args.seeds,
        device=args.device,
        output_dir=args.output
    )
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print("\nNext steps:")
    print("1. Check which component creates most variance")
    print("2. If RKS alone ≈ 0, check σ auto-estimation")
    print("3. If GRU reduces variance, try disabling it")
    print("4. Run comprehensive_diagnostics.py on the results")
