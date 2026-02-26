"""Compare variance between old and new observers"""
import torch
import numpy as np
from pathlib import Path

# Get project root (parent of scripts folder if we're in scripts)
project_root = Path(__file__).parent
if project_root.name == 'scripts':
    project_root = project_root.parent

outputs_dir = project_root / 'outputs'

# Load old observers (42-46)
print(f"Looking for observers in: {outputs_dir}")
print("\nLoading observers...")
obs_old = {}
for seed in [42, 43, 44, 45, 46]:
    file_path = outputs_dir / f'observer_{seed}.pt'
    obs_old[seed] = torch.load(file_path)

# Load new observers (diverse_*)
obs_new = {}
for seed in [1, 1000, 100000, 1000000, 10000000]:
    file_path = outputs_dir / f'diverse_observer_{seed}.pt'
    obs_new[seed] = torch.load(file_path)

def calc_variance(observers):
    """Calculate both pointwise variance AND pairwise differences"""
    seeds = sorted(observers.keys())
    
    # Stack all attention matrices
    attention_stack = torch.stack([
        observers[seed]['attention_matrix'].cpu() 
        for seed in seeds
    ])  # Shape: (n_observers, n_articles, n_articles)
    
    # 1. POINTWISE VARIANCE (original metric)
    # Variance at each (i,j) position across observers
    pointwise_var = attention_stack.var(dim=0).mean().item()
    
    # 2. PAIRWISE DIFFERENCES (alternative metric)
    pairwise_diffs = []
    for i, seed1 in enumerate(seeds):
        for seed2 in seeds[i+1:]:
            attn1 = observers[seed1]['attention_matrix'].cpu()
            attn2 = observers[seed2]['attention_matrix'].cpu()
            diff = (attn1 - attn2).abs().mean().item()
            pairwise_diffs.append(diff)
    
    pairwise_mean = np.mean(pairwise_diffs)
    
    return pointwise_var, pairwise_mean

# Calculate variances
print("\n" + "="*70)
print("OLD observers (42-46):")
print("="*70)
point_old, pair_old = calc_variance(obs_old)
print(f"  Pointwise variance:   {point_old:.6f}  <-- ORIGINAL METRIC")
print(f"  Pairwise differences: {pair_old:.6f}")

print("\n" + "="*70)
print("NEW observers (diverse_1 through diverse_10000000):")
print("="*70)
point_new, pair_new = calc_variance(obs_new)
print(f"  Pointwise variance:   {point_new:.6f}  <-- ORIGINAL METRIC")
print(f"  Pairwise differences: {pair_new:.6f}")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print(f"  Pointwise ratio:  {point_old/point_new:.2f}x")
print(f"  Pairwise ratio:   {pair_old/pair_new:.2f}x")

print("\n" + "="*70)
print("VERDICT:")
print("="*70)

if point_old > 0.001:
    print("✓ OLD observers have STRONG pointwise variance (>0.001)")
    print("  These match your original strong results!")
else:
    print("✗ OLD observers have WEAK pointwise variance (<0.001)")

if point_new > 0.001:
    print("✓ NEW observers have STRONG pointwise variance (>0.001)")
else:
    print("✗ NEW observers have WEAK pointwise variance (<0.001)")

print("\n" + "="*70)
print("EXPLANATION:")
print("="*70)
print("Your original result (0.001090) was POINTWISE variance.")
print("This measures: variance at each attention position across observers.")
print("")
print("Observer 42-46 use CLOSE seeds (42,43,44,45,46)")
print("  → Similar random initializations → Low variance")
print("")
print("Diverse observers use DISTANT seeds (1, 1000, 100000, ...)")
print("  → Different random initializations → Higher variance")
print("")
if point_new > point_old:
    print(f"✓ Diverse observers have {point_new/point_old:.1f}x MORE variance")
    print("  This is EXPECTED - they're designed to be more different!")
else:
    print("⚠ Something is wrong - diverse should have MORE variance")
