"""
Procrustes Alignment Analysis Script - CORRECTED

Runs high-dimensional Procrustes analysis on observer embeddings.
Uses functions from core.procrustes module.

CRITICAL FIXES (Jan 2026):
--------------------------
1. Energy-based variance decomposition (no double-counting)
2. Imports from core.procrustes (proper module structure)
3. Handles both seed_*.pt and observer_*.pt naming patterns

Usage:
    python procrustes_alignment.py --data-dir "experiments_*/real" --output-dir "outputs/procrustes_real"
    
Reference: ChatGPT analysis (Jan 7, 2026)
"""

import torch
import numpy as np
from pathlib import Path
import json
import argparse
from typing import List, Dict

# Import from core module
from core.procrustes import (
    procrustes_align,
    compute_consensus_and_residuals,
    interpret_residuals
)


def load_and_align_observers(
    observer_files: List[Path],
    reference_idx: int = 0
) -> Dict:
    """
    Load observer files and perform Procrustes alignment with correct variance.
    
    Parameters
    ----------
    observer_files : List[Path]
        Paths to observer .pt files
    reference_idx : int
        Which observer to use as reference
    
    Returns
    -------
    dict
        Complete alignment results including CORRECTED variance decomposition
    """
    
    print("\n" + "="*70)
    print("PROCRUSTES ALIGNMENT (HIGH-DIMENSIONAL)")
    print("="*70)
    
    # Load observers
    embeddings_list = []
    observer_info = []
    
    for i, filepath in enumerate(observer_files):
        print(f"\nLoading observer {i+1}/{len(observer_files)}: {filepath.name}")
        
        data = torch.load(filepath, map_location='cpu')
        
        # Get embeddings - try multiple field names
        embeddings = None
        for field in ['final_features', 'embeddings', 'features']:
            if field in data:
                embeddings = data[field]
                break
        
        if embeddings is None:
            raise ValueError(f"No embeddings found in {filepath}")
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu()
        else:
            embeddings = torch.FloatTensor(embeddings)
        
        embeddings_list.append(embeddings)
        
        # Get observer info
        info = {
            'seed': data.get('seed', data.get('random_seed', data.get('observer_seed', i))),
            'shape': list(embeddings.shape),
            'frobenius_norm': float(torch.norm(embeddings).item())
        }
        observer_info.append(info)
        
        print(f"  Seed: {info['seed']}")
        print(f"  Shape: {info['shape']}")
        print(f"  Frobenius norm: {info['frobenius_norm']:.4f}")
    
    # Align using core.procrustes function
    print(f"\n{'='*70}")
    print(f"ALIGNING TO REFERENCE (Observer {reference_idx+1})")
    print(f"{'='*70}")
    
    alignment_results = procrustes_align(embeddings_list, reference_idx)
    
    # Print alignment quality
    print("\nAlignment distances (Frobenius norm):")
    for i, (before, after) in enumerate(zip(
        alignment_results['distances_before'],
        alignment_results['distances_after']
    )):
        if i == reference_idx:
            print(f"  Observer {i+1}: REFERENCE")
        else:
            reduction = (1 - after/before) * 100 if before > 0 else 0
            print(f"  Observer {i+1}: {before:.4f} → {after:.4f} ({reduction:.1f}% reduction)")
    
    # Compute consensus and residuals with CORRECT variance
    print(f"\n{'='*70}")
    print("ENERGY-BASED VARIANCE DECOMPOSITION (CORRECTED)")
    print(f"{'='*70}")
    
    consensus, residuals, var_decomp = compute_consensus_and_residuals(
        alignment_results['aligned']
    )
    
    print(f"\nEnergy decomposition (Frobenius norms):")
    print(f"  E_total:     {var_decomp['E_total']:.6f}")
    print(f"  E_consensus: {var_decomp['E_consensus']:.6f}")
    print(f"  E_residual:  {var_decomp['E_residual']:.6f}")
    
    print(f"\nVariance fractions (CORRECT - sums to 1.0):")
    print(f"  Consensus:  {var_decomp['consensus_fraction']*100:.2f}%")
    print(f"  Residual:   {var_decomp['residual_fraction']*100:.2f}%")
    print(f"  Sum:        {var_decomp['fraction_sum']*100:.2f}% ← should be ~100%")
    
    print("\nPer-observer residual energy:")
    for i, energy in enumerate(var_decomp['per_observer_residual_energy']):
        print(f"  Observer {i+1}: {energy:.6f}")
    
    # Combine results
    results = {
        'aligned_embeddings': alignment_results['aligned'],
        'transformations': alignment_results['transformations'],
        'consensus': consensus,
        'residuals': residuals,
        'variance_decomposition': var_decomp,
        'alignment_distances': {
            'before': alignment_results['distances_before'],
            'after': alignment_results['distances_after']
        },
        'observer_info': observer_info
    }
    
    return results


def save_alignment_results(results: Dict, output_dir: Path):
    """Save alignment results to disk."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save consensus
    torch.save({
        'consensus': results['consensus'],
        'variance_decomposition': results['variance_decomposition']
    }, output_dir / 'consensus.pt')
    
    # Save aligned embeddings
    for i, aligned in enumerate(results['aligned_embeddings']):
        torch.save({
            'embeddings': aligned,
            'transformation': results['transformations'][i],
            'residual': results['residuals'][i]
        }, output_dir / f'aligned_observer_{i}.pt')
    
    # Save statistics as JSON
    stats = {
        'variance_decomposition': results['variance_decomposition'],
        'alignment_distances': results['alignment_distances'],
        'observer_info': results['observer_info']
    }
    
    with open(output_dir / 'alignment_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Saved alignment results to {output_dir}")


def compare_corpora(
    corpus_dirs: Dict[str, Path],
    seeds: List[int] = [42, 43, 44],
    output_dir: Path = None
) -> Dict:
    """
    Compare observer variance across multiple corpora using CORRECTED metrics.
    
    Parameters
    ----------
    corpus_dirs : dict
        Mapping corpus_name -> directory path
    seeds : List[int]
        Which observer seeds to load
    output_dir : Path, optional
        Where to save comparison results
    
    Returns
    -------
    dict
        Comparison of consensus/residual fractions across corpora
    """
    
    print("\n" + "="*70)
    print("COMPARING OBSERVER VARIANCE ACROSS CORPORA")
    print("="*70)
    
    corpus_results = {}
    
    for corpus_name, corpus_dir in corpus_dirs.items():
        print(f"\n{'='*70}")
        print(f"CORPUS: {corpus_name.upper()}")
        print(f"{'='*70}")
        
        # Find observer files (handle both naming patterns)
        observer_files = []
        for seed in seeds:
            # Try multiple naming patterns
            patterns = [
                corpus_dir / f"observer_{seed}.pt",
                corpus_dir / f"seed_{seed}.pt",
            ]
            
            for pattern in patterns:
                if pattern.exists():
                    observer_files.append(pattern)
                    break
        
        if len(observer_files) < 2:
            print(f"⚠️  Skipping {corpus_name}: found only {len(observer_files)} observers")
            continue
        
        # Run alignment with CORRECTED variance
        results = load_and_align_observers(observer_files, reference_idx=0)
        corpus_results[corpus_name] = results
    
    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS CORPORA (CORRECTED METRICS)")
    print(f"{'='*70}")
    
    comparison = {}
    for corpus_name, results in corpus_results.items():
        var_decomp = results['variance_decomposition']
        comparison[corpus_name] = {
            'consensus_fraction': var_decomp['consensus_fraction'],
            'residual_fraction': var_decomp['residual_fraction'],
            'E_consensus': var_decomp['E_consensus'],
            'E_residual': var_decomp['E_residual'],
            'fraction_sum': var_decomp['fraction_sum']
        }
    
    # Print table
    print(f"\n{'Corpus':<20} {'Consensus %':<15} {'Residual %':<15} {'Sum %':<10}")
    print("-" * 60)
    for corpus_name, stats in comparison.items():
        print(f"{corpus_name:<20} {stats['consensus_fraction']*100:>13.2f}% "
              f"{stats['residual_fraction']*100:>13.2f}% "
              f"{stats['fraction_sum']*100:>8.2f}%")
    
    # Save if output dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'corpus_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✅ Saved comparison to {output_dir / 'corpus_comparison.json'}")
    
    return comparison


def find_observer_files(data_dir: Path) -> List[Path]:
    """Find observer files regardless of naming pattern."""
    
    # Try different patterns
    patterns = ['observer_*.pt', 'seed_*.pt', '*_observer_*.pt']
    
    for pattern in patterns:
        files = sorted(data_dir.glob(pattern))
        if files:
            return files
    
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procrustes alignment with CORRECTED energy-based variance"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing observer files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='File pattern for observers (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/procrustes_alignment',
        help='Output directory for results'
    )
    parser.add_argument(
        '--reference-idx',
        type=int,
        default=0,
        help='Index of reference observer'
    )
    
    args = parser.parse_args()
    
    # Find observer files
    data_dir = Path(args.data_dir)
    
    if args.pattern:
        observer_files = sorted(data_dir.glob(args.pattern))
    else:
        observer_files = find_observer_files(data_dir)
    
    if not observer_files:
        print(f"❌ No files found in {data_dir}")
        print("   Tried patterns: observer_*.pt, seed_*.pt, *_observer_*.pt")
        exit(1)
    
    print(f"✅ Found {len(observer_files)} observer files")
    
    # Run alignment with CORRECTED variance
    results = load_and_align_observers(observer_files, args.reference_idx)
    
    # Save results
    save_alignment_results(results, args.output_dir)
    
    print("\n" + "="*70)
    print("✅ ALIGNMENT COMPLETE (CORRECTED VARIANCE)")
    print("="*70)