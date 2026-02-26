"""
Procrustes alignment for comparing observer geometries

Aligns multiple observer embeddings to measure consensus vs observer-specific variance
"""

import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
from pathlib import Path
import json


def procrustes_align(embeddings_list, reference_idx=0):
    """
    Align multiple observer geometries using Procrustes analysis
    
    Finds optimal rotation to align each observer to reference observer,
    minimizing Frobenius norm: ||X_i @ R_i - X_ref||_F
    
    Parameters:
    -----------
    embeddings_list : List[Tensor]
        List of [N, D] embedding matrices from different observers
    reference_idx : int
        Which observer to use as reference (default: first)
    
    Returns:
    --------
    aligned : List[Tensor]
        Aligned embeddings (same dimensions as input)
    transformations : List[Tensor]
        Rotation matrices [D, D] used for alignment
    distances : List[float]
        Distance between each observer and reference (before alignment)
    """
    
    n_observers = len(embeddings_list)
    reference = embeddings_list[reference_idx]
    
    # Convert to numpy for scipy
    if torch.is_tensor(reference):
        reference = reference.cpu().numpy()
    
    aligned = []
    transformations = []
    distances_before = []
    distances_after = []
    
    for i, emb in enumerate(embeddings_list):
        if torch.is_tensor(emb):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = emb
        
        if i == reference_idx:
            # Reference doesn't need alignment
            aligned.append(torch.FloatTensor(emb_np))
            transformations.append(torch.eye(emb_np.shape[1]))
            distances_before.append(0.0)
            distances_after.append(0.0)
            continue
        
        # Measure distance before alignment
        dist_before = np.linalg.norm(emb_np - reference, 'fro')
        distances_before.append(dist_before)
        
        # Solve for optimal rotation: min ||emb @ R - reference||_F
        R, scale = orthogonal_procrustes(emb_np, reference)
        
        # Apply transformation
        aligned_emb = emb_np @ R
        aligned.append(torch.FloatTensor(aligned_emb))
        transformations.append(torch.FloatTensor(R))
        
        # Measure distance after alignment
        dist_after = np.linalg.norm(aligned_emb - reference, 'fro')
        distances_after.append(dist_after)
    
    results = {
        'aligned': aligned,
        'transformations': transformations,
        'distances_before': distances_before,
        'distances_after': distances_after,
        'reference_idx': reference_idx,
        'n_observers': n_observers
    }
    
    return results


def compute_consensus_and_residuals(aligned_embeddings):
    """
    Decompose aligned embeddings into consensus + residuals
    
    Consensus = mean across observers (shared structure)
    Residuals = deviations from consensus (observer-specific)
    
    Parameters:
    -----------
    aligned_embeddings : List[Tensor]
        Aligned embeddings from procrustes_align
    
    Returns:
    --------
    consensus : Tensor [N, D]
        Mean embedding across observers
    residuals : List[Tensor]
        Per-observer residuals [N, D]
    variance_decomposition : dict
        Variance statistics
    """
    
    # Stack all observers: [n_observers, N, D]
    stacked = torch.stack(aligned_embeddings, dim=0)
    n_observers, n_articles, dim = stacked.shape
    
    # Consensus = mean across observers
    consensus = stacked.mean(dim=0)  # [N, D]
    
    # Residuals = deviation from consensus
    residuals = []
    for i in range(n_observers):
        residual = stacked[i] - consensus
        residuals.append(residual)
    
    # Variance decomposition
    total_var = stacked.var().item()
    consensus_var = consensus.var().item()
    
    # Residual variance = mean variance of residuals
    residual_vars = [r.var().item() for r in residuals]
    mean_residual_var = np.mean(residual_vars)
    
    # Fraction explained
    consensus_fraction = consensus_var / total_var if total_var > 0 else 0
    residual_fraction = mean_residual_var / total_var if total_var > 0 else 0
    
    variance_decomposition = {
        'total_variance': total_var,
        'consensus_variance': consensus_var,
        'mean_residual_variance': mean_residual_var,
        'consensus_fraction': consensus_fraction,
        'residual_fraction': residual_fraction,
        'per_observer_residual_variance': residual_vars
    }
    
    return consensus, residuals, variance_decomposition


def load_and_align_observers(observer_files, reference_idx=0):
    """
    Load observer files and perform Procrustes alignment
    
    Parameters:
    -----------
    observer_files : List[Path]
        Paths to observer .pt files
    reference_idx : int
        Which observer to use as reference
    
    Returns:
    --------
    Complete alignment results including variance decomposition
    """
    
    print("\n" + "="*70)
    print("PROCRUSTES ALIGNMENT")
    print("="*70)
    
    # Load observers
    embeddings_list = []
    observer_info = []
    
    for i, filepath in enumerate(observer_files):
        print(f"\nLoading observer {i+1}/{len(observer_files)}: {filepath.name}")
        
        data = torch.load(filepath, map_location='cpu')
        
        # Get embeddings
        embeddings = data.get('embeddings') or data.get('features')
        if embeddings is None:
            raise ValueError(f"No embeddings found in {filepath}")
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu()
        
        embeddings_list.append(embeddings)
        
        # Get observer info
        info = {
            'seed': data.get('seed', data.get('random_seed', i)),
            'shape': embeddings.shape,
            'norm': embeddings.norm().item()
        }
        observer_info.append(info)
        
        print(f"  Seed: {info['seed']}")
        print(f"  Shape: {info['shape']}")
        print(f"  Norm: {info['norm']:.4f}")
    
    # Align
    print(f"\n{'='*70}")
    print(f"ALIGNING TO REFERENCE (Observer {reference_idx+1})")
    print(f"{'='*70}")
    
    alignment_results = procrustes_align(embeddings_list, reference_idx)
    
    # Print alignment quality
    print("\nAlignment distances:")
    for i, (before, after) in enumerate(zip(
        alignment_results['distances_before'],
        alignment_results['distances_after']
    )):
        if i == reference_idx:
            print(f"  Observer {i+1}: REFERENCE")
        else:
            reduction = (1 - after/before) * 100 if before > 0 else 0
            print(f"  Observer {i+1}: {before:.4f} → {after:.4f} ({reduction:.1f}% reduction)")
    
    # Compute consensus and residuals
    print(f"\n{'='*70}")
    print("VARIANCE DECOMPOSITION")
    print(f"{'='*70}")
    
    consensus, residuals, var_decomp = compute_consensus_and_residuals(
        alignment_results['aligned']
    )
    
    print(f"\nTotal variance: {var_decomp['total_variance']:.6f}")
    print(f"Consensus variance: {var_decomp['consensus_variance']:.6f} "
          f"({var_decomp['consensus_fraction']*100:.2f}%)")
    print(f"Mean residual variance: {var_decomp['mean_residual_variance']:.6f} "
          f"({var_decomp['residual_fraction']*100:.2f}%)")
    
    print("\nPer-observer residual variance:")
    for i, var in enumerate(var_decomp['per_observer_residual_variance']):
        print(f"  Observer {i+1}: {var:.6f}")
    
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


def save_alignment_results(results, output_dir):
    """Save alignment results to disk"""
    
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
    
    print(f"\n✓ Saved alignment results to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Procrustes alignment of observer embeddings")
    parser.add_argument('--data_dir', type=str, default='outputs',
                       help='Directory containing observer files')
    parser.add_argument('--pattern', type=str, default='*observer*.pt',
                       help='File pattern for observers')
    parser.add_argument('--output_dir', type=str, default='outputs/alignment',
                       help='Output directory for results')
    parser.add_argument('--reference_idx', type=int, default=0,
                       help='Index of reference observer')
    
    args = parser.parse_args()
    
    # Find observer files
    data_dir = Path(args.data_dir)
    observer_files = sorted(data_dir.glob(args.pattern))
    
    if not observer_files:
        print(f"No files found matching {args.pattern} in {data_dir}")
        exit(1)
    
    print(f"Found {len(observer_files)} observer files")
    
    # Run alignment
    results = load_and_align_observers(observer_files, args.reference_idx)
    
    # Save results
    save_alignment_results(results, args.output_dir)
