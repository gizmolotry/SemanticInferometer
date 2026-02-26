"""
Belief Transformer Experimental Runner - COMPLETE VERSION

Supports:
- Multiple experiment modes (standard, enhanced, shared_pca, multi_kernel, etc.)
- Real corpus, temporal batches, AND control corpora (NEW!)
- Multi-kernel experiments  
- Sigma/alpha sweeps
- Batch temporal processing

Usage:
    # Real corpus
    python run_experiments.py --corpus real --mode enhanced --seeds 42 43 44
    
    # Control corpora (NEW!)
    python run_experiments.py --corpus control_constant --mode enhanced --seeds 42 43 44
    python run_experiments.py --corpus control_shuffled --mode enhanced --seeds 42 43 44
    python run_experiments.py --corpus control_random --mode enhanced --seeds 42 43 44
    
    # Temporal batches
    python run_experiments.py --corpus temporal --mode enhanced --seeds 42 43 --batch-temporal
    
    # Legacy control (backward compatible)
    python run_experiments.py --corpus control --mode enhanced --seeds 42 43 44
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
import torch

# Project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

from core.complete_pipeline import run_multi_observer_experiment


# =========================================================================
# EXPERIMENT CONFIGURATIONS
# =========================================================================

EXPERIMENT_MODES = {
    'standard': {
        'name': 'Standard Mode',
        'description': 'Basic pipeline (no contrastive, no PCA)',
        'use_contrastive': False,
        'use_pca_removal': False,
        'shared_pca': False,
        'kernel_type': 'rbf'
    },
    
    'contrastive': {
        'name': 'Contrastive Only',
        'description': 'Contrastive NLI features only',
        'use_contrastive': True,
        'use_pca_removal': False,
        'shared_pca': False,
        'kernel_type': 'rbf'
    },
    
    'pca': {
        'name': 'PCA Removal Only',
        'description': 'PCA topic removal only',
        'use_contrastive': False,
        'use_pca_removal': True,
        'shared_pca': False,
        'kernel_type': 'rbf'
    },
    
    'enhanced': {
        'name': 'Enhanced Mode',
        'description': 'Contrastive NLI + PCA removal (original)',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': False,
        'kernel_type': 'rbf'
    },
    
    'shared_pca': {
        'name': 'Shared PCA Control',
        'description': 'Tests if observer variance is real vs PCA artifact',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'kernel_type': 'rbf'
    },
    
    'multi_kernel': {
        'name': 'Multi-Kernel Observers',
        'description': 'Different kernels = genuinely different observers',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'kernel_types': ['rbf', 'laplacian', 'rq', 'imq'],
        'seed': 42
    },
    
    'sigma_sweep': {
        'name': 'Kernel Bandwidth Sweep',
        'description': 'Different σ values = observers with different scales',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'kernel_type': 'rbf',
        'sigmas': [0.3, 0.6, 0.9, 1.2, 1.5],
        'seed': 42
    },
    
    'rq_sweep': {
        'name': 'Rational Quadratic α Sweep',
        'description': 'Different smoothness parameters',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'kernel_type': 'rq',
        'alphas': [0.5, 1.0, 2.0, 5.0],
        'sigma': 0.6,
        'seed': 42
    },
    
    'laplacian': {
        'name': 'Laplacian Kernel',
        'description': 'Sharp, local kernel (better for clusters)',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'kernel_type': 'laplacian'
    },
    
    'framing_kernels': {
        'name': 'Multi-Kernel Per Framing',
        'description': 'Different kernel for each of 8 framings (maintains structure)',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'use_multi_framing_rks': True,
        'kernel_types_per_framing': ['rbf', 'laplacian', 'rq', 'imq', 'matern', 'rbf', 'laplacian', 'rq']
    },
    
    'framing_kernels_uniform': {
        'name': 'Same Kernel All Framings',
        'description': 'All framings use same kernel (baseline for framing_kernels)',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'use_multi_framing_rks': True,
        'kernel_type': 'rbf'
    },
    
    'minimal': {
        'name': 'Minimal Architecture',
        'description': 'NLI → RKS → Attention (no GRU)',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'use_gru': False,
        'kernel_type': 'rbf'
    },
    
    'no_gru': {
        'name': 'No Temporal GRU',
        'description': 'Tests if temporal processing affects variance',
        'use_contrastive': True,
        'use_pca_removal': True,
        'shared_pca': True,
        'use_gru': False,
        'kernel_type': 'rbf'
    }
}


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def load_articles(filepath):
    """Load articles from JSONL file"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Article file not found: {filepath}")
    
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))
    
    return articles


def load_corpus(corpus_type, limit=None):
    """
    Load corpus based on type - UPDATED with control corpus support!
    
    Supported corpus types:
    - real: Scraped news articles
    - temporal: Combined temporal corpus
    - control: Legacy control corpus
    - control_constant: Identical text control (NEW!)
    - control_shuffled: Shuffled tokens control (NEW!)
    - control_random: Random text control (NEW!)
    """
    # Control corpora (NEW!)
    if corpus_type == 'control_constant':
        corpus_file = DATA_DIR / 'control_constant.jsonl'
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Control constant corpus not found: {corpus_file}\n"
                f"Generate it: python controls/make_control_corpus.py"
            )
        articles = load_articles(corpus_file)
        print(f"✓ Loaded CONSTANT control: {len(articles)} identical articles")
    
    elif corpus_type == 'control_shuffled':
        corpus_file = DATA_DIR / 'control_shuffled.jsonl'
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Control shuffled corpus not found: {corpus_file}\n"
                f"Generate it: python controls/make_control_corpus.py"
            )
        articles = load_articles(corpus_file)
        print(f"✓ Loaded SHUFFLED control: {len(articles)} articles (same tokens, random order)")
    
    elif corpus_type == 'control_random':
        corpus_file = DATA_DIR / 'control_random.jsonl'
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Control random corpus not found: {corpus_file}\n"
                f"Generate it: python controls/make_control_corpus.py"
            )
        articles = load_articles(corpus_file)
        print(f"✓ Loaded RANDOM control: {len(articles)} articles (random text)")
    
    # Legacy control (backward compatible)
    elif corpus_type == 'control':
        print("⚠ Using legacy 'control' corpus name")
        print("  Recommend: Use --corpus control_constant, control_shuffled, or control_random")
        corpus_file = DATA_DIR / 'control_corpus.jsonl'
        if not corpus_file.exists():
            # Try the combined control file
            corpus_file = DATA_DIR / 'control_combined.jsonl'
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Control corpus not found.\n"
                f"Generate it: python controls/make_control_corpus.py"
            )
        articles = load_articles(corpus_file)
        print(f"✓ Loaded control corpus: {len(articles)} articles")
    
    # Real corpus
    elif corpus_type == 'real':
        # Try multiple possible filenames
        possible_files = [
            DATA_DIR / 'scraped_articles.jsonl',
            DATA_DIR / 'real_corpus.jsonl',
            DATA_DIR / 'articles.jsonl',
            DATA_DIR / 'corpus.jsonl'
        ]
        
        corpus_file = None
        for f in possible_files:
            if f.exists():
                corpus_file = f
                break
        
        if corpus_file is None:
            raise FileNotFoundError(
                f"Real corpus not found. Tried:\n" +
                "\n".join([f"  - {f}" for f in possible_files])
            )
        
        articles = load_articles(corpus_file)
        print(f"✓ Loaded REAL corpus: {len(articles)} articles from {corpus_file.name}")
    
    # Temporal combined (for non-batch mode)
    elif corpus_type == 'temporal':
        corpus_file = DATA_DIR / 'temporal_combined.jsonl'
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Temporal combined corpus not found: {corpus_file}\n"
                f"For batch processing, use --batch-temporal flag"
            )
        articles = load_articles(corpus_file)
        print(f"✓ Loaded TEMPORAL corpus: {len(articles)} articles")
    
    else:
        raise ValueError(
            f"Unknown corpus type: {corpus_type}\n"
            f"Valid types: real, temporal, control, control_constant, control_shuffled, control_random"
        )
    
    # Apply limit if specified
    if limit is not None:
        articles = articles[:limit]
        print(f"  ⚡ LIMITED to {len(articles)} articles for testing")
    
    return articles


def print_experiment_header(mode_name, mode_config):
    """Print experiment header"""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {mode_name}")
    print("="*70)
    print(f"Description: {mode_config['description']}")
    print(f"Contrastive: {mode_config.get('use_contrastive', False)}")
    print(f"PCA Removal: {mode_config.get('use_pca_removal', False)}")
    print(f"Shared PCA: {mode_config.get('shared_pca', False)}")
    print(f"Kernel Type: {mode_config.get('kernel_type', 'rbf')}")
    print("="*70)


# =========================================================================
# EXPERIMENT RUNNERS
# =========================================================================

def run_standard_experiment(articles, mode_config, seeds, corpus_name='real'):
    """Run standard experiment (single kernel type, multiple seeds)"""
    print_experiment_header(mode_config['name'], mode_config)
    
    # Prepare config for pipeline
    pipeline_config = {
        'use_contrastive': mode_config.get('use_contrastive', False),
        'use_pca_removal': mode_config.get('use_pca_removal', False),
        'kernel_type': mode_config.get('kernel_type', 'rbf'),
        'kernel_params': {},
        'use_multi_framing_rks': mode_config.get('use_multi_framing_rks', False),
        'use_gru': mode_config.get('use_gru', True)
    }
    
    # Add kernel types per framing if specified
    if 'kernel_types_per_framing' in mode_config:
        pipeline_config['kernel_types'] = mode_config['kernel_types_per_framing']
    
    # Add kernel-specific params
    if mode_config.get('kernel_type') == 'rq':
        pipeline_config['kernel_params']['alpha'] = mode_config.get('alpha', 1.0)
    
    # Run pipeline
    print(f"\nRunning with seeds: {seeds}")
    print(f"Use GRU: {pipeline_config['use_gru']}")
    print(f"Use MultiFramingRKS: {pipeline_config['use_multi_framing_rks']}")
    
    results = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_contrastive=pipeline_config['use_contrastive'],
        use_pca_removal=pipeline_config['use_pca_removal'],
        shared_pca=mode_config.get('shared_pca', False),
        kernel_type=pipeline_config.get('kernel_type', 'rbf'),
        kernel_types=pipeline_config.get('kernel_types'),
        kernel_params=pipeline_config.get('kernel_params', {}),
        use_gru=pipeline_config['use_gru'],
        use_multi_framing_rks=pipeline_config['use_multi_framing_rks'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Save results
    mode_slug = mode_config['name'].lower().replace(' ', '_').replace('-', '_')
    output_prefix = f"{corpus_name}_{mode_slug}"
    
    for seed in seeds:
        output_path = OUTPUT_DIR / f"{output_prefix}_observer_{seed}.pt"
        torch.save(results[seed], output_path)
        print(f"✓ Saved observer {seed} to {output_path.name}")
    
    return results


def run_multi_kernel_experiment(articles, mode_config, corpus_name='real'):
    """Run multi-kernel experiment (different kernels, same seed)"""
    print_experiment_header(mode_config['name'], mode_config)
    
    kernel_types = mode_config['kernel_types']
    seed = mode_config.get('seed', 42)
    
    results_by_kernel = {}
    
    for ktype in kernel_types:
        print(f"\n{'='*70}")
        print(f"KERNEL: {ktype.upper()}")
        print(f"{'='*70}")
        
        pipeline_config = {
            'use_contrastive': mode_config.get('use_contrastive', True),
            'use_pca_removal': mode_config.get('use_pca_removal', True),
            'kernel_type': ktype,
            'kernel_params': {}
        }
        
        if ktype == 'rq':
            pipeline_config['kernel_params']['alpha'] = 1.0
        
        results = run_multi_observer_experiment(
            articles=articles,
            seeds=[seed],
            use_contrastive=pipeline_config['use_contrastive'],
            use_pca_removal=pipeline_config['use_pca_removal'],
            shared_pca=mode_config.get('shared_pca', True),
            kernel_type=pipeline_config['kernel_type'],
            kernel_params=pipeline_config.get('kernel_params', {}),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        results_by_kernel[ktype] = results
        
        output_path = OUTPUT_DIR / f"{corpus_name}_multikernel_{ktype}_seed{seed}.pt"
        torch.save(results[seed], output_path)
        print(f"✓ Saved {ktype} observer to {output_path.name}")
    
    return results_by_kernel


def run_sigma_sweep_experiment(articles, mode_config, corpus_name='real'):
    """Run sigma sweep experiment (different bandwidths, same seed)"""
    print_experiment_header(mode_config['name'], mode_config)
    
    sigmas = mode_config['sigmas']
    seed = mode_config.get('seed', 42)
    
    for sigma in sigmas:
        print(f"\n{'='*70}")
        print(f"SIGMA: {sigma}")
        print(f"{'='*70}")
        
        pipeline_config = {
            'use_contrastive': mode_config.get('use_contrastive', True),
            'use_pca_removal': mode_config.get('use_pca_removal', True),
            'kernel_type': mode_config.get('kernel_type', 'rbf'),
            'kernel_params': {},
            'rks_sigma': sigma
        }
        
        results = run_multi_observer_experiment(
            articles=articles,
            seeds=[seed],
            use_contrastive=pipeline_config['use_contrastive'],
            use_pca_removal=pipeline_config['use_pca_removal'],
            shared_pca=mode_config.get('shared_pca', True),
            kernel_type=pipeline_config['kernel_type'],
            rks_sigma=sigma,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        output_path = OUTPUT_DIR / f"{corpus_name}_sigma{sigma:.1f}_seed{seed}.pt"
        torch.save(results[seed], output_path)
        print(f"✓ Saved σ={sigma} observer to {output_path.name}")


def run_rq_sweep_experiment(articles, mode_config, corpus_name='real'):
    """Run RQ alpha sweep experiment (different smoothness, same seed)"""
    print_experiment_header(mode_config['name'], mode_config)
    
    alphas = mode_config['alphas']
    sigma = mode_config.get('sigma', 0.6)
    seed = mode_config.get('seed', 42)
    
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA: {alpha}")
        print(f"{'='*70}")
        
        pipeline_config = {
            'use_contrastive': mode_config.get('use_contrastive', True),
            'use_pca_removal': mode_config.get('use_pca_removal', True),
            'kernel_type': 'rq',
            'kernel_params': {'alpha': alpha},
            'rks_sigma': sigma
        }
        
        results = run_multi_observer_experiment(
            articles=articles,
            seeds=[seed],
            use_contrastive=pipeline_config['use_contrastive'],
            use_pca_removal=pipeline_config['use_pca_removal'],
            shared_pca=mode_config.get('shared_pca', True),
            kernel_type='rq',
            kernel_params={'alpha': alpha},
            rks_sigma=sigma,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        output_path = OUTPUT_DIR / f"{corpus_name}_rq_alpha{alpha:.1f}_seed{seed}.pt"
        torch.save(results[seed], output_path)
        print(f"✓ Saved α={alpha} observer to {output_path.name}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Belief Transformer Experimental Runner"
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        default='real',
        choices=['real', 'control', 'temporal', 'control_constant', 'control_shuffled', 'control_random'],
        help='Which corpus to use (NEW: separate control types!)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='enhanced',
        choices=list(EXPERIMENT_MODES.keys()),
        help='Experiment mode'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 43, 44, 45, 46],
        help='Observer seeds (for standard modes)'
    )
    
    parser.add_argument(
        '--use-contrastive',
        action='store_true',
        help='Force contrastive NLI (overrides mode)'
    )
    
    parser.add_argument(
        '--use-pca-removal',
        action='store_true',
        help='Force PCA removal (overrides mode)'
    )
    
    parser.add_argument(
        '--queries-config',
        type=str,
        default=None,
        help='Path to contrastive queries config'
    )
    
    parser.add_argument(
        '--compare-modes',
        action='store_true',
        help='Run all modes and compare'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of articles (for testing)'
    )
    
    parser.add_argument(
        '--batch-temporal',
        action='store_true',
        help='Process temporal batches separately'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*70)
    print("BELIEF TRANSFORMER EXPERIMENTAL RUNNER")
    print("="*70)
    print(f"Corpus: {args.corpus}")
    print(f"Mode: {args.mode}")
    print(f"Seeds: {args.seeds}")
    if args.limit:
        print(f"Limit: {args.limit} articles")
    print("="*70)
    
    # Get mode config
    mode_config = EXPERIMENT_MODES[args.mode].copy()
    
    # Apply overrides
    if args.use_contrastive:
        mode_config['use_contrastive'] = True
    if args.use_pca_removal:
        mode_config['use_pca_removal'] = True
    
    # BATCH TEMPORAL PROCESSING
    if args.batch_temporal and args.corpus == 'temporal':
        print("\n" + "="*70)
        print("🕐 BATCH PROCESSING MODE - Temporal Evolution")
        print("="*70)
        
        temporal_dir = DATA_DIR / 'temporal_cleaned'
        
        if not temporal_dir.exists():
            raise FileNotFoundError(f"Temporal directory not found: {temporal_dir}")
        
        batch_files = sorted(temporal_dir.glob('*.jsonl'))
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {temporal_dir}")
        
        print(f"Found {len(batch_files)} temporal batches\n")
        
        for i, batch_file in enumerate(batch_files):
            # Extract date range from filename
            date_range = batch_file.stem
            
            print("="*70)
            print(f"BATCH {i+1}/{len(batch_files)}: {date_range}")
            print("="*70)
            
            # Load batch
            batch_articles = load_articles(batch_file)
            
            if args.limit:
                batch_articles = batch_articles[:args.limit]
            
            print(f"Articles: {len(batch_articles)}")
            
            # Run experiment
            corpus_name = f"temporal_{date_range}"
            
            if 'kernel_types' in mode_config:
                run_multi_kernel_experiment(batch_articles, mode_config, corpus_name)
            elif 'sigmas' in mode_config:
                run_sigma_sweep_experiment(batch_articles, mode_config, corpus_name)
            elif 'alphas' in mode_config:
                run_rq_sweep_experiment(batch_articles, mode_config, corpus_name)
            else:
                run_standard_experiment(batch_articles, mode_config, args.seeds, corpus_name)
            
            print(f"\n✓ Batch {date_range} complete\n")
        
        print("="*70)
        print("✓ ALL TEMPORAL BATCHES COMPLETE")
        print("="*70)
        return
    
    # NORMAL PROCESSING
    articles = load_corpus(args.corpus, limit=args.limit)
    
    # Run appropriate experiment
    if 'kernel_types' in mode_config:
        run_multi_kernel_experiment(articles, mode_config, args.corpus)
    elif 'sigmas' in mode_config:
        run_sigma_sweep_experiment(articles, mode_config, args.corpus)
    elif 'alphas' in mode_config:
        run_rq_sweep_experiment(articles, mode_config, args.corpus)
    else:
        run_standard_experiment(articles, mode_config, args.seeds, args.corpus)
    
    print("\n" + "="*70)
    print("✓ EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    # Control-specific next steps
    if args.corpus.startswith('control_'):
        print(f"\nNext steps:")
        print(f"  1. Run other controls if not done:")
        print(f"     python run_experiments.py --corpus control_constant --mode {args.mode} --seeds {' '.join(map(str, args.seeds))}")
        print(f"     python run_experiments.py --corpus control_shuffled --mode {args.mode} --seeds {' '.join(map(str, args.seeds))}")
        print(f"     python run_experiments.py --corpus control_random --mode {args.mode} --seeds {' '.join(map(str, args.seeds))}")
        print(f"  2. Compare results:")
        print(f"     python controls/compare_controls.py --mode {args.mode}")
    else:
        print(f"\nNext steps:")
        print(f"  1. Run Procrustes alignment:")
        print(f"     cd analysis && python procrustes_alignment.py")
        print(f"  2. Generate visualization:")
        print(f"     python rks_viz_CLEAN.py")
    
    print("="*70)


if __name__ == "__main__":
    main()