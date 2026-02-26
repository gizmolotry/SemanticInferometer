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

# ============================================================================
# WINDOWS UNICODE FIX - Must be before all other imports
# ============================================================================
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# ============================================================================

from pathlib import Path
import json
import argparse
from datetime import datetime
import torch
import hashlib

# Parsed CLI args (set in main)
args = None

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


def verify_articles(articles, manifest):
    """
    Verify a list of article records against a manifest of expected hashes.

    This helper examines each article's text content, computes an MD5 hash,
    and retains only those whose digest appears in ``manifest['article_hashes']``.
    Articles lacking a textual field are conservatively retained. A summary
    of the verification process is printed to stdout.

    Parameters
    ----------
    articles : list
        Articles loaded from a corpus via ``load_corpus``.
    manifest : dict or None
        Manifest object loaded from JSON. Must contain the key
        ``'article_hashes'`` mapping to a list of hexadecimal digests.

    Returns
    -------
    list
        The filtered list of articles. If no manifest is provided or the
        manifest lacks the ``'article_hashes'`` key, the original
        ``articles`` list is returned unchanged.
    """
    if not manifest or 'article_hashes' not in manifest:
        return articles
    expected_hashes = set(manifest.get('article_hashes', []))
    if not expected_hashes:
        return articles
    verified = []
    discarded = 0
    for article in articles:
        try:
            # Determine the best available textual field for hashing
            if isinstance(article, dict):
                if 'text' in article and isinstance(article['text'], str):
                    body = article['text']
                elif 'content' in article and isinstance(article['content'], str):
                    body = article['content']
                else:
                    # Concatenate all string values as a fallback
                    body_parts = [str(v) for v in article.values() if isinstance(v, str)]
                    body = ''.join(body_parts)
            else:
                # Non-dict articles are retained without verification
                verified.append(article)
                continue
            digest = hashlib.md5(body.encode('utf-8')).hexdigest()
            if digest in expected_hashes:
                verified.append(article)
            else:
                discarded += 1
        except Exception:
            # In case of any unexpected failure, keep the article
            verified.append(article)
    print(f"✓ Verified articles via manifest: kept {len(verified)} / {len(articles)}")
    if discarded:
        print(f"  Discarded {discarded} articles not present in manifest hashes")
    return verified


# =========================================================================
# EXPERIMENT RUNNERS
# =========================================================================

def run_standard_experiment(articles, mode_config, seeds, corpus_name='real', *, output_root: str = None, gru_mode: str = 'intra'):
    """
    Run a standard experiment (single kernel type, multiple seeds).

    Parameters
    ----------
    articles : list
        A list of article records for the experiment.
    mode_config : dict
        Configuration dictionary for the selected experiment mode.
    seeds : list[int]
        List of random seeds to initialise independent observers.
    corpus_name : str, optional
        Name of the corpus being processed. Used in default output paths.
    output_root : str, optional
        If provided, this directory is used as the base output directory for
        saving the results. Each seed will be saved as
        ``Path(output_root) / f'seed_{seed}.pt'``. If ``None`` (default),
        the results will be stored under ``outputs/<corpus_name>/<gru_mode>/``.
    gru_mode : str, optional
        Indicates the GRU aggregation mode (``'intra'`` or ``'inter'``). Used
        only when ``output_root`` is ``None`` to organise the output files.
    """
    print_experiment_header(mode_config['name'], mode_config)
    
    # Prepare config for pipeline
    pipeline_config = {
        'use_contrastive': mode_config.get('use_contrastive', False),
        'use_pca_removal': mode_config.get('use_pca_removal', False),
        'kernel_type': mode_config.get('kernel_type', 'rbf'),
        'kernel_params': {},
        'use_multi_framing_rks': mode_config.get('use_multi_framing_rks', True),  # CHANGED: Default True
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
        device='cuda' if torch.cuda.is_available() else 'cpu',
        track_variance=args.track_variance,  # NEW
        output_dir=Path(args.output_root) if args.output_root else Path('outputs')  # NEW
    )
    
    # Save results
    mode_slug = mode_config['name'].lower().replace(' ', '_').replace('-', '_')
    output_prefix = f"{corpus_name}_{mode_slug}"
    
    for seed in seeds:
        # Determine output path per seed. When neither a custom output_root
        # nor a non-default GRU mode is provided, retain the original
        # naming scheme for backwards compatibility. Otherwise, follow
        # the new structured directory convention.
        if output_root is None and (gru_mode == 'intra' or gru_mode is None):
            # Original behaviour: e.g. outputs/<corpus>_<mode_slug>_observer_<seed>.pt
            output_path = OUTPUT_DIR / f"{output_prefix}_observer_{seed}.pt"
        else:
            if output_root:
                base_dir = Path(output_root)
            else:
                base_dir = Path('outputs') / corpus_name / gru_mode
            output_path = base_dir / f"seed_{seed}.pt"
            # Ensure the parent directories exist before saving
            output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(results[seed], output_path)
        print(f"✓ Saved observer {seed} to {output_path}")
    
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
            device='cuda' if torch.cuda.is_available() else 'cpu',
            track_variance=args.track_variance,  # NEW
            output_dir=Path(args.output_root) if args.output_root else Path('outputs')  # NEW
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
            device='cuda' if torch.cuda.is_available() else 'cpu',
            track_variance=args.track_variance,  # NEW
            output_dir=Path(args.output_root) if args.output_root else Path('outputs')  # NEW
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
            device='cuda' if torch.cuda.is_available() else 'cpu',
            track_variance=args.track_variance,  # NEW
            output_dir=Path(args.output_root) if args.output_root else Path('outputs')  # NEW
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
    
    parser.add_argument(
        '--track-variance',
        action='store_true',
        help='Enable variance tracking at each pipeline stage'
    )

    # ------------------------------------------------------------------
    # NEW ARGUMENTS
    #
    # --gru-mode: Choose GRU inter/intra attention mode. Defaults to intra.
    # --output-root: Optional root for structured experiment outputs.
    # --manifest: Optional path to a manifest JSON file containing
    #             article hashes for verifying control corpora.
    #
    # These arguments are added to support more flexible experiment
    # configuration without breaking backwards compatibility. If
    # unspecified, the runner behaves exactly as before.
    parser.add_argument(
        '--gru-mode',
        type=str,
        choices=['intra', 'inter'],
        default='intra',
        help='GRU aggregation mode (intra or inter)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help='Root directory for structured output files'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default=None,
        help='Path to manifest file (for control corpus verification)'
    )
    
    global args
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load manifest (if provided)
    #
    # If a manifest JSON file is supplied via --manifest, we read it
    # here and optionally use it later to verify that the articles in a
    # control corpus match the expected set of hashes. This is useful for
    # reproducibility and to detect any accidental drift in the control
    # datasets. The manifest is assumed to contain a key called
    # 'article_hashes' which is a list of hex-digests. If this key is
    # missing, the manifest is ignored.
    manifest = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            try:
                manifest = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse manifest JSON: {e}")

    
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
            # Verify batch via manifest if provided
            if manifest:
                batch_articles = verify_articles(batch_articles, manifest)
            
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
                run_standard_experiment(
                    batch_articles,
                    mode_config,
                    args.seeds,
                    corpus_name,
                    output_root=args.output_root,
                    gru_mode=args.gru_mode
                )
            
            print(f"\n✓ Batch {date_range} complete\n")
        
        print("="*70)
        print("✓ ALL TEMPORAL BATCHES COMPLETE")
        print("="*70)
        return
    
    # NORMAL PROCESSING
    articles = load_corpus(args.corpus, limit=args.limit)
    # Verify articles via manifest if provided
    if manifest:
        articles = verify_articles(articles, manifest)
    
    # Run appropriate experiment
    if 'kernel_types' in mode_config:
        run_multi_kernel_experiment(articles, mode_config, args.corpus)
    elif 'sigmas' in mode_config:
        run_sigma_sweep_experiment(articles, mode_config, args.corpus)
    elif 'alphas' in mode_config:
        run_rq_sweep_experiment(articles, mode_config, args.corpus)
    else:
        run_standard_experiment(
            articles,
            mode_config,
            args.seeds,
            args.corpus,
            output_root=args.output_root,
            gru_mode=args.gru_mode
        )
    
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