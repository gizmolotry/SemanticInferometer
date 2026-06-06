"""
variant_integrator.py

BELIEF TRANSFORMER INTEGRATION PROTOCOL (v1.2)
==============================================
Implements "Endogenous Stress-Testing" by generating scientific variants
from raw V-Layer inputs.

LEGACY STATUS:
This script is retained for historical variant replay only. It is not the
canonical experiment or ablation entrypoint for ASTER v3.2+, and it still
assumes older raw-embedding contracts in multiple paths. New ablation work
should flow through `core/master_ablation.py`.

CORE LOGIC:
1. Intercepts RAW 768d embeddings (before Kernel distortion).
2. Modifies geometry (Whitening, Phase Space Fusion).
3. Re-Projects through fresh RKS Kernels (SharedRKSBasis).
4. Re-Fuses via Dirichlet Observers.

VARIANTS:
  1. Geometric Whitening (Removes Anisotropy Cone)
  2. Phase Space Fusion (State + Momentum)
  3. Geometry Switch (Gaussian -> Laplacian)
  4. Null Hypothesis (Green vs Purple Probe)
  5. Semantic Walker (Monte Carlo Semantic Exploration)

USAGE:
  python variant_integrator.py --exp-dir experiments_YYYYMMDD_...
"""

import argparse
import torch
import numpy as np
import sys
import json
from pathlib import Path

# =============================================================================
# 0. DEPENDENCY INJECTION & SAFETY
# =============================================================================
print(f"{'='*60}")
print("INITIATING BELIEF INTEGRATION PROTOCOL v1.2")
print(f"{'='*60}")

# Module A: PCA Removal
try:
    from core.pca_removal import remove_first_pc
    print("  Loaded: pca_removal")
except ImportError:
    try:
        from pca_removal import remove_first_pc
        print("  Loaded: pca_removal (root)")
    except ImportError:
        print("  Critical: 'pca_removal.py' not found.")
        sys.exit(1)

# Module B: RKS Kernel (SharedRKSBasis)
try:
    from core.dirichlet_fusion import SharedRKSBasis
    print("  Loaded: SharedRKSBasis")
except ImportError:
    try:
        from core.rks_feature_map import SharedBasis as SharedRKSBasis
        print("  Loaded: SharedBasis (as SharedRKSBasis)")
    except ImportError:
        print("  Critical: SharedRKSBasis not found.")
        sys.exit(1)

# Module C: Dirichlet Fusion
try:
    from core.dirichlet_fusion import DirichletFusion, DirichletFusionConfig
    print("  Loaded: DirichletFusion")
except ImportError:
    print("  Critical: DirichletFusion not found.")
    sys.exit(1)

# Module D: Metric Gradients (The Probe)
try:
    from core.metric_gradients import MetricGradientExtractor, MetricGradientConfig
    HAS_METRIC_GRADS = True
    print("  Loaded: MetricGradientExtractor")
except ImportError:
    HAS_METRIC_GRADS = False
    print("  Warning: MetricGradientExtractor not found. Variant 4 will be skipped.")

# Module E: Semantic Walker (Track 4 - The Path)
try:
    from core.physarum_walk import SemanticWalker
    HAS_SEMANTIC_WALKER = True
    print("  Loaded: SemanticWalker")
except ImportError:
    HAS_SEMANTIC_WALKER = False
    print("  Warning: core.physarum_walk not found. Variant 5 will be skipped.")


# =============================================================================
# UTILITIES
# =============================================================================

def make_basis(input_dim: int, output_dim: int = 2048, kernel_type: str = 'rbf',
               seed: int = 42) -> SharedRKSBasis:
    """Create a SharedRKSBasis and return it (sigma must be estimated from data)."""
    return SharedRKSBasis(
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
        kernel_type=kernel_type,
    )


def project_through_basis(basis: SharedRKSBasis, data: torch.Tensor) -> torch.Tensor:
    """Estimate sigma from data if needed, then project."""
    if basis._sigma is None:
        sigma = basis.estimate_sigma(data)
        basis.set_sigma(sigma)
        print(f"    Estimated sigma = {sigma:.4f}")
    return basis(data)


def dirichlet_fuse(projected: torch.Tensor, seed: int, n_bots: int,
                   alpha: float = 1.0) -> torch.Tensor:
    """
    Simple Dirichlet fusion: sample weights, mix bot RKHS features.

    Args:
        projected: [N, B, D] per-bot RKHS features
        seed: random seed for Dirichlet weights
        n_bots: number of bots (B dimension)
        alpha: Dirichlet concentration

    Returns:
        [N, D] fused features
    """
    gen = torch.Generator().manual_seed(seed)
    alpha_vec = torch.full((n_bots,), alpha)
    dirichlet = torch.distributions.Dirichlet(alpha_vec)
    # Sample one weight vector for this observer
    weights = dirichlet.sample(torch.Size([]), generator=gen if hasattr(dirichlet, '_validate_args') else None)
    # Manual seeded sampling since Dirichlet doesn't accept generator
    torch.manual_seed(seed)
    weights = torch.distributions.Dirichlet(alpha_vec).sample()  # [B]
    # Fuse: weighted sum over bots
    # projected: [N, B, D], weights: [B] -> [N, D]
    fused = torch.einsum('nbd,b->nd', projected, weights)
    return fused


def save_observer_variant(
    output_dir: Path,
    seed: int,
    fused_features: torch.Tensor,
    meta_tag: str,
    original_metadata,
):
    """Saves a re-projected observer in the standard format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"observer_{seed}.pt"

    payload = {
        'seed': seed,
        'embeddings': fused_features,
        'article_metadata': original_metadata,
        'metadata': {
            'processing': meta_tag,
            'generator': 'variant_integrator_v1.2'
        }
    }
    torch.save(payload, out_path)
    print(f"    -> Saved: {out_path.name}")


# =============================================================================
# MODULE A: VARIANT 1 - GEOMETRIC WHITENING
# =============================================================================
def run_variant_whitening(exp_dir: Path, seeds: list):
    print(f"\n[MODULE A] VARIANT 1: GEOMETRIC WHITENING")
    src_path = exp_dir / "v_layer" / "real" / "bot_embeddings.pt"
    tgt_dir = exp_dir / "rbf" / "whitened" / "real"

    if not src_path.exists():
        print(f"  Source Missing: {src_path}")
        return

    print("  1. Loading Raw Ingredients (768d)...")
    raw_data = torch.load(src_path, map_location='cpu', weights_only=False)
    embeddings = raw_data['embeddings']  # [N, 8, 768]
    metadata = raw_data.get('metadata', [])
    N, B, D = embeddings.shape

    print("  2. Removing Anisotropy Cone (PC1)...")
    flat = embeddings.view(-1, D)
    whitened_flat = remove_first_pc(flat)
    whitened = whitened_flat.view(N, B, D)

    print("  3. Re-Projecting through RBF Kernel...")
    basis = make_basis(input_dim=D, output_dim=2048, kernel_type='rbf', seed=42)
    projected = project_through_basis(basis, whitened)  # [N, 8, 2048]

    print(f"  4. Fusing {len(seeds)} Observers...")
    for seed in seeds:
        fused = dirichlet_fuse(projected, seed=seed, n_bots=B)
        save_observer_variant(tgt_dir, seed, fused, "whitened_pc1_removal", metadata)


# =============================================================================
# MODULE B: VARIANT 2 - PHASE SPACE FUSION
# =============================================================================
def run_variant_phase_space(exp_dir: Path, seeds: list):
    print(f"\n[MODULE B] VARIANT 2: PHASE SPACE FUSION")
    src_emb = exp_dir / "v_layer" / "real" / "bot_embeddings.pt"
    src_grad = exp_dir / "gradient" / "real" / "bot_gradients.pt"
    tgt_dir = exp_dir / "rbf" / "phase_space" / "real"

    if not src_emb.exists() or not src_grad.exists():
        print(f"  Skipping: Missing input artifacts.")
        print(f"    Embeddings: {src_emb.exists()}")
        print(f"    Gradients:  {src_grad.exists()}")
        return

    print("  1. Loading State (Embeddings) and Momentum (Gradients)...")
    data_emb = torch.load(src_emb, map_location='cpu', weights_only=False)
    data_grad = torch.load(src_grad, map_location='cpu', weights_only=False)

    emb_tensor = data_emb['embeddings']   # [N, 8, 768]
    grad_tensor = data_grad['gradients']  # [N, 8, 768]

    if emb_tensor.shape != grad_tensor.shape:
        print(f"  Shape Mismatch: {emb_tensor.shape} vs {grad_tensor.shape}")
        return

    print("  2. Constructing Phase Vectors (1536d)...")
    phase_tensor = torch.cat([emb_tensor, grad_tensor], dim=-1)  # [N, 8, 1536]
    N, B, D_phase = phase_tensor.shape

    print(f"  3. Re-Projecting (Input Dim: {D_phase})...")
    basis = make_basis(input_dim=D_phase, output_dim=2048, kernel_type='rbf', seed=42)
    projected = project_through_basis(basis, phase_tensor)

    print(f"  4. Fusing Observers...")
    for seed in seeds:
        fused = dirichlet_fuse(projected, seed=seed, n_bots=B)
        save_observer_variant(tgt_dir, seed, fused, "phase_space_fusion", data_emb.get('metadata', []))


# =============================================================================
# MODULE C: VARIANT 3 - GEOMETRY SWITCH (LAPLACIAN)
# =============================================================================
def run_variant_geometry_switch(exp_dir: Path, seeds: list):
    print(f"\n[MODULE C] VARIANT 3: GEOMETRY SWITCH (LAPLACIAN)")
    src_path = exp_dir / "v_layer" / "real" / "bot_embeddings.pt"
    tgt_dir = exp_dir / "laplacian" / "cls" / "real"

    if not src_path.exists():
        print(f"  Source Missing: {src_path}")
        return

    raw_data = torch.load(src_path, map_location='cpu', weights_only=False)
    embeddings = raw_data['embeddings']
    N, B, D = embeddings.shape

    print("  1. Initializing LAPLACIAN Kernel (Sharp Geometry)...")
    basis = make_basis(input_dim=D, output_dim=2048, kernel_type='laplacian', seed=42)

    print("  2. Projecting & Fusing...")
    projected = project_through_basis(basis, embeddings)

    for seed in seeds:
        fused = dirichlet_fuse(projected, seed=seed, n_bots=B)
        save_observer_variant(tgt_dir, seed, fused, "kernel_laplacian", raw_data.get('metadata', []))


# =============================================================================
# MODULE D: VARIANT 4 - NULL HYPOTHESIS
# =============================================================================
def run_variant_null_hypothesis(exp_dir: Path):
    print(f"\n[MODULE D] VARIANT 4: NULL HYPOTHESIS PROBE")
    if not HAS_METRIC_GRADS:
        print("  Skipping: core.metric_gradients missing.")
        return

    src_path = exp_dir / "v_layer" / "real" / "bot_embeddings.pt"
    tgt_path = exp_dir / "gradient" / "control_null" / "gradient_null.pt"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print("  No source found.")
        return

    data = torch.load(src_path, map_location='cpu', weights_only=False)
    meta = data.get('metadata', [])
    if isinstance(meta, dict):
        meta = [meta]

    texts = [m.get('text', m.get('content', '')) for m in meta]
    texts = [t for t in texts if t][:50]  # Limit to 50 for speed

    if not texts:
        print("  No text found in metadata.")
        return

    print(f"  1. Running Probe on {len(texts)} articles...")
    print("     Anchors: 'Green' vs 'Purple'")

    try:
        null_anchors = {
            "green": "This text is about the color green.",
            "purple": "This text is about the color purple.",
        }
        config = MetricGradientConfig(
            anchors=null_anchors,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        extractor = MetricGradientExtractor(config)

        corrs = []
        for txt in texts:
            res = extractor.compute_tension(txt, "green", "purple")
            corrs.append(res['correlation'])

        stats = {
            'tension_stats': {
                'green_purple': {
                    'mean': float(np.mean(corrs)),
                    'std': float(np.std(corrs)),
                    'raw': corrs,
                }
            },
            'variant': 'null_hypothesis',
        }
        torch.save(stats, tgt_path)
        print(f"  Saved Null Control: {tgt_path.name}")
        print(f"  Mean Noise Correlation: {np.mean(corrs):.4f}")

    except Exception as e:
        print(f"  Probe Failed: {e}")


# =============================================================================
# MODULE E: VARIANT 5 - PHYSARUM EXPLORER (SLIME MOLD)
# =============================================================================
def run_variant_physarum(exp_dir: Path):
    print(f"\n[MODULE E] VARIANT 5: PHYSARUM EXPLORER (SLIME MOLD)")

    src_emb = exp_dir / "v_layer" / "real" / "bot_embeddings.pt"
    src_grad = exp_dir / "gradient" / "real" / "bot_gradients.pt"
    tgt_dir = exp_dir / "physarum" / "trace" / "real"
    tgt_dir.mkdir(parents=True, exist_ok=True)

    if not src_emb.exists() or not src_grad.exists():
        print("  Skipping: Missing artifacts.")
        print(f"    Embeddings: {src_emb.exists()}")
        print(f"    Gradients:  {src_grad.exists()}")
        return
    if not HAS_SEMANTIC_WALKER:
        print("  Skipping: core.physarum_walk missing.")
        return

    d_emb = torch.load(src_emb, map_location='cpu', weights_only=False)
    d_grad = torch.load(src_grad, map_location='cpu', weights_only=False)

    N_sample = min(20, d_emb['embeddings'].shape[0])
    print(f"  1. Releasing Swarm on first {N_sample} articles...")

    embeddings_batch = d_emb['embeddings'][:N_sample]  # [N, 8, 768]
    gradients_batch = d_grad['gradients'][:N_sample]    # [N, 8, 768]

    # Initialize Kernel for Projection (768d fused embedding -> 2048d)
    basis = make_basis(input_dim=768, output_dim=2048, kernel_type='rbf', seed=42)
    # Estimate sigma from the embeddings (use first bot as representative)
    sigma = basis.estimate_sigma(embeddings_batch[:, 0, :])
    basis.set_sigma(sigma)
    print(f"    Estimated sigma = {sigma:.4f}")

    results = []

    for i in range(N_sample):
        emb = embeddings_batch[i]   # [8, 768]
        grad = gradients_batch[i]   # [8, 768]

        # MCMC Simulation
        explorer = SemanticWalker(emb, grad, basis, temperature=0.1)
        weights_path = explorer.run_swarm(n_walkers=50, n_steps=25)

        # Project final positions to RKS space
        final_weights = weights_path[-1]  # [50, 8]
        fused_emb = torch.matmul(final_weights, emb.float())  # [50, 768]
        projected = basis(fused_emb)  # [50, 2048]

        results.append(projected)

    final_tensor = torch.stack(results)  # [N_sample, 50, 2048]

    out_path = tgt_dir / "observer_semantic_walker.pt"
    torch.save({
        'embeddings': final_tensor,
        'metadata': {
            'processing': 'semantic_walker_mcmc_trace',
            'n_articles': N_sample,
            'n_walkers': 50,
            'n_steps': 25,
            'temperature': 0.1,
            'generator': 'variant_integrator_v1.2',
        }
    }, out_path)

    print(f"  Saved Semantic Walker Trace: {out_path.name} (shape: {final_tensor.shape})")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Belief Transformer Variant Integrator v1.2"
    )
    parser.add_argument("--exp-dir", required=True, help="Root experiment folder")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 420, 4200])
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=['whitening', 'phase_space', 'laplacian', 'null', 'physarum'],
                        help="Variants to skip")
    args = parser.parse_args()

    exp_path = Path(args.exp_dir)
    if not exp_path.exists():
        print(f"Experiment directory not found: {exp_path}")
        sys.exit(1)

    with torch.no_grad():
        if 'whitening' not in args.skip:
            run_variant_whitening(exp_path, args.seeds)
        if 'phase_space' not in args.skip:
            run_variant_phase_space(exp_path, args.seeds)
        if 'laplacian' not in args.skip:
            run_variant_geometry_switch(exp_path, args.seeds)
        if 'null' not in args.skip:
            run_variant_null_hypothesis(exp_path)
        if 'physarum' not in args.skip:
            run_variant_physarum(exp_path)

    print(f"\n{'='*60}")
    print("INTEGRATION COMPLETE")
    print(f"{'='*60}")
    print("Manifest:")
    print(f"  1. Whitened:    {exp_path}/rbf/whitened/real/")
    print(f"  2. PhaseSpace:  {exp_path}/rbf/phase_space/real/")
    print(f"  3. Laplacian:   {exp_path}/laplacian/cls/real/")
    print(f"  4. Null Probe:  {exp_path}/gradient/control_null/")
    print(f"  5. Physarum:    {exp_path}/physarum/trace/real/")
