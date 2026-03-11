#!/usr/bin/env python3
"""
run_full_experiment_suite.py

ONE COMMAND TO RULE THEM ALL

Runs all experiments (Real + 3 Controls)  (N seeds)
Outputs to a timestamped folder with clear structure
Optionally runs an NLI minimal-pair probe after each corpus (synthetic + optional corpus perturbations)
Tracks variance at each pipeline stage (if supported by run_experiments.py)
Generates comparison report automatically

Usage:
    python run_full_experiment_suite.py --limit 500
    python run_full_experiment_suite.py --limit 500 --probe --probe-hypotheses hypotheses.json

Output structure:
    experiments_YYYYMMDD_HHMMSS/
        real/
            observer_42.pt
            observer_43.pt
            ...
            nli_probe_results.json           (if --probe)
        control_constant/
        control_shuffled/
        control_random/
        experiment_manifest.json
        comparison_results.json            (produced by compare_controls.py, if it saves)
"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import sys

# Fix Windows console encoding for Unicode characters (do once at module load)
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Already wrapped

# For alpha sweep (optional - graceful fallback if not available)
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# For gradient channel (Track 3)
try:
    from core.metric_gradients import MetricGradientExtractor, MetricGradientAnalyzer, MetricGradientConfig
    GRADIENT_AVAILABLE = True
except ImportError:
    GRADIENT_AVAILABLE = False


# -----------------------------
# Synthetic Corpus Loading & Validation
# -----------------------------

def load_and_mask_corpus(corpus_path: Path) -> tuple:
    """
    Load synthetic corpus and extract ground truth labels.

    Returns:
        articles: List of article dicts
        ground_truth: Dict mapping article index to perspective label
    """
    articles = []
    ground_truth = {}

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                article = json.loads(line.strip())
                
                # --- FIX: Ensure 'text' field exists ---
                if 'text' not in article and 'content' in article:
                    article['text'] = article['content']
                # ---------------------------------------

                articles.append(article)
                # Extract ground truth from perspective_tag
                if 'perspective_tag' in article:
                    ground_truth[i] = article['perspective_tag']
            except json.JSONDecodeError:
                continue

    return articles, ground_truth


def validate_against_ground_truth(
    result: Dict[str, Any],
    ground_truth: Dict[int, str],
    n_clusters: int = 4,
    use_integrated: bool = True,  # ASTER v3.2: Use Track 5 integrated vectors
) -> Dict[str, Any]:
    """
    Validate pipeline results against ground truth labels.

    Uses KMeans clustering on features and compares to ground truth via NMI/ARI.

    ASTER v3.2: When use_integrated=True, prefers integrated_vectors (Track 5 Hadamard fusion)
    over raw features (Track 2 Dirichlet output) for validation.
    """
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "torch/numpy not available"}

    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    # Get features - prefer integrated vectors (Track 5) over raw features (Track 2)
    features = None
    feature_source = "features"

    if use_integrated and 'integrated_vectors' in result and result['integrated_vectors'] is not None:
        features = result['integrated_vectors']
        feature_source = "integrated_vectors (Track 5)"
    else:
        features = result.get('features')
        feature_source = "features (Track 2)"

    if features is None:
        return {"status": "failed", "error": "No features in result"}

    if hasattr(features, 'numpy'):
        features = features.numpy()

    # Build ground truth labels aligned with features
    n_samples = features.shape[0]
    label_set = sorted(set(ground_truth.values()))
    label_to_idx = {label: i for i, label in enumerate(label_set)}

    true_labels = []
    for i in range(n_samples):
        if i in ground_truth:
            true_labels.append(label_to_idx[ground_truth[i]])
        else:
            true_labels.append(-1)  # Unknown

    true_labels = np.array(true_labels)
    valid_mask = true_labels >= 0

    if valid_mask.sum() < 10:
        return {"status": "failed", "error": "Not enough valid ground truth labels"}

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features)

    # Compute metrics on valid samples
    nmi = normalized_mutual_info_score(true_labels[valid_mask], pred_labels[valid_mask])
    ari = adjusted_rand_score(true_labels[valid_mask], pred_labels[valid_mask])

    print(f"    [Validation] Using {feature_source}, shape={features.shape}, NMI={nmi:.3f}, ARI={ari:.3f}")

    return {
        "status": "success",
        "nmi": float(nmi),
        "ari": float(ari),
        "n_samples": int(n_samples),
        "n_valid": int(valid_mask.sum()),
        "n_clusters": n_clusters,
        "label_set": label_set,
        "feature_source": feature_source,
    }


# -----------------------------
# CRN (Common Random Numbers) for Dirichlet
# -----------------------------

def generate_crn_weights(
    n_bots: int,
    n_observers: int,
    alphas: List[float],
    crn_seed: int,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Pre-generate Dirichlet weights for all alpha values.
    
    This ensures exact reproducibility across conditions (real vs control).
    The same weights are used for each alpha, making "difference" meaningful.
    """
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "torch not available"}
    
    torch.manual_seed(crn_seed)
    
    weights_by_alpha = {}
    for alpha in alphas:
        alpha_vec = torch.full((n_bots,), alpha)
        dirichlet = torch.distributions.Dirichlet(alpha_vec)
        weights = dirichlet.sample((n_observers,))
        weights_by_alpha[f"alpha_{alpha}"] = weights.numpy().tolist()
    
    crn_data = {
        "n_bots": n_bots,
        "n_observers": n_observers,
        "alphas": alphas,
        "crn_seed": crn_seed,
        "weights": weights_by_alpha,
        "generated_at": datetime.now().isoformat(),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(crn_data, f, indent=2)
    
    print(f"[CRN] Saved Dirichlet weights to {output_path}")
    return {"status": "success", "path": str(output_path), "alphas": alphas}


def run_alpha_sweep(
    cls_embeddings_path: Path,
    alphas: List[float],
    crn_weights_path: Optional[Path],
    output_dir: Path,
    n_observers: int = 50,
    rks_dim: int = 2048,
    basis_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run Dirichlet alpha sweep on pre-extracted CLS embeddings.
    
    This is the O-observer probing:  is the probe distribution parameter.
    """
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "torch not available"}
    
    try:
        # Try to import from the core module
        from core.nli_extraction import (
            DirichletFusion, DirichletConfig,
            compute_gram_matrix, compute_wavelength_energy,
            compute_knn_flip_rate, compute_curvature_stats
        )
    except ImportError:
        return {"status": "skipped", "reason": "core.nli_extraction not available"}
    
    # Load CLS embeddings
    if not cls_embeddings_path.exists():
        return {"status": "failed", "error": f"CLS embeddings not found: {cls_embeddings_path}"}
    
    data = torch.load(cls_embeddings_path, weights_only=False)
    cls_per_bot = data.get('cls_per_bot')
    if cls_per_bot is None:
        return {"status": "failed", "error": "cls_per_bot not found in embeddings file"}
    
    results = {"alphas": {}, "wavelength": {}}
    grams = {}
    
    for alpha in alphas:
        config = DirichletConfig(
            n_bots=cls_per_bot.shape[1] if cls_per_bot.dim() == 3 else 8,
            hidden_dim=cls_per_bot.shape[-1],
            rks_dim=rks_dim,
            n_observers=n_observers,
            alpha=alpha,
            basis_seed=basis_seed,
            crn_enabled=True,
            crn_weights_path=str(crn_weights_path) if crn_weights_path else None,
        )
        
        fusion = DirichletFusion(config)
        output = fusion(cls_per_bot)
        
        # Compute Gram matrix and curvature
        gram = compute_gram_matrix(output['fused'])
        curvature = compute_curvature_stats(output['fused_std'])
        
        grams[alpha] = gram
        
        results["alphas"][f"alpha_{alpha}"] = {
            "curvature": curvature,
            "fused_mean_norm": float(output['fused'].norm(dim=-1).mean().item()),
            "fused_std_mean": float(output['fused_std'].mean().item()),
            "provenance": output['provenance'],
        }
        
        # Save fused embeddings
        alpha_output = output_dir / f"fused_alpha_{alpha}.pt"
        torch.save({
            'fused': output['fused'],
            'fused_std': output['fused_std'],
            'gram': gram,
            'alpha': alpha,
            'provenance': output['provenance'],
        }, alpha_output)
    
    # Compute wavelength metrics between adjacent alphas
    sorted_alphas = sorted(alphas)
    for i in range(len(sorted_alphas) - 1):
        a1, a2 = sorted_alphas[i], sorted_alphas[i+1]
        g1, g2 = grams[a1], grams[a2]
        
        energy = compute_wavelength_energy(g1, g2)
        flip_rate = compute_knn_flip_rate(g1, g2)
        
        results["wavelength"][f"{a1}_to_{a2}"] = {
            "energy": energy,
            "knn_flip_rate": flip_rate,
        }
    
    # Save summary
    summary_path = output_dir / "alpha_sweep_results.json"
    with open(summary_path, 'w') as f:
        # Convert any non-serializable types
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    return {"status": "success", "results": results, "output_dir": str(output_dir)}


# -----------------------------
# Directory / orchestration
# -----------------------------

def create_experiment_directory() -> Path:
    """Create timestamped experiment directory under canonical runs root."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = Path("outputs") / "experiments" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    exp_dir = runs_root / f"experiments_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: Subdirectories are now created dynamically in the kernel/channel/corpus loop
    # No longer pre-creating flat corpus dirs here
    return exp_dir
# -----------------------------
# Resume / idempotency helpers
# -----------------------------

_SUITE_CONFIG_NAME = "suite_config.json"

def _ensure_corpus_dirs(exp_dir: Path) -> None:
    """Legacy function - no longer pre-creates flat dirs."""
    # Subdirectories are now created dynamically: kernel/channel/corpus/
    pass

def _suite_config_path(exp_dir: Path) -> Path:
    return exp_dir / _SUITE_CONFIG_NAME

def _write_suite_config(exp_dir: Path, config: Dict) -> None:
    # Write early so we can resume even if the run crashes mid-way.
    try:
        p = _suite_config_path(exp_dir)
        with p.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception:
        # Never fail the run because config couldn't be written.
        pass

def _load_suite_config(exp_dir: Path) -> Optional[Dict]:
    p = _suite_config_path(exp_dir)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _corpus_done_path(corpus_dir: Path) -> Path:
    return corpus_dir / "_CORPUS_DONE.json"

def _mark_corpus_done(corpus_dir: Path, corpus: str, seeds: List[int], limit: int, mode: str) -> None:
    payload = {
        "corpus": corpus,
        "status": "success",
        "seeds": list(seeds),
        "limit": int(limit),
        "mode": str(mode),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with _corpus_done_path(corpus_dir).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

def _is_corpus_done(corpus_dir: Path, seeds: List[int]) -> bool:
    # Prefer explicit marker.
    m = _corpus_done_path(corpus_dir)
    if m.exists():
        try:
            payload = json.loads(m.read_text(encoding="utf-8"))
            if payload.get("status") == "success":
                # If seeds mismatch, be conservative and re-run.
                done_seeds = payload.get("seeds")
                if isinstance(done_seeds, list) and sorted(done_seeds) == sorted(list(seeds)):
                    return True
        except Exception:
            pass

    # Fallback heuristic: all observer_*.pt exist.
    for s in seeds:
        if not (corpus_dir / f"observer_{s}.pt").exists():
            return False
    return True

def _safe_standardize_seed_file(source_file: Path, dest_file: Path) -> None:
    '''
    Windows-safe, idempotent rename/move:
    - Never crashes if dest exists.
    - If dest exists, archive source as dest__dupN.ext (or delete if redundant).
    '''
    if source_file == dest_file:
        return

    if dest_file.exists():
        # If redundant (same size), delete source.
        try:
            if source_file.exists() and source_file.stat().st_size == dest_file.stat().st_size:
                source_file.unlink()
                print(f"  Warning: {dest_file.name} exists; deleted redundant {source_file.name}")
                return
        except Exception:
            pass

        # Otherwise archive as a unique name.
        i = 1
        archived = dest_file.with_name(f"{dest_file.stem}__dup{i}{dest_file.suffix}")
        while archived.exists():
            i += 1
            archived = dest_file.with_name(f"{dest_file.stem}__dup{i}{dest_file.suffix}")
        source_file.replace(archived)
        print(f"  Warning: {dest_file.name} exists; archived {source_file.name}  {archived.name}")
        return

    # Normal move (replace is Windows-friendly and overwrites only if target doesn't exist here)
    source_file.replace(dest_file)

def _pick_latest_experiments_dir() -> Optional[Path]:
    # Pick newest experiments_YYYYMMDD_HHMMSS folder by name (lexicographic matches time format).
    candidates = [p for p in Path(".").iterdir() if p.is_dir() and p.name.startswith("experiments_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]

def _should_resume(exp_dir: Path, seeds: List[int]) -> bool:
    # Resume if at least one corpus is done but not all.
    corpora = ["real", "control_constant", "control_shuffled", "control_random"]
    done = [c for c in corpora if _is_corpus_done(exp_dir / c, seeds)]
    return (len(done) > 0) and (len(done) < len(corpora))



def run_single_corpus(
    corpus: str,
    seeds: List[int],
    limit: int,
    output_dir: Path,
    mode: str = "enhanced",
    track_variance: bool = True,
    kernel_type: str = None,
    nli_cache_path: str = None,
    extra_flags: List[str] = None
) -> Dict:
    """Run experiments for one corpus via run_experiments.py, then move per-seed output files into output_dir."""

    kernel_info = f" kernel={kernel_type}" if kernel_type else ""
    cache_info = " [CACHED]" if nli_cache_path and Path(nli_cache_path).exists() else ""
    print(f"\n{'='*80}")
    print(f"RUNNING: {corpus.upper()} (mode: {mode}{kernel_info}){cache_info}")
    print(f"{'='*80}")

    # Build command - run_experiments.py already has UTF-8 fix built in
    cmd = [
        sys.executable,
        "run_experiments.py",
        "--corpus", corpus,
        "--mode", mode,
        "--seeds",
    ] + [str(s) for s in seeds] + [
        "--limit", str(limit),
        "--output-root", str(output_dir),
    ]

    if track_variance:
        cmd.append("--track-variance")
    
    # Add kernel type override if specified
    if kernel_type:
        cmd.extend(["--kernel-type", kernel_type])
    
    # Add NLI cache path for reuse across kernels (4x speedup)
    if extra_flags:
        cmd.extend(extra_flags)

    if nli_cache_path:
        cmd.extend(["--nli-cache-path", str(nli_cache_path)])
    
    # Set UTF-8 environment for subprocess with unbuffered output
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output for real-time streaming

    print(f"Command: {' '.join(cmd)}")
    print(f"[DEBUG] Starting subprocess for {corpus}...")
    sys.stdout.flush()

    # Run experiment with real-time output streaming (not captured)
    # This shows progress immediately instead of buffering until completion
    try:
        result = subprocess.run(cmd, env=env)
        print(f"[DEBUG] Subprocess completed with return code: {result.returncode}")
    except Exception as e:
        print(f"[DEBUG] Subprocess exception: {e}")
        raise
    sys.stdout.flush()

    if result.returncode != 0:
        print(f" FAILED: {corpus}")
        # Note: stdout/stderr not captured when streaming
        return {
            "corpus": corpus,
            "status": "failed",
            "error": f"Exit code {result.returncode}",
            "stdout": "",
            "returncode": result.returncode,
        }

    print(f"[OK] COMPLETED: {corpus}")

    # Files are saved by complete_pipeline.py directly as observer_{seed}.pt
    found_files: List[str] = []
    missing: List[str] = []

    for seed in seeds:
        expected_file = output_dir / f"observer_{seed}.pt"
        if expected_file.exists():
            found_files.append(str(expected_file))
            print(f"  [OK] Found: observer_{seed}.pt")
        else:
            missing.append(f"observer_{seed}.pt")
            print(f"  [WARN] Missing: observer_{seed}.pt")

    if missing:
        print(f"[WARN] Missing {len(missing)} output files for {corpus}")

    # Mark corpus complete for resume logic (only if all expected outputs exist)
    if not missing:
        _mark_corpus_done(output_dir, corpus=corpus, seeds=seeds, limit=limit, mode=mode)

    return {
        "corpus": corpus,
        "status": "success" if not missing else "partial",
        "seeds": seeds,
        "files": found_files,
        "missing_files": missing,
        "output_dir": str(output_dir),
        "stdout": result.stdout,
        "returncode": result.returncode,
    }


def materialize_baseline_bundle(run_dir: Path, strict: bool = True) -> Dict[str, Any]:
    """
    Emit thesis-facing baseline artifacts for Dash browsing:
      - MONOLITH.html
      - observer_manifest.json
      - observer_<idx>/MONOLITH.html links (or copies)
    """
    run_dir = Path(run_dir)
    target_dir = _resolve_bundle_target_dir(run_dir)
    monolith_out = target_dir / "MONOLITH.html"
    monolith_csv = target_dir / "MONOLITH_DATA.csv"
    if not monolith_csv.exists():
        monolith_ready = _ensure_monolith_csv_ready(target_dir)
        if monolith_ready.get("status") not in {"success", "already_exists"}:
            payload = {
                "status": monolith_ready.get("status", "skipped"),
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
            }
            if "reason" in monolith_ready:
                payload["reason"] = monolith_ready["reason"]
            if "error" in monolith_ready:
                payload["error"] = monolith_ready["error"]
            return payload
    if not monolith_csv.exists():
        return {
            "status": "skipped",
            "reason": f"missing MONOLITH_DATA.csv at {monolith_csv}",
            "run_dir": str(run_dir),
            "target_dir": str(target_dir),
        }
    if _bundle_outputs_are_fresh(target_dir):
        return {
            "status": "skipped",
            "reason": "bundle already fresh",
            "run_dir": str(run_dir),
            "target_dir": str(target_dir),
            "monolith": str(monolith_out),
            "observer_manifest": str(target_dir / "observer_manifest.json"),
            "baseline_meta": str(target_dir / "baseline_meta.json"),
            "baseline_state": str(target_dir / "baseline_state.json"),
            "validation_json": str(target_dir / "validation.json"),
        }

    viz_cmd = [
        sys.executable,
        "-m",
        "analysis.MONOLITH_VIZ",
        str(target_dir),
        "--output",
        str(monolith_out),
        "--mode",
        "synthesis",
    ]
    if strict:
        viz_cmd.append("--strict")

    precompute_cmd = [
        sys.executable,
        "-m",
        "analysis.regression.precompute_observer_artifacts",
        str(target_dir),
        "--variant",
        "MONOLITH.html",
        "--mode",
        "link",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        print(f"[BUNDLE] Rendering MONOLITH baseline for {target_dir}...")
        viz_res = subprocess.run(viz_cmd, env=env)
        if viz_res.returncode != 0:
            return {
                "status": "failed",
                "stage": "monolith_render",
                "returncode": viz_res.returncode,
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
            }

        print(f"[BUNDLE] Materializing observer manifest for {target_dir}...")
        pre_res = subprocess.run(precompute_cmd, env=env)
        if pre_res.returncode != 0:
            return {
                "status": "failed",
                "stage": "observer_manifest",
                "returncode": pre_res.returncode,
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
            }

        print(f"[BUNDLE] Emitting consumer contract bundle for {target_dir}...")
        contract_res = emit_consumer_contract_bundle(target_dir)
        if contract_res.get("status") != "success":
            return {
                "status": "failed",
                "stage": "contract_bundle",
                "error": contract_res.get("error", "unknown contract bundle error"),
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
            }
        missing = _validate_required_bundle_outputs(target_dir)
        if missing:
            return {
                "status": "failed",
                "stage": "post_emit_validation",
                "error": f"missing required bundle outputs: {', '.join(missing)}",
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
            }
    except Exception as exc:
        return {
            "status": "failed",
            "stage": "exception",
            "error": str(exc),
            "run_dir": str(run_dir),
            "target_dir": str(target_dir),
        }

    return {
        "status": "success",
        "run_dir": str(run_dir),
        "target_dir": str(target_dir),
        "monolith": str(monolith_out),
        "observer_manifest": str(target_dir / "observer_manifest.json"),
        "baseline_meta": str(target_dir / "baseline_meta.json"),
        "baseline_state": str(target_dir / "baseline_state.json"),
        "validation_json": str(target_dir / "validation.json"),
    }


def _load_ground_truth_for_corpus(corpus: str) -> Optional[Dict[int, str]]:
    corpus_path = Path(corpus)
    if not corpus_path.exists():
        return None
    try:
        _, ground_truth = load_and_mask_corpus(corpus_path)
    except Exception as exc:
        print(f"[WATERFALL][WARN] Failed to load ground truth for {corpus}: {exc}")
        return None
    return ground_truth or None


def _pick_primary_observer_file(run_dir: Path) -> Optional[Path]:
    observer_files = sorted(run_dir.glob("observer_*.pt"), key=lambda p: p.name)
    return observer_files[0] if observer_files else None


def _write_article_metadata_csv(metadata: List[Dict[str, Any]], output_path: Path) -> bool:
    if not metadata:
        return False
    fieldnames: List[str] = []
    for row in metadata:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            key_str = str(key)
            if key_str not in fieldnames:
                fieldnames.append(key_str)
    if not fieldnames:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in metadata:
            if not isinstance(row, dict):
                continue
            writer.writerow({key: row.get(key) for key in fieldnames})
    return True


def _hydrate_run_leaf_from_observer(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    observer_path = _pick_primary_observer_file(run_dir)
    if observer_path is None:
        return {
            "status": "skipped",
            "reason": f"missing observer_*.pt in {run_dir}",
            "attempted": False,
        }
    if not TORCH_AVAILABLE:
        return {
            "status": "failed",
            "error": "torch not available for observer hydration",
            "attempted": True,
        }

    try:
        observer = torch.load(observer_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"failed loading {observer_path.name}: {exc}",
            "attempted": True,
        }

    written: List[str] = []

    def _save_npy(name: str, key: str) -> None:
        if (run_dir / name).exists():
            return
        value = observer.get(key)
        if value is None:
            return
        np.save(run_dir / name, np.asarray(value))
        written.append(name)

    _save_npy("features.npy", "features")
    _save_npy("walker_work_integrals.npy", "walker_work_integrals")
    _save_npy("spectral_u_axis.npy", "spectral_u_axis")
    _save_npy("spectral_probe_magnitudes.npy", "spectral_probe_magnitudes")

    walker_states_path = run_dir / "walker_states.json"
    if not walker_states_path.exists() and observer.get("walker_states") is not None:
        walker_states_path.write_text(
            json.dumps(observer["walker_states"], indent=2, default=str),
            encoding="utf-8",
        )
        written.append("walker_states.json")

    phantom_path = run_dir / "phantom_verdicts.json"
    if not phantom_path.exists() and observer.get("phantom_verdicts") is not None:
        phantom_path.write_text(
            json.dumps(observer["phantom_verdicts"], indent=2, default=str),
            encoding="utf-8",
        )
        written.append("phantom_verdicts.json")

    metadata = observer.get("article_metadata")
    metadata_json_path = run_dir / "article_metadata.json"
    if not metadata_json_path.exists() and metadata is not None:
        metadata_json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        written.append("article_metadata.json")
    metadata_csv_path = run_dir / "article_metadata.csv"
    if not metadata_csv_path.exists() and isinstance(metadata, list):
        if _write_article_metadata_csv(metadata, metadata_csv_path):
            written.append("article_metadata.csv")

    missing = [
        name for name in [
            "features.npy",
            "walker_work_integrals.npy",
            "walker_states.json",
            "phantom_verdicts.json",
            "article_metadata.csv",
            "spectral_u_axis.npy",
        ]
        if not (run_dir / name).exists()
    ]
    if missing:
        return {
            "status": "failed",
            "error": f"observer hydration incomplete; missing {', '.join(missing)}",
            "attempted": True,
            "observer_path": str(observer_path),
            "written": written,
        }

    return {
        "status": "success",
        "attempted": True,
        "observer_path": str(observer_path),
        "written": written,
    }


def _ensure_monolith_csv_ready(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    if monolith_csv.exists():
        return {"status": "already_exists", "run_dir": str(run_dir)}

    hydrate_result = _hydrate_run_leaf_from_observer(run_dir)
    if hydrate_result.get("status") not in {"success", "already_exists"}:
        return hydrate_result

    try:
        from core.metric_fusion import calculate_unified_metric
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"metric fusion import failed: {exc}",
            "attempted": True,
        }

    try:
        calculate_unified_metric(
            embeddings_path=run_dir / "features.npy",
            gradients_path=run_dir / "spectral_u_axis.npy",
            metadata_path=run_dir / "article_metadata.csv",
            output_path=monolith_csv,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"metric fusion failed: {exc}",
            "attempted": True,
        }

    return {
        "status": "success",
        "run_dir": str(run_dir),
        "monolith_csv": str(monolith_csv),
        "hydration": hydrate_result,
    }


def generate_waterfall_dashboards(
    run_dir: Path,
    ground_truth: Optional[Dict[int, str]] = None,
    projection_method: str = "pca",
) -> Dict[str, Any]:
    """Generate waterfall dashboards for a standard run leaf if checkpoints exist."""
    run_dir = Path(run_dir)
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.exists():
        return {
            "status": "skipped",
            "reason": f"missing checkpoints at {checkpoints_root}",
            "run_dir": str(run_dir),
        }

    checkpoint_dirs = sorted([p for p in checkpoints_root.iterdir() if p.is_dir()])
    if not checkpoint_dirs:
        return {
            "status": "skipped",
            "reason": f"no checkpoint directories under {checkpoints_root}",
            "run_dir": str(run_dir),
        }

    try:
        from analysis.waterfall_viz import run_waterfall_analysis
    except ImportError as exc:
        return {
            "status": "failed",
            "reason": f"waterfall import unavailable: {exc}",
            "run_dir": str(run_dir),
        }

    results: List[Dict[str, Any]] = []
    multiple = len(checkpoint_dirs) > 1
    for checkpoint_dir in checkpoint_dirs:
        output_dir = run_dir / "waterfall_analysis"
        if multiple:
            output_dir = output_dir / checkpoint_dir.name
        try:
            result = run_waterfall_analysis(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                ground_truth=ground_truth,
                projection_method=projection_method,
            )
            results.append(
                {
                    "checkpoint": checkpoint_dir.name,
                    "status": result.get("status", "unknown"),
                    "dashboard_path": result.get("dashboard_path"),
                    "report_path": result.get("report_path"),
                    "metrics_path": result.get("metrics_path"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "checkpoint": checkpoint_dir.name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_path = run_dir / "waterfall_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    success_count = sum(1 for item in results if item.get("status") == "success")
    status = "success" if success_count == len(results) else "partial" if success_count > 0 else "failed"
    payload: Dict[str, Any] = {
        "status": status,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "results": results,
    }
    if len(results) == 1:
        payload.update(results[0])
    return payload


def _resolve_bundle_target_dir(run_dir: Path) -> Path:
    """
    Canonicalize bundle emission to the nearest directory that actually owns MONOLITH_DATA.csv.
    This keeps producer and Dash on the same run leaf even when callers pass a parent directory.
    """
    run_dir = Path(run_dir)
    direct = run_dir / "MONOLITH_DATA.csv"
    if direct.exists():
        return run_dir

    candidates = list(run_dir.rglob("MONOLITH_DATA.csv"))
    if not candidates:
        return run_dir
    # Deterministic: newest CSV wins, then lexical path to break ties.
    candidates.sort(key=lambda p: (-int(p.stat().st_mtime_ns), str(p)))
    return candidates[0].parent


def _validate_required_bundle_outputs(run_dir: Path) -> List[str]:
    required = [
        run_dir / "MONOLITH.html",
        run_dir / "observer_manifest.json",
        run_dir / "baseline_meta.json",
        run_dir / "baseline_state.json",
        run_dir / "validation.json",
    ]
    missing = [p.name for p in required if not p.exists()]
    rel_dir = run_dir / "relativity_cache"
    state_count = len(list(rel_dir.glob("state_*.json"))) if rel_dir.exists() else 0
    delta_count = len(list(rel_dir.glob("delta_*.json"))) if rel_dir.exists() else 0
    if state_count == 0:
        missing.append("relativity_cache/state_*.json")
    if delta_count == 0:
        missing.append("relativity_cache/delta_*.json")
    return missing


def _bundle_outputs_are_fresh(run_dir: Path) -> bool:
    """
    Idempotent bundle guard:
    - required outputs exist
    - outputs are not older than key inputs in the run leaf
    """
    missing = _validate_required_bundle_outputs(run_dir)
    if missing:
        return False

    input_files = [run_dir / "MONOLITH_DATA.csv"]
    for name in ("verification_report.json", "verification_summary.csv"):
        p = run_dir / name
        if p.exists():
            input_files.append(p)
    newest_input = max(int(p.stat().st_mtime_ns) for p in input_files if p.exists())

    output_files = [
        run_dir / "MONOLITH.html",
        run_dir / "observer_manifest.json",
        run_dir / "baseline_meta.json",
        run_dir / "baseline_state.json",
        run_dir / "validation.json",
    ]
    rel_dir = run_dir / "relativity_cache"
    output_files.extend(sorted(rel_dir.glob("state_*.json")))
    output_files.extend(sorted(rel_dir.glob("delta_*.json")))
    oldest_output = min(int(p.stat().st_mtime_ns) for p in output_files if p.exists())
    return oldest_output >= newest_input


def _find_nearby_file(run_dir: Path, filename: str) -> Optional[Path]:
    candidates = [
        run_dir / filename,
        run_dir.parent / filename,
        run_dir.parent.parent / filename if run_dir.parent else None,
        run_dir.parent.parent.parent / filename if run_dir.parent and run_dir.parent.parent else None,
        run_dir.parent.parent / filename if run_dir.parent and run_dir.parent.parent else None,
    ]
    for cand in candidates:
        if cand and cand.exists():
            return cand
    return None


def _copy_if_missing(src: Optional[Path], dst: Path) -> bool:
    if not src or not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _load_monolith_rows(monolith_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not monolith_csv.exists():
        return rows
    with monolith_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for i, row in enumerate(reader):
            ridx = row.get("index", "")
            try:
                idx = int(ridx)
            except Exception:
                idx = i
            rows.append(
                {
                    "index": idx,
                    "bt_uid": row.get("bt_uid", f"article_{idx}"),
                    "title": (row.get("title", "") or "")[:200],
                    "zone": row.get("zone", "unknown"),
                    "density": row.get("density", "0"),
                    "stress": row.get("stress", "0"),
                }
            )
    return rows


def _emit_baseline_state(run_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    manifest_path = run_dir / "observer_manifest.json"
    paths: List[str] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for obs in manifest.get("observers", []):
                rel = str(obs.get("relative_path", "")).strip()
                if rel:
                    paths.append(rel)
        except Exception:
            pass
    if not paths:
        for row in rows:
            paths.append(f"observer_{row['index']}/MONOLITH.html")

    payload = {
        "articles": rows,
        "paths": paths,
        "axes": {"x": "density", "y": "stress"},
        "metrics": {"source": "MONOLITH_DATA.csv"},
    }
    out = run_dir / "baseline_state.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _emit_baseline_meta(run_dir: Path) -> Path:
    report_path = run_dir / "verification_report.json"
    verification_status = "UNVERIFIED"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("global_pass") is True:
                verification_status = "VERIFIED"
            else:
                candidate = str(report.get("status") or report.get("verification_status") or "").upper().strip()
                if candidate in {"VERIFIED", "NON_COMPARABLE", "MISSING_ARTIFACTS", "UNVERIFIED"}:
                    verification_status = candidate
        except Exception:
            pass

    payload = {
        "schema_version": "1.0",
        "cache_version": "1.0",
        "dataset_hash": "suite-generated",
        "code_hash_or_commit": "suite-generated",
        "weights_hash": "suite-generated",
        "kernel_params": {"kernel": "unknown"},
        "rks_dim": 2048,
        "crn_seed": 0,
        "alpha": 1.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": verification_status,
    }
    out = run_dir / "baseline_meta.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _emit_relativity_defaults(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    written_state = 0
    written_delta = 0
    indices = [int(r.get("index", i)) for i, r in enumerate(rows)]
    for idx in indices:
        state_path = rel_dir / f"state_{idx}.json"
        delta_path = rel_dir / f"delta_{idx}.json"
        if not state_path.exists():
            state_payload = {
                "observer_id": idx,
                "articles": rows,
                "paths": [f"observer_{idx}/MONOLITH.html"],
                "axes": {"x": "density", "y": "stress"},
                "metrics": {},
                "provenance": {"source": "suite-default"},
            }
            state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
            written_state += 1
        if not delta_path.exists():
            delta_payload = {
                "observer_id": idx,
                "null_observer_equivalence": {"max_coord_delta": 0.0, "path_flip_count": 0, "axis_rotation_deg": 0.0},
                "path_flip_delta": {},
                "metrics_delta": {"d_rupture_rate": 0.0, "d_mean_work": 0.0, "d_survival_pct": 0.0},
                "axis_delta": {"rotation_deg": 0.0, "d_explained_variance_axis1": 0.0},
            }
            delta_path.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")
            written_delta += 1
    return {"state_files": written_state, "delta_files": written_delta}


def _emit_label_derivatives(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    labels_dir = run_dir / "labels"
    derived_dir = labels_dir / "derived"
    labels_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    hidden_csv = labels_dir / "hidden_groups.csv"
    if not hidden_csv.exists():
        with hidden_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["article_id", "group_topic"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"article_id": int(row.get("index", 0)), "group_topic": row.get("zone", "unknown")})

    counts: Dict[str, int] = {}
    sums: Dict[str, Dict[str, float]] = {}
    for row in rows:
        grp = str(row.get("zone", "unknown"))
        counts[grp] = counts.get(grp, 0) + 1
        if grp not in sums:
            sums[grp] = {"density": 0.0, "stress": 0.0}
        try:
            sums[grp]["density"] += float(row.get("density", 0.0))
        except Exception:
            pass
        try:
            sums[grp]["stress"] += float(row.get("stress", 0.0))
        except Exception:
            pass

    groups = sorted(counts.keys())
    summaries = []
    for g in groups:
        n = max(counts.get(g, 0), 1)
        summaries.append(
            {
                "group_name": g,
                "n_articles": counts.get(g, 0),
                "mean_density": sums[g]["density"] / n,
                "mean_stress": sums[g]["stress"] / n,
            }
        )

    group_summaries = {"groups": summaries}
    (derived_dir / "group_summaries.json").write_text(json.dumps(group_summaries, indent=2), encoding="utf-8")

    matrix = []
    for gi in groups:
        row_vals = []
        for gj in groups:
            if gi == gj:
                row_vals.append(0.0)
            else:
                di = sums[gi]["density"] / max(counts[gi], 1)
                dj = sums[gj]["density"] / max(counts[gj], 1)
                si = sums[gi]["stress"] / max(counts[gi], 1)
                sj = sums[gj]["stress"] / max(counts[gj], 1)
                row_vals.append(abs(di - dj) + abs(si - sj))
        matrix.append(row_vals)

    group_matrix = {"groups": groups, "cost_matrix": matrix}
    (derived_dir / "group_matrix.json").write_text(json.dumps(group_matrix, indent=2), encoding="utf-8")
    return {
        "hidden_groups": str(hidden_csv),
        "group_summaries": str(derived_dir / "group_summaries.json"),
        "group_matrix": str(derived_dir / "group_matrix.json"),
    }


def _emit_validation_json(run_dir: Path) -> Path:
    validation_path = run_dir / "validation.json"
    if validation_path.exists():
        return validation_path

    payload = {
        "nmi": 0.0,
        "ari": 0.0,
        "source": "suite-default",
    }
    validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return validation_path


def emit_consumer_contract_bundle(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    if not monolith_csv.exists():
        return {"status": "failed", "error": f"missing MONOLITH_DATA.csv at {monolith_csv}"}

    # Verification artifacts are leaf-local by contract. Do not copy from parent
    # directories; inherited reports can misstate leaf verification status.
    copied: List[str] = []

    rows = _load_monolith_rows(monolith_csv)
    baseline_meta = _emit_baseline_meta(run_dir)
    baseline_state = _emit_baseline_state(run_dir, rows)
    validation_json = _emit_validation_json(run_dir)
    rel_stats = _emit_relativity_defaults(run_dir, rows)
    label_paths = _emit_label_derivatives(run_dir, rows)

    return {
        "status": "success",
        "baseline_meta": str(baseline_meta),
        "baseline_state": str(baseline_state),
        "validation_json": str(validation_json),
        "copied": copied,
        "relativity": rel_stats,
        "labels": label_paths,
    }


# -----------------------------
# Gradient Channel Runner (Track 3)
# -----------------------------

def run_gradient_channel(
    corpus: str,
    seeds: List[int],
    limit: int,
    output_dir: Path,
    anchor_pairs: List[tuple] = None,
) -> Dict:
    """
    Run metric gradient analysis for a corpus (Track 3: Sensitivity Analysis).

    Uses bi-encoder approach to compute gradients toward semantic anchors,
    measuring "force vectors" that indicate framing direction.

    Args:
        corpus: Corpus name (real, control_constant, etc.)
        seeds: Random seeds (used for reproducibility in sampling)
        limit: Max articles to process
        output_dir: Where to save results
        anchor_pairs: Pairs of anchors to compute tension between

    Returns:
        Dict with status and results
    """
    if not GRADIENT_AVAILABLE:
        print(f"[WARN] Gradient channel not available (import failed)")
        return {"corpus": corpus, "status": "skipped", "reason": "gradient module not available"}

    if not TORCH_AVAILABLE:
        print(f"[WARN] Gradient channel requires torch")
        return {"corpus": corpus, "status": "skipped", "reason": "torch not available"}

    print(f"\n{'='*80}")
    print(f"GRADIENT ANALYSIS: {corpus.upper()}")
    print(f"{'='*80}")

    # Default anchor pairs (semantic tension axes)
    if anchor_pairs is None:
        anchor_pairs = [
            ('victim', 'aggressor'),
            ('emotional', 'neutral'),
            ('humanitarian', 'security'),
            ('conflict', 'peace'),
        ]

    # Load articles using run_experiments.py's load_corpus function
    # Import here to avoid circular imports
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_experiments", Path(__file__).parent / "run_experiments.py")
    run_exp_module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(run_exp_module)
        articles = run_exp_module.load_corpus(corpus, limit=limit)
        print(f"  Loaded {len(articles)} articles from {corpus}")
    except Exception as e:
        print(f"  [ERROR] Failed to load corpus: {e}")
        import traceback
        traceback.print_exc()
        return {"corpus": corpus, "status": "failed", "error": str(e)}

    # Extract article texts
    article_texts = []
    article_ids = []
    for i, art in enumerate(articles):
        if isinstance(art, dict):
            text = art.get('text') or art.get('content') or art.get('body') or ''
            aid = art.get('id') or art.get('article_id') or f"article_{i}"
        else:
            text = str(art)
            aid = f"article_{i}"
        article_texts.append(text)
        article_ids.append(aid)

    # Initialize extractor
    config = MetricGradientConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    analyzer = MetricGradientAnalyzer(MetricGradientExtractor(config))

    print(f"  Anchor pairs: {anchor_pairs}")
    print(f"  Processing {len(article_texts)} articles...")

    # Run analysis
    try:
        results = analyzer.analyze_corpus(
            articles=article_texts,
            anchor_pairs=anchor_pairs,
            article_ids=article_ids,
            verbose=True,
        )
    except Exception as e:
        print(f"  [ERROR] Gradient analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"corpus": corpus, "status": "failed", "error": str(e)}

    # Save results for each seed (gradient analysis is deterministic, but we save
    # per-seed for consistency with other channels)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        output_file = output_dir / f"gradient_{seed}.pt"

        # Save as torch file for consistency
        save_data = {
            'corpus': corpus,
            'seed': seed,
            'n_articles': results['n_articles'],
            'anchor_pairs': results['anchor_pairs'],
            'tension_stats': results['tension_stats'],
            'gradient_variance': results['gradient_variance'],
            'per_article': results['per_article'],
            'timestamp': results['timestamp'],
        }

        torch.save(save_data, output_file)
        print(f"  [OK] Saved: gradient_{seed}.pt")

    # Also save human-readable JSON
    json_file = output_dir / "gradient_analysis.json"
    with open(json_file, 'w') as f:
        # Convert numpy arrays to lists for JSON
        json_safe = {
            'corpus': corpus,
            'seeds': seeds,
            'n_articles': results['n_articles'],
            'anchor_pairs': [list(p) for p in results['anchor_pairs']],
            'tension_stats': results['tension_stats'],
            'gradient_variance': results['gradient_variance'],
            'timestamp': results['timestamp'],
        }
        json.dump(json_safe, f, indent=2)
    print(f"  [OK] Saved: gradient_analysis.json")

    # Print summary
    print(f"\n  TENSION SUMMARY:")
    for pair_key, stats in results['tension_stats'].items():
        print(f"    {pair_key}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    _mark_corpus_done(output_dir, corpus=corpus, seeds=seeds, limit=limit, mode="gradient")

    return {
        "corpus": corpus,
        "status": "success",
        "seeds": seeds,
        "n_articles": results['n_articles'],
        "tension_stats": results['tension_stats'],
        "output_dir": str(output_dir),
    }


# -----------------------------
# Synthetic Experiment Runner (ASTER v3.2 Validation)
# -----------------------------

def run_synthetic_experiment_suite(
    output_dir: Path,
    seeds: List[int],
    kernels: List[str],
    n_articles_per_cluster: int = 15,
    n_clusters: int = 4,
    enable_checkpoints: bool = False,
) -> Dict:
    """
    Run synthetic controlled experiment with embedded ground truth labels.

    This validates the full ASTER v3.2 pipeline (Tracks 1-6) on synthetic
    data where ground truth clusters are known, enabling NMI/ARI measurement.

    Args:
        output_dir: Base output directory
        seeds: Random seeds to test
        kernels: Kernel types to test
        n_articles_per_cluster: Articles per ground truth cluster
        n_clusters: Number of ground truth clusters
        enable_checkpoints: Enable waterfall checkpoint saving for forensic debugging

    Returns:
        Dict with results including NMI scores for all runs
    """
    print(f"\n{'='*80}")
    print(f"SYNTHETIC EXPERIMENT (ASTER v3.2 Validation)")
    print(f"{'='*80}")
    print(f"  Kernels: {kernels}")
    print(f"  Seeds: {seeds}")
    print(f"  Waterfall Checkpoints: {'ENABLED' if enable_checkpoints else 'disabled'}")

    # Import pipeline components (functions load_and_mask_corpus and validate_against_ground_truth are defined above)
    try:
        from core.complete_pipeline import initialize_full_pipeline, BeliefTransformerPipeline
        from core.pipeline_config import PipelineRuntimeConfig, DEFAULT_PIPELINE_RUNTIME_CONFIG
    except ImportError as e:
        print(f"[ERROR] Could not import required modules: {e}")
        return {"status": "failed", "error": str(e)}

    # Check for synthetic corpus
    corpus_path = Path("sythgen/high_quality_articles.jsonl")
    if not corpus_path.exists():
        corpus_path = Path("synthetic_corpus.jsonl")
    if not corpus_path.exists():
        print(f"[ERROR] No synthetic corpus found")
        return {"status": "failed", "error": "synthetic corpus not found"}

    # Load and mask corpus
    print(f"\n  Loading corpus from: {corpus_path}")
    articles, ground_truth = load_and_mask_corpus(corpus_path)
    max_articles = n_articles_per_cluster * n_clusters
    if max_articles < len(articles):
        articles = articles[:max_articles]
        ground_truth = {k: v for k, v in ground_truth.items() if k < max_articles}
    print(f"  Loaded {len(articles)} articles")

    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary = {
        "n_runs": 0,
        "successful_runs": 0,
        "nmi_scores": [],
        "ari_scores": [],
        "by_kernel": {},
        "by_seed": {},
    }

    for kernel in kernels:
        summary["by_kernel"][kernel] = {"nmi": [], "ari": []}
        for seed in seeds:
            run_key = f"{kernel}_seed{seed}"
            run_dir = synthetic_dir / run_key
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Running: {run_key}...")

            try:
                # Initialize pipeline components for this kernel/seed combo
                # Full ASTER v3.2 pipeline: CLS views -> Dirichlet Fusion -> RKS -> Spectral Polarity
                # MUST match run_synthetic_experiment.py parameters exactly!
                actual_hidden_dim = 1536  # DeBERTa-v3-large hidden size
                runtime_cfg = PipelineRuntimeConfig(
                    kernel_type=kernel,
                    use_cls_tokens=True,
                    use_dirichlet_fusion=True,
                    dirichlet_rks_dim=512,
                    dirichlet_n_observers=10,
                    dirichlet_alpha=1.0,
                    dirichlet_hidden_dim=actual_hidden_dim,
                    mix_in_rkhs=True,
                    geometry_mode="rks",
                    normalize_features=DEFAULT_PIPELINE_RUNTIME_CONFIG.normalize_features,
                )
                components = initialize_full_pipeline(
                    random_seed=seed,
                    device="cuda" if TORCH_AVAILABLE else "cpu",
                    **runtime_cfg.to_initialize_kwargs(),
                )

                # Wrap in BeliefTransformerPipeline
                pipeline = BeliefTransformerPipeline(
                    components=components,
                    random_seed=seed,
                    enable_provenance=True,
                    provenance_dir=str(run_dir),
                )

                # Run pipeline with checkpoint config
                pipeline_config = {
                    "enable_checkpoints": enable_checkpoints,
                    "checkpoint_dir": str(run_dir),
                    "output_dir": str(run_dir),
                }
                result = pipeline.process_month(
                    articles=articles,
                    month_name=run_key,
                    config=pipeline_config,
                )

                # --- FIX: Save full observer state as .pt file ---
                torch.save(result, run_dir / f"observer_{seed}.pt")
                # -----------------------------------------------

                # Validate against ground truth
                validation = validate_against_ground_truth(
                    result=result,
                    ground_truth=ground_truth,
                )

                # Save artifacts
                np.save(run_dir / "features.npy", result['features'])
                with open(run_dir / "validation.json", 'w') as f:
                    json.dump(validation, f, indent=2)

                # Save additional pipeline outputs
                if 'integrated_vectors' in result:
                    np.save(run_dir / "integrated_vectors.npy", result['integrated_vectors'])
                if 'dirichlet_fused' in result:
                    np.save(run_dir / "dirichlet_fused.npy", result['dirichlet_fused'])
                if 'spectral_evr' in result:
                    np.save(run_dir / "spectral_evr.npy", result['spectral_evr'])
                if 'spectral_probe_magnitudes' in result:
                    np.save(run_dir / "spectral_probe_magnitudes.npy", result['spectral_probe_magnitudes'])
                if 'spectral_dipole_valid' in result:
                    np.save(run_dir / "spectral_dipole_valid.npy", result['spectral_dipole_valid'])
                if 'spectral_u_axis' in result:
                    # Directional spectral axis (Track 1.5 canonical contract)
                    np.save(run_dir / "spectral_u_axis.npy", result['spectral_u_axis'])
                if 'spectral_antagonism' in result:
                    # Scaled force variant, used by wind-field consumers when present.
                    np.save(run_dir / "antagonism.npy", result['spectral_antagonism'])
                if 'logit_confidence' in result:
                    np.save(run_dir / "logit_confidence.npy", result['logit_confidence'])
                if 'dirichlet_fused_std' in result:
                    np.save(run_dir / "dirichlet_fused_std.npy", result['dirichlet_fused_std'])

                # Track 4/5: Walker and Phantom Path Data
                if 'walker_work_integrals' in result:
                    np.save(run_dir / "walker_work_integrals.npy", result['walker_work_integrals'])
                if 'walker_states' in result:
                    with open(run_dir / "walker_states.json", 'w') as wf:
                        json.dump(result['walker_states'], wf, indent=2)
                if 'd_spectral' in result:
                    np.save(run_dir / "d_spectral.npy", result['d_spectral'])
                if 'phantom_verdicts' in result:
                    with open(run_dir / "phantom_verdicts.json", 'w') as pf:
                        json.dump(result['phantom_verdicts'], pf, indent=2, default=str)

                # Article metadata for visualization popups
                if 'article_metadata' in result:
                    with open(run_dir / "article_metadata.json", 'w') as mf:
                        json.dump(result['article_metadata'], mf, indent=2)
                    print(f"    Saved article_metadata.json ({len(result['article_metadata'])} articles)")

                # Petal glyph visualization (Track 1.5)
                if 'spectral_probe_magnitudes' in result and 'spectral_evr' in result:
                    try:
                        from core.viz_engine import render_petal_grid, render_evr_histogram
                        petal_fig = render_petal_grid(
                            probe_magnitudes_batch=result['spectral_probe_magnitudes'][:16],
                            evr_batch=result['spectral_evr'][:16],
                            dipole_valid_batch=result.get('spectral_dipole_valid', np.ones(16, dtype=bool))[:16],
                            article_ids=[str(i) for i in range(min(16, len(result['spectral_evr'])))],
                            title=f"Spectral Petal Glyphs - {run_key}",
                        )
                        if petal_fig is not None:
                            petal_fig.write_html(str(run_dir / "petal_glyphs.html"))
                            print(f"    Saved petal_glyphs.html")

                        evr_fig = render_evr_histogram(
                            evr_batch=result['spectral_evr'],
                            evr_threshold=0.5,
                            title=f"EVR Distribution - {run_key}",
                        )
                        if evr_fig is not None:
                            evr_fig.write_html(str(run_dir / "evr_histogram.html"))
                            print(f"    Saved evr_histogram.html")
                    except Exception as viz_e:
                        print(f"    Visualization failed: {viz_e}")

                # HoTT Sidecar proofs (Track 6)
                if 'hott_proofs' in result:
                    with open(run_dir / "hott_proofs.json", 'w') as hf:
                        json.dump(result['hott_proofs'], hf, indent=2)
                    print(f"    Saved hott_proofs.json ({len(result['hott_proofs'])} proofs)")
                if 'hott_summary' in result:
                    with open(run_dir / "hott_summary.json", 'w') as hs:
                        json.dump(result['hott_summary'], hs, indent=2)

                # Extract metrics (validation.json uses 'nmi'/'ari' keys)
                nmi = validation.get("nmi", validation.get("normalized_mutual_info", 0))
                ari = validation.get("ari", validation.get("adjusted_rand_index", 0))

                summary["n_runs"] += 1
                summary["successful_runs"] += 1
                summary["nmi_scores"].append(nmi)
                summary["ari_scores"].append(ari)
                summary["by_kernel"][kernel]["nmi"].append(nmi)
                summary["by_kernel"][kernel]["ari"].append(ari)

                if seed not in summary["by_seed"]:
                    summary["by_seed"][seed] = {"nmi": [], "ari": []}
                summary["by_seed"][seed]["nmi"].append(nmi)
                summary["by_seed"][seed]["ari"].append(ari)

                all_results.append({
                    "run_key": run_key,
                    "kernel": kernel,
                    "seed": seed,
                    "nmi": nmi,
                    "ari": ari,
                    "status": "success",
                })

                print(f"    NMI: {nmi:.3f}, ARI: {ari:.3f}")

            except Exception as e:
                import traceback
                print(f"    FAILED: {e}")
                traceback.print_exc()
                summary["n_runs"] += 1
                all_results.append({
                    "run_key": run_key,
                    "kernel": kernel,
                    "seed": seed,
                    "status": "failed",
                    "error": str(e),
                })

    # Compute summary statistics
    if summary["nmi_scores"]:
        summary["mean_nmi"] = float(np.mean(summary["nmi_scores"]))
        summary["std_nmi"] = float(np.std(summary["nmi_scores"]))
        summary["mean_ari"] = float(np.mean(summary["ari_scores"]))
        summary["std_ari"] = float(np.std(summary["ari_scores"]))

    # Save summary
    summary_path = synthetic_dir / "synthetic_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "summary": summary,
            "results": all_results,
            "config": {
                "corpus_path": str(corpus_path),
                "n_articles": len(articles),
                "kernels": kernels,
                "seeds": seeds,
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SYNTHETIC EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total runs: {summary['n_runs']}, Successful: {summary['successful_runs']}")
    if "mean_nmi" in summary:
        print(f"  Mean NMI: {summary['mean_nmi']:.3f} (+/- {summary['std_nmi']:.3f})")
        print(f"  Mean ARI: {summary['mean_ari']:.3f} (+/- {summary['std_ari']:.3f})")
        print(f"\n  By Kernel:")
        for k, v in summary["by_kernel"].items():
            if v["nmi"]:
                print(f"    {k}: NMI={np.mean(v['nmi']):.3f}, ARI={np.mean(v['ari']):.3f}")
    print(f"  Results saved to: {summary_path}")

    return {
        "status": "success",
        "summary": summary,
        "results": all_results,
        "output_dir": str(synthetic_dir),
    }


# -----------------------------
# Alpha Stability Analysis (Track 3 - Modern System)
# -----------------------------

def run_alpha_stability_analysis(
    cls_observer_path: Path,
    output_dir: Path,
    alphas: List[float] = None,
    n_dirichlet_samples: int = 50,
    crn_seed: int = 12345,
) -> Dict:
    """
    Run alpha-sweep geometric stability analysis on CLS embeddings.

    This is the "Modern System" Track 3: measures how the article manifold
    deforms as we change the Dirichlet concentration parameter .

    The gradient S/ tells us where geometry "breaks" - high gradient
    means small  change causes large geometric shift (phase transition).

    Args:
        cls_observer_path: Path to observer_*.pt file from CLS channel
        output_dir: Where to save alpha stability results
        alphas: List of alpha values to sweep (default: [0.1, 0.5, 1.0, 5.0, 20.0])
        n_dirichlet_samples: Number of O-observer samples per alpha
        crn_seed: Common random numbers seed for reproducibility

    Returns:
        Dict with stability analysis results
    """
    if not GRADIENT_AVAILABLE:
        print(f"[WARN] Alpha stability requires gradient module")
        return {"status": "skipped", "reason": "gradient module not available"}

    if not TORCH_AVAILABLE:
        print(f"[WARN] Alpha stability requires torch")
        return {"status": "skipped", "reason": "torch not available"}

    if alphas is None:
        alphas = [0.1, 0.5, 1.0, 5.0, 20.0]

    print(f"\n{'='*80}")
    print(f"ALPHA STABILITY ANALYSIS (Track 3 - Modern System)")
    print(f"{'='*80}")
    print(f"  Source: {cls_observer_path}")
    print(f"  Alphas: {alphas}")

    # Load the CLS observer file
    try:
        observer_data = torch.load(cls_observer_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  [ERROR] Failed to load observer file: {e}")
        return {"status": "failed", "error": str(e)}

    # Extract cls_per_bot (bot-level CLS embeddings)
    cls_per_bot = observer_data.get('bot_cls') or observer_data.get('cls_per_bot')

    if cls_per_bot is None:
        # Try to reconstruct from embeddings if bot-level not saved
        print(f"  [WARN] No bot-level CLS found, attempting fallback...")
        embeddings = observer_data.get('embeddings')
        if embeddings is None:
            embeddings = observer_data.get('features')
        if embeddings is not None:
            print(f"  [INFO] Using fused embeddings (less accurate than bot-level)")
            # Fake bot structure: treat as single bot
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings)
            cls_per_bot = embeddings.unsqueeze(1)  # [N, 1, hidden]
        else:
            print(f"  [ERROR] No usable embeddings found in observer file")
            return {"status": "failed", "error": "No cls_per_bot or embeddings in observer file"}

    # Convert to tensor if needed
    if isinstance(cls_per_bot, np.ndarray):
        cls_per_bot = torch.from_numpy(cls_per_bot)

    print(f"  CLS shape: {cls_per_bot.shape}")

    # Import and run the analyzer
    from core.metric_gradients import AlphaStabilityAnalyzer, AlphaStabilityConfig

    config = AlphaStabilityConfig(
        alphas=alphas,
        n_dirichlet_samples=n_dirichlet_samples,
        crn_enabled=True,
        crn_seed=crn_seed,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    analyzer = AlphaStabilityAnalyzer(config)

    try:
        # Get articles list for provenance (if available)
        n_articles = cls_per_bot.shape[0]
        articles = [{"id": f"article_{i}"} for i in range(n_articles)]

        results = analyzer.analyze_corpus(articles, cls_per_bot)
    except Exception as e:
        print(f"  [ERROR] Alpha stability analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as torch file
    stability_file = output_dir / "alpha_stability.pt"
    torch.save({
        'metric_gradients': results['metric_gradients'],
        'config': results['config'],
        'n_articles': results['n_articles'],
        'alphas': alphas,
    }, stability_file)
    print(f"  [OK] Saved: alpha_stability.pt")

    # Save human-readable JSON
    json_file = output_dir / "alpha_stability.json"
    with open(json_file, 'w') as f:
        json_safe = {
            'metric_gradients': results['metric_gradients'],
            'config': results['config'],
            'n_articles': results['n_articles'],
            'alphas': alphas,
        }
        json.dump(json_safe, f, indent=2, default=str)
    print(f"  [OK] Saved: alpha_stability.json")

    # Print summary
    mg = results['metric_gradients']
    print(f"\n  STABILITY SUMMARY:")
    print(f"    Stability Score: {mg.get('stability_score', 0):.4f}")
    print(f"    Mean Tension: {mg.get('mean_tension', 0):.4f}")
    print(f"    Max Tension: {mg.get('max_tension', 0):.4f}")
    print(f"    Critical Interval: {mg.get('critical_interval', 'N/A')}")

    return {
        "status": "success",
        "metric_gradients": results['metric_gradients'],
        "output_dir": str(output_dir),
    }


# -----------------------------
# NLI Probe integration (Option B + optional shuffle probes)
# -----------------------------

def _first_existing(path_candidates: List[Path]) -> Optional[Path]:
    for p in path_candidates:
        if p.exists():
            return p
    return None


def _detect_premises_jsonl(output_dir: Path) -> Optional[Path]:
    """
    Best-effort auto-detection of a JSONL file containing raw premises/texts inside a corpus output folder.
    If none found, the probe will still run synthetic minimal pairs (if entities provided).
    """
    candidates = [
        output_dir / "premises.jsonl",
        output_dir / "articles.jsonl",
        output_dir / "corpus.jsonl",
        output_dir / "texts.jsonl",
        output_dir / "data.jsonl",
    ]
    return _first_existing(candidates)


def run_nli_probe_for_corpus(
    corpus: str,
    output_dir: Path,
    probe_script: str,
    probe_model: str,
    probe_hypotheses_path: Path,
    probe_entities: str,
    probe_n_synth: int,
    probe_corpus_jsonl: Optional[Path],
    probe_corpus_field: str,
    probe_corpus_limit: int,
    probe_max_length: int,
    probe_batch_size: int,
) -> Dict:
    """
    Runs nli_probe.py (or compatible script) and writes results into output_dir/nli_probe_results.json.
    This is intentionally subprocess-based so you can drop it into an existing repo without refactoring imports.
    """
    out_path = output_dir / "nli_probe_results.json"

    if not Path(probe_script).exists():
        raise FileNotFoundError(
            f"Probe script not found: {probe_script}. "
            f"Expected a file path relative to the repo root or an absolute path."
        )

    cmd = [
        sys.executable,
        probe_script,
        "--model", probe_model,
        "--hypotheses", str(probe_hypotheses_path),
        "--out", str(out_path),
        "--n-synth", str(probe_n_synth),
        "--max-length", str(probe_max_length),
        "--batch-size", str(probe_batch_size),
    ]

    if probe_entities.strip():
        cmd.extend(["--entities", probe_entities.strip()])

    if probe_corpus_jsonl is not None:
        cmd.extend([
            "--corpus-jsonl", str(probe_corpus_jsonl),
            "--corpus-field", probe_corpus_field,
            "--corpus-limit", str(probe_corpus_limit),
        ])

    print(f"\n{'-'*80}")
    print(f"NLI PROBE: {corpus}")
    print(f"{'-'*80}")
    print(f"Probe command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode != 0:
        print(f" PROBE FAILED: {corpus}")
        print(f"STDERR: {result.stderr}")
        return {
            "status": "failed",
            "corpus": corpus,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output": str(out_path),
        }

    print(f" PROBE COMPLETED: {corpus}")
    # Keep stdout in the log (useful summary)
    if result.stdout.strip():
        print(result.stdout)

    return {
        "status": "success",
        "corpus": corpus,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": str(out_path),
        "premises_jsonl_used": str(probe_corpus_jsonl) if probe_corpus_jsonl else None,
    }


# -----------------------------
# Reporting
# -----------------------------

def save_manifest(exp_dir: Path, results: List[Dict], config: Dict):
    """Save experiment manifest."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "experiments": results,
        "directory_structure": {
            "real": str(exp_dir / "real"),
            "control_constant": str(exp_dir / "control_constant"),
            "control_shuffled": str(exp_dir / "control_shuffled"),
            "control_random": str(exp_dir / "control_random"),
        }
    }

    manifest_path = exp_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n Saved manifest: {manifest_path}")


def run_post_thesis_sync(run_validation: bool = False) -> None:
    """
    Sync thesis-facing registry/docs after a suite run.

    - Always tries to refresh RESULTS.md from manifest files.
    - Optionally runs thesis artifact validation.
    - Never raises hard exceptions (post-run convenience only).
    """
    print(f"\n{'='*80}")
    print("POST-RUN THESIS SYNC")
    print(f"{'='*80}")

    builder = Path("scripts/build_results_registry.py")
    if builder.exists():
        print("Syncing RESULTS.md registry from manifests...")
        res = subprocess.run(
            [sys.executable, str(builder)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode == 0:
            if res.stdout.strip():
                print(res.stdout.strip())
            else:
                print("[OK] RESULTS.md sync complete.")
        else:
            print("[WARN] RESULTS.md sync failed.")
            if res.stdout.strip():
                print(res.stdout.strip())
            if res.stderr.strip():
                print(res.stderr.strip())
    else:
        print("[WARN] scripts/build_results_registry.py not found; skipping registry sync.")

    if not run_validation:
        return

    validator = Path("scripts/validate_thesis_artifacts.py")
    if validator.exists():
        print("Running thesis artifact validation...")
        res = subprocess.run(
            [sys.executable, str(validator), "--no-registry-sync"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.stdout.strip():
            print(res.stdout.strip())
        if res.returncode != 0 and res.stderr.strip():
            print(res.stderr.strip())
    else:
        print("[WARN] scripts/validate_thesis_artifacts.py not found; skipping validation.")


def run_comparison(exp_dir: Path, seeds: List[int], kernels: List[str] = None, channels: List[str] = None) -> bool:
    """Run comparison analysis via compare_controls.py.

    New directory structure: exp_dir/kernel/channel/corpus/observer_*.pt
    compare_controls.py expects: data_dir/corpus/observer_*.pt
    So we run comparison for each kernel/channel combo.
    """

    print(f"\n{'='*80}")
    print("RUNNING COMPARISON ANALYSIS")
    print(f"{'='*80}")

    kernels = kernels or ['rbf']
    # Only compare embedding channels (logits, cls), not gradient
    embed_channels = [c for c in (channels or ['cls']) if c != 'gradient']

    any_success = False
    for kernel in kernels:
        for channel in embed_channels:
            data_dir = exp_dir / kernel / channel
            if not data_dir.exists() or not (data_dir / 'real').exists():
                continue

            print(f"\n--- Comparison: {kernel}/{channel} ---")

            cmd = [
                sys.executable,
                "compare_controls.py",
                "--data-dir", str(data_dir),
                "--seeds",
            ] + [str(s) for s in seeds]

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[:500]}")
            else:
                any_success = True
                print(f"  Comparison complete for {kernel}/{channel}")
                if result.stdout.strip():
                    for line in result.stdout.strip().split(chr(10)):
                        if any(kw in line.lower() for kw in ["verdict", "real vs", "effect size", "p-value", "loaded", "shape", "separation"]):
                            print(f"  {line}")

    if not any_success:
        print("No comparisons succeeded.")
        return False

    return True


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run complete experiment suite: Real + 3 Controls"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of articles per corpus (default: 500)"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 420, 4200],
        help="Seeds to run (default: 42 420 4200)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing experiment directory (e.g., experiments_20260105_215516)"
    )

    parser.add_argument(
        "--no-variance-tracking",
        action="store_true",
        help="Disable variance tracking (if run_experiments.py supports it)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="enhanced",
        choices=["enhanced", "cls_logits", "cls_logits_no_pca", "cls_logits_paragraph", 
                 "standard", "contrastive", "pca", "shared_pca"],
        help="Experiment mode to run (default: enhanced). Use cls_logits for CLS+Logits stacking mode."
    )
    
    parser.add_argument(
        "--kernels",
        type=str,
        nargs="+",
        default=["rbf", "laplacian", "rq", "imq", "matern"],
        help="Kernel types to run (default: all five). Options: rbf, laplacian, rq, imq, matern"
    )
    
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["logits", "cls"],
        help="Feature channels to extract (default: logits, cls). Options: logits, cls, gradient"
    )
    
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=["real", "control_constant", "control_shuffled", "control_random"],
        help="Corpora to process (default: all five)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable NLI embedding caching (slower but uses less disk space)"
    )
    
    # ---- Alpha Sweep (O-observers) ----
    parser.add_argument(
        "--alpha-sweep",
        action="store_true",
        help="Enable Dirichlet alpha sweep for O-observer analysis (CLS channel only)"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 5.0, 20.0],
        help="Alpha values for Dirichlet sweep (default: 0.1 0.5 1.0 5.0 20.0)"
    )
    parser.add_argument(
        "--dirichlet-n-observers",
        type=int,
        default=50,
        help="Number of Dirichlet observers per alpha (default: 50)"
    )
    parser.add_argument(
        "--dirichlet-rks-dim",
        type=int,
        default=2048,
        help="RKS dimension for Dirichlet fusion (default: 2048)"
    )
    parser.add_argument(
        "--dirichlet-basis-seed",
        type=int,
        default=42,
        help="Seed for RKS basis (M-observer control, default: 42)"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification harness after experiment suite"
    )
    parser.add_argument(
        "--no-post-sync-results",
        action="store_true",
        help="Skip automatic RESULTS.md registry sync after suite completion."
    )
    parser.add_argument(
        "--post-validate-thesis",
        action="store_true",
        help="Also run thesis artifact validator after post-run registry sync."
    )
    parser.add_argument(
        "--crn-seed",
        type=int,
        default=12345,
        help="Seed for Common Random Numbers in Dirichlet sampling (default: 12345)"
    )
    parser.add_argument(
        "--save-crn-weights",
        action="store_true",
        help="Save pre-generated Dirichlet weights for exact reproducibility"
    )
    
    # ---- Atmospheric Annealing Analysis ----
    parser.add_argument(
        "--physarum",
        action="store_true",
        help="Run atmospheric annealing crack/bond analysis after experiments"
    )
    parser.add_argument(
        "--physarum-alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="Alpha values for Physarum sweep (default: 0.1 0.3 0.5 1.0 2.0 5.0 10.0 20.0)"
    )
    parser.add_argument(
        "--physarum-observers",
        type=int,
        default=50,
        help="Observers per alpha in atmospheric annealing analysis (default: 50)"
    )
    parser.add_argument(
        "--physarum-top-k",
        type=int,
        default=100,
        help="Number of top cracks/bonds to store in detail (default: 100)"
    )
    
    # ---- Metric Gradient Lane (SEPARATE from embeddings) ----
    parser.add_argument(
        "--metric-gradients",
        action="store_true",
        help="Run metric gradient analysis (semantic tension mapping) - SEPARATE LANE"
    )
    parser.add_argument(
        "--metric-anchors",
        type=str,
        nargs="+",
        default=['victim', 'aggressor', 'humanitarian', 'security'],
        help="Anchor concepts for metric gradients (default: victim aggressor humanitarian security)"
    )
    parser.add_argument(
        "--metric-pairs",
        type=str,
        nargs="+",
        default=['victim:aggressor', 'humanitarian:security'],
        help="Anchor pairs for tension analysis (format: anchor1:anchor2)"
    )

    # ---- Synthetic Experiment Mode (ASTER v3.2 Validation) ----
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic controlled experiment with embedded ground truth labels"
    )
    parser.add_argument(
        "--synthetic-n-articles",
        type=int,
        default=15,
        help="Number of synthetic articles per cluster (default: 15, total = 4 clusters * N)"
    )
    parser.add_argument(
        "--synthetic-clusters",
        type=int,
        default=4,
        help="Number of ground truth clusters in synthetic data (default: 4)"
    )

    # ---- Waterfall Ablation Mode (Forensic Pipeline Debugging) ----
    parser.add_argument(
        "--waterfall",
        action="store_true",
        help="Enable waterfall checkpoints and generate 4-panel forensic dashboard"
    )
    parser.add_argument(
        "--waterfall-viz-only",
        type=Path,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Generate waterfall visualization from existing checkpoint directory (skip pipeline run)"
    )

    # ---- Probe flags (ALWAYS ON with defaults) ----
    parser.add_argument(
        "--probe-script",
        default="nli_probe.py",
        help="Path to probe script (default: nli_probe.py)"
    )
    parser.add_argument(
        "--probe-model",
        default="roberta-large-mnli",
        help="HF model name for probe (default: roberta-large-mnli)"
    )
    parser.add_argument(
        "--probe-hypotheses",
        default="probe_hypotheses.json",  # DEFAULT to file in root
        help="Path to JSON file containing hypothesis strings (default: probe_hypotheses.json)"
    )
    parser.add_argument(
        "--probe-entities",
        default="Israel,Palestine,Gaza,Hamas,IDF,West Bank,Netanyahu,Fatah",
        help="Comma-separated entities for synthetic minimal pairs (default: Gaza/Israel entities)"
    )
    parser.add_argument(
        "--probe-n-synth",
        type=int,
        default=50,
        help="Pairs per synthetic transform type (default: 50)"
    )
    parser.add_argument(
        "--probe-corpus-jsonl",
        default="",
        help="Optional JSONL file of premises to run shuffle probes on. If omitted, auto-detect within each corpus output dir."
    )
    parser.add_argument(
        "--probe-corpus-field",
        default="text",
        help="Field name in JSONL for premise text (default: text)"
    )
    parser.add_argument(
        "--probe-corpus-limit",
        type=int,
        default=200,
        help="How many premises to sample for shuffle probes (default: 200)"
    )
    parser.add_argument(
        "--probe-max-length",
        type=int,
        default=256,
        help="Tokenizer max_length for probe (default: 256)"
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=16,
        help="Batch size for probe (default: 16)"
    )
    parser.add_argument(
        "--probe-nonfatal",
        action="store_true",
        default=True,  # DEFAULT: Continue even if probe fails
        help="Continue suite even if probe fails (default: True, use --no-probe-nonfatal to make fatal)"
    )
    parser.add_argument(
        "--no-probe-nonfatal",
        action="store_false",
        dest="probe_nonfatal",
        help="Make probe failures fatal (stop suite)"
    )

    args = parser.parse_args()

    # Validate probe files (always try to use probe)
    probe_hyp_path = Path(args.probe_hypotheses)
    probe_script_path = Path(args.probe_script)
    
    probe_enabled = True
    if not probe_hyp_path.exists():
        print(f"  WARNING: Probe hypotheses not found: {probe_hyp_path}")
        print(f"  Probe will be DISABLED")
        probe_enabled = False
    elif not probe_script_path.exists():
        print(f"  WARNING: Probe script not found: {probe_script_path}")
        print(f"  Probe will be DISABLED")
        probe_enabled = False

    # Create or resume experiment directory
    if args.resume:
        exp_dir = Path(args.resume)
        if not exp_dir.exists():
            raise SystemExit(f"ERROR: Resume directory not found: {exp_dir}")
        print(f"\n{'='*80}")
        print(" RESUMING EXPERIMENT SUITE")
        print(f"{'='*80}")
        print(f"Resuming from: {exp_dir.absolute()}")
    else:
        exp_dir = create_experiment_directory()
        print(f"\n{'='*80}")
        print("FULL EXPERIMENT SUITE")
        print(f"{'='*80}")
        print(f"Output directory: {exp_dir.absolute()}")

    # Print config
    print(f"Mode: {args.mode}")
    print(f"Kernels: {args.kernels}")
    print(f"Channels: {args.channels}")
    print(f"Corpora: {args.corpora}")
    print(f"Articles per corpus: {args.limit}")
    print(f"Seeds: {args.seeds}")
    print(f"Variance tracking: {not args.no_variance_tracking}")
    
    # Calculate total experiments
    total_experiments = len(args.kernels) * len(args.channels) * len(args.corpora)
    total_files = total_experiments * len(args.seeds)
    print(f"Total experiment combinations: {total_experiments}")
    print(f"Total observer files: {total_files}")
    
    if probe_enabled:
        print(f"Probe: ENABLED")
        print(f"  Script: {args.probe_script}")
        print(f"  Model: {args.probe_model}")
        print(f"  Hypotheses: {probe_hyp_path}")
        print(f"  Entities: {args.probe_entities or '(auto-detect)'}")
        print(f"  Non-fatal: {args.probe_nonfatal} (suite will {'continue' if args.probe_nonfatal else 'stop'} on probe failure)")
    else:
        print(f"Probe: DISABLED (missing files)")
    print(f"{'='*80}")

    # ============================================================
    # WATERFALL VIZ-ONLY MODE (Generate dashboard from existing checkpoints)
    # ============================================================
    if getattr(args, 'waterfall_viz_only', None):
        print(f"\n*** WATERFALL VISUALIZATION MODE ***")
        checkpoint_dir = Path(args.waterfall_viz_only)
        if not checkpoint_dir.exists():
            print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
            return

        try:
            from analysis.waterfall_viz import run_waterfall_analysis
            waterfall_result = run_waterfall_analysis(
                checkpoint_dir=checkpoint_dir,
                output_dir=checkpoint_dir / "waterfall_analysis",
                projection_method="pca",
            )
            print(f"\nWaterfall dashboard: {waterfall_result.get('dashboard_path', 'N/A')}")
        except Exception as e:
            print(f"Waterfall visualization failed: {e}")
            import traceback
            traceback.print_exc()
        return

    # ============================================================
    # SYNTHETIC EXPERIMENT MODE (ASTER v3.2 Validation)
    # ============================================================
    if getattr(args, 'synthetic', False):
        print(f"\n*** SYNTHETIC EXPERIMENT MODE ***")

        # Enable checkpoints if --waterfall flag is set
        enable_checkpoints = getattr(args, 'waterfall', False)
        if enable_checkpoints:
            print(f"  Waterfall checkpoints: ENABLED")

        synthetic_result = run_synthetic_experiment_suite(
            output_dir=exp_dir,
            seeds=args.seeds,
            kernels=args.kernels,
            n_articles_per_cluster=getattr(args, 'synthetic_n_articles', 15),
            n_clusters=getattr(args, 'synthetic_clusters', 4),
            enable_checkpoints=enable_checkpoints,
        )

        # Generate Waterfall Visualization for each run (if checkpoints enabled)
        if enable_checkpoints and synthetic_result.get("status") == "success":
            print(f"\n{'='*80}")
            print("GENERATING WATERFALL ABLATION DASHBOARDS")
            print(f"{'='*80}")

            try:
                from analysis.waterfall_viz import run_waterfall_analysis, load_waterfall_checkpoints

                synthetic_dir = exp_dir / "synthetic"
                waterfall_results = []

                # Load ground truth for NMI computation
                corpus_path = Path("sythgen/high_quality_articles.jsonl")
                if not corpus_path.exists():
                    corpus_path = Path("synthetic_corpus.jsonl")
                ground_truth = {}
                if corpus_path.exists():
                    _, ground_truth = load_and_mask_corpus(corpus_path)
                    print(f"  Loaded ground truth: {len(ground_truth)} labels")

                # Find all checkpoint directories
                for run_dir in synthetic_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "checkpoints").exists():
                        for ckpt_dir in (run_dir / "checkpoints").iterdir():
                            if ckpt_dir.is_dir():
                                print(f"\n  Processing: {ckpt_dir.name}")
                                try:
                                    result = run_waterfall_analysis(
                                        checkpoint_dir=ckpt_dir,
                                        output_dir=ckpt_dir / "waterfall_analysis",
                                        ground_truth=ground_truth,  # Pass ground truth for NMI
                                        projection_method="pca",
                                    )
                                    waterfall_results.append({
                                        "run": run_dir.name,
                                        "checkpoint": ckpt_dir.name,
                                        "dashboard": result.get("dashboard_path"),
                                        "status": result.get("status"),
                                    })
                                except Exception as wf_e:
                                    print(f"    Waterfall failed: {wf_e}")
                                    waterfall_results.append({
                                        "run": run_dir.name,
                                        "checkpoint": ckpt_dir.name,
                                        "status": "failed",
                                        "error": str(wf_e),
                                    })

                # Save waterfall summary
                waterfall_summary_path = synthetic_dir / "waterfall_summary.json"
                with open(waterfall_summary_path, 'w') as f:
                    json.dump(waterfall_results, f, indent=2)
                print(f"\n  Waterfall summary saved: {waterfall_summary_path}")

            except ImportError as ie:
                print(f"  Waterfall visualization not available: {ie}")
            except Exception as e:
                print(f"  Waterfall generation failed: {e}")
                import traceback
                traceback.print_exc()

        # Save manifest and exit
        manifest = {
            "experiment_type": "synthetic",
            "output_dir": str(exp_dir),
            "timestamp": datetime.now().isoformat(),
            "synthetic_result": synthetic_result,
            "waterfall_enabled": enable_checkpoints,
        }
        with open(exp_dir / "experiment_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*80}")
        print("SYNTHETIC EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {exp_dir.absolute()}")
        if not args.no_post_sync_results:
            run_post_thesis_sync(run_validation=args.post_validate_thesis)
        return

    corpora = args.corpora
    results: List[Dict] = []
    
    #  CHECKPOINT DETECTION - Skip already completed corpora
    completed_corpora = set()
    seeds_to_run = {}  # corpus -> list of remaining seeds
    
    if args.resume:
        print(f"\n{'='*80}")
        print(" SCANNING FOR CHECKPOINTS")
        print(f"{'='*80}")
        
        for corpus in corpora:
            # For modern layout, a corpus is complete only if all kernel/channel combos
            # have observer_{seed}.pt for all requested seeds.
            combo_dirs = []
            for channel in args.channels:
                if channel == "gradient":
                    combo_dirs.append(exp_dir / "gradient" / corpus)
                else:
                    for kernel in args.kernels:
                        combo_dirs.append(exp_dir / kernel / channel / corpus)

            completed_seeds = []
            missing_seeds = []
            for seed in args.seeds:
                expected_name = f"observer_{seed}.pt"
                found_all = all((combo_dir / expected_name).exists() for combo_dir in combo_dirs)
                if found_all:
                    completed_seeds.append(seed)
                else:
                    missing_seeds.append(seed)
            
            if missing_seeds:
                seeds_to_run[corpus] = missing_seeds
                print(f"  {corpus}: PARTIAL ({len(completed_seeds)}/{len(args.seeds)} seeds done)")
                print(f"     Completed: {completed_seeds}")
                print(f"     Remaining: {missing_seeds}")
            else:
                completed_corpora.add(corpus)
                print(f"  {corpus}:  COMPLETE (all {len(args.seeds)} seeds done)")
        
        print(f"{'='*80}")
        
        if completed_corpora == set(corpora):
            print("\n ALL CORPORA COMPLETE! Nothing to resume.")
            if not getattr(args, "verify", False):
                print("Run comparison analysis or start a new experiment.")
                return
            print("Proceeding to verification/reporting as requested by --verify.")
    else:
        # Fresh run - all seeds for all corpora
        for corpus in corpora:
            seeds_to_run[corpus] = args.seeds

    # Helper to get mode for channel
    def get_mode_for_channel(channel: str, base_mode: str) -> str:
        if channel == 'cls':
            return 'cls_logits'
        else:
            return base_mode
    
    # Helper to check if a specific kernel/channel/corpus combo is done
    def check_combo_done(exp_dir: Path, kernel: str, channel: str, corpus: str, seeds: List[int]) -> bool:
        combo_dir = exp_dir / kernel / channel / corpus
        if not combo_dir.exists():
            return False
        for seed in seeds:
            if not (combo_dir / f"observer_{seed}.pt").exists():
                return False
        return True

    # Main experiment loop: OPTIMIZED ORDER for NLI caching
    # Order: corpus -> channel -> kernel
    # This allows NLI embeddings (expensive) to be computed once per (corpus, channel)
    # and reused across all kernels (cheap RKS projection only)
    #
    # NOTE: Gradient channel is handled specially - it doesn't use kernels
    experiment_count = 0

    # Calculate total experiments (gradient doesn't multiply by kernels)
    embedding_channels = [c for c in args.channels if c != 'gradient']
    gradient_channels = [c for c in args.channels if c == 'gradient']
    total_experiments = len(args.kernels) * len(embedding_channels) * len(corpora) + len(gradient_channels) * len(corpora)

    # Create cache directory (unless caching disabled)
    use_cache = not getattr(args, 'no_cache', False)
    cache_dir = exp_dir / ".nli_cache" if use_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nNLI Cache: {cache_dir}")
    else:
        print(f"\nNLI Cache: DISABLED")

    for corpus in corpora:
        for channel in args.channels:
            # GRADIENT CHANNEL: Handle specially (no kernels)
            if channel == 'gradient':
                experiment_count += 1
                combo_key = f"gradient/{corpus}"

                # Check if already complete (for resume)
                gradient_output_dir = exp_dir / "gradient" / corpus
                if args.resume and (gradient_output_dir / "_CORPUS_DONE.json").exists():
                    print(f"\n[{experiment_count}/{total_experiments}] SKIP: {combo_key} (complete)")
                    continue

                print(f"\n{'='*80}")
                print(f"[{experiment_count}/{total_experiments}] {combo_key} [GRADIENT]")
                print(f"{'='*80}")

                result = run_gradient_channel(
                    corpus=corpus,
                    seeds=args.seeds,
                    limit=args.limit,
                    output_dir=gradient_output_dir,
                )

                result["kernel"] = "gradient"
                result["channel"] = "gradient"

                if result["status"] == "failed":
                    results.append(result)
                    print(f"\n Gradient analysis failed: {corpus}")
                    print("Stopping experiment suite.")
                    break

                results.append(result)
                print(f"\n Gradient analysis completed: {corpus}")
                continue  # Skip kernel loop for gradient channel

            # EMBEDDING CHANNELS (logits, cls): Normal processing with kernels
            # NLI cache path for this (corpus, channel) combination
            # All kernels will reuse this cache
            nli_cache_path = (cache_dir / f"{corpus}_{channel}_nli.pt") if cache_dir else None

            for kernel in args.kernels:
                experiment_count += 1
                combo_key = f"{kernel}/{channel}/{corpus}"
                
                # Check if already complete (for resume)
                if args.resume and check_combo_done(exp_dir, kernel, channel, corpus, args.seeds):
                    print(f"\n[{experiment_count}/{total_experiments}] SKIP: {combo_key} (complete)")
                    continue
                
                cache_status = "[CACHE]" if (nli_cache_path and nli_cache_path.exists()) else "[EXTRACT]"
                print(f"\n{'='*80}")
                print(f"[{experiment_count}/{total_experiments}] {combo_key} {cache_status}")
                print(f"{'='*80}")
                
                # Build output directory: exp_dir/kernel/channel/corpus/
                output_dir = exp_dir / kernel / channel / corpus
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get appropriate mode for this channel
                mode = get_mode_for_channel(channel, args.mode)
                
                result = run_single_corpus(
                    corpus=corpus,
                    seeds=args.seeds,
                    limit=args.limit,
                    output_dir=output_dir,
                    mode=mode,
                    track_variance=not args.no_variance_tracking,
                    kernel_type=kernel,
                    nli_cache_path=str(nli_cache_path) if nli_cache_path else None,
                )
                
                result["kernel"] = kernel
                result["channel"] = channel

                # Stop early on failure
                if result["status"] == "failed":
                    results.append(result)
                    print(f"\n Experiment failed: {corpus}")
                    print("Stopping experiment suite.")
                    break

                # Emit baseline + observer manifest artifacts (no online recompute in Dash).
                bundle_result = materialize_baseline_bundle(output_dir, strict=True)
                result["baseline_bundle"] = bundle_result
                if bundle_result.get("status") != "success":
                    print(f"[BUNDLE][WARN] {bundle_result}")
                else:
                    print(f"[BUNDLE][OK] {bundle_result.get('observer_manifest')}")

                if getattr(args, 'waterfall', False):
                    waterfall_result = generate_waterfall_dashboards(
                        output_dir,
                        ground_truth=_load_ground_truth_for_corpus(corpus),
                        projection_method="pca",
                    )
                    result["waterfall"] = waterfall_result
                    if waterfall_result.get("status") in {"success", "partial"}:
                        print(f"[WATERFALL][OK] {waterfall_result.get('summary_path')}")
                    else:
                        print(f"[WATERFALL][WARN] {waterfall_result}")
        
                # Probe (OPTIONAL - only if hypotheses provided)
                if probe_enabled:
                    try:
                        if args.probe_corpus_jsonl:
                            premises_path = Path(args.probe_corpus_jsonl)
                            if not premises_path.exists():
                                raise FileNotFoundError(f"--probe-corpus-jsonl not found: {premises_path}")
                            probe_corpus_jsonl = premises_path
                        else:
                            probe_corpus_jsonl = _detect_premises_jsonl(output_dir)
        
                        probe_result = run_nli_probe_for_corpus(
                            corpus=corpus,
                            output_dir=output_dir,
                            probe_script=args.probe_script,
                            probe_model=args.probe_model,
                            probe_hypotheses_path=probe_hyp_path,
                            probe_entities=args.probe_entities,
                            probe_n_synth=args.probe_n_synth,
                            probe_corpus_jsonl=probe_corpus_jsonl,
                            probe_corpus_field=args.probe_corpus_field,
                            probe_corpus_limit=args.probe_corpus_limit,
                            probe_max_length=args.probe_max_length,
                            probe_batch_size=args.probe_batch_size,
                        )
                        result["probe"] = probe_result
        
                        # If probe failed and nonfatal is off, stop suite
                        if probe_result.get("status") != "success" and not args.probe_nonfatal:
                            results.append(result)
                            print("\n Probe failed and --probe-nonfatal is not set.")
                            print("Stopping experiment suite.")
                            break
        
                    except Exception as e:
                        # Treat probe exceptions as failures unless nonfatal
                        result["probe"] = {
                            "status": "failed",
                            "corpus": corpus,
                            "error": str(e),
                        }
                        if not args.probe_nonfatal:
                            results.append(result)
                            print(f"\n Probe exception: {e}")
                            print("Stopping experiment suite.")
                            break
                        else:
                            print(f" Probe exception (nonfatal): {e}")
        
                # Alpha Stability Analysis (Track 3 Modern - CLS channel only)
                if getattr(args, 'alpha_sweep', False) and channel == 'cls':
                    try:
                        # Find the first observer file for this experiment
                        first_seed = args.seeds[0]
                        observer_path = output_dir / f"observer_{first_seed}.pt"

                        if observer_path.exists():
                            print(f"\n{'-'*80}")
                            print(f"ALPHA STABILITY: {corpus}")
                            print(f"{'-'*80}")

                            stability_output_dir = output_dir / "alpha_stability"
                            stability_result = run_alpha_stability_analysis(
                                cls_observer_path=observer_path,
                                output_dir=stability_output_dir,
                                alphas=getattr(args, 'alphas', [0.1, 0.5, 1.0, 5.0, 20.0]),
                                n_dirichlet_samples=getattr(args, 'dirichlet_n_observers', 50),
                                crn_seed=getattr(args, 'crn_seed', 12345),
                            )

                            result["alpha_stability"] = stability_result

                            if stability_result.get("status") == "success":
                                mg = stability_result.get("metric_gradients", {})
                                print(f"[OK] ALPHA STABILITY COMPLETED: {corpus}")
                                print(f"  Stability Score: {mg.get('stability_score', 0):.4f}")
                            else:
                                print(f"[WARN] Alpha stability skipped/failed: {stability_result.get('reason', 'unknown')}")
                        else:
                            print(f"  [SKIP] No observer file found for alpha stability")
                    except Exception as e:
                        print(f"[WARN] Alpha stability exception (nonfatal): {e}")
                        result["alpha_stability"] = {"status": "failed", "error": str(e)}

                results.append(result)

                # Progress update
                completed_so_far = len([r for r in results if r.get("status") == "success"])
                print(f"\n{'='*80}")
                print(f"PROGRESS: {experiment_count}/{total_experiments} experiments")
                print(f"Successful: {completed_so_far}")
                print(f"{'='*80}\n")
                sys.stdout.flush()

    # Save manifest
    config = {
        "limit": args.limit,
        "seeds": args.seeds,
        "kernels": args.kernels,
        "channels": args.channels,
        "corpora": args.corpora,
        "variance_tracking": not args.no_variance_tracking,
        "structure": "kernel/channel/corpus/observer_SEED.pt",
        "probe": {
            "enabled": True,  # FIXED: Always enabled now
            "script": args.probe_script,
            "model": args.probe_model,
            "hypotheses": str(probe_hyp_path),
            "entities": args.probe_entities,
            "n_synth": args.probe_n_synth,
            "corpus_jsonl": args.probe_corpus_jsonl or "auto",
            "corpus_field": args.probe_corpus_field,
            "corpus_limit": args.probe_corpus_limit,
            "max_length": args.probe_max_length,
            "batch_size": args.probe_batch_size,
            "nonfatal": args.probe_nonfatal,
        }
    }
    save_manifest(exp_dir, results, config)

    # =========================================================================
    # ALPHA SWEEP (O-observer probing) - CLS channel only
    # =========================================================================
    if args.alpha_sweep and 'cls' in args.channels:
        print(f"\n{'='*80}")
        print("ALPHA SWEEP (O-OBSERVER PROBING)")
        print(f"{'='*80}")
        print(f"Alphas: {args.alphas}")
        print(f"N observers per alpha: {args.dirichlet_n_observers}")
        print(f"RKS dimension: {args.dirichlet_rks_dim}")
        print(f"Basis seed (M-observer): {args.dirichlet_basis_seed}")
        print(f"CRN seed: {args.crn_seed}")
        
        # Generate and save CRN weights if requested
        crn_weights_path = None
        if args.save_crn_weights:
            crn_weights_path = exp_dir / "crn_weights.json"
            crn_result = generate_crn_weights(
                n_bots=8,
                n_observers=args.dirichlet_n_observers,
                alphas=args.alphas,
                crn_seed=args.crn_seed,
                output_path=crn_weights_path,
            )
            print(f"CRN weights: {crn_result.get('status')}")
        
        # Run alpha sweep for each corpus (CLS channel only)
        alpha_results = {}
        for corpus in corpora:
            print(f"\n--- Alpha sweep for {corpus} ---")
            
            # Find the CLS embeddings for this corpus
            cls_path = None
            for kernel in args.kernels:
                candidate = exp_dir / kernel / "cls" / corpus / f"observer_{args.seeds[0]}.pt"
                if candidate.exists():
                    cls_path = candidate
                    break
            
            if cls_path is None:
                print(f"  [WARN] No CLS embeddings found for {corpus}, skipping")
                alpha_results[corpus] = {"status": "skipped", "reason": "no CLS embeddings"}
                continue
            
            # Create output directory for alpha sweep results
            sweep_dir = exp_dir / "alpha_sweep" / corpus
            sweep_dir.mkdir(parents=True, exist_ok=True)
            
            # Run the sweep
            sweep_result = run_alpha_sweep(
                cls_embeddings_path=cls_path,
                alphas=args.alphas,
                crn_weights_path=crn_weights_path,
                output_dir=sweep_dir,
                n_observers=args.dirichlet_n_observers,
                rks_dim=args.dirichlet_rks_dim,
                basis_seed=args.dirichlet_basis_seed,
            )
            
            alpha_results[corpus] = sweep_result
            print(f"  Status: {sweep_result.get('status')}")
        
        # Save alpha sweep summary
        alpha_summary_path = exp_dir / "alpha_sweep_summary.json"
        with open(alpha_summary_path, 'w') as f:
            json.dump(alpha_results, f, indent=2, default=str)
        print(f"\nAlpha sweep summary saved to {alpha_summary_path}")

    # =========================================================================
    # ATMOSPHERIC ANNEALING ANALYSIS
    # =========================================================================
    if args.physarum:
        print(f"\n{'='*80}")
        print("ATMOSPHERIC ANNEALING ANALYSIS")
        print(f"{'='*80}")
        print(f"Alphas: {args.physarum_alphas}")
        print(f"Observers per alpha: {args.physarum_observers}")
        print(f"Top-k pairs: {args.physarum_top_k}")
        
        try:
            from dirichlet_fusion import (
                DirichletFusion, DirichletFusionConfig,
                AtmosphericAnnealer, AnnealingResult,
                compare_annealing_results,
                prepare_crack_heatmap_data,
                prepare_alpha_response_curves,
                prepare_network_graph_data,
            )

            # Create annealing output directory (Track 3 atmospheric analysis)
            annealing_dir = exp_dir / "annealing"
            annealing_dir.mkdir(parents=True, exist_ok=True)

            # Initialize fusion with locked basis
            fusion_config = DirichletFusionConfig(
                rks_dim=args.dirichlet_rks_dim,
                n_observers=args.physarum_observers,
                alpha=1.0,  # Will be overridden in analyzer
                basis_seed=args.dirichlet_basis_seed,
            )
            fusion = DirichletFusion(fusion_config)

            # Create analyzer
            analyzer = AtmosphericAnnealer(
                fusion=fusion,
                alphas=args.physarum_alphas,
                n_observers_per_alpha=args.physarum_observers,
                device='cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu',
            )
            
            # Run analysis for each corpus
            annealing_results = {}
            for corpus in corpora:
                print(f"\n--- Atmospheric annealing: {corpus} ---")
                
                # Find CLS embeddings (need bot_rkhs or raw CLS)
                cls_path = None
                for kernel in args.kernels:
                    candidate = exp_dir / kernel / "cls" / corpus / f"observer_{args.seeds[0]}.pt"
                    if candidate.exists():
                        cls_path = candidate
                        break
                
                if cls_path is None:
                    print(f"  [WARN] No embeddings found for {corpus}, skipping")
                    annealing_results[corpus] = {"status": "skipped", "reason": "no embeddings"}
                    continue
                
                # Load embeddings
                artifact = torch.load(cls_path, map_location='cpu', weights_only=False)
                
                # Get bot_rkhs or reconstruct from CLS
                bot_rkhs = artifact.get('bot_rkhs') or artifact.get('rkhs_views')
                
                if bot_rkhs is None:
                    # Need raw CLS - check if we have it
                    # For now, skip if no bot_rkhs
                    print(f"  [WARN] No bot_rkhs in artifact, skipping atmospheric annealing for {corpus}")
                    print(f"  (Re-run with --mode that saves bot_rkhs)")
                    annealing_results[corpus] = {"status": "skipped", "reason": "no bot_rkhs"}
                    continue
                
                # Get canonical IDs if available
                article_ids = artifact.get('ids') or artifact.get('bt_uid_list')

                # Run atmospheric annealing analysis
                analysis = analyzer.analyze(
                    cls_per_bot=bot_rkhs,  # Actually bot_rkhs here
                    corpus_name=corpus,
                    article_ids=article_ids,
                    top_k=args.physarum_top_k,
                    store_full_matrices=True,
                    verbose=True,
                )
                
                # Save analysis
                corpus_annealing_dir = annealing_dir / corpus
                corpus_annealing_dir.mkdir(parents=True, exist_ok=True)

                analysis.save(corpus_annealing_dir / "annealing_analysis.json")
                
                # Save Plotly-ready visualization data
                heatmap_data = prepare_crack_heatmap_data(analysis)
                with open(corpus_annealing_dir / "crack_heatmap_data.json", 'w') as f:
                    json.dump(heatmap_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

                curve_data = prepare_alpha_response_curves(analysis)
                with open(corpus_annealing_dir / "alpha_response_curves.json", 'w') as f:
                    json.dump(curve_data, f, indent=2)

                network_data = prepare_network_graph_data(analysis)
                with open(corpus_annealing_dir / "network_graph_data.json", 'w') as f:
                    json.dump(network_data, f, indent=2)

                annealing_results[corpus] = {
                    "status": "success",
                    "n_articles": analysis.n_articles,
                    "n_cracks": analysis.n_cracks,
                    "n_bonds": analysis.n_bonds,
                    "n_contested": analysis.n_contested,
                    "mean_crack_score": analysis.mean_crack_score,
                    "crack_fraction": analysis.crack_fraction,
                    "output_dir": str(corpus_annealing_dir),
                }

                print(f"  Saved to {corpus_annealing_dir}")
            
            # Run cross-corpus comparison
            print(f"\n--- Annealing Comparison ---")

            # Load analyses for comparison
            loaded_analyses = {}
            for corpus in corpora:
                analysis_path = annealing_dir / corpus / "annealing_analysis.json"
                if analysis_path.exists():
                    loaded_analyses[corpus] = AnnealingResult.load(analysis_path)

            if 'real' in loaded_analyses:
                comparison = compare_annealing_results(
                    real=loaded_analyses.get('real'),
                    shuffled=loaded_analyses.get('control_shuffled'),
                    constant=loaded_analyses.get('control_constant'),
                    random=loaded_analyses.get('control_random'),
                )

                # Save comparison
                comparison_path = annealing_dir / "comparison.json"
                with open(comparison_path, 'w') as f:
                    json.dump(comparison, f, indent=2)

                print(f"\nAnnealing Comparison:")
                print(f"  Verdict: {comparison['verdict']}")
                for detail in comparison.get('ordering_details', []):
                    print(f"    {detail}")

                annealing_results['comparison'] = comparison

            # Save overall annealing summary
            annealing_summary_path = annealing_dir / "annealing_summary.json"
            with open(annealing_summary_path, 'w') as f:
                json.dump(annealing_results, f, indent=2, default=str)

            print(f"\nAnnealing summary saved to {annealing_summary_path}")
            
        except ImportError as e:
            print(f"[WARN] Atmospheric annealing analysis requires dirichlet_fusion.py with AtmosphericAnnealer")
            print(f"  Import error: {e}")
        except Exception as e:
            print(f"[ERROR] Atmospheric annealing analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # PATH B: GRADIENT ANNEALING (The "Force Field")
    # Treats gradient directions AS embeddings - anchors become V-observers
    # =========================================================================
    if args.physarum and args.metric_gradients:
        print(f"\n{'='*80}")
        print("PATH B: GRADIENT TENSION TOPOLOGY (Dual-Path Annealing)")
        print(f"{'='*80}")
        print("Treating Anchor Gradients as V-observers for atmospheric annealing analysis")
        print("This reveals WHERE articles are framed differently despite similar words")
        
        try:
            from metric_gradients import MetricGradientConfig, MetricGradientExtractor
            from dirichlet_fusion import (
                DirichletFusion, DirichletFusionConfig,
                AtmosphericAnnealer, AnnealingResult,
                compare_annealing_results,
            )

            # Helper: Bridge gradients to annealing-compatible tensor
            def extract_gradient_tensor(
                extractor,
                articles: List[str],
                anchors: List[str],
                device: str,
                verbose: bool = True,
            ) -> torch.Tensor:
                """
                Bridge: Converts Metric Gradients into [N, B, H] tensor for atmospheric annealing.
                Here, 'Bots' (B) are replaced by 'Anchors' (framing concepts).
                """
                if verbose:
                    print(f"  Extracting gradients: {len(articles)} articles  {len(anchors)} anchors...")
                
                tensor_list = []
                
                for i, text in enumerate(articles):
                    if verbose and (i + 1) % 50 == 0:
                        print(f"    {i + 1}/{len(articles)}...")
                    
                    # Get gradients for ALL anchors for this article
                    grads_dict = extractor.get_all_gradients(text, anchor_names=anchors)
                    
                    # Stack anchors: [n_anchors, hidden_dim]
                    article_grads = torch.stack([grads_dict[a] for a in anchors])
                    tensor_list.append(article_grads)
                
                # Stack articles: [N, n_anchors, hidden_dim]
                full_tensor = torch.stack(tensor_list)
                return full_tensor.to(device)
            
            # Setup gradient extractor
            gradient_anchors = args.metric_anchors
            print(f"Gradient anchors (V-observers): {gradient_anchors}")
            
            default_anchors = {
                'victim': "This text describes victims and suffering.",
                'aggressor': "This text describes aggression and violence.",
                'neutral': "This text is a neutral factual report.",
                'emotional': "This text is emotionally charged.",
                'humanitarian': "This text focuses on humanitarian concerns.",
                'security': "This text focuses on security threats.",
            }
            anchor_prompts = {k: default_anchors.get(k, f"This text is about {k}.") 
                            for k in gradient_anchors}
            
            grad_config = MetricGradientConfig(
                anchors=anchor_prompts,
                device='cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu',
            )
            extractor = MetricGradientExtractor(grad_config)
            
            # Load article texts (same as metric_gradients section)
            article_texts = []
            article_ids = []
            real_artifact_path = None
            
            for kernel in args.kernels:
                candidate = exp_dir / kernel / "cls" / "real" / f"observer_{args.seeds[0]}.pt"
                if candidate.exists():
                    real_artifact_path = candidate
                    break
            
            if real_artifact_path:
                artifact = torch.load(real_artifact_path, map_location='cpu', weights_only=False)
                metadata = artifact.get('article_metadata', [])
                if metadata:
                    article_texts = [m.get('text', m.get('title', '')) for m in metadata]
                    article_ids = artifact.get('ids', artifact.get('bt_uid_list', []))
            
            if not article_texts:
                print("[WARN] Could not find article texts - skipping gradient annealing")
            else:
                # Limit articles
                n_articles = min(len(article_texts), args.limit)
                article_texts = article_texts[:n_articles]
                article_ids = article_ids[:n_articles] if article_ids else [f"art_{i}" for i in range(n_articles)]
                
                # BRIDGE: Create [N, Anchors, 768] tensor
                print(f"\n--- Extracting gradient tensor ---")
                gradient_tensor = extract_gradient_tensor(
                    extractor,
                    article_texts,
                    gradient_anchors,
                    grad_config.device,
                    verbose=True,
                )
                print(f"Gradient tensor shape: {gradient_tensor.shape}")
                print(f"  = [N={gradient_tensor.shape[0]}, Anchors={gradient_tensor.shape[1]}, H={gradient_tensor.shape[2]}]")
                
                # Initialize Fusion for Gradients (B = n_anchors, not n_bots)
                grad_fusion_config = DirichletFusionConfig(
                    n_bots=len(gradient_anchors),  # Anchors ARE the V-observers now
                    hidden_dim=gradient_tensor.shape[2],
                    rks_dim=args.dirichlet_rks_dim,
                    n_observers=args.physarum_observers,
                    alpha=1.0,
                    basis_seed=args.dirichlet_basis_seed,
                )
                grad_fusion = DirichletFusion(grad_fusion_config)

                # Run atmospheric annealing on Gradients
                print(f"\n--- Running atmospheric annealing on gradient space ---")
                grad_annealer = AtmosphericAnnealer(
                    fusion=grad_fusion,
                    alphas=args.physarum_alphas,
                    n_observers_per_alpha=args.physarum_observers,
                    device=grad_config.device,
                )

                grad_analysis = grad_annealer.analyze(
                    cls_per_bot=gradient_tensor,  # GRADIENTS AS EMBEDDINGS
                    corpus_name="real_gradients",
                    article_ids=article_ids,
                    top_k=args.physarum_top_k,
                    store_full_matrices=True,
                    verbose=True,
                )

                # Save Path B results
                grad_annealing_dir = exp_dir / "physarum" / "gradient_topology"
                grad_annealing_dir.mkdir(parents=True, exist_ok=True)

                grad_analysis.save(grad_annealing_dir / "gradient_annealing_analysis.json")
                
                # Save visualization data
                from dirichlet_fusion import prepare_crack_heatmap_data, prepare_alpha_response_curves

                heatmap_data = prepare_crack_heatmap_data(grad_analysis)
                with open(grad_annealing_dir / "gradient_crack_heatmap.json", 'w') as f:
                    json.dump(heatmap_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

                curve_data = prepare_alpha_response_curves(grad_analysis)
                with open(grad_annealing_dir / "gradient_alpha_curves.json", 'w') as f:
                    json.dump(curve_data, f, indent=2)

                print(f"\nPath B (Gradient Topology) saved to {grad_annealing_dir}")
                print(f"  Cracks: {grad_analysis.n_cracks}")
                print(f"  Bonds: {grad_analysis.n_bonds}")
                print(f"  Contested: {grad_analysis.n_contested}")
                
                # =========================================================
                # INTERFERENCE PATTERN: Compare Path A vs Path B
                # =========================================================
                print(f"\n--- Dual-Path Interference Analysis ---")
                
                # Load Path A results if they exist
                path_a_file = exp_dir / "physarum" / "real" / "annealing_analysis.json"
                if path_a_file.exists():
                    path_a_analysis = AnnealingResult.load(path_a_file)
                    
                    # Find pairs that differ between paths
                    # Path A Bond + Path B Crack = "Wolf in Sheep's Clothing"
                    # (Same words, different framing)
                    
                    interference = {
                        'path_a_cracks': path_a_analysis.n_cracks,
                        'path_a_bonds': path_a_analysis.n_bonds,
                        'path_b_cracks': grad_analysis.n_cracks,
                        'path_b_bonds': grad_analysis.n_bonds,
                    }
                    
                    # Compare crack matrices if both exist
                    if path_a_analysis.crack_matrix is not None and grad_analysis.crack_matrix is not None:
                        a_cracks = path_a_analysis.crack_matrix
                        b_cracks = grad_analysis.crack_matrix
                        
                        # Ensure same size
                        min_n = min(a_cracks.shape[0], b_cracks.shape[0])
                        a_cracks = a_cracks[:min_n, :min_n]
                        b_cracks = b_cracks[:min_n, :min_n]
                        
                        # Wolf detection: Low crack in A (bond) but high crack in B
                        a_threshold = float(a_cracks.median())
                        b_threshold = float(b_cracks.median())
                        
                        wolves = ((a_cracks < a_threshold) & (b_cracks > b_threshold)).sum().item()
                        sheep = ((a_cracks < a_threshold) & (b_cracks < b_threshold)).sum().item()
                        
                        interference['wolves_in_sheeps_clothing'] = int(wolves)
                        interference['genuine_bonds'] = int(sheep)
                        interference['correlation'] = float(torch.corrcoef(
                            torch.stack([a_cracks.flatten(), b_cracks.flatten()])
                        )[0, 1].item())
                        
                        print(f"\n  INTERFERENCE PATTERN:")
                        print(f"    Path A (Embedding) cracks: {path_a_analysis.n_cracks}")
                        print(f"    Path B (Gradient) cracks: {grad_analysis.n_cracks}")
                        print(f"    Crack matrix correlation: {interference['correlation']:.3f}")
                        print(f"    'Wolves in Sheep's Clothing': {wolves} pairs")
                        print(f"    (Same words, different framing)")
                    
                    # Save interference analysis
                    interference_path = exp_dir / "physarum" / "dual_path_interference.json"
                    with open(interference_path, 'w') as f:
                        json.dump(interference, f, indent=2)
                    
                    print(f"\n  Interference analysis saved to {interference_path}")
                else:
                    print("  [WARN] Path A results not found - skipping interference analysis")
        
        except ImportError as e:
            print(f"[WARN] Gradient Annealing requires metric_gradients.py and dirichlet_fusion.py")
            print(f"  Import error: {e}")
        except Exception as e:
            print(f"[ERROR] Gradient Annealing failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # METRIC GRADIENT ANALYSIS (SEPARATE - tension stats only, no Physarum)
    # =========================================================================
    if args.metric_gradients and not args.physarum:
        print(f"\n{'='*80}")
        print("METRIC GRADIENT ANALYSIS (Semantic Tension Mapping)")
        print(f"{'='*80}")
        print(f"Anchors: {args.metric_anchors}")
        print(f"Pairs: {args.metric_pairs}")
        print("NOTE: This is a SEPARATE LANE from embedding analysis")
        
        try:
            from metric_gradients import (
                MetricGradientConfig,
                MetricGradientExtractor,
                MetricGradientAnalyzer,
                run_gradient_controls,
            )
            
            # Create output directory
            metric_dir = exp_dir / "metric_gradients"
            metric_dir.mkdir(parents=True, exist_ok=True)
            
            # Build anchor dict
            default_anchors = {
                'victim': "This text describes victims and suffering.",
                'aggressor': "This text describes aggression and violence.",
                'neutral': "This text is a neutral factual report.",
                'emotional': "This text is emotionally charged.",
                'humanitarian': "This text focuses on humanitarian concerns.",
                'security': "This text focuses on security threats.",
            }
            anchors = {k: default_anchors.get(k, f"This text is about {k}.") 
                      for k in args.metric_anchors}
            
            # Parse anchor pairs
            anchor_pairs = [tuple(p.split(':')) for p in args.metric_pairs]
            
            # Initialize
            config = MetricGradientConfig(
                anchors=anchors,
                device='cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu',
            )
            extractor = MetricGradientExtractor(config)
            analyzer = MetricGradientAnalyzer(extractor)
            
            # Load article texts from real corpus
            article_texts = []
            real_artifact_path = None
            
            for kernel in args.kernels:
                candidate = exp_dir / kernel / "cls" / "real" / f"observer_{args.seeds[0]}.pt"
                if candidate.exists():
                    real_artifact_path = candidate
                    break
            
            if real_artifact_path:
                artifact = torch.load(real_artifact_path, map_location='cpu', weights_only=False)
                metadata = artifact.get('article_metadata', [])
                if metadata:
                    article_texts = [m.get('text', m.get('title', '')) for m in metadata]
            
            if not article_texts:
                print("[WARN] Could not find article texts in artifacts")
                print("  Metric gradient analysis requires raw text")
                print("  Skipping metric gradient analysis")
            else:
                print(f"\n--- Analyzing {len(article_texts)} articles ---")
                
                analysis = analyzer.analyze_corpus(
                    articles=article_texts[:min(len(article_texts), args.limit)],
                    anchor_pairs=anchor_pairs,
                    verbose=True,
                )
                
                analysis_path = metric_dir / "real_analysis.json"
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                
                print(f"\n--- Running gradient controls ---")
                controls = run_gradient_controls(
                    extractor=extractor,
                    real_articles=article_texts[:min(100, len(article_texts))],
                    n_samples=50,
                    verbose=True,
                )
                
                controls_path = metric_dir / "gradient_controls.json"
                with open(controls_path, 'w') as f:
                    json.dump(controls, f, indent=2)
                
                summary = {
                    'n_articles': len(article_texts),
                    'anchor_pairs': args.metric_pairs,
                    'tension_stats': analysis['tension_stats'],
                    'control_verdict': controls['verdict'],
                    'control_ordering': controls['ordering_satisfied'],
                }
                
                summary_path = metric_dir / "metric_gradient_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\nMetric gradient results saved to {metric_dir}")
                print(f"  Control verdict: {controls['verdict']}")
        
        except ImportError as e:
            print(f"[WARN] Metric gradient analysis requires metric_gradients.py")
            print(f"  Import error: {e}")
        except Exception as e:
            print(f"[ERROR] Metric gradient analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Only run comparison if all experiments succeeded
    all_success = all(r.get("status") == "success" for r in results if r.get("corpus") in corpora)
    if all_success:
        run_comparison(exp_dir, args.seeds)
    else:
        print("\n[WARN] Skipping comparison due to earlier failures.")

    
    # =========================================================================
    # VERIFICATION HARNESS
    # =========================================================================
    if getattr(args, 'verify', False):
        print("\n" + "="*80)
        print("RUNNING VERIFICATION HARNESS")
        print("="*80)
        try:
            # Ensure analysis is in path
            analysis_path = Path("analysis").resolve()
            if str(analysis_path) not in sys.path:
                sys.path.append(str(analysis_path))
            
            from verification.verify_run import discover_all_layers, verify_layer_data, write_reports_to_leaves
            
            reports = []
            # -----------------------------
            # RUNNING AUTOMATED ABLATIONS
            # -----------------------------
            # Pick first representative layer for ablation
            rep_channel = [c for c in args.channels if c != 'gradient'][0]
            rep_kernel = args.kernels[0]
            
            print(f"\n[VERIFY] Running Ablation A1: CRN OFF ({rep_kernel}/{rep_channel})")
            for corpus in args.corpora:
                a1_out = exp_dir / "ablation" / "crn_off" / rep_kernel / rep_channel / corpus
                a1_out.mkdir(parents=True, exist_ok=True)
                run_single_corpus(corpus, args.seeds, args.limit, a1_out, 
                                 mode=get_mode_for_channel(rep_channel, args.mode), 
                                 track_variance=False, kernel_type=rep_kernel,
                                 extra_flags=["--no-crn"])
            
            if getattr(args, 'alpha_sweep', False):
                print(f"\n[VERIFY] Running Ablation A2: ALPHA COLLAPSE ({rep_kernel}/{rep_channel})")
                for corpus in args.corpora:
                    a2_out = exp_dir / "ablation" / "alpha_collapse" / rep_kernel / rep_channel / corpus
                    a2_out.mkdir(parents=True, exist_ok=True)
                    run_single_corpus(corpus, args.seeds, args.limit, a2_out, 
                                     mode=get_mode_for_channel(rep_channel, args.mode), 
                                     track_variance=False, kernel_type=rep_kernel,
                                     extra_flags=["--alpha-collapse"])

            # Verify each discovered layer in the current experiment layout.
            all_layers = discover_all_layers(exp_dir)
            for layer in all_layers:
                print(f"Verifying Layer: {layer['layer_id']}")
                report = verify_layer_data(
                    layer['layer_id'],
                    layer['layer_name'],
                    layer['artifacts'],
                    layer['layer_dir'],
                    exp_dir,
                )
                reports.append(report)
            
            if reports:
                leaf_dirs = write_reports_to_leaves(reports, exp_dir, all_layers)
                if leaf_dirs:
                    print(f"[VERIFY] Verification reports saved to {len(leaf_dirs)} leaf directories")
                else:
                    print("[VERIFY] No leaf directories with MONOLITH_DATA.csv found; no verification files written.")
        except Exception as e:
            print(f"[VERIFY] Error during verification: {e}")
            import traceback
            traceback.print_exc()

    if not args.no_post_sync_results:
        run_post_thesis_sync(run_validation=args.post_validate_thesis)

    # Summary
    print(f"\n{'='*80}")
    print("SUITE COMPLETE")
    print(f"{'='*80}")
    print("Results summary:")
    for r in results:
        status_emoji = "[OK]" if r.get("status") == "success" else "[FAIL]"
        print(f"  {status_emoji} {r.get('corpus')}: {r.get('status')}")

        if "probe" in r:
            p = r["probe"]
            p_emoji = "[OK]" if p.get("status") == "success" else "[FAIL]"
            print(f"      Probe: {p_emoji} {p.get('status')}")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print(f"1. View results: cd {exp_dir}")
    print(f"2. Check variance: cat */variance_tracking.json (if present)")
    print(f"3. View comparison: cat comparison_results.json (if produced)")
    print(f"4. View probe results: cat */nli_probe_results.json")
    if args.alpha_sweep:
        print(f"5. View alpha sweep: cat alpha_sweep_summary.json")
    if getattr(args, 'verify', False):
        print(f"11. View verification report: cat {exp_dir / 'verification_report.json'}")
    if args.physarum:
        print(f"6. View atmospheric annealing analysis: cat physarum/annealing_summary.json")
        print(f"7. View crack/bond topology: ls physarum/*/")
        print(f"8. Plotly data ready in: physarum/*/crack_heatmap_data.json")
    if args.metric_gradients and args.physarum:
        print(f"9. DUAL-PATH: cat physarum/gradient_topology/gradient_annealing_analysis.json")
        print(f"10. INTERFERENCE: cat physarum/dual_path_interference.json")
        print(f"    (Shows 'wolves in sheep's clothing' - same words, different framing)")
    elif args.metric_gradients:
        print(f"9. View metric gradients: cat metric_gradients/metric_gradient_summary.json")
        print(f"10. Gradient controls: cat metric_gradients/gradient_controls.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()



