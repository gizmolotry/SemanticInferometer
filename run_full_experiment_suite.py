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
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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

    def _consumer_contract_failure_payload() -> Optional[Dict[str, Any]]:
        from analysis.verification.contract import evaluate_consumer_contract

        diagnostics = evaluate_consumer_contract(target_dir)
        if diagnostics.contract_ok:
            return None
        reasons = diagnostics.missing_required_artifacts + diagnostics.schema_errors
        return {
            "status": "failed",
            "stage": "consumer_contract",
            "error": "; ".join(reasons) if reasons else "consumer contract invalid",
            "run_dir": str(run_dir),
            "target_dir": str(target_dir),
        }

    if not monolith_csv.exists():
        monolith_ready = _ensure_monolith_csv_ready(target_dir)
        if monolith_ready.get("status") not in {"success", "already_exists"}:
            err_text = str(monolith_ready.get("error", "") or "")
            if "metric fusion failed:" in err_text and "refusing to emit flat manifold metrics" in err_text:
                return _emit_non_comparable_contract_bundle(
                    target_dir,
                    "flat manifold metrics collapsed during bundle materialization",
                )
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
        contract_failure = _consumer_contract_failure_payload()
        if contract_failure is not None:
            return contract_failure
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
        "focused",
        "--overwrite",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    try:
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

        print(f"[BUNDLE] Rendering MONOLITH baseline for {target_dir}...")
        viz_res = subprocess.run(viz_cmd, env=env)
        if viz_res.returncode != 0:
            return {
                "status": "failed",
                "stage": "monolith_render",
                "returncode": viz_res.returncode,
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
                "contract_bundle": contract_res,
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
        missing = _validate_required_bundle_outputs(target_dir)
        if missing:
            return {
                "status": "failed",
                "stage": "post_emit_validation",
                "error": f"missing required bundle outputs: {', '.join(missing)}",
                "run_dir": str(run_dir),
                "target_dir": str(target_dir),
            }
        contract_failure = _consumer_contract_failure_payload()
        if contract_failure is not None:
            return contract_failure
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
    global_payload = run_dir / "observer_global.pt"
    if global_payload.exists():
        return global_payload
    observer_files = sorted(run_dir.glob("observer_*.pt"), key=lambda p: p.name)
    return observer_files[0] if observer_files else None


def _load_primary_observer_payload(run_dir: Path) -> Optional[Dict[str, Any]]:
    observer_path = _pick_primary_observer_file(run_dir)
    if observer_path is None or not TORCH_AVAILABLE:
        return None
    try:
        payload = torch.load(observer_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_text_for_match(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _candidate_corpus_paths_for_run(run_dir: Path, payload: Optional[Dict[str, Any]]) -> List[Path]:
    payload = payload if isinstance(payload, dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    seen: set[str] = set()
    candidates: List[Path] = []

    def _push(path_value: Any) -> None:
        if not path_value:
            return
        try:
            candidate = Path(str(path_value)).resolve()
        except Exception:
            return
        key = str(candidate).lower()
        if key in seen or not candidate.exists():
            return
        seen.add(key)
        candidates.append(candidate)

    _push(meta.get("corpus_path"))

    corpus_name = str(meta.get("corpus", "") or "").strip().lower()
    if corpus_name in {"high_quality_articles", "synthetic", "synthetic30"}:
        _push(Path("sythgen/high_quality_articles.jsonl"))
        _push(Path("synthetic_corpus.jsonl"))
    elif corpus_name in {"synthetic_corpus", "synthetic_corpus.jsonl"}:
        _push(Path("synthetic_corpus.jsonl"))
        _push(Path("sythgen/high_quality_articles.jsonl"))

    for manifest_name in ("experiment_manifest.json", "suite_config.json"):
        manifest_path = run_dir.parent / manifest_name
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        _push(manifest.get("corpus_path"))

    return candidates


def _metadata_row_matches_corpus_row(meta_row: Dict[str, Any], corpus_row: Dict[str, Any]) -> bool:
    meta_title = _normalize_text_for_match(meta_row.get("title"))
    corpus_title = _normalize_text_for_match(corpus_row.get("title"))
    if meta_title and corpus_title and meta_title != corpus_title:
        return False

    meta_perspective = _normalize_text_for_match(meta_row.get("perspective_tag"))
    corpus_perspective = _normalize_text_for_match(corpus_row.get("perspective_tag"))
    if meta_perspective and corpus_perspective and meta_perspective != corpus_perspective:
        return False

    meta_preview = _normalize_text_for_match(
        meta_row.get("content_preview") or meta_row.get("snippet") or meta_row.get("summary")
    )
    corpus_text = _normalize_text_for_match(
        corpus_row.get("content") or corpus_row.get("text") or corpus_row.get("body")
    )
    if meta_preview:
        preview_prefix = meta_preview[:120]
        if not corpus_text or not corpus_text.startswith(preview_prefix):
            return False

    return True


def _load_metadata_rows_for_backfill(run_dir: Path, payload: Optional[Dict[str, Any]], n_articles: int) -> List[Dict[str, Any]]:
    payload = payload if isinstance(payload, dict) else {}
    metadata = payload.get("article_metadata")
    if isinstance(metadata, list) and metadata:
        return [dict(row) for row in metadata[:n_articles] if isinstance(row, dict)]

    metadata_json = run_dir / "article_metadata.json"
    if metadata_json.exists():
        try:
            rows = json.loads(metadata_json.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return [dict(row) for row in rows[:n_articles] if isinstance(row, dict)]
        except Exception:
            pass

    metadata_csv = run_dir / "article_metadata.csv"
    if metadata_csv.exists():
        try:
            with metadata_csv.open("r", encoding="utf-8", errors="replace") as f:
                return [dict(row) for row in csv.DictReader(f)][:n_articles]
        except Exception:
            pass

    return []


def _recover_articles_for_observer_backfill(
    run_dir: Path,
    payload: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    payload = payload if isinstance(payload, dict) else {}
    n_articles = int(payload.get("n_articles") or 0)
    metadata_rows = _load_metadata_rows_for_backfill(run_dir, payload, n_articles)
    if n_articles <= 0 or len(metadata_rows) < n_articles:
        return [], "insufficient article metadata for observer backfill"

    for candidate in _candidate_corpus_paths_for_run(run_dir, payload):
        try:
            candidate_rows, _ = load_and_mask_corpus(candidate)
        except Exception:
            continue
        if len(candidate_rows) < n_articles:
            continue

        prefix_rows = candidate_rows[:n_articles]
        if all(
            _metadata_row_matches_corpus_row(metadata_rows[i], prefix_rows[i])
            for i in range(n_articles)
        ):
            return prefix_rows, f"corpus_prefix:{candidate}"

        sequential_rows: List[Dict[str, Any]] = []
        cursor = 0
        success = True
        for meta_row in metadata_rows:
            match_idx = None
            for pos in range(cursor, len(candidate_rows)):
                if _metadata_row_matches_corpus_row(meta_row, candidate_rows[pos]):
                    match_idx = pos
                    break
            if match_idx is None:
                success = False
                break
            sequential_rows.append(candidate_rows[match_idx])
            cursor = match_idx + 1
        if success and len(sequential_rows) == n_articles:
            return sequential_rows, f"corpus_match:{candidate}"

    fallback_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(metadata_rows[:n_articles]):
        hydrated = dict(row)
        content = (
            hydrated.get("content")
            or hydrated.get("text")
            or hydrated.get("body")
            or hydrated.get("content_preview")
            or hydrated.get("snippet")
            or hydrated.get("summary")
            or hydrated.get("title")
            or f"Observer backfill article {idx}"
        )
        hydrated["content"] = str(content)
        fallback_rows.append(hydrated)
    return fallback_rows, "article_metadata_fallback"


def _infer_runtime_config_from_payload(payload: Dict[str, Any]) -> "PipelineRuntimeConfig":
    from core.pipeline_config import PipelineRuntimeConfig, DEFAULT_PIPELINE_RUNTIME_CONFIG

    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    basis_state = payload.get("rks_basis_state", {}) if isinstance(payload.get("rks_basis_state"), dict) else {}
    cls_per_bot = payload.get("cls_per_bot")
    features = payload.get("features")

    hidden_dim = DEFAULT_PIPELINE_RUNTIME_CONFIG.dirichlet_hidden_dim
    if cls_per_bot is not None:
        try:
            hidden_dim = int(np.asarray(cls_per_bot).shape[-1])
        except Exception:
            pass

    rks_dim = DEFAULT_PIPELINE_RUNTIME_CONFIG.dirichlet_rks_dim
    if features is not None:
        try:
            feature_arr = np.asarray(features)
            if feature_arr.ndim >= 2:
                rks_dim = int(feature_arr.shape[-1])
        except Exception:
            pass

    return PipelineRuntimeConfig(
        use_contrastive=bool(meta.get("use_contrastive", True)),
        use_pca_removal=bool(meta.get("use_pca_removal", False)),
        use_cls_tokens=bool(cls_per_bot is not None or str(meta.get("channel", "")).lower() == "cls"),
        use_gru=bool(meta.get("use_gru", True)),
        use_multi_framing_rks=bool(meta.get("use_multi_framing_rks", DEFAULT_PIPELINE_RUNTIME_CONFIG.use_multi_framing_rks)),
        use_attention=bool(meta.get("use_attention", DEFAULT_PIPELINE_RUNTIME_CONFIG.use_attention)),
        use_dirichlet_fusion=bool(meta.get("use_dirichlet_fusion", payload.get("T3_topology") is not None)),
        normalize_features=bool(meta.get("normalize_features", DEFAULT_PIPELINE_RUNTIME_CONFIG.normalize_features)),
        normalize_before_projection=bool(
            meta.get("normalize_before_projection", DEFAULT_PIPELINE_RUNTIME_CONFIG.normalize_before_projection)
        ),
        geometry_mode=str(meta.get("geometry_mode", DEFAULT_PIPELINE_RUNTIME_CONFIG.geometry_mode)),
        kernel_type=str(meta.get("kernel", basis_state.get("kernel_type", DEFAULT_PIPELINE_RUNTIME_CONFIG.kernel_type))),
        mix_in_rkhs=bool(meta.get("mix_in_rkhs", DEFAULT_PIPELINE_RUNTIME_CONFIG.mix_in_rkhs)),
        projection_dim=int(meta.get("projection_dim", DEFAULT_PIPELINE_RUNTIME_CONFIG.projection_dim)),
        apply_pca_to_cls=bool(meta.get("apply_pca_to_cls", DEFAULT_PIPELINE_RUNTIME_CONFIG.apply_pca_to_cls)),
        dirichlet_rks_dim=int(meta.get("dirichlet_rks_dim", rks_dim)),
        dirichlet_n_observers=int(meta.get("dirichlet_n_observers", DEFAULT_PIPELINE_RUNTIME_CONFIG.dirichlet_n_observers)),
        dirichlet_alpha=float(
            meta.get(
                "dirichlet_alpha",
                (payload.get("provenance", {}) or {}).get("alpha", DEFAULT_PIPELINE_RUNTIME_CONFIG.dirichlet_alpha),
            )
        ),
        dirichlet_hidden_dim=int(meta.get("dirichlet_hidden_dim", hidden_dim)),
    )


def _save_observer_run_leaf(obs_sub_dir: Path, obs_result: Dict[str, Any]) -> None:
    obs_sub_dir.mkdir(parents=True, exist_ok=True)

    def _save_npy(name: str, value: Any) -> None:
        if value is None:
            return
        try:
            np.save(obs_sub_dir / name, np.asarray(value))
        except Exception:
            pass

    _save_npy("features.npy", obs_result.get("features"))
    _save_npy("integrated_vectors.npy", obs_result.get("integrated_vectors"))
    _save_npy("dirichlet_fused.npy", obs_result.get("dirichlet_fused"))
    _save_npy("dirichlet_fused_std.npy", obs_result.get("dirichlet_fused_std"))
    _save_npy("walker_work_integrals.npy", obs_result.get("walker_work_integrals"))

    t15 = obs_result.get("T1.5_spectral", {}) if isinstance(obs_result.get("T1.5_spectral"), dict) else {}
    spectral_evr = obs_result.get("spectral_evr")
    if spectral_evr is None:
        spectral_evr = t15.get("evr")
    spectral_probe_magnitudes = obs_result.get("spectral_probe_magnitudes")
    if spectral_probe_magnitudes is None:
        spectral_probe_magnitudes = t15.get("probe_magnitudes")
    spectral_u_axis = obs_result.get("spectral_u_axis")
    if spectral_u_axis is None:
        spectral_u_axis = t15.get("u_axis")
    antagonism = obs_result.get("spectral_antagonism")
    if antagonism is None:
        antagonism = obs_result.get("antagonism")
    if antagonism is None:
        antagonism = t15.get("antagonism")

    _save_npy("spectral_evr.npy", spectral_evr)
    _save_npy("spectral_probe_magnitudes.npy", spectral_probe_magnitudes)
    _save_npy("spectral_dipole_valid.npy", obs_result.get("spectral_dipole_valid"))
    _save_npy("spectral_u_axis.npy", spectral_u_axis)
    _save_npy("antagonism.npy", antagonism)

    walker_states = obs_result.get("walker_states")
    if isinstance(walker_states, list):
        (obs_sub_dir / "walker_states.json").write_text(json.dumps(walker_states, indent=2), encoding="utf-8")

    phantom_verdicts = obs_result.get("phantom_verdicts")
    if isinstance(phantom_verdicts, list):
        (obs_sub_dir / "phantom_verdicts.json").write_text(json.dumps(phantom_verdicts, indent=2), encoding="utf-8")

    walker_paths = obs_result.get("walker_paths")
    if isinstance(walker_paths, list) and walker_paths:
        try:
            article_idx = np.arange(len(walker_paths), dtype=np.int64)
            path_xyz = np.array([np.asarray(path, dtype=float) for path in walker_paths], dtype=object)
            path_space = np.array(["embedding"], dtype=object)
            np.savez(obs_sub_dir / "walker_paths.npz", article_idx=article_idx, path_xyz=path_xyz, path_space=path_space)
        except Exception:
            pass


def _ensure_observer_universes_materialized(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)

    primary_payload = _load_primary_observer_payload(run_dir)
    if not isinstance(primary_payload, dict):
        return {"status": "failed", "reason": "primary observer payload unavailable"}

    n_articles = int(primary_payload.get("n_articles") or len(primary_payload.get("article_metadata", [])) or 0)
    if n_articles <= 0:
        return {"status": "failed", "reason": "observer payload missing article count"}

    existing_payloads = len(list(rel_dir.glob("observer_*.pt")))
    existing_obs_dirs = len([p for p in rel_dir.glob("obs_*") if p.is_dir()])
    if existing_payloads >= n_articles and existing_obs_dirs >= n_articles:
        if not (run_dir / "observer_global.pt").exists() and TORCH_AVAILABLE:
            try:
                torch.save(primary_payload, run_dir / "observer_global.pt")
            except Exception:
                pass
        return {
            "status": "already_exists",
            "observer_payloads": existing_payloads,
            "observer_dirs": existing_obs_dirs,
        }

    articles, article_source = _recover_articles_for_observer_backfill(run_dir, primary_payload)
    if len(articles) != n_articles:
        return {
            "status": "failed",
            "reason": f"observer backfill article recovery failed ({article_source})",
            "article_source": article_source,
        }

    try:
        from core.complete_pipeline import initialize_full_pipeline, BeliefTransformerPipeline
    except Exception as exc:
        return {"status": "failed", "reason": f"pipeline imports unavailable: {exc}"}

    runtime_cfg = _infer_runtime_config_from_payload(primary_payload)
    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    seed = int((primary_payload.get("meta", {}) or {}).get("seed", primary_payload.get("seed", 42)) or 42)

    try:
        components = initialize_full_pipeline(
            random_seed=seed,
            device=device,
            **runtime_cfg.to_initialize_kwargs(),
        )
        pipeline = BeliefTransformerPipeline(
            components=components,
            random_seed=seed,
            enable_provenance=True,
            provenance_dir=str(rel_dir),
        )
    except Exception as exc:
        return {"status": "failed", "reason": f"observer backfill pipeline init failed: {exc}"}

    run_meta = dict(primary_payload.get("meta", {}) if isinstance(primary_payload.get("meta"), dict) else {})
    if run_meta:
        primary_payload.setdefault("meta", run_meta)
    normalized_global_provenance = _normalize_run_provenance(primary_payload, run_meta)
    primary_payload["provenance"] = normalized_global_provenance
    primary_payload.setdefault("meta", {})["provenance"] = normalized_global_provenance
    if TORCH_AVAILABLE and not (run_dir / "observer_global.pt").exists():
        try:
            torch.save(primary_payload, run_dir / "observer_global.pt")
        except Exception:
            pass

    written_payloads = 0
    written_dirs = 0
    for obs_i in range(n_articles):
        obs_pt_path = rel_dir / f"observer_{obs_i}.pt"
        obs_sub_dir = rel_dir / f"obs_{obs_i}"
        if obs_pt_path.exists() and obs_sub_dir.exists() and (obs_sub_dir / "features.npy").exists():
            continue

        print(f"[RELATIVITY][BACKFILL] Materializing observer universe {obs_i + 1}/{n_articles} for {run_dir.name}...")
        obs_config = {
            "enable_checkpoints": True,
            "output_dir": str(obs_sub_dir),
            "checkpoint_dir": str(obs_sub_dir),
        }
        obs_result = pipeline.process_month(
            articles=articles,
            month_name=f"{run_dir.name}_obs{obs_i}",
            config=obs_config,
            observer_idx=obs_i,
        )
        obs_result["meta"] = run_meta
        normalized_obs_provenance = _normalize_run_provenance(obs_result, run_meta)
        obs_result["provenance"] = normalized_obs_provenance
        obs_result.setdefault("meta", {})["provenance"] = normalized_obs_provenance
        _save_observer_run_leaf(obs_sub_dir, obs_result)
        if TORCH_AVAILABLE:
            torch.save(obs_result, obs_pt_path)
        written_payloads += 1
        written_dirs += 1

        del obs_result
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "status": "success",
        "observer_payloads": len(list(rel_dir.glob("observer_*.pt"))),
        "observer_dirs": len([p for p in rel_dir.glob("obs_*") if p.is_dir()]),
        "article_source": article_source,
        "written_payloads": written_payloads,
        "written_dirs": written_dirs,
    }


def _json_ready(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    except Exception:
        return value


def _resolve_control_family_root(run_dir: Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    corpus_markers = ("real", "control_constant", "control_shuffled", "control_random")
    if any((run_dir / name).exists() for name in corpus_markers):
        return run_dir
    if any((run_dir.parent / name).exists() for name in corpus_markers):
        return run_dir.parent
    return None


def _center_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    return arr - np.mean(arr, axis=0, keepdims=True)


def _normalize_frobenius(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        return arr.copy()
    return arr / norm


def _orthogonal_procrustes_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = a.T @ b
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def _procrustes_residual(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = _normalize_frobenius(_center_rows(a))
    b_norm = _normalize_frobenius(_center_rows(b))
    rotation = _orthogonal_procrustes_rotation(a_norm, b_norm)
    aligned = a_norm @ rotation
    return float(np.linalg.norm(aligned - b_norm))


def _pairwise_distance_vector(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    deltas = arr[:, None, :] - arr[None, :, :]
    dist = np.linalg.norm(deltas, axis=2)
    iu = np.triu_indices(arr.shape[0], k=1)
    return dist[iu]


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[sorter]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[sorter[i:j]] = avg_rank
        i = j
    return ranks


def _distance_correlation_residual(a: np.ndarray, b: np.ndarray) -> float:
    da = _pairwise_distance_vector(a)
    db = _pairwise_distance_vector(b)
    if da.size == 0 or db.size == 0 or da.size != db.size:
        return 1.0
    ra = _rankdata_average(da)
    rb = _rankdata_average(db)
    ra -= np.mean(ra)
    rb -= np.mean(rb)
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom <= 1e-12:
        return 1.0
    rho = float(np.dot(ra, rb) / denom)
    rho = max(-1.0, min(1.0, rho))
    return float(1.0 - rho)


def _consensus_residual_summary(feature_mats: List[np.ndarray]) -> Dict[str, float]:
    if not feature_mats:
        return {"consensus_pct": 0.0, "residual_pct": 0.0}
    stacked = np.stack(feature_mats, axis=0)
    consensus = stacked.mean(axis=0)
    residuals = stacked - consensus[None, :, :]
    total_var = float(np.var(stacked))
    if total_var <= 1e-12:
        return {"consensus_pct": 0.0, "residual_pct": 0.0}
    consensus_pct = float(np.var(consensus) / total_var * 100.0)
    residual_pct = float(np.var(residuals) / total_var * 100.0)
    return {
        "consensus_pct": consensus_pct,
        "residual_pct": residual_pct,
    }


def _load_control_observers(corpus_dir: Path) -> Dict[int, Dict[str, Any]]:
    observers: Dict[int, Dict[str, Any]] = {}
    if not TORCH_AVAILABLE or not corpus_dir.exists():
        return observers
    for fpath in sorted(corpus_dir.glob("observer_*.pt")):
        seed_txt = fpath.stem.replace("observer_", "", 1)
        if not seed_txt.isdigit():
            continue
        try:
            payload = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception:
            continue
        matrix = payload.get("features") if isinstance(payload, dict) else payload
        if matrix is None and isinstance(payload, dict):
            matrix = payload.get("embeddings")
        if matrix is None:
            continue
        if TORCH_AVAILABLE and torch.is_tensor(matrix):
            matrix = matrix.detach().cpu().numpy()
        try:
            arr = np.asarray(matrix, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        observers[int(seed_txt)] = {
            "features": arr,
            "shape": arr.shape,
            "file": fpath.name,
        }
    return observers


def _reconcile_control_shapes(observers: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    if not observers:
        return {}
    shapes = [tuple(obs.get("shape", ())) for obs in observers.values()]
    if len(set(shapes)) <= 1:
        return observers
    shape_counts: Dict[Tuple[int, ...], int] = {}
    for shape in shapes:
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    dominant_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
    return {seed: obs for seed, obs in observers.items() if tuple(obs.get("shape", ())) == dominant_shape}


def _compute_direct_control_metrics(observers: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    seeds = sorted(observers.keys())
    if len(seeds) < 2:
        return None
    procrustes_vals: List[float] = []
    distance_vals: List[float] = []
    feature_mats: List[np.ndarray] = [np.asarray(observers[s]["features"], dtype=np.float64) for s in seeds]
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            obs_a = np.asarray(observers[seed_a]["features"], dtype=np.float64)
            obs_b = np.asarray(observers[seed_b]["features"], dtype=np.float64)
            procrustes_vals.append(_procrustes_residual(obs_a, obs_b))
            distance_vals.append(_distance_correlation_residual(obs_a, obs_b))
    if not procrustes_vals or not distance_vals:
        return None
    return {
        "procrustes": {
            "mean": float(np.mean(procrustes_vals)),
            "std": float(np.std(procrustes_vals)),
            "values": [float(v) for v in procrustes_vals],
        },
        "distance_corr": {
            "mean": float(np.mean(distance_vals)),
            "std": float(np.std(distance_vals)),
            "values": [float(v) for v in distance_vals],
        },
        "consensus_residual": _consensus_residual_summary(feature_mats),
        "n_observers": len(seeds),
        "seeds": seeds,
    }


def _build_direct_control_results_blob(run_dir: Path) -> Optional[Dict[str, Any]]:
    family_root = _resolve_control_family_root(run_dir)
    if family_root is None:
        return None

    corpus_dirs = {
        "Real": family_root / "real",
        "Constant": family_root / "control_constant",
        "Shuffled": family_root / "control_shuffled",
        "Random": family_root / "control_random",
    }

    results_dict: Dict[str, Dict[str, Any]] = {}
    for display_name, corpus_dir in corpus_dirs.items():
        observers = _reconcile_control_shapes(_load_control_observers(corpus_dir))
        results = _compute_direct_control_metrics(observers)
        if isinstance(results, dict):
            results_dict[display_name] = _json_ready(results)

    if "Real" not in results_dict or len(results_dict) < 2:
        # Single-seed suite runs still produce one observer payload per corpus.
        # Build direct real-vs-control comparisons from that family instead of
        # silently collapsing to NO_DATA.
        primary_family: Dict[str, Dict[str, Any]] = {}
        for display_name, corpus_dir in corpus_dirs.items():
            observers = _load_control_observers(corpus_dir)
            if not observers:
                continue
            primary_seed = sorted(observers.keys())[0]
            primary_family[display_name] = dict(observers[primary_seed])
            primary_family[display_name]["seed"] = primary_seed

        if "Real" not in primary_family or len(primary_family) < 2:
            return None

        shared_rows = min(
            int(np.asarray(obs["features"]).shape[0])
            for obs in primary_family.values()
            if np.asarray(obs["features"]).ndim == 2
        )
        shared_dim = min(
            int(np.asarray(obs["features"]).shape[1])
            for obs in primary_family.values()
            if np.asarray(obs["features"]).ndim == 2
        )
        if shared_rows < 2 or shared_dim < 1:
            return None

        aligned_family: Dict[str, np.ndarray] = {}
        for display_name, obs in primary_family.items():
            arr = np.asarray(obs["features"], dtype=np.float64)
            if arr.ndim != 2:
                continue
            aligned_family[display_name] = arr[:shared_rows, :shared_dim]
        if "Real" not in aligned_family or len(aligned_family) < 2:
            return None

        real_arr = aligned_family["Real"]
        control_names = [name for name in aligned_family.keys() if name != "Real"]
        control_pair_procrustes: List[float] = []
        control_pair_distcorr: List[float] = []
        if len(control_names) >= 2:
            for i, name_a in enumerate(control_names):
                for name_b in control_names[i + 1:]:
                    arr_a = aligned_family[name_a]
                    arr_b = aligned_family[name_b]
                    control_pair_procrustes.append(_procrustes_residual(arr_a, arr_b))
                    control_pair_distcorr.append(_distance_correlation_residual(arr_a, arr_b))

        real_vs_control_procrustes: List[float] = []
        real_vs_control_distcorr: List[float] = []
        fallback_results: Dict[str, Dict[str, Any]] = {}
        for control_name in control_names:
            ctrl_arr = aligned_family[control_name]
            p_val = _procrustes_residual(real_arr, ctrl_arr)
            d_val = _distance_correlation_residual(real_arr, ctrl_arr)
            real_vs_control_procrustes.append(p_val)
            real_vs_control_distcorr.append(d_val)
            fallback_results[control_name] = {
                "comparison_to_real": {
                    "procrustes": float(p_val),
                    "distance_corr": float(d_val),
                },
                "n_observers": 1,
                "seeds": [int(primary_family[control_name]["seed"])],
                "single_observer_mode": True,
            }

        if not real_vs_control_procrustes or not real_vs_control_distcorr:
            return None

        procrustes_control_avg = (
            float(np.mean(control_pair_procrustes))
            if control_pair_procrustes
            else 1.0
        )
        distcorr_control_avg = (
            float(np.mean(control_pair_distcorr))
            if control_pair_distcorr
            else 1.0
        )
        procrustes_real_value = float(np.mean(real_vs_control_procrustes))
        distcorr_real_value = float(np.mean(real_vs_control_distcorr))

        interpretation = {
            "has_real": True,
            "has_controls": True,
            "single_observer_mode": True,
            "metrics": {
                "procrustes": {
                    "real_value": procrustes_real_value,
                    "control_avg": procrustes_control_avg,
                    "ratio": (
                        float(procrustes_real_value / procrustes_control_avg)
                        if abs(procrustes_control_avg) > 1e-12
                        else None
                    ),
                    "separates": bool(
                        abs(procrustes_real_value - procrustes_control_avg) > 1e-9
                        and (
                            abs(procrustes_real_value / procrustes_control_avg - 1.0) > 0.2
                            if abs(procrustes_control_avg) > 1e-12
                            else True
                        )
                    ),
                },
                "distance_corr": {
                    "real_value": distcorr_real_value,
                    "control_avg": distcorr_control_avg,
                    "ratio": (
                        float(distcorr_real_value / distcorr_control_avg)
                        if abs(distcorr_control_avg) > 1e-12
                        else None
                    ),
                    "separates": bool(
                        abs(distcorr_real_value - distcorr_control_avg) > 1e-9
                        and (
                            abs(distcorr_real_value / distcorr_control_avg - 1.0) > 0.2
                            if abs(distcorr_control_avg) > 1e-12
                            else True
                        )
                    ),
                },
            },
            "consensus_residual": {
                "real": {
                    "consensus_pct": None,
                    "residual_pct": None,
                }
            },
        }
        fallback_results["Real"] = {
            "n_observers": 1,
            "seeds": [int(primary_family["Real"]["seed"])],
            "single_observer_mode": True,
            "consensus_residual": {"consensus_pct": None, "residual_pct": None},
        }
        return {
            "results": _json_ready(fallback_results),
            "interpretation": _json_ready(interpretation),
            "direct_source": str(family_root),
        }

    interpretation: Dict[str, Any] = {
        "has_real": True,
        "has_controls": True,
        "metrics": {},
        "consensus_residual": {
            "real": results_dict["Real"].get("consensus_residual", {}),
        },
    }
    for metric_name in ("procrustes", "distance_corr"):
        if metric_name not in results_dict["Real"]:
            continue
        real_val = float(results_dict["Real"][metric_name]["mean"])
        control_vals = [
            float(corpus_metrics[metric_name]["mean"])
            for corpus_name, corpus_metrics in results_dict.items()
            if corpus_name != "Real" and metric_name in corpus_metrics
        ]
        if not control_vals:
            continue
        control_avg = float(np.mean(control_vals))
        ratio = float(real_val / control_avg) if abs(control_avg) > 1e-12 else None
        interpretation["metrics"][metric_name] = {
            "real_value": real_val,
            "control_avg": control_avg,
            "ratio": ratio,
            "separates": bool(ratio is not None and abs(ratio - 1.0) > 0.2),
        }

    return {
        "results": results_dict,
        "interpretation": _json_ready(interpretation),
        "direct_source": str(family_root),
    }


def _build_control_metrics_payload(results_blob: Optional[Dict[str, Any]], source_path: Optional[Path]) -> Dict[str, Any]:
    def _control_explanation(controls_blob: Any) -> str:
        if not isinstance(controls_blob, dict):
            return (
                "Type 1 controls compare the real leaf against matched control siblings "
                "to show whether the manifold is carrying structured signal rather than noise."
            )
        present_set = set()
        for raw_name in controls_blob.keys():
            name = str(raw_name).strip().lower()
            if name in {"real", ""}:
                continue
            if name == "constant":
                present_set.add("constant")
            elif name == "shuffled":
                present_set.add("shuffled")
            elif name == "random":
                present_set.add("random")
            else:
                present_set.add(name)
        ordered = ["constant", "shuffled", "random"]
        present = [name for name in ordered if name in present_set]
        present.extend(sorted(name for name in present_set if name not in ordered))
        if not present:
            sibling_phrase = "matched control siblings"
        elif len(present) == 1:
            sibling_phrase = f"the matched {present[0]} control"
        elif len(present) == 2:
            sibling_phrase = f"matched {present[0]} and {present[1]} controls"
        else:
            sibling_phrase = "matched " + ", ".join(present[:-1]) + f", and {present[-1]} controls"
        return (
            f"Type 1 controls compare the real leaf against {sibling_phrase} "
            "to show whether the manifold is carrying structured signal rather than noise."
        )

    controls_blob = results_blob.get("results", {}) if isinstance(results_blob, dict) else {}
    payload: Dict[str, Any] = {
        "status": "NO_DATA",
        "panel_type": "type_1_epistemic_controls",
        "source": str(source_path) if source_path else "NOT FOUND",
        "synthetic_placeholder": True,
        "message": "control analysis not run for this leaf",
        "explanation": _control_explanation(controls_blob),
        "metrics": {
            "procrustes_ratio": None,
            "distance_corr_ratio": None,
            "separates_count": 0,
            "consensus_pct": None,
            "residual_pct": None,
        },
        "controls": {},
        "summary": {
            "real_vs_controls": "not evaluated",
            "interpretation": "no control comparison available for this leaf",
        },
    }
    if not isinstance(results_blob, dict) or not results_blob:
        return payload

    status_marker = str(results_blob.get("status", "")).strip().upper()
    if status_marker in {"NO_DATA", "UNAVAILABLE", "MISSING"} or bool(results_blob.get("synthetic_placeholder", False)):
        payload["status"] = "NO_DATA" if status_marker in {"", "NO_DATA"} or bool(results_blob.get("synthetic_placeholder", False)) else status_marker
        payload["message"] = str(results_blob.get("message") or results_blob.get("reason") or payload["message"])
        return payload

    interpretation = results_blob.get("interpretation", {})
    if not isinstance(interpretation, dict):
        return payload

    if interpretation.get("error"):
        payload["status"] = "UNAVAILABLE"
        payload["message"] = str(interpretation.get("error"))
        return payload

    metrics = interpretation.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    procrustes_ratio = _safe_float((metrics.get("procrustes", {}) or {}).get("ratio"), default=float("nan"))
    distance_corr_ratio = _safe_float((metrics.get("distance_corr", {}) or {}).get("ratio"), default=float("nan"))
    separates_count = sum(1 for m in metrics.values() if isinstance(m, dict) and bool(m.get("separates")))
    consensus_residual = ((interpretation.get("consensus_residual", {}) or {}).get("real", {}) or {})
    controls = controls_blob
    controls = controls if isinstance(controls, dict) else {}

    payload.update(
        {
            "status": "OK",
            "synthetic_placeholder": bool(results_blob.get("synthetic_placeholder", False)),
            "message": "loaded",
            "explanation": _control_explanation(controls),
            "metrics": {
                "procrustes_ratio": None if str(procrustes_ratio) == "nan" else procrustes_ratio,
                "distance_corr_ratio": None if str(distance_corr_ratio) == "nan" else distance_corr_ratio,
                "separates_count": int(separates_count),
                "consensus_pct": _safe_float(consensus_residual.get("consensus_pct"), default=None),
                "residual_pct": _safe_float(consensus_residual.get("residual_pct"), default=None),
            },
            "controls": controls,
            "summary": {
                "real_vs_controls": "evaluated",
                "interpretation": (
                    "Ratios near 1.0 indicate the real manifold behaves like controls; "
                    "larger separation indicates signal survives against noise baselines."
                ),
            },
        }
    )
    return payload


def _emit_control_metrics_json(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "control_metrics.json"
    direct_blob = _build_direct_control_results_blob(run_dir)
    if isinstance(direct_blob, dict) and direct_blob:
        source_path = Path(str(direct_blob.get("direct_source", run_dir)))
        payload = _build_control_metrics_payload(direct_blob, source_path)
        payload["message"] = "loaded from direct control observer payloads"
        payload["synthetic_placeholder"] = False
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    source_path = run_dir / "comprehensive_results.json"
    if source_path.exists():
        try:
            results_blob = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception:
            results_blob = {}
    else:
        results_blob = {}
    payload = _build_control_metrics_payload(results_blob, source_path if source_path.exists() else None)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _infer_validation_nmi_from_payload(validation_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(validation_payload, dict):
        return None
    track_metrics_existing = validation_payload.get("track_metrics")
    if isinstance(track_metrics_existing, dict):
        syn_blob = track_metrics_existing.get("SYN")
        if isinstance(syn_blob, dict) and isinstance(syn_blob.get("nmi"), (int, float)) and not isinstance(syn_blob.get("nmi"), bool):
            inferred = float(syn_blob.get("nmi"))
            if math.isfinite(inferred) and 0.0 <= inferred <= 1.0:
                return inferred
    track_nmi_existing = validation_payload.get("track_nmi")
    if isinstance(track_nmi_existing, dict) and isinstance(track_nmi_existing.get("SYN"), (int, float)) and not isinstance(track_nmi_existing.get("SYN"), bool):
        inferred = float(track_nmi_existing.get("SYN"))
        if math.isfinite(inferred) and 0.0 <= inferred <= 1.0:
            return inferred
    return None


def _repair_validation_payload(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inferred_nmi = _infer_validation_nmi_from_payload(existing)
    if inferred_nmi is None:
        return None
    repaired = dict(existing)
    repaired["nmi"] = inferred_nmi
    if "status" in repaired and str(repaired.get("status", "")).strip().lower() == "failed":
        repaired["status"] = "success"
    if str(repaired.get("trust_level", "")).strip().upper() in {"", "UNAVAILABLE", "FAILED"}:
        repaired["trust_level"] = "MEASURED"
    return repaired


def _normalize_validation_payload(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(existing, dict) or not existing:
        return None
    current_nmi = existing.get("nmi")
    if not isinstance(current_nmi, (int, float)) or isinstance(current_nmi, bool):
        return None
    current_nmi = float(current_nmi)
    if not math.isfinite(current_nmi) or not (0.0 <= current_nmi <= 1.0):
        return None
    normalized = dict(existing)
    changed = False
    if str(normalized.get("status", "")).strip().lower() == "failed":
        normalized["status"] = "success"
        changed = True
    if str(normalized.get("trust_level", "")).strip().upper() in {"", "UNAVAILABLE", "FAILED"}:
        normalized["trust_level"] = "MEASURED"
        changed = True
    return normalized if changed else None


def _build_ablation_summary_payload(summary_blob: Optional[Dict[str, Any]], source_path: Optional[Path]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "NO_DATA",
        "panel_type": "ablation_laboratory",
        "source": str(source_path) if source_path else "NOT FOUND",
        "synthetic_placeholder": True,
        "reason": "ablation flow not executed for this run",
        "message": "ablation flow not executed for this run",
        "explanation": (
            "Ablations compare null and mature pipeline branches so we can see which structural "
            "choices preserve the manifold and which ones wash it out. Alignment scores are "
            "distance-derived proxies, not literal NMI."
        ),
        "stage_1_alignment_score": None,
        "stage_2_alignment_score": None,
        "stage_3_survival_rate": None,
        "delta_alignment_score": None,
        "stage_1_nmi": None,
        "stage_2_nmi": None,
        "stage_3_nmi": None,
        "delta_nmi": None,
        "retained_percentage": None,
        "retained_pct": None,
        "legacy_mean_variance": None,
        "mean_variance": None,
        "metrics": {
            "stage_1_alignment_score": None,
            "stage_2_alignment_score": None,
            "stage_3_survival_rate": None,
            "delta_alignment_score": None,
            "stage_1_nmi": None,
            "stage_2_nmi": None,
            "stage_3_nmi": None,
            "delta_nmi": None,
            "retained_pct": None,
            "legacy_mean_variance": None,
        },
        "summary": {
            "stage_map": {
                "stage_1_alignment_score": "null or pre-refactor alignment score",
                "stage_2_alignment_score": "post-repair alignment score",
                "stage_3_survival_rate": "structural invariant survival rate",
                "stage_1_nmi": "null or pre-refactor alignment proxy",
                "stage_2_nmi": "post-repair alignment proxy",
                "stage_3_nmi": "structural invariant survival proxy",
            },
            "interpretation": "no ablation laboratory output available for this run",
        },
    }
    if not isinstance(summary_blob, dict) or not summary_blob:
        return payload

    status_marker = str(summary_blob.get("status", "")).strip().upper()
    if status_marker in {"NO_DATA", "UNAVAILABLE"} or bool(summary_blob.get("synthetic_placeholder", False)):
        payload["status"] = status_marker or "NO_DATA"
        payload["reason"] = str(summary_blob.get("reason", payload["reason"]))
        payload["message"] = str(summary_blob.get("message", payload["message"]))
        return payload

    metrics = summary_blob.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}

    def _pick(name: str, metric_name: Optional[str] = None):
        metric_key = metric_name or name
        val = summary_blob.get(name)
        if val is None:
            val = metrics.get(metric_key)
        return val

    stage_1_alignment = _pick("stage_1_alignment_score", "stage_1_alignment_score")
    stage_2_alignment = _pick("stage_2_alignment_score", "stage_2_alignment_score")
    stage_3_survival = _pick("stage_3_survival_rate", "stage_3_survival_rate")
    delta_alignment = _pick("delta_alignment_score", "delta_alignment_score")
    stage_1_nmi = _pick("stage_1_nmi")
    stage_2_nmi = _pick("stage_2_nmi")
    stage_3_nmi = _pick("stage_3_nmi")
    delta_nmi = _pick("delta_nmi")
    if stage_1_alignment is None:
        stage_1_alignment = stage_1_nmi
    if stage_2_alignment is None:
        stage_2_alignment = stage_2_nmi
    if stage_3_survival is None:
        stage_3_survival = stage_3_nmi
    if delta_alignment is None:
        delta_alignment = delta_nmi
    stage_1_nmi = stage_1_alignment
    stage_2_nmi = stage_2_alignment
    stage_3_nmi = stage_3_survival
    delta_nmi = delta_alignment
    retained_pct = _pick("retained_percentage", "retained_pct")
    legacy_mean_variance = _pick("legacy_mean_variance")

    payload.update(
        {
            "status": "OK",
            "synthetic_placeholder": False,
            "reason": None,
            "message": str(summary_blob.get("message", "loaded")),
            "explanation": str(summary_blob.get("explanation", payload["explanation"])),
            "stage_1_alignment_score": stage_1_alignment,
            "stage_2_alignment_score": stage_2_alignment,
            "stage_3_survival_rate": stage_3_survival,
            "delta_alignment_score": delta_alignment,
            "stage_1_nmi": stage_1_nmi,
            "stage_2_nmi": stage_2_nmi,
            "stage_3_nmi": stage_3_nmi,
            "delta_nmi": delta_nmi,
            "retained_percentage": retained_pct,
            "retained_pct": retained_pct,
            "legacy_mean_variance": legacy_mean_variance,
            "mean_variance": legacy_mean_variance,
            "metrics": {
                "stage_1_alignment_score": stage_1_alignment,
                "stage_2_alignment_score": stage_2_alignment,
                "stage_3_survival_rate": stage_3_survival,
                "delta_alignment_score": delta_alignment,
                "stage_1_nmi": stage_1_nmi,
                "stage_2_nmi": stage_2_nmi,
                "stage_3_nmi": stage_3_nmi,
                "delta_nmi": delta_nmi,
                "retained_pct": retained_pct,
                "legacy_mean_variance": legacy_mean_variance,
            },
        }
    )
    return payload


def _is_placeholder_status_blob(blob: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(blob, dict) or not blob:
        return True
    if bool(blob.get("synthetic_placeholder", False)):
        return True
    status = str(blob.get("status", "")).strip().upper()
    if status in {"NO_DATA", "UNAVAILABLE", "MISSING"}:
        return True
    if status:
        return False
    meaningful_metric_keys = {
        "stage_1_alignment_score",
        "stage_2_alignment_score",
        "stage_3_survival_rate",
        "delta_alignment_score",
        "stage_1_nmi",
        "stage_2_nmi",
        "stage_3_nmi",
        "delta_nmi",
        "retained_percentage",
        "retained_pct",
        "legacy_mean_variance",
        "mean_variance",
    }
    if any(blob.get(key) is not None for key in meaningful_metric_keys):
        return False
    metrics = blob.get("metrics")
    if isinstance(metrics, dict) and any(metrics.get(key) is not None for key in meaningful_metric_keys):
        return False
    return True


def _translate_lab_diagnostics_to_ablation_summary(lab_blob: Optional[Dict[str, Any]], source_path: Optional[Path]) -> Dict[str, Any]:
    payload = _build_ablation_summary_payload({}, source_path)
    if not isinstance(lab_blob, dict) or not lab_blob:
        return payload

    procrustes = lab_blob.get("procrustes", {})
    invariants = lab_blob.get("structural_invariants", {})
    if not isinstance(procrustes, dict):
        procrustes = {}
    if not isinstance(invariants, dict):
        invariants = {}

    mean_before = _safe_float(procrustes.get("mean_distance_before"), default=None)
    mean_after = _safe_float(procrustes.get("mean_distance_after"), default=None)
    survival_rate = _safe_float(invariants.get("mean_survival_rate"), default=None)

    def _distance_similarity(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(1.0 / (1.0 + max(float(value), 0.0)))
        except Exception:
            return None

    stage_1 = _distance_similarity(mean_before)
    stage_2 = _distance_similarity(mean_after)
    stage_3 = survival_rate
    delta_nmi = None
    if stage_1 is not None and stage_2 is not None:
        delta_nmi = float(stage_2 - stage_1)
    retained_pct = float(survival_rate * 100.0) if survival_rate is not None else None

    payload.update(
        {
            "status": "OK",
            "synthetic_placeholder": False,
            "reason": None,
            "source": str(source_path) if source_path else "lab_diagnostics.json",
            "message": "translated from lab_diagnostics.json",
            "explanation": (
                "Ablation laboratory output translated from observer alignment distances and "
                "invariant survival rates. Alignment scores are distance-derived proxies, not literal NMI."
            ),
            "stage_1_alignment_score": stage_1,
            "stage_2_alignment_score": stage_2,
            "stage_3_survival_rate": stage_3,
            "delta_alignment_score": delta_nmi,
            "stage_1_nmi": stage_1,
            "stage_2_nmi": stage_2,
            "stage_3_nmi": stage_3,
            "delta_nmi": delta_nmi,
            "retained_percentage": retained_pct,
            "retained_pct": retained_pct,
            "legacy_mean_variance": None,
            "mean_variance": None,
            "metrics": {
                "stage_1_alignment_score": stage_1,
                "stage_2_alignment_score": stage_2,
                "stage_3_survival_rate": stage_3,
                "delta_alignment_score": delta_nmi,
                "stage_1_nmi": stage_1,
                "stage_2_nmi": stage_2,
                "stage_3_nmi": stage_3,
                "delta_nmi": delta_nmi,
                "retained_pct": retained_pct,
                "legacy_mean_variance": None,
            },
            "translation": {
                "source_type": "lab_diagnostics",
                "lab_metrics": {
                    "mean_distance_before": mean_before,
                    "mean_distance_after": mean_after,
                    "mean_survival_rate": survival_rate,
                    "consensus_fraction": _safe_float(procrustes.get("consensus_fraction"), default=None),
                    "residual_fraction": _safe_float(procrustes.get("residual_fraction"), default=None),
                },
            },
            "summary": {
                "stage_map": {
                    "stage_1_alignment_score": "alignment before repair / null-side similarity proxy",
                    "stage_2_alignment_score": "alignment after repair / mature-side similarity proxy",
                    "stage_3_survival_rate": "structural invariant survival rate",
                    "stage_1_nmi": "alignment before repair / null-side similarity proxy",
                    "stage_2_nmi": "alignment after repair / mature-side similarity proxy",
                    "stage_3_nmi": "structural invariant survival rate",
                },
                "interpretation": (
                    "Higher stage 2 and stage 3 values mean the mature ablation path preserved "
                    "observer agreement and invariant survival more effectively."
                ),
            },
        }
    )
    return payload


def _emit_ablation_results_json(run_dir: Path, payload: Dict[str, Any]) -> Path:
    out = Path(run_dir) / "ablation_results.json"
    legacy_blob = dict(payload)
    legacy_blob.setdefault("retained_pct", payload.get("retained_percentage"))
    legacy_blob.setdefault("mean_variance", payload.get("legacy_mean_variance"))
    out.write_text(json.dumps(legacy_blob, indent=2), encoding="utf-8")
    return out


def _emit_ablation_summary_json(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "ablation_summary.json"

    existing_blob: Dict[str, Any] = {}
    if out.exists():
        try:
            loaded_existing = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(loaded_existing, dict):
                existing_blob = loaded_existing
        except Exception:
            existing_blob = {}

    source_path: Optional[Path] = None
    summary_blob: Dict[str, Any] = {}

    preferred_sources = [
        (run_dir / "lab_diagnostics.json", "lab"),
        (run_dir / "ablation_results.json", "json"),
        (run_dir / "critical_ablation_summary.csv", "csv"),
    ]

    for candidate_path, source_kind in preferred_sources:
        if not candidate_path.exists():
            continue
        source_path = candidate_path
        try:
            if source_kind == "csv":
                with candidate_path.open("r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    summary_blob = dict(next(reader, {}) or {})
            else:
                loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
                summary_blob = loaded if isinstance(loaded, dict) else {}
        except Exception:
            summary_blob = {}

        if source_kind == "lab":
            payload = _translate_lab_diagnostics_to_ablation_summary(summary_blob, source_path)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _emit_ablation_results_json(run_dir, payload)
            return out

        if not _is_placeholder_status_blob(summary_blob):
            payload = _build_ablation_summary_payload(summary_blob, source_path)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _emit_ablation_results_json(run_dir, payload)
            return out

    payload = _build_ablation_summary_payload(existing_blob, out if out.exists() else source_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit_ablation_results_json(run_dir, payload)
    return out


def _emit_relativity_deltas_json(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    rel_dir = run_dir / "relativity_cache"
    out = run_dir / "relativity_deltas.json"
    state_paths = sorted(rel_dir.glob("state_*.json"), key=lambda p: p.name)
    delta_paths = {
        p.stem.replace("delta_", "", 1): p
        for p in sorted(rel_dir.glob("delta_*.json"), key=lambda p: p.name)
    }
    observers: List[Dict[str, Any]] = []
    coord_deltas_all: List[float] = []
    synthetic_observer_count = 0
    for state_path in state_paths:
        observer_key = state_path.stem.replace("state_", "", 1)
        try:
            state_blob = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state_blob = {}
        if observer_key in delta_paths:
            try:
                delta_blob = json.loads(delta_paths[observer_key].read_text(encoding="utf-8"))
            except Exception:
                delta_blob = {}
        else:
            delta_blob = {}
        if not isinstance(state_blob, dict):
            continue
        metrics = state_blob.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        articles = state_blob.get("articles", [])
        articles = articles if isinstance(articles, list) else []
        synthetic_placeholder = bool(state_blob.get("synthetic_placeholder", False)) or bool(delta_blob.get("synthetic_placeholder", False))
        if synthetic_placeholder:
            synthetic_observer_count += 1
        observer_density = _safe_float(metrics.get("observer_density"), default=0.0)
        observer_stress = _safe_float(metrics.get("observer_stress"), default=0.0)
        observer_z_height = _safe_float(metrics.get("observer_z_height"), default=0.0)
        vectors: List[Dict[str, Any]] = []
        for article in articles:
            if not isinstance(article, dict):
                continue
            baseline_x = article.get("baseline_x")
            baseline_y = article.get("baseline_y")
            baseline_z = article.get("baseline_z")
            if baseline_x is None or baseline_y is None or baseline_z is None:
                baseline_x = _safe_float(article.get("density"), default=0.0) - observer_density
                baseline_y = _safe_float(article.get("stress"), default=0.0) - observer_stress
                baseline_z = _safe_float(article.get("z_height"), default=0.0) - observer_z_height
            else:
                baseline_x = _safe_float(baseline_x, default=0.0)
                baseline_y = _safe_float(baseline_y, default=0.0)
                baseline_z = _safe_float(baseline_z, default=0.0)
            observer_x = _safe_float(article.get("observer_x"), default=0.0)
            observer_y = _safe_float(article.get("observer_y"), default=0.0)
            observer_z = _safe_float(article.get("observer_z"), default=0.0)
            delta_x = _safe_float(article.get("delta_x"), default=observer_x - baseline_x)
            delta_y = _safe_float(article.get("delta_y"), default=observer_y - baseline_y)
            delta_z = _safe_float(article.get("delta_z"), default=observer_z - baseline_z)
            coord_delta = _safe_float(
                article.get("coord_delta"),
                default=float(np.linalg.norm([delta_x, delta_y, delta_z])),
            )
            vectors.append(
                {
                    "index": int(article.get("index", len(vectors))),
                    "bt_uid": str(article.get("bt_uid", "")),
                    "title": str(article.get("title", "")),
                    "baseline": {"x": baseline_x, "y": baseline_y, "z": baseline_z},
                    "observer": {"x": observer_x, "y": observer_y, "z": observer_z},
                    "delta": {"x": delta_x, "y": delta_y, "z": delta_z},
                    "coord_delta": coord_delta,
                }
            )
            if math.isfinite(coord_delta) and not synthetic_placeholder:
                coord_deltas_all.append(coord_delta)
        observers.append(
            {
                "observer_id": int(state_blob.get("observer_id", observer_key)),
                "observer_bt_uid": str(metrics.get("observer_bt_uid", "")),
                "observer_title": str(metrics.get("observer_title", "")),
                "observer_conditioned_nmi": metrics.get("observer_conditioned_nmi"),
                "global_nmi": metrics.get("global_nmi"),
                "null_observer_equivalence": delta_blob.get("null_observer_equivalence", {}),
                "path_flip_delta": delta_blob.get("path_flip_delta", {}),
                "metrics_delta": delta_blob.get("metrics_delta", {}),
                "axis_delta": delta_blob.get("axis_delta", {}),
                "translation_only_comparison": delta_blob.get("translation_only_comparison", {}),
                "synthetic_placeholder": synthetic_placeholder,
                "vectors": vectors,
            }
        )

    all_synthetic = bool(observers) and synthetic_observer_count == len(observers)
    payload = {
        "status": "NO_DATA" if (not observers or all_synthetic) else "OK",
        "panel_type": "type_2_relativity",
        "source": str(rel_dir),
        "observer_count": len(observers),
        "synthetic_placeholder": all_synthetic,
        "message": (
            "observer-conditioned relativity payloads were not materialized for this leaf"
            if all_synthetic
            else "loaded"
        ),
        "explanation": (
            "Type 2 relativity measures vector displacement between the global mean manifold "
            "and each observer-conditioned manifold in a shared coordinate frame."
            if not all_synthetic
            else "Type 2 relativity is unavailable because this leaf does not carry observer-conditioned local universes in a shared displacement frame."
        ),
        "summary": {
            "mean_coord_delta": float(np.mean(coord_deltas_all)) if coord_deltas_all else None,
            "max_coord_delta": float(np.max(coord_deltas_all)) if coord_deltas_all else None,
            "interpretation": (
                "Larger displacement magnitudes indicate stronger ideological shear between "
                "the observer frame and the global mean frame."
                if not all_synthetic
                else "No displacement field was computed for this leaf."
            ),
        },
        "observers": observers,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


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
        run_dir / "verification_report.json",
        run_dir / "verification_summary.csv",
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

    rel_dir = run_dir / "relativity_cache"
    for rel_path in list(sorted(rel_dir.glob("state_*.json"))) + list(sorted(rel_dir.glob("delta_*.json"))):
        try:
            blob = json.loads(rel_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if bool(blob.get("synthetic_placeholder", False)):
            return False

    verification_report = run_dir / "verification_report.json"
    if verification_report.exists():
        try:
            report_blob = json.loads(verification_report.read_text(encoding="utf-8"))
        except Exception:
            return False
        layer_failures = []
        for layer in report_blob.get("layers", []) if isinstance(report_blob.get("layers"), list) else []:
            if isinstance(layer, dict):
                layer_failures.extend(str(reason) for reason in (layer.get("fail_reasons") or []))
        if any("verification layer not materialized during bundle emission" in reason for reason in layer_failures):
            return False

    input_files = [run_dir / "MONOLITH_DATA.csv"]
    for name in ("verification_report.json", "verification_summary.csv"):
        p = run_dir / name
        if p.exists():
            input_files.append(p)
    for name in (
        "features.npy",
        "walker_states.json",
        "phantom_verdicts.json",
        "article_metadata.csv",
        "article_metadata.json",
        "validation.json",
        "EPISTEMIC_CONTRACT.json",
        "observer_manifest.json",
    ):
        p = run_dir / name
        if p.exists():
            input_files.append(p)
    input_files.extend(sorted(run_dir.glob("observer_*/MONOLITH*.html")))
    input_files.extend(sorted(run_dir.glob("observer_*.pt")))
    newest_input = max(int(p.stat().st_mtime_ns) for p in input_files if p.exists())

    output_files = [
        run_dir / "MONOLITH.html",
        run_dir / "observer_manifest.json",
        run_dir / "baseline_meta.json",
        run_dir / "baseline_state.json",
        run_dir / "validation.json",
    ]
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
            hydrated = {str(k): v for k, v in row.items() if k is not None}
            hydrated["index"] = idx
            hydrated["bt_uid"] = row.get("bt_uid", f"article_{idx}")
            hydrated["title"] = (row.get("title", "") or "")[:200]
            hydrated["zone"] = row.get("zone", "unknown")
            hydrated["density"] = row.get("density", "0")
            hydrated["stress"] = row.get("stress", "0")
            rows.append(hydrated)
    return rows


def _load_metadata_rows(metadata_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not metadata_csv.exists():
        return rows
    with metadata_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for i, row in enumerate(reader):
            ridx = row.get("index", "")
            try:
                idx = int(ridx)
            except Exception:
                idx = i
            title = (row.get("title", "") or row.get("article_title", "") or "")[:200]
            rows.append(
                {
                    "index": idx,
                    "bt_uid": row.get("bt_uid", f"article_{idx}"),
                    "title": title,
                    "zone": row.get("zone", "non_comparable"),
                    "density": "0",
                    "stress": "0",
                    "z_height": "0",
                    "source": row.get("source", ""),
                    "perspective_tag": row.get("perspective_tag", ""),
                }
            )
    return rows


def _emit_observer_backed_contract_bundle(run_dir: Path, reason: str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    hydrate_result = _hydrate_run_leaf_from_observer(run_dir)
    observer_payload = _load_primary_observer_payload(run_dir)
    rows = _load_metadata_rows(run_dir / "article_metadata.csv")
    if not rows and isinstance(observer_payload, dict):
        metadata = observer_payload.get("article_metadata")
        if isinstance(metadata, list):
            for i, item in enumerate(metadata):
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "index": int(item.get("index", i)),
                        "bt_uid": str(item.get("bt_uid", f"article_{i}")),
                        "title": str(item.get("title", ""))[:200],
                        "zone": str(item.get("zone", "observer_backed")) or "observer_backed",
                        "density": str(item.get("density", 0)),
                        "stress": str(item.get("stress", 0)),
                        "z_height": str(item.get("z_height", 0)),
                        "source": str(item.get("source", "")),
                        "perspective_tag": str(item.get("perspective_tag", "")),
                    }
                )

    if not rows:
        return {
            "status": "failed",
            "stage": "observer_backed_bundle",
            "error": f"unable to build fallback article rows: {reason}",
            "run_dir": str(run_dir),
            "hydration": hydrate_result,
        }

    _ensure_verification_report(run_dir)
    baseline_meta = _emit_baseline_meta(run_dir)
    baseline_state = _emit_baseline_state(run_dir, rows)
    validation_json = _emit_validation_json(run_dir)
    try:
        validation_blob = json.loads(validation_json.read_text(encoding="utf-8"))
    except Exception:
        validation_blob = {}
    if validation_blob.get("nmi") is None:
        return _emit_non_comparable_contract_bundle(
            run_dir,
            f"{reason}; validation metrics unavailable",
        )
    control_metrics = _emit_control_metrics_json(run_dir)
    ablation_summary = _emit_ablation_summary_json(run_dir)
    label_paths = _emit_label_derivatives(run_dir, rows)

    return {
        "status": "success",
        "mode": "observer_backed",
        "reason": reason,
        "run_dir": str(run_dir),
        "baseline_meta": str(baseline_meta),
        "baseline_state": str(baseline_state),
        "validation_json": str(validation_json),
        "control_metrics": str(control_metrics),
        "ablation_summary": str(ablation_summary),
        "labels": label_paths,
    }


def _emit_non_comparable_contract_bundle(run_dir: Path, reason: str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    hydrate_result = _hydrate_run_leaf_from_observer(run_dir)
    observer_payload = _load_primary_observer_payload(run_dir)
    rows = _load_metadata_rows(run_dir / "article_metadata.csv")
    if not rows and isinstance(observer_payload, dict):
        metadata = observer_payload.get("article_metadata")
        if isinstance(metadata, list):
            for i, item in enumerate(metadata):
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "index": int(item.get("index", i)),
                        "bt_uid": str(item.get("bt_uid", f"article_{i}")),
                        "title": str(item.get("title", ""))[:200],
                        "zone": "non_comparable",
                        "density": "0",
                        "stress": "0",
                        "z_height": "0",
                        "source": str(item.get("source", "")),
                        "perspective_tag": str(item.get("perspective_tag", "")),
                    }
                )

    if not rows:
        return {
            "status": "failed",
            "stage": "non_comparable_bundle",
            "error": f"unable to build fallback article rows: {reason}",
            "run_dir": str(run_dir),
            "hydration": hydrate_result,
        }

    summary = _build_leaf_provenance_summary(run_dir, observer_payload)
    summary["verification_status"] = "NON_COMPARABLE"
    summary["provenance_source"] = "observer_payload_non_comparable"
    baseline_meta = run_dir / "baseline_meta.json"
    baseline_meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    baseline_state = {
        "articles": rows,
        "paths": [],
        "axes": {"x": "density", "y": "stress"},
        "metrics": {
            "source": "article_metadata.csv",
            "comparability_status": "NON_COMPARABLE",
            "reason": reason,
        },
    }
    baseline_state_path = run_dir / "baseline_state.json"
    baseline_state_path.write_text(json.dumps(baseline_state, indent=2), encoding="utf-8")

    validation_payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_observers": 1,
        "nmi": None,
        "ari": None,
        "nmi_std": None,
        "ari_std": None,
        "metric_source": "non_comparable_flat_manifold",
        "source": "non_comparable_flat_manifold",
        "provenance_source": "observer_payload_non_comparable",
        "comparability_status": "NON_COMPARABLE",
        "track_nmi": {},
        "track_metrics": {},
        "trust_level": "NON_COMPARABLE",
        "reason": reason,
    }
    validation_path = run_dir / "validation.json"
    validation_path.write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")

    report_payload = {
        "schema_version": "1.0",
        "contract_version": "1.0",
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_pass": False,
        "status": "NON_COMPARABLE",
        "verification_status": "NON_COMPARABLE",
        "comparability_status": "NON_COMPARABLE",
        "dataset_hash": summary["dataset_hash"],
        "code_hash_or_commit": summary["code_hash_or_commit"],
        "weights_hash": summary["weights_hash"],
        "kernel_params": summary["kernel_params"],
        "rks_dim": summary["rks_dim"],
        "crn_seed": summary["crn_seed"],
        "alpha": summary["alpha"],
        "layers": [
            {
                "layer_id": run_dir.name,
                "layer_name": run_dir.name,
                "status": "NON_COMPARABLE",
                "checks": [],
                "fail_reasons": [reason],
            }
        ],
    }
    report_path = run_dir / "verification_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    walker_counts = _summarize_walker_state_counts(run_dir)
    _write_verification_summary_csv(
        run_dir,
        layer_id=run_dir.name,
        layer_name=run_dir.name,
        status="NON_COMPARABLE",
        fail_reasons=[reason],
        n_broken=walker_counts["n_broken"],
        n_trapped=walker_counts["n_trapped"],
        n_total=walker_counts["n_total"],
    )

    _emit_no_data_ablation_summary(run_dir)
    _emit_control_metrics_json(run_dir)

    return {
        "status": "success",
        "mode": "non_comparable",
        "reason": reason,
        "run_dir": str(run_dir),
        "baseline_meta": str(baseline_meta),
        "baseline_state": str(baseline_state_path),
        "validation_json": str(validation_path),
        "verification_report": str(report_path),
    }


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


def _stable_short_hash(value: Any) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _normalize_run_provenance(payload: Optional[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    meta = meta if isinstance(meta, dict) else {}

    existing = payload.get("provenance")
    provenance: Dict[str, Any] = {}
    if isinstance(existing, dict):
        provenance.update(existing)
    elif isinstance(existing, list):
        for entry in existing:
            if isinstance(entry, dict):
                metadata = entry.get("metadata", {})
                if isinstance(metadata, dict):
                    provenance.update(metadata)
                for key in ("basis_hash", "weights_hash", "crn_seed", "alpha", "canonical_ids"):
                    if entry.get(key) is not None:
                        provenance[key] = entry.get(key)

    meta_provenance = meta.get("provenance")
    if isinstance(meta_provenance, dict):
        provenance.update(meta_provenance)

    dirichlet_provenance = payload.get("dirichlet_provenance")
    if isinstance(dirichlet_provenance, dict):
        for key in ("basis_hash", "weights_hash", "crn_seed", "alpha"):
            if dirichlet_provenance.get(key) is not None and key not in provenance:
                provenance[key] = dirichlet_provenance.get(key)

    rks_basis_state = payload.get("rks_basis_state")
    if isinstance(rks_basis_state, dict):
        if rks_basis_state.get("hash"):
            provenance.setdefault("basis_hash", rks_basis_state.get("hash"))
        if rks_basis_state.get("seed") is not None:
            provenance.setdefault("basis_seed", rks_basis_state.get("seed"))
        if rks_basis_state.get("kernel_type"):
            provenance.setdefault("kernel_type", rks_basis_state.get("kernel_type"))
        if rks_basis_state.get("sigma") is not None:
            provenance.setdefault("kernel_sigma", rks_basis_state.get("sigma"))

    canonical_ids = payload.get("bt_uid_list")
    if canonical_ids is not None:
        provenance.setdefault("canonical_ids", canonical_ids)

    if not provenance.get("basis_hash"):
        provenance["basis_hash"] = _stable_short_hash(
            canonical_ids if canonical_ids is not None else payload.get("article_metadata", [])
        )

    if not provenance.get("weights_hash"):
        weight_source = "missing"
        for candidate in (
            payload.get("cls_per_bot_contract"),
            payload.get("spectral_u_axis"),
            payload.get("features"),
            payload.get("embeddings"),
            canonical_ids,
        ):
            if candidate is not None:
                weight_source = candidate
                break
        provenance["weights_hash"] = _stable_short_hash(weight_source)

    if provenance.get("crn_seed") is None:
        provenance["crn_seed"] = meta.get("seed", payload.get("seed", 0))

    if provenance.get("alpha") is None:
        kernel_params = meta.get("kernel_params", {}) if isinstance(meta.get("kernel_params"), dict) else {}
        provenance["alpha"] = kernel_params.get("alpha", 1.0)

    return provenance


def _build_leaf_provenance_summary(run_dir: Path, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    payload = payload if isinstance(payload, dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    provenance = _normalize_run_provenance(payload, meta)

    kernel_params = dict(meta.get("kernel_params", {})) if isinstance(meta.get("kernel_params"), dict) else {}
    if "kernel" in meta:
        kernel_params.setdefault("kernel", str(meta["kernel"]))
    if "channel" in meta:
        kernel_params.setdefault("channel", str(meta["channel"]))
    kernel_params.setdefault("kernel", "unknown")
    kernel_params.setdefault("channel", "main")

    rks_dim = 0
    try:
        arr = np.asarray(payload.get("features"))
        if arr.ndim >= 2:
            rks_dim = int(arr.shape[-1])
    except Exception:
        rks_dim = 0

    return {
        "schema_version": "1.0",
        "cache_version": "1.0",
        "dataset_hash": str(provenance.get("basis_hash") or _stable_short_hash(run_dir.name)),
        "code_hash_or_commit": str(meta.get("git_hash") or _stable_short_hash(meta)),
        "weights_hash": str(provenance.get("weights_hash") or _stable_short_hash("missing")),
        "kernel_params": kernel_params,
        "rks_dim": rks_dim if rks_dim > 0 else 2048,
        "crn_seed": int(provenance.get("crn_seed", meta.get("seed", 0) or 0)),
        "alpha": float(provenance.get("alpha", 1.0)),
        "timestamp_utc": str(meta.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        "verification_status": "UNVERIFIED",
        "provenance_source": "observer_payload",
    }


def _write_verification_summary_csv(
    run_dir: Path,
    *,
    layer_id: str,
    layer_name: str,
    status: str,
    fail_reasons: List[str],
    crn_locked: Optional[bool] = None,
    ordering_pass: Optional[bool] = None,
    seed_stability: Optional[bool] = None,
    mi: Optional[float] = None,
    n_broken: int = 0,
    n_trapped: int = 0,
    n_total: int = 0,
) -> Path:
    summary_path = Path(run_dir) / "verification_summary.csv"
    fieldnames = ["layer_id", "layer_name", "status", "crn_locked", "ordering_pass", "seed_stability", "mi", "n_broken", "n_trapped", "n_total", "fail_reasons"]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "layer_id": str(layer_id),
                "layer_name": str(layer_name),
                "status": str(status),
                "crn_locked": crn_locked,
                "ordering_pass": ordering_pass,
                "seed_stability": seed_stability,
                "mi": mi,
                "n_broken": int(n_broken),
                "n_trapped": int(n_trapped),
                "n_total": int(n_total),
                "fail_reasons": "; ".join(str(reason) for reason in (fail_reasons or []) if str(reason).strip()),
            }
        )
    return summary_path


def _summarize_walker_state_counts(run_dir: Path) -> Dict[str, int]:
    walker_states_path = Path(run_dir) / "walker_states.json"
    if not walker_states_path.exists():
        return {"n_broken": 0, "n_trapped": 0, "n_total": 0}
    try:
        payload = json.loads(walker_states_path.read_text(encoding="utf-8"))
    except Exception:
        return {"n_broken": 0, "n_trapped": 0, "n_total": 0}
    if not isinstance(payload, list):
        return {"n_broken": 0, "n_trapped": 0, "n_total": 0}

    n_broken = 0
    n_trapped = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        state = str(item.get("raw_state") or item.get("label") or item.get("status") or "").strip().lower()
        if state == "broken":
            n_broken += 1
        elif state == "trapped":
            n_trapped += 1
    return {"n_broken": int(n_broken), "n_trapped": int(n_trapped), "n_total": int(len(payload))}


def _infer_verification_exp_dir(run_dir: Path) -> Optional[Path]:
    run_dir = Path(run_dir).resolve()
    for candidate in [run_dir, *run_dir.parents]:
        if candidate.name.startswith("experiments_"):
            return candidate

    if run_dir.name in {"real", "control_random", "control_shuffled", "control_constant"}:
        try:
            return run_dir.parents[2]
        except IndexError:
            return run_dir.parent
    if run_dir.parent.name in {"cls", "logits"}:
        try:
            return run_dir.parents[2]
        except IndexError:
            return run_dir.parent
    return run_dir.parent if run_dir.parent != run_dir else None


def _resolve_verification_layer_for_leaf(all_layers: List[Dict[str, Any]], run_dir: Path, exp_dir: Path) -> Optional[Dict[str, Any]]:
    resolved_run_dir = Path(run_dir).resolve()
    for layer in all_layers:
        layer_dir = Path(layer.get("layer_dir", exp_dir))
        if layer_dir.exists() and layer_dir.resolve() == resolved_run_dir:
            return layer

        artifacts = layer.get("artifacts", {}) or {}
        corpora = list(artifacts.keys()) if isinstance(artifacts, dict) and artifacts else ["real"]
        for corpus in corpora:
            candidate = layer_dir / str(corpus)
            if (candidate / "MONOLITH_DATA.csv").exists() and candidate.resolve() == resolved_run_dir:
                return layer

        if (layer_dir / "MONOLITH_DATA.csv").exists() and layer_dir.resolve() == resolved_run_dir:
            return layer
    return None


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

    observer = _load_primary_observer_payload(run_dir)

    if isinstance(observer, dict):
        payload = _build_leaf_provenance_summary(run_dir, observer)
        payload["verification_status"] = verification_status
        out = run_dir / "baseline_meta.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    payload = {
        "schema_version": "1.0",
        "cache_version": "1.0",
        "dataset_hash": "suite-generated",
        "code_hash_or_commit": "suite-generated",
        "weights_hash": "suite-generated",
        "synthetic_placeholder": True,
        "provenance_source": "suite-generated-placeholder",
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


def _ensure_verification_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    report_path = run_dir / "verification_report.json"
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                required = {"run_id", "timestamp", "layers", "global_pass"}
                if required.issubset(existing.keys()):
                    return report_path
        except Exception:
            pass

    observer_payload = _load_primary_observer_payload(run_dir)
    provenance_summary = _build_leaf_provenance_summary(run_dir, observer_payload)
    now_iso = datetime.now(timezone.utc).isoformat()
    placeholder = {
        "schema_version": "1.0",
        "contract_version": "1.0",
        "run_id": run_dir.name,
        "timestamp": now_iso,
        "global_pass": False,
        "layers": [
            {
                "layer_id": run_dir.name,
                "layer_name": run_dir.name,
                "status": "UNVERIFIED",
                "checks": [],
                "fail_reasons": ["verification layer not materialized during bundle emission"],
            }
        ],
        "dataset_hash": provenance_summary["dataset_hash"],
        "code_hash_or_commit": provenance_summary["code_hash_or_commit"],
        "weights_hash": provenance_summary["weights_hash"],
        "kernel_params": provenance_summary["kernel_params"],
        "rks_dim": provenance_summary["rks_dim"],
        "crn_seed": provenance_summary["crn_seed"],
        "alpha": provenance_summary["alpha"],
        "verification_status": "UNVERIFIED",
    }
    report_path.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
    walker_counts = _summarize_walker_state_counts(run_dir)
    _write_verification_summary_csv(
        run_dir,
        layer_id=run_dir.name,
        layer_name=run_dir.name,
        status="UNVERIFIED",
        fail_reasons=["verification layer not materialized during bundle emission"],
        n_broken=walker_counts["n_broken"],
        n_trapped=walker_counts["n_trapped"],
        n_total=walker_counts["n_total"],
    )
    return report_path


def _safe_float(value: Any, default: Any = 0.0) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        if default is None:
            return None
        return float(default)
    if not math.isfinite(out):
        if default is None:
            return None
        return float(default)
    return out


def _safe_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "pass"}:
        return True
    if text in {"false", "0", "no", "fail"}:
        return False
    return None


def _normalize_rows_for_relativity(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        idx = int(row.get("index", i))
        normalized_row = dict(row)
        normalized_row.update(
            {
                "index": idx,
                "bt_uid": str(row.get("bt_uid", f"article_{idx}")),
                "title": str(row.get("title", "")),
                "zone": str(row.get("zone", "unknown")),
                "verdict": str(row.get("verdict", "UNKNOWN")).upper(),
                "density": _safe_float(row.get("density", 0.0)),
                "stress": _safe_float(row.get("stress", 0.0)),
                "z_height": _safe_float(row.get("z_height", 0.0)),
                "source": str(row.get("source", "")),
                "perspective_tag": str(row.get("perspective_tag", "")),
            }
        )
        normalized.append(normalized_row)
    return normalized


def _build_article_maps(payload: Dict[str, Any], n_articles: int) -> Dict[str, Dict[int, Any]]:
    article_metadata = payload.get("article_metadata") if isinstance(payload.get("article_metadata"), list) else []
    walker_states = payload.get("walker_states") if isinstance(payload.get("walker_states"), list) else []
    phantom_verdicts = payload.get("phantom_verdicts") if isinstance(payload.get("phantom_verdicts"), list) else []
    walker_paths = payload.get("walker_paths") if isinstance(payload.get("walker_paths"), list) else []

    metadata_by_idx: Dict[int, Dict[str, Any]] = {}
    for i, item in enumerate(article_metadata):
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", i))
        metadata_by_idx[idx] = item

    state_by_idx: Dict[int, Dict[str, Any]] = {}
    for i, item in enumerate(walker_states):
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", i))
        state_by_idx[idx] = item

    verdict_by_idx: Dict[int, Dict[str, Any]] = {}
    for i, item in enumerate(phantom_verdicts):
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", i))
        verdict_by_idx[idx] = item

    path_by_idx: Dict[int, Dict[str, Any]] = {}
    for i, item in enumerate(walker_paths):
        if not isinstance(item, dict):
            continue
        idx = int(item.get("article_idx", i))
        path_by_idx[idx] = item

    for idx in range(n_articles):
        metadata_by_idx.setdefault(idx, {"index": idx})
        state_by_idx.setdefault(idx, {})
        verdict_by_idx.setdefault(idx, {})

    return {
        "metadata": metadata_by_idx,
        "state": state_by_idx,
        "verdict": verdict_by_idx,
        "path": path_by_idx,
    }


def _summarize_path_trace(path_blob: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(path_blob, dict):
        return {
            "mean_work": 0.0,
            "event_rate": 0.0,
            "dominant_axis_index": -1,
            "dominant_axis_label": "unknown",
        }
    diags = path_blob.get("step_diagnostics")
    if not isinstance(diags, list) or not diags:
        return {
            "mean_work": 0.0,
            "event_rate": 0.0,
            "dominant_axis_index": -1,
            "dominant_axis_label": "unknown",
        }
    total_work = 0.0
    event_hits = 0
    axis_weights: Optional[np.ndarray] = None
    label_counts: Dict[str, int] = {}
    for item in diags:
        if not isinstance(item, dict):
            continue
        total_work += _safe_float(item.get("step_work", 0.0))
        event_hits += 1 if bool(item.get("event_active", False)) else 0
        axis_vec = item.get("step_axis_vector")
        if axis_vec is not None:
            arr = np.asarray(axis_vec, dtype=float).reshape(-1)
            if arr.size:
                if axis_weights is None:
                    axis_weights = np.zeros_like(arr, dtype=float)
                if arr.shape == axis_weights.shape:
                    axis_weights += arr
        axis_label = str(item.get("dominant_axis_label", "") or "").strip()
        if axis_label:
            label_counts[axis_label] = label_counts.get(axis_label, 0) + 1

    dominant_axis_index = -1
    if axis_weights is not None and axis_weights.size:
        dominant_axis_index = int(np.argmax(axis_weights))
    dominant_axis_label = "unknown"
    if label_counts:
        dominant_axis_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
    elif dominant_axis_index >= 0:
        dominant_axis_label = f"bot_{dominant_axis_index}"

    return {
        "mean_work": float(total_work / max(len(diags), 1)),
        "event_rate": float(event_hits / max(len(diags), 1)),
        "dominant_axis_index": dominant_axis_index,
        "dominant_axis_label": dominant_axis_label,
    }


def _nearest_neighbor_indices(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n <= 1:
        return np.full((n,), -1, dtype=int)
    deltas = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.argmin(dist, axis=1).astype(int)


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=float).reshape(-1)
    b = np.asarray(vec_b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def _emit_relativity_from_payload(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _load_primary_observer_payload(run_dir)
    if not isinstance(payload, dict):
        return {"status": "fallback", "reason": "observer payload unavailable"}

    normalized_rows = _normalize_rows_for_relativity(rows)
    n_articles = len(normalized_rows)
    if n_articles == 0:
        return {"status": "fallback", "reason": "no MONOLITH rows available"}

    ordered_uids = [str(row.get("bt_uid", "")) for row in normalized_rows]

    def _ordered_metric_matrix_from_payload(candidate_payload: Dict[str, Any]) -> Optional[np.ndarray]:
        if not isinstance(candidate_payload, dict):
            return None
        matrix = candidate_payload.get("features")
        if matrix is None:
            matrix = candidate_payload.get("embeddings")
        if matrix is None:
            return None
        if TORCH_AVAILABLE and torch.is_tensor(matrix):
            matrix = matrix.detach().cpu().numpy()
        try:
            arr = np.asarray(matrix, dtype=np.float64)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        ids = candidate_payload.get("bt_uid_list")
        if not isinstance(ids, (list, tuple)):
            metadata = candidate_payload.get("article_metadata", [])
            if isinstance(metadata, list):
                ids = [str((row or {}).get("bt_uid", "")) for row in metadata if isinstance(row, dict)]
        if isinstance(ids, (list, tuple)) and len(ids) >= n_articles:
            id_to_idx = {str(uid): i for i, uid in enumerate(ids)}
            if all(uid in id_to_idx for uid in ordered_uids):
                return np.asarray([arr[id_to_idx[uid]] for uid in ordered_uids], dtype=np.float64)
        if arr.shape[0] < n_articles:
            return None
        return arr[:n_articles]

    def _project_shared_basis(global_matrix: np.ndarray, observer_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = np.asarray(global_matrix, dtype=np.float64)
        o = np.asarray(observer_matrix, dtype=np.float64)
        if g.ndim != 2 or o.ndim != 2:
            raise ValueError("shared-basis projection requires 2D matrices")
        d = min(g.shape[1], o.shape[1])
        if d <= 0:
            raise ValueError("shared-basis projection requires non-empty feature dims")
        g = g[:, :d]
        o = o[:, :d]
        mean_vec = np.mean(g, axis=0, keepdims=True)
        g_centered = g - mean_vec
        o_centered = o - mean_vec
        if g_centered.shape[0] < 2 or d == 1:
            g_proj = g_centered[:, :1]
            o_proj = o_centered[:, :1]
        else:
            _, _, vt = np.linalg.svd(g_centered, full_matrices=False)
            n_comp = max(1, min(3, vt.shape[0], d))
            basis = vt[:n_comp].T
            g_proj = g_centered @ basis
            o_proj = o_centered @ basis
        if g_proj.shape[1] < 3:
            pad = 3 - g_proj.shape[1]
            g_proj = np.pad(g_proj, ((0, 0), (0, pad)), constant_values=0.0)
            o_proj = np.pad(o_proj, ((0, 0), (0, pad)), constant_values=0.0)
        return g_proj[:, :3], o_proj[:, :3]

    probes = np.asarray(payload.get("spectral_probe_magnitudes"), dtype=float)
    works = np.asarray(payload.get("walker_work_integrals"), dtype=float).reshape(-1)
    if probes.ndim != 2 or probes.shape[0] < n_articles:
        return {"status": "fallback", "reason": "spectral_probe_magnitudes missing or malformed"}
    if works.ndim != 1 or works.shape[0] < n_articles:
        return {"status": "fallback", "reason": "walker_work_integrals missing or malformed"}

    article_maps = _build_article_maps(payload, n_articles)
    path_summaries = {
        idx: _summarize_path_trace(article_maps["path"].get(idx))
        for idx in range(n_articles)
    }

    baseline_metric_matrix = _ordered_metric_matrix_from_payload(payload)
    if baseline_metric_matrix is None:
        return {"status": "fallback", "reason": "root feature matrix unavailable"}
    try:
        baseline_coords, _ = _project_shared_basis(baseline_metric_matrix, baseline_metric_matrix)
    except Exception as proj_err:
        return {"status": "fallback", "reason": f"shared-basis projection failed: {proj_err}"}

    try:
        from core.complete_pipeline import _compute_alignment_metrics, _extract_validation_label_info
    except Exception:
        _compute_alignment_metrics = None
        _extract_validation_label_info = None

    label_info = None
    baseline_alignment = None
    if callable(_extract_validation_label_info) and callable(_compute_alignment_metrics):
        try:
            label_info = _extract_validation_label_info(
                articles_for_labels=normalized_rows,
                metadata_for_labels=normalized_rows,
                allow_source_fallback=True,
            )
            if label_info is not None:
                baseline_alignment = _compute_alignment_metrics(
                    baseline_metric_matrix if baseline_metric_matrix is not None else baseline_coords,
                    label_info,
                )
        except Exception:
            label_info = None
            baseline_alignment = None
    baseline_nn = _nearest_neighbor_indices(baseline_coords)
    global_probe = np.asarray(probes[:n_articles], dtype=float)
    probe_norms = np.linalg.norm(global_probe, axis=1, keepdims=True)
    probe_norms[probe_norms <= 0.0] = 1.0
    probe_unit = global_probe / probe_norms
    global_probe_mean = np.mean(global_probe, axis=0)
    global_work_mean = float(np.mean(works[:n_articles]))
    global_event_rate = float(np.mean([path_summaries[idx]["event_rate"] for idx in range(n_articles)]))
    global_anomaly_rate = float(
        np.mean(
            [
                1.0 if bool(article_maps["state"].get(idx, {}).get("anomaly_flag", False)) else 0.0
                for idx in range(n_articles)
            ]
        )
    )
    global_survival_pct = float(
        np.mean(
            [
                0.0 if bool(article_maps["state"].get(idx, {}).get("anomaly_flag", False)) else 100.0
                for idx in range(n_articles)
            ]
        )
    )
    global_top1_evr = 0.0
    if global_probe_mean.size:
        denom = float(np.sum(np.abs(global_probe_mean)))
        if denom > 0.0:
            global_top1_evr = float(np.max(np.abs(global_probe_mean)) / denom)

    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    observer_payload_paths = sorted(rel_dir.glob("observer_*.pt"))
    if not observer_payload_paths:
        return {
            "status": "fallback",
            "reason": "observer-conditioned relativity payloads missing",
        }

    written_state = 0
    written_delta = 0
    for idx in range(n_articles):
        # NUCLEAR DAG: Load the physically distinct universe for this observer
        obs_pt_path = rel_dir / f"observer_{idx}.pt"
        current_obs_payload = payload  # Default to global
        if obs_pt_path.exists() and TORCH_AVAILABLE:
            try:
                current_obs_payload = torch.load(obs_pt_path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        
        # Build map for CURRENT observer's results
        obs_article_maps = _build_article_maps(current_obs_payload, n_articles)
        obs_path_summaries = {
            j: _summarize_path_trace(obs_article_maps["path"].get(j))
            for j in range(n_articles)
        }
        observer_metric_matrix = _ordered_metric_matrix_from_payload(current_obs_payload)
        if observer_metric_matrix is None:
            observer_metric_matrix = baseline_metric_matrix

        focus_row = normalized_rows[idx]
        
        # Spectral axis from observer's own universe
        obs_probes = np.asarray(current_obs_payload.get("spectral_probe_magnitudes", probes), dtype=float)
        obs_focus_probe = obs_probes[idx]
        obs_probe_norms = np.linalg.norm(obs_probes, axis=1, keepdims=True)
        obs_probe_norms[obs_probe_norms <= 0.0] = 1.0
        obs_probe_unit = obs_probes / obs_probe_norms
        
        # Similarity in the warped space
        probe_distance = 1.0 - np.clip(np.sum(obs_probe_unit * (obs_focus_probe / np.linalg.norm(obs_focus_probe).clip(min=1e-9)), axis=1), -1.0, 1.0)

        try:
            centered_baseline, observer_coords = _project_shared_basis(baseline_metric_matrix, observer_metric_matrix)
        except Exception:
            centered_baseline = baseline_coords
            observer_coords = baseline_coords.copy()

        # Strict Einsteinian displacement in the shared Minkowski projection space.
        delta_vectors = observer_coords - centered_baseline
        coord_delta = np.linalg.norm(delta_vectors, axis=1)
        observer_nn = _nearest_neighbor_indices(observer_coords)
        translation_nn = _nearest_neighbor_indices(centered_baseline)
        flip_mask = observer_nn != baseline_nn
        flip_count = int(np.sum(flip_mask))
        translation_flip_count = int(np.sum(translation_nn != baseline_nn))
        
        observer_alignment = None
        if label_info is not None and callable(_compute_alignment_metrics):
            try:
                observer_alignment = _compute_alignment_metrics(
                    observer_metric_matrix if observer_metric_matrix is not None else observer_coords,
                    label_info,
                )
            except Exception:
                observer_alignment = None

        focus_meta = obs_article_maps["metadata"].get(idx, {})
        focus_state = obs_article_maps["state"].get(idx, {})
        focus_verdict = obs_article_maps["verdict"].get(idx, {})
        focus_path = obs_path_summaries.get(idx, {})
        focus_probe_raw = obs_probes[idx]
        probe_sum = float(np.sum(np.abs(focus_probe_raw)))
        focus_top1_evr = 0.0
        if probe_sum > 0.0:
            focus_top1_evr = float(np.max(np.abs(focus_probe_raw)) / probe_sum)

        articles_blob: List[Dict[str, Any]] = []
        for row_j, base_coord, obs_coord, delta_vec, delta_j, sim_j in zip(
            normalized_rows,
            centered_baseline,
            observer_coords,
            delta_vectors,
            coord_delta,
            1.0 - probe_distance,
        ):
            article_row = dict(row_j)
            article_row.update(
                {
                    "index": int(row_j["index"]),
                    "bt_uid": row_j["bt_uid"],
                    "title": row_j["title"],
                    "zone": row_j["zone"],
                    "verdict": row_j["verdict"],
                    "density": row_j["density"],
                    "stress": row_j["stress"],
                    "z_height": row_j["z_height"],
                    "baseline_x": float(base_coord[0]),
                    "baseline_y": float(base_coord[1]),
                    "baseline_z": float(base_coord[2]),
                    "observer_x": float(obs_coord[0]),
                    "observer_y": float(obs_coord[1]),
                    "observer_z": float(obs_coord[2]),
                    "delta_x": float(delta_vec[0]),
                    "delta_y": float(delta_vec[1]),
                    "delta_z": float(delta_vec[2]),
                    "observer_probe_similarity": float(sim_j),
                    "coord_delta": float(delta_j),
                }
            )
            articles_blob.append(article_row)

        ranked_flip_indices = np.argsort(-coord_delta)
        path_flip_delta: Dict[str, float] = {}
        for j in ranked_flip_indices[:8]:
            key = f"{normalized_rows[j]['bt_uid']}|{normalized_rows[j]['title'][:48]}"
            path_flip_delta[key] = float(coord_delta[j])

        state_payload = {
            "observer_id": idx,
            "articles": articles_blob,
            "paths": [f"observer_{idx}/MONOLITH.html"],
            "axes": {
                "x": "shared_basis_axis_1",
                "y": "shared_basis_axis_2",
                "z": "shared_basis_axis_3",
            },
            "metrics": {
                "observer_bt_uid": focus_row["bt_uid"],
                "observer_title": focus_row["title"],
                "observer_zone": focus_row["zone"],
                "observer_verdict": str(focus_verdict.get("verdict", focus_row["verdict"])).upper(),
                "observer_density": focus_row["density"],
                "observer_stress": focus_row["stress"],
                "observer_z_height": focus_row["z_height"],
                "observer_axis_index": int(focus_path.get("dominant_axis_index", -1)),
                "observer_axis_label": str(focus_path.get("dominant_axis_label", "unknown")),
                "observer_mean_work": float(focus_path.get("mean_work", _safe_float(works[idx]))),
                "observer_event_rate": float(focus_path.get("event_rate", 0.0)),
                "global_mean_work": global_work_mean,
                "global_event_rate": global_event_rate,
                "global_survival_pct": global_survival_pct,
                "observer_survival_pct": 0.0 if bool(focus_state.get("anomaly_flag", False)) else 100.0,
                "global_nmi": float(baseline_alignment.get("nmi")) if isinstance(baseline_alignment, dict) and isinstance(baseline_alignment.get("nmi"), (int, float)) else None,
                "global_ari": float(baseline_alignment.get("ari")) if isinstance(baseline_alignment, dict) and isinstance(baseline_alignment.get("ari"), (int, float)) else None,
                "observer_conditioned_nmi": float(observer_alignment.get("nmi")) if isinstance(observer_alignment, dict) and isinstance(observer_alignment.get("nmi"), (int, float)) else None,
                "observer_conditioned_ari": float(observer_alignment.get("ari")) if isinstance(observer_alignment, dict) and isinstance(observer_alignment.get("ari"), (int, float)) else None,
                "observer_track_nmi": {"SYN": float(observer_alignment.get("nmi"))} if isinstance(observer_alignment, dict) and isinstance(observer_alignment.get("nmi"), (int, float)) else {},
            },
            "provenance": {
                "source": "observer_payload_relativity_v1",
                "synthetic_placeholder": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "basis_hash": str(payload.get("provenance", {}).get("basis_hash", "")) if isinstance(payload.get("provenance"), dict) else "",
                "observer_bt_uid": focus_row["bt_uid"],
            },
        }

        state_path = rel_dir / f"state_{idx}.json"
        state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        written_state += 1

        delta_payload = {
            "observer_id": idx,
            "null_observer_equivalence": {
                "equivalent": bool(np.max(coord_delta) <= 1e-9 and flip_count == 0),
                "max_coord_delta": float(np.max(coord_delta)),
                "path_flip_count": flip_count,
                "axis_rotation_deg": float(_angle_deg(focus_probe_raw, global_probe_mean)),
            },
            "path_flip_delta": path_flip_delta,
            "metrics_delta": {
                "d_rupture_rate": float((1.0 if bool(focus_state.get("anomaly_flag", False)) else 0.0) - global_anomaly_rate),
                "d_mean_work": float(focus_path.get("mean_work", _safe_float(works[idx])) - global_work_mean),
                "d_survival_pct": float((0.0 if bool(focus_state.get("anomaly_flag", False)) else 100.0) - global_survival_pct),
                "d_nmi": (
                    float(observer_alignment.get("nmi") - baseline_alignment.get("nmi"))
                    if isinstance(observer_alignment, dict)
                    and isinstance(baseline_alignment, dict)
                    and isinstance(observer_alignment.get("nmi"), (int, float))
                    and isinstance(baseline_alignment.get("nmi"), (int, float))
                    else None
                ),
                "d_ari": (
                    float(observer_alignment.get("ari") - baseline_alignment.get("ari"))
                    if isinstance(observer_alignment, dict)
                    and isinstance(baseline_alignment, dict)
                    and isinstance(observer_alignment.get("ari"), (int, float))
                    and isinstance(baseline_alignment.get("ari"), (int, float))
                    else None
                ),
            },
            "axis_delta": {
                "rotation_deg": float(_angle_deg(focus_probe_raw, global_probe_mean)),
                "d_explained_variance_axis1": float(focus_top1_evr - global_top1_evr),
            },
            "translation_only_comparison": {
                "d_path_flip_count": int(flip_count - translation_flip_count),
                "d_mean_work": float(focus_path.get("mean_work", _safe_float(works[idx])) - global_work_mean),
            },
            "provenance": {
                "source": "observer_payload_relativity_v1",
                "synthetic_placeholder": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "observer_bt_uid": focus_row["bt_uid"],
            },
        }
        delta_path = rel_dir / f"delta_{idx}.json"
        delta_path.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")
        written_delta += 1

    return {
        "status": "success",
        "state_files": written_state,
        "delta_files": written_delta,
        "mode": "observer_payload_relativity_v1",
    }


def _emit_relativity_defaults(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    real = _emit_relativity_from_payload(run_dir, rows)
    if real.get("status") == "success":
        return {
            "state_files": int(real.get("state_files", 0)),
            "delta_files": int(real.get("delta_files", 0)),
            "mode": str(real.get("mode", "observer_payload_relativity_v1")),
        }

    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    written_state = 0
    written_delta = 0
    indices = [int(r.get("index", i)) for i, r in enumerate(rows)]
    for idx in indices:
        state_path = rel_dir / f"state_{idx}.json"
        delta_path = rel_dir / f"delta_{idx}.json"
        state_payload = {
            "observer_id": idx,
            "articles": rows,
            "paths": [f"observer_{idx}/MONOLITH.html"],
            "axes": {"x": "density", "y": "stress"},
            "metrics": {},
            "synthetic_placeholder": True,
            "message": str(real.get("reason", "observer payload unavailable")),
            "provenance": {"source": "suite-default", "synthetic_placeholder": True},
        }
        state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        written_state += 1
        delta_payload = {
            "observer_id": idx,
            "synthetic_placeholder": True,
            "message": str(real.get("reason", "observer payload unavailable")),
            "null_observer_equivalence": {"max_coord_delta": 0.0, "path_flip_count": 0, "axis_rotation_deg": 0.0},
            "path_flip_delta": {},
            "metrics_delta": {"d_rupture_rate": 0.0, "d_mean_work": 0.0, "d_survival_pct": 0.0},
            "axis_delta": {"rotation_deg": 0.0, "d_explained_variance_axis1": 0.0},
        }
        delta_path.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")
        written_delta += 1
    return {
        "state_files": written_state,
        "delta_files": written_delta,
        "mode": "suite-default",
        "fallback_reason": str(real.get("reason", "observer payload unavailable")),
    }


def _emit_label_derivatives(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    labels_dir = run_dir / "labels"
    derived_dir = labels_dir / "derived"
    labels_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    def _clean_group_value(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() in {"none", "nan", "null"}:
                continue
            return text
        return "unknown"

    hidden_rows: List[Dict[str, Any]] = []
    token_sets: Dict[str, set[str]] = {}
    counts: Dict[str, int] = {}

    for row in rows:
        group_source = _clean_group_value(row.get("source"))
        group_publication = _clean_group_value(row.get("publication"), row.get("source"))
        group_perspective_type = _clean_group_value(row.get("perspective_type"))
        group_perspective_tag = _clean_group_value(row.get("perspective_tag"))
        group_bias = _clean_group_value(row.get("bias"))
        group_affiliation = _clean_group_value(row.get("affiliation"))
        group_label = _clean_group_value(row.get("label"))
        group_topic = _clean_group_value(
            row.get("group_topic"),
            row.get("topic"),
            row.get("subtopic"),
            row.get("perspective_tag"),
            row.get("perspective_type"),
            row.get("label"),
            row.get("bias"),
            row.get("publication"),
            row.get("source"),
        )

        hidden_row = {
            "article_id": int(row.get("index", 0)),
            "bt_uid": row.get("bt_uid"),
            "group_topic": group_topic,
            "group_source": group_source,
            "group_publication": group_publication,
            "group_perspective_type": group_perspective_type,
            "group_perspective_tag": group_perspective_tag,
            "group_bias": group_bias,
            "group_affiliation": group_affiliation,
            "group_label": group_label,
        }
        hidden_rows.append(hidden_row)

        counts[group_topic] = counts.get(group_topic, 0) + 1
        token_bucket = token_sets.setdefault(group_topic, set())
        for token in (
            group_source,
            group_publication,
            group_perspective_type,
            group_perspective_tag,
            group_bias,
            group_affiliation,
            group_label,
        ):
            if token != "unknown":
                token_bucket.add(token)

    hidden_csv = labels_dir / "hidden_groups.csv"
    with hidden_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "article_id",
                "bt_uid",
                "group_topic",
                "group_source",
                "group_publication",
                "group_perspective_type",
                "group_perspective_tag",
                "group_bias",
                "group_affiliation",
                "group_label",
            ],
        )
        writer.writeheader()
        for hidden_row in hidden_rows:
            writer.writerow(hidden_row)

    groups = sorted(counts.keys())
    summaries = []
    for g in groups:
        tokens = sorted(token_sets.get(g, set()))
        summaries.append(
            {
                "group_name": g,
                "n_articles": counts.get(g, 0),
                "top_markers": tokens[:5],
                "label_source": "metadata_hidden_groups",
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
                left = token_sets.get(gi, set())
                right = token_sets.get(gj, set())
                union = left | right
                if not union:
                    row_vals.append(1.0)
                else:
                    overlap = len(left & right) / float(len(union))
                    row_vals.append(1.0 - overlap)
        matrix.append(row_vals)

    group_matrix = {"groups": groups, "cost_matrix": matrix, "label_source": "metadata_token_overlap"}
    (derived_dir / "group_matrix.json").write_text(json.dumps(group_matrix, indent=2), encoding="utf-8")
    return {
        "hidden_groups": str(hidden_csv),
        "group_summaries": str(derived_dir / "group_summaries.json"),
        "group_matrix": str(derived_dir / "group_matrix.json"),
    }


def _emit_validation_json(run_dir: Path) -> Path:
    validation_path = run_dir / "validation.json"
    existing: Dict[str, Any] = {}
    if validation_path.exists():
        try:
            loaded = json.loads(validation_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    current_nmi = existing.get("nmi")
    if isinstance(current_nmi, (int, float)) and not isinstance(current_nmi, bool) and math.isfinite(float(current_nmi)) and 0.0 <= float(current_nmi) <= 1.0:
        normalized = _normalize_validation_payload(existing)
        if normalized is not None:
            validation_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        return validation_path

    # Repair older validation payloads that already contain per-track metrics but
    # omitted the top-level synthesis NMI required by the consumer contract.
    repaired = _repair_validation_payload(existing)
    if repaired is not None:
        validation_path.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
        return validation_path

    observer_payload = _load_primary_observer_payload(run_dir)
    rows = _load_monolith_rows(run_dir / "MONOLITH_DATA.csv")
    if not rows:
        rows = _load_metadata_rows(run_dir / "article_metadata.csv")
        if not rows and isinstance(observer_payload, dict):
            metadata = observer_payload.get("article_metadata")
            if isinstance(metadata, list):
                for i, item in enumerate(metadata):
                    if not isinstance(item, dict):
                        continue
                    rows.append(
                        {
                            "index": int(item.get("index", i)),
                            "bt_uid": str(item.get("bt_uid", f"article_{i}")),
                            "title": str(item.get("title", ""))[:200],
                            "zone": str(item.get("zone", "non_comparable")) or "non_comparable",
                            "density": str(item.get("density", 0)),
                            "stress": str(item.get("stress", 0)),
                            "z_height": str(item.get("z_height", 0)),
                            "source": str(item.get("source", "")),
                            "perspective_tag": str(item.get("perspective_tag", "")),
                        }
                    )
    validation_payload = dict(existing) if isinstance(existing, dict) else {}

    try:
        from core.complete_pipeline import _compute_alignment_metrics, _extract_validation_label_info
    except Exception:
        _compute_alignment_metrics = None
        _extract_validation_label_info = None

    alignment_metrics = None
    track_metrics: Dict[str, Dict[str, Any]] = {}

    def _as_metric_array_local(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if TORCH_AVAILABLE:
            try:
                import torch as torch_lib
                if torch_lib.is_tensor(value):
                    value = value.detach().cpu().numpy()
            except Exception:
                pass
        try:
            arr = np.asarray(value, dtype=np.float64)
        except Exception:
            return None
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2:
            return None
        return arr

    def _simple_zone_alignment_metrics(matrix: Any) -> Optional[Dict[str, Any]]:
        arr = _as_metric_array_local(matrix)
        if arr is None or arr.shape[0] < 2:
            return None
        zone_labels = [str(row.get("zone", "")).strip() for row in rows if str(row.get("zone", "")).strip()]
        if len(zone_labels) < 2:
            return None
        n = min(arr.shape[0], len(zone_labels))
        arr = arr[:n]
        zone_labels = zone_labels[:n]
        unique_labels = sorted(set(zone_labels))
        if len(unique_labels) < 2:
            return None
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        except Exception:
            return None
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        truth = np.asarray([label_to_idx[label] for label in zone_labels], dtype=int)
        try:
            pred = KMeans(n_clusters=len(unique_labels), n_init=10, random_state=42).fit_predict(arr)
        except Exception:
            return None
        return {
            "nmi": float(normalized_mutual_info_score(truth, pred)),
            "ari": float(adjusted_rand_score(truth, pred)),
            "n_clusters": int(len(unique_labels)),
            "label_cardinality": int(len(unique_labels)),
            "label_source": "zone_fallback",
            "metric_source": "kmeans_on_final_features",
        }

    if isinstance(observer_payload, dict) and callable(_compute_alignment_metrics) and callable(_extract_validation_label_info):
        label_rows = rows or (observer_payload.get("article_metadata") if isinstance(observer_payload.get("article_metadata"), list) else [])
        try:
            label_info = _extract_validation_label_info(
                articles_for_labels=label_rows,
                metadata_for_labels=observer_payload.get("article_metadata"),
                allow_source_fallback=True,
            )
        except Exception:
            label_info = None
        features = observer_payload.get("features")
        if label_info is not None and features is not None:
            try:
                alignment_metrics = _compute_alignment_metrics(features, label_info)
            except Exception:
                alignment_metrics = None
            try:
                result_like = {
                    "features": observer_payload.get("features"),
                    "spectral_probe_magnitudes": observer_payload.get("spectral_probe_magnitudes"),
                    "dirichlet_fused_std": observer_payload.get("dirichlet_fused_std"),
                    "checkpoints": {
                        "T1_embeddings": observer_payload.get("T1_embeddings"),
                        "T2_kernels": observer_payload.get("T2_kernels"),
                        "T1.5_spectral": observer_payload.get("T1.5_spectral"),
                        "T3_topology": observer_payload.get("T3_topology"),
                    },
                }
                track_metrics = _collect_track_metrics_from_result(
                    result_like,
                    label_info,
                    _compute_alignment_metrics,
                )
            except Exception:
                track_metrics = {}

    if alignment_metrics is None and isinstance(observer_payload, dict):
        alignment_metrics = _simple_zone_alignment_metrics(observer_payload.get("features"))
        if isinstance(alignment_metrics, dict):
            track_inputs = {
                "SYN": observer_payload.get("features"),
                "T1": observer_payload.get("T1_embeddings"),
                "T2": observer_payload.get("T2_kernels"),
                "T1.5": observer_payload.get("spectral_probe_magnitudes"),
            }
            t3_payload = observer_payload.get("T3_topology")
            if isinstance(t3_payload, dict):
                for key in ("dirichlet_fused", "bond_matrix", "crack_matrix"):
                    if t3_payload.get(key) is not None:
                        track_inputs["T3"] = t3_payload.get(key)
                        break
            track_metrics = {}
            for track_key, track_value in track_inputs.items():
                metrics_payload = _simple_zone_alignment_metrics(track_value)
                if isinstance(metrics_payload, dict):
                    track_metrics[track_key] = metrics_payload

    if isinstance(alignment_metrics, dict):
        validation_payload.update(
            {
                "schema_version": "1.0",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "nmi": float(alignment_metrics.get("nmi")) if isinstance(alignment_metrics.get("nmi"), (int, float)) else None,
                "ari": float(alignment_metrics.get("ari")) if isinstance(alignment_metrics.get("ari"), (int, float)) else None,
                "metric_source": alignment_metrics.get("metric_source", "kmeans_on_final_features"),
                "label_source": alignment_metrics.get("label_source", "metadata"),
                "label_cardinality": alignment_metrics.get("label_cardinality"),
                "n_clusters": alignment_metrics.get("n_clusters"),
                "track_nmi": {
                    k: float(v["nmi"])
                    for k, v in track_metrics.items()
                    if isinstance(v, dict) and isinstance(v.get("nmi"), (int, float))
                },
                "track_metrics": track_metrics,
                "trust_level": "MEASURED",
            }
        )
    elif not validation_payload:
        validation_payload = {
            "source": "suite-default",
            "synthetic_placeholder": True,
            "reason": "validation.json missing during bundle emission; canonical validation metrics unavailable",
        }

    validation_path.write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")
    return validation_path


def _emit_no_data_ablation_summary(run_dir: Path) -> Path:
    out = run_dir / "ablation_summary.json"
    if out.exists():
        return _emit_ablation_summary_json(run_dir)
    payload = _build_ablation_summary_payload({}, None)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit_ablation_results_json(run_dir, payload)
    return out


def _emit_no_data_control_results(run_dir: Path) -> Path:
    out = run_dir / "comprehensive_results.json"
    if not out.exists():
        payload = {
            "status": "NO_DATA",
            "source": "suite-default-placeholder",
            "synthetic_placeholder": True,
            "reason": "control analysis not run for this leaf",
            "message": "control analysis not run for this leaf",
            "interpretation": {
                "error": "control analysis not run for this leaf",
                "metrics": {},
                "consensus_residual": {
                    "real": {
                        "consensus_pct": None,
                        "residual_pct": None,
                    }
                },
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit_control_metrics_json(run_dir)
    return out


def _normalize_observer_payloads_for_verification(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    payload_paths: List[Path] = []
    global_payload = run_dir / "observer_global.pt"
    if global_payload.exists():
        payload_paths.append(global_payload)
    payload_paths.extend(list(run_dir.glob("observer_*.pt")))
    payload_paths.extend(sorted((run_dir / "relativity_cache").glob("observer_*.pt")))
    if not payload_paths or not TORCH_AVAILABLE:
        return

    import torch as torch_lib

    for payload_path in payload_paths:
        try:
            payload = torch_lib.load(payload_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                continue
            meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
            normalized = _normalize_run_provenance(payload, meta)
            payload["provenance"] = normalized
            payload.setdefault("meta", {})["provenance"] = normalized
            torch_lib.save(payload, payload_path)
        except Exception:
            continue


def _write_leaf_verification_artifacts(run_dir: Path, exp_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    exp_dir = Path(exp_dir)
    try:
        _normalize_observer_payloads_for_verification(run_dir)
        analysis_path = Path("analysis").resolve()
        if str(analysis_path) not in sys.path:
            sys.path.append(str(analysis_path))
        try:
            from analysis.verification.verify_run import discover_all_layers, verify_layer_data, write_report
        except ImportError:
            from verification.verify_run import discover_all_layers, verify_layer_data, write_report

        all_layers = discover_all_layers(exp_dir)
        target_layer = _resolve_verification_layer_for_leaf(all_layers, run_dir, exp_dir)
        if target_layer is None:
            return {
                "status": "skipped",
                "reason": f"no verification layer discovered for {run_dir} under {exp_dir}",
            }

        report = verify_layer_data(
            target_layer["layer_id"],
            target_layer["layer_name"],
            target_layer["artifacts"],
            Path(target_layer["layer_dir"]),
            exp_dir,
        )
        write_report(
            [report],
            run_dir,
            global_pass_override=(str(report.get("status", "")).upper() == "VERIFIED"),
        )
        walker_counts = _summarize_walker_state_counts(run_dir)
        report_path = run_dir / "verification_report.json"
        if report_path.exists():
            observer_payload = _load_primary_observer_payload(run_dir)
            provenance_summary = _build_leaf_provenance_summary(run_dir, observer_payload)
            report_json = json.loads(report_path.read_text(encoding="utf-8"))
            report_json.update({
                "dataset_hash": provenance_summary["dataset_hash"],
                "code_hash_or_commit": provenance_summary["code_hash_or_commit"],
                "weights_hash": provenance_summary["weights_hash"],
                "kernel_params": provenance_summary["kernel_params"],
                "rks_dim": provenance_summary["rks_dim"],
                "crn_seed": provenance_summary["crn_seed"],
                "alpha": provenance_summary["alpha"],
                "verification_status": str(report.get("status", "")).upper() or "UNVERIFIED",
            })
            report_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        summary_path = run_dir / "verification_summary.csv"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                rows = []
            if rows:
                row = dict(rows[0])
                row["n_broken"] = str(walker_counts["n_broken"])
                row["n_trapped"] = str(walker_counts["n_trapped"])
                row["n_total"] = str(walker_counts["n_total"])
                _write_verification_summary_csv(
                    run_dir,
                    layer_id=str(row.get("layer_id", report.get("layer_id", run_dir.name))),
                    layer_name=str(row.get("layer_name", report.get("layer_name", run_dir.name))),
                    status=str(row.get("status", report.get("status", "UNVERIFIED"))),
                    fail_reasons=[
                        reason.strip()
                        for reason in str(row.get("fail_reasons", "")).split(";")
                        if reason.strip()
                    ],
                    crn_locked=_safe_optional_bool(row.get("crn_locked")),
                    ordering_pass=_safe_optional_bool(row.get("ordering_pass")),
                    seed_stability=_safe_optional_bool(row.get("seed_stability")),
                    mi=_safe_float(row.get("mi"), default=None),
                    n_broken=walker_counts["n_broken"],
                    n_trapped=walker_counts["n_trapped"],
                    n_total=walker_counts["n_total"],
                )
        return {
            "status": "success",
            "layer_id": str(target_layer.get("layer_id", "")),
            "report_path": str(run_dir / "verification_report.json"),
            "summary_path": str(run_dir / "verification_summary.csv"),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "report_path": str(run_dir / "verification_report.json"),
            "summary_path": str(run_dir / "verification_summary.csv"),
        }


def _collect_track_metrics_from_result(
    result: Dict[str, Any],
    label_info: Optional[Dict[str, Any]],
    compute_alignment_metrics,
) -> Dict[str, Dict[str, Any]]:
    if label_info is None or not callable(compute_alignment_metrics):
        return {}

    pipeline_checkpoints = result.get("checkpoints", {}) if isinstance(result.get("checkpoints"), dict) else {}

    def _as_metric_array(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if TORCH_AVAILABLE:
            try:
                import torch as torch_lib
                if torch_lib.is_tensor(value):
                    value = value.detach().cpu().numpy()
            except Exception:
                pass
        try:
            return np.asarray(value, dtype=np.float64)
        except Exception:
            return None

    track_feature_inputs: Dict[str, Any] = {
        "SYN": result.get("features"),
    }
    t1_src = pipeline_checkpoints.get("T1_embeddings")
    if t1_src is None:
        t1_src = pipeline_checkpoints.get("T0_substrate")
    if t1_src is not None:
        t1_arr = _as_metric_array(t1_src)
        if t1_arr is None:
            t1_arr = np.array([], dtype=np.float64)
        if t1_arr.ndim > 2:
            t1_arr = t1_arr.reshape(t1_arr.shape[0], -1)
        track_feature_inputs["T1"] = t1_arr
    t2_src = pipeline_checkpoints.get("T2_kernels")
    if t2_src is not None:
        track_feature_inputs["T2"] = _as_metric_array(t2_src)
    t15_src = result.get("spectral_probe_magnitudes")
    if t15_src is None:
        t15_ckpt = pipeline_checkpoints.get("T1.5_spectral")
        if isinstance(t15_ckpt, dict):
            t15_src = t15_ckpt.get("probe_magnitudes")
    if t15_src is not None:
        track_feature_inputs["T1.5"] = _as_metric_array(t15_src)
    t3_src = result.get("dirichlet_fused_std")
    if t3_src is None:
        t3_ckpt = pipeline_checkpoints.get("T3_topology")
        if isinstance(t3_ckpt, dict):
            for key in ("dirichlet_fused", "bond_matrix", "crack_matrix"):
                if t3_ckpt.get(key) is not None:
                    t3_src = t3_ckpt.get(key)
                    break
    if t3_src is not None:
        track_feature_inputs["T3"] = _as_metric_array(t3_src)

    track_metrics: Dict[str, Dict[str, Any]] = {}
    for track_key, features_np in track_feature_inputs.items():
        arr = _as_metric_array(features_np)
        if arr is None or arr.ndim != 2:
            continue
        metrics_payload = compute_alignment_metrics(arr, label_info)
        if metrics_payload is not None:
            track_metrics[track_key] = metrics_payload
    return track_metrics


def _finalize_synthetic_leaf_artifacts(
    run_dir: Path,
    exp_root: Path,
    *,
    result: Dict[str, Any],
    validation: Dict[str, Any],
    articles: List[Dict[str, Any]],
    kernel: str,
    seed: int,
    compute_alignment_metrics,
    extract_validation_label_info,
) -> Dict[str, Any]:
    label_info = extract_validation_label_info(
        articles_for_labels=articles,
        metadata_for_labels=result.get("article_metadata"),
        allow_source_fallback=True,
    ) if callable(extract_validation_label_info) else None

    alignment_metrics = compute_alignment_metrics(
        result.get("features"),
        label_info,
    ) if callable(compute_alignment_metrics) and label_info is not None else None

    track_metrics = _collect_track_metrics_from_result(
        result,
        label_info,
        compute_alignment_metrics,
    )

    validation_payload = dict(validation)
    validation_payload.update(
        {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "kernel": str(kernel),
            "channel": "cls",
            "seed": int(seed),
            "metric_source": (
                alignment_metrics.get("metric_source")
                if isinstance(alignment_metrics, dict)
                else validation.get("feature_source", "kmeans_on_final_features")
            ),
            "label_source": (
                alignment_metrics.get("label_source")
                if isinstance(alignment_metrics, dict)
                else "unavailable"
            ),
            "label_cardinality": (
                alignment_metrics.get("label_cardinality")
                if isinstance(alignment_metrics, dict)
                else None
            ),
            "n_clusters": (
                alignment_metrics.get("n_clusters")
                if isinstance(alignment_metrics, dict)
                else validation.get("n_clusters")
            ),
            "track_nmi": {
                k: float(v["nmi"])
                for k, v in track_metrics.items()
                if isinstance(v, dict) and isinstance(v.get("nmi"), (int, float))
            },
            "track_metrics": track_metrics,
            "trust_level": "MEASURED" if validation.get("status") == "success" else "UNAVAILABLE",
        }
    )
    with open(run_dir / "validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2)

    _emit_no_data_ablation_summary(run_dir)
    _emit_no_data_control_results(run_dir)
    verification_res = _write_leaf_verification_artifacts(run_dir, exp_root)
    return {
        "validation_payload": validation_payload,
        "verification_res": verification_res,
    }


def emit_consumer_contract_bundle(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    if not monolith_csv.exists():
        has_observer_payload = (run_dir / "observer_global.pt").exists() or any(run_dir.glob("observer_*.pt"))
        if has_observer_payload:
            monolith_ready = _ensure_monolith_csv_ready(run_dir)
            if monolith_ready.get("status") in {"success", "already_exists"}:
                monolith_csv = run_dir / "MONOLITH_DATA.csv"
            elif run_dir.name == "control_constant":
                fallback = _emit_non_comparable_contract_bundle(
                    run_dir,
                    "MONOLITH_DATA.csv unavailable; leaf treated as non-comparable",
                )
                if fallback.get("status") == "success":
                    return fallback
            else:
                fallback = _emit_observer_backed_contract_bundle(
                    run_dir,
                    "MONOLITH_DATA.csv unavailable; emitted observer-backed contract bundle",
                )
                if fallback.get("status") == "success":
                    return fallback
        if not monolith_csv.exists():
            return {"status": "failed", "error": f"missing MONOLITH_DATA.csv at {monolith_csv}"}

    # Verification artifacts are leaf-local by contract. Do not copy from parent
    # directories; inherited reports can misstate leaf verification status.
    copied: List[str] = []

    rows = _load_monolith_rows(monolith_csv)
    observer_backfill = None
    rel_dir = run_dir / "relativity_cache"
    has_observer_payloads = rel_dir.exists() and any(rel_dir.glob("observer_*.pt"))
    if not has_observer_payloads:
        observer_backfill = _ensure_observer_universes_materialized(run_dir)
    verification_res: Dict[str, Any]
    exp_root = _infer_verification_exp_dir(run_dir)
    if exp_root is not None and exp_root.exists():
        verification_res = _write_leaf_verification_artifacts(run_dir, exp_root)
    else:
        verification_res = {
            "status": "skipped",
            "reason": f"unable to infer experiment root for {run_dir}",
        }
    if verification_res.get("status") != "success":
        _ensure_verification_report(run_dir)
    baseline_meta = _emit_baseline_meta(run_dir)
    baseline_state = _emit_baseline_state(run_dir, rows)
    validation_json = _emit_validation_json(run_dir)
    rel_stats = _emit_relativity_defaults(run_dir, rows)
    control_metrics = _emit_control_metrics_json(run_dir)
    ablation_summary = _emit_ablation_summary_json(run_dir)
    relativity_deltas = _emit_relativity_deltas_json(run_dir)
    label_paths = _emit_label_derivatives(run_dir, rows)

    return {
        "status": "success",
        "baseline_meta": str(baseline_meta),
        "baseline_state": str(baseline_state),
        "validation_json": str(validation_json),
        "ablation_summary": str(ablation_summary),
        "control_metrics": str(control_metrics),
        "copied": copied,
        "observer_backfill": observer_backfill,
        "verification": verification_res,
        "relativity": rel_stats,
        "relativity_deltas": str(relativity_deltas),
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
        from core.complete_pipeline import (
            initialize_full_pipeline,
            BeliefTransformerPipeline,
            _compute_alignment_metrics,
            _extract_validation_label_info,
        )
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
                # ASTER v3.2 NUCLEAR DAG: Re-run the entire pipeline for EVERY observer
                # to perform true root-level metric warping (Track 1.5).
                print(f"    [NUCLEAR DAG] Re-running physics for {len(articles)} observers...")
                
                # Base metadata for provenance
                try:
                    from core.complete_pipeline import get_git_hash
                    git_hash = get_git_hash()
                except ImportError:
                    git_hash = "unknown"

                run_meta = {
                    "kernel": kernel,
                    "seed": seed,
                    "channel": "cls",
                    "n_articles": len(articles),
                    "git_hash": git_hash,
                }

                # We still need a 'global' run for baseline_state.json and the root MONOLITH.html
                # We treat the first article (or a mean) as global if no global_idx is set.
                global_result = pipeline.process_month(
                    articles=articles,
                    month_name=run_key,
                    config=pipeline_config,
                )
                global_result["meta"] = run_meta
                normalized_global_provenance = _normalize_run_provenance(global_result, run_meta)
                global_result["provenance"] = normalized_global_provenance
                global_result.setdefault("meta", {})["provenance"] = normalized_global_provenance
                if TORCH_AVAILABLE:
                    import torch as torch_lib
                    torch_lib.save(global_result, run_dir / "observer_global.pt")
                 
                # Materialize relativity_cache directory
                rel_cache_dir = run_dir / "relativity_cache"
                rel_cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Loop through every article as an observer
                for obs_i in range(len(articles)):
                    print(f"\r      -> Computing Universe {obs_i}/{len(articles)}...", end="")
                    
                    # Each universe needs its own isolated directory for internal pipeline persistence
                    obs_sub_dir = rel_cache_dir / f"obs_{obs_i}"
                    obs_sub_dir.mkdir(parents=True, exist_ok=True)
                    
                    obs_config = dict(pipeline_config)
                    obs_config["output_dir"] = str(obs_sub_dir)
                    obs_config["checkpoint_dir"] = str(obs_sub_dir)

                    # RE-RUN PHYSICS (Kernels -> Topology -> Walkers)
                    obs_result = pipeline.process_month(
                        articles=articles,
                        month_name=f"{run_key}_obs{obs_i}",
                        config=obs_config,
                        observer_idx=obs_i,
                    )
                    obs_result["meta"] = run_meta
                    normalized_obs_provenance = _normalize_run_provenance(obs_result, run_meta)
                    obs_result["provenance"] = normalized_obs_provenance
                    obs_result.setdefault("meta", {})["provenance"] = normalized_obs_provenance
                    
                    # Redundant save for safety (the pipeline already writes some to obs_sub_dir)
                    np.save(obs_sub_dir / "features.npy", obs_result['features'])
                    if 'dirichlet_fused' in obs_result:
                         np.save(obs_sub_dir / "gradients.npy", obs_result['dirichlet_fused'])
                    
                    # Save the full state payload for later hydration
                    if TORCH_AVAILABLE:
                        import torch as torch_lib
                        torch_lib.save(obs_result, rel_cache_dir / f"observer_{obs_i}.pt")
                    
                    # ASTER v3.2: Prevent memory accumulation in the N-Universe loop
                    del obs_result
                    if TORCH_AVAILABLE:
                        import torch as torch_lib
                        torch_lib.cuda.empty_cache()
                    import gc
                    gc.collect()

                print(f"\n    [OK] N-Universe expansion complete.")
                result = global_result  # Main result for standard artifacts

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
                    _write_article_metadata_csv(result['article_metadata'], run_dir / "article_metadata.csv")
                    print(f"    Saved article_metadata.json and .csv ({len(result['article_metadata'])} articles)")

                leaf_artifacts = _finalize_synthetic_leaf_artifacts(
                    run_dir,
                    output_dir,
                    result=result,
                    validation=validation,
                    articles=articles,
                    kernel=kernel,
                    seed=seed,
                    compute_alignment_metrics=_compute_alignment_metrics,
                    extract_validation_label_info=_extract_validation_label_info,
                )
                verification_res = leaf_artifacts.get("verification_res", {})
                if verification_res.get("status") != "success":
                    print(f"    [VERIFY][WARN] {verification_res}")
                else:
                    print(f"    [VERIFY][OK] {verification_res.get('report_path')}")

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
                    "run_dir": str(run_dir),
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


def _iter_suite_leaf_dirs(exp_dir: Path, config: Dict[str, Any]) -> List[Path]:
    kernels = list(config.get("kernels") or [])
    channels = list(config.get("channels") or [])
    corpora = list(config.get("corpora") or [])
    leafs: List[Path] = []
    seen: set[str] = set()
    for corpus in corpora:
        for channel in channels:
            if channel == "gradient":
                candidate = exp_dir / "gradient" / corpus
                key = str(candidate)
                if key not in seen:
                    leafs.append(candidate)
                    seen.add(key)
                continue
            for kernel in kernels:
                candidate = exp_dir / kernel / channel / corpus
                key = str(candidate)
                if key not in seen:
                    leafs.append(candidate)
                    seen.add(key)
    return leafs


def _restore_args_from_resume(args: argparse.Namespace, exp_dir: Path) -> argparse.Namespace:
    """Restore core run settings from prior suite config/manifest when resuming."""
    stored: Optional[Dict[str, Any]] = _load_suite_config(exp_dir)
    if not isinstance(stored, dict):
        manifest_path = exp_dir / "experiment_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                maybe_cfg = manifest.get("config")
                if isinstance(maybe_cfg, dict):
                    stored = maybe_cfg
            except Exception:
                stored = None

    if not isinstance(stored, dict):
        return args

    for key in ("limit", "mode"):
        if key in stored and stored[key] is not None:
            setattr(args, key, stored[key])

    for key in ("seeds", "kernels", "channels", "corpora"):
        value = stored.get(key)
        if isinstance(value, list) and value:
            setattr(args, key, value)

    variance_tracking = stored.get("variance_tracking")
    if isinstance(variance_tracking, bool):
        args.no_variance_tracking = not variance_tracking

    probe_cfg = stored.get("probe")
    if isinstance(probe_cfg, dict):
        for arg_key, cfg_key in (
            ("probe_script", "script"),
            ("probe_model", "model"),
            ("probe_hypotheses", "hypotheses"),
            ("probe_entities", "entities"),
            ("probe_corpus_jsonl", "corpus_jsonl"),
            ("probe_corpus_field", "corpus_field"),
        ):
            if probe_cfg.get(cfg_key):
                setattr(args, arg_key, probe_cfg[cfg_key])
        for arg_key, cfg_key in (
            ("probe_n_synth", "n_synth"),
            ("probe_corpus_limit", "corpus_limit"),
            ("probe_max_length", "max_length"),
            ("probe_batch_size", "batch_size"),
        ):
            if probe_cfg.get(cfg_key) is not None:
                setattr(args, arg_key, probe_cfg[cfg_key])
        if "nonfatal" in probe_cfg:
            args.probe_nonfatal = bool(probe_cfg["nonfatal"])

    print("[RESUME] Restored config from prior suite metadata.")
    return args


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
        args = _restore_args_from_resume(args, exp_dir)
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

    base_config = {
        "limit": args.limit,
        "seeds": args.seeds,
        "kernels": args.kernels,
        "channels": args.channels,
        "corpora": args.corpora,
        "variance_tracking": not args.no_variance_tracking,
        "structure": "kernel/channel/corpus/observer_SEED.pt",
        "probe": {
            "enabled": bool(probe_enabled),
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
        },
    }
    _write_suite_config(exp_dir, base_config)
    save_manifest(exp_dir, [], base_config)

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

        # Emit thesis-facing baseline artifacts for each synthetic run
        if synthetic_result.get("status") == "success":
            print(f"\n{'='*80}")
            print("MATERIALIZING SYNTHETIC BASELINE BUNDLES")
            print(f"{'='*80}")
            for res in synthetic_result.get("results", []):
                if res.get("status") == "success" and res.get("run_dir"):
                    run_dir = Path(res["run_dir"])
                    print(f"\n  Processing: {res['run_key']}")
                    bundle_res = materialize_baseline_bundle(run_dir, strict=True)
                    res["baseline_bundle"] = bundle_res
                    if bundle_res.get("status") == "success":
                        print(f"    [BUNDLE][OK] {bundle_res.get('observer_manifest')}")
                    else:
                        print(f"    [BUNDLE][WARN] {bundle_res.get('error', 'Unknown error')}")

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
                    save_manifest(exp_dir, results, base_config)
                    print(f"\n Gradient analysis failed: {corpus}")
                    print("Stopping experiment suite.")
                    break

                results.append(result)
                save_manifest(exp_dir, results, base_config)
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
                    save_manifest(exp_dir, results, base_config)
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
                            save_manifest(exp_dir, results, base_config)
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
                            save_manifest(exp_dir, results, base_config)
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
                save_manifest(exp_dir, results, base_config)

                # Progress update
                completed_so_far = len([r for r in results if r.get("status") == "success"])
                print(f"\n{'='*80}")
                print(f"PROGRESS: {experiment_count}/{total_experiments} experiments")
                print(f"Successful: {completed_so_far}")
                print(f"{'='*80}\n")
                sys.stdout.flush()

    # Refresh direct control metrics after all sibling corpora exist.
    refreshed_control_metrics: List[str] = []
    refreshed_contract_bundles: List[str] = []
    refreshed_control_dirs: set[str] = set()
    for output_dir in _iter_suite_leaf_dirs(exp_dir, base_config):
        if not output_dir.exists():
            continue
        key = str(output_dir)
        if key in refreshed_control_dirs:
            continue
        refreshed_control_dirs.add(key)
        try:
            refreshed_path = _emit_control_metrics_json(output_dir)
            refreshed_control_metrics.append(str(refreshed_path))
        except Exception as refresh_exc:
            print(f"[CONTROL][WARN] Failed to refresh direct control metrics for {output_dir}: {refresh_exc}")
        try:
            bundle_res = emit_consumer_contract_bundle(output_dir)
            if bundle_res.get("status") == "success":
                refreshed_contract_bundles.append(str(output_dir))
            else:
                print(f"[CONTRACT][WARN] Failed to refresh consumer bundle for {output_dir}: {bundle_res}")
        except Exception as bundle_exc:
            print(f"[CONTRACT][WARN] Failed to refresh consumer bundle for {output_dir}: {bundle_exc}")

    # Save manifest
    config = dict(base_config)
    config.update({
        "refreshed_control_metrics": refreshed_control_metrics,
        "refreshed_contract_bundles": refreshed_contract_bundles,
    })
    _write_suite_config(exp_dir, config)
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
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
