"""Run and summarize Track 4 feature-basis probes through the full pipeline.

This is a thin orchestration wrapper around ``run_experiments.py``. It does not
re-implement the Semantic Interferometer; it standardizes a small ablation
matrix so Track 4 can be tested over different traversal coordinate systems
without changing production defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.verification.scientific_summaries import summarize_track4_traversal


TRACK4_PIPELINE_BASES = ("track2", "logits_flat", "bot_norms", "cls_stacked", "spectral_pc1")


@dataclass(frozen=True)
class ProbeConfig:
    corpus: Path
    output_root: Path
    mode: str = "cls_logits"
    seed: int = 42
    limit: int = 60
    kernel_type: str = "rbf"
    rks_dim: int = 512
    track5_assembly_mode: str = "hadamard_strict"
    nli_cache_path: Path | None = None
    proposal_mode: str = "metric_softmax"
    adaptive_tpt_connectivity: bool = False
    walker_temperature: float = 0.75
    walker_gamma: float = 5.0
    walker_k_neighbors: int = 8


def basis_output_dir(output_root: Path, basis: str) -> Path:
    return Path(output_root) / f"track4_basis_{basis}"


def build_basis_command(config: ProbeConfig, basis: str) -> List[str]:
    command = [
        sys.executable,
        str(ROOT / "run_experiments.py"),
        "--corpus",
        str(config.corpus),
        "--mode",
        str(config.mode),
        "--seeds",
        str(int(config.seed)),
        "--limit",
        str(int(config.limit)),
        "--kernel-type",
        str(config.kernel_type),
        "--rks-dim",
        str(int(config.rks_dim)),
        "--output-root",
        str(basis_output_dir(config.output_root, basis)),
        "--track5-assembly-mode",
        str(config.track5_assembly_mode),
        "--track4-basis",
        str(basis),
        "--track4-proposal-mode",
        str(config.proposal_mode),
        "--walker-temperature",
        str(float(config.walker_temperature)),
        "--walker-gamma",
        str(float(config.walker_gamma)),
        "--walker-k-neighbors",
        str(int(config.walker_k_neighbors)),
        "--no-freeze-good-run",
    ]
    if config.nli_cache_path is not None:
        command.extend(["--nli-cache-path", str(config.nli_cache_path)])
    if config.adaptive_tpt_connectivity:
        command.append("--track4-adaptive-tpt-connectivity")
    return command


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _load_observer_runtime(run_dir: Path, seed: int) -> Dict[str, Any]:
    observer_path = Path(run_dir) / f"observer_{int(seed)}.pt"
    if not observer_path.exists():
        return {"observer_path": str(observer_path), "observer_exists": False}
    try:
        artifact = torch.load(observer_path, map_location="cpu", weights_only=False)
        return {
            "observer_path": str(observer_path),
            "observer_exists": True,
            "meta": dict(artifact.get("meta", {}) or {}),
            "track4_runtime_config": dict(artifact.get("track4_runtime_config", {}) or {}),
            "track4_markov_observables": dict(artifact.get("track4_markov_observables", {}) or {}),
        }
    except Exception as exc:
        return {
            "observer_path": str(observer_path),
            "observer_exists": True,
            "observer_load_error": str(exc),
        }


def summarize_basis_run(output_root: Path, basis: str, seed: int) -> Dict[str, Any]:
    run_dir = basis_output_dir(output_root, basis)
    runtime = _load_observer_runtime(run_dir, seed)
    try:
        traversal = summarize_track4_traversal(run_dir)
    except Exception as exc:
        traversal = {
            "status": "SUMMARY_ERROR",
            "safe_for_thesis_claim": False,
            "failure_reasons": [str(exc)],
        }
    runtime_config = runtime.get("track4_runtime_config", {}) if isinstance(runtime, dict) else {}
    markov = runtime.get("track4_markov_observables", {}) if isinstance(runtime, dict) else {}
    markov_summary = markov.get("summary", {}) if isinstance(markov, dict) else {}
    return {
        "basis": str(basis),
        "run_dir": str(run_dir),
        "observer_exists": bool(runtime.get("observer_exists")),
        "requested_basis": runtime_config.get("requested_basis"),
        "effective_basis": runtime_config.get("effective_basis"),
        "basis_embedding_dim": runtime_config.get("basis_embedding_dim"),
        "proposal_mode": runtime_config.get("proposal_mode"),
        "adaptive_tpt_connectivity": runtime_config.get("adaptive_tpt_connectivity"),
        "effective_k_neighbors": runtime_config.get("effective_k_neighbors"),
        "markov_status": markov.get("status") if isinstance(markov, dict) else None,
        "reactive_flux_total": _safe_float(markov_summary.get("reactive_flux_total")),
        "committor_mean": _safe_float(markov_summary.get("committor_mean")),
        "traversal_status": traversal.get("status"),
        "safe_for_thesis_claim": bool(traversal.get("safe_for_thesis_claim", False)),
        "path_count": traversal.get("path_count"),
        "closed_loop_rate": _safe_float(traversal.get("closed_loop_rate")),
        "mean_work_integral": _safe_float(traversal.get("mean_work_integral")),
        "path_shape_entropy_norm": _safe_float(traversal.get("path_shape_entropy_norm")),
        "path_edge_entropy_norm": _safe_float(traversal.get("path_edge_entropy_norm")),
        "primary_zone_count": traversal.get("primary_zone_count"),
        "feature_basis_counts": traversal.get("feature_basis_counts", {}),
        "proposal_mode_counts": traversal.get("proposal_mode_counts", {}),
        "failure_reasons": traversal.get("failure_reasons", []),
        "warnings": traversal.get("warnings", []),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    scalar_keys = [
        "basis",
        "run_dir",
        "observer_exists",
        "requested_basis",
        "effective_basis",
        "basis_embedding_dim",
        "proposal_mode",
        "adaptive_tpt_connectivity",
        "effective_k_neighbors",
        "markov_status",
        "reactive_flux_total",
        "committor_mean",
        "traversal_status",
        "safe_for_thesis_claim",
        "path_count",
        "closed_loop_rate",
        "mean_work_integral",
        "path_shape_entropy_norm",
        "path_edge_entropy_norm",
        "primary_zone_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def basis_probe_score(row: Dict[str, Any]) -> float:
    """Diagnostic ranking score for Track 4 feature-basis probes.

    This is deliberately not a thesis pass/fail rule. It is a compact triage
    score for deciding which basis deserves the next expensive run.
    """
    shape_entropy = _safe_float(row.get("path_shape_entropy_norm")) or 0.0
    edge_entropy = _safe_float(row.get("path_edge_entropy_norm")) or 0.0
    closed_loop_rate = _safe_float(row.get("closed_loop_rate")) or 0.0
    zone_count = _safe_float(row.get("primary_zone_count")) or 0.0
    reactive_flux = _safe_float(row.get("reactive_flux_total")) or 0.0
    zone_component = min(max(zone_count / 4.0, 0.0), 1.0)
    flux_component = 1.0 if reactive_flux > 0.0 else 0.0
    thesis_bonus = 0.15 if bool(row.get("safe_for_thesis_claim")) else 0.0
    return float(
        0.30 * shape_entropy
        + 0.25 * edge_entropy
        + 0.20 * zone_component
        + 0.15 * closed_loop_rate
        + 0.10 * flux_component
        + thesis_bonus
    )


def rank_basis_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["basis_probe_score"] = basis_probe_score(enriched)
        ranked.append(enriched)
    ranked.sort(key=lambda item: float(item.get("basis_probe_score", 0.0)), reverse=True)
    return ranked


def probe_claim_boundary(
    ranked_rows: Sequence[Dict[str, Any]],
    *,
    min_basis_score_margin: float = 0.05,
) -> Dict[str, Any]:
    """Separate engineering evidence from scientific/terrain evidence.

    A small diagnostic win is useful for choosing the next run, but it is not a
    publishable "basis is better" or "terrain is valid" claim. This contract
    makes that distinction explicit in every probe summary.
    """
    rows = list(ranked_rows)
    best = rows[0] if rows else {}
    runner_healthy = bool(rows) and all(
        bool(row.get("observer_exists"))
        and row.get("effective_basis")
        and (_safe_float(row.get("path_count")) or 0.0) > 0.0
        for row in rows
    )
    terrain_valid_rows = [row for row in rows if bool(row.get("safe_for_thesis_claim"))]
    best_score = _safe_float(best.get("basis_probe_score")) if best else None
    second_score = _safe_float(rows[1].get("basis_probe_score")) if len(rows) > 1 else None
    score_margin = (
        float(best_score - second_score)
        if best_score is not None and second_score is not None
        else None
    )
    basis_superiority_supported = bool(
        score_margin is not None
        and score_margin >= float(min_basis_score_margin)
        and bool(best.get("safe_for_thesis_claim"))
    )

    if not rows:
        basis_status = "no_probe_rows"
    elif score_margin is None:
        basis_status = "single_basis_no_comparison"
    elif score_margin < float(min_basis_score_margin):
        basis_status = "tentative_margin_too_small"
    elif not bool(best.get("safe_for_thesis_claim")):
        basis_status = "diagnostic_only_top_basis_not_terrain_safe"
    else:
        basis_status = "basis_superiority_candidate"

    failure_reasons = sorted(
        {
            str(reason)
            for row in rows
            for reason in (row.get("failure_reasons") or [])
            if str(reason).strip()
        }
    )

    return {
        "instrumentation_supported": runner_healthy,
        "terrain_validity_supported": bool(terrain_valid_rows),
        "basis_superiority_supported": basis_superiority_supported,
        "recommended_basis": best.get("basis"),
        "basis_score_margin": score_margin,
        "minimum_basis_score_margin": float(min_basis_score_margin),
        "basis_claim_status": basis_status,
        "terrain_safe_basis_count": len(terrain_valid_rows),
        "terrain_failure_reasons": failure_reasons,
        "safe_language": (
            "Track 4 basis/proposal instrumentation is working; terrain validity "
            "and basis superiority remain unproven unless this boundary reports support."
        ),
    }


def write_probe_summary(output_root: Path, rows: Sequence[Dict[str, Any]], commands: Sequence[Sequence[str]]) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ranked_rows = rank_basis_rows(rows)
    recommended_basis = ranked_rows[0]["basis"] if ranked_rows else None
    claim_boundary = probe_claim_boundary(ranked_rows)
    payload = {
        "schema_version": "1.0",
        "probe_type": "track4_pipeline_feature_basis",
        "basis_count": len(rows),
        "rows": list(rows),
        "ranked_rows": ranked_rows,
        "recommended_basis": recommended_basis,
        "claim_boundary": claim_boundary,
        "ranking_basis": (
            "Diagnostic blend of path-shape entropy, edge entropy, terrain coverage, "
            "closed-loop rate, nonzero reactive flux, and a small thesis-safe bonus."
        ),
        "commands": [" ".join(command) for command in commands],
        "interpretation": (
            "Feature basis probes are Track 4 ablations. They test where traversal "
            "telemetry is strongest while preserving the production default unless "
            "a basis is explicitly selected."
        ),
    }
    out_path = output_root / "track4_pipeline_basis_probe_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_root / "track4_pipeline_basis_probe_summary.csv", rows)
    return out_path


def run_probe(config: ProbeConfig, bases: Iterable[str], *, dry_run: bool = False, skip_existing: bool = False) -> Dict[str, Any]:
    selected_bases = [str(basis) for basis in bases]
    commands = [build_basis_command(config, basis) for basis in selected_bases]
    run_records: List[Dict[str, Any]] = []

    for basis, command in zip(selected_bases, commands):
        run_dir = basis_output_dir(config.output_root, basis)
        observer_path = run_dir / f"observer_{int(config.seed)}.pt"
        if dry_run:
            run_records.append({"basis": basis, "status": "DRY_RUN", "command": command})
            continue
        if skip_existing and observer_path.exists():
            run_records.append({"basis": basis, "status": "SKIPPED_EXISTING", "command": command})
            continue

        completed = subprocess.run(command, cwd=str(ROOT), text=True)
        run_records.append(
            {
                "basis": basis,
                "status": "OK" if completed.returncode == 0 else "FAILED",
                "returncode": int(completed.returncode),
                "command": command,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Track 4 basis probe failed for {basis} with exit code {completed.returncode}")

    rows = [] if dry_run else [summarize_basis_run(config.output_root, basis, config.seed) for basis in selected_bases]
    ranked_rows = rank_basis_rows(rows)
    summary_path = write_probe_summary(config.output_root, rows, commands)
    return {
        "summary_path": str(summary_path),
        "rows": rows,
        "ranked_rows": ranked_rows,
        "claim_boundary": probe_claim_boundary(ranked_rows),
        "run_records": run_records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", default="cls_logits")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=60, type=int)
    parser.add_argument("--kernel-type", default="rbf")
    parser.add_argument("--rks-dim", default=512, type=int)
    parser.add_argument("--track5-assembly-mode", default="hadamard_strict")
    parser.add_argument("--nli-cache-path", default=None, type=Path)
    parser.add_argument("--bases", nargs="+", default=list(TRACK4_PIPELINE_BASES), choices=list(TRACK4_PIPELINE_BASES))
    parser.add_argument("--track4-proposal-mode", default="metric_softmax")
    parser.add_argument("--track4-adaptive-tpt-connectivity", action="store_true")
    parser.add_argument("--walker-temperature", default=0.75, type=float)
    parser.add_argument("--walker-gamma", default=5.0, type=float)
    parser.add_argument("--walker-k-neighbors", default=8, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ProbeConfig(
        corpus=args.corpus,
        output_root=args.output_root,
        mode=args.mode,
        seed=args.seed,
        limit=args.limit,
        kernel_type=args.kernel_type,
        rks_dim=args.rks_dim,
        track5_assembly_mode=args.track5_assembly_mode,
        nli_cache_path=args.nli_cache_path,
        proposal_mode=args.track4_proposal_mode,
        adaptive_tpt_connectivity=args.track4_adaptive_tpt_connectivity,
        walker_temperature=args.walker_temperature,
        walker_gamma=args.walker_gamma,
        walker_k_neighbors=args.walker_k_neighbors,
    )
    result = run_probe(config, args.bases, dry_run=bool(args.dry_run), skip_existing=bool(args.skip_existing))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
