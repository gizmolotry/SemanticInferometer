#!/usr/bin/env python3
"""Run lab-only Track 4 proposal-mode sweeps on a saved feature environment.

The production walker stays ``metric_softmax``.  This harness asks a narrower
question: if we keep the same article graph, anchors, cyclic contract, and
Markov/TPT exports, do alternative proposal dynamics improve Track 4 evidence
quality on a controlled micro-environment?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.verification.scientific_summaries import summarize_track4_traversal  # noqa: E402
from core.physarum_walk import SemanticWalker  # noqa: E402
from scripts.property_theft_microprobe import MicroArticle, _articles  # noqa: E402


DEFAULT_FEATURES = REPO_ROOT / "outputs/microprobes/property_theft/deberta_20260515/deberta_features.npz"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/microprobes/property_theft/track4_method_sweep"
DEFAULT_MODES = (
    "metric_softmax",
    "stress_biased",
    "committor_guided",
    "deterministic_low_cost",
)


def _csv_numbers(values: str, *, cast):
    return [cast(part.strip()) for part in str(values).split(",") if part.strip()]


def _unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    span = float(arr.max() - arr.min()) if arr.size else 0.0
    if span <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - arr.min()) / span


def _pca(features: np.ndarray, n_components: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if min(x.shape) <= 1:
        return np.zeros((x.shape[0], n_components), dtype=np.float32)
    u, s, _vh = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords.astype(np.float32)


def _pairwise_euclidean(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _feature_density(features: np.ndarray, k: int = 4) -> np.ndarray:
    dmat = _pairwise_euclidean(features)
    if dmat.shape[0] <= 1:
        return np.ones((dmat.shape[0],), dtype=np.float32)
    k_eff = min(max(int(k), 1), dmat.shape[0] - 1)
    knn = np.sort(dmat, axis=1)[:, 1 : k_eff + 1]
    mean_knn = knn.mean(axis=1)
    return (1.0 - _unit(mean_knn)).astype(np.float32)


def _load_features(path: Path, feature_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        if feature_key not in payload.files:
            raise KeyError(f"{path} does not contain feature_key={feature_key!r}; available={payload.files}")
        features = np.asarray(payload[feature_key], dtype=np.float32)
    if features.ndim < 2:
        raise ValueError(f"{feature_key} must be at least 2D, got shape={features.shape}")
    return features.reshape(features.shape[0], -1).astype(np.float32)


def _zone_profile(zone: str) -> tuple[float, float]:
    profiles = {
        "Bridge": (0.92, 0.15),
        "Swamp": (0.90, 0.88),
        "Tightrope": (0.14, 0.15),
        "Void": (0.12, 0.90),
    }
    return profiles.get(str(zone), (0.5, 0.5))


def _environment_from_articles(
    articles: Sequence[MicroArticle],
    features: np.ndarray,
) -> Dict[str, np.ndarray]:
    coords = _pca(features, 2)
    feature_density = _feature_density(features)
    pc1_abs = _unit(np.abs(_pca(features, 1).reshape(-1)))

    density: List[float] = []
    stress: List[float] = []
    for idx, article in enumerate(articles):
        base_density, base_stress = _zone_profile(article.zone)
        density.append(float(np.clip(base_density + 0.06 * (feature_density[idx] - 0.5), 0.01, 1.0)))
        stress.append(float(np.clip(base_stress + 0.06 * (pc1_abs[idx] - 0.5), 0.0, 1.0)))

    return {
        "features": features.astype(np.float32),
        "coords": coords.astype(np.float32),
        "density": np.asarray(density, dtype=np.float32),
        "stress": np.asarray(stress, dtype=np.float32),
    }


def _write_monolith_csv(path: Path, articles: Sequence[MicroArticle], env: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "title",
                "zone",
                "frame",
                "hidden_label",
                "track3_density",
                "metric_stress",
                "x",
                "y",
            ],
        )
        writer.writeheader()
        for row_idx, article in enumerate(articles):
            writer.writerow(
                {
                    "index": int(article.index),
                    "title": article.title,
                    "zone": article.zone,
                    "frame": article.frame,
                    "hidden_label": article.hidden_label,
                    "track3_density": float(env["density"][row_idx]),
                    "metric_stress": float(env["stress"][row_idx]),
                    "x": float(env["coords"][row_idx, 0]),
                    "y": float(env["coords"][row_idx, 1]),
                }
            )


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: json.dumps(row.get(column), sort_keys=True)
                    if isinstance(row.get(column), (dict, list))
                    else row.get(column)
                    for column in columns
                }
            )


def _compact_cyclic_paths_npz(path: Path) -> bool:
    """Drop HD coordinate payloads from lab sweeps while preserving telemetry.

    The visualizer needs ``path_xyz`` for animation, but the method-sweep scorer
    only consumes path indices, work, closure, terrain metadata, and Markov/TPT
    fields.  Keeping every high-dimensional coordinate trace makes robustness
    sweeps balloon into multi-GB artifacts, so this lab-only compactor preserves
    scientific telemetry and removes only the visualization-heavy coordinate key.
    """
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files if key != "path_xyz"}
        had_path_xyz = "path_xyz" in payload.files
    if had_path_xyz:
        np.savez_compressed(path, **data)
    return bool(had_path_xyz)


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _std(values: Iterable[float]) -> float | None:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.std(vals)) if vals.size else None


def _score_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    closed = summary.get("closed_loop_rate")
    closed = float(closed) if isinstance(closed, (int, float)) and math.isfinite(float(closed)) else 0.0
    path_count = max(int(summary.get("path_count") or 0), 1)
    unique_paths = float(summary.get("unique_path_shape_count") or 0) / path_count
    route_entropy = summary.get("path_shape_entropy_norm")
    route_entropy = (
        float(route_entropy)
        if isinstance(route_entropy, (int, float)) and math.isfinite(float(route_entropy))
        else unique_paths
    )
    zone_coverage = min(float(summary.get("primary_zone_count") or 0) / 4.0, 1.0)
    primary_gap = (summary.get("primary_bridge_vs_void") or {}).get("work_integral_gap")
    mean_work = summary.get("mean_work_integral")
    gap_norm = 0.0
    if isinstance(primary_gap, (int, float)) and isinstance(mean_work, (int, float)) and abs(float(mean_work)) > 1e-9:
        gap_norm = max(min(float(primary_gap) / abs(float(mean_work)), 1.0), -1.0)
    flux_total = ((summary.get("reactive_flux_summary") or {}).get("total") or 0.0)
    flux_score = min(float(flux_total) * 10.0, 1.0) if math.isfinite(float(flux_total)) else 0.0
    score = (
        0.35 * closed
        + 0.15 * unique_paths
        + 0.10 * route_entropy
        + 0.20 * zone_coverage
        + 0.15 * max(gap_norm, 0.0)
        + 0.05 * flux_score
    )
    return {
        "proposal_quality_score": float(score),
        "closed_loop_rate_component": closed,
        "unique_path_component": float(unique_paths),
        "route_entropy_component": float(route_entropy),
        "zone_coverage_component": float(zone_coverage),
        "positive_bridge_void_work_gap_component": float(max(gap_norm, 0.0)),
        "reactive_flux_component": float(flux_score),
    }


def run_sweep(
    *,
    feature_path: Path,
    output_dir: Path,
    feature_key: str,
    modes: Sequence[str],
    seed: int,
    max_steps: int,
    k_neighbors: int,
    temperature: float = 0.75,
    gamma: float = 5.0,
    adaptive_tpt_connectivity: bool = False,
    compact_artifacts: bool = True,
) -> Dict[str, Any]:
    articles = _articles()
    features = _load_features(feature_path, feature_key)
    if features.shape[0] != len(articles):
        raise ValueError(f"feature row count {features.shape[0]} does not match microprobe article count {len(articles)}")

    env = _environment_from_articles(articles, features)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_rows: List[Dict[str, Any]] = []

    for mode in modes:
        mode_dir = output_dir / str(mode)
        mode_dir.mkdir(parents=True, exist_ok=True)
        _write_monolith_csv(mode_dir / "MONOLITH_DATA.csv", articles, env)
        walker = SemanticWalker(
            embeddings=torch.as_tensor(env["features"], dtype=torch.float32),
            rks_basis=lambda x: x,
            article_coords_2d=torch.as_tensor(env["coords"], dtype=torch.float32),
            track3_density=torch.as_tensor(env["density"], dtype=torch.float32),
            metric_stress=torch.as_tensor(env["stress"], dtype=torch.float32),
            temperature=float(temperature),
        )
        result = walker.run_stress_triggered_cyclic_walk(
            max_steps=max_steps,
            k_neighbors=k_neighbors,
            gamma=float(gamma),
            output_dir=str(mode_dir),
            start_seed=seed,
            proposal_mode=str(mode),
            adaptive_tpt_connectivity=adaptive_tpt_connectivity,
            feature_basis=str(feature_key),
        )
        compacted = _compact_cyclic_paths_npz(mode_dir / "cyclic_paths.npz") if compact_artifacts else False
        summary = summarize_track4_traversal(mode_dir)
        score = _score_summary(summary)
        row = {
            "proposal_mode": str(mode),
            "mode_dir": str(mode_dir),
            "catalyst_indices": result.get("catalyst_indices", []),
            "catalyst_zones": result.get("catalyst_zones", []),
            "anchor_summaries": result.get("anchor_summaries", []),
            "compact_artifacts": bool(compact_artifacts),
            "cyclic_paths_compacted": bool(compacted),
            "summary": summary,
            "score": score,
        }
        mode_rows.append(row)
        (mode_dir / "track4_method_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    baseline = next((row for row in mode_rows if row["proposal_mode"] == "metric_softmax"), None)
    baseline_score = float((baseline or {}).get("score", {}).get("proposal_quality_score", 0.0))
    for row in mode_rows:
        mode_score = float(row["score"]["proposal_quality_score"])
        row["score"]["delta_vs_metric_softmax"] = float(mode_score - baseline_score)
        row["score"]["improves_over_metric_softmax"] = bool(mode_score > baseline_score + 1e-9)

    ranked = sorted(
        [
            {
                "proposal_mode": row["proposal_mode"],
                "proposal_quality_score": row["score"]["proposal_quality_score"],
                "delta_vs_metric_softmax": row["score"]["delta_vs_metric_softmax"],
                "closed_loop_rate": row["summary"].get("closed_loop_rate"),
                "mean_work_integral": row["summary"].get("mean_work_integral"),
                "terrain_evidence_basis": row["summary"].get("terrain_evidence_basis"),
                "safe_for_thesis_claim": row["summary"].get("safe_for_thesis_claim"),
                "failure_reasons": row["summary"].get("failure_reasons", []),
            }
            for row in mode_rows
        ],
        key=lambda item: float(item["proposal_quality_score"]),
        reverse=True,
    )
    report = {
        "status": "OK",
        "feature_path": str(feature_path),
        "feature_key": feature_key,
        "output_dir": str(output_dir),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "k_neighbors": int(k_neighbors),
        "temperature": float(temperature),
        "gamma": float(gamma),
        "adaptive_tpt_connectivity": bool(adaptive_tpt_connectivity),
        "compact_artifacts": bool(compact_artifacts),
        "modes": [str(mode) for mode in modes],
        "ranked_modes": ranked,
        "mode_rows": mode_rows,
        "interpretation": (
            "This is a lab-only Track 4 dynamics sweep. It does not change the "
            "production metric_softmax walker; it ranks candidate proposal "
            "rules by survival, path diversity, terrain coverage, bridge/void "
            "work gap, and reactive-flux availability."
        ),
    }
    (output_dir / "track4_method_sweep_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_rows_csv(
        output_dir / "track4_ranked_modes.csv",
        ranked,
        [
            "proposal_mode",
            "proposal_quality_score",
            "delta_vs_metric_softmax",
            "closed_loop_rate",
            "mean_work_integral",
            "terrain_evidence_basis",
            "safe_for_thesis_claim",
            "failure_reasons",
        ],
    )
    return report


def _sanitize_float_for_path(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _flatten_grid_reports(reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in reports:
        for row in report.get("mode_rows", []):
            summary = row.get("summary") or {}
            score = row.get("score") or {}
            markov = summary.get("markov_summary") or {}
            rows.append(
                {
                    "proposal_mode": row.get("proposal_mode"),
                    "k_neighbors": int(report.get("k_neighbors") or 0),
                    "seed": int(report.get("seed") or 0),
                    "temperature": float(report.get("temperature") or 0.0),
                    "gamma": float(report.get("gamma") or 0.0),
                    "adaptive_tpt_connectivity": bool(report.get("adaptive_tpt_connectivity")),
                    "proposal_quality_score": float(score.get("proposal_quality_score") or 0.0),
                    "delta_vs_metric_softmax": float(score.get("delta_vs_metric_softmax") or 0.0),
                    "closed_loop_rate": summary.get("closed_loop_rate"),
                    "mean_work_integral": summary.get("mean_work_integral"),
                    "unique_path_shape_count": summary.get("unique_path_shape_count"),
                    "path_shape_entropy_norm": summary.get("path_shape_entropy_norm"),
                    "path_edge_entropy_norm": summary.get("path_edge_entropy_norm"),
                    "mean_path_edge_count": summary.get("mean_path_edge_count"),
                    "safe_for_thesis_claim": bool(summary.get("safe_for_thesis_claim")),
                    "terrain_evidence_basis": summary.get("terrain_evidence_basis"),
                    "reactive_flux_total": (summary.get("reactive_flux_summary") or {}).get("total"),
                    "effective_k_neighbors": markov.get("effective_k_neighbors"),
                    "connectivity_repair_applied": markov.get("connectivity_repair_applied"),
                    "failure_reasons": summary.get("failure_reasons", []),
                    "mode_dir": row.get("mode_dir"),
                }
            )
    return rows


def _mode_robustness(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("proposal_mode")), []).append(row)
    aggregate: List[Dict[str, Any]] = []
    for mode, mode_rows in sorted(by_mode.items()):
        scores = [float(row["proposal_quality_score"]) for row in mode_rows]
        deltas = [float(row["delta_vs_metric_softmax"]) for row in mode_rows]
        closed = [
            float(row["closed_loop_rate"])
            for row in mode_rows
            if isinstance(row.get("closed_loop_rate"), (int, float))
            and math.isfinite(float(row["closed_loop_rate"]))
        ]
        flux = [
            float(row["reactive_flux_total"])
            for row in mode_rows
            if isinstance(row.get("reactive_flux_total"), (int, float))
            and math.isfinite(float(row["reactive_flux_total"]))
        ]
        safe_count = sum(1 for row in mode_rows if bool(row.get("safe_for_thesis_claim")))
        aggregate.append(
            {
                "proposal_mode": mode,
                "condition_count": len(mode_rows),
                "safe_condition_count": int(safe_count),
                "safe_rate": float(safe_count / max(len(mode_rows), 1)),
                "mean_score": _mean(scores),
                "std_score": _std(scores),
                "min_score": float(min(scores)) if scores else None,
                "max_score": float(max(scores)) if scores else None,
                "mean_delta_vs_metric_softmax": _mean(deltas),
                "mean_closed_loop_rate": _mean(closed),
                "mean_reactive_flux_total": _mean(flux),
            }
        )
    return sorted(
        aggregate,
        key=lambda item: (
            float(item.get("safe_rate") or 0.0),
            float(item.get("mean_score") or 0.0),
            float(item.get("mean_delta_vs_metric_softmax") or 0.0),
        ),
        reverse=True,
    )


def _method_recommendation(
    ranked_conditions: Sequence[Dict[str, Any]],
    mode_robustness: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    best_peak = dict(ranked_conditions[0]) if ranked_conditions else {}
    best_robust = dict(mode_robustness[0]) if mode_robustness else {}
    peak_mode = str(best_peak.get("proposal_mode", ""))
    robust_mode = str(best_robust.get("proposal_mode", ""))
    if peak_mode and robust_mode and peak_mode != robust_mode:
        decision = "keep_robust_default_and_report_peak_ablation"
        rationale = (
            f"{peak_mode} achieved the strongest individual condition, but "
            f"{robust_mode} had the best aggregate robustness across the grid."
        )
    elif robust_mode:
        decision = "promote_robust_mode_candidate"
        rationale = f"{robust_mode} is both the best aggregate mode and the best observed peak mode."
    else:
        decision = "insufficient_data"
        rationale = "No ranked Track 4 method conditions were available."
    return {
        "decision": decision,
        "recommended_default_proposal_mode": robust_mode or None,
        "best_peak_proposal_mode": peak_mode or None,
        "best_peak_condition": best_peak,
        "best_robust_mode": best_robust,
        "rationale": rationale,
    }


def run_grid(
    *,
    feature_path: Path,
    output_dir: Path,
    feature_key: str,
    modes: Sequence[str],
    seeds: Sequence[int],
    k_values: Sequence[int],
    temperatures: Sequence[float],
    gammas: Sequence[float],
    max_steps: int,
    adaptive_tpt_connectivity: bool = False,
    compact_artifacts: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: List[Dict[str, Any]] = []
    for seed in seeds:
        for k_value in k_values:
            for temperature in temperatures:
                for gamma in gammas:
                    condition_dir = (
                        output_dir
                        / f"seed{int(seed)}"
                        / f"k{int(k_value)}"
                        / f"tau{_sanitize_float_for_path(float(temperature))}_gamma{_sanitize_float_for_path(float(gamma))}"
                    )
                    reports.append(
                        run_sweep(
                            feature_path=feature_path,
                            output_dir=condition_dir,
                            feature_key=feature_key,
                            modes=modes,
                            seed=int(seed),
                            max_steps=max_steps,
                            k_neighbors=int(k_value),
                            temperature=float(temperature),
                            gamma=float(gamma),
                            adaptive_tpt_connectivity=adaptive_tpt_connectivity,
                            compact_artifacts=compact_artifacts,
                        )
                    )

    condition_rows = _flatten_grid_reports(reports)
    ranked_conditions = sorted(
        condition_rows,
        key=lambda item: float(item["proposal_quality_score"]),
        reverse=True,
    )
    mode_robustness = _mode_robustness(condition_rows)
    ranked_k_modes = sorted(
        [
            {
                "k_neighbors": int(sub_report["k_neighbors"]),
                "seed": int(sub_report["seed"]),
                "temperature": float(sub_report["temperature"]),
                "gamma": float(sub_report["gamma"]),
                **sub_report["ranked_modes"][0],
            }
            for sub_report in reports
            if sub_report.get("ranked_modes")
        ],
        key=lambda item: float(item["proposal_quality_score"]),
        reverse=True,
    )
    report = {
        "status": "OK",
        "feature_path": str(feature_path),
        "feature_key": feature_key,
        "output_dir": str(output_dir),
        "seeds": [int(seed) for seed in seeds],
        "k_grid": [int(k_value) for k_value in k_values],
        "temperature_grid": [float(value) for value in temperatures],
        "gamma_grid": [float(value) for value in gammas],
        "max_steps": int(max_steps),
        "adaptive_tpt_connectivity": bool(adaptive_tpt_connectivity),
        "compact_artifacts": bool(compact_artifacts),
        "modes": [str(mode) for mode in modes],
        "condition_count": len(reports),
        "ranked_conditions": ranked_conditions,
        "ranked_k_modes": ranked_k_modes,
        "mode_robustness": mode_robustness,
        "method_recommendation": _method_recommendation(ranked_conditions, mode_robustness),
        "reports": reports,
    }
    (output_dir / "track4_grid_method_sweep_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    # Backward-compatible filename for earlier k-only sweeps and tests.
    (output_dir / "track4_kgrid_method_sweep_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_rows_csv(
        output_dir / "track4_ranked_conditions.csv",
        ranked_conditions,
        [
            "proposal_mode",
            "k_neighbors",
            "seed",
            "temperature",
            "gamma",
            "adaptive_tpt_connectivity",
            "proposal_quality_score",
            "delta_vs_metric_softmax",
            "closed_loop_rate",
            "mean_work_integral",
            "unique_path_shape_count",
            "path_shape_entropy_norm",
            "path_edge_entropy_norm",
            "mean_path_edge_count",
            "safe_for_thesis_claim",
            "reactive_flux_total",
            "effective_k_neighbors",
            "connectivity_repair_applied",
            "failure_reasons",
            "mode_dir",
        ],
    )
    _write_rows_csv(
        output_dir / "track4_mode_robustness.csv",
        mode_robustness,
        [
            "proposal_mode",
            "condition_count",
            "safe_condition_count",
            "safe_rate",
            "mean_score",
            "std_score",
            "min_score",
            "max_score",
            "mean_delta_vs_metric_softmax",
            "mean_closed_loop_rate",
            "mean_reactive_flux_total",
        ],
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-key", default="cls_stacked")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument(
        "--k-grid",
        nargs="*",
        type=int,
        default=None,
        help="Optional neighbor-count grid. When set, writes one sweep under output-dir/k{K}.",
    )
    parser.add_argument(
        "--seed-grid",
        nargs="*",
        type=str,
        default=None,
        help="Optional seed grid. Also accepts a single comma-separated value such as 42,420,4200.",
    )
    parser.add_argument(
        "--temperature-grid",
        nargs="*",
        type=str,
        default=None,
        help="Optional walker-temperature grid. Also accepts comma-separated values.",
    )
    parser.add_argument(
        "--gamma-grid",
        nargs="*",
        type=str,
        default=None,
        help="Optional retreat-penalty grid. Also accepts comma-separated values.",
    )
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--adaptive-tpt-connectivity", action="store_true")
    parser.add_argument(
        "--keep-path-xyz",
        action="store_true",
        help="Keep high-dimensional path coordinates in cyclic_paths.npz. Default compacts lab sweep artifacts.",
    )
    args = parser.parse_args()

    k_grid = args.k_grid or [int(args.k_neighbors)]
    seed_grid: List[int] = []
    for raw in args.seed_grid or [str(args.seed)]:
        seed_grid.extend(_csv_numbers(raw, cast=int))
    temperature_grid: List[float] = []
    for raw in args.temperature_grid or [str(args.temperature)]:
        temperature_grid.extend(_csv_numbers(raw, cast=float))
    gamma_grid: List[float] = []
    for raw in args.gamma_grid or [str(args.gamma)]:
        gamma_grid.extend(_csv_numbers(raw, cast=float))

    should_run_grid = (
        bool(args.k_grid)
        or bool(args.seed_grid)
        or bool(args.temperature_grid)
        or bool(args.gamma_grid)
    )
    if should_run_grid:
        report = run_grid(
            feature_path=args.features,
            output_dir=args.output_dir,
            feature_key=args.feature_key,
            modes=args.modes,
            seeds=[int(value) for value in seed_grid],
            k_values=[int(value) for value in k_grid],
            temperatures=[float(value) for value in temperature_grid],
            gammas=[float(value) for value in gamma_grid],
            max_steps=args.max_steps,
            adaptive_tpt_connectivity=bool(args.adaptive_tpt_connectivity),
            compact_artifacts=not bool(args.keep_path_xyz),
        )
    else:
        report = run_sweep(
            feature_path=args.features,
            output_dir=args.output_dir,
            feature_key=args.feature_key,
            modes=args.modes,
            seed=args.seed,
            max_steps=args.max_steps,
            k_neighbors=args.k_neighbors,
            temperature=args.temperature,
            gamma=args.gamma,
            adaptive_tpt_connectivity=bool(args.adaptive_tpt_connectivity),
            compact_artifacts=not bool(args.keep_path_xyz),
        )
    print("Track 4 method sweep complete")
    print(f"- output_dir: {report['output_dir']}")
    if "ranked_conditions" in report:
        print("- ranked conditions:")
        for row in report["ranked_conditions"][:10]:
            print(
                "  "
                f"seed={row['seed']} k={row['k_neighbors']} tau={row['temperature']} gamma={row['gamma']} "
                f"{row['proposal_mode']}: "
                f"score={row['proposal_quality_score']:.4f}, "
                f"delta={row['delta_vs_metric_softmax']:.4f}, "
                f"closed={row['closed_loop_rate']}"
            )
        print("- mode robustness:")
        for row in report["mode_robustness"]:
            print(
                "  "
                f"{row['proposal_mode']}: mean={row['mean_score']:.4f}, "
                f"std={row['std_score']:.4f}, safe={row['safe_rate']:.2f}"
            )
    else:
        print("- ranked modes:")
        for row in report["ranked_modes"]:
            print(
                "  "
                f"{row['proposal_mode']}: score={row['proposal_quality_score']:.4f}, "
                f"delta={row['delta_vs_metric_softmax']:.4f}, "
                f"closed={row['closed_loop_rate']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
