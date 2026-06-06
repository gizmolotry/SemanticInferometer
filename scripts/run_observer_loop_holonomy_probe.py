#!/usr/bin/env python3
"""Probe higher-order observer-atlas holonomy with closed triangular loops.

The pairwise commutator asks whether "move then switch observer" differs from
"switch observer then move."  This probe asks a stronger, more geometric
question: around a closed article/observer triangle, does the action depend on
loop orientation?  That is closer to a Wilson-loop style curvature diagnostic
than a single two-step commutator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observer_slice_transport import ObserverSliceTransportConfig  # noqa: E402
from scripts.run_observer_slice_transport_scale_suite import (  # noqa: E402
    _infer_run_identity,
    _load_payload,
    _observer_switch_vector,
    _pairwise_sq_dists,
    _safe_float,
    _semantic_action_vector,
)
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _config_id,
    _dedupe_paths,
    _null_slices,
    _payloads_for_mode,
    _prepare_slices,
    _stable_seed,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_loop_holonomy_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_loop_holonomy_probe" / "latest"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _rate(values: Iterable[Any]) -> float:
    vals = [bool(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _default_configs() -> list[dict[str, Any]]:
    return [
        {
            "label": "rupture_raw_cls_wide_loop",
            "feature_source": "raw_cls",
            "projection_dim_cap": 512,
            "normalization": "zscore_per_slice",
            "pair_mode": "farthest",
            "triangle_mode": "wide",
            "null_mode": "independent_article_shuffle",
            "weight_profile": "switch_heavy",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.5,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "rupture_raw_cls_stratified_loop",
            "feature_source": "raw_cls",
            "projection_dim_cap": 512,
            "normalization": "global_zscore",
            "pair_mode": "distance_stratified",
            "triangle_mode": "stratified",
            "null_mode": "shared_article_shuffle",
            "weight_profile": "semantic_heavy",
            "semantic_weight": 1.5,
            "observer_switch_weight": 1.0,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "local_rks_nearest_loop",
            "feature_source": "rks",
            "projection_dim_cap": 128,
            "normalization": "zscore_per_slice",
            "pair_mode": "nearest",
            "triangle_mode": "local",
            "null_mode": "dimension_signflip_by_slice",
            "weight_profile": "switch_heavy",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.5,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "current_baseline_loop",
            "feature_source": "rks",
            "projection_dim_cap": 512,
            "normalization": "none",
            "pair_mode": "mixed",
            "triangle_mode": "mixed",
            "null_mode": "zero",
            "weight_profile": "default_action",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.0,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
    ]


def _action_config(config: Mapping[str, Any]) -> ObserverSliceTransportConfig:
    return ObserverSliceTransportConfig(
        semantic_weight=float(config["semantic_weight"]),
        observer_switch_weight=float(config["observer_switch_weight"]),
        stress_weight=float(config["stress_weight"]),
        density_weight=float(config["density_weight"]),
    )


def _select_article_triples(
    reference: np.ndarray,
    *,
    triangle_mode: str,
    max_triangles: int,
    neighbor_count: int,
    random_seed: int,
    triangle_dim_cap: Optional[int],
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    ref = np.asarray(reference, dtype=np.float64)
    if triangle_dim_cap is not None and int(triangle_dim_cap) > 0:
        ref = ref[:, : min(int(triangle_dim_cap), int(ref.shape[1]))]
    n_items = int(ref.shape[0])
    if n_items < 3:
        return [], {"triangle_mode": triangle_mode, "reason": "fewer_than_three_articles"}
    d2 = _pairwise_sq_dists(ref)
    rng = np.random.default_rng(int(random_seed))
    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def add(i: int, j: int, k: int) -> None:
        if len({int(i), int(j), int(k)}) < 3:
            return
        key = tuple(sorted((int(i), int(j), int(k))))
        if key not in seen:
            seen.add(key)
            triples.append((int(i), int(j), int(k)))

    if triangle_mode == "local":
        for i in range(n_items):
            order = [int(j) for j in np.argsort(d2[i]) if int(j) != i]
            neighbors = order[: max(2, int(neighbor_count))]
            for pos in range(len(neighbors) - 1):
                add(i, neighbors[pos], neighbors[pos + 1])
                if len(triples) >= int(max_triangles):
                    break
            if len(triples) >= int(max_triangles):
                break
    elif triangle_mode == "wide":
        for i in range(n_items):
            order = [int(j) for j in np.argsort(d2[i]) if int(j) != i]
            nearest = order[: max(1, int(neighbor_count))]
            farthest = order[-max(1, int(neighbor_count)) :]
            for j in nearest:
                for k in reversed(farthest):
                    add(i, j, k)
                    if len(triples) >= int(max_triangles):
                        break
                if len(triples) >= int(max_triangles):
                    break
            if len(triples) >= int(max_triangles):
                break
    elif triangle_mode == "stratified":
        all_triples = [(i, j, k) for i in range(n_items) for j in range(i + 1, n_items) for k in range(j + 1, n_items)]
        if len(all_triples) <= int(max_triangles):
            triples = all_triples
        else:
            perimeter = np.asarray([d2[i, j] + d2[j, k] + d2[k, i] for i, j, k in all_triples], dtype=np.float64)
            order = np.argsort(perimeter)
            buckets = np.array_split(order, max(1, min(8, int(max_triangles))))
            for bucket in buckets:
                if len(triples) >= int(max_triangles) or len(bucket) == 0:
                    continue
                size = max(1, int(max_triangles) // len(buckets))
                picks = rng.choice(bucket, size=min(size, len(bucket)), replace=False)
                for raw in np.asarray(picks).reshape(-1):
                    add(*all_triples[int(raw)])
                    if len(triples) >= int(max_triangles):
                        break
    elif triangle_mode == "mixed":
        all_triples = [(i, j, k) for i in range(n_items) for j in range(i + 1, n_items) for k in range(j + 1, n_items)]
        idx = rng.choice(len(all_triples), size=min(len(all_triples), int(max_triangles)), replace=False)
        triples = [all_triples[int(raw)] for raw in np.asarray(idx).reshape(-1)]
    else:
        raise ValueError(f"unknown triangle mode: {triangle_mode}")
    return triples[: int(max_triangles)], {
        "triangle_mode": triangle_mode,
        "triangle_count": len(triples[: int(max_triangles)]),
        "candidate_article_count": n_items,
        "neighbor_count": int(neighbor_count),
        "random_seed": int(random_seed),
        "triangle_dim_cap": triangle_dim_cap,
    }


def _edge_semantic(
    coords: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    density: np.ndarray,
    stress: np.ndarray,
    config: ObserverSliceTransportConfig,
) -> np.ndarray:
    action, _ = _semantic_action_vector(
        coords,
        source,
        target,
        density=density,
        stress=stress,
        config=config,
    )
    return action


def _edge_switch(
    source_slice: np.ndarray,
    target_slice: np.ndarray,
    article: np.ndarray,
    *,
    config: ObserverSliceTransportConfig,
) -> np.ndarray:
    action, _ = _observer_switch_vector(source_slice, target_slice, article, config=config)
    return action


def _loop_summary(
    slices: Mapping[str, np.ndarray],
    *,
    article_triples: Sequence[tuple[int, int, int]],
    density: np.ndarray,
    stress: np.ndarray,
    config: ObserverSliceTransportConfig,
) -> dict[str, Any]:
    names = list(slices)
    if not article_triples or len(names) < 3:
        return {
            "record_count": 0,
            "mean_orientation_holonomy": None,
            "mean_relative_orientation_holonomy": None,
            "positive_orientation_rate": 0.0,
            "top_records": [],
        }
    idx_i = np.asarray([triple[0] for triple in article_triples], dtype=np.int64)
    idx_j = np.asarray([triple[1] for triple in article_triples], dtype=np.int64)
    idx_k = np.asarray([triple[2] for triple in article_triples], dtype=np.int64)
    record_count = 0
    holonomy_sum = 0.0
    relative_sum = 0.0
    positive = 0
    top: list[dict[str, Any]] = []
    for a_name in names:
        a_chart = slices[a_name]
        for b_name in names:
            if b_name == a_name:
                continue
            b_chart = slices[b_name]
            for c_name in names:
                if c_name in {a_name, b_name}:
                    continue
                c_chart = slices[c_name]
                forward = (
                    _edge_semantic(a_chart, idx_i, idx_j, density=density, stress=stress, config=config)
                    + _edge_switch(a_chart, b_chart, idx_j, config=config)
                    + _edge_semantic(b_chart, idx_j, idx_k, density=density, stress=stress, config=config)
                    + _edge_switch(b_chart, c_chart, idx_k, config=config)
                    + _edge_semantic(c_chart, idx_k, idx_i, density=density, stress=stress, config=config)
                    + _edge_switch(c_chart, a_chart, idx_i, config=config)
                )
                reverse = (
                    _edge_semantic(a_chart, idx_i, idx_k, density=density, stress=stress, config=config)
                    + _edge_switch(a_chart, c_chart, idx_k, config=config)
                    + _edge_semantic(c_chart, idx_k, idx_j, density=density, stress=stress, config=config)
                    + _edge_switch(c_chart, b_chart, idx_j, config=config)
                    + _edge_semantic(b_chart, idx_j, idx_i, density=density, stress=stress, config=config)
                    + _edge_switch(b_chart, a_chart, idx_i, config=config)
                )
                gap = forward - reverse
                holonomy = np.abs(gap)
                route_min = np.maximum(np.minimum(forward, reverse), 1e-12)
                relative = holonomy / route_min
                record_count += int(holonomy.size)
                holonomy_sum += float(np.sum(holonomy))
                relative_sum += float(np.sum(relative))
                positive += int(np.sum(holonomy > 1e-9))
                if holonomy.size:
                    top_idx = int(np.argmax(holonomy))
                    top.append(
                        {
                            "observer_loop": f"{a_name}->{b_name}->{c_name}->{a_name}",
                            "article_triangle": [
                                int(idx_i[top_idx]),
                                int(idx_j[top_idx]),
                                int(idx_k[top_idx]),
                            ],
                            "forward_action": float(forward[top_idx]),
                            "reverse_action": float(reverse[top_idx]),
                            "orientation_gap": float(gap[top_idx]),
                            "orientation_holonomy": float(holonomy[top_idx]),
                            "relative_orientation_holonomy": float(relative[top_idx]),
                        }
                    )
                    top = sorted(top, key=lambda row: float(row["orientation_holonomy"]), reverse=True)[:12]
    return {
        "record_count": record_count,
        "mean_orientation_holonomy": float(holonomy_sum / record_count) if record_count else None,
        "mean_relative_orientation_holonomy": float(relative_sum / record_count) if record_count else None,
        "positive_orientation_rate": float(positive / record_count) if record_count else 0.0,
        "top_records": top,
    }


def _calibrate_summary(summary: Mapping[str, Any], null_summary: Optional[Mapping[str, Any]], *, null_mode: str) -> dict[str, Any]:
    raw = _safe_float(summary.get("mean_orientation_holonomy"))
    raw_rel = _safe_float(summary.get("mean_relative_orientation_holonomy"))
    null = _safe_float((null_summary or {}).get("mean_orientation_holonomy")) if null_summary else 0.0
    null_rel = _safe_float((null_summary or {}).get("mean_relative_orientation_holonomy")) if null_summary else 0.0
    return {
        **dict(summary),
        "null_mode": null_mode,
        "mean_null_orientation_holonomy": null,
        "mean_null_relative_orientation_holonomy": null_rel,
        "mean_calibrated_orientation_holonomy": None if raw is None or null is None else float(raw - null),
        "mean_calibrated_relative_orientation_holonomy": None
        if raw_rel is None or null_rel is None
        else float(raw_rel - null_rel),
        "null_record_count": (null_summary or {}).get("record_count", 0) if null_summary else 0,
    }


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    max_triangles: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    triangle_dim_cap: Optional[int],
) -> dict[str, Any]:
    identity = _infer_run_identity(payload_path)
    slices, density, stress, metadata = _prepare_slices(
        payload_path,
        payload,
        config,
        device=device,
        normalization=str(config["normalization"]),
        article_cap=article_cap,
        random_seed=random_seed,
    )
    reference = np.mean(np.stack(list(slices.values()), axis=1), axis=1)
    triples, triangle_meta = _select_article_triples(
        reference,
        triangle_mode=str(config["triangle_mode"]),
        max_triangles=max_triangles,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        triangle_dim_cap=triangle_dim_cap,
    )
    action_config = _action_config(config)
    summary = _loop_summary(
        slices,
        article_triples=triples,
        density=density,
        stress=stress,
        config=action_config,
    )
    null_seed = _stable_seed(payload_path, _config_id(config), "loop_holonomy", random_seed)
    null_chart = _null_slices(slices, mode=str(config["null_mode"]), random_seed=null_seed)
    null_summary = None
    if null_chart is not None:
        null_summary = _loop_summary(
            null_chart,
            article_triples=triples,
            density=density,
            stress=stress,
            config=action_config,
        )
    calibrated = _calibrate_summary(summary, null_summary, null_mode=str(config["null_mode"]))
    return {
        **identity,
        "config_id": _config_id(config),
        "config_label": config.get("label"),
        "config": {key: value for key, value in dict(config).items() if key != "label"},
        "n_articles": int(next(iter(slices.values())).shape[0]),
        "n_slices": len(slices),
        "projection": metadata.get("projection"),
        "normalization": metadata.get("normalization"),
        "field_sources": metadata.get("field_sources"),
        "triangle_selection": triangle_meta,
        "loop_holonomy": calibrated,
        "null_loop_holonomy": null_summary,
    }


def _corpus_metric(rows: Sequence[Mapping[str, Any]], corpus: str, metric: str) -> Optional[float]:
    return _mean((row.get("loop_holonomy") or {}).get(metric) for row in rows if row.get("corpus") == corpus)


def _config_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    real = _corpus_metric(rows, "real", "mean_calibrated_orientation_holonomy")
    real_rel = _corpus_metric(rows, "real", "mean_calibrated_relative_orientation_holonomy")
    controls = _mean(
        _corpus_metric(rows, corpus, "mean_calibrated_orientation_holonomy")
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    controls_rel = _mean(
        _corpus_metric(rows, corpus, "mean_calibrated_relative_orientation_holonomy")
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    synthetic = _corpus_metric(rows, "synthetic", "mean_calibrated_orientation_holonomy")
    gap = float(real - controls) if real is not None and controls is not None else None
    rel_gap = float(real_rel - controls_rel) if real_rel is not None and controls_rel is not None else None
    ratio = float(real / max(controls, 1e-12)) if real is not None and controls is not None else None
    return {
        "config_label": first.get("config_label"),
        "config_id": first.get("config_id"),
        "config": first.get("config"),
        "payload_count": len(rows),
        "real_mean_calibrated_orientation_holonomy": real,
        "control_mean_calibrated_orientation_holonomy": controls,
        "synthetic_mean_calibrated_orientation_holonomy": synthetic,
        "real_minus_control_orientation_gap": gap,
        "real_minus_control_relative_orientation_gap": rel_gap,
        "real_to_control_orientation_ratio": ratio,
        "effect_pass_rate": _rate((row.get("loop_holonomy") or {}).get("mean_calibrated_orientation_holonomy", 0.0) > 0.0 for row in rows),
        "mean_record_count": _mean((row.get("loop_holonomy") or {}).get("record_count") for row in rows),
    }


def build_loop_holonomy_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    device_name: str,
    max_triangles: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    triangle_dim_cap: Optional[int],
    configs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    selected_configs = [dict(config) for config in (configs or _default_configs())]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for payload_path in payload_paths:
        try:
            payload = _load_payload(payload_path)
        except Exception as exc:
            failures.append({"payload_path": str(payload_path), "error": str(exc)})
            continue
        for config in selected_configs:
            try:
                rows.append(
                    _run_one(
                        payload_path,
                        payload,
                        config,
                        device=device,
                        max_triangles=max_triangles,
                        neighbor_count=neighbor_count,
                        random_seed=random_seed,
                        article_cap=article_cap,
                        triangle_dim_cap=triangle_dim_cap,
                    )
                )
            except Exception as exc:
                failures.append({"payload_path": str(payload_path), "config_label": config.get("label"), "error": str(exc)})
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("config_label") or row.get("config_id"))].append(row)
    config_summaries = sorted(
        [_config_summary(config_rows) for config_rows in grouped.values()],
        key=lambda row: float(row.get("real_minus_control_orientation_gap") or -1e12),
        reverse=True,
    )
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_loop_holonomy_claim_matrix",
        "claim_scope": "engineering_curvature_diagnostic_not_thesis_truth_claim",
        "claims": [
            {
                "claim_id": f"loop_holonomy_{summary.get('config_label')}",
                "claim_type": "engineering_hypothesis",
                "pass": bool((summary.get("real_minus_control_orientation_gap") or 0.0) > 0.0),
                "engineering_safe": bool((summary.get("real_minus_control_orientation_gap") or 0.0) > 0.0),
                "thesis_safe": False,
                "point_estimate": summary.get("real_minus_control_orientation_gap"),
                "artifact_family": "observer_loop_holonomy_probe.json",
            }
            for summary in config_summaries
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at_utc": claim_matrix["generated_at_utc"],
        "output_dir": str(output_dir),
        "device": str(device),
        "config": {
            "payload_count": len(payload_paths),
            "max_triangles": int(max_triangles),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "triangle_dim_cap": triangle_dim_cap,
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "orientation-dependent closed-loop action exists in observer/article triangles",
            "unsafe_claim": "this proves literal Riemannian curvature or ideological correctness without further controls",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "config_summaries": config_summaries,
            "interpretation": (
                "Positive real/control loop gap suggests higher-order atlas curvature beyond pairwise commutators."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_loop_holonomy_probe.json"
    csv_path = output_dir / "observer_loop_holonomy_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_label",
                "corpus",
                "kernel",
                "cell_id",
                "triangle_mode",
                "mean_calibrated_orientation_holonomy",
                "mean_calibrated_relative_orientation_holonomy",
                "record_count",
                "payload_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            loop = row.get("loop_holonomy") or {}
            triangle = row.get("triangle_selection") or {}
            writer.writerow(
                {
                    "config_label": row.get("config_label"),
                    "corpus": row.get("corpus"),
                    "kernel": row.get("kernel"),
                    "cell_id": row.get("cell_id"),
                    "triangle_mode": triangle.get("triangle_mode"),
                    "mean_calibrated_orientation_holonomy": loop.get("mean_calibrated_orientation_holonomy"),
                    "mean_calibrated_relative_orientation_holonomy": loop.get(
                        "mean_calibrated_relative_orientation_holonomy"
                    ),
                    "record_count": loop.get("record_count"),
                    "payload_path": row.get("payload_path"),
                }
            )
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="defaults")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-triangles", type=int, default=96)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--triangle-dim-cap", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_loop_holonomy_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        device_name=str(args.device),
        max_triangles=int(args.max_triangles),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        triangle_dim_cap=int(args.triangle_dim_cap) if int(args.triangle_dim_cap) > 0 else None,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "config_summaries": artifact["summary"]["config_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
