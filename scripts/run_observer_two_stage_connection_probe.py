#!/usr/bin/env python3
"""Test two-stage observer chart connections for Track 4.

Stage 1 learns a chart connection from one set of route endpoints.  Stage 2
evaluates Track 4 action on a disjoint set of route pairs.  This checks whether
the observer connection transfers, rather than merely fitting the same path
endpoints used for alignment.
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

from scripts.run_observer_gauge_invariance_probe import (  # noqa: E402
    _default_configs,
    _run_summary,
    _transform_slices,
)
from scripts.run_observer_gauge_repair_probe import (  # noqa: E402
    _farthest_indices,
    _repair_slices,
)
from scripts.run_observer_route_reserved_anchor_probe import (  # noqa: E402
    _nearest_route_distance,
    _route_endpoint_indices,
)
from scripts.run_observer_slice_transport_scale_suite import (  # noqa: E402
    _infer_run_identity,
    _load_payload,
    _safe_float,
)
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _config_id,
    _dedupe_paths,
    _payloads_for_mode,
    _prepare_slices,
    _select_pairs_by_mode,
    _stable_seed,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_two_stage_connection_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_two_stage_connection_probe" / "latest"


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


def _gap(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    real = _mean((row.get("transport") or {}).get(metric) for row in rows if row.get("corpus") == "real")
    controls = _mean(
        _mean((row.get("transport") or {}).get(metric) for row in rows if row.get("corpus") == corpus)
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    return float(real - controls) if real is not None and controls is not None else None


def _excess(summary: Mapping[str, Any]) -> Optional[float]:
    return _safe_float(summary.get("mean_calibrated_excess_holonomy_action"))


def _recovery(identity: Optional[float], damaged: Optional[float], repaired: Optional[float]) -> Optional[float]:
    if identity is None or damaged is None or repaired is None:
        return None
    denom = abs(float(damaged) - float(identity))
    if denom <= 1e-12:
        return None
    return float(1.0 - (abs(float(repaired) - float(identity)) / denom))


def _split_connection_eval_pairs(
    article_pairs: Sequence[tuple[int, int]],
    *,
    n_items: int,
    mean_chart: np.ndarray,
    slices: Mapping[str, np.ndarray],
    stress: Optional[np.ndarray],
    connection_pair_count: int,
    eval_pair_count: int,
    max_connection_endpoint_fraction: float,
    eval_pair_strategy: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, Any]]:
    endpoint_limit = max(2, min(int(n_items) - 2, int(math.floor(int(n_items) * float(max_connection_endpoint_fraction)))))
    connection: list[tuple[int, int]] = []
    endpoints: set[int] = set()
    for left, right in article_pairs:
        pair = (int(left), int(right))
        candidate = endpoints | {pair[0], pair[1]}
        if len(connection) < int(connection_pair_count) and len(candidate) <= endpoint_limit:
            connection.append(pair)
            endpoints = candidate
        if len(connection) >= int(connection_pair_count):
            break
    if not connection and article_pairs:
        first = (int(article_pairs[0][0]), int(article_pairs[0][1]))
        connection = [first]
        endpoints = {first[0], first[1]}
    candidates: list[tuple[int, int]] = []
    for left, right in article_pairs:
        pair = (int(left), int(right))
        if pair in connection:
            continue
        if pair[0] in endpoints or pair[1] in endpoints:
            continue
        candidates.append(pair)
    ordered_candidates = _order_eval_pairs(
        candidates,
        mean_chart=mean_chart,
        slices=slices,
        stress=stress,
        strategy=eval_pair_strategy,
    )
    eval_pairs: list[tuple[int, int]] = []
    for pair in ordered_candidates:
        eval_pairs.append(pair)
        if len(eval_pairs) >= int(eval_pair_count):
            break
    return connection, eval_pairs, {
        "source_pair_count": int(len(article_pairs)),
        "connection_pair_count": int(len(connection)),
        "connection_endpoint_count": int(len(endpoints)),
        "connection_endpoint_limit": int(endpoint_limit),
        "eval_pair_strategy": str(eval_pair_strategy),
        "eval_candidate_count": int(len(candidates)),
        "eval_pair_count": int(len(eval_pairs)),
        "eval_disjoint_from_connection": True,
        "max_connection_endpoint_fraction": float(max_connection_endpoint_fraction),
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    scale = float(np.std(arr))
    if scale <= 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / scale


def _order_eval_pairs(
    candidate_pairs: Sequence[tuple[int, int]],
    *,
    mean_chart: np.ndarray,
    slices: Mapping[str, np.ndarray],
    stress: Optional[np.ndarray],
    strategy: str,
) -> list[tuple[int, int]]:
    pairs = [(int(left), int(right)) for left, right in candidate_pairs]
    if strategy == "ordered" or not pairs:
        return pairs
    chart = np.asarray(mean_chart, dtype=np.float64)
    pair_index = np.asarray(pairs, dtype=np.int64)
    distances = np.linalg.norm(chart[pair_index[:, 0]] - chart[pair_index[:, 1]], axis=1)
    stack = np.stack([np.asarray(coords, dtype=np.float64) for coords in slices.values()], axis=1)
    disagreement = np.linalg.norm(np.var(stack, axis=1), axis=1)
    pair_disagreement = (
        disagreement[pair_index[:, 0]]
        + disagreement[pair_index[:, 1]]
        + np.abs(disagreement[pair_index[:, 0]] - disagreement[pair_index[:, 1]])
    )
    if stress is None:
        pair_stress = np.zeros(len(pairs), dtype=np.float64)
    else:
        stress_arr = np.asarray(stress, dtype=np.float64).reshape(-1)
        if stress_arr.size < chart.shape[0]:
            pair_stress = np.zeros(len(pairs), dtype=np.float64)
        else:
            pair_stress = (
                stress_arr[pair_index[:, 0]]
                + stress_arr[pair_index[:, 1]]
                + np.abs(stress_arr[pair_index[:, 0]] - stress_arr[pair_index[:, 1]])
            )
    if strategy == "farthest":
        score = distances
    elif strategy == "high_disagreement":
        score = pair_disagreement
    elif strategy == "high_stress":
        score = pair_stress
    elif strategy == "mixed_hard":
        score = _zscore(distances) + _zscore(pair_disagreement) + _zscore(pair_stress)
    else:
        raise ValueError(f"unknown eval pair strategy: {strategy}")
    order = np.argsort(score)[::-1]
    return [pairs[int(idx)] for idx in order]


def _augment_connection_anchors(
    slices: Mapping[str, np.ndarray],
    base_anchor_idx: np.ndarray,
    *,
    target_anchor_count: Optional[int],
    strategy: str,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    refs = {name: np.asarray(coords, dtype=np.float64) for name, coords in slices.items()}
    names = list(refs)
    n_items = int(next(iter(refs.values())).shape[0])
    base = np.unique(np.asarray(base_anchor_idx, dtype=np.int64).reshape(-1))
    target = n_items if target_anchor_count is None else max(int(base.size), min(int(target_anchor_count), n_items))
    if base.size >= target:
        return base, {
            "anchor_strategy": strategy,
            "base_anchor_count": int(base.size),
            "actual_anchor_count": int(base.size),
            "added_anchor_count": 0,
        }
    candidates = np.asarray([idx for idx in range(n_items) if int(idx) not in set(base.tolist())], dtype=np.int64)
    add_count = min(target - int(base.size), int(candidates.size))
    stack = np.stack([refs[name] for name in names], axis=1)
    mean_chart = np.mean(stack, axis=1)
    rng = np.random.default_rng(int(random_seed))
    if strategy == "connection_endpoints":
        extra = np.asarray([], dtype=np.int64)
    elif strategy == "connection_plus_random":
        extra = np.sort(rng.choice(candidates, size=add_count, replace=False)).astype(np.int64)
    elif strategy == "connection_plus_near_shell":
        dist = _nearest_route_distance(mean_chart, base, candidates)
        extra = np.sort(candidates[np.argsort(dist)[:add_count]]).astype(np.int64)
    elif strategy == "connection_plus_farthest":
        local = _farthest_indices(mean_chart[candidates], add_count)
        extra = np.sort(candidates[local]).astype(np.int64)
    elif strategy == "connection_plus_high_disagreement":
        disagreement = np.linalg.norm(np.var(stack, axis=1), axis=1)
        extra = np.sort(candidates[np.argsort(disagreement[candidates])[::-1][:add_count]]).astype(np.int64)
    else:
        raise ValueError(f"unknown two-stage anchor strategy: {strategy}")
    anchors = np.unique(np.concatenate([base, extra]))
    return anchors, {
        "anchor_strategy": strategy,
        "base_anchor_count": int(base.size),
        "target_anchor_count": None if target_anchor_count is None else int(target_anchor_count),
        "actual_anchor_count": int(anchors.size),
        "added_anchor_count": int(max(0, anchors.size - base.size)),
    }


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    connection_pair_count: int,
    target_anchor_count: Optional[int],
    anchor_strategy: str,
    transform_mode: str,
    repair_mode: str,
    device: torch.device,
    max_pairs: int,
    eval_pair_count: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_connection_endpoint_fraction: float,
    eval_pair_strategy: str,
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
    mean_chart = np.mean(np.stack(list(slices.values()), axis=1), axis=1)
    article_pairs, pair_meta = _select_pairs_by_mode(
        mean_chart,
        pair_mode=str(config["pair_mode"]),
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        pair_dim_cap=pair_dim_cap,
    )
    connection_pairs, eval_pairs, split_meta = _split_connection_eval_pairs(
        article_pairs,
        n_items=int(mean_chart.shape[0]),
        mean_chart=mean_chart,
        slices=slices,
        stress=stress,
        connection_pair_count=connection_pair_count,
        eval_pair_count=eval_pair_count,
        max_connection_endpoint_fraction=max_connection_endpoint_fraction,
        eval_pair_strategy=eval_pair_strategy,
    )
    connection_idx = _route_endpoint_indices(connection_pairs)
    eval_idx = _route_endpoint_indices(eval_pairs)
    anchors, anchor_meta = _augment_connection_anchors(
        slices,
        connection_idx,
        target_anchor_count=target_anchor_count,
        strategy=anchor_strategy,
        random_seed=_stable_seed(payload_path, _config_id(config), anchor_strategy, target_anchor_count or 0, random_seed),
    )
    transformed, transform_meta = _transform_slices(
        slices,
        mode=transform_mode,
        seed=_stable_seed(payload_path, _config_id(config), transform_mode, random_seed),
    )
    repaired, repair_meta = _repair_slices(
        transformed,
        slices,
        mode=repair_mode,
        anchor_count=target_anchor_count,
        anchor_strategy=anchor_strategy,
        random_seed=random_seed,
        anchor_indices=anchors,
    )
    identity_summary = _run_summary(
        slices,
        article_pairs=eval_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode="identity",
        random_seed=random_seed,
    )
    damaged_summary = _run_summary(
        transformed,
        article_pairs=eval_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode=transform_mode,
        random_seed=random_seed,
    )
    repaired_summary = _run_summary(
        repaired,
        article_pairs=eval_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode=transform_mode,
        random_seed=random_seed,
    )
    identity_excess = _excess(identity_summary)
    damaged_excess = _excess(damaged_summary)
    repaired_excess = _excess(repaired_summary)
    overlap = len(set(connection_idx.tolist()) & set(eval_idx.tolist()))
    return {
        **identity,
        "config_label": config.get("label"),
        "config_id": _config_id(config),
        "config": {key: value for key, value in dict(config).items() if key != "label"},
        "connection_pair_count": int(connection_pair_count),
        "target_anchor_count": None if target_anchor_count is None else int(target_anchor_count),
        "target_anchor_label": "all" if target_anchor_count is None else str(int(target_anchor_count)),
        "anchor_strategy": anchor_strategy,
        "transform_mode": transform_mode,
        "repair_mode": repair_mode,
        "n_articles": int(next(iter(slices.values())).shape[0]),
        "n_slices": len(slices),
        "projection": metadata.get("projection"),
        "normalization": metadata.get("normalization"),
        "field_sources": metadata.get("field_sources"),
        "pair_selection": pair_meta,
        "split": split_meta,
        "connection_eval_endpoint_overlap": int(overlap),
        "anchor": anchor_meta,
        "transform": transform_meta,
        "repair": repair_meta,
        "identity_transport": identity_summary,
        "damaged_transport": damaged_summary,
        "transport": repaired_summary,
        "identity_excess": identity_excess,
        "damaged_excess": damaged_excess,
        "repaired_excess": repaired_excess,
        "recovery_fraction": _recovery(identity_excess, damaged_excess, repaired_excess),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("config_label")),
                int(row.get("connection_pair_count") or 0),
                str(row.get("target_anchor_label")),
                str(row.get("anchor_strategy")),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_label, connection_count, target_label, strategy), group_rows in grouped.items():
        real_rows = [row for row in group_rows if row.get("corpus") == "real"]
        real_identity = _mean(row.get("identity_excess") for row in real_rows)
        real_damaged = _mean(row.get("damaged_excess") for row in real_rows)
        real_repaired = _mean(row.get("repaired_excess") for row in real_rows)
        real_recovery = _recovery(real_identity, real_damaged, real_repaired)
        mean_eval_pairs = _mean((row.get("split") or {}).get("eval_pair_count") for row in group_rows)
        mean_overlap = _mean(row.get("connection_eval_endpoint_overlap") for row in group_rows)
        pass_flag = bool(
            real_recovery is not None
            and real_recovery >= 0.75
            and mean_eval_pairs is not None
            and mean_eval_pairs >= 8
            and mean_overlap == 0.0
        )
        summaries.append(
            {
                "config_label": config_label,
                "connection_pair_count": int(connection_count),
                "target_anchor_label": target_label,
                "target_anchor_count": None if target_label == "all" else int(target_label),
                "anchor_strategy": strategy,
                "payload_count": len(group_rows),
                "mean_eval_pair_count": mean_eval_pairs,
                "mean_connection_endpoint_count": _mean(
                    (row.get("split") or {}).get("connection_endpoint_count") for row in group_rows
                ),
                "mean_actual_anchor_count": _mean((row.get("anchor") or {}).get("actual_anchor_count") for row in group_rows),
                "mean_connection_eval_endpoint_overlap": mean_overlap,
                "real_identity_excess": real_identity,
                "real_damaged_excess": real_damaged,
                "real_repaired_excess": real_repaired,
                "real_recovery_fraction": real_recovery,
                "mean_row_recovery_fraction": _mean(row.get("recovery_fraction") for row in group_rows),
                "real_minus_control_gap": _gap(group_rows, "mean_calibrated_excess_holonomy_action"),
                "pass": pass_flag,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["config_label"],
            row["connection_pair_count"],
            10**12 if row["target_anchor_count"] is None else int(row["target_anchor_count"]),
            row["anchor_strategy"],
        ),
    )


def _best_by_config(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[str(row["config_label"])].append(row)
    out: list[dict[str, Any]] = []
    for config_label, rows in grouped.items():
        best = sorted(
            rows,
            key=lambda row: (
                0 if row.get("pass") else 1,
                10**12 if row.get("target_anchor_count") is None else int(row["target_anchor_count"]),
                int(row["connection_pair_count"]),
                -1e9 if row.get("real_recovery_fraction") is None else -float(row["real_recovery_fraction"]),
            ),
        )[0]
        out.append(
            {
                "config_label": config_label,
                "best_anchor_strategy": best.get("anchor_strategy"),
                "best_connection_pair_count": best.get("connection_pair_count"),
                "best_target_anchor_count": best.get("target_anchor_count"),
                "best_real_recovery_fraction": best.get("real_recovery_fraction"),
                "best_pass": bool(best.get("pass")),
            }
        )
    return sorted(out, key=lambda row: row["config_label"])


def build_two_stage_connection_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    connection_pair_counts: Sequence[int],
    target_anchor_counts: Sequence[Optional[int]],
    anchor_strategies: Sequence[str],
    transform_mode: str,
    repair_mode: str,
    device_name: str,
    max_pairs: int,
    eval_pair_count: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_connection_endpoint_fraction: float,
    eval_pair_strategy: str,
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
            for connection_pair_count in connection_pair_counts:
                for target_anchor_count in target_anchor_counts:
                    for anchor_strategy in anchor_strategies:
                        try:
                            rows.append(
                                _run_one(
                                    payload_path,
                                    payload,
                                    config,
                                    connection_pair_count=int(connection_pair_count),
                                    target_anchor_count=target_anchor_count,
                                    anchor_strategy=anchor_strategy,
                                    transform_mode=transform_mode,
                                    repair_mode=repair_mode,
                                    device=device,
                                    max_pairs=max_pairs,
                                    eval_pair_count=eval_pair_count,
                                    neighbor_count=neighbor_count,
                                    random_seed=random_seed,
                                    article_cap=article_cap,
                                    pair_dim_cap=pair_dim_cap,
                                    max_connection_endpoint_fraction=max_connection_endpoint_fraction,
                                    eval_pair_strategy=eval_pair_strategy,
                                )
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "payload_path": str(payload_path),
                                    "config_label": config.get("label"),
                                    "connection_pair_count": connection_pair_count,
                                    "target_anchor_count": target_anchor_count,
                                    "anchor_strategy": anchor_strategy,
                                    "error": str(exc),
                                }
                            )
    summaries = _summarize(rows)
    best_rows = _best_by_config(summaries)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_two_stage_connection_claim_matrix",
        "claim_scope": "engineering_transfer_of_chart_connection_not_semantic_truth_claim",
        "claims": [
            {
                "claim_id": f"two_stage_connection_{row['config_label']}",
                "claim_type": "engineering_hypothesis",
                "pass": bool(row.get("best_pass")),
                "engineering_safe": bool(row.get("best_pass")),
                "thesis_safe": False,
                "point_estimate": row.get("best_real_recovery_fraction"),
                "best_anchor_strategy": row.get("best_anchor_strategy"),
                "best_connection_pair_count": row.get("best_connection_pair_count"),
                "best_target_anchor_count": row.get("best_target_anchor_count"),
                "artifact_family": "observer_two_stage_connection_probe.json",
            }
            for row in best_rows
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
            "connection_pair_counts": [int(value) for value in connection_pair_counts],
            "target_anchor_counts": ["all" if value is None else int(value) for value in target_anchor_counts],
            "anchor_strategies": list(anchor_strategies),
            "transform_mode": transform_mode,
            "repair_mode": repair_mode,
            "max_pairs": int(max_pairs),
            "eval_pair_count": int(eval_pair_count),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "max_connection_endpoint_fraction": float(max_connection_endpoint_fraction),
            "eval_pair_strategy": str(eval_pair_strategy),
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "a chart connection learned from one route set can be evaluated on a disjoint route set",
            "unsafe_claim": "two-stage transfer proves all path semantics or article ideology",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "two_stage_summaries": summaries,
            "best_by_config": best_rows,
            "interpretation": (
                "A pass means the observer connection transfers to disjoint evaluation routes with recovery >= 0.75."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_two_stage_connection_probe.json"
    csv_path = output_dir / "observer_two_stage_connection_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "connection_pair_count",
            "target_anchor_label",
            "anchor_strategy",
            "mean_eval_pair_count",
            "mean_connection_endpoint_count",
            "mean_actual_anchor_count",
            "real_identity_excess",
            "real_damaged_excess",
            "real_repaired_excess",
            "real_recovery_fraction",
            "real_minus_control_gap",
            "pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})
    return payload


def _parse_optional_counts(values: Sequence[str]) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    for raw in values:
        value = str(raw).strip().lower()
        if value in {"all", "none", "0"}:
            out.append(None)
        else:
            out.append(int(value))
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="representative")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--connection-pair-counts", nargs="+", default=["8", "16", "32"])
    parser.add_argument("--target-anchor-counts", nargs="+", default=["0", "16", "32", "64"])
    parser.add_argument(
        "--anchor-strategies",
        nargs="+",
        default=[
            "connection_endpoints",
            "connection_plus_near_shell",
            "connection_plus_farthest",
            "connection_plus_high_disagreement",
        ],
    )
    parser.add_argument("--transform-mode", default="per_slice_orthogonal")
    parser.add_argument("--repair-mode", default="procrustes_to_original_slice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--eval-pair-count", type=int, default=96)
    parser.add_argument(
        "--eval-pair-strategy",
        choices=("ordered", "farthest", "high_disagreement", "high_stress", "mixed_hard"),
        default="ordered",
    )
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    parser.add_argument("--max-connection-endpoint-fraction", type=float, default=0.4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_two_stage_connection_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        connection_pair_counts=[int(value) for value in args.connection_pair_counts],
        target_anchor_counts=_parse_optional_counts(args.target_anchor_counts),
        anchor_strategies=[str(strategy) for strategy in args.anchor_strategies],
        transform_mode=str(args.transform_mode),
        repair_mode=str(args.repair_mode),
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        eval_pair_count=int(args.eval_pair_count),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
        max_connection_endpoint_fraction=float(args.max_connection_endpoint_fraction),
        eval_pair_strategy=str(args.eval_pair_strategy),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "best_by_config": artifact["summary"]["best_by_config"],
                    "two_stage_summaries": artifact["summary"]["two_stage_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
