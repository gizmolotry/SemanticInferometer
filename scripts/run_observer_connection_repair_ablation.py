#!/usr/bin/env python3
"""Ablate concrete repairs for hard observer-slice path transfer.

This suite starts from the two-stage connection failure case: local RKS-nearest
can transfer chart repair on ordinary held-out routes, but collapses on
mixed-hard routes.  The ablation tests five engineering repairs without
changing the recovery target:

* multi_chart: several local Procrustes charts instead of one global chart.
* edge_level_connection: include the current route endpoints as runtime anchors.
* stress_boundary_anchors: spend anchors on stress/disagreement seam nodes.
* graph_stable_filter: admit only hard routes with stable cross-slice edge lengths.
* hybrid_path_basis: concatenate the tested chart with the current baseline chart.
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
    _orthogonal_procrustes,
    _repair_slices,
)
from scripts.run_observer_route_reserved_anchor_probe import _route_endpoint_indices  # noqa: E402
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
from scripts.run_observer_two_stage_connection_probe import (  # noqa: E402
    _augment_connection_anchors,
    _gap,
    _json_safe,
    _recovery,
    _split_connection_eval_pairs,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_connection_repair_ablation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_connection_repair_ablation" / "latest"
REPAIR_VARIANTS = (
    "baseline_global",
    "multi_chart",
    "edge_level_connection",
    "stress_boundary_anchors",
    "graph_stable_filter",
    "hybrid_path_basis",
)


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _excess(summary: Mapping[str, Any]) -> Optional[float]:
    return _safe_float(summary.get("mean_calibrated_excess_holonomy_action"))


def _standardize(arr: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    coords = np.asarray(arr, dtype=np.float64)
    center = coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    return (coords - center) / np.where(scale > eps, scale, 1.0)


def _hybrid_slices(
    primary: Mapping[str, np.ndarray],
    fallback: Mapping[str, np.ndarray],
    *,
    fallback_weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    names = [name for name in primary if name in fallback]
    if len(names) < 2:
        raise ValueError("hybrid_path_basis requires at least two shared observer slices")
    out = {
        name: np.concatenate(
            [
                _standardize(np.asarray(primary[name], dtype=np.float64)),
                float(fallback_weight) * _standardize(np.asarray(fallback[name], dtype=np.float64)),
            ],
            axis=1,
        )
        for name in names
    }
    first = names[0]
    return out, {
        "hybrid_primary_dim": int(np.asarray(primary[first]).shape[1]),
        "hybrid_fallback_dim": int(np.asarray(fallback[first]).shape[1]),
        "hybrid_output_dim": int(out[first].shape[1]),
        "hybrid_fallback_weight": float(fallback_weight),
    }


def _node_boundary_score(
    slices: Mapping[str, np.ndarray],
    stress: Optional[np.ndarray],
) -> np.ndarray:
    stack = np.stack([np.asarray(coords, dtype=np.float64) for coords in slices.values()], axis=1)
    disagreement = np.linalg.norm(np.var(stack, axis=1), axis=1)
    if stress is None:
        stress_score = np.zeros_like(disagreement)
    else:
        stress_arr = np.asarray(stress, dtype=np.float64).reshape(-1)
        stress_score = stress_arr[: disagreement.shape[0]] if stress_arr.size >= disagreement.shape[0] else np.zeros_like(disagreement)
    def z(values: np.ndarray) -> np.ndarray:
        scale = float(np.std(values))
        return np.zeros_like(values) if scale <= 1e-12 else (values - float(np.mean(values))) / scale

    return z(disagreement) + z(stress_score)


def _stress_boundary_anchors(
    slices: Mapping[str, np.ndarray],
    base_anchor_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    target_anchor_count: Optional[int],
    stress: Optional[np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    n_items = int(next(iter(slices.values())).shape[0])
    base = np.unique(np.asarray(base_anchor_idx, dtype=np.int64).reshape(-1))
    target = n_items if target_anchor_count is None else max(int(base.size), min(int(target_anchor_count), n_items))
    forbidden = set(int(idx) for idx in np.asarray(eval_idx, dtype=np.int64).reshape(-1))
    candidates = np.asarray(
        [idx for idx in range(n_items) if idx not in forbidden and idx not in set(base.tolist())],
        dtype=np.int64,
    )
    add_count = min(max(0, target - int(base.size)), int(candidates.size))
    score = _node_boundary_score(slices, stress)
    extra = candidates[np.argsort(score[candidates])[::-1][:add_count]] if add_count else np.asarray([], dtype=np.int64)
    anchors = np.unique(np.concatenate([base, extra]))
    return anchors, {
        "anchor_policy": "stress_boundary_anchors",
        "base_anchor_count": int(base.size),
        "target_anchor_count": None if target_anchor_count is None else int(target_anchor_count),
        "actual_anchor_count": int(anchors.size),
        "excluded_eval_endpoint_count": int(len(forbidden)),
        "added_anchor_count": int(max(0, anchors.size - base.size)),
    }


def _augment_connection_anchors_excluding_eval(
    slices: Mapping[str, np.ndarray],
    base_anchor_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    target_anchor_count: Optional[int],
    strategy: str,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    refs = {name: np.asarray(coords, dtype=np.float64) for name, coords in slices.items()}
    names = list(refs)
    n_items = int(next(iter(refs.values())).shape[0])
    base = np.unique(np.asarray(base_anchor_idx, dtype=np.int64).reshape(-1))
    forbidden = set(int(idx) for idx in np.asarray(eval_idx, dtype=np.int64).reshape(-1))
    target = n_items if target_anchor_count is None else max(int(base.size), min(int(target_anchor_count), n_items))
    candidates = np.asarray(
        [idx for idx in range(n_items) if idx not in forbidden and idx not in set(base.tolist())],
        dtype=np.int64,
    )
    add_count = min(max(0, target - int(base.size)), int(candidates.size))
    stack = np.stack([refs[name] for name in names], axis=1)
    mean_chart = np.mean(stack, axis=1)
    rng = np.random.default_rng(int(random_seed))
    if add_count <= 0 or strategy == "connection_endpoints":
        extra = np.asarray([], dtype=np.int64)
    elif strategy == "connection_plus_random":
        extra = np.sort(rng.choice(candidates, size=add_count, replace=False)).astype(np.int64)
    elif strategy == "connection_plus_near_shell":
        route = base if base.size else np.arange(n_items, dtype=np.int64)
        dist = np.min(np.linalg.norm(mean_chart[candidates, None, :] - mean_chart[route][None, :, :], axis=2), axis=1)
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
        "anchor_policy": "connection_anchors_excluding_eval",
        "anchor_strategy": str(strategy),
        "base_anchor_count": int(base.size),
        "target_anchor_count": None if target_anchor_count is None else int(target_anchor_count),
        "actual_anchor_count": int(anchors.size),
        "excluded_eval_endpoint_count": int(len(forbidden)),
        "added_anchor_count": int(max(0, anchors.size - base.size)),
    }


def _edge_level_anchors(
    base_anchor_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    n_items: int,
    target_anchor_count: Optional[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    base = np.unique(np.asarray(base_anchor_idx, dtype=np.int64).reshape(-1))
    eval_nodes = np.unique(np.asarray(eval_idx, dtype=np.int64).reshape(-1))
    anchors = np.unique(np.concatenate([base, eval_nodes]))
    if target_anchor_count is not None and anchors.size < int(target_anchor_count):
        candidates = np.asarray([idx for idx in range(n_items) if idx not in set(anchors.tolist())], dtype=np.int64)
        add_count = min(int(target_anchor_count) - int(anchors.size), int(candidates.size))
        anchors = np.unique(np.concatenate([anchors, candidates[:add_count]]))
    return anchors, {
        "anchor_policy": "edge_level_connection",
        "base_anchor_count": int(base.size),
        "eval_endpoint_anchor_count": int(eval_nodes.size),
        "target_anchor_count": None if target_anchor_count is None else int(target_anchor_count),
        "actual_anchor_count": int(anchors.size),
    }


def _edge_length_stability(slices: Mapping[str, np.ndarray], pair: tuple[int, int]) -> float:
    lengths = []
    for coords in slices.values():
        arr = np.asarray(coords, dtype=np.float64)
        lengths.append(float(np.linalg.norm(arr[int(pair[0])] - arr[int(pair[1])])))
    vals = np.asarray(lengths, dtype=np.float64)
    mean = float(np.mean(vals))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(vals) / mean)


def _graph_stable_pairs(
    slices: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[int, int]],
    *,
    eval_pair_count: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    scored = [(_edge_length_stability(slices, pair), int(pos), pair) for pos, pair in enumerate(pairs)]
    scored.sort(key=lambda row: (row[0], row[1]))
    selected = [pair for _, _, pair in scored[: int(eval_pair_count)]]
    return selected, {
        "graph_stable_candidate_count": int(len(pairs)),
        "graph_stable_eval_pair_count": int(len(selected)),
        "mean_selected_edge_cv": _mean(score for score, _, _ in scored[: int(eval_pair_count)]),
        "mean_rejected_edge_cv": _mean(score for score, _, _ in scored[int(eval_pair_count) :]),
    }


def _repair_multi_chart(
    transformed: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    anchor_idx: np.ndarray,
    chart_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {name: np.asarray(coords, dtype=np.float64) for name, coords in transformed.items()}
    refs = {name: np.asarray(coords, dtype=np.float64) for name, coords in reference.items()}
    names = list(arrays)
    mean_chart = np.mean(np.stack([refs[name] for name in names], axis=1), axis=1)
    n_items = int(mean_chart.shape[0])
    centers = _farthest_indices(mean_chart, min(max(1, int(chart_count)), n_items))
    dists = np.linalg.norm(mean_chart[:, None, :] - mean_chart[centers][None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    anchor_idx = np.unique(np.asarray(anchor_idx, dtype=np.int64).reshape(-1))
    repaired = {name: np.zeros_like(arrays[name], dtype=np.float64) for name in names}
    residuals = []
    fallback = anchor_idx
    for cluster in range(int(centers.size)):
        members = np.where(labels == cluster)[0]
        if members.size == 0:
            continue
        local_anchors = np.asarray([idx for idx in anchor_idx if labels[int(idx)] == cluster], dtype=np.int64)
        # Below three anchors, the local chart is underidentified; use the full
        # anchor set rather than hallucinating a tiny local rotation.
        if local_anchors.size < 3:
            local_anchors = fallback
        for name in names:
            src = arrays[name]
            tgt = refs[name]
            src_mean = np.mean(src[local_anchors], axis=0, keepdims=True)
            tgt_mean = np.mean(tgt[local_anchors], axis=0, keepdims=True)
            rotation = _orthogonal_procrustes(src[local_anchors] - src_mean, tgt[local_anchors] - tgt_mean)
            aligned_members = (src[members] - src_mean) @ rotation + tgt_mean
            repaired[name][members] = aligned_members
            residuals.append(float(np.linalg.norm(((src[local_anchors] - src_mean) @ rotation + tgt_mean) - tgt[local_anchors]) / max(1, int(local_anchors.size))))
    return repaired, {
        "repair_mode": "multi_chart_procrustes",
        "chart_count": int(centers.size),
        "anchor_count": int(anchor_idx.size),
        "mean_anchor_residual": float(np.mean(residuals)) if residuals else None,
    }


def _candidate_configs(labels: Sequence[str]) -> list[dict[str, Any]]:
    configs = _default_configs()
    if not labels:
        return configs
    wanted = set(str(label) for label in labels)
    return [config for config in configs if str(config.get("label")) in wanted]


def _run_one_variant(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    repair_variant: str,
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
    chart_count: int,
    graph_candidate_multiplier: int,
    hybrid_fallback_weight: float,
) -> dict[str, Any]:
    identity = _infer_run_identity(payload_path)
    primary_slices, density, stress, metadata = _prepare_slices(
        payload_path,
        payload,
        config,
        device=device,
        normalization=str(config["normalization"]),
        article_cap=article_cap,
        random_seed=random_seed,
    )
    eval_source_slices = primary_slices
    variant_meta: dict[str, Any] = {"repair_variant": repair_variant}
    if repair_variant == "hybrid_path_basis":
        fallback_config = _default_configs()[0]
        fallback_slices, _, _, fallback_meta = _prepare_slices(
            payload_path,
            payload,
            fallback_config,
            device=device,
            normalization=str(fallback_config["normalization"]),
            article_cap=article_cap,
            random_seed=random_seed,
        )
        eval_source_slices, hybrid_meta = _hybrid_slices(
            primary_slices,
            fallback_slices,
            fallback_weight=hybrid_fallback_weight,
        )
        variant_meta.update({"fallback_config_label": fallback_config.get("label"), **hybrid_meta})

    mean_chart = np.mean(np.stack(list(primary_slices.values()), axis=1), axis=1)
    article_pairs, pair_meta = _select_pairs_by_mode(
        mean_chart,
        pair_mode=str(config["pair_mode"]),
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        pair_dim_cap=pair_dim_cap,
    )
    requested_eval_count = int(eval_pair_count)
    split_eval_count = requested_eval_count
    if repair_variant == "graph_stable_filter":
        split_eval_count = max(requested_eval_count, requested_eval_count * max(1, int(graph_candidate_multiplier)))
    connection_pairs, eval_pairs, split_meta = _split_connection_eval_pairs(
        article_pairs,
        n_items=int(mean_chart.shape[0]),
        mean_chart=mean_chart,
        slices=primary_slices,
        stress=stress,
        connection_pair_count=connection_pair_count,
        eval_pair_count=split_eval_count,
        max_connection_endpoint_fraction=max_connection_endpoint_fraction,
        eval_pair_strategy=eval_pair_strategy,
    )
    if repair_variant == "graph_stable_filter":
        eval_pairs, stable_meta = _graph_stable_pairs(primary_slices, eval_pairs, eval_pair_count=requested_eval_count)
        split_meta = {**split_meta, **stable_meta, "eval_pair_count": int(len(eval_pairs))}
    connection_idx = _route_endpoint_indices(connection_pairs)
    eval_idx = _route_endpoint_indices(eval_pairs)

    if repair_variant == "stress_boundary_anchors":
        anchors, anchor_meta = _stress_boundary_anchors(
            primary_slices,
            connection_idx,
            eval_idx,
            target_anchor_count=target_anchor_count,
            stress=stress,
        )
    elif repair_variant == "edge_level_connection":
        anchors, anchor_meta = _edge_level_anchors(
            connection_idx,
            eval_idx,
            n_items=int(mean_chart.shape[0]),
            target_anchor_count=target_anchor_count,
        )
    else:
        anchors, anchor_meta = _augment_connection_anchors_excluding_eval(
            primary_slices,
            connection_idx,
            eval_idx,
            target_anchor_count=target_anchor_count,
            strategy=anchor_strategy,
            random_seed=_stable_seed(
                payload_path,
                _config_id(config),
                repair_variant,
                anchor_strategy,
                target_anchor_count or 0,
                random_seed,
            ),
        )

    transformed, transform_meta = _transform_slices(
        eval_source_slices,
        mode=transform_mode,
        seed=_stable_seed(payload_path, _config_id(config), repair_variant, transform_mode, random_seed),
    )
    if repair_variant == "multi_chart":
        repaired, repair_meta = _repair_multi_chart(
            transformed,
            eval_source_slices,
            anchor_idx=anchors,
            chart_count=chart_count,
        )
    else:
        repaired, repair_meta = _repair_slices(
            transformed,
            eval_source_slices,
            mode=repair_mode,
            anchor_count=target_anchor_count,
            anchor_strategy=anchor_strategy,
            random_seed=random_seed,
            anchor_indices=anchors,
        )

    identity_summary = _run_summary(
        eval_source_slices,
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
    anchor_eval_overlap = len(set(np.asarray(anchors, dtype=np.int64).tolist()) & set(eval_idx.tolist()))
    return {
        **identity,
        "config_label": config.get("label"),
        "config_id": _config_id(config),
        "repair_variant": repair_variant,
        "variant": variant_meta,
        "connection_pair_count": int(connection_pair_count),
        "target_anchor_count": None if target_anchor_count is None else int(target_anchor_count),
        "target_anchor_label": "all" if target_anchor_count is None else str(int(target_anchor_count)),
        "anchor_strategy": anchor_strategy,
        "transform_mode": transform_mode,
        "repair_mode": repair_mode,
        "n_articles": int(next(iter(primary_slices.values())).shape[0]),
        "n_slices": len(primary_slices),
        "projection": metadata.get("projection"),
        "normalization": metadata.get("normalization"),
        "field_sources": metadata.get("field_sources"),
        "pair_selection": pair_meta,
        "split": split_meta,
        "connection_eval_endpoint_overlap": int(overlap),
        "anchor_eval_endpoint_overlap": int(anchor_eval_overlap),
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
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("config_label")),
                str(row.get("repair_variant")),
                int(row.get("connection_pair_count") or 0),
                str(row.get("target_anchor_label")),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_label, repair_variant, connection_count, target_label), group_rows in grouped.items():
        real_rows = [row for row in group_rows if row.get("corpus") == "real"]
        real_identity = _mean(row.get("identity_excess") for row in real_rows)
        real_damaged = _mean(row.get("damaged_excess") for row in real_rows)
        real_repaired = _mean(row.get("repaired_excess") for row in real_rows)
        real_recovery = _recovery(real_identity, real_damaged, real_repaired)
        mean_eval_pairs = _mean((row.get("split") or {}).get("eval_pair_count") for row in group_rows)
        mean_overlap = _mean(row.get("connection_eval_endpoint_overlap") for row in group_rows)
        mean_anchor_eval_overlap = _mean(row.get("anchor_eval_endpoint_overlap") for row in group_rows)
        pass_flag = bool(
            real_recovery is not None
            and real_recovery >= 0.75
            and mean_eval_pairs is not None
            and mean_eval_pairs >= 8
            and mean_overlap == 0.0
        )
        strict_transfer_pass = bool(pass_flag and (mean_anchor_eval_overlap is None or mean_anchor_eval_overlap <= 1e-12))
        transductive_oracle_pass = bool(pass_flag and mean_anchor_eval_overlap is not None and mean_anchor_eval_overlap > 1e-12)
        summaries.append(
            {
                "config_label": config_label,
                "repair_variant": repair_variant,
                "connection_pair_count": int(connection_count),
                "target_anchor_label": target_label,
                "target_anchor_count": None if target_label == "all" else int(target_label),
                "payload_count": len(group_rows),
                "mean_eval_pair_count": mean_eval_pairs,
                "mean_connection_endpoint_count": _mean((row.get("split") or {}).get("connection_endpoint_count") for row in group_rows),
                "mean_actual_anchor_count": _mean((row.get("anchor") or {}).get("actual_anchor_count") for row in group_rows),
                "mean_anchor_eval_endpoint_overlap": mean_anchor_eval_overlap,
                "mean_connection_eval_endpoint_overlap": mean_overlap,
                "real_identity_excess": real_identity,
                "real_damaged_excess": real_damaged,
                "real_repaired_excess": real_repaired,
                "real_recovery_fraction": real_recovery,
                "mean_row_recovery_fraction": _mean(row.get("recovery_fraction") for row in group_rows),
                "real_minus_control_gap": _gap(group_rows, "mean_calibrated_excess_holonomy_action"),
                "pass": pass_flag,
                "strict_transfer_pass": strict_transfer_pass,
                "transductive_oracle_pass": transductive_oracle_pass,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["config_label"],
            row["repair_variant"],
            row["connection_pair_count"],
            10**12 if row["target_anchor_count"] is None else int(row["target_anchor_count"]),
        ),
    )


def _best_by_config_variant(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["config_label"]), str(row["repair_variant"]))].append(row)
    out = []
    for (config_label, repair_variant), rows in grouped.items():
        best = sorted(
            rows,
            key=lambda row: (
                0 if row.get("strict_transfer_pass") else 1,
                0 if row.get("pass") else 1,
                10**12 if row.get("target_anchor_count") is None else int(row["target_anchor_count"]),
                int(row["connection_pair_count"]),
                -1e9 if row.get("real_recovery_fraction") is None else -float(row["real_recovery_fraction"]),
            ),
        )[0]
        out.append(
            {
                "config_label": config_label,
                "repair_variant": repair_variant,
                "best_connection_pair_count": best.get("connection_pair_count"),
                "best_target_anchor_count": best.get("target_anchor_count"),
                "best_real_recovery_fraction": best.get("real_recovery_fraction"),
                "best_pass": bool(best.get("pass")),
                "best_strict_transfer_pass": bool(best.get("strict_transfer_pass")),
                "best_transductive_oracle_pass": bool(best.get("transductive_oracle_pass")),
                "best_mean_actual_anchor_count": best.get("mean_actual_anchor_count"),
                "best_mean_anchor_eval_endpoint_overlap": best.get("mean_anchor_eval_endpoint_overlap"),
            }
        )
    return sorted(out, key=lambda row: (row["config_label"], row["repair_variant"]))


def build_connection_repair_ablation_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    repair_variants: Sequence[str],
    config_labels: Sequence[str],
    connection_pair_counts: Sequence[int],
    target_anchor_counts: Sequence[Optional[int]],
    anchor_strategy: str,
    transform_mode: str,
    repair_mode: str,
    device_name: str,
    max_pairs: int,
    eval_pair_count: int,
    eval_pair_strategy: str,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_connection_endpoint_fraction: float,
    chart_count: int,
    graph_candidate_multiplier: int,
    hybrid_fallback_weight: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    selected_configs = _candidate_configs(config_labels)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for payload_path in payload_paths:
        try:
            payload = _load_payload(payload_path)
        except Exception as exc:
            failures.append({"payload_path": str(payload_path), "error": str(exc)})
            continue
        for config in selected_configs:
            for repair_variant in repair_variants:
                for connection_pair_count in connection_pair_counts:
                    for target_anchor_count in target_anchor_counts:
                        try:
                            rows.append(
                                _run_one_variant(
                                    payload_path,
                                    payload,
                                    config,
                                    repair_variant=str(repair_variant),
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
                                    chart_count=chart_count,
                                    graph_candidate_multiplier=graph_candidate_multiplier,
                                    hybrid_fallback_weight=hybrid_fallback_weight,
                                )
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "payload_path": str(payload_path),
                                    "config_label": config.get("label"),
                                    "repair_variant": str(repair_variant),
                                    "connection_pair_count": int(connection_pair_count),
                                    "target_anchor_count": target_anchor_count,
                                    "error": str(exc),
                                }
                            )
    summaries = _summarize(rows)
    best_rows = _best_by_config_variant(summaries)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_connection_repair_ablation_claim_matrix",
        "claim_scope": "engineering_repair_candidates_for_hard_observer_paths",
        "claims": [
            {
                "claim_id": f"connection_repair_{row['config_label']}_{row['repair_variant']}",
                "claim_type": "engineering_hypothesis",
                "pass": bool(row.get("best_strict_transfer_pass")),
                "engineering_safe": bool(row.get("best_strict_transfer_pass")),
                "transductive_oracle_pass": bool(row.get("best_transductive_oracle_pass")),
                "thesis_safe": False,
                "point_estimate": row.get("best_real_recovery_fraction"),
                "best_connection_pair_count": row.get("best_connection_pair_count"),
                "best_target_anchor_count": row.get("best_target_anchor_count"),
                "artifact_family": "observer_connection_repair_ablation.json",
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
            "repair_variants": list(repair_variants),
            "connection_pair_counts": [int(value) for value in connection_pair_counts],
            "target_anchor_counts": ["all" if value is None else int(value) for value in target_anchor_counts],
            "anchor_strategy": str(anchor_strategy),
            "transform_mode": str(transform_mode),
            "repair_mode": str(repair_mode),
            "max_pairs": int(max_pairs),
            "eval_pair_count": int(eval_pair_count),
            "eval_pair_strategy": str(eval_pair_strategy),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "max_connection_endpoint_fraction": float(max_connection_endpoint_fraction),
            "chart_count": int(chart_count),
            "graph_candidate_multiplier": int(graph_candidate_multiplier),
            "hybrid_fallback_weight": float(hybrid_fallback_weight),
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "which repair candidates restore hard-route chart-transfer recovery under this ledger",
            "unsafe_claim": "a passing repair proves semantic ideology correctness or universal path validity",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "repair_summaries": summaries,
            "best_by_config_variant": best_rows,
            "interpretation": "A pass means hard-route recovery >= 0.75 with disjoint connection/eval endpoints.",
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_connection_repair_ablation.json"
    csv_path = output_dir / "observer_connection_repair_ablation.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "repair_variant",
            "connection_pair_count",
            "target_anchor_label",
            "mean_eval_pair_count",
            "mean_actual_anchor_count",
            "mean_anchor_eval_endpoint_overlap",
            "strict_transfer_pass",
            "transductive_oracle_pass",
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
        out.append(None if value in {"all", "none", "0"} else int(value))
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="representative")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-variants", nargs="+", default=list(REPAIR_VARIANTS))
    parser.add_argument("--config-labels", nargs="*", default=[])
    parser.add_argument("--connection-pair-counts", nargs="+", default=["8"])
    parser.add_argument("--target-anchor-counts", nargs="+", default=["16", "32", "64"])
    parser.add_argument("--anchor-strategy", default="connection_plus_farthest")
    parser.add_argument("--transform-mode", default="per_slice_orthogonal")
    parser.add_argument("--repair-mode", default="procrustes_to_original_slice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--eval-pair-count", type=int, default=96)
    parser.add_argument(
        "--eval-pair-strategy",
        choices=("ordered", "farthest", "high_disagreement", "high_stress", "mixed_hard"),
        default="mixed_hard",
    )
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    parser.add_argument("--max-connection-endpoint-fraction", type=float, default=0.4)
    parser.add_argument("--chart-count", type=int, default=6)
    parser.add_argument("--graph-candidate-multiplier", type=int, default=4)
    parser.add_argument("--hybrid-fallback-weight", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    unknown = sorted(set(str(variant) for variant in args.repair_variants) - set(REPAIR_VARIANTS))
    if unknown:
        raise SystemExit(f"unknown repair variants: {unknown}")
    artifact = build_connection_repair_ablation_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        repair_variants=[str(variant) for variant in args.repair_variants],
        config_labels=[str(label) for label in args.config_labels],
        connection_pair_counts=[int(value) for value in args.connection_pair_counts],
        target_anchor_counts=_parse_optional_counts(args.target_anchor_counts),
        anchor_strategy=str(args.anchor_strategy),
        transform_mode=str(args.transform_mode),
        repair_mode=str(args.repair_mode),
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        eval_pair_count=int(args.eval_pair_count),
        eval_pair_strategy=str(args.eval_pair_strategy),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
        max_connection_endpoint_fraction=float(args.max_connection_endpoint_fraction),
        chart_count=int(args.chart_count),
        graph_candidate_multiplier=int(args.graph_candidate_multiplier),
        hybrid_fallback_weight=float(args.hybrid_fallback_weight),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "best_by_config_variant": artifact["summary"]["best_by_config_variant"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
