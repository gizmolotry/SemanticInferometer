#!/usr/bin/env python3
"""Measure which Track 4 path classes are repaired by each chart strategy.

The connection repair ablation answers whether a repair restores hard-route
transport in aggregate.  This companion ledger keeps the same repair mechanics
but stratifies the evidence by observer path class:

* consensus_path
* consensus_but_warped
* observer_contingent_path
* universal_barrier

That lets us separate a real observer-gated repair from a repair that only
helps easy consensus edges, or an endpoint-leaking oracle that should not count
as a strict transfer claim.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from core.observer_slice_transport import summarize_observer_path_contingency  # noqa: E402
from scripts.run_observer_connection_repair_ablation import (  # noqa: E402
    REPAIR_VARIANTS,
    _augment_connection_anchors_excluding_eval,
    _edge_level_anchors,
    _excess,
    _graph_stable_pairs,
    _hybrid_slices,
    _repair_multi_chart,
    _stress_boundary_anchors,
)
from scripts.run_observer_gauge_repair_probe import _repair_slices  # noqa: E402
from scripts.run_observer_gauge_invariance_probe import (  # noqa: E402
    _default_configs,
    _run_summary,
    _transform_slices,
)
from scripts.run_observer_path_contingency_probe import _action_config  # noqa: E402
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
    _json_safe,
    _recovery,
    _split_connection_eval_pairs,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_repair_by_path_class_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_repair_by_path_class_probe" / "latest"
PATH_CLASSES = (
    "consensus_path",
    "consensus_but_warped",
    "observer_contingent_path",
    "universal_barrier",
)


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _candidate_configs(labels: Sequence[str]) -> list[dict[str, Any]]:
    configs = _default_configs()
    if not labels:
        return configs
    wanted = set(str(label) for label in labels)
    available = {str(config.get("label")) for config in configs}
    unknown = sorted(wanted - available)
    if unknown:
        raise ValueError(f"unknown config labels: {unknown}; available={sorted(available)}")
    return [config for config in configs if str(config.get("label")) in wanted]


def _transport_for_pairs(
    slices: Mapping[str, np.ndarray],
    *,
    article_pairs: Sequence[tuple[int, int]],
    density: np.ndarray,
    stress: np.ndarray,
    config: Mapping[str, Any],
    payload_path: Path,
    transform_mode: str,
    random_seed: int,
) -> tuple[Mapping[str, Any], Optional[float]]:
    summary = _run_summary(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode=transform_mode,
        random_seed=random_seed,
    )
    return summary, _excess(summary)


def _path_class_pair_groups(
    summary: Mapping[str, Any],
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, dict[str, Any]]]:
    groups: dict[str, list[tuple[int, int]]] = {path_class: [] for path_class in PATH_CLASSES}
    metrics: dict[str, dict[str, Any]] = {
        path_class: {
            "support_fractions": [],
            "edge_action_cvs": [],
            "gate_scores": [],
            "null_contingent_probabilities": [],
            "null_subset_required_probabilities": [],
        }
        for path_class in PATH_CLASSES
    }
    for record in summary.get("records") or []:
        path_class = str(record.get("path_class"))
        if path_class not in groups:
            groups[path_class] = []
            metrics[path_class] = {
                "support_fractions": [],
                "edge_action_cvs": [],
                "gate_scores": [],
                "null_contingent_probabilities": [],
                "null_subset_required_probabilities": [],
            }
        pair = (int(record["source_idx"]), int(record["target_idx"]))
        groups[path_class].append(pair)
        metrics[path_class]["support_fractions"].append(record.get("support_fraction"))
        metrics[path_class]["edge_action_cvs"].append(record.get("edge_action_cv"))
        metrics[path_class]["gate_scores"].append(record.get("observer_gate_score"))
        metrics[path_class]["null_contingent_probabilities"].append(
            record.get("null_observer_contingent_probability")
        )
        metrics[path_class]["null_subset_required_probabilities"].append(
            record.get("null_subset_required_probability")
        )
    return groups, metrics


def _class_result(
    *,
    path_class: str,
    pairs: Sequence[tuple[int, int]],
    class_metrics: Mapping[str, Any],
    eval_pair_total: int,
    identity_slices: Mapping[str, np.ndarray],
    transformed_slices: Mapping[str, np.ndarray],
    repaired_slices: Mapping[str, np.ndarray],
    density: np.ndarray,
    stress: np.ndarray,
    config: Mapping[str, Any],
    payload_path: Path,
    transform_mode: str,
    random_seed: int,
    min_class_pair_count: int,
    recovery_threshold: float,
    connection_eval_endpoint_overlap: int,
    anchor_eval_endpoint_overlap: int,
) -> dict[str, Any]:
    pair_count = int(len(pairs))
    if pair_count:
        identity_summary, identity_excess = _transport_for_pairs(
            identity_slices,
            article_pairs=pairs,
            density=density,
            stress=stress,
            config=config,
            payload_path=payload_path,
            transform_mode="identity",
            random_seed=random_seed,
        )
        damaged_summary, damaged_excess = _transport_for_pairs(
            transformed_slices,
            article_pairs=pairs,
            density=density,
            stress=stress,
            config=config,
            payload_path=payload_path,
            transform_mode=transform_mode,
            random_seed=random_seed,
        )
        repaired_summary, repaired_excess = _transport_for_pairs(
            repaired_slices,
            article_pairs=pairs,
            density=density,
            stress=stress,
            config=config,
            payload_path=payload_path,
            transform_mode=transform_mode,
            random_seed=random_seed,
        )
    else:
        identity_summary = {"status": "NO_RECORDS", "record_count": 0}
        damaged_summary = {"status": "NO_RECORDS", "record_count": 0}
        repaired_summary = {"status": "NO_RECORDS", "record_count": 0}
        identity_excess = damaged_excess = repaired_excess = None

    recovery = _recovery(identity_excess, damaged_excess, repaired_excess)
    pass_flag = bool(
        recovery is not None
        and recovery >= float(recovery_threshold)
        and pair_count >= int(min_class_pair_count)
        and connection_eval_endpoint_overlap == 0
    )
    strict_transfer_pass = bool(pass_flag and anchor_eval_endpoint_overlap == 0)
    transductive_oracle_pass = bool(pass_flag and anchor_eval_endpoint_overlap > 0)
    return {
        "path_class": str(path_class),
        "pair_count": pair_count,
        "pair_fraction": float(pair_count / max(1, int(eval_pair_total))),
        "mean_support_fraction": _mean(class_metrics.get("support_fractions") or []),
        "mean_edge_action_cv": _mean(class_metrics.get("edge_action_cvs") or []),
        "mean_observer_gate_score": _mean(class_metrics.get("gate_scores") or []),
        "mean_null_contingent_probability": _mean(
            class_metrics.get("null_contingent_probabilities") or []
        ),
        "mean_null_subset_required_probability": _mean(
            class_metrics.get("null_subset_required_probabilities") or []
        ),
        "identity_excess": identity_excess,
        "damaged_excess": damaged_excess,
        "repaired_excess": repaired_excess,
        "recovery_fraction": recovery,
        "identity_transport": identity_summary,
        "damaged_transport": damaged_summary,
        "transport": repaired_summary,
        "pass": pass_flag,
        "strict_transfer_pass": strict_transfer_pass,
        "transductive_oracle_pass": transductive_oracle_pass,
    }


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
    availability_quantile: float,
    availability_action_cutoff: Optional[float],
    consensus_fraction: float,
    stable_cv_threshold: float,
    min_class_pair_count: int,
    recovery_threshold: float,
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
        fallback_slices, _, _, _ = _prepare_slices(
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
        variant_meta.update({"fallback_config": fallback_config.get("label"), **hybrid_meta})

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
        eval_pairs, stable_meta = _graph_stable_pairs(
            primary_slices,
            eval_pairs,
            eval_pair_count=requested_eval_count,
        )
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

    connection_eval_overlap = len(set(connection_idx.tolist()) & set(eval_idx.tolist()))
    anchor_eval_overlap = len(set(np.asarray(anchors, dtype=np.int64).tolist()) & set(eval_idx.tolist()))
    path_summary = summarize_observer_path_contingency(
        eval_source_slices,
        article_pairs=eval_pairs,
        density=density,
        stress=stress,
        config=_action_config(config),
        row_to_article_index=metadata.get("article_indices"),
        availability_quantile=availability_quantile,
        availability_action_cutoff=availability_action_cutoff,
        consensus_fraction=consensus_fraction,
        stable_cv_threshold=stable_cv_threshold,
    )
    pair_groups, group_metrics = _path_class_pair_groups(path_summary)
    path_class_results = [
        _class_result(
            path_class=path_class,
            pairs=pair_groups.get(path_class, []),
            class_metrics=group_metrics.get(path_class, {}),
            eval_pair_total=len(eval_pairs),
            identity_slices=eval_source_slices,
            transformed_slices=transformed,
            repaired_slices=repaired,
            density=density,
            stress=stress,
            config=config,
            payload_path=payload_path,
            transform_mode=transform_mode,
            random_seed=random_seed,
            min_class_pair_count=min_class_pair_count,
            recovery_threshold=recovery_threshold,
            connection_eval_endpoint_overlap=connection_eval_overlap,
            anchor_eval_endpoint_overlap=anchor_eval_overlap,
        )
        for path_class in PATH_CLASSES
    ]
    return {
        **identity,
        "config_label": config.get("label"),
        "config_id": _config_id(config),
        "config": {key: value for key, value in dict(config).items() if key != "label"},
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
        "connection_eval_endpoint_overlap": int(connection_eval_overlap),
        "anchor_eval_endpoint_overlap": int(anchor_eval_overlap),
        "anchor": anchor_meta,
        "transform": transform_meta,
        "repair": repair_meta,
        "path_contingency": {
            key: value
            for key, value in dict(path_summary).items()
            if key != "records"
        },
        "path_class_results": path_class_results,
    }


def _iter_class_results(rows: Sequence[Mapping[str, Any]]) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for row in rows:
        for class_result in row.get("path_class_results") or []:
            yield row, class_result


def _summarize(rows: Sequence[Mapping[str, Any]], *, min_class_pair_count: int, recovery_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for row, class_result in _iter_class_results(rows):
        grouped[
            (
                str(row.get("config_label")),
                str(row.get("repair_variant")),
                int(row.get("connection_pair_count") or 0),
                str(row.get("target_anchor_label")),
                str(class_result.get("path_class")),
            )
        ].append((row, class_result))

    summaries: list[dict[str, Any]] = []
    for (config_label, repair_variant, connection_count, target_label, path_class), group in grouped.items():
        real_group = [(row, result) for row, result in group if row.get("corpus") == "real"]
        real_identity = _mean(result.get("identity_excess") for _, result in real_group)
        real_damaged = _mean(result.get("damaged_excess") for _, result in real_group)
        real_repaired = _mean(result.get("repaired_excess") for _, result in real_group)
        real_recovery = _recovery(real_identity, real_damaged, real_repaired)
        mean_pair_count = _mean(result.get("pair_count") for _, result in group)
        mean_anchor_overlap = _mean(row.get("anchor_eval_endpoint_overlap") for row, _ in group)
        mean_connection_overlap = _mean(row.get("connection_eval_endpoint_overlap") for row, _ in group)
        pass_flag = bool(
            real_recovery is not None
            and real_recovery >= float(recovery_threshold)
            and mean_pair_count is not None
            and mean_pair_count >= int(min_class_pair_count)
            and mean_connection_overlap == 0.0
        )
        strict_transfer_pass = bool(pass_flag and mean_anchor_overlap == 0.0)
        transductive_oracle_pass = bool(pass_flag and mean_anchor_overlap is not None and mean_anchor_overlap > 0.0)
        summaries.append(
            {
                "config_label": config_label,
                "repair_variant": repair_variant,
                "connection_pair_count": connection_count,
                "target_anchor_label": target_label,
                "target_anchor_count": None if target_label == "all" else int(target_label),
                "path_class": path_class,
                "payload_count": len(group),
                "mean_pair_count": mean_pair_count,
                "mean_pair_fraction": _mean(result.get("pair_fraction") for _, result in group),
                "mean_support_fraction": _mean(result.get("mean_support_fraction") for _, result in group),
                "mean_observer_gate_score": _mean(result.get("mean_observer_gate_score") for _, result in group),
                "mean_edge_action_cv": _mean(result.get("mean_edge_action_cv") for _, result in group),
                "mean_null_contingent_probability": _mean(
                    result.get("mean_null_contingent_probability") for _, result in group
                ),
                "mean_anchor_eval_endpoint_overlap": mean_anchor_overlap,
                "mean_connection_eval_endpoint_overlap": mean_connection_overlap,
                "real_identity_excess": real_identity,
                "real_damaged_excess": real_damaged,
                "real_repaired_excess": real_repaired,
                "real_recovery_fraction": real_recovery,
                "mean_row_recovery_fraction": _mean(result.get("recovery_fraction") for _, result in group),
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
            row["path_class"],
            row["connection_pair_count"],
            10**12 if row["target_anchor_count"] is None else int(row["target_anchor_count"]),
        ),
    )


def _best_by_config_variant_class(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["config_label"]), str(row["repair_variant"]), str(row["path_class"]))].append(row)
    out = []
    for (config_label, repair_variant, path_class), rows in grouped.items():
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
                "path_class": path_class,
                "best_connection_pair_count": best.get("connection_pair_count"),
                "best_target_anchor_count": best.get("target_anchor_count"),
                "best_real_recovery_fraction": best.get("real_recovery_fraction"),
                "best_mean_pair_count": best.get("mean_pair_count"),
                "best_strict_transfer_pass": bool(best.get("strict_transfer_pass")),
                "best_transductive_oracle_pass": bool(best.get("transductive_oracle_pass")),
                "best_mean_anchor_eval_endpoint_overlap": best.get("mean_anchor_eval_endpoint_overlap"),
            }
        )
    return sorted(out, key=lambda row: (row["config_label"], row["repair_variant"], row["path_class"]))


def build_repair_by_path_class_probe(
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
    availability_quantile: float,
    availability_action_cutoff: Optional[float],
    consensus_fraction: float,
    stable_cv_threshold: float,
    min_class_pair_count: int,
    recovery_threshold: float,
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
                                    availability_quantile=availability_quantile,
                                    availability_action_cutoff=availability_action_cutoff,
                                    consensus_fraction=consensus_fraction,
                                    stable_cv_threshold=stable_cv_threshold,
                                    min_class_pair_count=min_class_pair_count,
                                    recovery_threshold=recovery_threshold,
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
    summaries = _summarize(
        rows,
        min_class_pair_count=min_class_pair_count,
        recovery_threshold=recovery_threshold,
    )
    best = _best_by_config_variant_class(summaries)
    generated_at = datetime.now(timezone.utc).isoformat()
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "summary_type": "observer_repair_by_path_class_claim_matrix",
        "claim_scope": "which_track4_path_classes_are_restored_by_repair_variants",
        "claims": [
            {
                "claim_id": f"repair_by_path_class_{row['config_label']}_{row['repair_variant']}_{row['path_class']}",
                "claim_type": "engineering_hypothesis",
                "path_class": row["path_class"],
                "pass": bool(row.get("best_strict_transfer_pass")),
                "engineering_safe": bool(row.get("best_strict_transfer_pass")),
                "transductive_oracle_pass": bool(row.get("best_transductive_oracle_pass")),
                "thesis_safe": bool(row.get("best_strict_transfer_pass") and row["path_class"] != "universal_barrier"),
                "point_estimate": row.get("best_real_recovery_fraction"),
                "best_mean_pair_count": row.get("best_mean_pair_count"),
                "artifact_family": "observer_repair_by_path_class_probe.json",
            }
            for row in best
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at_utc": generated_at,
        "output_dir": str(output_dir),
        "device": str(device),
        "config": {
            "payload_count": len(payload_paths),
            "repair_variants": list(repair_variants),
            "config_labels": [config.get("label") for config in selected_configs],
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
            "availability_quantile": float(availability_quantile),
            "availability_action_cutoff": availability_action_cutoff,
            "consensus_fraction": float(consensus_fraction),
            "stable_cv_threshold": float(stable_cv_threshold),
            "min_class_pair_count": int(min_class_pair_count),
            "recovery_threshold": float(recovery_threshold),
        },
        "claim_boundary": {
            "safe_claim": "which path classes a repair restores under this ledger and split",
            "unsafe_claim": "a path-class repair pass proves ideology labels or real-world correctness",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "path_class_repair_summaries": summaries,
            "best_by_config_variant_class": best,
            "interpretation": "Strict pass means class-specific recovery meets threshold without eval endpoint anchors.",
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_repair_by_path_class_probe.json"
    csv_path = output_dir / "observer_repair_by_path_class_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "claim_matrix": str(claim_path),
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "repair_variant",
            "path_class",
            "connection_pair_count",
            "target_anchor_label",
            "mean_pair_count",
            "mean_pair_fraction",
            "mean_support_fraction",
            "mean_observer_gate_score",
            "mean_edge_action_cv",
            "mean_anchor_eval_endpoint_overlap",
            "strict_transfer_pass",
            "transductive_oracle_pass",
            "real_identity_excess",
            "real_damaged_excess",
            "real_repaired_excess",
            "real_recovery_fraction",
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
        out.append(None if value in {"all", "none"} else int(value))
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults"), default="representative")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-variants", nargs="+", default=["graph_stable_filter", "edge_level_connection"])
    parser.add_argument("--config-labels", nargs="*", default=["local_rks_nearest"])
    parser.add_argument("--connection-pair-counts", nargs="+", default=["8"])
    parser.add_argument("--target-anchor-counts", nargs="+", default=["16", "64"])
    parser.add_argument("--anchor-strategy", default="connection_plus_farthest")
    parser.add_argument("--transform-mode", default="per_slice_orthogonal")
    parser.add_argument("--repair-mode", default="procrustes_to_original_slice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--eval-pair-count", type=int, default=96)
    parser.add_argument(
        "--eval-pair-strategy",
        choices=("ordered", "farthest", "mixed_hard", "high_disagreement", "high_stress"),
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
    parser.add_argument("--availability-quantile", type=float, default=0.35)
    parser.add_argument("--availability-action-cutoff", type=float, default=None)
    parser.add_argument("--consensus-fraction", type=float, default=0.75)
    parser.add_argument("--stable-cv-threshold", type=float, default=0.25)
    parser.add_argument("--min-class-pair-count", type=int, default=4)
    parser.add_argument("--recovery-threshold", type=float, default=0.75)
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
    artifact = build_repair_by_path_class_probe(
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
        article_cap=int(args.article_cap) if args.article_cap > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if args.pair_dim_cap > 0 else None,
        max_connection_endpoint_fraction=float(args.max_connection_endpoint_fraction),
        chart_count=int(args.chart_count),
        graph_candidate_multiplier=int(args.graph_candidate_multiplier),
        hybrid_fallback_weight=float(args.hybrid_fallback_weight),
        availability_quantile=float(args.availability_quantile),
        availability_action_cutoff=args.availability_action_cutoff,
        consensus_fraction=float(args.consensus_fraction),
        stable_cv_threshold=float(args.stable_cv_threshold),
        min_class_pair_count=int(args.min_class_pair_count),
        recovery_threshold=float(args.recovery_threshold),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "best_by_config_variant_class": artifact["summary"]["best_by_config_variant_class"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
