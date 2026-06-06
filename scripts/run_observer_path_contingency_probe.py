#!/usr/bin/env python3
"""Measure observer-contingent path existence in Track 4 slices.

This is not a repair/optimization pass.  It records when an article-to-article
move is traversable in some observer slices but blocked in others, preserving
the nuance that failed paths may be semantically meaningful observer gates.
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

from core.observer_slice_transport import (  # noqa: E402
    ObserverSliceTransportConfig,
    summarize_observer_path_contingency,
)
from scripts.run_observer_gauge_invariance_probe import _default_configs  # noqa: E402
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
    _split_connection_eval_pairs,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_path_contingency_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_path_contingency_probe" / "latest"


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _action_config(config: Mapping[str, Any]) -> ObserverSliceTransportConfig:
    return ObserverSliceTransportConfig(
        semantic_weight=float(config["semantic_weight"]),
        observer_switch_weight=float(config["observer_switch_weight"]),
        stress_weight=float(config["stress_weight"]),
        density_weight=float(config["density_weight"]),
    )


def _candidate_configs(labels: Sequence[str]) -> list[dict[str, Any]]:
    configs = _default_configs()
    if not labels:
        return configs
    wanted = set(str(label) for label in labels)
    available = {str(config.get("label")) for config in configs}
    unknown = sorted(wanted - available)
    if unknown:
        raise ValueError(f"unknown config labels: {unknown}; available={sorted(available)}")
    selected = [config for config in configs if str(config.get("label")) in wanted]
    if not selected:
        raise ValueError("no configs selected")
    return selected


def _class_count(summary: Mapping[str, Any], key: str) -> int:
    counts = summary.get("path_class_counts") if isinstance(summary.get("path_class_counts"), Mapping) else {}
    try:
        return int(counts.get(key, 0))
    except Exception:
        return 0


def _top_gatekeeping_observers(summary: Mapping[str, Any]) -> list[str]:
    counts = summary.get("observer_blocker_counts") if isinstance(summary.get("observer_blocker_counts"), Mapping) else {}
    if not counts:
        return []
    max_count = max(int(value) for value in counts.values())
    return sorted(str(name) for name, value in counts.items() if int(value) == max_count)


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    max_pairs: int,
    eval_pair_count: int,
    eval_pair_strategy: str,
    connection_pair_count: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_connection_endpoint_fraction: float,
    availability_quantile: float,
    availability_action_cutoff: Optional[float],
    consensus_fraction: float,
    stable_cv_threshold: float,
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
    summary = summarize_observer_path_contingency(
        slices,
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
    top_gatekeepers = _top_gatekeeping_observers(summary)
    return {
        **identity,
        "config_label": config.get("label"),
        "config_id": _config_id(config),
        "config": {key: value for key, value in dict(config).items() if key != "label"},
        "n_articles": int(next(iter(slices.values())).shape[0]),
        "n_slices": len(slices),
        "projection": metadata.get("projection"),
        "normalization": metadata.get("normalization"),
        "field_sources": metadata.get("field_sources"),
        "row_to_article_index": metadata.get("article_indices"),
        "pair_selection": pair_meta,
        "split": split_meta,
        "connection_pair_count": int(len(connection_pairs)),
        "eval_pair_count": int(len(eval_pairs)),
        "path_contingency": summary,
        "observer_contingent_rate": summary.get("observer_contingent_rate"),
        "null_observer_contingent_rate": summary.get("null_observer_contingent_rate"),
        "excess_observer_contingent_rate": summary.get("excess_observer_contingent_rate"),
        "observer_subset_required_rate": summary.get("observer_subset_required_rate"),
        "null_observer_subset_required_rate": summary.get("null_observer_subset_required_rate"),
        "excess_observer_subset_required_rate": summary.get("excess_observer_subset_required_rate"),
        "consensus_path_count": _class_count(summary, "consensus_path"),
        "consensus_but_warped_count": _class_count(summary, "consensus_but_warped"),
        "observer_contingent_path_count": _class_count(summary, "observer_contingent_path"),
        "universal_barrier_count": _class_count(summary, "universal_barrier"),
        "top_gatekeeping_observers": top_gatekeepers,
        "top_gatekeeping_observer": top_gatekeepers[0] if top_gatekeepers else None,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("config_label")), str(row.get("corpus")))].append(row)
    out: list[dict[str, Any]] = []
    for (config_label, corpus), group_rows in grouped.items():
        out.append(
            {
                "config_label": config_label,
                "corpus": corpus,
                "payload_count": len(group_rows),
                "mean_eval_pair_count": _mean(row.get("eval_pair_count") for row in group_rows),
                "mean_observer_contingent_rate": _mean(row.get("observer_contingent_rate") for row in group_rows),
                "mean_null_observer_contingent_rate": _mean(row.get("null_observer_contingent_rate") for row in group_rows),
                "mean_excess_observer_contingent_rate": _mean(row.get("excess_observer_contingent_rate") for row in group_rows),
                "mean_observer_subset_required_rate": _mean(row.get("observer_subset_required_rate") for row in group_rows),
                "mean_null_observer_subset_required_rate": _mean(row.get("null_observer_subset_required_rate") for row in group_rows),
                "mean_excess_observer_subset_required_rate": _mean(row.get("excess_observer_subset_required_rate") for row in group_rows),
                "mean_consensus_path_count": _mean(row.get("consensus_path_count") for row in group_rows),
                "mean_consensus_but_warped_count": _mean(row.get("consensus_but_warped_count") for row in group_rows),
                "mean_observer_contingent_path_count": _mean(row.get("observer_contingent_path_count") for row in group_rows),
                "mean_universal_barrier_count": _mean(row.get("universal_barrier_count") for row in group_rows),
                "top_gatekeeping_observers": sorted(
                    {
                        observer
                        for row in group_rows
                        for observer in (row.get("top_gatekeeping_observers") or [])
                    }
                ),
            }
        )
    return sorted(out, key=lambda row: (row["config_label"], row["corpus"]))


def build_path_contingency_probe(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    config_labels: Sequence[str],
    device_name: str,
    max_pairs: int,
    eval_pair_count: int,
    eval_pair_strategy: str,
    connection_pair_count: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_connection_endpoint_fraction: float,
    availability_quantile: float,
    availability_action_cutoff: Optional[float],
    consensus_fraction: float,
    stable_cv_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_configs = _candidate_configs(config_labels)
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
                        max_pairs=max_pairs,
                        eval_pair_count=eval_pair_count,
                        eval_pair_strategy=eval_pair_strategy,
                        connection_pair_count=connection_pair_count,
                        neighbor_count=neighbor_count,
                        random_seed=_stable_seed(payload_path, _config_id(config), random_seed),
                        article_cap=article_cap,
                        pair_dim_cap=pair_dim_cap,
                        max_connection_endpoint_fraction=max_connection_endpoint_fraction,
                        availability_quantile=availability_quantile,
                        availability_action_cutoff=availability_action_cutoff,
                        consensus_fraction=consensus_fraction,
                        stable_cv_threshold=stable_cv_threshold,
                    )
                )
            except Exception as exc:
                failures.append({"payload_path": str(payload_path), "config_label": config.get("label"), "error": str(exc)})
    summaries = _summarize(rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "summary_type": "observer_path_contingency_claim_matrix",
        "claim_scope": "observer_contingent_traversability_not_ideology_correctness",
        "claims": [
            {
                "claim_id": f"path_contingency_{row['config_label']}_{row['corpus']}",
                "claim_type": "diagnostic_observation",
                "pass": bool((row.get("mean_observer_contingent_rate") or 0.0) > 0.0),
                "engineering_safe": True,
                "thesis_safe": False,
                "point_estimate": row.get("mean_observer_contingent_rate"),
                "artifact_family": "observer_path_contingency_probe.json",
            }
            for row in summaries
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
            "config_labels": [config.get("label") for config in selected_configs],
            "max_pairs": int(max_pairs),
            "eval_pair_count": int(eval_pair_count),
            "eval_pair_strategy": str(eval_pair_strategy),
            "connection_pair_count": int(connection_pair_count),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "max_connection_endpoint_fraction": float(max_connection_endpoint_fraction),
            "availability_quantile": float(availability_quantile),
            "availability_action_cutoff": availability_action_cutoff,
            "consensus_fraction": float(consensus_fraction),
            "stable_cv_threshold": float(stable_cv_threshold),
        },
        "claim_boundary": {
            "safe_claim": "some hard paths are traversable only for subsets of observer slices",
            "unsafe_claim": "observer-contingent traversability proves real-world ideology labels are correct",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "path_contingency_summaries": summaries,
            "interpretation": "Observer-contingent paths are edges whose low-action existence depends on observer slice.",
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_path_contingency_probe.json"
    csv_path = output_dir / "observer_path_contingency_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "corpus",
            "payload_count",
            "mean_eval_pair_count",
            "mean_observer_contingent_rate",
            "mean_null_observer_contingent_rate",
            "mean_excess_observer_contingent_rate",
            "mean_observer_subset_required_rate",
            "mean_null_observer_subset_required_rate",
            "mean_excess_observer_subset_required_rate",
            "mean_consensus_path_count",
            "mean_consensus_but_warped_count",
            "mean_observer_contingent_path_count",
            "mean_universal_barrier_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="representative")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-labels", nargs="*", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--eval-pair-count", type=int, default=96)
    parser.add_argument(
        "--eval-pair-strategy",
        choices=("ordered", "farthest", "high_disagreement", "high_stress", "mixed_hard"),
        default="mixed_hard",
    )
    parser.add_argument("--connection-pair-count", type=int, default=8)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    parser.add_argument("--max-connection-endpoint-fraction", type=float, default=0.4)
    parser.add_argument("--availability-quantile", type=float, default=0.35)
    parser.add_argument("--availability-action-cutoff", type=float, default=None)
    parser.add_argument("--consensus-fraction", type=float, default=0.75)
    parser.add_argument("--stable-cv-threshold", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_path_contingency_probe(
        payload_paths=payloads,
        output_dir=args.output_dir,
        config_labels=[str(label) for label in args.config_labels],
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        eval_pair_count=int(args.eval_pair_count),
        eval_pair_strategy=str(args.eval_pair_strategy),
        connection_pair_count=int(args.connection_pair_count),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
        max_connection_endpoint_fraction=float(args.max_connection_endpoint_fraction),
        availability_quantile=float(args.availability_quantile),
        availability_action_cutoff=args.availability_action_cutoff,
        consensus_fraction=float(args.consensus_fraction),
        stable_cv_threshold=float(args.stable_cv_threshold),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "path_contingency_summaries": artifact["summary"]["path_contingency_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
