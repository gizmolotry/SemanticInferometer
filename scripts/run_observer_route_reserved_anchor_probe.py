#!/usr/bin/env python3
"""Test chart repair when alignment anchors are disjoint from route endpoints.

The holdout probe showed that farthest anchors repair rupture charts well, but
they can consume the same extreme articles used as route endpoints.  This suite
selects Track 4 route pairs first, reserves every route endpoint for evaluation,
then chooses chart-alignment anchors only from the remaining article nodes.
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

from scripts.run_observer_alignment_anchor_sparsity_probe import _anchor_label, _parse_anchor_counts  # noqa: E402
from scripts.run_observer_gauge_invariance_probe import (  # noqa: E402
    _default_configs,
    _run_summary,
    _transform_slices,
)
from scripts.run_observer_gauge_repair_probe import (  # noqa: E402
    _farthest_indices,
    _repair_slices,
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
DIAGNOSTIC_TYPE = "observer_route_reserved_anchor_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_route_reserved_anchor_probe" / "latest"


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


def _route_endpoint_indices(article_pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    endpoints: set[int] = set()
    for left, right in article_pairs:
        endpoints.add(int(left))
        endpoints.add(int(right))
    return np.asarray(sorted(endpoints), dtype=np.int64)


def _reserve_route_pairs(
    article_pairs: Sequence[tuple[int, int]],
    *,
    n_items: int,
    max_endpoint_fraction: float,
    min_pairs: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    if not article_pairs:
        return [], {
            "source_pair_count": 0,
            "reserved_pair_count": 0,
            "endpoint_limit": 0,
            "endpoint_count": 0,
        }
    endpoint_limit = max(2, min(int(n_items) - 1, int(math.floor(float(n_items) * float(max_endpoint_fraction)))))
    selected: list[tuple[int, int]] = []
    endpoints: set[int] = set()
    for left, right in article_pairs:
        pair = (int(left), int(right))
        candidate_endpoints = endpoints | {pair[0], pair[1]}
        if len(selected) < int(min_pairs) or len(candidate_endpoints) <= endpoint_limit:
            selected.append(pair)
            endpoints = candidate_endpoints
    if not selected:
        selected = [(int(article_pairs[0][0]), int(article_pairs[0][1]))]
        endpoints = {selected[0][0], selected[0][1]}
    return selected, {
        "source_pair_count": int(len(article_pairs)),
        "reserved_pair_count": int(len(selected)),
        "endpoint_limit": int(endpoint_limit),
        "endpoint_count": int(len(endpoints)),
        "max_endpoint_fraction": float(max_endpoint_fraction),
        "min_pairs": int(min_pairs),
    }


def _nearest_route_distance(mean_chart: np.ndarray, route_idx: np.ndarray, candidate_idx: np.ndarray) -> np.ndarray:
    if route_idx.size == 0 or candidate_idx.size == 0:
        return np.zeros(candidate_idx.size, dtype=np.float64)
    route_points = mean_chart[route_idx]
    candidate_points = mean_chart[candidate_idx]
    best = np.full(candidate_idx.size, np.inf, dtype=np.float64)
    # Chunk to avoid a large temporary for bigger future manifolds.
    for start in range(0, route_points.shape[0], 128):
        block = route_points[start : start + 128]
        d2 = np.sum((candidate_points[:, None, :] - block[None, :, :]) ** 2, axis=2)
        best = np.minimum(best, np.min(d2, axis=1))
    return np.sqrt(best)


def _select_reserved_anchors(
    slices: Mapping[str, np.ndarray],
    route_idx: np.ndarray,
    *,
    anchor_count: Optional[int],
    strategy: str,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    refs = {name: np.asarray(coords, dtype=np.float64) for name, coords in slices.items()}
    names = list(refs)
    n_items = int(next(iter(refs.values())).shape[0])
    route_set = {int(idx) for idx in route_idx.reshape(-1)}
    candidates = np.asarray([idx for idx in range(n_items) if idx not in route_set], dtype=np.int64)
    requested = n_items if anchor_count is None else int(anchor_count)
    count = min(max(1, requested), int(candidates.size))
    if candidates.size == 0:
        raise ValueError("no non-route candidate anchors remain")
    stack = np.stack([refs[name] for name in names], axis=1)
    mean_chart = np.mean(stack, axis=1)
    rng = np.random.default_rng(int(random_seed))
    if strategy == "random_complement":
        chosen = np.sort(rng.choice(candidates, size=count, replace=False)).astype(np.int64)
    elif strategy == "farthest_complement":
        local = _farthest_indices(mean_chart[candidates], count)
        chosen = np.sort(candidates[local]).astype(np.int64)
    elif strategy == "near_route_shell":
        dist = _nearest_route_distance(mean_chart, route_idx, candidates)
        chosen = np.sort(candidates[np.argsort(dist)[:count]]).astype(np.int64)
    elif strategy == "mid_route_shell":
        dist = _nearest_route_distance(mean_chart, route_idx, candidates)
        median = float(np.median(dist))
        chosen = np.sort(candidates[np.argsort(np.abs(dist - median))[:count]]).astype(np.int64)
    elif strategy == "high_disagreement_complement":
        disagreement = np.linalg.norm(np.var(stack, axis=1), axis=1)
        order = candidates[np.argsort(disagreement[candidates])[::-1]]
        chosen = np.sort(order[:count]).astype(np.int64)
    else:
        raise ValueError(f"unknown reserved anchor strategy: {strategy}")
    overlap = int(len(set(chosen.tolist()) & route_set))
    return chosen, {
        "anchor_strategy": strategy,
        "requested_anchor_count": None if anchor_count is None else int(anchor_count),
        "actual_anchor_count": int(chosen.size),
        "route_endpoint_count": int(route_idx.size),
        "candidate_count": int(candidates.size),
        "route_anchor_overlap": overlap,
    }


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    anchor_count: Optional[int],
    anchor_strategy: str,
    transform_mode: str,
    repair_mode: str,
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_route_endpoint_fraction: float,
    min_route_pairs: int,
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
    article_pairs, route_budget_meta = _reserve_route_pairs(
        article_pairs,
        n_items=int(mean_chart.shape[0]),
        max_endpoint_fraction=max_route_endpoint_fraction,
        min_pairs=min_route_pairs,
    )
    route_idx = _route_endpoint_indices(article_pairs)
    anchor_idx, anchor_meta = _select_reserved_anchors(
        slices,
        route_idx,
        anchor_count=anchor_count,
        strategy=anchor_strategy,
        random_seed=_stable_seed(payload_path, _config_id(config), anchor_strategy, anchor_count or 0, random_seed),
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
        anchor_count=anchor_count,
        anchor_strategy=anchor_strategy,
        random_seed=random_seed,
        anchor_indices=anchor_idx,
    )
    identity_summary = _run_summary(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode="identity",
        random_seed=random_seed,
    )
    damaged_summary = _run_summary(
        transformed,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode=transform_mode,
        random_seed=random_seed,
    )
    repaired_summary = _run_summary(
        repaired,
        article_pairs=article_pairs,
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
    return {
        **identity,
        "config_label": config.get("label"),
        "config_id": _config_id(config),
        "config": {key: value for key, value in dict(config).items() if key != "label"},
        "anchor_count": None if anchor_count is None else int(anchor_count),
        "anchor_label": _anchor_label(anchor_count),
        "anchor_strategy": anchor_strategy,
        "transform_mode": transform_mode,
        "repair_mode": repair_mode,
        "n_articles": int(next(iter(slices.values())).shape[0]),
        "n_slices": len(slices),
        "projection": metadata.get("projection"),
        "normalization": metadata.get("normalization"),
        "field_sources": metadata.get("field_sources"),
        "pair_selection": pair_meta,
        "route_budget": route_budget_meta,
        "path_pair_count": int(len(article_pairs)),
        "route_endpoint_count": int(route_idx.size),
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
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("config_label")), str(row.get("anchor_label")), str(row.get("anchor_strategy")))].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_label, anchor_label, anchor_strategy), group_rows in grouped.items():
        real_rows = [row for row in group_rows if row.get("corpus") == "real"]
        real_identity = _mean(row.get("identity_excess") for row in real_rows)
        real_damaged = _mean(row.get("damaged_excess") for row in real_rows)
        real_repaired = _mean(row.get("repaired_excess") for row in real_rows)
        real_recovery = _recovery(real_identity, real_damaged, real_repaired)
        mean_overlap = _mean((row.get("anchor") or {}).get("route_anchor_overlap") for row in group_rows)
        mean_candidates = _mean((row.get("anchor") or {}).get("candidate_count") for row in group_rows)
        mean_actual_anchors = _mean((row.get("anchor") or {}).get("actual_anchor_count") for row in group_rows)
        pass_flag = bool(real_recovery is not None and real_recovery >= 0.75 and mean_overlap == 0.0)
        summaries.append(
            {
                "config_label": config_label,
                "anchor_label": anchor_label,
                "anchor_count": None if anchor_label == "all" else int(anchor_label),
                "anchor_strategy": anchor_strategy,
                "payload_count": len(group_rows),
                "mean_actual_anchor_count": mean_actual_anchors,
                "mean_candidate_count": mean_candidates,
                "mean_route_anchor_overlap": mean_overlap,
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
            str(row["config_label"]),
            10**12 if row["anchor_count"] is None else int(row["anchor_count"]),
            str(row["anchor_strategy"]),
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
                10**12 if row.get("anchor_count") is None else int(row["anchor_count"]),
                -1e9 if row.get("real_recovery_fraction") is None else -float(row["real_recovery_fraction"]),
            ),
        )[0]
        out.append(
            {
                "config_label": config_label,
                "best_anchor_strategy": best.get("anchor_strategy"),
                "best_anchor_count": best.get("anchor_count"),
                "best_real_recovery_fraction": best.get("real_recovery_fraction"),
                "best_pass": bool(best.get("pass")),
            }
        )
    return sorted(out, key=lambda row: row["config_label"])


def build_route_reserved_anchor_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    anchor_counts: Sequence[Optional[int]],
    anchor_strategies: Sequence[str],
    transform_mode: str,
    repair_mode: str,
    device_name: str,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    max_route_endpoint_fraction: float,
    min_route_pairs: int,
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
            for anchor_count in anchor_counts:
                for anchor_strategy in anchor_strategies:
                    try:
                        rows.append(
                            _run_one(
                                payload_path,
                                payload,
                                config,
                                anchor_count=anchor_count,
                                anchor_strategy=anchor_strategy,
                                transform_mode=transform_mode,
                                repair_mode=repair_mode,
                                device=device,
                                max_pairs=max_pairs,
                                neighbor_count=neighbor_count,
                                random_seed=random_seed,
                                article_cap=article_cap,
                                pair_dim_cap=pair_dim_cap,
                                max_route_endpoint_fraction=max_route_endpoint_fraction,
                                min_route_pairs=min_route_pairs,
                            )
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "payload_path": str(payload_path),
                                "config_label": config.get("label"),
                                "anchor_count": anchor_count,
                                "anchor_strategy": anchor_strategy,
                                "error": str(exc),
                            }
                        )
    summaries = _summarize(rows)
    best_rows = _best_by_config(summaries)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_route_reserved_anchor_claim_matrix",
        "claim_scope": "engineering_disjoint_alignment_and_route_landmarks_not_semantic_truth_claim",
        "claims": [
            {
                "claim_id": f"route_reserved_{row['config_label']}",
                "claim_type": "engineering_hypothesis",
                "pass": bool(row.get("best_pass")),
                "engineering_safe": bool(row.get("best_pass")),
                "thesis_safe": False,
                "point_estimate": row.get("best_real_recovery_fraction"),
                "best_anchor_strategy": row.get("best_anchor_strategy"),
                "best_anchor_count": row.get("best_anchor_count"),
                "artifact_family": "observer_route_reserved_anchor_probe.json",
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
            "anchor_counts": [_anchor_label(count) for count in anchor_counts],
            "anchor_strategies": list(anchor_strategies),
            "transform_mode": transform_mode,
            "repair_mode": repair_mode,
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "max_route_endpoint_fraction": float(max_route_endpoint_fraction),
            "min_route_pairs": int(min_route_pairs),
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "alignment anchors can be made disjoint from route endpoints and still recover path action",
            "unsafe_claim": "route-reserved repair validates every semantic interpretation of paths",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "route_reserved_summaries": summaries,
            "best_by_config": best_rows,
            "interpretation": (
                "A pass means chart alignment recovered the original action while using no route endpoints as anchors."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_route_reserved_anchor_probe.json"
    csv_path = output_dir / "observer_route_reserved_anchor_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "anchor_label",
            "anchor_strategy",
            "mean_actual_anchor_count",
            "mean_candidate_count",
            "mean_route_anchor_overlap",
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="representative")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-counts", nargs="+", default=["8", "16", "32", "64"])
    parser.add_argument(
        "--anchor-strategies",
        nargs="+",
        default=[
            "random_complement",
            "farthest_complement",
            "near_route_shell",
            "mid_route_shell",
            "high_disagreement_complement",
        ],
    )
    parser.add_argument("--transform-mode", default="per_slice_orthogonal")
    parser.add_argument("--repair-mode", default="procrustes_to_original_slice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    parser.add_argument("--max-route-endpoint-fraction", type=float, default=0.65)
    parser.add_argument("--min-route-pairs", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_route_reserved_anchor_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        anchor_counts=_parse_anchor_counts(args.anchor_counts),
        anchor_strategies=[str(strategy) for strategy in args.anchor_strategies],
        transform_mode=str(args.transform_mode),
        repair_mode=str(args.repair_mode),
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
        max_route_endpoint_fraction=float(args.max_route_endpoint_fraction),
        min_route_pairs=int(args.min_route_pairs),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "best_by_config": artifact["summary"]["best_by_config"],
                    "route_reserved_summaries": artifact["summary"]["route_reserved_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
