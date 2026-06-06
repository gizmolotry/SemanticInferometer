#!/usr/bin/env python3
"""Measure whether observer-slice transport is dominated by a few observer pairs.

This is a creative Track 4 stress test: decompose the transport effect by
ordered observer-pair, then recompute the real/control gap after removing the
largest pair contributions.  If the signal collapses after dropping one pair,
the engineering fix is observer-pair calibration/regularization.  If it
survives, the effect is atlas-wide rather than a single-slice artifact.
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
    _safe_float,
    _semantic_action_vector,
)
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _config_id,
    _dedupe_paths,
    _null_slices,
    _payloads_for_mode,
    _prepare_slices,
    _select_pairs_by_mode,
    _stable_seed,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_pair_dominance_ablation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_pair_dominance_ablation" / "latest"


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


def _weighted_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    num = 0.0
    den = 0
    for row in rows:
        value = _safe_float(row.get(metric))
        count = int(row.get("record_count") or 0)
        if value is None or count <= 0:
            continue
        num += float(value) * count
        den += count
    return float(num / den) if den else None


def _rate(values: Iterable[Any]) -> float:
    vals = [bool(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _action_config(config: Mapping[str, Any]) -> ObserverSliceTransportConfig:
    return ObserverSliceTransportConfig(
        semantic_weight=float(config["semantic_weight"]),
        observer_switch_weight=float(config["observer_switch_weight"]),
        stress_weight=float(config["stress_weight"]),
        density_weight=float(config["density_weight"]),
    )


def _default_configs() -> list[dict[str, Any]]:
    return [
        {
            "label": "current_baseline",
            "feature_source": "rks",
            "projection_dim_cap": 512,
            "normalization": "none",
            "pair_mode": "mixed",
            "null_mode": "zero",
            "weight_profile": "default_action",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.0,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "rupture_raw_cls_farthest",
            "feature_source": "raw_cls",
            "projection_dim_cap": 512,
            "normalization": "zscore_per_slice",
            "pair_mode": "farthest",
            "null_mode": "independent_article_shuffle",
            "weight_profile": "switch_heavy",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.5,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "local_rks_nearest",
            "feature_source": "rks",
            "projection_dim_cap": 128,
            "normalization": "zscore_per_slice",
            "pair_mode": "nearest",
            "null_mode": "dimension_signflip_by_slice",
            "weight_profile": "switch_heavy",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.5,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "local_rks_mixed",
            "feature_source": "rks",
            "projection_dim_cap": 256,
            "normalization": "zscore_per_slice",
            "pair_mode": "mixed",
            "null_mode": "dimension_signflip_by_slice",
            "weight_profile": "stress_forward",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.0,
            "stress_weight": 0.75,
            "density_weight": 0.15,
        },
    ]


def _pair_rows(
    slices: Mapping[str, np.ndarray],
    *,
    article_pairs: Sequence[tuple[int, int]],
    density: np.ndarray,
    stress: np.ndarray,
    config: ObserverSliceTransportConfig,
) -> list[dict[str, Any]]:
    names = list(slices)
    source_idx = np.asarray([pair[0] for pair in article_pairs], dtype=np.int64)
    target_idx = np.asarray([pair[1] for pair in article_pairs], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    if source_idx.size == 0:
        return rows
    for source_name in names:
        src_chart = slices[source_name]
        for target_name in names:
            if source_name == target_name:
                continue
            tgt_chart = slices[target_name]
            sem_first_sem, _ = _semantic_action_vector(
                src_chart,
                source_idx,
                target_idx,
                density=density,
                stress=stress,
                config=config,
            )
            sem_first_switch, _ = _observer_switch_vector(
                src_chart,
                tgt_chart,
                target_idx,
                config=config,
            )
            obs_first_switch, _ = _observer_switch_vector(
                src_chart,
                tgt_chart,
                source_idx,
                config=config,
            )
            obs_first_sem, _ = _semantic_action_vector(
                tgt_chart,
                source_idx,
                target_idx,
                density=density,
                stress=stress,
                config=config,
            )
            semantic_first = sem_first_sem + sem_first_switch
            observer_first = obs_first_switch + obs_first_sem
            holonomy = np.abs(semantic_first - observer_first)
            route_min = np.maximum(np.minimum(semantic_first, observer_first), 1e-12)
            relative = holonomy / route_min
            rows.append(
                {
                    "source_slice": source_name,
                    "target_slice": target_name,
                    "observer_pair": f"{source_name}->{target_name}",
                    "record_count": int(holonomy.size),
                    "mean_holonomy_action": float(np.mean(holonomy)) if holonomy.size else None,
                    "mean_relative_holonomy": float(np.mean(relative)) if relative.size else None,
                    "max_holonomy_action": float(np.max(holonomy)) if holonomy.size else None,
                    "positive_rate": float(np.mean(holonomy > 1e-9)) if holonomy.size else 0.0,
                }
            )
    return rows


def _calibrate_pair_rows(
    real_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]] | None,
    *,
    null_mode: str,
) -> list[dict[str, Any]]:
    null_by_pair = {
        str(row.get("observer_pair")): row
        for row in (null_rows or [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in real_rows:
        null = null_by_pair.get(str(row.get("observer_pair")), {})
        raw = _safe_float(row.get("mean_holonomy_action")) or 0.0
        raw_relative = _safe_float(row.get("mean_relative_holonomy")) or 0.0
        null_mean = _safe_float(null.get("mean_holonomy_action")) if null else 0.0
        null_relative = _safe_float(null.get("mean_relative_holonomy")) if null else 0.0
        out.append(
            {
                **dict(row),
                "null_mode": null_mode,
                "mean_null_holonomy_action": null_mean,
                "mean_null_relative_holonomy": null_relative,
                "mean_calibrated_excess_holonomy_action": float(raw - (null_mean or 0.0)),
                "mean_calibrated_relative_holonomy": float(raw_relative - (null_relative or 0.0)),
            }
        )
    return out


def _dominance_summary(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        [dict(row) for row in pair_rows],
        key=lambda row: float(row.get("mean_calibrated_excess_holonomy_action") or -1e12),
        reverse=True,
    )
    positive = [
        max(0.0, float(row.get("mean_calibrated_excess_holonomy_action") or 0.0))
        * int(row.get("record_count") or 0)
        for row in ordered
    ]
    total_pos = float(sum(positive))
    top1 = ordered[0] if ordered else {}
    top2 = ordered[:2]
    top3 = ordered[:3]
    top1_share = float(positive[0] / total_pos) if total_pos > 1e-12 and positive else None
    top3_share = float(sum(positive[:3]) / total_pos) if total_pos > 1e-12 and positive else None
    mean_full = _weighted_mean(ordered, "mean_calibrated_excess_holonomy_action")
    rel_full = _weighted_mean(ordered, "mean_calibrated_relative_holonomy")
    mean_drop_top1 = _weighted_mean(ordered[1:], "mean_calibrated_excess_holonomy_action")
    mean_drop_top2 = _weighted_mean(ordered[2:], "mean_calibrated_excess_holonomy_action")
    mean_drop_top3 = _weighted_mean(ordered[3:], "mean_calibrated_excess_holonomy_action")
    return {
        "observer_pair_count": len(ordered),
        "full_mean_calibrated_excess": mean_full,
        "full_mean_calibrated_relative": rel_full,
        "drop_top1_mean_calibrated_excess": mean_drop_top1,
        "drop_top2_mean_calibrated_excess": mean_drop_top2,
        "drop_top3_mean_calibrated_excess": mean_drop_top3,
        "top1_pair_share_of_positive_excess": top1_share,
        "top3_pair_share_of_positive_excess": top3_share,
        "top_pair": {
            "observer_pair": top1.get("observer_pair"),
            "mean_calibrated_excess_holonomy_action": top1.get("mean_calibrated_excess_holonomy_action"),
            "mean_calibrated_relative_holonomy": top1.get("mean_calibrated_relative_holonomy"),
        },
        "top_pairs": [
            {
                "observer_pair": row.get("observer_pair"),
                "mean_calibrated_excess_holonomy_action": row.get("mean_calibrated_excess_holonomy_action"),
                "mean_calibrated_relative_holonomy": row.get("mean_calibrated_relative_holonomy"),
            }
            for row in top3
        ],
    }


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
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
    pairs, pair_meta = _select_pairs_by_mode(
        reference,
        pair_mode=str(config["pair_mode"]),
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        pair_dim_cap=pair_dim_cap,
    )
    action_config = _action_config(config)
    real_pair_rows = _pair_rows(
        slices,
        article_pairs=pairs,
        density=density,
        stress=stress,
        config=action_config,
    )
    null_seed = _stable_seed(payload_path, _config_id(config), "pair_dominance", random_seed)
    null_chart = _null_slices(slices, mode=str(config["null_mode"]), random_seed=null_seed)
    null_rows = None
    if null_chart is not None:
        null_rows = _pair_rows(
            null_chart,
            article_pairs=pairs,
            density=density,
            stress=stress,
            config=action_config,
        )
    calibrated = _calibrate_pair_rows(real_pair_rows, null_rows, null_mode=str(config["null_mode"]))
    dominance = _dominance_summary(calibrated)
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
        "pair_selection": pair_meta,
        "dominance": dominance,
        "pair_rows": calibrated,
    }


def _corpus_gap(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    by_corpus: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _safe_float((row.get("dominance") or {}).get(metric))
        if value is None:
            continue
        by_corpus[str(row.get("corpus") or "unknown")].append(value)
    real = _mean(by_corpus.get("real", []))
    controls = _mean(
        _mean(by_corpus.get(corpus, []))
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    return float(real - controls) if real is not None and controls is not None else None


def _config_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    full_gap = _corpus_gap(rows, "full_mean_calibrated_excess")
    drop1_gap = _corpus_gap(rows, "drop_top1_mean_calibrated_excess")
    drop2_gap = _corpus_gap(rows, "drop_top2_mean_calibrated_excess")
    drop3_gap = _corpus_gap(rows, "drop_top3_mean_calibrated_excess")
    top1_share = _mean((row.get("dominance") or {}).get("top1_pair_share_of_positive_excess") for row in rows)
    top3_share = _mean((row.get("dominance") or {}).get("top3_pair_share_of_positive_excess") for row in rows)
    collapse_ratio = None
    if full_gap is not None and drop1_gap is not None and abs(full_gap) > 1e-12:
        collapse_ratio = float(drop1_gap / full_gap)
    robust_after_top1 = bool(drop1_gap is not None and full_gap is not None and drop1_gap > 0.0 and drop1_gap >= 0.5 * full_gap)
    dominance_risk = bool(top1_share is not None and top1_share > 0.25)
    return {
        "config_id": first.get("config_id"),
        "config_label": first.get("config_label"),
        "config": first.get("config"),
        "payload_count": len(rows),
        "real_minus_control_full_gap": full_gap,
        "real_minus_control_drop_top1_gap": drop1_gap,
        "real_minus_control_drop_top2_gap": drop2_gap,
        "real_minus_control_drop_top3_gap": drop3_gap,
        "drop_top1_gap_ratio": collapse_ratio,
        "mean_top1_pair_share_of_positive_excess": top1_share,
        "mean_top3_pair_share_of_positive_excess": top3_share,
        "robust_after_top1_removal": robust_after_top1,
        "dominance_risk": dominance_risk,
        "effect_pass_rate": _rate((row.get("dominance") or {}).get("full_mean_calibrated_excess", 0.0) > 0 for row in rows),
    }


def build_pair_dominance_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    device_name: str,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
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
                        max_pairs=max_pairs,
                        neighbor_count=neighbor_count,
                        random_seed=random_seed,
                        article_cap=article_cap,
                        pair_dim_cap=pair_dim_cap,
                    )
                )
            except Exception as exc:
                failures.append({"payload_path": str(payload_path), "config_label": config.get("label"), "error": str(exc)})
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("config_label") or row.get("config_id"))].append(row)
    config_summaries = sorted(
        [_config_summary(config_rows) for config_rows in grouped.values()],
        key=lambda row: float(row.get("real_minus_control_drop_top1_gap") or -1e12),
        reverse=True,
    )
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_pair_dominance_claim_matrix",
        "claim_scope": "engineering_dominance_diagnostic_not_thesis_truth_claim",
        "claims": [
            {
                "claim_id": f"pair_dominance_{summary.get('config_label')}",
                "claim_type": "engineering_hypothesis",
                "pass": bool(summary.get("robust_after_top1_removal")),
                "engineering_safe": bool(summary.get("robust_after_top1_removal")),
                "thesis_safe": False,
                "dominance_risk": bool(summary.get("dominance_risk")),
                "point_estimate": summary.get("real_minus_control_drop_top1_gap"),
                "artifact_family": "observer_pair_dominance_ablation.json",
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
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "observer-pair dominance and robustness of transport gaps after top-pair removal",
            "unsafe_claim": "observer pair names correspond to human-interpretable ideological agents without independent validation",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "config_summaries": config_summaries,
            "dominance_warning": (
                "High top-pair share means a config should add observer-pair calibration even if the gap is large."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_pair_dominance_ablation.json"
    csv_path = output_dir / "observer_pair_dominance_ablation.csv"
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
                "full_mean_calibrated_excess",
                "drop_top1_mean_calibrated_excess",
                "top1_pair_share_of_positive_excess",
                "top_pair",
                "payload_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            dominance = row.get("dominance") or {}
            writer.writerow(
                {
                    "config_label": row.get("config_label"),
                    "corpus": row.get("corpus"),
                    "kernel": row.get("kernel"),
                    "cell_id": row.get("cell_id"),
                    "full_mean_calibrated_excess": dominance.get("full_mean_calibrated_excess"),
                    "drop_top1_mean_calibrated_excess": dominance.get("drop_top1_mean_calibrated_excess"),
                    "top1_pair_share_of_positive_excess": dominance.get("top1_pair_share_of_positive_excess"),
                    "top_pair": (dominance.get("top_pair") or {}).get("observer_pair"),
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
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_pair_dominance_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
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
