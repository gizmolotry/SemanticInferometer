#!/usr/bin/env python3
"""Test whether sparse chart repair generalizes to held-out Track 4 paths.

Previous anchor probes evaluated repaired observer charts on the same global
path sample, which can include articles used as Procrustes anchors.  This
diagnostic filters evaluation paths so neither endpoint is an anchor, then
measures whether the repaired cross-observer action still recovers the identity
chart signal.
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
    _repair_slices,
    _select_anchor_indices,
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
DIAGNOSTIC_TYPE = "observer_alignment_holdout_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_alignment_holdout_probe" / "latest"


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


def _corpus_metric(rows: Sequence[Mapping[str, Any]], corpus: str, metric: str) -> Optional[float]:
    return _mean((row.get("transport") or {}).get(metric) for row in rows if row.get("corpus") == corpus)


def _gap(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    real = _corpus_metric(rows, "real", metric)
    controls = _mean(
        _corpus_metric(rows, corpus, metric)
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    return float(real - controls) if real is not None and controls is not None else None


def _filter_pairs(
    pairs: Sequence[tuple[int, int]],
    anchor_idx: np.ndarray,
    *,
    eval_mode: str,
) -> list[tuple[int, int]]:
    anchors = {int(idx) for idx in np.asarray(anchor_idx, dtype=np.int64).reshape(-1)}
    if eval_mode == "all_pairs":
        return [(int(i), int(j)) for i, j in pairs]
    if eval_mode == "heldout_endpoints":
        return [(int(i), int(j)) for i, j in pairs if int(i) not in anchors and int(j) not in anchors]
    if eval_mode == "anchor_to_holdout":
        return [
            (int(i), int(j))
            for i, j in pairs
            if (int(i) in anchors) != (int(j) in anchors)
        ]
    raise ValueError(f"unknown eval mode: {eval_mode}")


def _excess(summary: Mapping[str, Any]) -> Optional[float]:
    return _safe_float(summary.get("mean_calibrated_excess_holonomy_action"))


def _rel(summary: Mapping[str, Any]) -> Optional[float]:
    return _safe_float(summary.get("mean_calibrated_relative_holonomy"))


def _recovery(identity: Optional[float], damaged: Optional[float], repaired: Optional[float]) -> Optional[float]:
    if identity is None or damaged is None or repaired is None:
        return None
    denom = abs(float(damaged) - float(identity))
    if denom <= 1e-12:
        return None
    return float(1.0 - (abs(float(repaired) - float(identity)) / denom))


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    anchor_count: Optional[int],
    anchor_strategy: str,
    transform_mode: str,
    repair_mode: str,
    eval_modes: Sequence[str],
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
) -> list[dict[str, Any]]:
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
    article_pairs, pair_meta = _select_pairs_by_mode(
        reference,
        pair_mode=str(config["pair_mode"]),
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        pair_dim_cap=pair_dim_cap,
    )
    repair_seed = _stable_seed(payload_path, _config_id(config), transform_mode, repair_mode, random_seed)
    anchor_idx = _select_anchor_indices(
        slices,
        anchor_count=anchor_count,
        strategy=anchor_strategy,
        random_seed=repair_seed,
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
        random_seed=repair_seed,
    )
    rows: list[dict[str, Any]] = []
    for eval_mode in eval_modes:
        eval_pairs = _filter_pairs(article_pairs, anchor_idx, eval_mode=eval_mode)
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
        rows.append(
            {
                **identity,
                "config_label": config.get("label"),
                "config_id": _config_id(config),
                "config": {key: value for key, value in dict(config).items() if key != "label"},
                "anchor_count": None if anchor_count is None else int(anchor_count),
                "anchor_label": _anchor_label(anchor_count),
                "anchor_strategy": anchor_strategy,
                "transform_mode": transform_mode,
                "repair_mode": repair_mode,
                "eval_mode": eval_mode,
                "n_articles": int(next(iter(slices.values())).shape[0]),
                "n_slices": len(slices),
                "anchor_size": int(anchor_idx.size),
                "path_pair_count": int(len(eval_pairs)),
                "path_pair_source_count": int(len(article_pairs)),
                "projection": metadata.get("projection"),
                "normalization": metadata.get("normalization"),
                "field_sources": metadata.get("field_sources"),
                "pair_selection": pair_meta,
                "transform": transform_meta,
                "repair": repair_meta,
                "identity_transport": identity_summary,
                "damaged_transport": damaged_summary,
                "transport": repaired_summary,
                "identity_excess": identity_excess,
                "damaged_excess": damaged_excess,
                "repaired_excess": repaired_excess,
                "identity_relative": _rel(identity_summary),
                "damaged_relative": _rel(damaged_summary),
                "repaired_relative": _rel(repaired_summary),
                "recovery_fraction": _recovery(identity_excess, damaged_excess, repaired_excess),
            }
        )
    return rows


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("config_label")),
                str(row.get("anchor_label")),
                str(row.get("anchor_strategy")),
                str(row.get("transform_mode")),
                str(row.get("eval_mode")),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_label, anchor_label, anchor_strategy, transform_mode, eval_mode), group_rows in grouped.items():
        real_gap = _gap(group_rows, "mean_calibrated_excess_holonomy_action")
        identity_gap = _mean(row.get("identity_excess") for row in group_rows if row.get("corpus") == "real")
        damaged_gap = _mean(row.get("damaged_excess") for row in group_rows if row.get("corpus") == "real")
        repaired_gap = _mean(row.get("repaired_excess") for row in group_rows if row.get("corpus") == "real")
        recovery = _recovery(identity_gap, damaged_gap, repaired_gap)
        mean_pair_count = _mean(row.get("path_pair_count") for row in group_rows)
        pass_flag = bool(recovery is not None and recovery >= 0.75 and mean_pair_count is not None and mean_pair_count >= 16)
        summaries.append(
            {
                "config_label": config_label,
                "anchor_label": anchor_label,
                "anchor_count": None if anchor_label == "all" else int(anchor_label),
                "anchor_strategy": anchor_strategy,
                "transform_mode": transform_mode,
                "eval_mode": eval_mode,
                "payload_count": len(group_rows),
                "mean_path_pair_count": mean_pair_count,
                "real_minus_control_gap": real_gap,
                "real_identity_excess": identity_gap,
                "real_damaged_excess": damaged_gap,
                "real_repaired_excess": repaired_gap,
                "real_recovery_fraction": recovery,
                "mean_row_recovery_fraction": _mean(row.get("recovery_fraction") for row in group_rows),
                "pass": pass_flag,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["config_label"],
            row["eval_mode"],
            10**12 if row["anchor_count"] is None else int(row["anchor_count"]),
            row["anchor_strategy"],
        ),
    )


def build_alignment_holdout_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    anchor_counts: Sequence[Optional[int]],
    anchor_strategies: Sequence[str],
    transform_mode: str,
    repair_mode: str,
    eval_modes: Sequence[str],
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
            for anchor_strategy in anchor_strategies:
                for anchor_count in anchor_counts:
                    try:
                        rows.extend(
                            _run_one(
                                payload_path,
                                payload,
                                config,
                                anchor_count=anchor_count,
                                anchor_strategy=anchor_strategy,
                                transform_mode=transform_mode,
                                repair_mode=repair_mode,
                                eval_modes=eval_modes,
                                device=device,
                                max_pairs=max_pairs,
                                neighbor_count=neighbor_count,
                                random_seed=random_seed,
                                article_cap=article_cap,
                                pair_dim_cap=pair_dim_cap,
                            )
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "payload_path": str(payload_path),
                                "config_label": config.get("label"),
                                "anchor_strategy": anchor_strategy,
                                "anchor_count": anchor_count,
                                "error": str(exc),
                            }
                        )
    summaries = _summarize(rows)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_alignment_holdout_claim_matrix",
        "claim_scope": "engineering_out_of_sample_chart_repair_not_semantic_truth_claim",
        "claims": [
            {
                "claim_id": (
                    f"holdout_{summary['config_label']}_{summary['anchor_strategy']}_"
                    f"{summary['anchor_label']}_{summary['eval_mode']}"
                ),
                "claim_type": "engineering_hypothesis",
                "pass": bool(summary.get("pass")),
                "engineering_safe": bool(summary.get("pass")),
                "thesis_safe": False,
                "point_estimate": summary.get("real_recovery_fraction"),
                "artifact_family": "observer_alignment_holdout_probe.json",
            }
            for summary in summaries
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
            "eval_modes": list(eval_modes),
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "sparse chart repair can be evaluated on held-out non-anchor path endpoints",
            "unsafe_claim": "held-out repair proves article ideology or all path semantics",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "holdout_summaries": summaries,
            "interpretation": (
                "A pass means repaired action recovers the identity action on held-out path endpoints, not just anchors."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_alignment_holdout_probe.json"
    csv_path = output_dir / "observer_alignment_holdout_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "anchor_label",
            "anchor_strategy",
            "eval_mode",
            "mean_path_pair_count",
            "real_identity_excess",
            "real_damaged_excess",
            "real_repaired_excess",
            "real_recovery_fraction",
            "mean_row_recovery_fraction",
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
    parser.add_argument("--anchor-strategies", nargs="+", default=["random", "farthest_mean", "leverage"])
    parser.add_argument("--transform-mode", default="per_slice_orthogonal")
    parser.add_argument("--repair-mode", default="procrustes_to_original_slice")
    parser.add_argument("--eval-modes", nargs="+", default=["all_pairs", "heldout_endpoints"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=256)
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
    artifact = build_alignment_holdout_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        anchor_counts=_parse_anchor_counts(args.anchor_counts),
        anchor_strategies=[str(strategy) for strategy in args.anchor_strategies],
        transform_mode=str(args.transform_mode),
        repair_mode=str(args.repair_mode),
        eval_modes=[str(mode) for mode in args.eval_modes],
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
                    "holdout_summaries": artifact["summary"]["holdout_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
