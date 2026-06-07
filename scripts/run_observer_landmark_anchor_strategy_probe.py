#!/usr/bin/env python3
"""Compare landmark anchor strategies for observer-chart alignment repair.

Anchor sparsity told us how many random correspondences were needed.  This
suite asks whether better anchor selection can lower that budget, especially
for per-slice rotations where Track 4 is most chart-sensitive.
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
from typing import Any, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_observer_alignment_anchor_sparsity_probe import (  # noqa: E402
    _anchor_label,
    _json_safe,
    _mean,
    _parse_anchor_counts,
)
from scripts.run_observer_gauge_invariance_probe import _default_configs  # noqa: E402
from scripts.run_observer_gauge_repair_probe import build_gauge_repair_suite  # noqa: E402
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _dedupe_paths,
    _payloads_for_mode,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_landmark_anchor_strategy_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_landmark_anchor_strategy_probe" / "latest"


def _strategy_slug(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("config_label")),
                str(row.get("transform_mode")),
                str(row.get("repair_mode")),
                str(row.get("anchor_strategy")),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (config_label, transform_mode, repair_mode, strategy), group_rows in grouped.items():
        sparse = [row for row in group_rows if row.get("anchor_count") is not None]
        passing = [
            row
            for row in sparse
            if bool(row.get("repair_pass"))
            and row.get("recovery_fraction") is not None
            and float(row["recovery_fraction"]) >= 0.75
        ]
        passing = sorted(passing, key=lambda row: int(row["anchor_count"]))
        best = max(
            sparse,
            key=lambda row: float(row.get("recovery_fraction") or -1e9),
            default=None,
        )
        summaries.append(
            {
                "config_label": config_label,
                "transform_mode": transform_mode,
                "repair_mode": repair_mode,
                "anchor_strategy": strategy,
                "minimum_anchor_count_for_pass": int(passing[0]["anchor_count"]) if passing else None,
                "mean_recovery_fraction": _mean(
                    [row.get("recovery_fraction") for row in sparse if row.get("recovery_fraction") is not None]
                ),
                "pass_rate": _mean([1.0 if bool(row.get("repair_pass")) else 0.0 for row in sparse]),
                "best_anchor_count": None if best is None else best.get("anchor_count"),
                "best_recovery_fraction": None if best is None else best.get("recovery_fraction"),
                "anchor_rows": sorted(
                    group_rows,
                    key=lambda row: (
                        10**12 if row.get("anchor_count") is None else int(row["anchor_count"]),
                        str(row.get("anchor_strategy")),
                    ),
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            str(row["config_label"]),
            str(row["transform_mode"]),
            str(row["repair_mode"]),
            10**12 if row["minimum_anchor_count_for_pass"] is None else int(row["minimum_anchor_count_for_pass"]),
            -1e9 if row["mean_recovery_fraction"] is None else -float(row["mean_recovery_fraction"]),
            str(row["anchor_strategy"]),
        ),
    )


def _best_by_cell(strategy_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in strategy_summaries:
        grouped[(str(row["config_label"]), str(row["transform_mode"]), str(row["repair_mode"]))].append(row)
    out: list[dict[str, Any]] = []
    for (config_label, transform_mode, repair_mode), rows in grouped.items():
        best = sorted(
            rows,
            key=lambda row: (
                10**12 if row["minimum_anchor_count_for_pass"] is None else int(row["minimum_anchor_count_for_pass"]),
                -1e9 if row["mean_recovery_fraction"] is None else -float(row["mean_recovery_fraction"]),
            ),
        )[0]
        random_row = next((row for row in rows if row.get("anchor_strategy") == "random"), None)
        out.append(
            {
                "config_label": config_label,
                "transform_mode": transform_mode,
                "repair_mode": repair_mode,
                "best_strategy": best.get("anchor_strategy"),
                "best_minimum_anchor_count_for_pass": best.get("minimum_anchor_count_for_pass"),
                "best_mean_recovery_fraction": best.get("mean_recovery_fraction"),
                "random_minimum_anchor_count_for_pass": None
                if random_row is None
                else random_row.get("minimum_anchor_count_for_pass"),
                "random_mean_recovery_fraction": None if random_row is None else random_row.get("mean_recovery_fraction"),
            }
        )
    return sorted(out, key=lambda row: (row["config_label"], row["transform_mode"], row["repair_mode"]))


def build_landmark_anchor_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    anchor_counts: Sequence[Optional[int]],
    anchor_strategies: Sequence[str],
    transform_modes: Sequence[str],
    repair_modes: Sequence[str],
    device_name: str,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    configs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_configs = [dict(config) for config in (configs or _default_configs())]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sub_artifacts: list[dict[str, Any]] = []
    for strategy in anchor_strategies:
        for anchor_count in anchor_counts:
            subdir = output_dir / f"strategy_{_strategy_slug(strategy)}" / f"anchors_{_anchor_label(anchor_count)}"
            try:
                artifact = build_gauge_repair_suite(
                    payload_paths=payload_paths,
                    output_dir=subdir,
                    transform_modes=transform_modes,
                    repair_modes=repair_modes,
                    device_name=device_name,
                    max_pairs=max_pairs,
                    neighbor_count=neighbor_count,
                    random_seed=random_seed,
                    article_cap=article_cap,
                    pair_dim_cap=pair_dim_cap,
                    anchor_count=anchor_count,
                    anchor_strategy=strategy,
                    configs=selected_configs,
                )
            except Exception as exc:
                failures.append({"anchor_strategy": strategy, "anchor_count": anchor_count, "error": str(exc)})
                continue
            sub_artifacts.append(
                {
                    "anchor_strategy": strategy,
                    "anchor_count": anchor_count,
                    "anchor_label": _anchor_label(anchor_count),
                    "artifacts": artifact.get("artifacts"),
                    "run_count": (artifact.get("summary") or {}).get("run_count"),
                    "failure_count": (artifact.get("summary") or {}).get("failure_count"),
                }
            )
            failures.extend(artifact.get("failures") or [])
            for summary in (artifact.get("summary") or {}).get("config_transform_repair_summaries", []):
                if summary.get("transform_mode") == "identity" or summary.get("repair_mode") == "none":
                    continue
                rows.append(
                    {
                        "anchor_strategy": strategy,
                        "anchor_count": anchor_count,
                        "anchor_label": _anchor_label(anchor_count),
                        "config_label": summary.get("config_label"),
                        "config_id": summary.get("config_id"),
                        "transform_mode": summary.get("transform_mode"),
                        "repair_mode": summary.get("repair_mode"),
                        "identity_gap": summary.get("identity_gap"),
                        "damaged_gap": summary.get("damaged_gap"),
                        "real_minus_control_gap": summary.get("real_minus_control_gap"),
                        "gap_error_vs_identity": summary.get("gap_error_vs_identity"),
                        "recovery_fraction": summary.get("recovery_fraction"),
                        "repair_pass": summary.get("repair_pass"),
                        "mean_anchor_residual": summary.get("mean_anchor_residual"),
                    }
                )
    strategy_summaries = _summarize(rows)
    best_rows = _best_by_cell(strategy_summaries)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_landmark_anchor_strategy_claim_matrix",
        "claim_scope": "engineering_landmark_selection_for_chart_alignment_not_semantic_truth_claim",
        "claims": [
            {
                "claim_id": f"landmark_{row['config_label']}_{row['transform_mode']}_{row['repair_mode']}",
                "claim_type": "engineering_hypothesis",
                "pass": row["best_minimum_anchor_count_for_pass"] is not None,
                "engineering_safe": row["best_minimum_anchor_count_for_pass"] is not None,
                "thesis_safe": False,
                "point_estimate": row["best_minimum_anchor_count_for_pass"],
                "best_strategy": row["best_strategy"],
                "artifact_family": "observer_landmark_anchor_strategy_probe.json",
            }
            for row in best_rows
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at_utc": claim_matrix["generated_at_utc"],
        "output_dir": str(output_dir),
        "config": {
            "payload_count": len(payload_paths),
            "anchor_counts": [_anchor_label(count) for count in anchor_counts],
            "anchor_strategies": list(anchor_strategies),
            "transform_modes": list(transform_modes),
            "repair_modes": list(repair_modes),
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "landmark choice changes chart-alignment recovery at fixed anchor count",
            "unsafe_claim": "best landmark strategy proves article ideology or validates all semantic interpretation",
        },
        "summary": {
            "strategy_cell_count": len(rows),
            "failure_count": len(failures),
            "sub_artifacts": sub_artifacts,
            "strategy_summaries": strategy_summaries,
            "best_by_cell": best_rows,
            "interpretation": (
                "If a landmark strategy beats random at the same anchor count, chart repair can be made cheaper."
            ),
        },
        "strategy_rows": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_landmark_anchor_strategy_probe.json"
    csv_path = output_dir / "observer_landmark_anchor_strategy_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "anchor_strategy",
            "anchor_count",
            "anchor_label",
            "config_label",
            "transform_mode",
            "repair_mode",
            "identity_gap",
            "damaged_gap",
            "real_minus_control_gap",
            "gap_error_vs_identity",
            "recovery_fraction",
            "repair_pass",
            "mean_anchor_residual",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
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
            "random",
            "farthest_mean",
            "high_observer_disagreement",
            "leverage",
            "hybrid_disagreement_farthest",
        ],
    )
    parser.add_argument("--transform-modes", nargs="+", default=["identity", "per_slice_orthogonal"])
    parser.add_argument("--repair-modes", nargs="+", default=["none", "procrustes_to_original_slice"])
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
    artifact = build_landmark_anchor_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        anchor_counts=_parse_anchor_counts(args.anchor_counts),
        anchor_strategies=[str(strategy) for strategy in args.anchor_strategies],
        transform_modes=[str(mode) for mode in args.transform_modes],
        repair_modes=[str(mode) for mode in args.repair_modes],
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
                    "strategy_cell_count": artifact["summary"]["strategy_cell_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "best_by_cell": artifact["summary"]["best_by_cell"],
                    "strategy_summaries": artifact["summary"]["strategy_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
