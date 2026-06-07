#!/usr/bin/env python3
"""Measure how many article anchors are needed to repair observer-chart gauge drift.

The gauge repair probe showed that Procrustes alignment to each observer's own
chart can restore Track 4 after per-slice translations/rotations.  This suite
asks whether that repair needs every article, or whether a sparse set of anchor
correspondences is enough.
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

from scripts.run_observer_gauge_invariance_probe import _default_configs  # noqa: E402
from scripts.run_observer_gauge_repair_probe import build_gauge_repair_suite  # noqa: E402
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _dedupe_paths,
    _payloads_for_mode,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_alignment_anchor_sparsity_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_alignment_anchor_sparsity_probe" / "latest"


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


def _anchor_label(anchor_count: Optional[int]) -> str:
    return "all" if anchor_count is None else str(int(anchor_count))


def _mean(values: Sequence[Any]) -> Optional[float]:
    finite: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            finite.append(number)
    return float(sum(finite) / len(finite)) if finite else None


def _summarize_anchor_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row.get("config_label")), str(row.get("transform_mode")), str(row.get("repair_mode")))
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        config_label, transform_mode, repair_mode = key
        sorted_rows = sorted(
            group_rows,
            key=lambda row: (10**12 if row.get("anchor_count") is None else int(row["anchor_count"])),
        )
        passing = [
            row
            for row in sorted_rows
            if row.get("anchor_count") is not None
            and bool(row.get("repair_pass"))
            and row.get("recovery_fraction") is not None
            and float(row["recovery_fraction"]) >= 0.75
        ]
        all_anchor = next((row for row in sorted_rows if row.get("anchor_count") is None), None)
        summaries.append(
            {
                "config_label": config_label,
                "transform_mode": transform_mode,
                "repair_mode": repair_mode,
                "minimum_sparse_anchor_count_for_pass": int(passing[0]["anchor_count"]) if passing else None,
                "all_anchor_recovery_fraction": None if all_anchor is None else all_anchor.get("recovery_fraction"),
                "mean_sparse_recovery_fraction": _mean(
                    [
                        row.get("recovery_fraction")
                        for row in sorted_rows
                        if row.get("anchor_count") is not None and row.get("recovery_fraction") is not None
                    ]
                ),
                "sparse_pass_rate": _mean(
                    [
                        1.0 if bool(row.get("repair_pass")) else 0.0
                        for row in sorted_rows
                        if row.get("anchor_count") is not None
                    ]
                ),
                "anchor_rows": sorted_rows,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (row["config_label"], row["transform_mode"], row["repair_mode"]),
    )


def build_anchor_sparsity_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    anchor_counts: Sequence[Optional[int]],
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
    anchor_rows: list[dict[str, Any]] = []
    sub_artifacts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for anchor_count in anchor_counts:
        subdir = output_dir / f"anchors_{_anchor_label(anchor_count)}"
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
                configs=selected_configs,
            )
        except Exception as exc:
            failures.append({"anchor_count": anchor_count, "error": str(exc)})
            continue
        sub_artifacts.append(
            {
                "anchor_count": anchor_count,
                "anchor_label": _anchor_label(anchor_count),
                "artifacts": artifact.get("artifacts"),
                "run_count": (artifact.get("summary") or {}).get("run_count"),
                "failure_count": (artifact.get("summary") or {}).get("failure_count"),
            }
        )
        failures.extend((artifact.get("failures") or []))
        for summary in (artifact.get("summary") or {}).get("config_transform_repair_summaries", []):
            if summary.get("transform_mode") == "identity" or summary.get("repair_mode") == "none":
                continue
            anchor_rows.append(
                {
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
    summaries = _summarize_anchor_rows(anchor_rows)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_alignment_anchor_sparsity_claim_matrix",
        "claim_scope": "engineering_anchor_budget_for_chart_alignment_not_semantic_truth_claim",
        "claims": [
            {
                "claim_id": f"anchor_budget_{row['config_label']}_{row['transform_mode']}_{row['repair_mode']}",
                "claim_type": "engineering_hypothesis",
                "pass": row["minimum_sparse_anchor_count_for_pass"] is not None,
                "engineering_safe": row["minimum_sparse_anchor_count_for_pass"] is not None,
                "thesis_safe": False,
                "point_estimate": row["minimum_sparse_anchor_count_for_pass"],
                "artifact_family": "observer_alignment_anchor_sparsity_probe.json",
            }
            for row in summaries
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
            "safe_claim": "chart repair can be characterized by an anchor-correspondence budget",
            "unsafe_claim": "a low anchor budget proves the observer manifold is semantically correct",
        },
        "summary": {
            "anchor_cell_count": len(anchor_rows),
            "failure_count": len(failures),
            "sub_artifacts": sub_artifacts,
            "anchor_summaries": summaries,
            "interpretation": (
                "Lower minimum sparse anchor counts indicate a cheaper, more robust chart-connection repair."
            ),
        },
        "anchor_rows": anchor_rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_alignment_anchor_sparsity_probe.json"
    csv_path = output_dir / "observer_alignment_anchor_sparsity_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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
        for row in anchor_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return payload


def _parse_anchor_counts(values: Sequence[str]) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    for raw in values:
        value = str(raw).strip().lower()
        if value in {"all", "none", "0"}:
            out.append(None)
        else:
            count = int(value)
            if count <= 0:
                out.append(None)
            else:
                out.append(count)
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="defaults")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-counts", nargs="+", default=["4", "8", "16", "32", "64", "128", "all"])
    parser.add_argument("--transform-modes", nargs="+", default=["identity", "per_slice_translation", "per_slice_orthogonal"])
    parser.add_argument("--repair-modes", nargs="+", default=["none", "center_only", "procrustes_to_original_slice"])
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
    artifact = build_anchor_sparsity_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        anchor_counts=_parse_anchor_counts(args.anchor_counts),
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
                    "anchor_cell_count": artifact["summary"]["anchor_cell_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "anchor_summaries": artifact["summary"]["anchor_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
