#!/usr/bin/env python3
"""Compile Track 4 method-sweep summaries into a feature-basis comparison.

This is intentionally small: it reads existing ``track4_grid_method_sweep``
JSON artifacts, extracts the robust-mode and peak-condition summaries, and
emits reviewer-friendly JSON/CSV tables.  It does not rerun the walker.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _load_summary(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _basis_row(summary_path: Path) -> Dict[str, Any]:
    summary = _load_summary(summary_path)
    recommendation = summary.get("method_recommendation") or {}
    robust = recommendation.get("best_robust_mode") or {}
    peak = recommendation.get("best_peak_condition") or {}
    return {
        "feature_key": summary.get("feature_key"),
        "summary_path": str(summary_path),
        "condition_count": summary.get("condition_count"),
        "compact_artifacts": summary.get("compact_artifacts"),
        "recommended_default_proposal_mode": recommendation.get("recommended_default_proposal_mode"),
        "best_peak_proposal_mode": recommendation.get("best_peak_proposal_mode"),
        "decision": recommendation.get("decision"),
        "robust_mean_score": robust.get("mean_score"),
        "robust_std_score": robust.get("std_score"),
        "robust_safe_rate": robust.get("safe_rate"),
        "robust_mean_closed_loop_rate": robust.get("mean_closed_loop_rate"),
        "peak_score": peak.get("proposal_quality_score"),
        "peak_k_neighbors": peak.get("k_neighbors"),
        "peak_seed": peak.get("seed"),
        "peak_temperature": peak.get("temperature"),
        "peak_gamma": peak.get("gamma"),
        "peak_path_shape_entropy_norm": peak.get("path_shape_entropy_norm"),
        "peak_path_edge_entropy_norm": peak.get("path_edge_entropy_norm"),
        "peak_reactive_flux_total": peak.get("reactive_flux_total"),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    columns = [
        "feature_key",
        "condition_count",
        "compact_artifacts",
        "recommended_default_proposal_mode",
        "best_peak_proposal_mode",
        "decision",
        "robust_mean_score",
        "robust_std_score",
        "robust_safe_rate",
        "robust_mean_closed_loop_rate",
        "peak_score",
        "peak_k_neighbors",
        "peak_seed",
        "peak_temperature",
        "peak_gamma",
        "peak_path_shape_entropy_norm",
        "peak_path_edge_entropy_norm",
        "peak_reactive_flux_total",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compile_basis_comparison(summary_paths: Iterable[Path], output_dir: Path) -> Dict[str, Any]:
    rows = [_basis_row(Path(path)) for path in summary_paths]
    rows = sorted(rows, key=lambda row: float(row.get("robust_mean_score") or 0.0), reverse=True)
    report = {
        "status": "OK",
        "basis_count": len(rows),
        "ranked_bases": rows,
        "interpretation": (
            "Higher robust_mean_score means the feature basis produced stronger "
            "Track 4 traversal evidence across the tested seed/k/temperature/gamma grid. "
            "This is basis sensitivity evidence, not a replacement for full corpus validation."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "track4_basis_comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "track4_basis_comparison.csv", rows)
    # Reviewer packets historically looked for paper_basis_comparison.csv.
    # Keep both names so Track 4 basis evidence cannot fall out of the ledger.
    (output_dir / "paper_basis_comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "paper_basis_comparison.csv", rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compile_basis_comparison(args.summaries, args.output_dir)
    print("Track 4 basis comparison complete")
    print(f"- output_dir: {args.output_dir}")
    for row in report["ranked_bases"]:
        print(
            "  "
            f"{row['feature_key']}: robust_mean={float(row['robust_mean_score']):.4f}, "
            f"default={row['recommended_default_proposal_mode']}, "
            f"peak={row['best_peak_proposal_mode']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
