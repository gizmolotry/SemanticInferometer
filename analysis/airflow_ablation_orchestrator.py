"""
Airflow orchestrator for manifold gradient ablation (2x2 matrix).

This module is intentionally isolated:
- No changes to rendering code paths.
- No mutation of existing MONOLITH_DATA.csv.
- Writes ablation artifacts to a separate output directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from airflow.decorators import dag, task  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dag = None
    task = None


STRESS_MODES: Tuple[str, str] = ("raw_l2", "normalized_l2")
ZONE_RULES: Tuple[str, str] = ("fixed_0_5_threshold", "median_threshold")


@dataclass(frozen=True)
class AblationConfig:
    base_csv: Path
    output_dir: Path
    scalar_bins: int = 8


def _ensure_columns(df: pd.DataFrame) -> None:
    required = {"density", "stress"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in MONOLITH_DATA.csv: {missing}")


def _stress_view(stress: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw_l2":
        return stress.astype(float)
    if mode == "normalized_l2":
        lo = float(np.nanmin(stress))
        hi = float(np.nanmax(stress))
        if hi - lo <= 1e-12:
            return np.full_like(stress, 0.5, dtype=float)
        return (stress - lo) / (hi - lo)
    raise ValueError(f"Unknown stress mode: {mode}")


def _zone_counts(density: np.ndarray, stress_view: np.ndarray, rule: str) -> Dict[str, int]:
    if rule == "fixed_0_5_threshold":
        d_thr = 0.5
        s_thr = 0.5
    elif rule == "median_threshold":
        d_thr = float(np.nanmedian(density))
        s_thr = float(np.nanmedian(stress_view))
    else:
        raise ValueError(f"Unknown zone rule: {rule}")

    bridge = int(((density >= d_thr) & (stress_view < s_thr)).sum())
    swamp = int(((density >= d_thr) & (stress_view >= s_thr)).sum())
    tightrope = int(((density < d_thr) & (stress_view < s_thr)).sum())
    void = int(((density < d_thr) & (stress_view >= s_thr)).sum())
    return {
        "Bridge": bridge,
        "Swamp": swamp,
        "Tightrope": tightrope,
        "Void": void,
    }


def _scalar_metrics(density: np.ndarray, stress_view: np.ndarray, bins: int) -> Dict[str, float]:
    scalar = 0.25 - 0.25 * density + 0.75 * stress_view
    scalar_min = float(np.nanmin(scalar))
    scalar_max = float(np.nanmax(scalar))
    # Coverage is measured on [0,1] expected manifold range.
    scalar_clip = np.clip(scalar, 0.0, 1.0)
    hist, _ = np.histogram(scalar_clip, bins=bins, range=(0.0, 1.0))
    coverage = float((hist > 0).sum()) / float(bins)
    return {
        "scalar_min": scalar_min,
        "scalar_max": scalar_max,
        "scalar_bin_coverage": coverage,
    }


def compute_ablation_cell(config: AblationConfig, stress_mode: str, zone_rule: str) -> Dict[str, object]:
    df = pd.read_csv(config.base_csv)
    _ensure_columns(df)

    density = np.asarray(df["density"], dtype=float)
    stress = np.asarray(df["stress"], dtype=float)
    stress_view = _stress_view(stress, stress_mode)

    out: Dict[str, object] = {
        "stress_mode": stress_mode,
        "zone_rule": zone_rule,
        "n_rows": int(len(df)),
        "density_min": float(np.nanmin(density)),
        "density_max": float(np.nanmax(density)),
        "stress_min": float(np.nanmin(stress)),
        "stress_max": float(np.nanmax(stress)),
        "stress_view_min": float(np.nanmin(stress_view)),
        "stress_view_max": float(np.nanmax(stress_view)),
    }
    out.update(_scalar_metrics(density, stress_view, config.scalar_bins))
    out["zone_counts"] = _zone_counts(density, stress_view, zone_rule)
    out["missing_zone_count"] = int(sum(1 for c in out["zone_counts"].values() if c == 0))
    return out


def run_ablation_matrix(base_csv: Path, output_dir: Path, scalar_bins: int = 8) -> Path:
    config = AblationConfig(base_csv=Path(base_csv), output_dir=Path(output_dir), scalar_bins=scalar_bins)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for stress_mode in STRESS_MODES:
        for zone_rule in ZONE_RULES:
            row = compute_ablation_cell(config, stress_mode=stress_mode, zone_rule=zone_rule)
            rows.append(row)
            slug = f"{stress_mode}__{zone_rule}".replace("/", "_")
            (config.output_dir / f"ablation_cell_{slug}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )

    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "base_csv": str(config.base_csv),
        "cells": rows,
    }
    summary_json = config.output_dir / "manifold_ablation_summary.json"
    summary_csv = config.output_dir / "manifold_ablation_summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flat_rows: List[Dict[str, object]] = []
    for r in rows:
        zone_counts = r.get("zone_counts", {})
        flat_rows.append(
            {
                "stress_mode": r["stress_mode"],
                "zone_rule": r["zone_rule"],
                "n_rows": r["n_rows"],
                "scalar_min": r["scalar_min"],
                "scalar_max": r["scalar_max"],
                "scalar_bin_coverage": r["scalar_bin_coverage"],
                "missing_zone_count": r["missing_zone_count"],
                "Bridge": zone_counts.get("Bridge", 0),
                "Swamp": zone_counts.get("Swamp", 0),
                "Tightrope": zone_counts.get("Tightrope", 0),
                "Void": zone_counts.get("Void", 0),
            }
        )
    pd.DataFrame(flat_rows).to_csv(summary_csv, index=False)
    return summary_json


if dag is not None and task is not None:  # pragma: no cover - exercised in Airflow
    @dag(
        dag_id="manifold_gradient_ablation_v1",
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["ablation", "manifold", "monolith"],
    )
    def manifold_gradient_ablation_v1():
        @task
        def run_matrix(base_csv: str, output_dir: str, scalar_bins: int = 8) -> str:
            out = run_ablation_matrix(Path(base_csv), Path(output_dir), scalar_bins=scalar_bins)
            return str(out)

        run_matrix(
            base_csv="{{ dag_run.conf.get('base_csv', 'outputs/honest_matern/MONOLITH_DATA.csv') }}",
            output_dir="{{ dag_run.conf.get('output_dir', 'analysis/ablation_outputs') }}",
            scalar_bins=8,
        )

    manifold_gradient_ablation_v1_dag = manifold_gradient_ablation_v1()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifold ablation matrix without Airflow.")
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=Path("outputs/honest_matern/MONOLITH_DATA.csv"),
        help="Path to MONOLITH_DATA.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ablation_outputs"),
        help="Directory for ablation outputs",
    )
    parser.add_argument("--scalar-bins", type=int, default=8, help="Histogram bins for [0,1] coverage metric")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = run_ablation_matrix(base_csv=args.base_csv, output_dir=args.output_dir, scalar_bins=args.scalar_bins)
    print(f"[ablation] summary written to: {out}")
