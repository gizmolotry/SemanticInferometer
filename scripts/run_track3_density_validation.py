#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _rank_corr(left: Iterable[float], right: Iterable[float]) -> Optional[float]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _cell_dir(synthetic_root: Path, kernel: str, seed: int) -> Path:
    return Path(synthetic_root) / f"{kernel}_seed{int(seed)}"


def _read_cell(cell_dir: Path) -> Dict[str, Any]:
    csv_path = cell_dir / "MONOLITH_DATA.csv"
    if not csv_path.exists():
        return {
            "cell": cell_dir.name,
            "status": "NO_DATA",
            "failure_reasons": ["MONOLITH_DATA.csv missing"],
        }
    density: List[float] = []
    stress: List[float] = []
    work: List[float] = []
    label_rows = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            d = _to_float(row.get("density"))
            s = _to_float(row.get("stress"))
            w = _to_float(row.get("w_actual") or row.get("work_integral") or row.get("walker_mean_action"))
            label = str(row.get("perspective_tag") or row.get("label") or "").strip()
            if label:
                label_rows += 1
            if d is None or s is None or w is None:
                continue
            density.append(d)
            stress.append(s)
            work.append(w)
    density_work = _rank_corr(density, work)
    stress_work = _rank_corr(stress, work)
    aligned_density = density_work is not None and density_work <= -0.15
    aligned_stress = stress_work is not None and stress_work >= 0.15
    pass_cell = bool(aligned_density or aligned_stress)
    failures: List[str] = []
    if len(work) < 12:
        failures.append("fewer_than_12_usable_density_work_rows")
    if not pass_cell:
        failures.append("density/stress does not align with traversal-work proxy")
    return {
        "cell": cell_dir.name,
        "status": "OK" if len(work) >= 3 else "INVALID",
        "csv_path": str(csv_path),
        "row_count": sum(1 for _ in csv_path.open("r", encoding="utf-8", errors="replace")) - 1,
        "label_row_count": label_rows,
        "usable_density_work_rows": len(work),
        "density_work_rank_corr": density_work,
        "stress_work_rank_corr": stress_work,
        "density_expected_direction_pass": aligned_density,
        "stress_expected_direction_pass": aligned_stress,
        "cell_pass": pass_cell and len(work) >= 12,
        "failure_reasons": failures,
    }


def run_validation(
    *,
    synthetic_root: Path,
    output_dir: Path,
    kernels: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_read_cell(_cell_dir(synthetic_root, kernel, int(seed))) for kernel in kernels for seed in seeds]
    valid_rows = [row for row in rows if row.get("status") == "OK"]
    pass_rows = [row for row in valid_rows if bool(row.get("cell_pass"))]
    density_corrs = [row.get("density_work_rank_corr") for row in valid_rows if row.get("density_work_rank_corr") is not None]
    stress_corrs = [row.get("stress_work_rank_corr") for row in valid_rows if row.get("stress_work_rank_corr") is not None]
    pass_rate = float(len(pass_rows) / len(valid_rows)) if valid_rows else 0.0
    failures: List[str] = []
    if len(valid_rows) < 3:
        failures.append("fewer_than_three_valid_cells")
    if pass_rate < 0.67:
        failures.append("density_proxy_pass_rate_below_0_67")
    payload = {
        "schema_version": "1.0",
        "summary_type": "track3_density_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_root": str(synthetic_root),
        "kernels": list(kernels),
        "seeds": [int(seed) for seed in seeds],
        "target": "Track4 work proxy from MONOLITH_DATA.w_actual",
        "target_independence_level": "downstream_proxy_not_external_ground_truth",
        "valid_cell_count": len(valid_rows),
        "cell_count": len(rows),
        "pass_count": len(pass_rows),
        "pass_rate": pass_rate,
        "mean_density_work_rank_corr": float(np.mean(density_corrs)) if density_corrs else None,
        "mean_stress_work_rank_corr": float(np.mean(stress_corrs)) if stress_corrs else None,
        "track3_density_semantic_pass": bool(not failures),
        "failure_reasons": failures,
        "claim_boundary": {
            "safe_claim": "rho/density was tested against a traversal-work proxy",
            "unsafe_claim": "Track 3 density has independent human-semantic validity",
        },
        "cells": rows,
    }
    json_path = output_dir / "track3_density_validation.json"
    csv_path = output_dir / "track3_density_validation.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cell",
            "status",
            "row_count",
            "label_row_count",
            "usable_density_work_rows",
            "density_work_rank_corr",
            "stress_work_rank_corr",
            "cell_pass",
            "failure_reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "runs" / "experiments_20260506_192553" / "synthetic",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "track3_density_validation" / "latest",
    )
    parser.add_argument("--kernels", nargs="+", default=["rbf", "matern", "imq"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 420, 4200])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_validation(
        synthetic_root=args.synthetic_root,
        output_dir=args.output_dir,
        kernels=[str(k) for k in args.kernels],
        seeds=[int(seed) for seed in args.seeds],
    )
    print(
        json.dumps(
            {
                "track3_density_semantic_pass": payload["track3_density_semantic_pass"],
                "pass_count": payload["pass_count"],
                "valid_cell_count": payload["valid_cell_count"],
                "failure_reasons": payload["failure_reasons"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
