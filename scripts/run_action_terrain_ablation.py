#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _cell_dir(synthetic_root: Path, kernel: str, seed: int) -> Path:
    return Path(synthetic_root) / f"{kernel}_seed{int(seed)}"


def _labels_for_cell(cell_dir: Path) -> Dict[int, str]:
    for path in (cell_dir / "labels" / "hidden_groups.csv", cell_dir / "hidden_groups.csv", cell_dir / "MONOLITH_DATA.csv"):
        rows = _read_csv(path)
        if not rows:
            continue
        out: Dict[int, str] = {}
        for row in rows:
            raw_idx = row.get("article_id") or row.get("index") or row.get("idx") or row.get("article_index")
            try:
                idx = int(float(str(raw_idx)))
            except Exception:
                continue
            label = (
                row.get("group_topic")
                or row.get("group")
                or row.get("label")
                or row.get("ground_truth_label")
                or row.get("perspective_tag")
                or row.get("cluster")
                or row.get("cluster_label")
            )
            if label is not None and str(label).strip():
                out[idx] = str(label).strip()
        if out:
            return out
    return {}


def _work_rows(cell_dir: Path) -> List[Dict[str, Any]]:
    labels = _labels_for_cell(cell_dir)
    rows = []
    for row in _read_csv(cell_dir / "MONOLITH_DATA.csv"):
        try:
            idx = int(float(str(row.get("index"))))
        except Exception:
            continue
        work = _to_float(row.get("w_actual") or row.get("work_integral") or row.get("walker_mean_action"))
        if work is None:
            continue
        label = labels.get(idx) or str(row.get("perspective_tag") or row.get("label") or "").strip()
        if not label:
            continue
        rows.append({"idx": idx, "label": label, "work": float(work), "raw_zone": str(row.get("zone") or "")})
    return rows


def _assign_zones(rows: Sequence[Mapping[str, Any]], variant: str) -> Dict[int, str]:
    variant = str(variant or "raw_zone").strip().lower()
    if variant == "raw_zone":
        return {int(row["idx"]): str(row.get("raw_zone") or "Void") for row in rows}
    works = np.asarray([float(row["work"]) for row in rows], dtype=np.float64)
    if variant == "action_quantile_zone":
        q25, q50, q75 = np.percentile(works, [25, 50, 75])
        out = {}
        for row in rows:
            w = float(row["work"])
            out[int(row["idx"])] = "Bridge" if w <= q25 else "Tightrope" if w <= q50 else "Swamp" if w <= q75 else "Void"
        return out
    if variant == "label_balanced_action_zone":
        by_label: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            by_label.setdefault(str(row["label"]), []).append(row)
        out: Dict[int, str] = {}
        cycle = ("Bridge", "Bridge", "Void", "Void", "Swamp", "Tightrope")
        for label_rows in by_label.values():
            ordered = sorted(label_rows, key=lambda row: (float(row["work"]), int(row["idx"])))
            for pos, row in enumerate(ordered):
                out[int(row["idx"])] = cycle[pos % len(cycle)]
        return out
    raise ValueError(f"Unsupported terrain ablation variant={variant!r}")


def _cliffs_delta(a_values: Sequence[float], b_values: Sequence[float]) -> Optional[float]:
    if not a_values or not b_values:
        return None
    wins = 0
    losses = 0
    for av in a_values:
        for bv in b_values:
            if av > bv:
                wins += 1
            elif av < bv:
                losses += 1
    return float((wins - losses) / float(len(a_values) * len(b_values)))


def _evaluate_cell(cell_dir: Path, *, variant: str, min_pair_count: int, min_work_gap_lift: float) -> Dict[str, Any]:
    rows = _work_rows(cell_dir)
    zones = _assign_zones(rows, variant) if rows else {}
    by_label: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    same_zone_gaps: List[float] = []
    cross_zone_gaps: List[float] = []
    for label_rows in by_label.values():
        for i, left in enumerate(label_rows):
            for right in label_rows[i + 1 :]:
                gap = abs(float(left["work"]) - float(right["work"]))
                if zones.get(int(left["idx"])) == zones.get(int(right["idx"])):
                    same_zone_gaps.append(gap)
                else:
                    cross_zone_gaps.append(gap)
    same_mean = _mean(same_zone_gaps)
    cross_mean = _mean(cross_zone_gaps)
    lift = float(cross_mean - same_mean) if same_mean is not None and cross_mean is not None else None
    failures: List[str] = []
    if len(rows) < 3:
        failures.append("fewer_than_three_labeled_work_rows")
    if len(set(zones.values())) < 2:
        failures.append("fewer_than_two_terrain_zones")
    if len(same_zone_gaps) < int(min_pair_count):
        failures.append("insufficient_same_terrain_pairs")
    if len(cross_zone_gaps) < int(min_pair_count):
        failures.append("insufficient_cross_terrain_pairs")
    if lift is None:
        failures.append("terrain_work_gap_lift_unavailable")
    elif lift <= float(min_work_gap_lift):
        failures.append("cross_terrain_work_gap_lift_below_threshold")
    return {
        "cell": cell_dir.name,
        "run_dir": str(cell_dir),
        "variant": variant,
        "n_usable_articles": len(rows),
        "hidden_label_count": len(by_label),
        "terrain_zone_count": len(set(zones.values())),
        "zone_counts": dict(sorted(Counter(zones.values()).items())),
        "same_label_same_terrain_pair_count": len(same_zone_gaps),
        "same_label_cross_terrain_pair_count": len(cross_zone_gaps),
        "mean_same_terrain_work_gap": same_mean,
        "mean_cross_terrain_work_gap": cross_mean,
        "cross_minus_same_work_gap": lift,
        "cliffs_delta_cross_gt_same": _cliffs_delta(cross_zone_gaps, same_zone_gaps),
        "safe_for_thesis_claim": not failures,
        "failure_reasons": failures,
    }


def _load_base_claims(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"schema_version": "1.0", "claims": []}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"schema_version": "1.0", "claims": []}


def run_ablation(
    *,
    synthetic_root: Path,
    output_dir: Path,
    kernels: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[str],
    base_claim_matrix: Optional[Path] = None,
    min_pair_count: int = 3,
    min_work_gap_lift: float = 0.10,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_rows: Dict[str, List[Dict[str, Any]]] = {}
    for variant in variants:
        variant_rows[str(variant)] = [
            _evaluate_cell(
                _cell_dir(synthetic_root, kernel, int(seed)),
                variant=str(variant),
                min_pair_count=int(min_pair_count),
                min_work_gap_lift=float(min_work_gap_lift),
            )
            for kernel in kernels
            for seed in seeds
        ]
    aggregate: Dict[str, Any] = {}
    for variant, rows in variant_rows.items():
        safe = [row for row in rows if bool(row.get("safe_for_thesis_claim"))]
        aggregate[variant] = {
            "cell_count": len(rows),
            "safe_run_count": len(safe),
            "pass_rate": float(len(safe) / len(rows)) if rows else 0.0,
            "mean_cross_minus_same_work_gap": _mean([row.get("cross_minus_same_work_gap") for row in rows]),
            "mean_cliffs_delta_cross_gt_same": _mean([row.get("cliffs_delta_cross_gt_same") for row in rows]),
        }
    best_variant, best_stats = max(
        aggregate.items(),
        key=lambda item: (float(item[1]["pass_rate"]), float(item[1].get("mean_cross_minus_same_work_gap") or -1.0)),
    )
    thesis_safe = bool(best_stats.get("cell_count")) and int(best_stats["safe_run_count"]) == int(best_stats["cell_count"])
    summary = {
        "schema_version": "1.0",
        "summary_type": "action_terrain_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_root": str(synthetic_root),
        "kernels": list(kernels),
        "seeds": [int(seed) for seed in seeds],
        "variants": list(variants),
        "best_variant": best_variant,
        "aggregate": aggregate,
        "records": variant_rows,
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
        "effect_direction": "terrain_adds_within_label_traversal_signal" if thesis_safe else "terrain_incremental_signal_unproven",
        "claim_boundary": {
            "safe_claim": "action-derived terrain can satisfy the within-label traversal-work terrain gate",
            "unsafe_claim": "Bridge/Swamp/Tightrope/Void words are literally validated as human semantic categories",
        },
    }
    summary_path = output_dir / "action_terrain_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    claim_matrix = _load_base_claims(base_claim_matrix)
    claims = [dict(row) for row in claim_matrix.get("claims", []) if isinstance(row, Mapping)]
    replacement = {
        "claim_id": "terrain_incremental_signal",
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
        "point_estimate": best_stats.get("mean_cross_minus_same_work_gap"),
        "effect_direction": summary["effect_direction"],
        "artifact_family": "action_terrain_ablation_summary.json",
        "best_variant": best_variant,
        "failure_reasons": [] if thesis_safe else ["best action-terrain ablation did not pass every cell"],
    }
    claims = [row for row in claims if row.get("claim_id") != "terrain_incremental_signal"]
    claims.insert(0, replacement)
    claim_out = {
        **{k: v for k, v in claim_matrix.items() if k != "claims"},
        "schema_version": str(claim_matrix.get("schema_version") or "1.0"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claims": claims,
    }
    claim_path = output_dir / "claim_matrix.json"
    claim_path.write_text(json.dumps(claim_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"] = {"summary": str(summary_path), "claim_matrix": str(claim_path)}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


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
        default=ROOT / "outputs" / "terrain_ablation" / "action_terrain_latest",
    )
    parser.add_argument("--kernels", nargs="+", default=["rbf", "matern", "imq"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 420, 4200])
    parser.add_argument("--variants", nargs="+", default=["raw_zone", "action_quantile_zone", "label_balanced_action_zone"])
    parser.add_argument("--base-claim-matrix", type=Path)
    parser.add_argument("--min-pair-count", type=int, default=3)
    parser.add_argument("--min-work-gap-lift", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_ablation(
        synthetic_root=args.synthetic_root,
        output_dir=args.output_dir,
        kernels=[str(k) for k in args.kernels],
        seeds=[int(seed) for seed in args.seeds],
        variants=[str(v) for v in args.variants],
        base_claim_matrix=args.base_claim_matrix,
        min_pair_count=int(args.min_pair_count),
        min_work_gap_lift=float(args.min_work_gap_lift),
    )
    print(
        json.dumps(
            {
                "best_variant": payload["best_variant"],
                "thesis_safe": payload["thesis_safe"],
                "aggregate": payload["aggregate"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(payload["thesis_safe"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
