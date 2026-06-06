"""Diagnose what Track 4 terrain semantics currently measure.

This harness consumes existing Track 4 validation/probe artifacts. It does not
rerun DeBERTa or the full pipeline. The intent is to split the failed terrain
claim into smaller falsifiable hypotheses:

1. Terrain construct validity.
2. Walker sensitivity.
3. Work decomposition / proxy decomposition.
4. Real-vs-control specificity.
5. Alternative terrain contrasts.
6. Targeted event-pair evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.verification.scientific_summaries import summarize_track4_traversal  # noqa: E402


DEFAULT_VALIDATION_SUMMARY = (
    ROOT
    / "outputs"
    / "track4_focused_basis_validation"
    / "seed42_kernel_slice_20260519_033359"
    / "track4_focused_basis_validation_summary.json"
)
DEFAULT_METHOD_SWEEP = (
    ROOT
    / "outputs"
    / "microprobes"
    / "property_theft"
    / "track4_entropy_compact_grid_20260515_093357"
    / "track4_grid_method_sweep_summary.json"
)

TERRAIN_ZONES = ("Bridge", "Swamp", "Tightrope", "Void")
CONTRAST_PAIRS = (
    ("Bridge", "Void"),
    ("Bridge", "Tightrope"),
    ("Bridge", "Swamp"),
    ("Swamp", "Void"),
    ("Swamp", "Tightrope"),
    ("Tightrope", "Void"),
)


def _safe_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _safe_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(v) for v in (_safe_float(value) for value in values) if v is not None]
    return float(np.mean(finite)) if finite else None


def _std(values: Iterable[Any]) -> float | None:
    finite = [float(v) for v in (_safe_float(value) for value in values) if v is not None]
    return float(np.std(finite)) if len(finite) >= 2 else None


def _pearson(x_values: Iterable[Any], y_values: Iterable[Any]) -> float | None:
    pairs: List[Tuple[float, float]] = []
    for x_raw, y_raw in zip(x_values, y_values):
        x = _safe_float(x_raw)
        y = _safe_float(y_raw)
        if x is not None and y is not None:
            pairs.append((float(x), float(y)))
    if len(pairs) < 3:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _cliffs_delta(a_values: Iterable[Any], b_values: Iterable[Any]) -> float | None:
    a = [float(v) for v in (_safe_float(value) for value in a_values) if v is not None]
    b = [float(v) for v in (_safe_float(value) for value in b_values) if v is not None]
    if not a or not b:
        return None
    wins = 0
    losses = 0
    for av in a:
        for bv in b:
            if av > bv:
                wins += 1
            elif av < bv:
                losses += 1
    return float((wins - losses) / float(len(a) * len(b)))


def _rate(flags: Iterable[Any]) -> float:
    values = [bool(flag) for flag in flags]
    return float(sum(1 for flag in values if flag) / len(values)) if values else 0.0


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1", "same_event"}:
            return True
        if text in {"false", "f", "no", "n", "0", "different_event"}:
            return False
    return None


def load_validation_payload(path: Path) -> Dict[str, Any]:
    payload = _safe_json(Path(path))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Validation summary does not contain a list-valued 'rows': {path}")
    return payload


def load_traversal_records(validation_summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in _as_list(validation_summary.get("rows")):
        if not isinstance(row, dict):
            continue
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        try:
            traversal = summarize_track4_traversal(run_dir)
        except Exception as exc:
            traversal = {
                "status": "SUMMARY_ERROR",
                "safe_for_thesis_claim": False,
                "failure_reasons": [str(exc)],
            }
        records.append(
            {
                "row": dict(row),
                "run_dir": str(run_dir),
                "summary": traversal,
            }
        )
    return records


def _zone_observations(records: Sequence[Mapping[str, Any]], *, primary_only: bool = True) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for record in records:
        row = record.get("row", {}) if isinstance(record.get("row"), dict) else {}
        summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
        zone_summary = summary.get("primary_zone_summary" if primary_only else "anchor_zone_summary")
        if not isinstance(zone_summary, dict):
            continue
        for zone, metrics in zone_summary.items():
            if not isinstance(metrics, dict):
                continue
            observations.append(
                {
                    "zone": str(zone),
                    "corpus": row.get("corpus"),
                    "corpus_kind": row.get("corpus_kind"),
                    "basis": row.get("basis"),
                    "kernel": row.get("kernel"),
                    "seed": row.get("seed"),
                    "path_count": _safe_float(metrics.get("path_count")),
                    "mean_work_integral": _safe_float(metrics.get("mean_work_integral")),
                    "closed_loop_rate": _safe_float(metrics.get("closed_loop_rate")),
                    "run_dir": record.get("run_dir"),
                }
            )
    return observations


def _group_by(items: Iterable[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(key)), []).append(item)
    return grouped


def _zone_distinctness_summary(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_zone = _group_by(observations, "zone")
    zone_summary = {
        zone: {
            "observation_count": len(group),
            "mean_path_count": _mean(item.get("path_count") for item in group),
            "mean_work_integral": _mean(item.get("mean_work_integral") for item in group),
            "std_work_integral": _std(item.get("mean_work_integral") for item in group),
            "mean_closed_loop_rate": _mean(item.get("closed_loop_rate") for item in group),
            "std_closed_loop_rate": _std(item.get("closed_loop_rate") for item in group),
        }
        for zone, group in sorted(by_zone.items())
    }
    work_values = [v["mean_work_integral"] for v in zone_summary.values() if v["mean_work_integral"] is not None]
    closed_values = [v["mean_closed_loop_rate"] for v in zone_summary.values() if v["mean_closed_loop_rate"] is not None]
    work_range = float(max(work_values) - min(work_values)) if len(work_values) >= 2 else None
    closed_loop_range = float(max(closed_values) - min(closed_values)) if len(closed_values) >= 2 else None
    zones_observed = sorted(zone_summary)
    supported = bool(
        len(zones_observed) >= 3
        and (
            (work_range is not None and work_range > 0.05)
            or (closed_loop_range is not None and closed_loop_range > 0.05)
        )
    )
    return {
        "status": "OK" if observations else "NO_DATA",
        "evidence_basis": "primary_path_touched_zone_summary",
        "zone_count": len(zones_observed),
        "zones_observed": zones_observed,
        "zone_summary": zone_summary,
        "work_range": work_range,
        "closed_loop_range": closed_loop_range,
        "construct_distinctness_supported": supported,
    }


def _zone_distinctness_by_key(observations: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    return {
        value: _zone_distinctness_summary(group)
        for value, group in sorted(_group_by(observations, key).items())
        if value not in {"None", ""}
    }


def terrain_construct_validity(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    observations = _zone_observations(records)
    global_summary = _zone_distinctness_summary(observations)
    by_corpus = _zone_distinctness_by_key(observations, "corpus")
    by_basis = _zone_distinctness_by_key(observations, "basis")
    by_kernel = _zone_distinctness_by_key(observations, "kernel")
    real_summary = by_corpus.get("real", {})
    real_supported = bool(real_summary.get("construct_distinctness_supported"))
    control_summaries = {
        corpus: summary
        for corpus, summary in by_corpus.items()
        if str(corpus).startswith("control_")
    }
    control_support_rate = _rate(
        summary.get("construct_distinctness_supported")
        for summary in control_summaries.values()
    )
    return {
        "status": "OK" if observations else "NO_DATA",
        "evidence_basis": "primary_path_touched_zone_summary",
        **global_summary,
        "global_construct_distinctness_supported": global_summary.get("construct_distinctness_supported"),
        "matched_real_construct_distinctness_supported": real_supported,
        "control_construct_distinctness_support_rate": control_support_rate,
        "by_corpus": by_corpus,
        "by_basis": by_basis,
        "by_kernel": by_kernel,
        "construct_distinctness_supported": real_supported,
        "pooling_warning": (
            "Global zone ranges pool corpora, kernels, bases, and seeds. The primary support flag "
            "requires matched real-corpus distinctness so controls cannot create the construct result alone."
        ),
        "interpretation": (
            "This only tests whether terrain bins occupy measurably different telemetry regimes. "
            "It does not prove semantic specificity against controls."
        ),
    }


def walker_sensitivity(method_sweep_path: Path | None) -> Dict[str, Any]:
    if method_sweep_path is None or not Path(method_sweep_path).exists():
        return {
            "status": "NO_DATA",
            "failure_reasons": ["method sweep summary missing"],
        }
    payload = _safe_json(Path(method_sweep_path))
    rows = _as_list(payload.get("ranked_conditions"))
    mode_robustness = _as_list(payload.get("mode_robustness"))
    if not rows:
        return {"status": "NO_DATA", "failure_reasons": ["ranked_conditions missing"]}

    def group_mean(field: str, group_key: str) -> Dict[str, float | None]:
        grouped = _group_by([row for row in rows if isinstance(row, dict)], group_key)
        return {
            key: _mean(item.get(field) for item in group)
            for key, group in sorted(grouped.items())
        }

    score_by_mode = group_mean("proposal_quality_score", "proposal_mode")
    score_by_k = group_mean("proposal_quality_score", "k_neighbors")
    score_by_temperature = group_mean("proposal_quality_score", "temperature")
    score_by_gamma = group_mean("proposal_quality_score", "gamma")
    all_scores = [value for value in (_safe_float(row.get("proposal_quality_score")) for row in rows if isinstance(row, dict)) if value is not None]
    score_range = float(max(all_scores) - min(all_scores)) if len(all_scores) >= 2 else None
    supported = bool(score_range is not None and score_range > 0.05)
    return {
        "status": "OK",
        "source": str(method_sweep_path),
        "condition_count": len(rows),
        "score_range": score_range,
        "walker_sensitivity_detected": supported,
        "score_by_mode": score_by_mode,
        "score_by_k_neighbors": score_by_k,
        "score_by_temperature": score_by_temperature,
        "score_by_gamma": score_by_gamma,
        "mode_robustness": mode_robustness,
    }


def work_decomposition_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row = record.get("row", {}) if isinstance(record.get("row"), dict) else {}
        summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
        mean_work = _safe_float(summary.get("mean_work_integral"))
        mean_edges = _safe_float(summary.get("mean_path_edge_count"))
        mean_nodes = _safe_float(summary.get("mean_path_node_count"))
        rows.append(
            {
                "corpus": row.get("corpus"),
                "basis": row.get("basis"),
                "kernel": row.get("kernel"),
                "seed": row.get("seed"),
                "mean_work_integral": mean_work,
                "mean_path_edge_count": mean_edges,
                "mean_path_node_count": mean_nodes,
                "work_per_edge_proxy": (
                    float(mean_work / mean_edges)
                    if mean_work is not None and mean_edges is not None and mean_edges > 0
                    else None
                ),
                "closed_loop_rate": _safe_float(summary.get("closed_loop_rate")),
                "primary_zone_count": _safe_float(summary.get("primary_zone_count")),
            }
        )
    work = [row.get("mean_work_integral") for row in rows]
    work_per_edge = [row.get("work_per_edge_proxy") for row in rows]
    return {
        "status": "OK" if rows else "NO_DATA",
        "native_component_decomposition_available": False,
        "decomposition_basis": "proxy_from_exported_path_lengths_and_summary_metrics",
        "warning": (
            "Current cyclic_paths exports do not separate edge-distance, stress, density, retreat, "
            "and shear costs. This summary is a proxy and should not be cited as component-level work."
        ),
        "row_count": len(rows),
        "mean_work_integral": _mean(work),
        "std_work_integral": _std(work),
        "mean_work_per_edge_proxy": _mean(work_per_edge),
        "std_work_per_edge_proxy": _std(work_per_edge),
        "confound_correlations": {
            "work_vs_path_edge_count": _pearson(
                (row.get("mean_work_integral") for row in rows),
                (row.get("mean_path_edge_count") for row in rows),
            ),
            "work_vs_path_node_count": _pearson(
                (row.get("mean_work_integral") for row in rows),
                (row.get("mean_path_node_count") for row in rows),
            ),
            "work_vs_closed_loop_rate": _pearson(
                (row.get("mean_work_integral") for row in rows),
                (row.get("closed_loop_rate") for row in rows),
            ),
            "work_vs_primary_zone_count": _pearson(
                (row.get("mean_work_integral") for row in rows),
                (row.get("primary_zone_count") for row in rows),
            ),
            "work_per_edge_vs_closed_loop_rate": _pearson(
                (row.get("work_per_edge_proxy") for row in rows),
                (row.get("closed_loop_rate") for row in rows),
            ),
        },
        "component_level_claim_safe": False,
        "rows": rows,
    }


def _summarize_validation_group(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "row_count": len(group),
        "terrain_safe_rate": _rate(row.get("safe_for_thesis_claim") for row in group),
        "mean_score": _mean(row.get("basis_probe_score") for row in group),
        "mean_closed_loop_rate": _mean(row.get("closed_loop_rate") for row in group),
        "mean_work_integral": _mean(row.get("mean_work_integral") for row in group),
        "mean_primary_zone_count": _mean(row.get("primary_zone_count") for row in group),
    }


def _specificity_comparison(
    *,
    real_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    real = _summarize_validation_group(real_rows)
    controls = _summarize_validation_group(control_rows)
    safe_gap = (real["terrain_safe_rate"] - controls["terrain_safe_rate"]) if control_rows else None
    score_gap = (
        float(real["mean_score"] - controls["mean_score"])
        if real["mean_score"] is not None and controls["mean_score"] is not None
        else None
    )
    return {
        "real": real,
        "controls": controls,
        "real_minus_control_terrain_safe_rate": safe_gap,
        "real_minus_control_mean_score": score_gap,
        "terrain_specificity_supported": bool(
            safe_gap is not None and safe_gap > 0.10 and (score_gap is None or score_gap > 0.0)
        ),
    }


def _specificity_by_key(
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> Dict[str, Any]:
    values = sorted({str(row.get(key)) for row in validation_rows if row.get(key) is not None})
    out: Dict[str, Any] = {}
    for value in values:
        scoped = [row for row in validation_rows if str(row.get(key)) == value]
        real = [row for row in scoped if row.get("corpus") == "real"]
        controls = [row for row in scoped if str(row.get("corpus", "")).startswith("control_")]
        if real or controls:
            out[value] = _specificity_comparison(real_rows=real, control_rows=controls)
    return out


def real_vs_control_specificity(validation_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    real_rows = [row for row in validation_rows if row.get("corpus") == "real"]
    control_rows = [row for row in validation_rows if str(row.get("corpus", "")).startswith("control_")]
    overall = _specificity_comparison(real_rows=real_rows, control_rows=control_rows)
    control_families = sorted({str(row.get("corpus")) for row in control_rows})
    by_control_family = {
        family: _specificity_comparison(
            real_rows=real_rows,
            control_rows=[row for row in control_rows if str(row.get("corpus")) == family],
        )
        for family in control_families
    }
    stochastic_controls = [
        row for row in control_rows if str(row.get("corpus")) in {"control_shuffled", "control_random"}
    ]
    if stochastic_controls:
        by_control_family["stochastic_controls"] = _specificity_comparison(
            real_rows=real_rows,
            control_rows=stochastic_controls,
        )
    return {
        "status": "OK" if real_rows and control_rows else "NO_DATA",
        **overall,
        "by_basis": _specificity_by_key(validation_rows, key="basis"),
        "by_kernel": _specificity_by_key(validation_rows, key="kernel"),
        "by_control_family": by_control_family,
        "missing_control_families": sorted(
            set(["control_constant", "control_shuffled", "control_random"]) - set(control_families)
        ),
        "interpretation": (
            "Specificity requires real runs to outperform controls, not merely produce terrain zones."
        ),
    }


def terrain_contrast_matrix(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    contrast_rows: List[Dict[str, Any]] = []
    for record in records:
        row = record.get("row", {}) if isinstance(record.get("row"), dict) else {}
        summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
        zone_summary = summary.get("primary_zone_summary")
        if not isinstance(zone_summary, dict):
            continue
        for left, right in CONTRAST_PAIRS:
            left_metrics = zone_summary.get(left)
            right_metrics = zone_summary.get(right)
            if not isinstance(left_metrics, dict) or not isinstance(right_metrics, dict):
                continue
            left_work = _safe_float(left_metrics.get("mean_work_integral"))
            right_work = _safe_float(right_metrics.get("mean_work_integral"))
            left_closed = _safe_float(left_metrics.get("closed_loop_rate"))
            right_closed = _safe_float(right_metrics.get("closed_loop_rate"))
            contrast_rows.append(
                {
                    "contrast": f"{left}_vs_{right}",
                    "corpus": row.get("corpus"),
                    "basis": row.get("basis"),
                    "kernel": row.get("kernel"),
                    "seed": row.get("seed"),
                    "work_gap_abs": abs(left_work - right_work) if left_work is not None and right_work is not None else None,
                    "closed_loop_gap_abs": abs(left_closed - right_closed) if left_closed is not None and right_closed is not None else None,
                    "left": left,
                    "right": right,
                }
            )
    by_contrast = _group_by(contrast_rows, "contrast")
    contrast_summary = {
        contrast: {
            "observation_count": len(group),
            "mean_work_gap_abs": _mean(item.get("work_gap_abs") for item in group),
            "mean_closed_loop_gap_abs": _mean(item.get("closed_loop_gap_abs") for item in group),
            "max_work_gap_abs": max(
                [float(v) for v in (_safe_float(item.get("work_gap_abs")) for item in group) if v is not None],
                default=None,
            ),
            "max_closed_loop_gap_abs": max(
                [float(v) for v in (_safe_float(item.get("closed_loop_gap_abs")) for item in group) if v is not None],
                default=None,
            ),
        }
        for contrast, group in sorted(by_contrast.items())
    }
    ranked = sorted(
        [
            {"contrast": contrast, **metrics}
            for contrast, metrics in contrast_summary.items()
        ],
        key=lambda item: (
            _safe_float(item.get("mean_work_gap_abs")) or 0.0,
            _safe_float(item.get("mean_closed_loop_gap_abs")) or 0.0,
        ),
        reverse=True,
    )
    return {
        "status": "OK" if contrast_rows else "NO_DATA",
        "contrast_count": len(contrast_summary),
        "contrast_summary": contrast_summary,
        "ranked_contrasts": ranked,
        "interpretation": "This tests whether contrasts other than Bridge/Void are more predictive.",
    }


def _unit_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-9:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def soft_terrain_membership(density: float, stress: float) -> Dict[str, float]:
    """Continuous Bridge/Swamp/Tightrope/Void membership on the unit square.

    The hard terrain bins are useful for screenshots and anchor provenance, but
    they throw away exactly the boundary information Track 4 walkers care about.
    These four bilinear memberships sum to one for density/stress in [0, 1].
    """
    d = float(np.clip(_safe_float(density) if _safe_float(density) is not None else 0.0, 0.0, 1.0))
    s = float(np.clip(_safe_float(stress) if _safe_float(stress) is not None else 0.0, 0.0, 1.0))
    return {
        "Bridge": d * (1.0 - s),
        "Swamp": d * s,
        "Tightrope": (1.0 - d) * (1.0 - s),
        "Void": (1.0 - d) * s,
    }


def _soft_membership_matrix(density_unit: np.ndarray, stress_unit: np.ndarray) -> np.ndarray:
    rows = [
        [soft_terrain_membership(float(d), float(s))[zone] for zone in TERRAIN_ZONES]
        for d, s in zip(density_unit, stress_unit)
    ]
    return np.asarray(rows, dtype=np.float64)


def _load_soft_terrain_fields(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    density_path = run_dir / "track3_density_rho.npy"
    stress_path = run_dir / "d_spectral.npy"
    cyclic_path = run_dir / "cyclic_paths.npz"
    missing = [
        str(path.name)
        for path in (density_path, stress_path, cyclic_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "MISSING_ARTIFACTS",
            "run_dir": str(run_dir),
            "missing_artifacts": missing,
        }
    try:
        density_raw = np.load(density_path, allow_pickle=False)
        stress_raw = np.load(stress_path, allow_pickle=False)
    except Exception as exc:
        return {
            "status": "LOAD_ERROR",
            "run_dir": str(run_dir),
            "error": str(exc),
        }
    density_unit = _unit_interval(np.asarray(density_raw).reshape(-1))
    stress_unit = _unit_interval(np.abs(np.asarray(stress_raw).reshape(-1)))
    if density_unit.size == 0 or density_unit.size != stress_unit.size:
        return {
            "status": "INVALID_FIELDS",
            "run_dir": str(run_dir),
            "density_length": int(density_unit.size),
            "stress_length": int(stress_unit.size),
        }
    return {
        "status": "OK",
        "run_dir": str(run_dir),
        "density_unit": density_unit,
        "stress_unit": stress_unit,
        "membership": _soft_membership_matrix(density_unit, stress_unit),
        "cyclic_path": cyclic_path,
    }


def _soft_path_rows_for_record(record: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    row = record.get("row", {}) if isinstance(record.get("row"), dict) else {}
    run_dir = Path(str(record.get("run_dir") or row.get("run_dir") or ""))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    fields = _load_soft_terrain_fields(run_dir)
    if fields.get("status") != "OK":
        return [], fields
    density_unit = np.asarray(fields["density_unit"], dtype=np.float64)
    stress_unit = np.asarray(fields["stress_unit"], dtype=np.float64)
    membership = np.asarray(fields["membership"], dtype=np.float64)
    n_nodes = int(density_unit.size)
    path_rows: List[Dict[str, Any]] = []
    try:
        with np.load(fields["cyclic_path"], allow_pickle=True) as npz:
            path_indices = list(npz["path_indices"])
            work_integral = np.asarray(npz["work_integral"], dtype=np.float64)
            closed_loop = np.asarray(npz["closed_loop"], dtype=bool)
            anchor_indices = np.asarray(npz["path_anchor_idx"], dtype=np.int64) if "path_anchor_idx" in npz.files else None
            path_hot = np.asarray(npz["path_is_hot"], dtype=bool) if "path_is_hot" in npz.files else None
    except Exception as exc:
        return [], {
            "status": "CYCLIC_LOAD_ERROR",
            "run_dir": str(run_dir),
            "error": str(exc),
        }

    for path_idx, raw_indices in enumerate(path_indices):
        try:
            indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        except Exception:
            continue
        indices = indices[(indices >= 0) & (indices < n_nodes)]
        if indices.size == 0:
            continue
        path_membership = membership[indices].mean(axis=0)
        member_by_zone = {
            f"soft_{zone.lower()}_mass": float(path_membership[pos])
            for pos, zone in enumerate(TERRAIN_ZONES)
        }
        work = _safe_float(work_integral[path_idx]) if path_idx < work_integral.size else None
        is_closed = bool(closed_loop[path_idx]) if path_idx < closed_loop.size else None
        dominant_zone = TERRAIN_ZONES[int(np.argmax(path_membership))]
        path_rows.append(
            {
                "corpus": row.get("corpus"),
                "corpus_kind": row.get("corpus_kind"),
                "basis": row.get("basis"),
                "kernel": row.get("kernel"),
                "seed": row.get("seed"),
                "run_dir": str(run_dir),
                "path_ordinal": int(path_idx),
                "anchor_idx": int(anchor_indices[path_idx]) if anchor_indices is not None and path_idx < anchor_indices.size else None,
                "is_hot": bool(path_hot[path_idx]) if path_hot is not None and path_idx < path_hot.size else None,
                "path_node_count": int(indices.size),
                "unique_node_count": int(np.unique(indices).size),
                "work_integral": work,
                "closed_loop": is_closed,
                "closed_loop_failure": 0.0 if is_closed else 1.0,
                "soft_density_mean": float(density_unit[indices].mean()),
                "soft_stress_mean": float(stress_unit[indices].mean()),
                "soft_barrier_mass": float(path_membership[1] + path_membership[3]),
                "soft_low_stress_mass": float(path_membership[0] + path_membership[2]),
                "soft_dominant_zone": dominant_zone,
                **member_by_zone,
            }
        )
    return path_rows, {"status": "OK", "run_dir": str(run_dir), "path_count": len(path_rows)}


def _soft_group_summary(path_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(path_rows)
    work_values = [row.get("work_integral") for row in rows]
    barrier_corr = _pearson((row.get("soft_barrier_mass") for row in rows), work_values)
    stress_corr = _pearson((row.get("soft_stress_mean") for row in rows), work_values)
    bridge_corr = _pearson((row.get("soft_bridge_mass") for row in rows), work_values)
    density_corr = _pearson((row.get("soft_density_mean") for row in rows), work_values)
    closed_failure_values = [row.get("closed_loop_failure") for row in rows]
    return {
        "path_count": len(rows),
        "run_count": len({str(row.get("run_dir")) for row in rows}),
        "mean_work_integral": _mean(work_values),
        "mean_soft_barrier_mass": _mean(row.get("soft_barrier_mass") for row in rows),
        "mean_soft_bridge_mass": _mean(row.get("soft_bridge_mass") for row in rows),
        "mean_soft_void_mass": _mean(row.get("soft_void_mass") for row in rows),
        "mean_soft_swamp_mass": _mean(row.get("soft_swamp_mass") for row in rows),
        "mean_soft_density": _mean(row.get("soft_density_mean") for row in rows),
        "mean_soft_stress": _mean(row.get("soft_stress_mean") for row in rows),
        "closed_loop_failure_rate": _mean(closed_failure_values),
        "corr_soft_barrier_mass_work": barrier_corr,
        "corr_soft_stress_mean_work": stress_corr,
        "corr_soft_bridge_mass_work": bridge_corr,
        "corr_soft_density_mean_work": density_corr,
        "corr_soft_void_mass_work": _pearson((row.get("soft_void_mass") for row in rows), work_values),
        "corr_soft_swamp_mass_work": _pearson((row.get("soft_swamp_mass") for row in rows), work_values),
        "corr_soft_barrier_mass_closed_loop_failure": _pearson(
            (row.get("soft_barrier_mass") for row in rows),
            closed_failure_values,
        ),
    }


def _soft_summary_by_key(path_rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    return {
        value: _soft_group_summary(group)
        for value, group in sorted(_group_by(path_rows, key).items())
        if value not in {"None", ""}
    }


def _soft_matched_cell_specificity(
    path_rows: Sequence[Mapping[str, Any]],
    *,
    control_families: Sequence[str] = ("control_random", "control_shuffled"),
    min_paths_per_side: int = 4,
    min_real_corr: float = 0.10,
    min_excess_corr: float = 0.05,
    min_matched_cells: int = 3,
    min_cell_pass_rate: float = 0.67,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in path_rows:
        key = (
            str(row.get("basis")),
            str(row.get("kernel")),
            str(row.get("seed")),
        )
        grouped.setdefault(key, []).append(row)

    cells: List[Dict[str, Any]] = []
    for (basis, kernel, seed), group in sorted(grouped.items()):
        real_rows = [row for row in group if row.get("corpus") == "real"]
        controls_by_family = {
            family: [row for row in group if row.get("corpus") == family]
            for family in control_families
        }
        usable_control_summaries: Dict[str, Any] = {}
        for family, rows in controls_by_family.items():
            corr = _pearson(
                (row.get("soft_barrier_mass") for row in rows),
                (row.get("work_integral") for row in rows),
            )
            usable_control_summaries[family] = {
                "path_count": len(rows),
                "corr_soft_barrier_mass_work": corr,
                "mean_work_integral": _mean(row.get("work_integral") for row in rows),
                "mean_soft_barrier_mass": _mean(row.get("soft_barrier_mass") for row in rows),
            }
        real_corr = _pearson(
            (row.get("soft_barrier_mass") for row in real_rows),
            (row.get("work_integral") for row in real_rows),
        )
        control_corrs = [
            value
            for value in (
                _safe_float(summary.get("corr_soft_barrier_mass_work"))
                for summary in usable_control_summaries.values()
            )
            if value is not None
        ]
        mean_control_corr = float(np.mean(control_corrs)) if control_corrs else None
        excess_corr = (
            float(real_corr - mean_control_corr)
            if real_corr is not None and mean_control_corr is not None
            else None
        )
        has_enough_paths = bool(
            len(real_rows) >= int(min_paths_per_side)
            and any(len(rows) >= int(min_paths_per_side) for rows in controls_by_family.values())
        )
        soft_support = bool(
            has_enough_paths
            and real_corr is not None
            and real_corr >= float(min_real_corr)
            and excess_corr is not None
            and excess_corr >= float(min_excess_corr)
        )
        cells.append(
            {
                "basis": basis,
                "kernel": kernel,
                "seed": seed,
                "cell_key": f"{basis}|{kernel}|seed{seed}",
                "real_path_count": len(real_rows),
                "control_path_count": sum(len(rows) for rows in controls_by_family.values()),
                "control_families": list(control_families),
                "real_corr_soft_barrier_mass_work": real_corr,
                "mean_control_corr_soft_barrier_mass_work": mean_control_corr,
                "excess_corr_real_minus_control": excess_corr,
                "real_mean_work_integral": _mean(row.get("work_integral") for row in real_rows),
                "control_mean_work_integral": _mean(
                    row.get("work_integral")
                    for rows in controls_by_family.values()
                    for row in rows
                ),
                "real_mean_soft_barrier_mass": _mean(row.get("soft_barrier_mass") for row in real_rows),
                "control_mean_soft_barrier_mass": _mean(
                    row.get("soft_barrier_mass")
                    for rows in controls_by_family.values()
                    for row in rows
                ),
                "has_enough_paths": has_enough_paths,
                "soft_support": soft_support,
                "control_summaries": usable_control_summaries,
            }
        )

    usable_cells = [
        cell
        for cell in cells
        if cell.get("has_enough_paths")
        and _safe_float(cell.get("real_corr_soft_barrier_mass_work")) is not None
        and _safe_float(cell.get("mean_control_corr_soft_barrier_mass_work")) is not None
    ]
    pass_rate = _rate(cell.get("soft_support") for cell in usable_cells)
    excess_values = [
        float(value)
        for value in (_safe_float(cell.get("excess_corr_real_minus_control")) for cell in usable_cells)
        if value is not None
    ]
    supported = bool(
        len(usable_cells) >= int(min_matched_cells)
        and pass_rate >= float(min_cell_pass_rate)
    )
    return {
        "status": "OK" if cells else "NO_DATA",
        "control_policy": "stochastic_controls",
        "control_families": list(control_families),
        "threshold_min_paths_per_side": int(min_paths_per_side),
        "threshold_min_real_corr": float(min_real_corr),
        "threshold_min_excess_corr": float(min_excess_corr),
        "threshold_min_matched_cells": int(min_matched_cells),
        "threshold_min_cell_pass_rate": float(min_cell_pass_rate),
        "matched_cell_count": len(cells),
        "usable_matched_cell_count": len(usable_cells),
        "supporting_cell_count": sum(1 for cell in usable_cells if bool(cell.get("soft_support"))),
        "matched_cell_pass_rate": pass_rate,
        "mean_excess_corr_real_minus_control": float(np.mean(excess_values)) if excess_values else None,
        "median_excess_corr_real_minus_control": float(np.median(excess_values)) if excess_values else None,
        "min_excess_corr_real_minus_control": float(np.min(excess_values)) if excess_values else None,
        "max_excess_corr_real_minus_control": float(np.max(excess_values)) if excess_values else None,
        "matched_soft_terrain_specificity_supported": supported,
        "cells": cells,
    }


def soft_terrain_work_coupling(
    records: Sequence[Mapping[str, Any]],
    *,
    min_path_count: int = 30,
    min_abs_work_corr: float = 0.10,
    min_real_minus_control_corr: float = 0.05,
    min_matched_cells: int = 3,
    min_matched_cell_pass_rate: float = 0.67,
) -> Dict[str, Any]:
    """Test terrain as a continuous field instead of four brittle bins."""
    path_rows: List[Dict[str, Any]] = []
    run_statuses: List[Dict[str, Any]] = []
    for record in records:
        rows, status = _soft_path_rows_for_record(record)
        run_statuses.append(status)
        path_rows.extend(rows)

    global_summary = _soft_group_summary(path_rows)
    by_corpus = _soft_summary_by_key(path_rows, "corpus")
    by_basis = _soft_summary_by_key(path_rows, "basis")
    by_kernel = _soft_summary_by_key(path_rows, "kernel")
    real_summary = by_corpus.get("real", {})
    control_summaries = {
        corpus: summary
        for corpus, summary in by_corpus.items()
        if str(corpus).startswith("control_")
    }
    control_barrier_corrs = [
        value
        for value in (
            _safe_float(summary.get("corr_soft_barrier_mass_work"))
            for summary in control_summaries.values()
        )
        if value is not None
    ]
    real_barrier_corr = _safe_float(real_summary.get("corr_soft_barrier_mass_work"))
    mean_control_barrier_corr = float(np.mean(control_barrier_corrs)) if control_barrier_corrs else None
    real_minus_control_barrier_corr = (
        float(real_barrier_corr - mean_control_barrier_corr)
        if real_barrier_corr is not None and mean_control_barrier_corr is not None
        else None
    )
    real_path_count = int(real_summary.get("path_count") or 0)
    real_soft_work_coupling_supported = bool(
        real_path_count >= int(min_path_count)
        and real_barrier_corr is not None
        and abs(float(real_barrier_corr)) >= float(min_abs_work_corr)
    )
    pooled_soft_terrain_specificity_supported = bool(
        real_soft_work_coupling_supported
        and real_minus_control_barrier_corr is not None
        and real_minus_control_barrier_corr >= float(min_real_minus_control_corr)
    )
    matched_specificity = _soft_matched_cell_specificity(
        path_rows,
        min_real_corr=float(min_abs_work_corr),
        min_excess_corr=float(min_real_minus_control_corr),
        min_matched_cells=int(min_matched_cells),
        min_cell_pass_rate=float(min_matched_cell_pass_rate),
    )
    matched_soft_terrain_specificity_supported = bool(
        matched_specificity.get("matched_soft_terrain_specificity_supported")
    )
    return {
        "status": "OK" if path_rows else "NO_DATA",
        "evidence_basis": "continuous_path_touched_density_stress_membership",
        "field_sources": {
            "density": "track3_density_rho.npy normalized like Track 4 _terrain_fields",
            "stress": "d_spectral.npy normalized like Track 4 _terrain_fields",
            "paths": "cyclic_paths.npz path_indices/work_integral/closed_loop",
        },
        "threshold_min_path_count": int(min_path_count),
        "threshold_min_abs_work_corr": float(min_abs_work_corr),
        "threshold_min_real_minus_control_corr": float(min_real_minus_control_corr),
        "threshold_min_matched_cells": int(min_matched_cells),
        "threshold_min_matched_cell_pass_rate": float(min_matched_cell_pass_rate),
        "run_count": len(run_statuses),
        "runs_with_soft_fields": sum(1 for status in run_statuses if status.get("status") == "OK"),
        "run_status_counts": {
            status: sum(1 for item in run_statuses if item.get("status") == status)
            for status in sorted({str(item.get("status")) for item in run_statuses})
        },
        "global": global_summary,
        "real": real_summary,
        "controls": {
            "family_count": len(control_summaries),
            "mean_barrier_work_corr": mean_control_barrier_corr,
            "by_family": control_summaries,
        },
        "real_minus_control_barrier_work_corr": real_minus_control_barrier_corr,
        "by_corpus": by_corpus,
        "by_basis": by_basis,
        "by_kernel": by_kernel,
        "real_soft_work_coupling_supported": real_soft_work_coupling_supported,
        "pooled_soft_terrain_specificity_supported": pooled_soft_terrain_specificity_supported,
        "matched_soft_terrain_specificity_supported": matched_soft_terrain_specificity_supported,
        "soft_terrain_specificity_supported": matched_soft_terrain_specificity_supported,
        "matched_cell_specificity": matched_specificity,
        "path_rows": path_rows,
        "interpretation": (
            "Positive matched specificity means fuzzy high-stress terrain predicts walker work in real data "
            "more than stochastic controls within matched basis/kernel/seed cells. Pooled specificity is "
            "reported separately because pooling can hide weak cells or inflate controls."
        ),
    }


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in {None, ""}:
            return row.get(key)
    return None


def _load_targeted_event_rows(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _safe_csv_rows(path), "csv_rows"
    if suffix == ".json":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)], "json_list"
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("pairs") or payload.get("records") or []
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)], "json_rows"
            return [], "json_no_rows"
    return [], "unsupported"


def _normalize_frame_relation(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"same", "same_frame", "sameframe", "matched", "within_frame", "same_event_same_frame"}:
        return "same_frame"
    if text in {
        "different",
        "different_frame",
        "differentframe",
        "cross",
        "cross_frame",
        "opposed_frame",
        "same_event_different_frame",
    }:
        return "different_frame"
    return text or "unknown"


def _normalize_targeted_pair_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    relation_text = str(_first_value(row, ("relation", "pair_relation", "event_frame_relation")) or "").strip().lower()
    same_event_raw = _first_value(row, ("same_event", "event_match", "is_same_event"))
    same_event = _safe_bool(same_event_raw)
    if same_event is None:
        relation = str(_first_value(row, ("event_relation", "event_match_type")) or "").strip().lower()
        if relation in {"same", "same_event", "matched_event"}:
            same_event = True
        elif relation in {"different", "different_event", "unmatched_event"}:
            same_event = False
    if same_event is None and relation_text in {"same_event_same_frame", "same_event_different_frame"}:
        same_event = True
    frame_relation = _normalize_frame_relation(
        _first_value(row, ("frame_relation", "frame_match", "frame_pair_type", "relation", "pair_relation"))
    )
    terrain_relation = str(
        _first_value(row, ("terrain_relation", "zone_relation", "terrain_pair_type")) or ""
    ).strip()
    left_frame = _first_value(row, ("left_frame", "frame_left", "source_frame_a"))
    right_frame = _first_value(row, ("right_frame", "frame_right", "source_frame_b"))
    if frame_relation == "unknown" and left_frame is not None and right_frame is not None:
        frame_relation = "same_frame" if str(left_frame) == str(right_frame) else "different_frame"
    left_work = _safe_float(_first_value(row, ("left_work", "work_left", "left_work_integral")))
    right_work = _safe_float(_first_value(row, ("right_work", "work_right", "right_work_integral")))
    work_gap = _safe_float(_first_value(row, ("work_gap", "track4_work_gap", "mean_work_gap")))
    if work_gap is None and left_work is not None and right_work is not None:
        work_gap = abs(float(left_work) - float(right_work))
    left_closed = _safe_float(_first_value(row, ("left_closed_loop_rate", "left_survival", "closed_left")))
    right_closed = _safe_float(_first_value(row, ("right_closed_loop_rate", "right_survival", "closed_right")))
    closed_gap = _safe_float(_first_value(row, ("closed_loop_gap", "survival_gap", "track4_survival_gap")))
    if closed_gap is None and left_closed is not None and right_closed is not None:
        closed_gap = abs(float(left_closed) - float(right_closed))
    left_source = _first_value(row, ("left_source", "left_source_id", "source_left", "source_a"))
    right_source = _first_value(row, ("right_source", "right_source_id", "source_right", "source_b"))
    same_source = _safe_bool(_first_value(row, ("same_source", "source_match")))
    if same_source is None and left_source is not None and right_source is not None:
        same_source = str(left_source) == str(right_source)
    return {
        "pair_id": _first_value(row, ("pair_id", "id")) or "",
        "event_id": _first_value(row, ("event_id", "event", "topic_id")) or "",
        "left_article_idx": _first_value(row, ("left_article_idx", "left_article_id", "left_idx", "article_a", "a_idx")),
        "right_article_idx": _first_value(row, ("right_article_idx", "right_article_id", "right_idx", "article_b", "b_idx")),
        "same_event": same_event,
        "frame_relation": frame_relation,
        "left_frame": left_frame,
        "right_frame": right_frame,
        "left_source": left_source,
        "right_source": right_source,
        "same_source": same_source,
        "terrain_relation": terrain_relation,
        "left_work": left_work,
        "right_work": right_work,
        "work_gap": work_gap,
        "closed_loop_gap": closed_gap,
        "track2_distance": _safe_float(_first_value(row, ("track2_distance", "embedding_distance", "semantic_distance"))),
        "raw": dict(row),
    }


def targeted_event_pair_results(
    targeted_event_path: Path | None,
    *,
    min_mixed_frame_events: int = 2,
    min_frame_pair_count: int = 3,
    min_cross_source_different_frame_pairs: int = 3,
    min_work_gap_lift: float = 0.10,
    min_cliffs_delta: float = 0.0,
    min_event_direction_pass_rate: float = 0.67,
) -> Dict[str, Any]:
    if targeted_event_path is None:
        return {
            "status": "NO_DATA",
            "failure_reasons": ["targeted event-pair corpus/results not provided"],
            "recommended_next_step": (
                "Create matched same-event article groups with source/frame metadata, then rerun Track 4 "
                "to test whether terrain predicts perspectival traversability inside event clusters."
            ),
        }
    path = Path(targeted_event_path)
    if not path.exists():
        return {
            "status": "NO_DATA",
            "failure_reasons": [f"targeted event-pair path missing: {path}"],
        }
    rows, schema = _load_targeted_event_rows(path)
    if not rows:
        if path.suffix.lower() == ".json":
            payload = _safe_json(path)
            return {
                "status": "AVAILABLE_UNINTERPRETED",
                "source": str(path),
                "payload_keys": sorted(payload.keys()),
                "schema": schema,
                "warning": "Targeted event-pair schema is not standardized yet.",
            }
        return {
            "status": "AVAILABLE_UNINTERPRETED",
            "source": str(path),
            "schema": schema,
            "warning": "Targeted event-pair schema is not standardized yet.",
        }

    normalized = [_normalize_targeted_pair_row(row) for row in rows]
    usable = [
        row
        for row in normalized
        if row.get("same_event") is not None and row.get("frame_relation") in {"same_frame", "different_frame"}
    ]
    same_event = [row for row in usable if row.get("same_event") is True]
    same_frame = [row for row in same_event if row.get("frame_relation") == "same_frame"]
    different_frame = [row for row in same_event if row.get("frame_relation") == "different_frame"]
    cross_source_different = [row for row in different_frame if row.get("same_source") is False]
    same_frame_work = [row.get("work_gap") for row in same_frame]
    different_frame_work = [row.get("work_gap") for row in different_frame]
    same_mean = _mean(same_frame_work)
    different_mean = _mean(different_frame_work)
    work_lift = (
        float(different_mean - same_mean)
        if same_mean is not None and different_mean is not None
        else None
    )
    cliffs = _cliffs_delta(different_frame_work, same_frame_work)
    by_event: Dict[str, Dict[str, Any]] = {}
    for row in same_event:
        event_id = str(row.get("event_id") or "unknown")
        event = by_event.setdefault(event_id, {"same_frame": [], "different_frame": []})
        if row.get("frame_relation") in {"same_frame", "different_frame"}:
            event[str(row.get("frame_relation"))].append(row)
    event_rows: List[Dict[str, Any]] = []
    event_direction_flags: List[bool] = []
    mixed_frame_event_count = 0
    for event_id, event in sorted(by_event.items()):
        event_same_mean = _mean(row.get("work_gap") for row in event["same_frame"])
        event_diff_mean = _mean(row.get("work_gap") for row in event["different_frame"])
        is_mixed = bool(event["same_frame"] and event["different_frame"])
        if is_mixed:
            mixed_frame_event_count += 1
        direction_pass = bool(
            event_same_mean is not None
            and event_diff_mean is not None
            and float(event_diff_mean) > float(event_same_mean)
        )
        if is_mixed:
            event_direction_flags.append(direction_pass)
        event_rows.append(
            {
                "event_id": event_id,
                "same_frame_pair_count": len(event["same_frame"]),
                "different_frame_pair_count": len(event["different_frame"]),
                "mean_same_frame_work_gap": event_same_mean,
                "mean_different_frame_work_gap": event_diff_mean,
                "direction_pass": direction_pass if is_mixed else None,
            }
        )
    event_direction_pass_rate = _rate(event_direction_flags)
    failures: List[str] = []
    if mixed_frame_event_count < int(min_mixed_frame_events):
        failures.append("insufficient mixed-frame events")
    if len(same_frame) < int(min_frame_pair_count):
        failures.append("insufficient same-event same-frame pairs")
    if len(different_frame) < int(min_frame_pair_count):
        failures.append("insufficient same-event different-frame pairs")
    if len(cross_source_different) < int(min_cross_source_different_frame_pairs):
        failures.append("insufficient cross-source different-frame pairs")
    if work_lift is None:
        failures.append("targeted event work-gap lift unavailable")
    elif work_lift <= float(min_work_gap_lift):
        failures.append("different-frame work gap lift below threshold")
    if cliffs is None:
        failures.append("targeted event Cliff's delta unavailable")
    elif cliffs <= float(min_cliffs_delta):
        failures.append("different-frame Cliff's delta below threshold")
    if event_direction_pass_rate < float(min_event_direction_pass_rate):
        failures.append("event direction pass rate below threshold")
    return {
        "status": "VALIDATED" if not failures else "INVALID",
        "source": str(path),
        "schema": schema,
        "pair_count": len(rows),
        "usable_pair_count": len(usable),
        "mixed_frame_event_count": mixed_frame_event_count,
        "same_event_pair_count": len(same_event),
        "same_event_same_frame_pair_count": len(same_frame),
        "same_event_different_frame_pair_count": len(different_frame),
        "cross_source_different_frame_pair_count": len(cross_source_different),
        "mean_same_event_same_frame_work_gap": same_mean,
        "mean_same_event_different_frame_work_gap": different_mean,
        "different_minus_same_frame_work_gap": work_lift,
        "cliffs_delta_different_gt_same": cliffs,
        "event_direction_pass_rate": event_direction_pass_rate,
        "mean_same_event_same_frame_track2_distance": _mean(row.get("track2_distance") for row in same_frame),
        "mean_same_event_different_frame_track2_distance": _mean(row.get("track2_distance") for row in different_frame),
        "threshold_min_mixed_frame_events": int(min_mixed_frame_events),
        "threshold_min_frame_pair_count": int(min_frame_pair_count),
        "threshold_min_cross_source_different_frame_pairs": int(min_cross_source_different_frame_pairs),
        "threshold_min_work_gap_lift": float(min_work_gap_lift),
        "threshold_min_cliffs_delta": float(min_cliffs_delta),
        "threshold_min_event_direction_pass_rate": float(min_event_direction_pass_rate),
        "safe_for_thesis_claim": not failures,
        "failure_reasons": failures,
        "event_rows": event_rows,
        "rows": normalized,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_diagnostics(
    *,
    validation_summary_path: Path,
    output_dir: Path,
    method_sweep_path: Path | None = DEFAULT_METHOD_SWEEP,
    targeted_event_path: Path | None = None,
) -> Dict[str, Any]:
    validation_payload = load_validation_payload(validation_summary_path)
    records = load_traversal_records(validation_payload)
    validation_rows = [record["row"] for record in records if isinstance(record.get("row"), dict)]

    diagnostics = {
        "terrain_construct_validity": terrain_construct_validity(records),
        "walker_sensitivity_matrix": walker_sensitivity(method_sweep_path),
        "work_decomposition_summary": work_decomposition_summary(records),
        "real_vs_control_terrain_specificity": real_vs_control_specificity(validation_rows),
        "terrain_contrast_matrix": terrain_contrast_matrix(records),
        "soft_terrain_work_coupling": soft_terrain_work_coupling(records),
        "targeted_event_pair_results": targeted_event_pair_results(targeted_event_path),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_map = {}
    for name, payload in diagnostics.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        file_map[name] = str(path)

    targeted_status = diagnostics["targeted_event_pair_results"].get("status")
    targeted_claim_ready = targeted_status in {"OK", "VALIDATED", "MEASURED"}
    summary = {
        "schema_version": "1.0",
        "diagnostic_type": "track4_terrain_semantics",
        "validation_summary_path": str(validation_summary_path),
        "method_sweep_path": str(method_sweep_path) if method_sweep_path is not None else None,
        "record_count": len(records),
        "outputs": file_map,
        "claim_boundary": {
            "terrain_construct_distinctness": diagnostics["terrain_construct_validity"].get("construct_distinctness_supported"),
            "walker_sensitivity_detected": diagnostics["walker_sensitivity_matrix"].get("walker_sensitivity_detected"),
            "native_work_decomposition_available": diagnostics["work_decomposition_summary"].get("native_component_decomposition_available"),
            "terrain_specificity_supported": diagnostics["real_vs_control_terrain_specificity"].get("terrain_specificity_supported"),
            "soft_terrain_work_coupling_supported": diagnostics["soft_terrain_work_coupling"].get("real_soft_work_coupling_supported"),
            "pooled_soft_terrain_specificity_supported": diagnostics["soft_terrain_work_coupling"].get("pooled_soft_terrain_specificity_supported"),
            "matched_soft_terrain_specificity_supported": diagnostics["soft_terrain_work_coupling"].get("matched_soft_terrain_specificity_supported"),
            "soft_terrain_specificity_supported": diagnostics["soft_terrain_work_coupling"].get("soft_terrain_specificity_supported"),
            "targeted_event_pair_status": targeted_status,
            "targeted_event_pair_available": targeted_claim_ready,
            "targeted_event_pair_claim_ready": targeted_claim_ready,
        },
    }
    summary_path = output_dir / "track4_terrain_semantics_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    file_map["summary"] = str(summary_path)

    contrast_rows = diagnostics["terrain_contrast_matrix"].get("ranked_contrasts", [])
    if contrast_rows:
        write_csv(
            output_dir / "terrain_contrast_matrix.csv",
            contrast_rows,
            [
                "contrast",
                "observation_count",
                "mean_work_gap_abs",
                "mean_closed_loop_gap_abs",
                "max_work_gap_abs",
                "max_closed_loop_gap_abs",
            ],
        )
    soft_rows = diagnostics["soft_terrain_work_coupling"].get("path_rows", [])
    if soft_rows:
        write_csv(
            output_dir / "soft_terrain_path_rows.csv",
            soft_rows,
            [
                "corpus",
                "corpus_kind",
                "basis",
                "kernel",
                "seed",
                "path_ordinal",
                "anchor_idx",
                "is_hot",
                "path_node_count",
                "unique_node_count",
                "work_integral",
                "closed_loop",
                "soft_density_mean",
                "soft_stress_mean",
                "soft_barrier_mass",
                "soft_bridge_mass",
                "soft_swamp_mass",
                "soft_tightrope_mass",
                "soft_void_mass",
                "soft_dominant_zone",
                "run_dir",
            ],
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--method-sweep-summary", type=Path, default=DEFAULT_METHOD_SWEEP)
    parser.add_argument("--targeted-event-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_diagnostics(
        validation_summary_path=args.validation_summary,
        output_dir=args.output_dir,
        method_sweep_path=args.method_sweep_summary,
        targeted_event_path=args.targeted_event_results,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
