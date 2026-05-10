from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _safe_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _safe_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            f = float(text)
        except Exception:
            return None
        return f if math.isfinite(f) else None
    return None


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except Exception:
            return None
    return None


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(mean(vals)) if vals else None


def _median_or_none(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(median(vals)) if vals else None


def _std_or_none(values: Iterable[float]) -> Optional[float]:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    return float(np.std(vals)) if vals.size else None


def _bridge_void_from_zone_summary(
    zone_summary: Dict[str, Dict[str, Any]],
    *,
    min_semantic_gap: float,
    coverage_label: str = "path-touched",
) -> tuple[Dict[str, float], List[str]]:
    bridge = zone_summary.get("Bridge")
    void = zone_summary.get("Void")
    bridge_void: Dict[str, float] = {}
    failures: List[str] = []
    if len(zone_summary) < 3:
        if coverage_label == "anchor":
            failures.append("track 4 anchors cover fewer than three terrain zones")
        else:
            failures.append("track 4 path-touched terrain coverage fewer than three terrain zones")
    if bridge and void:
        if bridge.get("closed_loop_rate") is not None and void.get("closed_loop_rate") is not None:
            bridge_void["closed_loop_rate_gap"] = float(bridge["closed_loop_rate"]) - float(void["closed_loop_rate"])
        if bridge.get("mean_work_integral") is not None and void.get("mean_work_integral") is not None:
            bridge_void["work_integral_gap"] = float(void["mean_work_integral"]) - float(bridge["mean_work_integral"])
        closed_gap = bridge_void.get("closed_loop_rate_gap")
        work_gap = bridge_void.get("work_integral_gap")
        if closed_gap is not None and closed_gap <= min_semantic_gap:
            failures.append("bridge/void closed-loop gap below minimum semantic effect size")
        if work_gap is not None and work_gap <= min_semantic_gap:
            failures.append("bridge/void work-integral gap below minimum semantic effect size")
    else:
        failures.append("bridge/void comparison unavailable for this run")
    return bridge_void, failures


def _load_article_records(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    article_map: Dict[int, Dict[str, Any]] = {}

    state_path = run_dir / "baseline_state.json"
    if state_path.exists():
        state = _safe_json(state_path)
        for article in state.get("articles", []):
            if isinstance(article, dict):
                idx = _to_int(article.get("index"))
                if idx is not None:
                    article_map[idx] = dict(article)

    view_state_path = run_dir / "MONOLITH.view_state.json"
    if view_state_path.exists():
        view_state = _safe_json(view_state_path)
        for article in view_state.get("articles", []):
            if isinstance(article, dict):
                idx = _to_int(article.get("idx"))
                if idx is None:
                    continue
                merged = article_map.setdefault(idx, {})
                merged.update(article)

    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    if monolith_csv.exists():
        for row in _safe_csv_rows(monolith_csv):
            idx = _to_int(row.get("index"))
            if idx is None:
                continue
            merged = article_map.setdefault(idx, {})
            merged.update(row)

    return article_map


def _iter_observer_delta_records(run_dir: Path) -> List[Dict[str, Any]]:
    rel_path = run_dir / "relativity_deltas.json"
    if rel_path.exists():
        blob = _safe_json(rel_path)
        observers = blob.get("observers", [])
        if isinstance(observers, list):
            return [obs for obs in observers if isinstance(obs, dict)]

    rel_cache = run_dir / "relativity_cache"
    delta_records: List[Dict[str, Any]] = []
    if rel_cache.exists():
        for path in sorted(rel_cache.glob("delta_*.json"), key=lambda p: p.name):
            blob = _safe_json(path)
            if blob:
                delta_records.append(blob)
    return delta_records


def summarize_observer_relativity(run_dir: Path, *, coord_epsilon: float = 1e-6, rotation_epsilon: float = 1e-6) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    observer_records = _iter_observer_delta_records(run_dir)
    failures: List[str] = []

    if not observer_records:
        return {
            "status": "NO_DATA",
            "run_dir": str(run_dir),
            "observer_count": 0,
            "valid_observer_count": 0,
            "placeholder_count": 0,
            "safe_for_thesis_claim": False,
            "failure_reasons": ["observer relativity artifacts missing"],
        }

    max_coord_deltas: List[float] = []
    abs_rotations: List[float] = []
    path_flip_counts: List[float] = []
    nmi_deltas: List[float] = []
    survival_deltas: List[float] = []
    placeholders = 0
    equivalent_count = 0
    nonzero_coord = 0
    nonzero_rotation = 0
    positive_path_flip = 0

    for obs in observer_records:
        provenance = obs.get("provenance", {}) if isinstance(obs.get("provenance"), dict) else {}
        null_eq = obs.get("null_observer_equivalence", {}) if isinstance(obs.get("null_observer_equivalence"), dict) else {}
        axis_delta = obs.get("axis_delta", {}) if isinstance(obs.get("axis_delta"), dict) else {}
        metrics_delta = obs.get("metrics_delta", {}) if isinstance(obs.get("metrics_delta"), dict) else {}
        translation = obs.get("translation_only_comparison", {}) if isinstance(obs.get("translation_only_comparison"), dict) else {}
        path_flip_delta = obs.get("path_flip_delta", {}) if isinstance(obs.get("path_flip_delta"), dict) else {}

        is_placeholder = bool(provenance.get("synthetic_placeholder", False))
        if is_placeholder:
            placeholders += 1

        equivalent = bool(null_eq.get("equivalent", False))
        if equivalent:
            equivalent_count += 1

        max_coord_delta = _to_float(null_eq.get("max_coord_delta"))
        rotation_deg = _to_float(axis_delta.get("rotation_deg"))
        nmi_delta = _to_float(metrics_delta.get("d_nmi"))
        survival_delta = _to_float(metrics_delta.get("d_survival_pct"))

        path_flip_count = _to_float(null_eq.get("path_flip_count"))
        if path_flip_count is None:
            path_flip_count = _to_float(translation.get("d_path_flip_count"))
        if path_flip_count is None and path_flip_delta:
            positives = sum(1 for value in path_flip_delta.values() if (_to_float(value) or 0.0) > coord_epsilon)
            path_flip_count = float(positives)

        if max_coord_delta is not None:
            max_coord_deltas.append(max_coord_delta)
            if max_coord_delta > coord_epsilon:
                nonzero_coord += 1
        if rotation_deg is not None:
            abs_rot = abs(rotation_deg)
            abs_rotations.append(abs_rot)
            if abs_rot > rotation_epsilon:
                nonzero_rotation += 1
        if path_flip_count is not None:
            path_flip_counts.append(path_flip_count)
            if path_flip_count > 0:
                positive_path_flip += 1
        if nmi_delta is not None:
            nmi_deltas.append(nmi_delta)
        if survival_delta is not None:
            survival_deltas.append(survival_delta)

    valid_observers = len(observer_records) - placeholders
    if valid_observers <= 0:
        failures.append("all observer relativity artifacts are placeholders")
    if nonzero_coord <= 0:
        failures.append("observer relativity produced no nonzero coordinate displacement")
    if nonzero_rotation <= 0:
        failures.append("observer relativity produced no nonzero axis rotation")
    if positive_path_flip <= 0:
        failures.append("observer relativity produced no path flips")
    if equivalent_count == len(observer_records):
        failures.append("all observer universes are null-equivalent")

    return {
        "status": "OK" if not failures else "INVALID",
        "run_dir": str(run_dir),
        "observer_count": len(observer_records),
        "valid_observer_count": valid_observers,
        "placeholder_count": placeholders,
        "null_equivalent_count": equivalent_count,
        "nonzero_coord_delta_count": nonzero_coord,
        "nonzero_rotation_count": nonzero_rotation,
        "positive_path_flip_count": positive_path_flip,
        "mean_max_coord_delta": _mean_or_none(max_coord_deltas),
        "max_coord_delta": max(max_coord_deltas) if max_coord_deltas else None,
        "mean_abs_rotation_deg": _mean_or_none(abs_rotations),
        "max_abs_rotation_deg": max(abs_rotations) if abs_rotations else None,
        "mean_path_flip_count": _mean_or_none(path_flip_counts),
        "mean_nmi_delta": _mean_or_none(nmi_deltas),
        "mean_survival_delta": _mean_or_none(survival_deltas),
        "safe_for_thesis_claim": not failures,
        "failure_reasons": failures,
    }


def _resolve_anchor_article_index(path_anchor_value: int, anchor_indices: np.ndarray) -> Optional[int]:
    if anchor_indices.size == 0:
        return None
    if 0 <= path_anchor_value < int(anchor_indices.shape[0]):
        return int(anchor_indices[path_anchor_value])
    if path_anchor_value in set(int(v) for v in anchor_indices.tolist()):
        return int(path_anchor_value)
    return None


def summarize_track4_traversal(
    run_dir: Path,
    *,
    work_epsilon: float = 1e-9,
    min_semantic_gap: float = 0.05,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    cyclic_path = run_dir / "cyclic_paths.npz"
    article_map = _load_article_records(run_dir)
    failures: List[str] = []
    warnings: List[str] = []

    if not cyclic_path.exists():
        legacy_work = run_dir / "walker_work_integrals.npy"
        if not legacy_work.exists():
            return {
                "status": "NO_DATA",
                "run_dir": str(run_dir),
                "safe_for_thesis_claim": False,
                "failure_reasons": ["track 4 traversal artifacts missing"],
            }
        work = np.load(legacy_work, allow_pickle=True).reshape(-1)
        view_state_path = run_dir / "MONOLITH.view_state.json"
        closed_loop_rate = None
        if view_state_path.exists():
            view_state = _safe_json(view_state_path)
            metrics = view_state.get("metrics", {}) if isinstance(view_state.get("metrics"), dict) else {}
            closed_loop_rate = _to_float(metrics.get("walker_survival_rate"))
        failures.append("cyclic_paths.npz missing; using legacy Track 4 fallback")
        return {
            "status": "LEGACY_FALLBACK",
            "run_dir": str(run_dir),
            "path_count": int(work.size),
            "mean_work_integral": _mean_or_none(work.tolist()),
            "median_work_integral": _median_or_none(work.tolist()),
            "std_work_integral": _std_or_none(work.tolist()),
            "closed_loop_rate": closed_loop_rate,
            "safe_for_thesis_claim": False,
            "failure_reasons": failures,
        }

    npz = np.load(cyclic_path, allow_pickle=True)
    required_keys = {"work_integral", "closed_loop", "path_anchor_idx", "anchor_indices"}
    missing_keys = sorted(required_keys - set(npz.files))
    if missing_keys:
        return {
            "status": "INVALID",
            "run_dir": str(run_dir),
            "path_count": 0,
            "mean_work_integral": None,
            "median_work_integral": None,
            "std_work_integral": None,
            "closed_loop_rate": None,
            "terrain_evidence_basis": "path_touched",
            "safe_for_thesis_claim": False,
            "failure_reasons": [
                f"cyclic_paths.npz missing required Track 4 keys: {', '.join(missing_keys)}"
            ],
            "warnings": [f"available cyclic_paths.npz keys: {sorted(npz.files)}"],
        }
    work = np.asarray(npz["work_integral"], dtype=float).reshape(-1)
    closed_loop = np.asarray(npz["closed_loop"], dtype=bool).reshape(-1)
    path_anchor_idx = np.asarray(npz["path_anchor_idx"], dtype=int).reshape(-1)
    anchor_indices = np.asarray(npz["anchor_indices"], dtype=int).reshape(-1)
    path_is_hot = np.asarray(npz["path_is_hot"], dtype=bool).reshape(-1) if "path_is_hot" in npz.files else None
    path_indices = None
    if "path_indices" in npz.files:
        raw_path_indices = np.asarray(npz["path_indices"], dtype=object)
        if raw_path_indices.ndim == 0:
            path_indices = [raw_path_indices.item()]
        elif raw_path_indices.ndim == 1:
            path_indices = list(raw_path_indices)
        else:
            # Preserve each exported walker row; flattening makes repeated traces look diverse.
            path_indices = [raw_path_indices[idx] for idx in range(raw_path_indices.shape[0])]
    markov_summary = _safe_json(run_dir / "track4_markov_summary.json") if (run_dir / "track4_markov_summary.json").exists() else {}
    if "track4_markov_status" in npz.files and "status" not in markov_summary:
        status_arr = np.asarray(npz["track4_markov_status"], dtype=object).reshape(-1)
        if status_arr.size:
            markov_summary["status"] = str(status_arr[0])
    committor_summary: Dict[str, Any] = {}
    if "committor_to_void" in npz.files:
        committor = np.asarray(npz["committor_to_void"], dtype=float).reshape(-1)
        finite = committor[np.isfinite(committor)]
        committor_summary = {
            "available": bool(finite.size),
            "finite_fraction": float(finite.size / max(committor.size, 1)),
            "mean": _mean_or_none(finite.tolist()),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    mfpt_summary: Dict[str, Any] = {}
    for key in ("mfpt_to_bridge", "mfpt_to_void"):
        if key in npz.files:
            values = np.asarray(npz[key], dtype=float).reshape(-1)
            finite = values[np.isfinite(values)]
            mfpt_summary[key] = {
                "available": bool(finite.size),
                "finite_fraction": float(finite.size / max(values.size, 1)),
                "mean": _mean_or_none(finite.tolist()),
                "median": _median_or_none(finite.tolist()),
            }
    reactive_flux_summary: Dict[str, Any] = {}
    if "reactive_flux_values" in npz.files:
        flux = np.asarray(npz["reactive_flux_values"], dtype=float).reshape(-1)
        finite_flux = flux[np.isfinite(flux)]
        reactive_flux_summary = {
            "available": bool(finite_flux.size),
            "edge_count": int(finite_flux.size),
            "total": float(np.sum(finite_flux)) if finite_flux.size else 0.0,
            "max": float(np.max(finite_flux)) if finite_flux.size else None,
        }

    if work.size == 0:
        failures.append("track 4 exported zero paths")
    if work.size != closed_loop.size or work.size != path_anchor_idx.size:
        failures.append("track 4 arrays have inconsistent lengths")

    finite_work = np.isfinite(work)
    finite_fraction = float(finite_work.mean()) if work.size else 0.0
    if finite_fraction < 1.0:
        failures.append("track 4 work integrals contain non-finite values")

    nonzero_fraction = float((np.abs(work) > work_epsilon).mean()) if work.size else 0.0
    if nonzero_fraction <= 0.0:
        failures.append("track 4 work integrals are all zero")

    anchor_stats: Dict[str, Dict[str, Any]] = {}
    zone_buckets: Dict[str, Dict[str, List[float]]] = {}
    touched_zone_buckets: Dict[str, Dict[str, List[float]]] = {}

    for idx in range(min(work.size, closed_loop.size, path_anchor_idx.size)):
        anchor_article_idx = _resolve_anchor_article_index(int(path_anchor_idx[idx]), anchor_indices)
        hot = bool(path_is_hot[idx]) if path_is_hot is not None and idx < path_is_hot.size else None
        zone = None
        if anchor_article_idx is not None and anchor_article_idx in article_map:
            zone = str(article_map[anchor_article_idx].get("zone", "")).strip() or None
        touched_zones: List[str] = []
        if path_indices is not None and idx < len(path_indices):
            raw_path = np.atleast_1d(np.asarray(path_indices[idx], dtype=int)).reshape(-1).tolist()
            touched_zones = sorted(
                {
                    str(article_map.get(path_idx, {}).get("zone", "")).strip()
                    for path_idx in raw_path
                    if str(article_map.get(path_idx, {}).get("zone", "")).strip()
                }
            )

        anchor_key = str(anchor_article_idx) if anchor_article_idx is not None else f"unknown:{idx}"
        stats = anchor_stats.setdefault(
            anchor_key,
            {
                "anchor_article_idx": anchor_article_idx,
                "zone": zone,
                "path_count": 0,
                "closed_loop_count": 0,
                "hot_count": 0,
                "cold_count": 0,
                "work_integrals": [],
                "touched_zones": set(),
            },
        )
        stats["path_count"] += 1
        stats["closed_loop_count"] += int(bool(closed_loop[idx]))
        if hot is True:
            stats["hot_count"] += 1
        elif hot is False:
            stats["cold_count"] += 1
        stats["work_integrals"].append(float(work[idx]))
        stats["touched_zones"].update(touched_zones)

        if zone:
            bucket = zone_buckets.setdefault(zone, {"work": [], "closed": []})
            bucket["work"].append(float(work[idx]))
            bucket["closed"].append(float(bool(closed_loop[idx])))
        for touched_zone in touched_zones:
            touched_bucket = touched_zone_buckets.setdefault(touched_zone, {"work": [], "closed": []})
            touched_bucket["work"].append(float(work[idx]))
            touched_bucket["closed"].append(float(bool(closed_loop[idx])))

    anchor_rows: List[Dict[str, Any]] = []
    for stats in anchor_stats.values():
        path_count = int(stats["path_count"])
        anchor_rows.append(
            {
                "anchor_article_idx": stats["anchor_article_idx"],
                "zone": stats["zone"],
                "path_count": path_count,
                "closed_loop_rate": (float(stats["closed_loop_count"]) / path_count) if path_count else None,
                "hot_count": int(stats["hot_count"]),
                "cold_count": int(stats["cold_count"]),
                "mean_work_integral": _mean_or_none(stats["work_integrals"]),
                "touched_zones": sorted(str(zone_name) for zone_name in stats["touched_zones"]),
            }
        )

    zone_summary = {
        zone: {
            "path_count": len(values["work"]),
            "mean_work_integral": _mean_or_none(values["work"]),
            "closed_loop_rate": _mean_or_none(values["closed"]),
        }
        for zone, values in zone_buckets.items()
    }
    touched_zone_summary = {
        zone: {
            "path_count": len(values["work"]),
            "mean_work_integral": _mean_or_none(values["work"]),
            "closed_loop_rate": _mean_or_none(values["closed"]),
        }
        for zone, values in touched_zone_buckets.items()
    }

    hot_count = int(path_is_hot.sum()) if path_is_hot is not None else None
    cold_count = int((~path_is_hot).sum()) if path_is_hot is not None else None
    if path_is_hot is not None and (hot_count == 0 or cold_count == 0):
        failures.append("track 4 did not preserve both hot and cold walkers")

    unique_path_shapes = 0
    if path_indices is not None:
        unique_path_shapes = len(
            {
                tuple(np.atleast_1d(np.asarray(path, dtype=int)).reshape(-1).tolist())
                for path in path_indices
            }
        )
        if unique_path_shapes <= 1:
            failures.append("all Track 4 paths collapse to one repeated index trace")

    primary_zone_summary = touched_zone_summary if touched_zone_summary else zone_summary
    terrain_evidence_basis = "path_touched" if touched_zone_summary else "anchor"
    primary_bridge_void, primary_failures = _bridge_void_from_zone_summary(
        primary_zone_summary,
        min_semantic_gap=min_semantic_gap,
        coverage_label=terrain_evidence_basis,
    )
    failures.extend(primary_failures)
    anchor_bridge_void, anchor_failures = _bridge_void_from_zone_summary(
        zone_summary,
        min_semantic_gap=min_semantic_gap,
        coverage_label="anchor",
    )
    if terrain_evidence_basis == "path_touched" and anchor_failures:
        warnings.extend([f"anchor diagnostic: {reason}" for reason in anchor_failures])
        if (
            "track 4 path-touched terrain coverage fewer than three terrain zones" in failures
            and "track 4 anchors cover fewer than three terrain zones" in anchor_failures
        ):
            failures.append("track 4 anchors cover fewer than three terrain zones")

    if not anchor_rows:
        failures.append("track 4 anchor summaries could not be constructed")

    return {
        "status": "OK" if not failures else "INVALID",
        "run_dir": str(run_dir),
        "path_count": int(work.size),
        "anchor_count": int(anchor_indices.size),
        "unique_anchor_article_count": len({row["anchor_article_idx"] for row in anchor_rows if row["anchor_article_idx"] is not None}),
        "finite_work_fraction": finite_fraction,
        "nonzero_work_fraction": nonzero_fraction,
        "mean_work_integral": _mean_or_none(work.tolist()),
        "median_work_integral": _median_or_none(work.tolist()),
        "std_work_integral": _std_or_none(work.tolist()),
        "closed_loop_rate": float(closed_loop.mean()) if closed_loop.size else None,
        "hot_count": hot_count,
        "cold_count": cold_count,
        "unique_path_shape_count": unique_path_shapes if path_indices is not None else None,
        "per_anchor": anchor_rows,
        "zone_summary": zone_summary,
        "anchor_zone_summary": zone_summary,
        "anchor_zone_count": len(zone_summary),
        "touched_zone_summary": touched_zone_summary,
        "touched_zone_count": len(touched_zone_summary),
        "primary_zone_summary": primary_zone_summary,
        "primary_zone_count": len(primary_zone_summary),
        "terrain_evidence_basis": terrain_evidence_basis,
        "primary_bridge_vs_void": primary_bridge_void,
        "anchor_bridge_vs_void": anchor_bridge_void,
        "bridge_vs_void": anchor_bridge_void,
        "markov_summary": markov_summary,
        "committor_summary": committor_summary,
        "mfpt_summary": mfpt_summary,
        "reactive_flux_summary": reactive_flux_summary,
        "safe_for_thesis_claim": not failures,
        "failure_reasons": failures,
        "warnings": warnings,
    }


def write_observer_relativity_summary(run_dir: Path, out_path: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path is not None else run_dir / "observer_relativity_summary.json"
    out_path.write_text(json.dumps(summarize_observer_relativity(run_dir), indent=2), encoding="utf-8")
    return out_path


def write_track4_traversal_summary(run_dir: Path, out_path: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path is not None else run_dir / "track4_traversal_summary.json"
    out_path.write_text(json.dumps(summarize_track4_traversal(run_dir), indent=2), encoding="utf-8")
    return out_path
