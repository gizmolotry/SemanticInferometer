"""Observer-slice transport diagnostics for Track 4.

Track 4 is most useful when it measures action across an atlas of
observer-conditioned charts, not only paths inside one static terrain.  This
module adds a small, deterministic diagnostic for that atlas view: compare the
cost of moving semantically first and switching observer second against the
cost of switching observer first and moving semantically second.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ObserverSliceTransportConfig:
    """Weights for observer-slice commutator action."""

    semantic_weight: float = 1.0
    observer_switch_weight: float = 1.0
    stress_weight: float = 0.0
    density_weight: float = 0.0
    simplex_weight: float = 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _coerce_slices(slices: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    expected_n: Optional[int] = None
    for name, value in slices.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"slice {name!r} must be a 2D array, got shape={arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError(f"slice {name!r} must contain at least two article nodes")
        if expected_n is None:
            expected_n = int(arr.shape[0])
        elif int(arr.shape[0]) != expected_n:
            raise ValueError("all observer slices must contain the same article count")
        out[str(name)] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(out) < 2:
        raise ValueError("at least two observer slices are required")
    return out


def _coerce_vector(value: Any | None, n_items: int, *, default: float) -> np.ndarray:
    if value is None:
        return np.full(n_items, float(default), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n_items:
        raise ValueError(f"expected vector length {n_items}, got {arr.shape[0]}")
    return np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))


def _coerce_simplex(value: Any | None, n_slices: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != n_slices:
        raise ValueError(f"slice_simplex must have shape [{n_slices}, B], got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(arr < 0.0):
        arr = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    row_sums = arr.sum(axis=1, keepdims=True)
    arr = arr / np.clip(row_sums, 1e-12, None)
    zero_rows = row_sums.reshape(-1) <= 1e-12
    if np.any(zero_rows):
        arr[zero_rows] = 1.0 / float(arr.shape[1])
    return np.clip(arr, 1e-12, 1.0)


def _simplex_l1(simplex: Optional[np.ndarray], left: int, right: int) -> float:
    if simplex is None:
        return 0.0
    return float(0.5 * np.linalg.norm(simplex[int(left)] - simplex[int(right)], ord=1))


def _point_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64), ord=2))


def _loop_corner(row_index: int, article_idx: int, slice_name: str, row_to_article_index: Any | None) -> Dict[str, Any]:
    corner = {"article_idx": int(article_idx), "slice": slice_name}
    if row_to_article_index is not None:
        corner["row_index"] = int(row_index)
    return corner


def _article_idx_for_row(row_index: int, row_to_article_index: Any | None) -> int:
    if row_to_article_index is None:
        return int(row_index)
    if isinstance(row_to_article_index, Mapping):
        try:
            return int(row_to_article_index[int(row_index)])
        except Exception:
            return int(row_index)
    if isinstance(row_to_article_index, Sequence) and not isinstance(row_to_article_index, (str, bytes)):
        try:
            return int(row_to_article_index[int(row_index)])
        except Exception:
            return int(row_index)
    return int(row_index)


def _semantic_action(
    coords: np.ndarray,
    source_idx: int,
    target_idx: int,
    *,
    density: np.ndarray,
    stress: np.ndarray,
    config: ObserverSliceTransportConfig,
) -> float:
    base = _point_distance(coords[int(source_idx)], coords[int(target_idx)])
    density_mid = 0.5 * (float(density[int(source_idx)]) + float(density[int(target_idx)]))
    stress_mid = 0.5 * (float(stress[int(source_idx)]) + float(stress[int(target_idx)]))
    density_penalty = float(config.density_weight) * max(0.0, 1.0 - density_mid) * base
    stress_penalty = float(config.stress_weight) * max(0.0, stress_mid) * base
    return float(float(config.semantic_weight) * base + density_penalty + stress_penalty)


def _observer_switch_action(
    source_slice: np.ndarray,
    target_slice: np.ndarray,
    article_idx: int,
    *,
    simplex: Optional[np.ndarray],
    source_slice_pos: int,
    target_slice_pos: int,
    config: ObserverSliceTransportConfig,
) -> float:
    base = _point_distance(source_slice[int(article_idx)], target_slice[int(article_idx)])
    simplex_penalty = float(config.simplex_weight) * _simplex_l1(simplex, source_slice_pos, target_slice_pos)
    return float(float(config.observer_switch_weight) * base + simplex_penalty)


def observer_slice_commutator(
    slices: Mapping[str, Any],
    *,
    source_idx: int,
    target_idx: int,
    source_slice: str,
    target_slice: str,
    density: Any | None = None,
    stress: Any | None = None,
    slice_simplex: Any | None = None,
    config: ObserverSliceTransportConfig | None = None,
    row_to_article_index: Any | None = None,
) -> Dict[str, Any]:
    """Measure whether semantic movement and observer switching commute."""

    cfg = config or ObserverSliceTransportConfig()
    slice_arrays = _coerce_slices(slices)
    names = list(slice_arrays)
    if source_slice not in slice_arrays or target_slice not in slice_arrays:
        raise ValueError(f"source_slice={source_slice!r} and target_slice={target_slice!r} must be present")
    n_items = int(next(iter(slice_arrays.values())).shape[0])
    source_idx = int(source_idx)
    target_idx = int(target_idx)
    if not (0 <= source_idx < n_items and 0 <= target_idx < n_items):
        raise ValueError(f"source/target out of bounds for n_items={n_items}")
    density_vec = _coerce_vector(density, n_items, default=1.0)
    stress_vec = _coerce_vector(stress, n_items, default=0.0)
    simplex = _coerce_simplex(slice_simplex, len(names))
    source_pos = names.index(source_slice)
    target_pos = names.index(target_slice)
    src_chart = slice_arrays[source_slice]
    tgt_chart = slice_arrays[target_slice]

    semantic_first_semantic = _semantic_action(
        src_chart,
        source_idx,
        target_idx,
        density=density_vec,
        stress=stress_vec,
        config=cfg,
    )
    semantic_first_switch = _observer_switch_action(
        src_chart,
        tgt_chart,
        target_idx,
        simplex=simplex,
        source_slice_pos=source_pos,
        target_slice_pos=target_pos,
        config=cfg,
    )
    observer_first_switch = _observer_switch_action(
        src_chart,
        tgt_chart,
        source_idx,
        simplex=simplex,
        source_slice_pos=source_pos,
        target_slice_pos=target_pos,
        config=cfg,
    )
    observer_first_semantic = _semantic_action(
        tgt_chart,
        source_idx,
        target_idx,
        density=density_vec,
        stress=stress_vec,
        config=cfg,
    )
    semantic_first_action = float(semantic_first_semantic + semantic_first_switch)
    observer_first_action = float(observer_first_switch + observer_first_semantic)
    commutator_gap = float(semantic_first_action - observer_first_action)
    holonomy_action = float(abs(commutator_gap))
    route_min = max(min(semantic_first_action, observer_first_action), 1e-12)
    source_article_idx = _article_idx_for_row(source_idx, row_to_article_index)
    target_article_idx = _article_idx_for_row(target_idx, row_to_article_index)
    return _json_safe(
        {
            "diagnostic_type": "observer_slice_transport_commutator",
            "source_idx": source_idx,
            "target_idx": target_idx,
            "source_row_index": source_idx,
            "target_row_index": target_idx,
            "source_article_idx": source_article_idx,
            "target_article_idx": target_article_idx,
            "source_slice": source_slice,
            "target_slice": target_slice,
            "semantic_first_action": semantic_first_action,
            "observer_first_action": observer_first_action,
            "commutator_gap": commutator_gap,
            "holonomy_action": holonomy_action,
            "relative_holonomy": float(holonomy_action / route_min),
            "semantic_first_components": {
                "semantic_move": semantic_first_semantic,
                "observer_switch": semantic_first_switch,
            },
            "observer_first_components": {
                "observer_switch": observer_first_switch,
                "semantic_move": observer_first_semantic,
            },
            "closed_loop": [
                _loop_corner(source_idx, source_article_idx, source_slice, row_to_article_index),
                _loop_corner(target_idx, target_article_idx, source_slice, row_to_article_index),
                _loop_corner(target_idx, target_article_idx, target_slice, row_to_article_index),
                _loop_corner(source_idx, source_article_idx, target_slice, row_to_article_index),
                _loop_corner(source_idx, source_article_idx, source_slice, row_to_article_index),
            ],
            "config": {
                "semantic_weight": float(cfg.semantic_weight),
                "observer_switch_weight": float(cfg.observer_switch_weight),
                "stress_weight": float(cfg.stress_weight),
                "density_weight": float(cfg.density_weight),
                "simplex_weight": float(cfg.simplex_weight),
            },
        }
    )


def summarize_observer_slice_transport(
    slices: Mapping[str, Any],
    *,
    article_pairs: Sequence[Tuple[int, int]],
    slice_pairs: Sequence[Tuple[str, str]] | None = None,
    density: Any | None = None,
    stress: Any | None = None,
    slice_simplex: Any | None = None,
    null_slices: Mapping[str, Any] | None = None,
    config: ObserverSliceTransportConfig | None = None,
    row_to_article_index: Any | None = None,
) -> Dict[str, Any]:
    """Summarize commutator/holonomy records over article and slice pairs."""

    slice_arrays = _coerce_slices(slices)
    names = list(slice_arrays)
    if slice_pairs is None:
        slice_pairs = [(left, right) for left in names for right in names if left != right]
    records: List[Dict[str, Any]] = []
    null_records: List[Dict[str, Any]] = []
    for source_idx, target_idx in article_pairs:
        for source_slice, target_slice in slice_pairs:
            records.append(
                observer_slice_commutator(
                    slice_arrays,
                    source_idx=int(source_idx),
                    target_idx=int(target_idx),
                    source_slice=str(source_slice),
                    target_slice=str(target_slice),
                    density=density,
                    stress=stress,
                    slice_simplex=slice_simplex,
                    config=config,
                    row_to_article_index=row_to_article_index,
                )
            )
            if null_slices is not None:
                null_records.append(
                    observer_slice_commutator(
                        null_slices,
                        source_idx=int(source_idx),
                        target_idx=int(target_idx),
                        source_slice=str(source_slice),
                        target_slice=str(target_slice),
                        density=density,
                        stress=stress,
                        slice_simplex=slice_simplex,
                        config=config,
                        row_to_article_index=row_to_article_index,
                    )
                )

    holonomies = np.asarray([float(row["holonomy_action"]) for row in records], dtype=np.float64)
    null_holonomies = np.asarray([float(row["holonomy_action"]) for row in null_records], dtype=np.float64)
    null_mean = float(np.mean(null_holonomies)) if null_holonomies.size else None
    mean_holonomy = float(np.mean(holonomies)) if holonomies.size else None
    excess = None
    if mean_holonomy is not None and null_mean is not None:
        excess = float(mean_holonomy - null_mean)
    return _json_safe(
        {
            "summary_type": "observer_slice_transport_summary",
            "status": "OK" if records else "NO_RECORDS",
            "slice_count": len(names),
            "article_pair_count": len(article_pairs),
            "slice_pair_count": len(slice_pairs),
            "record_count": len(records),
            "row_to_article_index": list(row_to_article_index) if isinstance(row_to_article_index, (list, tuple)) else row_to_article_index,
            "mean_holonomy_action": mean_holonomy,
            "max_holonomy_action": float(np.max(holonomies)) if holonomies.size else None,
            "mean_null_holonomy_action": null_mean,
            "mean_excess_holonomy_action": excess,
            "records": records,
            "null_records": null_records,
        }
    )


def write_observer_slice_transport_summary(summary: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "observer_slice_transport_summary.json"
    path.write_text(json.dumps(_json_safe(dict(summary)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": str(path)}
