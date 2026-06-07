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


def _availability_thresholds(
    action_matrix: np.ndarray,
    *,
    availability_quantile: float,
    availability_action_cutoff: Optional[float],
) -> np.ndarray:
    if availability_action_cutoff is not None:
        if float(availability_action_cutoff) < 0.0:
            raise ValueError("availability_action_cutoff must be nonnegative")
        return np.full(action_matrix.shape[1], float(availability_action_cutoff), dtype=np.float64)
    if not 0.0 <= float(availability_quantile) <= 1.0:
        raise ValueError("availability_quantile must be in [0, 1]")
    q = min(max(float(availability_quantile), 0.0), 1.0)
    return np.quantile(action_matrix, q, axis=0).astype(np.float64)


def _edge_cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if abs(mean) <= 1e-12:
        return 0.0
    return float(np.std(values) / abs(mean))


def _classify_path(
    *,
    support_count: int,
    observer_count: int,
    edge_action_cv: float,
    consensus_fraction: float,
    stable_cv_threshold: float,
) -> str:
    if support_count <= 0:
        return "universal_barrier"
    consensus_cutoff = max(1, int(math.ceil(float(consensus_fraction) * int(observer_count))))
    if support_count >= consensus_cutoff and float(edge_action_cv) <= float(stable_cv_threshold):
        return "consensus_path"
    if support_count >= consensus_cutoff:
        return "consensus_but_warped"
    return "observer_contingent_path"


def _is_consensus_class(path_class: str) -> bool:
    return path_class in {"consensus_path", "consensus_but_warped"}


def _removal_effect(
    *,
    base_class: str,
    removed_class: str,
    base_support_count: int,
    removed_support_count: int,
    base_support_fraction: float,
    removed_support_fraction: float,
) -> str:
    if base_support_count > 0 and removed_support_count <= 0:
        return "removes_last_path"
    if not _is_consensus_class(base_class) and _is_consensus_class(removed_class):
        return "unblocks_consensus"
    if _is_consensus_class(base_class) and not _is_consensus_class(removed_class):
        return "breaks_consensus"
    if removed_class != base_class:
        return "changes_path_class"
    if removed_support_fraction > base_support_fraction + 1e-12:
        return "removes_blocker"
    if removed_support_fraction < base_support_fraction - 1e-12:
        return "removes_enabler"
    return "neutral"


def _null_path_stats(
    action_row: np.ndarray,
    thresholds: np.ndarray,
    *,
    consensus_fraction: float,
    stable_cv_threshold: float,
) -> Dict[str, Any]:
    observer_count = int(action_row.shape[0])
    if observer_count <= 1:
        shifts = [0]
    else:
        shifts = list(range(1, observer_count))
    class_counts: Dict[str, int] = {}
    support_fractions = []
    subset_required = 0
    contingent = 0
    for shift in shifts:
        shifted = np.roll(action_row, int(shift))
        row = shifted <= thresholds
        support_count = int(np.sum(row))
        support_fractions.append(float(support_count / observer_count))
        path_class = _classify_path(
            support_count=support_count,
            observer_count=observer_count,
            edge_action_cv=_edge_cv(shifted),
            consensus_fraction=consensus_fraction,
            stable_cv_threshold=stable_cv_threshold,
        )
        class_counts[path_class] = class_counts.get(path_class, 0) + 1
        if path_class == "observer_contingent_path":
            contingent += 1
        if 0 < support_count < observer_count:
            subset_required += 1
    denom = max(1, len(shifts))
    modal_class = sorted(class_counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0] if class_counts else "universal_barrier"
    return {
        "null_shift_count": int(len(shifts)),
        "null_path_class": modal_class,
        "null_path_class_counts": class_counts,
        "null_observer_contingent_probability": float(contingent / denom),
        "null_subset_required_probability": float(subset_required / denom),
        "null_support_fraction": float(np.mean(support_fractions)) if support_fractions else 0.0,
    }


def summarize_observer_path_contingency(
    slices: Mapping[str, Any],
    *,
    article_pairs: Sequence[Tuple[int, int]],
    density: Any | None = None,
    stress: Any | None = None,
    config: ObserverSliceTransportConfig | None = None,
    row_to_article_index: Any | None = None,
    availability_quantile: float = 0.35,
    availability_action_cutoff: Optional[float] = None,
    consensus_fraction: float = 0.75,
    stable_cv_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Classify paths by which observers make them traversable.

    This is intentionally not an optimizer.  It preserves a Track 4 nuance:
    some article-to-article moves only exist as low-action paths inside a subset
    of observer slices.  Those failures are semantic evidence, not just noise.
    """

    cfg = config or ObserverSliceTransportConfig()
    if not 0.0 < float(consensus_fraction) <= 1.0:
        raise ValueError("consensus_fraction must be in (0, 1]")
    if float(stable_cv_threshold) < 0.0:
        raise ValueError("stable_cv_threshold must be nonnegative")
    slice_arrays = _coerce_slices(slices)
    names = list(slice_arrays)
    n_items = int(next(iter(slice_arrays.values())).shape[0])
    density_vec = _coerce_vector(density, n_items, default=1.0)
    stress_vec = _coerce_vector(stress, n_items, default=0.0)
    pairs = [(int(left), int(right)) for left, right in article_pairs if int(left) != int(right)]
    if not pairs:
        empty_counts = {
            "consensus_path": 0,
            "consensus_but_warped": 0,
            "observer_contingent_path": 0,
            "universal_barrier": 0,
        }
        return _json_safe(
            {
                "summary_type": "observer_path_contingency_summary",
                "status": "NO_RECORDS",
                "slice_count": len(names),
                "article_pair_count": 0,
                "record_count": 0,
                "availability_quantile": float(availability_quantile),
                "availability_action_cutoff": availability_action_cutoff,
                "consensus_fraction": float(consensus_fraction),
                "stable_cv_threshold": float(stable_cv_threshold),
                "observer_names": names,
                "observer_thresholds": {name: None for name in names},
                "path_class_counts": dict(empty_counts),
                "null_path_class_counts": dict(empty_counts),
                "observer_contingent_rate": 0.0,
                "null_observer_contingent_rate": 0.0,
                "excess_observer_contingent_rate": 0.0,
                "observer_subset_required_count": 0,
                "observer_subset_required_rate": 0.0,
                "null_observer_subset_required_rate": 0.0,
                "excess_observer_subset_required_rate": 0.0,
                "mean_support_fraction": None,
                "mean_null_support_fraction": None,
                "mean_edge_action_cv": None,
                "observer_enabler_counts": {name: 0 for name in names},
                "observer_blocker_counts": {name: 0 for name in names},
                "leave_one_out_observer_effects": {},
                "top_consensus_unblockers": [],
                "top_consensus_dependencies": [],
                "top_gatekeeping_paths": [],
                "records": [],
            }
        )
    actions = np.zeros((len(pairs), len(names)), dtype=np.float64)
    for pair_idx, (source_idx, target_idx) in enumerate(pairs):
        if not (0 <= source_idx < n_items and 0 <= target_idx < n_items):
            raise ValueError(f"source/target out of bounds for n_items={n_items}")
        for slice_idx, name in enumerate(names):
            actions[pair_idx, slice_idx] = _semantic_action(
                slice_arrays[name],
                source_idx,
                target_idx,
                density=density_vec,
                stress=stress_vec,
                config=cfg,
            )
    thresholds = _availability_thresholds(
        actions,
        availability_quantile=availability_quantile,
        availability_action_cutoff=availability_action_cutoff,
    )
    available = actions <= thresholds.reshape(1, -1)
    records: List[Dict[str, Any]] = []
    null_class_counts: Dict[str, float] = {}
    enabler_counts = {name: 0 for name in names}
    blocker_counts = {name: 0 for name in names}
    removal_effect_counts = {
        name: {
            "unblocks_consensus": 0,
            "breaks_consensus": 0,
            "removes_last_path": 0,
            "changes_path_class": 0,
            "removes_blocker": 0,
            "removes_enabler": 0,
            "neutral": 0,
        }
        for name in names
    }
    class_counts: Dict[str, int] = {}
    subset_required_count = 0
    null_subset_required_total = 0.0
    null_contingent_total = 0.0
    for pair_idx, (source_idx, target_idx) in enumerate(pairs):
        row = available[pair_idx]
        enabled = [names[idx] for idx, flag in enumerate(row.tolist()) if bool(flag)]
        blocked = [names[idx] for idx, flag in enumerate(row.tolist()) if not bool(flag)]
        for name in enabled:
            enabler_counts[name] += 1
        for name in blocked:
            blocker_counts[name] += 1
        support_count = len(enabled)
        support_fraction = float(support_count / len(names))
        cv = _edge_cv(actions[pair_idx])
        path_class = _classify_path(
            support_count=support_count,
            observer_count=len(names),
            edge_action_cv=cv,
            consensus_fraction=consensus_fraction,
            stable_cv_threshold=stable_cv_threshold,
        )
        class_counts[path_class] = class_counts.get(path_class, 0) + 1
        null_stats = _null_path_stats(
            actions[pair_idx],
            thresholds,
            consensus_fraction=consensus_fraction,
            stable_cv_threshold=stable_cv_threshold,
        )
        for class_name, count in (null_stats.get("null_path_class_counts") or {}).items():
            null_class_counts[str(class_name)] = null_class_counts.get(str(class_name), 0.0) + float(count) / float(max(1, null_stats["null_shift_count"]))
        if 0 < support_count < len(names):
            subset_required_count += 1
        null_subset_required_total += float(null_stats["null_subset_required_probability"])
        null_contingent_total += float(null_stats["null_observer_contingent_probability"])
        source_article_idx = _article_idx_for_row(source_idx, row_to_article_index)
        target_article_idx = _article_idx_for_row(target_idx, row_to_article_index)
        leave_one_out = []
        for observer_idx, observer_name in enumerate(names):
            remaining_actions = np.delete(actions[pair_idx], observer_idx)
            removed_support_count = int(support_count - (1 if bool(row[observer_idx]) else 0))
            remaining_count = max(1, len(names) - 1)
            removed_support_fraction = float(removed_support_count / remaining_count)
            removed_cv = _edge_cv(remaining_actions)
            removed_class = _classify_path(
                support_count=removed_support_count,
                observer_count=remaining_count,
                edge_action_cv=removed_cv,
                consensus_fraction=consensus_fraction,
                stable_cv_threshold=stable_cv_threshold,
            )
            effect = _removal_effect(
                base_class=path_class,
                removed_class=removed_class,
                base_support_count=support_count,
                removed_support_count=removed_support_count,
                base_support_fraction=support_fraction,
                removed_support_fraction=removed_support_fraction,
            )
            removal_effect_counts[observer_name][effect] += 1
            leave_one_out.append(
                {
                    "removed_observer": observer_name,
                    "observer_was_enabler": bool(row[observer_idx]),
                    "path_class_without_observer": removed_class,
                    "support_count_without_observer": int(removed_support_count),
                    "support_fraction_without_observer": removed_support_fraction,
                    "edge_action_cv_without_observer": removed_cv,
                    "removal_effect": effect,
                }
            )
        records.append(
            {
                "diagnostic_type": "observer_path_contingency_record",
                "source_idx": int(source_idx),
                "target_idx": int(target_idx),
                "source_row_index": int(source_idx),
                "target_row_index": int(target_idx),
                "source_article_idx": int(source_article_idx),
                "target_article_idx": int(target_article_idx),
                "path_class": path_class,
                "support_count": int(support_count),
                "support_fraction": support_fraction,
                "edge_action_cv": cv,
                "enabled_by_observers": enabled,
                "blocked_by_observers": blocked,
                "observer_actions": {name: float(actions[pair_idx, idx]) for idx, name in enumerate(names)},
                "observer_thresholds": {name: float(thresholds[idx]) for idx, name in enumerate(names)},
                "path_exists_for_some_observers": bool(support_count > 0),
                "path_requires_observer_subset": bool(0 < support_count < len(names)),
                "observer_gate_score": float(1.0 - support_fraction),
                "null_path_class": null_stats["null_path_class"],
                "null_shift_count": int(null_stats["null_shift_count"]),
                "null_observer_contingent_probability": float(null_stats["null_observer_contingent_probability"]),
                "null_subset_required_probability": float(null_stats["null_subset_required_probability"]),
                "null_support_fraction": float(null_stats["null_support_fraction"]),
                "leave_one_out": leave_one_out,
            }
        )
    contingent = class_counts.get("observer_contingent_path", 0)
    null_contingent = float(null_contingent_total)
    records_sorted = sorted(records, key=lambda row: (-float(row["observer_gate_score"]), -float(row["edge_action_cv"])))
    top_consensus_unblockers = sorted(
        (
            {"observer": name, "unblocks_consensus": counts["unblocks_consensus"]}
            for name, counts in removal_effect_counts.items()
        ),
        key=lambda row: (-int(row["unblocks_consensus"]), row["observer"]),
    )
    top_consensus_dependencies = sorted(
        (
            {"observer": name, "breaks_consensus": counts["breaks_consensus"], "removes_last_path": counts["removes_last_path"]}
            for name, counts in removal_effect_counts.items()
        ),
        key=lambda row: (-(int(row["breaks_consensus"]) + int(row["removes_last_path"])), row["observer"]),
    )
    return _json_safe(
        {
            "summary_type": "observer_path_contingency_summary",
            "status": "OK",
            "slice_count": len(names),
            "article_pair_count": len(pairs),
            "record_count": len(records),
            "availability_quantile": float(availability_quantile),
            "availability_action_cutoff": availability_action_cutoff,
            "consensus_fraction": float(consensus_fraction),
            "stable_cv_threshold": float(stable_cv_threshold),
            "observer_names": names,
            "observer_thresholds": {name: float(thresholds[idx]) for idx, name in enumerate(names)},
            "path_class_counts": class_counts,
            "null_path_class_counts": null_class_counts,
            "observer_contingent_rate": float(contingent / len(records)),
            "null_observer_contingent_rate": float(null_contingent / len(records)),
            "excess_observer_contingent_rate": float((contingent - null_contingent) / len(records)),
            "observer_subset_required_count": int(subset_required_count),
            "observer_subset_required_rate": float(subset_required_count / len(records)),
            "null_observer_subset_required_rate": float(null_subset_required_total / len(records)),
            "excess_observer_subset_required_rate": float((subset_required_count - null_subset_required_total) / len(records)),
            "mean_support_fraction": float(np.mean([row["support_fraction"] for row in records])),
            "mean_null_support_fraction": float(np.mean([row["null_support_fraction"] for row in records])),
            "mean_edge_action_cv": float(np.mean([row["edge_action_cv"] for row in records])),
            "observer_enabler_counts": enabler_counts,
            "observer_blocker_counts": blocker_counts,
            "leave_one_out_observer_effects": removal_effect_counts,
            "top_consensus_unblockers": top_consensus_unblockers,
            "top_consensus_dependencies": top_consensus_dependencies,
            "top_gatekeeping_paths": records_sorted[: min(25, len(records_sorted))],
            "records": records,
        }
    )


def write_observer_slice_transport_summary(summary: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "observer_slice_transport_summary.json"
    path.write_text(json.dumps(_json_safe(dict(summary)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": str(path)}
