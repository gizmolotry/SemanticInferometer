from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ablation_dag import build_cache_key, build_orchestration_contract
from core.observer_manifold import (
    load_observer_manifold_bundle,
    write_observer_manifold_bundle,
)
from core.observer_local_recompute import (
    LOCAL_RECOMPUTE_DEFAULT_VARIANT,
    LOCAL_RECOMPUTE_MODE,
    LOCAL_RECOMPUTE_VARIANTS,
    compute_local_observer_recenter_from_run,
    load_primary_observer_payload,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_recenter_meaning_probe"
SUMMARY_TYPE = "observer_scientific_meaning_probe"
CLAIM_SCOPE = "observer_recenter_scientific_meaning_diagnostic"
IDEOLOGICAL_LABEL_COLUMNS = ("perspective_tag", "perspective_type", "label", "bias", "affiliation")
PROVENANCE_LABEL_COLUMNS = ("source", "publication", "author")
DIAGNOSTIC_LABEL_COLUMNS = ("verdict", "zone")
LABEL_COLUMNS = IDEOLOGICAL_LABEL_COLUMNS + PROVENANCE_LABEL_COLUMNS + DIAGNOSTIC_LABEL_COLUMNS
DEFAULT_THRESHOLDS = {
    "min_nontranslation_shift": 0.25,
    "min_label_gap_gain": 0.05,
    "min_all_pairs_separation_gain": 0.05,
    "min_centroid_contraction_gain": 0.05,
    "min_silhouette_gain": 0.02,
    "min_cluster_recovery_gain": 0.02,
    "max_permutation_p_value": 0.05,
    "min_synthetic_label_gap_gain": 0.50,
    "min_heldout_correlation_gain": 0.10,
}
PERMUTATION_COUNT = 199


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    # Average ties deterministically.
    unique, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if len(unique) < len(arr):
        for group_idx, count in enumerate(counts):
            if count <= 1:
                continue
            members = np.where(inverse == group_idx)[0]
            ranks[members] = float(np.mean(ranks[members]))
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 3:
        return None
    xx = xx[mask] - float(np.mean(xx[mask]))
    yy = yy[mask] - float(np.mean(yy[mask]))
    denom = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    return float(np.dot(xx, yy) / denom) if denom > 1e-12 else None


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _comb2(value: int) -> float:
    value = int(value)
    return float(value * (value - 1) / 2) if value >= 2 else 0.0


def _xy(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    try:
        x = float(row.get("x", row.get("observer_x")))
        y = float(row.get("y", row.get("observer_y")))
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return np.asarray([x, y], dtype=np.float64)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _article_idx(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("idx", "index", "article_idx"):
        if key in row:
            try:
                return int(row[key])
            except Exception:
                return None
    return None


def _article_map(view_state: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    rows = view_state.get("articles") if isinstance(view_state.get("articles"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = _article_idx(row)
        if idx is not None:
            out[idx] = dict(row)
    return out


def _metadata_by_idx(run_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
    for candidate in (run_dir / "MONOLITH_DATA.csv", run_dir / "article_metadata.csv"):
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)
        if "index" not in df.columns:
            df = df.reset_index().rename(columns={"index": "index"})
        rows: Dict[int, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            rows[idx] = {str(k): row.get(k) for k in df.columns}
        return rows, str(candidate)
    json_path = run_dir / "article_metadata.json"
    if json_path.exists():
        payload = _load_json(json_path)
        raw_rows = payload.get("articles") or payload.get("rows") or payload.get("metadata")
        if isinstance(raw_rows, list):
            rows = {}
            for pos, row in enumerate(raw_rows):
                if not isinstance(row, dict):
                    continue
                idx = _article_idx(row)
                rows[int(idx if idx is not None else pos)] = dict(row)
            return rows, str(json_path)
    return {}, None


def _clean_label(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "unknown", "missing"}:
        return ""
    return text


def select_label_column(
    metadata: Mapping[int, Mapping[str, Any]],
    article_indices: Iterable[int],
    *,
    min_label_count: int,
    preferred_label_column: Optional[str] = None,
    label_mode: str = "auto",
) -> Tuple[Optional[str], Dict[str, int], Dict[str, Any]]:
    mode = str(label_mode or "auto").strip().lower()
    if mode not in {"auto", "ideological", "provenance", "diagnostic", "any"}:
        raise ValueError(f"Unsupported label_mode={label_mode!r}")
    best_col: Optional[str] = None
    best_counts: Dict[str, int] = {}
    best_score: Tuple[int, int, int] = (-1, -1, -1)
    indices = list(article_indices)
    if preferred_label_column:
        ordered_columns = (str(preferred_label_column),)
    elif mode == "ideological":
        ordered_columns = IDEOLOGICAL_LABEL_COLUMNS
    elif mode == "provenance":
        ordered_columns = PROVENANCE_LABEL_COLUMNS
    elif mode == "diagnostic":
        ordered_columns = DIAGNOSTIC_LABEL_COLUMNS
    elif mode == "any":
        ordered_columns = LABEL_COLUMNS
    else:
        ordered_columns = IDEOLOGICAL_LABEL_COLUMNS + PROVENANCE_LABEL_COLUMNS + DIAGNOSTIC_LABEL_COLUMNS
    candidates: List[Dict[str, Any]] = []
    for priority, col in enumerate(ordered_columns):
        labels = [_clean_label(metadata.get(idx, {}).get(col)) for idx in indices]
        counts = Counter(label for label in labels if label)
        repeated = sum(1 for count in counts.values() if count >= int(min_label_count))
        usable_count = sum(count for count in counts.values() if count >= int(min_label_count))
        basis = _label_basis_for_column(col)
        candidate = {
            "column": col,
            "basis": basis,
            "repeated_group_count": repeated,
            "usable_article_count": usable_count,
            "distinct_label_count": len(counts),
            "priority": priority,
        }
        candidates.append(candidate)
        # Repeated groups are the hard gate; usable rows break ties; lower priority wins last.
        score = (int(repeated), int(usable_count), -int(priority))
        if score > best_score:
            best_col = col if repeated > 0 else None
            best_counts = dict(sorted(counts.items()))
            best_score = score
    selected_basis = _label_basis_for_column(best_col) if best_col else "none"
    diagnostics = {
        "label_mode": mode,
        "preferred_label_column": preferred_label_column,
        "selected_column": best_col,
        "selected_basis": selected_basis,
        "candidate_columns": candidates,
        "semantic_label_basis": selected_basis == "ideological",
        "provenance_label_basis": selected_basis == "provenance",
    }
    return best_col, best_counts, diagnostics


def _label_basis_for_column(column: Optional[str]) -> str:
    if column in IDEOLOGICAL_LABEL_COLUMNS:
        return "ideological"
    if column in PROVENANCE_LABEL_COLUMNS:
        return "provenance"
    if column in DIAGNOSTIC_LABEL_COLUMNS:
        return "diagnostic"
    return "custom" if column else "none"


def _pairwise_distance_scale(points: np.ndarray) -> Optional[float]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    distances: List[float] = []
    for i in range(int(arr.shape[0])):
        for j in range(i + 1, int(arr.shape[0])):
            dist = float(np.linalg.norm(arr[i] - arr[j]))
            if math.isfinite(dist) and dist > 1e-12:
                distances.append(dist)
    if not distances:
        return None
    return float(np.median(distances))


def _anchor_label_gap(
    *,
    coords: Mapping[int, np.ndarray],
    anchor_idx: int,
    labels: Mapping[int, str],
) -> Dict[str, Any]:
    anchor = coords.get(anchor_idx)
    anchor_label = labels.get(anchor_idx)
    if anchor is None or not anchor_label:
        return {
            "status": "INSUFFICIENT_LABELS",
            "same_count": 0,
            "different_count": 0,
            "label_gap": None,
        }
    same: List[float] = []
    different: List[float] = []
    labeled_points: List[np.ndarray] = []
    for idx, coord in coords.items():
        if idx == anchor_idx or idx not in labels:
            continue
        labeled_points.append(np.asarray(coord, dtype=np.float64))
        dist = float(np.linalg.norm(coord - anchor))
        if labels[idx] == anchor_label:
            same.append(dist)
        else:
            different.append(dist)
    same_mean = _mean(same)
    diff_mean = _mean(different)
    gap = diff_mean - same_mean if same_mean is not None and diff_mean is not None else None
    scale = _pairwise_distance_scale(np.vstack(labeled_points)) if len(labeled_points) >= 2 else None
    normalized_gap = float(gap) / float(scale) if gap is not None and scale is not None and scale > 1e-12 else None
    return {
        "status": "OK" if gap is not None else "INSUFFICIENT_LABELS",
        "anchor_label": anchor_label,
        "same_count": len(same),
        "different_count": len(different),
        "same_mean_distance": same_mean,
        "different_mean_distance": diff_mean,
        "label_gap": gap,
        "distance_scale": scale,
        "scale_normalized_label_gap": normalized_gap,
    }


def _heldout_correlation(
    *,
    coords: Mapping[int, np.ndarray],
    anchor_idx: int,
    heldout_similarity: Mapping[int, float],
) -> Dict[str, Any]:
    anchor = coords.get(anchor_idx)
    if anchor is None:
        return {"status": "MISSING_ANCHOR", "spearman_closeness_vs_similarity": None}
    closeness: List[float] = []
    sim: List[float] = []
    for idx, coord in coords.items():
        if idx == anchor_idx or idx not in heldout_similarity:
            continue
        closeness.append(-float(np.linalg.norm(coord - anchor)))
        sim.append(float(heldout_similarity[idx]))
    corr = _spearman(closeness, sim)
    return {
        "status": "OK" if corr is not None else "INSUFFICIENT_HELDOUT",
        "n_compared": len(sim),
        "spearman_closeness_vs_similarity": corr,
    }


def evaluate_observer_recenter(
    *,
    global_articles: Mapping[int, Mapping[str, Any]],
    observer_articles: Mapping[int, Mapping[str, Any]],
    anchor_idx: int,
    labels: Optional[Mapping[int, str]] = None,
    heldout_similarity: Optional[Mapping[int, float]] = None,
) -> Dict[str, Any]:
    global_anchor = _xy(global_articles.get(anchor_idx, {}))
    observer_anchor = _xy(observer_articles.get(anchor_idx, {}))
    common = sorted(set(global_articles).intersection(observer_articles))
    translation_coords: Dict[int, np.ndarray] = {}
    observer_coords: Dict[int, np.ndarray] = {}
    shift_magnitudes: List[float] = []
    for idx in common:
        global_xy = _xy(global_articles[idx])
        observer_xy = _xy(observer_articles[idx])
        if global_xy is None or observer_xy is None or global_anchor is None:
            continue
        translation_xy = global_xy - global_anchor
        translation_coords[idx] = translation_xy
        observer_coords[idx] = observer_xy
        shift_magnitudes.append(float(np.linalg.norm(observer_xy - translation_xy)))

    translation_gap = _anchor_label_gap(coords=translation_coords, anchor_idx=anchor_idx, labels=labels or {})
    observer_gap = _anchor_label_gap(coords=observer_coords, anchor_idx=anchor_idx, labels=labels or {})
    label_gap_gain = None
    normalized_label_gap_gain = None
    if translation_gap.get("label_gap") is not None and observer_gap.get("label_gap") is not None:
        label_gap_gain = float(observer_gap["label_gap"]) - float(translation_gap["label_gap"])
    if (
        translation_gap.get("scale_normalized_label_gap") is not None
        and observer_gap.get("scale_normalized_label_gap") is not None
    ):
        normalized_label_gap_gain = float(observer_gap["scale_normalized_label_gap"]) - float(
            translation_gap["scale_normalized_label_gap"]
        )

    translation_heldout = _heldout_correlation(
        coords=translation_coords,
        anchor_idx=anchor_idx,
        heldout_similarity=heldout_similarity or {},
    )
    observer_heldout = _heldout_correlation(
        coords=observer_coords,
        anchor_idx=anchor_idx,
        heldout_similarity=heldout_similarity or {},
    )
    heldout_gain = None
    if (
        translation_heldout.get("spearman_closeness_vs_similarity") is not None
        and observer_heldout.get("spearman_closeness_vs_similarity") is not None
    ):
        heldout_gain = float(observer_heldout["spearman_closeness_vs_similarity"]) - float(
            translation_heldout["spearman_closeness_vs_similarity"]
        )

    return {
        "anchor_idx": int(anchor_idx),
        "common_article_count": len(common),
        "focus_xy_centered": bool(observer_anchor is not None and np.linalg.norm(observer_anchor) <= 1e-9),
        "nontranslation_shift_mean": _mean(shift_magnitudes),
        "nontranslation_shift_max": max(shift_magnitudes) if shift_magnitudes else None,
        "translation_null": translation_gap,
        "observer_recentered": observer_gap,
        "label_gap_gain_over_translation": label_gap_gain,
        "scale_normalized_label_gap_gain_over_translation": normalized_label_gap_gain,
        "translation_heldout": translation_heldout,
        "observer_heldout": observer_heldout,
        "heldout_correlation_gain_over_translation": heldout_gain,
    }


def _labeled_coordinate_arrays(
    *,
    global_articles: Mapping[int, Mapping[str, Any]],
    observer_articles: Mapping[int, Mapping[str, Any]],
    anchor_idx: int,
    labels: Mapping[int, str],
) -> Dict[str, Any]:
    global_anchor = _xy(global_articles.get(anchor_idx, {}))
    if global_anchor is None:
        return {"status": "MISSING_ANCHOR"}
    indices: List[int] = []
    label_values: List[str] = []
    translation_rows: List[np.ndarray] = []
    observer_rows: List[np.ndarray] = []
    for idx in sorted(set(global_articles).intersection(observer_articles)):
        label = _clean_label(labels.get(idx))
        if not label:
            continue
        global_xy = _xy(global_articles[idx])
        observer_xy = _xy(observer_articles[idx])
        if global_xy is None or observer_xy is None:
            continue
        indices.append(int(idx))
        label_values.append(label)
        translation_rows.append(global_xy - global_anchor)
        observer_rows.append(observer_xy)
    if len(indices) < 3 or len(set(label_values)) < 2:
        return {
            "status": "INSUFFICIENT_LABELS",
            "labeled_count": len(indices),
            "label_count": len(set(label_values)),
        }
    return {
        "status": "OK",
        "indices": indices,
        "labels": label_values,
        "translation": np.vstack(translation_rows).astype(np.float64),
        "observer": np.vstack(observer_rows).astype(np.float64),
    }


def _pairwise_label_stats(points: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    within: List[float] = []
    between: List[float] = []
    n_items = int(points.shape[0])
    for i in range(n_items):
        for j in range(i + 1, n_items):
            dist = float(np.linalg.norm(points[i] - points[j]))
            if labels[i] == labels[j]:
                within.append(dist)
            else:
                between.append(dist)
    within_mean = _mean(within)
    between_mean = _mean(between)
    separation = None
    if within_mean is not None and between_mean is not None:
        separation = float(between_mean - within_mean)
    scale = _pairwise_distance_scale(points)
    normalized_separation = (
        float(separation) / float(scale)
        if separation is not None and scale is not None and scale > 1e-12
        else None
    )
    return {
        "within_pair_count": len(within),
        "between_pair_count": len(between),
        "within_mean_distance": within_mean,
        "between_mean_distance": between_mean,
        "separation": separation,
        "distance_scale": scale,
        "scale_normalized_separation": normalized_separation,
    }


def _centroid_stats(points: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    label_values = sorted(set(labels))
    centroids: Dict[str, np.ndarray] = {
        label: points[[idx for idx, value in enumerate(labels) if value == label]].mean(axis=0)
        for label in label_values
    }
    within_distances: List[float] = []
    for idx, label in enumerate(labels):
        within_distances.append(float(np.linalg.norm(points[idx] - centroids[label])))
    centroid_distances: List[float] = []
    for i, left in enumerate(label_values):
        for right in label_values[i + 1 :]:
            centroid_distances.append(float(np.linalg.norm(centroids[left] - centroids[right])))
    within_mean = _mean(within_distances)
    between_mean = _mean(centroid_distances)
    ratio = None
    if within_mean is not None and between_mean is not None:
        ratio = float(between_mean / max(within_mean, 1e-12))
    return {
        "label_count": len(label_values),
        "within_centroid_mean_distance": within_mean,
        "between_centroid_mean_distance": between_mean,
        "between_over_within_ratio": ratio,
    }


def _silhouette_score(points: np.ndarray, labels: Sequence[str]) -> Optional[float]:
    label_values = sorted(set(labels))
    if len(label_values) < 2 or int(points.shape[0]) < 3:
        return None
    scores: List[float] = []
    for idx, label in enumerate(labels):
        same = [j for j, value in enumerate(labels) if value == label and j != idx]
        other_labels = [value for value in label_values if value != label]
        if not same or not other_labels:
            continue
        a = float(np.mean([np.linalg.norm(points[idx] - points[j]) for j in same]))
        b_values = []
        for other in other_labels:
            members = [j for j, value in enumerate(labels) if value == other]
            if members:
                b_values.append(float(np.mean([np.linalg.norm(points[idx] - points[j]) for j in members])))
        if not b_values:
            continue
        b = min(b_values)
        denom = max(a, b)
        if denom > 1e-12:
            scores.append(float((b - a) / denom))
    return _mean(scores)


def _deterministic_kmeans(points: np.ndarray, k: int, *, iterations: int = 50) -> Optional[np.ndarray]:
    n_items = int(points.shape[0])
    k = int(k)
    if k < 2 or n_items < k:
        return None
    center = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    centroid_indices = [first]
    while len(centroid_indices) < k:
        centroids = points[np.asarray(centroid_indices)]
        distances = np.min(np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2), axis=1)
        for used in centroid_indices:
            distances[used] = -1.0
        centroid_indices.append(int(np.argmax(distances)))
    centroids = points[np.asarray(centroid_indices)].copy()
    assignments = np.zeros(n_items, dtype=np.int64)
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        next_assignments = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster_idx in range(k):
            members = points[assignments == cluster_idx]
            if len(members):
                centroids[cluster_idx] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                centroids[cluster_idx] = points[farthest]
    return assignments


def _adjusted_rand_index(true_labels: Sequence[str], pred_labels: Sequence[int]) -> Optional[float]:
    if len(true_labels) != len(pred_labels) or len(true_labels) < 2:
        return None
    true_values = {label: idx for idx, label in enumerate(sorted(set(true_labels)))}
    pred_values = {label: idx for idx, label in enumerate(sorted(set(int(v) for v in pred_labels)))}
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for true, pred in zip(true_labels, pred_labels):
        contingency[true_values[true], pred_values[int(pred)]] += 1
    sum_comb = float(sum(_comb2(int(value)) for value in contingency.ravel()))
    row_comb = float(sum(_comb2(int(value)) for value in contingency.sum(axis=1)))
    col_comb = float(sum(_comb2(int(value)) for value in contingency.sum(axis=0)))
    total_comb = _comb2(len(true_labels))
    if total_comb <= 0:
        return None
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) <= 1e-12:
        return 1.0 if abs(sum_comb - expected) <= 1e-12 else 0.0
    return float((sum_comb - expected) / denom)


def _normalized_mutual_info(true_labels: Sequence[str], pred_labels: Sequence[int]) -> Optional[float]:
    if len(true_labels) != len(pred_labels) or len(true_labels) < 2:
        return None
    true_values = {label: idx for idx, label in enumerate(sorted(set(true_labels)))}
    pred_values = {label: idx for idx, label in enumerate(sorted(set(int(v) for v in pred_labels)))}
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.float64)
    for true, pred in zip(true_labels, pred_labels):
        contingency[true_values[true], pred_values[int(pred)]] += 1.0
    total = float(contingency.sum())
    if total <= 0:
        return None
    pxy = contingency / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += float(pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j])))
    hx = -float(sum(value * math.log(value) for value in px if value > 0))
    hy = -float(sum(value * math.log(value) for value in py if value > 0))
    denom = math.sqrt(hx * hy)
    return float(mi / denom) if denom > 1e-12 else None


def _cluster_recovery_stats(points: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    k = len(set(labels))
    pred = _deterministic_kmeans(points, k)
    if pred is None:
        return {"status": "INSUFFICIENT_LABELS", "ari": None, "nmi": None, "cluster_count": k}
    return {
        "status": "OK",
        "cluster_count": k,
        "ari": _adjusted_rand_index(labels, pred.tolist()),
        "nmi": _normalized_mutual_info(labels, pred.tolist()),
        "cluster_sizes": dict(sorted(Counter(int(value) for value in pred.tolist()).items())),
    }


def _permutation_separation_test(
    points: np.ndarray,
    labels: Sequence[str],
    *,
    permutations: int = PERMUTATION_COUNT,
    seed: int = 1729,
) -> Dict[str, Any]:
    observed_stats = _pairwise_label_stats(points, labels)
    observed = observed_stats.get("separation")
    if observed is None:
        return {"status": "INSUFFICIENT_LABELS", "p_value": None, "observed_separation": None}
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels, dtype=object)
    null_values: List[float] = []
    for _ in range(int(permutations)):
        shuffled = label_arr.copy()
        rng.shuffle(shuffled)
        sep = _pairwise_label_stats(points, shuffled.tolist()).get("separation")
        if sep is not None:
            null_values.append(float(sep))
    if not null_values:
        return {"status": "NO_NULL", "p_value": None, "observed_separation": observed}
    ge = sum(1 for value in null_values if value >= float(observed))
    p_value = float((ge + 1) / (len(null_values) + 1))
    return {
        "status": "OK",
        "permutations": len(null_values),
        "observed_separation": float(observed),
        "distance_scale": observed_stats.get("distance_scale"),
        "scale_normalized_observed_separation": observed_stats.get("scale_normalized_separation"),
        "null_mean_separation": float(np.mean(null_values)),
        "p_value": p_value,
    }


def evaluate_label_geometry_tests(
    *,
    global_articles: Mapping[int, Mapping[str, Any]],
    observer_articles: Mapping[int, Mapping[str, Any]],
    anchor_idx: int,
    labels: Mapping[int, str],
) -> Dict[str, Any]:
    arrays = _labeled_coordinate_arrays(
        global_articles=global_articles,
        observer_articles=observer_articles,
        anchor_idx=anchor_idx,
        labels=labels,
    )
    if arrays.get("status") != "OK":
        return {"status": arrays.get("status", "NO_DATA"), "tests": {}, **arrays}
    label_values = list(arrays["labels"])
    translation = np.asarray(arrays["translation"], dtype=np.float64)
    observer = np.asarray(arrays["observer"], dtype=np.float64)

    translation_pairs = _pairwise_label_stats(translation, label_values)
    observer_pairs = _pairwise_label_stats(observer, label_values)
    all_pairs_gain = None
    all_pairs_normalized_gain = None
    if translation_pairs.get("separation") is not None and observer_pairs.get("separation") is not None:
        all_pairs_gain = float(observer_pairs["separation"]) - float(translation_pairs["separation"])
    if (
        translation_pairs.get("scale_normalized_separation") is not None
        and observer_pairs.get("scale_normalized_separation") is not None
    ):
        all_pairs_normalized_gain = float(observer_pairs["scale_normalized_separation"]) - float(
            translation_pairs["scale_normalized_separation"]
        )

    translation_centroid = _centroid_stats(translation, label_values)
    observer_centroid = _centroid_stats(observer, label_values)
    centroid_contraction_gain = None
    centroid_ratio_gain = None
    if (
        translation_centroid.get("within_centroid_mean_distance") is not None
        and observer_centroid.get("within_centroid_mean_distance") is not None
    ):
        centroid_contraction_gain = float(translation_centroid["within_centroid_mean_distance"]) - float(
            observer_centroid["within_centroid_mean_distance"]
        )
    if (
        translation_centroid.get("between_over_within_ratio") is not None
        and observer_centroid.get("between_over_within_ratio") is not None
    ):
        centroid_ratio_gain = float(observer_centroid["between_over_within_ratio"]) - float(
            translation_centroid["between_over_within_ratio"]
        )

    translation_silhouette = _silhouette_score(translation, label_values)
    observer_silhouette = _silhouette_score(observer, label_values)
    silhouette_gain = None
    if translation_silhouette is not None and observer_silhouette is not None:
        silhouette_gain = float(observer_silhouette - translation_silhouette)

    translation_cluster = _cluster_recovery_stats(translation, label_values)
    observer_cluster = _cluster_recovery_stats(observer, label_values)
    nmi_gain = None
    ari_gain = None
    if translation_cluster.get("nmi") is not None and observer_cluster.get("nmi") is not None:
        nmi_gain = float(observer_cluster["nmi"]) - float(translation_cluster["nmi"])
    if translation_cluster.get("ari") is not None and observer_cluster.get("ari") is not None:
        ari_gain = float(observer_cluster["ari"]) - float(translation_cluster["ari"])

    translation_perm = _permutation_separation_test(translation, label_values)
    observer_perm = _permutation_separation_test(observer, label_values)
    permutation_gain = None
    permutation_normalized_gain = None
    if translation_perm.get("observed_separation") is not None and observer_perm.get("observed_separation") is not None:
        permutation_gain = float(observer_perm["observed_separation"]) - float(translation_perm["observed_separation"])
    if (
        translation_perm.get("scale_normalized_observed_separation") is not None
        and observer_perm.get("scale_normalized_observed_separation") is not None
    ):
        permutation_normalized_gain = float(observer_perm["scale_normalized_observed_separation"]) - float(
            translation_perm["scale_normalized_observed_separation"]
        )

    tests = {
        "all_pairs_separation": {
            "status": "OK" if all_pairs_gain is not None else "NO_DATA",
            "pass": bool(
                all_pairs_normalized_gain is not None
                and all_pairs_normalized_gain >= DEFAULT_THRESHOLDS["min_all_pairs_separation_gain"]
            ),
            "separation_gain_over_translation": all_pairs_gain,
            "scale_normalized_separation_gain_over_translation": all_pairs_normalized_gain,
            "metric_basis": "scale_normalized_separation",
            "min_gain": DEFAULT_THRESHOLDS["min_all_pairs_separation_gain"],
            "translation": translation_pairs,
            "observer": observer_pairs,
        },
        "centroid_contraction": {
            "status": "OK" if centroid_contraction_gain is not None else "NO_DATA",
            "pass": bool(
                centroid_contraction_gain is not None
                and centroid_contraction_gain >= DEFAULT_THRESHOLDS["min_centroid_contraction_gain"]
            ),
            "within_centroid_contraction_gain": centroid_contraction_gain,
            "between_over_within_ratio_gain": centroid_ratio_gain,
            "min_gain": DEFAULT_THRESHOLDS["min_centroid_contraction_gain"],
            "translation": translation_centroid,
            "observer": observer_centroid,
        },
        "silhouette": {
            "status": "OK" if silhouette_gain is not None else "NO_DATA",
            "pass": bool(
                silhouette_gain is not None
                and silhouette_gain >= DEFAULT_THRESHOLDS["min_silhouette_gain"]
            ),
            "silhouette_gain_over_translation": silhouette_gain,
            "min_gain": DEFAULT_THRESHOLDS["min_silhouette_gain"],
            "translation_silhouette": translation_silhouette,
            "observer_silhouette": observer_silhouette,
        },
        "ari_nmi": {
            "status": "OK" if nmi_gain is not None and ari_gain is not None else "NO_DATA",
            "pass": bool(
                (
                    nmi_gain is not None
                    and nmi_gain >= DEFAULT_THRESHOLDS["min_cluster_recovery_gain"]
                )
                or (
                    ari_gain is not None
                    and ari_gain >= DEFAULT_THRESHOLDS["min_cluster_recovery_gain"]
                )
            ),
            "min_gain": DEFAULT_THRESHOLDS["min_cluster_recovery_gain"],
            "nmi_gain_over_translation": nmi_gain,
            "ari_gain_over_translation": ari_gain,
            "translation": translation_cluster,
            "observer": observer_cluster,
        },
        "permutation_within_between": {
            "status": "OK" if observer_perm.get("status") == "OK" else observer_perm.get("status"),
            "pass": bool(
                observer_perm.get("p_value") is not None
                and float(observer_perm["p_value"]) <= DEFAULT_THRESHOLDS["max_permutation_p_value"]
                and permutation_normalized_gain is not None
                and permutation_normalized_gain >= DEFAULT_THRESHOLDS["min_all_pairs_separation_gain"]
            ),
            "max_p_value": DEFAULT_THRESHOLDS["max_permutation_p_value"],
            "separation_gain_over_translation": permutation_gain,
            "scale_normalized_separation_gain_over_translation": permutation_normalized_gain,
            "metric_basis": "scale_normalized_separation",
            "translation": translation_perm,
            "observer": observer_perm,
        },
    }
    return {
        "status": "PASS" if all(test.get("pass") for test in tests.values()) else "FAIL",
        "labeled_count": len(label_values),
        "label_count": len(set(label_values)),
        "label_counts": dict(sorted(Counter(label_values).items())),
        "tests": tests,
    }


def aggregate_ideological_validation_suite(
    observer_rows: Sequence[Mapping[str, Any]],
    *,
    label_column: Optional[str],
    label_basis: str,
    label_counts: Mapping[str, int],
    min_label_count: int,
) -> Dict[str, Any]:
    usable_label_counts = {
        str(label): int(count)
        for label, count in label_counts.items()
        if int(count) >= int(min_label_count)
    }
    anchor_labels = []
    observer_metrics = []
    for row in observer_rows:
        anchor_label = ((row.get("observer_recentered") or {}).get("anchor_label")) or ""
        if anchor_label:
            anchor_labels.append(str(anchor_label))
        metrics = row.get("label_geometry_tests")
        if isinstance(metrics, dict) and metrics.get("status") in {"PASS", "FAIL"}:
            observer_metrics.append(metrics)
    test_names = [
        "all_pairs_separation",
        "centroid_contraction",
        "silhouette",
        "ari_nmi",
        "permutation_within_between",
    ]
    aggregated_tests: Dict[str, Any] = {}
    for test_name in test_names:
        rows = [
            (metrics.get("tests") or {}).get(test_name)
            for metrics in observer_metrics
            if isinstance((metrics.get("tests") or {}).get(test_name), dict)
        ]
        pass_values = [bool(row.get("pass")) for row in rows]
        aggregated_tests[test_name] = {
            "status": "OK" if rows else "NO_DATA",
            "observer_count": len(rows),
            "pass_count": sum(1 for value in pass_values if value),
            "pass": bool(rows and all(pass_values)),
            "rows": rows,
        }
    support_contracts = {
        "ideological_label_basis": {
            "pass": label_basis == "ideological",
            "label_column": label_column,
            "label_basis": label_basis,
        },
        "balanced_label_packet": {
            "pass": bool(
                usable_label_counts
                and len(set(usable_label_counts.values())) == 1
                and len(usable_label_counts) >= 2
            ),
            "usable_label_counts": usable_label_counts,
            "min_label_count": int(min_label_count),
        },
        "multi_anchor_coverage": {
            "pass": bool(
                usable_label_counts
                and set(usable_label_counts).issubset(set(anchor_labels))
            ),
            "covered_anchor_labels": sorted(set(anchor_labels)),
            "missing_anchor_labels": sorted(set(usable_label_counts) - set(anchor_labels)),
            "observer_count": len(observer_rows),
        },
    }
    return {
        "status": "PASS" if all(row["pass"] for row in aggregated_tests.values()) else "FAIL",
        "suite_type": "ideological_label_geometry_validation",
        "label_column": label_column,
        "label_basis": label_basis,
        "support_contracts": support_contracts,
        "tests": aggregated_tests,
    }


def evaluate_property_theft_homotopy() -> Dict[str, Any]:
    global_articles = {
        0: {"idx": 0, "name": "property", "x": 0.0, "y": 0.0},
        1: {"idx": 1, "name": "theft", "x": 1.2, "y": 0.0},
        2: {"idx": 2, "name": "contract", "x": 0.9, "y": 0.1},
        3: {"idx": 3, "name": "exploitation", "x": 1.1, "y": -0.1},
        4: {"idx": 4, "name": "market", "x": 0.8, "y": 0.3},
    }
    anarchist = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 0.18, "y": 0.03},
        2: {"idx": 2, "x": 1.8, "y": 0.2},
        3: {"idx": 3, "x": 0.35, "y": -0.02},
        4: {"idx": 4, "x": 2.0, "y": 0.4},
    }
    liberal = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 1.9, "y": 0.3},
        2: {"idx": 2, "x": 0.22, "y": 0.02},
        3: {"idx": 3, "x": 1.7, "y": -0.2},
        4: {"idx": 4, "x": 0.36, "y": 0.03},
    }

    def dist(view: Mapping[int, Mapping[str, Any]], a: int, b: int) -> float:
        aa = _xy(view[a])
        bb = _xy(view[b])
        assert aa is not None and bb is not None
        return float(np.linalg.norm(aa - bb))

    anarchist_property_theft = dist(anarchist, 0, 1)
    liberal_property_theft = dist(liberal, 0, 1)
    anarchist_property_contract = dist(anarchist, 0, 2)
    liberal_property_contract = dist(liberal, 0, 2)
    pass_checks = {
        "property_theft_lower_under_anarchist": anarchist_property_theft < liberal_property_theft * 0.5,
        "property_contract_lower_under_liberal": liberal_property_contract < anarchist_property_contract * 0.5,
    }
    return {
        "status": "PASS" if all(pass_checks.values()) else "FAIL",
        "global_articles": list(global_articles.values()),
        "distances": {
            "anarchist_property_theft": anarchist_property_theft,
            "liberal_property_theft": liberal_property_theft,
            "anarchist_property_contract": anarchist_property_contract,
            "liberal_property_contract": liberal_property_contract,
        },
        "pass_checks": pass_checks,
    }


def synthetic_fixture_payload() -> Dict[str, Any]:
    global_articles = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 1.05, "y": 0.10},
        2: {"idx": 2, "x": 0.95, "y": -0.15},
        3: {"idx": 3, "x": 1.20, "y": 0.30},
        4: {"idx": 4, "x": 1.15, "y": -0.35},
        5: {"idx": 5, "x": 0.85, "y": 0.40},
        6: {"idx": 6, "x": 1.30, "y": -0.20},
    }
    observer_articles = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 0.16, "y": 0.05},
        2: {"idx": 2, "x": 0.20, "y": -0.08},
        3: {"idx": 3, "x": 2.40, "y": 0.50},
        4: {"idx": 4, "x": 2.10, "y": -0.65},
        5: {"idx": 5, "x": 2.70, "y": 0.95},
        6: {"idx": 6, "x": 2.85, "y": -0.25},
    }
    labels = {0: "frame_a", 1: "frame_a", 2: "frame_a", 3: "frame_b", 4: "frame_b", 5: "frame_c", 6: "frame_c"}
    heldout = {0: 1.0, 1: 0.93, 2: 0.89, 3: 0.24, 4: 0.31, 5: 0.10, 6: 0.16}
    recenter = evaluate_observer_recenter(
        global_articles=global_articles,
        observer_articles=observer_articles,
        anchor_idx=0,
        labels=labels,
        heldout_similarity=heldout,
    )
    pass_checks = {
        "focus_centered": bool(recenter.get("focus_xy_centered")),
        "nontranslation": float(recenter.get("nontranslation_shift_mean") or 0.0) > 0.25,
        "label_gain": float(recenter.get("label_gap_gain_over_translation") or 0.0) > 0.50,
        "heldout_gain": float(recenter.get("heldout_correlation_gain_over_translation") or 0.0) > 0.10,
    }
    return {
        "status": "PASS" if all(pass_checks.values()) else "FAIL",
        "pass_checks": pass_checks,
        "recenter": recenter,
        "property_theft_homotopy": evaluate_property_theft_homotopy(),
    }


def terrain_regime_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        zone = _clean_label(row.get("zone"))
        if zone:
            grouped.setdefault(zone, []).append(row)

    zone_metrics: Dict[str, Any] = {}
    for zone, zone_rows in sorted(grouped.items()):
        zone_metrics[zone] = {
            "count": len(zone_rows),
            "density_mean": _mean([float(r.get("density")) for r in zone_rows if str(r.get("density")) != "nan"]),
            "stress_mean": _mean([float(r.get("stress")) for r in zone_rows if str(r.get("stress")) != "nan"]),
            "work_mean": _mean([float(r.get("w_actual")) for r in zone_rows if str(r.get("w_actual")) != "nan"]),
        }

    bridge = zone_metrics.get("Bridge", {})
    void = zone_metrics.get("Void", {})
    tightrope = zone_metrics.get("Tightrope", {})
    pass_checks = {
        "bridge_and_void_present": bool(bridge and void),
        "bridge_density_ge_void_density": (
            bridge.get("density_mean") is not None
            and void.get("density_mean") is not None
            and float(bridge["density_mean"]) >= float(void["density_mean"])
        ),
        "void_stress_ge_bridge_stress": (
            bridge.get("stress_mean") is not None
            and void.get("stress_mean") is not None
            and float(void["stress_mean"]) >= float(bridge["stress_mean"])
        ),
        "terrain_zone_count_ge_three": len(zone_metrics) >= 3,
    }
    if tightrope and bridge.get("density_mean") is not None and tightrope.get("density_mean") is not None:
        pass_checks["tightrope_density_lt_bridge_density"] = float(tightrope["density_mean"]) <= float(
            bridge["density_mean"]
        )
    return {
        "status": "PASS" if all(pass_checks.values()) else ("NO_DATA" if not zone_metrics else "FAIL"),
        "zone_count": len(zone_metrics),
        "zone_metrics": zone_metrics,
        "pass_checks": pass_checks,
    }


def _translated_anchor_view(global_articles: Mapping[int, Mapping[str, Any]], anchor_idx: int) -> Dict[int, Dict[str, Any]]:
    anchor_xy = _xy(global_articles.get(anchor_idx, {}))
    if anchor_xy is None:
        return {int(idx): dict(row) for idx, row in global_articles.items()}
    translated: Dict[int, Dict[str, Any]] = {}
    for idx, row in global_articles.items():
        coord = _xy(row)
        materialized = dict(row)
        if coord is not None:
            shifted = coord - anchor_xy
            materialized["x"] = float(shifted[0])
            materialized["y"] = float(shifted[1])
        translated[int(idx)] = materialized
    return translated


def _observer_anchor_candidates(
    run_dir: Path,
    *,
    global_articles: Mapping[int, Mapping[str, Any]],
    metadata: Mapping[int, Mapping[str, Any]],
    label_col: Optional[str],
    label_counts: Mapping[str, int],
    min_label_count: int,
    recenter_mode: str,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_anchor_indices = set()
    seen_anchor_labels = set()

    for observer_dir in sorted(run_dir.glob("observer_*"), key=lambda p: p.name):
        if not observer_dir.is_dir():
            continue
        state = _load_json(observer_dir / "MONOLITH.view_state.json")
        observer_articles = _article_map(state)
        focus = state.get("observer_focus") if isinstance(state.get("observer_focus"), dict) else {}
        try:
            anchor_idx = int(focus.get("idx", observer_dir.name.split("_")[-1]))
        except Exception:
            continue
        label = _clean_label(metadata.get(anchor_idx, {}).get(label_col)) if label_col else ""
        candidates.append(
            {
                "observer_dir": observer_dir,
                "anchor_idx": anchor_idx,
                "observer_articles": observer_articles,
                "candidate_source": "materialized_observer_state",
                "anchor_label": label,
            }
        )
        seen_anchor_indices.add(anchor_idx)
        if label:
            seen_anchor_labels.add(label)

    if recenter_mode not in {"auto", LOCAL_RECOMPUTE_MODE}:
        return candidates

    payload, _, _ = load_primary_observer_payload(run_dir)
    if payload is None:
        return candidates

    if not label_col:
        return candidates

    for idx in sorted(global_articles):
        if int(idx) in seen_anchor_indices:
            continue
        label = _clean_label(metadata.get(int(idx), {}).get(label_col))
        if not label or label in seen_anchor_labels:
            continue
        if int(label_counts.get(label, 0) or 0) < int(min_label_count):
            continue
        candidates.append(
            {
                "observer_dir": run_dir / f"observer_local_{int(idx)}",
                "anchor_idx": int(idx),
                "observer_articles": _translated_anchor_view(global_articles, int(idx)),
                "candidate_source": "local_track_recompute_label_anchor",
                "anchor_label": label,
            }
        )
        seen_anchor_indices.add(int(idx))
        seen_anchor_labels.add(label)

    return candidates


def evaluate_run_dir(
    run_dir: Path,
    *,
    min_label_count: int = 2,
    preferred_label_column: Optional[str] = None,
    label_mode: str = "auto",
    recenter_mode: str = "auto",
    local_recompute_variant: str = LOCAL_RECOMPUTE_DEFAULT_VARIANT,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    recenter_mode = str(recenter_mode or "auto").strip().lower()
    if recenter_mode not in {"auto", "artifact_view", LOCAL_RECOMPUTE_MODE}:
        raise ValueError(f"Unsupported recenter_mode={recenter_mode!r}")
    local_recompute_variant = str(local_recompute_variant or LOCAL_RECOMPUTE_DEFAULT_VARIANT).strip().lower()
    if local_recompute_variant not in LOCAL_RECOMPUTE_VARIANTS:
        raise ValueError(f"Unsupported local_recompute_variant={local_recompute_variant!r}")
    global_state = _load_json(run_dir / "MONOLITH.view_state.json")
    global_articles = _article_map(global_state)
    metadata, metadata_path = _metadata_by_idx(run_dir)
    terrain = terrain_regime_metrics(list(metadata.values()))
    observer_rows: List[Dict[str, Any]] = []
    label_col, label_counts, label_diagnostics = select_label_column(
        metadata,
        global_articles.keys(),
        min_label_count=min_label_count,
        preferred_label_column=preferred_label_column,
        label_mode=label_mode,
    )

    anchor_candidates = _observer_anchor_candidates(
        run_dir,
        global_articles=global_articles,
        metadata=metadata,
        label_col=label_col,
        label_counts=label_counts,
        min_label_count=min_label_count,
        recenter_mode=recenter_mode,
    )

    for candidate in anchor_candidates:
        observer_dir = Path(candidate["observer_dir"])
        anchor_idx = int(candidate["anchor_idx"])
        observer_articles = {
            int(idx): dict(row)
            for idx, row in (candidate.get("observer_articles") or {}).items()
        }
        labels = {}
        if label_col:
            labels = {
                idx: _clean_label(row.get(label_col))
                for idx, row in metadata.items()
                if _clean_label(row.get(label_col))
            }
        observer_articles_for_eval = observer_articles
        local_recompute_summary: Dict[str, Any] = {
            "status": "SKIPPED",
            "mode": recenter_mode,
            "reason": "artifact_view_selected",
        }
        actual_recenter_mode = "artifact_view"
        local_recompute_ok = False
        if recenter_mode in {"auto", LOCAL_RECOMPUTE_MODE}:
            local_recompute, local_recompute_summary = compute_local_observer_recenter_from_run(
                run_dir,
                anchor_idx,
                variant=local_recompute_variant,
            )
            if local_recompute is not None:
                observer_articles_for_eval = local_recompute.article_map()
                actual_recenter_mode = LOCAL_RECOMPUTE_MODE
                local_recompute_ok = True
            elif recenter_mode == LOCAL_RECOMPUTE_MODE:
                actual_recenter_mode = LOCAL_RECOMPUTE_MODE
        if (
            candidate.get("candidate_source") == "local_track_recompute_label_anchor"
            and not local_recompute_ok
        ):
            continue
        row = evaluate_observer_recenter(
            global_articles=global_articles,
            observer_articles=observer_articles_for_eval,
            anchor_idx=anchor_idx,
            labels=labels,
            heldout_similarity=None,
        )
        row["observer_dir"] = str(observer_dir)
        row["label_column"] = label_col
        row["label_basis"] = label_diagnostics["selected_basis"]
        row["requested_recenter_mode"] = recenter_mode
        row["actual_recenter_mode"] = actual_recenter_mode
        row["local_recompute_variant"] = local_recompute_variant
        row["observer_candidate_source"] = candidate.get("candidate_source")
        row["anchor_label"] = candidate.get("anchor_label")
        row["local_track_recompute"] = local_recompute_summary
        row["label_geometry_tests"] = evaluate_label_geometry_tests(
            global_articles=global_articles,
            observer_articles=observer_articles_for_eval,
            anchor_idx=anchor_idx,
            labels=labels,
        )
        try:
            bundle = load_observer_manifold_bundle(run_dir, observer_dir, label_key=label_col or "source")
            row["observer_manifold_bundle_supported"] = True
            row["observer_manifold_bundle"] = {
                "bundle_type": "observer_manifold_bundle",
                "node_count": len(bundle.nodes),
                "edge_count": len(bundle.edges),
                "path_count": bundle.path_count,
                "focus_xy_centered": bundle.focus_xy_centered,
                "mean_nontranslation_shift": bundle.mean_nontranslation_shift,
                "edge_action_summary": bundle.edge_summary(),
                "provenance": bundle.provenance,
            }
        except Exception as exc:
            row["observer_manifold_bundle_supported"] = False
            row["observer_manifold_bundle_error"] = f"{type(exc).__name__}: {exc}"
        observer_rows.append(row)

    shifts = [float(row["nontranslation_shift_mean"]) for row in observer_rows if row.get("nontranslation_shift_mean") is not None]
    raw_gains = [
        float(row["label_gap_gain_over_translation"])
        for row in observer_rows
        if row.get("label_gap_gain_over_translation") is not None
    ]
    normalized_gains = [
        float(row["scale_normalized_label_gap_gain_over_translation"])
        for row in observer_rows
        if row.get("scale_normalized_label_gap_gain_over_translation") is not None
    ]
    primary_gains = normalized_gains or raw_gains
    source_status = "OK" if primary_gains else "INSUFFICIENT_LABELS"
    mean_gain = _mean(raw_gains)
    normalized_mean_gain = _mean(normalized_gains)
    primary_mean_gain = _mean(primary_gains)
    source_pass = bool(
        primary_mean_gain is not None
        and float(primary_mean_gain) >= float(DEFAULT_THRESHOLDS["min_label_gap_gain"])
    )
    label_contraction = {
        "status": source_status,
        "pass": source_pass,
        "mean_label_gap_gain_over_translation": mean_gain,
        "mean_scale_normalized_label_gap_gain_over_translation": normalized_mean_gain,
        "primary_mean_label_gap_gain_over_translation": primary_mean_gain,
        "metric_basis": "scale_normalized_label_gap" if normalized_gains else "raw_label_gap",
        "min_label_gap_gain": DEFAULT_THRESHOLDS["min_label_gap_gain"],
        "n_observers_with_label_test": len(primary_gains),
        "label_column": label_col,
        "label_basis": label_diagnostics["selected_basis"],
    }
    return {
        "run_dir": str(run_dir),
        "status": "OK" if observer_rows else "NO_OBSERVER_STATES",
        "metadata_path": metadata_path,
        "label_column": label_col,
        "label_basis": label_diagnostics["selected_basis"],
        "label_diagnostics": label_diagnostics,
        "label_counts": label_counts,
        "observer_count": len(observer_rows),
        "mean_nontranslation_shift": _mean(shifts),
        "label_contraction": label_contraction,
        "source_label_contraction": label_contraction,
        "requested_recenter_mode": recenter_mode,
        "local_recompute_variant": local_recompute_variant,
        "actual_recenter_modes": sorted(
            set(str(row.get("actual_recenter_mode", "artifact_view")) for row in observer_rows)
        ),
        "local_track_recompute_supported_count": sum(
            1 for row in observer_rows if row.get("actual_recenter_mode") == LOCAL_RECOMPUTE_MODE
        ),
        "ideological_validation_suite": aggregate_ideological_validation_suite(
            observer_rows,
            label_column=label_col,
            label_basis=label_diagnostics["selected_basis"],
            label_counts=label_counts,
            min_label_count=min_label_count,
        ),
        "observer_manifold_bundle_supported_count": sum(
            1 for row in observer_rows if bool(row.get("observer_manifold_bundle_supported"))
        ),
        "edge_action_ledger": _aggregate_edge_action(observer_rows),
        "terrain_regime": terrain,
        "observers": observer_rows,
    }


def _aggregate_edge_action(observer_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bundles = [
        row.get("observer_manifold_bundle") or {}
        for row in observer_rows
        if bool(row.get("observer_manifold_bundle_supported"))
    ]
    summaries = [(bundle.get("edge_action_summary") or {}) for bundle in bundles]
    edge_counts = [int(summary.get("edge_count") or 0) for summary in summaries]
    fresh_edge_counts = [
        int(summary.get("edge_count") or 0)
        for bundle, summary in zip(bundles, summaries)
        if bool((bundle.get("provenance") or {}).get("fresh_focused_observer_replay"))
    ]
    legacy_hydrated_count = sum(
        1 for bundle in bundles if bool((bundle.get("provenance") or {}).get("legacy_hydrated"))
    )
    mean_positive_excess = [
        float(summary.get("mean_positive_excess_action"))
        for summary in summaries
        if summary.get("mean_positive_excess_action") is not None
    ]
    mean_excess = [
        float(summary.get("mean_excess_action"))
        for summary in summaries
        if summary.get("mean_excess_action") is not None
    ]
    return {
        "status": "OK" if summaries else "NO_BUNDLES",
        "bundle_count": len(summaries),
        "edge_count": int(sum(edge_counts)),
        "fresh_edge_count": int(sum(fresh_edge_counts)),
        "legacy_hydrated_bundle_count": int(legacy_hydrated_count),
        "mean_excess_action": _mean(mean_excess),
        "mean_positive_excess_action": _mean(mean_positive_excess),
        "edge_action_supported": bool(sum(edge_counts) > 0),
        "fresh_edge_action_supported": bool(sum(fresh_edge_counts) > 0),
        "positive_excess_supported": bool(mean_positive_excess and float(np.mean(mean_positive_excess)) > 0.0),
    }


def build_orchestration(run_dirs: Sequence[Path], *, synthetic_fixture: bool) -> Dict[str, Any]:
    nodes = [
        {"node_id": "load_artifacts", "status": "completed"},
        {"node_id": "local_track_recompute", "status": "completed"},
        {"node_id": "translation_null", "status": "completed"},
        {"node_id": "label_basis_contract", "status": "completed"},
        {"node_id": "label_contraction", "status": "completed"},
        {"node_id": "all_pairs_separation", "status": "completed"},
        {"node_id": "centroid_contraction", "status": "completed"},
        {"node_id": "silhouette_validation", "status": "completed"},
        {"node_id": "ari_nmi_recovery", "status": "completed"},
        {"node_id": "permutation_within_between", "status": "completed"},
        {"node_id": "multi_anchor_coverage", "status": "completed"},
        {"node_id": "heldout_frame_correlation", "status": "completed" if synthetic_fixture else "skipped"},
        {"node_id": "property_theft_homotopy", "status": "completed" if synthetic_fixture else "skipped"},
        {"node_id": "terrain_regime", "status": "completed"},
        {"node_id": "summary", "status": "completed"},
    ]
    return build_orchestration_contract(
        dag_id="observer_recenter_meaning_probe_v1",
        nodes=nodes,
        dependencies={
            "local_track_recompute": ["load_artifacts"],
            "translation_null": ["local_track_recompute"],
            "label_basis_contract": ["load_artifacts"],
            "label_contraction": ["translation_null", "label_basis_contract"],
            "all_pairs_separation": ["translation_null", "label_basis_contract"],
            "centroid_contraction": ["translation_null", "label_basis_contract"],
            "silhouette_validation": ["translation_null", "label_basis_contract"],
            "ari_nmi_recovery": ["translation_null", "label_basis_contract"],
            "permutation_within_between": ["translation_null", "label_basis_contract"],
            "multi_anchor_coverage": ["label_basis_contract"],
            "heldout_frame_correlation": ["translation_null"],
            "property_theft_homotopy": ["translation_null"],
            "terrain_regime": ["load_artifacts"],
            "summary": [
                "label_contraction",
                "all_pairs_separation",
                "centroid_contraction",
                "silhouette_validation",
                "ari_nmi_recovery",
                "permutation_within_between",
                "multi_anchor_coverage",
                "heldout_frame_correlation",
                "property_theft_homotopy",
                "terrain_regime",
            ],
        },
        executor="sidecar_script",
        manifest_paths=[],
        metadata={
            "diagnostic_type": DIAGNOSTIC_TYPE,
            "run_dirs": [str(path) for path in run_dirs],
            "synthetic_fixture": bool(synthetic_fixture),
        },
    )


def summarize_payload(
    *,
    run_results: Sequence[Dict[str, Any]],
    synthetic: Optional[Dict[str, Any]],
    orchestration: Dict[str, Any],
) -> Dict[str, Any]:
    real_ok = [row for row in run_results if row.get("status") == "OK"]
    label_pass = [row for row in real_ok if bool((row.get("label_contraction") or {}).get("pass"))]
    terrain_pass = [row for row in real_ok if (row.get("terrain_regime") or {}).get("status") == "PASS"]
    ideology_suite_pass = [
        row for row in real_ok if (row.get("ideological_validation_suite") or {}).get("status") == "PASS"
    ]
    mean_shift = _mean(
        [
            float(row["mean_nontranslation_shift"])
            for row in real_ok
            if row.get("mean_nontranslation_shift") is not None
        ]
    )
    claim_readiness = {
        "mechanical_recenter_nontrivial": bool(
            mean_shift is not None and mean_shift > DEFAULT_THRESHOLDS["min_nontranslation_shift"]
        ),
        "synthetic_meaning_pass": bool(synthetic and synthetic.get("status") == "PASS"),
        "real_label_contraction_pass": bool(real_ok and len(label_pass) == len(real_ok)),
        "real_source_label_contraction_pass": bool(real_ok and len(label_pass) == len(real_ok)),
        "ideological_validation_suite_pass": bool(real_ok and len(ideology_suite_pass) == len(real_ok)),
        "real_terrain_regime_pass": bool(real_ok and len(terrain_pass) == len(real_ok)),
        "edge_action_supported": bool(
            any(
                int(((row.get("edge_action_ledger") or {}).get("edge_count") or 0)) > 0
                for row in run_results
            )
        ),
        "fresh_edge_action_supported": bool(
            any(
                int(((row.get("edge_action_ledger") or {}).get("fresh_edge_count") or 0)) > 0
                for row in run_results
            )
        ),
        "local_track_recompute_supported": bool(
            any(int(row.get("local_track_recompute_supported_count") or 0) > 0 for row in real_ok)
        ),
    }
    required_claim_readiness_keys = [
        "mechanical_recenter_nontrivial",
        "synthetic_meaning_pass",
        "real_label_contraction_pass",
        "real_terrain_regime_pass",
    ]
    thesis_safe = bool(
        all(claim_readiness[key] for key in required_claim_readiness_keys)
    )
    failure_reasons = [name for name in required_claim_readiness_keys if not claim_readiness[name]]
    return {
        "schema_version": SCHEMA_VERSION,
        "summary_type": SUMMARY_TYPE,
        "probe_type": DIAGNOSTIC_TYPE,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "generated_at_utc": _utc_now(),
        "cache_key": build_cache_key(
            {
                "diagnostic_type": DIAGNOSTIC_TYPE,
                "run_dirs": [row.get("run_dir") for row in run_results],
                "synthetic": bool(synthetic),
            }
        ),
        "orchestration_contract": orchestration,
        "status": "OK",
        "safe_for_thesis_claim": thesis_safe,
        "thesis_safe": thesis_safe,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "failure_reasons": failure_reasons,
        "synthetic_fixture": synthetic,
        "real_run_count": len(run_results),
        "real_ok_run_count": len(real_ok),
        "real_label_pass_count": len(label_pass),
        "real_source_label_pass_count": len(label_pass),
        "ideological_validation_suite_pass_count": len(ideology_suite_pass),
        "real_terrain_pass_count": len(terrain_pass),
        "mean_nontranslation_shift": mean_shift,
        "claim_readiness": claim_readiness,
        "required_claim_readiness_keys": required_claim_readiness_keys,
        "claim_boundary": {
            "mechanical_recenter_is_not_semantic_validation": True,
            "requires_translation_null_advantage": True,
            "requires_independent_labels_or_heldout_scores": True,
            "legacy_hydrated_paths_are_not_fresh_replay": True,
            "edge_action_not_required_for_observer_recenter_claim": True,
            "edge_action_required_for_track4_action_claim": True,
            "local_track_recompute_is_stronger_than_artifact_reprojection": True,
        },
        "architecture_interpretation": (
            "Observer recentering is scientifically meaningful only if it beats translation-only nulls "
            "on independent labels or held-out semantic scores. Mechanical non-translation alone is "
            "visual evidence, not semantic validation."
        ),
        "runs": list(run_results),
    }


def write_outputs(payload: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "observer_recenter_meaning_probe.json"
    csv_path = output_dir / "observer_recenter_meaning_probe.csv"
    dag_path = output_dir / "observer_recenter_meaning_dag_contract.json"
    bundle_artifacts = _write_observer_bundle_artifacts(payload, output_dir / "observer_manifold_bundles")
    artifacts = {"json": str(json_path), "csv": str(csv_path), "dag_contract": str(dag_path)}
    if bundle_artifacts:
        artifacts["observer_manifold_bundles"] = str(output_dir / "observer_manifold_bundles")
    payload["artifacts"] = dict(artifacts)
    payload["observer_manifold_bundle_artifacts"] = bundle_artifacts
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dag_path.write_text(
        json.dumps(_json_safe(payload["orchestration_contract"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "run_dir",
            "observer_dir",
            "anchor_idx",
            "requested_recenter_mode",
            "actual_recenter_mode",
            "local_recompute_variant",
            "focus_xy_centered",
            "nontranslation_shift_mean",
            "label_column",
            "label_basis",
            "label_gap_gain_over_translation",
            "scale_normalized_label_gap_gain_over_translation",
            "translation_label_gap",
            "translation_scale_normalized_label_gap",
            "observer_label_gap",
            "observer_scale_normalized_label_gap",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for run in payload.get("runs", []):
            for observer in run.get("observers", []):
                writer.writerow(
                    {
                        "run_dir": run.get("run_dir"),
                        "observer_dir": observer.get("observer_dir"),
                        "anchor_idx": observer.get("anchor_idx"),
                        "requested_recenter_mode": observer.get("requested_recenter_mode"),
                        "actual_recenter_mode": observer.get("actual_recenter_mode"),
                        "local_recompute_variant": observer.get("local_recompute_variant"),
                        "focus_xy_centered": observer.get("focus_xy_centered"),
                        "nontranslation_shift_mean": observer.get("nontranslation_shift_mean"),
                        "label_column": observer.get("label_column"),
                        "label_basis": run.get("label_basis"),
                        "label_gap_gain_over_translation": observer.get("label_gap_gain_over_translation"),
                        "scale_normalized_label_gap_gain_over_translation": observer.get(
                            "scale_normalized_label_gap_gain_over_translation"
                        ),
                        "translation_label_gap": (observer.get("translation_null") or {}).get("label_gap"),
                        "translation_scale_normalized_label_gap": (observer.get("translation_null") or {}).get(
                            "scale_normalized_label_gap"
                        ),
                        "observer_label_gap": (observer.get("observer_recentered") or {}).get("label_gap"),
                        "observer_scale_normalized_label_gap": (observer.get("observer_recentered") or {}).get(
                            "scale_normalized_label_gap"
                        ),
                    }
                )
    return artifacts


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")[:160] or "leaf"


def _write_observer_bundle_artifacts(payload: Mapping[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_dir = Path(str(run.get("run_dir") or ""))
        label_key = str(run.get("label_column") or "source")
        for observer in run.get("observers", []):
            if not isinstance(observer, dict) or not bool(observer.get("observer_manifold_bundle_supported")):
                continue
            observer_dir = Path(str(observer.get("observer_dir") or ""))
            try:
                bundle = load_observer_manifold_bundle(run_dir, observer_dir, label_key=label_key)
                rel_slug = _safe_slug(f"{run_dir.name}_{observer_dir.name}_{bundle.provenance.get('provenance_hash')}")
                written = write_observer_manifold_bundle(bundle, output_dir / rel_slug)
                artifacts.append(
                    {
                        "run_dir": str(run_dir),
                        "observer_dir": str(observer_dir),
                        "observer_idx": int(bundle.observer_idx),
                        **written,
                    }
                )
            except Exception as exc:
                artifacts.append(
                    {
                        "run_dir": str(run_dir),
                        "observer_dir": str(observer_dir),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return artifacts


def build_payload(
    *,
    run_dirs: Sequence[Path],
    include_synthetic_fixture: bool,
    min_label_count: int = 2,
    preferred_label_column: Optional[str] = None,
    label_mode: str = "auto",
    recenter_mode: str = "auto",
    local_recompute_variant: str = LOCAL_RECOMPUTE_DEFAULT_VARIANT,
) -> Dict[str, Any]:
    synthetic = synthetic_fixture_payload() if include_synthetic_fixture else None
    run_results = [
        evaluate_run_dir(
            path,
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
            recenter_mode=recenter_mode,
            local_recompute_variant=local_recompute_variant,
        )
        for path in run_dirs
    ]
    orchestration = build_orchestration(run_dirs, synthetic_fixture=include_synthetic_fixture)
    return summarize_payload(run_results=run_results, synthetic=synthetic, orchestration=orchestration)


def run_probe(
    *,
    run_dirs: Sequence[Path],
    output_dir: Path,
    include_synthetic_fixture: bool = True,
    min_label_count: int = 2,
    preferred_label_column: Optional[str] = None,
    label_mode: str = "auto",
    recenter_mode: str = "auto",
    local_recompute_variant: str = LOCAL_RECOMPUTE_DEFAULT_VARIANT,
) -> Dict[str, Any]:
    payload = build_payload(
        run_dirs=run_dirs,
        include_synthetic_fixture=include_synthetic_fixture,
        min_label_count=min_label_count,
        preferred_label_column=preferred_label_column,
        label_mode=label_mode,
        recenter_mode=recenter_mode,
        local_recompute_variant=local_recompute_variant,
    )
    artifacts = write_outputs(payload, output_dir)
    payload["artifacts"] = artifacts
    return payload


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / "observer_recenter_meaning_probe" / stamp


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--no-synthetic-fixture", action="store_true")
    parser.add_argument("--min-label-count", type=int, default=2)
    parser.add_argument(
        "--label-column",
        dest="preferred_label_column",
        default=None,
        help="Explicit metadata label column for contraction tests, e.g. perspective_tag.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("auto", "ideological", "provenance", "diagnostic", "any"),
        default="auto",
        help="Auto prefers ideological labels before provenance labels.",
    )
    parser.add_argument(
        "--recenter-mode",
        choices=("auto", "artifact_view", LOCAL_RECOMPUTE_MODE),
        default="auto",
        help="auto uses local Track 2/1.5/3 recompute when artifacts are available; artifact_view preserves the old view-state baseline.",
    )
    parser.add_argument(
        "--local-recompute-variant",
        choices=LOCAL_RECOMPUTE_VARIANTS,
        default=LOCAL_RECOMPUTE_DEFAULT_VARIANT,
        help="Variant for local Track 2/1.5/3 observer recentering; default preserves current behavior.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_probe(
        run_dirs=args.run_dir,
        output_dir=args.output_dir,
        include_synthetic_fixture=not bool(args.no_synthetic_fixture),
        min_label_count=int(args.min_label_count),
        preferred_label_column=args.preferred_label_column,
        label_mode=args.label_mode,
        recenter_mode=args.recenter_mode,
        local_recompute_variant=args.local_recompute_variant,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    **payload["artifacts"],
                    "status": payload["status"],
                    "thesis_safe": payload["thesis_safe"],
                    "claim_readiness": payload["claim_readiness"],
                    "failure_reasons": payload["failure_reasons"],
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
