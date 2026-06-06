#!/usr/bin/env python3
"""Run direct Track 2 necessity ablations from saved synthetic artifacts.

This is intentionally artifact-level: it does not rerun DeBERTa.  It answers
whether the existing Track 2 geometry behaves like a necessary foundation by
comparing saved Track 2 coordinates against Track 1.5, full integrated vectors,
exact kernel oracles, RKS dimension ramps, and observer mixing order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # pragma: no cover - exercised in integration where sklearn is present.
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except Exception:  # pragma: no cover
    KMeans = None
    adjusted_rand_score = None
    normalized_mutual_info_score = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SYNTHETIC_ROOT = (
    ROOT / "outputs" / "experiments" / "runs" / "experiments_20260506_192553" / "synthetic"
)
SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "track2_necessity_ablation"


@dataclass(frozen=True)
class CellArtifacts:
    kernel: str
    seed: int
    cell_id: str
    run_dir: Path
    labels: list[str]
    label_column: str
    track2: np.ndarray
    track15: np.ndarray
    full_integrated: np.ndarray
    cls_per_bot: np.ndarray


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _median(values: Iterable[Any]) -> Optional[float]:
    finite = sorted(value for value in (_safe_float(raw) for raw in values) if value is not None)
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float((finite[mid - 1] + finite[mid]) / 2.0)


def _rate(flags: Iterable[Any]) -> float:
    vals = [bool(flag) for flag in flags]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _load_labels(path: Path, label_column: Optional[str] = None) -> tuple[list[str], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic labels file: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Synthetic labels file is empty: {path}")

    candidates = [
        label_column,
        "group_perspective_tag",
        "group_topic",
        "group_label",
        "group_source",
        "source",
        "label",
        "cluster",
        "cluster_label",
    ]
    for candidate in candidates:
        if candidate and candidate in rows[0] and any(row.get(candidate) for row in rows):
            return [str(row.get(candidate) or "unknown") for row in rows], str(candidate)

    raise ValueError(f"No usable label column found in {path}")


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return np.asarray(np.load(path), dtype=np.float64)


def _load_cls_per_bot(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing observer payload: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cls_per_bot = payload.get("cls_per_bot") if isinstance(payload, dict) else None
    if cls_per_bot is None and isinstance(payload, dict):
        channel_extras = payload.get("channel_extras")
        if isinstance(channel_extras, dict):
            main = channel_extras.get("main")
            if isinstance(main, dict):
                cls_per_bot = main.get("cls_per_bot")
    if cls_per_bot is None:
        raise KeyError(f"cls_per_bot not found in {path}")
    arr = np.asarray(cls_per_bot, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected cls_per_bot [N,B,H], got {arr.shape} in {path}")
    return arr


def _load_cell(
    synthetic_root: Path,
    *,
    kernel: str,
    seed: int,
    label_column: Optional[str],
) -> CellArtifacts:
    cell_id = f"{kernel}_seed{seed}"
    run_dir = synthetic_root / cell_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing synthetic cell: {run_dir}")

    labels, resolved_label_column = _load_labels(run_dir / "labels" / "hidden_groups.csv", label_column)
    track2 = _load_npy(run_dir / "dirichlet_fused.npy")
    track15 = _load_npy(run_dir / "spectral_probe_magnitudes.npy")
    full_path = run_dir / "integrated_vectors.npy"
    if not full_path.exists():
        full_path = run_dir / "features.npy"
    full_integrated = _load_npy(full_path)
    cls_per_bot = _load_cls_per_bot(run_dir / "observer_global.pt")

    n = len(labels)
    for name, arr in {
        "track2": track2,
        "track15": track15,
        "full_integrated": full_integrated,
        "cls_per_bot": cls_per_bot,
    }.items():
        if arr.shape[0] != n:
            raise ValueError(f"{cell_id} {name} row count {arr.shape[0]} != labels {n}")

    return CellArtifacts(
        kernel=kernel,
        seed=seed,
        cell_id=cell_id,
        run_dir=run_dir,
        labels=labels,
        label_column=resolved_label_column,
        track2=track2,
        track15=track15,
        full_integrated=full_integrated,
        cls_per_bot=cls_per_bot,
    )


def _standardize(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    keep = (std.reshape(-1) > eps)
    if not np.any(keep):
        return np.zeros((arr.shape[0], 1), dtype=np.float64)
    return (arr[:, keep] - mean[:, keep]) / std[:, keep]


def _fit_standardizer(X: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    keep = (std.reshape(-1) > eps)
    if not np.any(keep):
        keep = np.zeros(arr.shape[1], dtype=bool)
        keep[0] = True
        std[:, 0] = 1.0
    return mean, std, keep


def _apply_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return (arr[:, keep] - mean[:, keep]) / std[:, keep]


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.sum(X * X, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * (X @ X.T), 0.0)


def _median_sigma_from_d2(d2: np.ndarray) -> float:
    if d2.shape[0] < 2:
        return 1.0
    tri = np.triu_indices(d2.shape[0], k=1)
    distances = np.sqrt(np.maximum(d2[tri], 0.0))
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 1e-12]
    if distances.size == 0:
        return 1.0
    return float(max(np.median(distances) / math.sqrt(2.0), 1e-6))


def exact_kernel(X: np.ndarray, kernel: str, sigma: Optional[float] = None) -> np.ndarray:
    Xz = _standardize(X)
    d2 = _pairwise_sq_dists(Xz)
    sigma = float(sigma or _median_sigma_from_d2(d2))
    r = np.sqrt(np.maximum(d2, 0.0))
    scale = max(sigma, 1e-6)
    kernel = kernel.lower()
    if kernel == "rbf":
        K = np.exp(-d2 / (2.0 * scale * scale))
    elif kernel == "imq":
        K = 1.0 / np.sqrt(1.0 + d2 / (scale * scale))
    elif kernel == "matern":
        root3 = math.sqrt(3.0)
        z = root3 * r / scale
        K = (1.0 + z) * np.exp(-z)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    np.fill_diagonal(K, 1.0)
    return np.asarray(K, dtype=np.float64)


def rbf_kernel_from_features(X: np.ndarray) -> np.ndarray:
    return exact_kernel(X, "rbf")


def linear_kernel_from_features(X: np.ndarray) -> np.ndarray:
    Xz = _standardize(X)
    norms = np.linalg.norm(Xz, axis=1, keepdims=True)
    Xn = Xz / np.maximum(norms, 1e-12)
    K = Xn @ Xn.T
    K = (K + 1.0) / 2.0
    np.fill_diagonal(K, 1.0)
    return np.clip(K, 0.0, 1.0)


def _kernel_upper_values(K: np.ndarray) -> np.ndarray:
    tri = np.triu_indices(K.shape[0], k=1)
    return np.asarray(K[tri], dtype=np.float64)


def kernel_correlation(A: np.ndarray, B: np.ndarray) -> Optional[float]:
    if A.shape != B.shape or A.shape[0] < 2:
        return None
    a = _kernel_upper_values(A)
    b = _kernel_upper_values(B)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def kernel_spearman(A: np.ndarray, B: np.ndarray) -> Optional[float]:
    if A.shape != B.shape or A.shape[0] < 2:
        return None
    a = _average_ranks(_kernel_upper_values(A))
    b = _average_ranks(_kernel_upper_values(B))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def relative_frobenius_error(A: np.ndarray, B: np.ndarray) -> Optional[float]:
    denom = float(np.linalg.norm(B, ord="fro"))
    if denom <= 1e-12:
        return None
    return float(np.linalg.norm(A - B, ord="fro") / denom)


def kernel_error_metrics(A: np.ndarray, B: Optional[np.ndarray]) -> dict[str, Optional[float]]:
    if B is None:
        return {
            "kernel_corr": None,
            "kernel_spearman": None,
            "relative_frobenius_error": None,
            "kernel_mae": None,
            "kernel_rmse": None,
            "diag_mean_abs_error": None,
        }
    diff = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    return {
        "kernel_corr": kernel_correlation(A, B),
        "kernel_spearman": kernel_spearman(A, B),
        "relative_frobenius_error": relative_frobenius_error(A, B),
        "kernel_mae": float(np.mean(np.abs(diff))),
        "kernel_rmse": float(np.sqrt(np.mean(diff * diff))),
        "diag_mean_abs_error": float(np.mean(np.abs(np.diag(diff)))),
    }


def rks_features(
    X: np.ndarray,
    *,
    kernel: str,
    dim: int,
    seed: int,
    sigma: Optional[float] = None,
) -> np.ndarray:
    Xz = _standardize(X)
    d2 = _pairwise_sq_dists(Xz)
    sigma = float(sigma or _median_sigma_from_d2(d2))
    rng = np.random.default_rng(seed)
    input_dim = Xz.shape[1]
    kernel = kernel.lower()
    if kernel == "rbf":
        omega = rng.normal(loc=0.0, scale=1.0 / max(sigma, 1e-6), size=(input_dim, dim))
    elif kernel == "imq":
        omega = rng.standard_t(df=1.0, size=(input_dim, dim)) / max(sigma, 1e-6)
    elif kernel == "matern":
        omega = rng.standard_t(df=3.0, size=(input_dim, dim)) / max(sigma, 1e-6)
    else:
        raise ValueError(f"Unsupported RKS kernel: {kernel}")
    bias = rng.uniform(0.0, 2.0 * math.pi, size=(dim,))
    return math.sqrt(2.0 / dim) * np.cos(Xz @ omega + bias)


def rks_features_with_standardizer(
    X: np.ndarray,
    *,
    kernel: str,
    dim: int,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    keep: np.ndarray,
    sigma: float,
) -> np.ndarray:
    Xz = _apply_standardizer(X, mean, std, keep)
    rng = np.random.default_rng(seed)
    input_dim = Xz.shape[1]
    kernel = kernel.lower()
    if kernel == "rbf":
        omega = rng.normal(loc=0.0, scale=1.0 / max(sigma, 1e-6), size=(input_dim, dim))
    elif kernel == "imq":
        omega = rng.standard_t(df=1.0, size=(input_dim, dim)) / max(sigma, 1e-6)
    elif kernel == "matern":
        omega = rng.standard_t(df=3.0, size=(input_dim, dim)) / max(sigma, 1e-6)
    else:
        raise ValueError(f"Unsupported RKS kernel: {kernel}")
    bias = rng.uniform(0.0, 2.0 * math.pi, size=(dim,))
    return math.sqrt(2.0 / dim) * np.cos(Xz @ omega + bias)


def rks_approx_kernel(Z: np.ndarray) -> np.ndarray:
    K = np.asarray(Z, dtype=np.float64) @ np.asarray(Z, dtype=np.float64).T
    K = (K + K.T) / 2.0
    np.fill_diagonal(K, 1.0)
    return K


def kernel_pca_embedding(K: np.ndarray, dims: Optional[int] = None) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    centered = H @ ((K + K.T) / 2.0) @ H
    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    dims = min(int(dims or max(2, min(n - 1, 8))), n)
    return vectors[:, :dims] * np.sqrt(values[:dims])[None, :]


def _fallback_kmeans(points: np.ndarray, k: int, *, iterations: int = 80) -> Optional[np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < k or k <= 1:
        return None
    order = np.lexsort(points[:, : min(points.shape[1], 4)].T[::-1])
    centroids = points[order[np.linspace(0, len(order) - 1, k).astype(int)]].copy()
    assignments = np.zeros(points.shape[0], dtype=int)
    for _ in range(iterations):
        d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = np.argmin(d2, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_idx in range(k):
            members = points[assignments == cluster_idx]
            if len(members):
                centroids[cluster_idx] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(d2, axis=1)))
                centroids[cluster_idx] = points[farthest]
    return assignments


def _comb2(value: int) -> float:
    return float(value * (value - 1) / 2)


def _fallback_ari(true_labels: Sequence[str], pred_labels: Sequence[int]) -> Optional[float]:
    n = len(true_labels)
    if n != len(pred_labels) or n < 2:
        return None
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    contingency: Counter[tuple[str, int]] = Counter(zip(true_labels, pred_labels))
    sum_comb = sum(_comb2(v) for v in contingency.values())
    sum_true = sum(_comb2(v) for v in true_counts.values())
    sum_pred = sum(_comb2(v) for v in pred_counts.values())
    total = _comb2(n)
    expected = (sum_true * sum_pred / total) if total else 0.0
    maximum = 0.5 * (sum_true + sum_pred)
    denom = maximum - expected
    return float((sum_comb - expected) / denom) if abs(denom) > 1e-12 else 0.0


def _fallback_nmi(true_labels: Sequence[str], pred_labels: Sequence[int]) -> Optional[float]:
    n = len(true_labels)
    if n != len(pred_labels) or n == 0:
        return None
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    contingency: Counter[tuple[str, int]] = Counter(zip(true_labels, pred_labels))
    mi = 0.0
    for (true, pred), count in contingency.items():
        if count:
            mi += (count / n) * math.log((count * n) / (true_counts[true] * pred_counts[pred]))
    h_true = -sum((count / n) * math.log(count / n) for count in true_counts.values())
    h_pred = -sum((count / n) * math.log(count / n) for count in pred_counts.values())
    denom = math.sqrt(h_true * h_pred)
    return float(mi / denom) if denom > 1e-12 else 0.0


def cluster_scores(points: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    unique = sorted(set(labels))
    k = len(unique)
    if k < 2 or len(labels) < k:
        return {"status": "INSUFFICIENT_LABELS", "nmi": None, "ari": None, "n_clusters": k}
    if KMeans is not None:
        pred = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(points)
    else:
        pred = _fallback_kmeans(points, k)
    if pred is None:
        return {"status": "KMEANS_FAILED", "nmi": None, "ari": None, "n_clusters": k}
    if normalized_mutual_info_score is not None and adjusted_rand_score is not None:
        nmi = float(normalized_mutual_info_score(labels, pred))
        ari = float(adjusted_rand_score(labels, pred))
    else:
        nmi = _fallback_nmi(labels, pred.tolist())
        ari = _fallback_ari(labels, pred.tolist())
    return {
        "status": "OK",
        "nmi": nmi,
        "ari": ari,
        "n_clusters": k,
        "cluster_sizes": dict(sorted(Counter(int(value) for value in pred.tolist()).items())),
    }


def kernel_label_metrics(K: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    labels = list(labels)
    same: list[float] = []
    different: list[float] = []
    pair_scores: list[tuple[float, int]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = float(K[i, j])
            is_same = int(labels[i] == labels[j])
            pair_scores.append((score, is_same))
            if is_same:
                same.append(score)
            else:
                different.append(score)
    nn_hits = 0
    for i in range(len(labels)):
        row = np.asarray(K[i], dtype=np.float64).copy()
        row[i] = -np.inf
        nearest = int(np.argmax(row))
        nn_hits += int(labels[i] == labels[nearest])

    auc = None
    if same and different:
        wins = 0.0
        total = 0
        for s in same:
            for d in different:
                wins += 1.0 if s > d else 0.5 if s == d else 0.0
                total += 1
        auc = wins / total if total else None

    embedding = kernel_pca_embedding(K)
    scores = cluster_scores(embedding, labels)
    return {
        "same_similarity_mean": float(np.mean(same)) if same else None,
        "different_similarity_mean": float(np.mean(different)) if different else None,
        "same_different_similarity_separation": (
            float(np.mean(same) - np.mean(different)) if same and different else None
        ),
        "pair_auc_same_vs_different": auc,
        "nearest_neighbor_label_agreement": float(nn_hits / len(labels)) if labels else None,
        "cluster_nmi": scores.get("nmi"),
        "cluster_ari": scores.get("ari"),
        "cluster_status": scores.get("status"),
        "n_clusters": scores.get("n_clusters"),
    }


def direct_feature_cluster_record(
    cell: CellArtifacts,
    *,
    variant: str,
    features: np.ndarray,
) -> dict[str, Any]:
    scores = cluster_scores(_standardize(features), cell.labels)
    return {
        "record_type": "variant_metric",
        "cell_id": cell.cell_id,
        "kernel": cell.kernel,
        "seed": cell.seed,
        "variant": variant,
        "kernel_mode": "direct_feature_kmeans",
        "label_column": cell.label_column,
        "n_samples": len(cell.labels),
        "n_features": int(features.shape[1]),
        "same_different_similarity_separation": None,
        "pair_auc_same_vs_different": None,
        "nearest_neighbor_label_agreement": None,
        "cluster_nmi": scores.get("nmi"),
        "cluster_ari": scores.get("ari"),
        "cluster_status": scores.get("status"),
        "comparison_target": None,
        "kernel_corr": None,
        "relative_frobenius_error": None,
    }


def kernel_record(
    cell: CellArtifacts,
    *,
    variant: str,
    K: np.ndarray,
    n_features: Optional[int] = None,
    kernel_mode: str = "rbf_on_coordinates",
    comparison_target: Optional[str] = None,
    target_kernel: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    metrics = kernel_label_metrics(K, cell.labels)
    error_metrics = kernel_error_metrics(K, target_kernel)
    return {
        "record_type": "variant_metric",
        "cell_id": cell.cell_id,
        "kernel": cell.kernel,
        "seed": cell.seed,
        "variant": variant,
        "kernel_mode": kernel_mode,
        "label_column": cell.label_column,
        "n_samples": len(cell.labels),
        "n_features": n_features,
        "comparison_target": comparison_target,
        **error_metrics,
        **metrics,
    }


def build_cell_records(
    cell: CellArtifacts,
    *,
    rks_dims: Sequence[int],
    rks_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mean_cls = cell.cls_per_bot.mean(axis=1)
    K_exact = exact_kernel(mean_cls, cell.kernel)
    K_track2_rbf = rbf_kernel_from_features(cell.track2)
    K_track2_dot = linear_kernel_from_features(cell.track2)
    K_track15 = rbf_kernel_from_features(cell.track15)
    K_full = rbf_kernel_from_features(cell.full_integrated)
    K_hadamard = np.clip(K_track2_rbf * K_track15, 0.0, 1.0)
    K_raw_cls = rbf_kernel_from_features(mean_cls)

    records.extend(
        [
            kernel_record(
                cell,
                variant="saved_track2_rbf",
                K=K_track2_rbf,
                n_features=cell.track2.shape[1],
                comparison_target="exact_kernel_oracle",
                target_kernel=K_exact,
            ),
            kernel_record(
                cell,
                variant="saved_track2_dot",
                K=K_track2_dot,
                n_features=cell.track2.shape[1],
                kernel_mode="dot_on_saved_rks_coordinates",
                comparison_target="exact_kernel_oracle",
                target_kernel=K_exact,
            ),
            kernel_record(
                cell,
                variant="track15_only",
                K=K_track15,
                n_features=cell.track15.shape[1],
                comparison_target="saved_track2_rbf",
                target_kernel=K_track2_rbf,
            ),
            kernel_record(
                cell,
                variant="full_integrated",
                K=K_full,
                n_features=cell.full_integrated.shape[1],
                comparison_target="saved_track2_rbf",
                target_kernel=K_track2_rbf,
            ),
            kernel_record(
                cell,
                variant="hadamard_track2_track15",
                K=K_hadamard,
                n_features=cell.track2.shape[1] + cell.track15.shape[1],
                kernel_mode="hadamard_rbf_track2_track15",
                comparison_target="track15_only",
                target_kernel=K_track15,
            ),
            kernel_record(
                cell,
                variant="raw_mean_cls_rbf",
                K=K_raw_cls,
                n_features=mean_cls.shape[1],
                comparison_target="exact_kernel_oracle",
                target_kernel=K_exact,
            ),
            kernel_record(
                cell,
                variant="exact_kernel_oracle",
                K=K_exact,
                n_features=mean_cls.shape[1],
                kernel_mode=f"exact_{cell.kernel}_on_mean_cls",
            ),
            direct_feature_cluster_record(
                cell,
                variant="raw_mean_cls_direct_kmeans",
                features=mean_cls,
            ),
            direct_feature_cluster_record(
                cell,
                variant="saved_track2_direct_kmeans",
                features=cell.track2,
            ),
        ]
    )

    for dim in rks_dims:
        for rks_seed in rks_seeds:
            Z = rks_features(mean_cls, kernel=cell.kernel, dim=dim, seed=rks_seed)
            K_rks_dot = rks_approx_kernel(Z)
            K_rks_downstream = rbf_kernel_from_features(Z)
            records.append(
                kernel_record(
                    cell,
                    variant=f"rks_dot_dim{dim}_seed{rks_seed}",
                    K=K_rks_dot,
                    n_features=dim,
                    kernel_mode=f"rks_dot_approx_{cell.kernel}",
                    comparison_target="exact_kernel_oracle",
                    target_kernel=K_exact,
                )
            )
            records.append(
                kernel_record(
                    cell,
                    variant=f"rks_downstream_rbf_dim{dim}_seed{rks_seed}",
                    K=K_rks_downstream,
                    n_features=dim,
                    kernel_mode="rbf_on_rks_coordinates",
                    comparison_target="exact_kernel_oracle",
                    target_kernel=K_exact,
                )
            )

            mix_records = observer_mixing_records(cell, dim=dim, rks_seed=rks_seed, exact=K_exact)
            records.extend(mix_records)

    return records


def observer_mixing_records(
    cell: CellArtifacts,
    *,
    dim: int,
    rks_seed: int,
    exact: np.ndarray,
) -> list[dict[str, Any]]:
    n, b, h = cell.cls_per_bot.shape
    flat = cell.cls_per_bot.reshape(n * b, h)
    mean, std, keep = _fit_standardizer(flat)
    flat_z = _apply_standardizer(flat, mean, std, keep)
    sigma = _median_sigma_from_d2(_pairwise_sq_dists(flat_z))

    mean_raw = cell.cls_per_bot.mean(axis=1)
    z_mix_then_project = rks_features_with_standardizer(
        mean_raw,
        kernel=cell.kernel,
        dim=dim,
        seed=rks_seed,
        mean=mean,
        std=std,
        keep=keep,
        sigma=sigma,
    )
    z_each = rks_features_with_standardizer(
        flat,
        kernel=cell.kernel,
        dim=dim,
        seed=rks_seed,
        mean=mean,
        std=std,
        keep=keep,
        sigma=sigma,
    ).reshape(n, b, dim)
    z_project_then_mix = z_each.mean(axis=1)

    K_mix_then_project = rks_approx_kernel(z_mix_then_project)
    K_project_then_mix = rks_approx_kernel(z_project_then_mix)
    records = [
        kernel_record(
            cell,
            variant=f"mix_then_project_dim{dim}_seed{rks_seed}",
            K=K_mix_then_project,
            n_features=dim,
            kernel_mode="observer_mix_then_rks_project",
            comparison_target="exact_kernel_oracle",
            target_kernel=exact,
        ),
        kernel_record(
            cell,
            variant=f"project_then_mix_dim{dim}_seed{rks_seed}",
            K=K_project_then_mix,
            n_features=dim,
            kernel_mode="observer_rks_project_then_mix",
            comparison_target="mix_then_project",
            target_kernel=K_mix_then_project,
        ),
    ]
    records.append(
        {
            "record_type": "mixing_commutativity",
            "cell_id": cell.cell_id,
            "kernel": cell.kernel,
            "seed": cell.seed,
            "variant": f"project_mix_vs_mix_project_dim{dim}_seed{rks_seed}",
            "kernel_mode": "observer_mixing_order_comparison",
            "label_column": cell.label_column,
            "n_samples": len(cell.labels),
            "n_features": dim,
            "comparison_target": "mix_then_project",
            "kernel_corr": kernel_correlation(K_project_then_mix, K_mix_then_project),
            "kernel_spearman": kernel_spearman(K_project_then_mix, K_mix_then_project),
            "relative_frobenius_error": relative_frobenius_error(
                K_project_then_mix, K_mix_then_project
            ),
            "kernel_mae": float(np.mean(np.abs(K_project_then_mix - K_mix_then_project))),
            "kernel_rmse": float(
                np.sqrt(np.mean((K_project_then_mix - K_mix_then_project) ** 2))
            ),
            "diag_mean_abs_error": float(
                np.mean(np.abs(np.diag(K_project_then_mix - K_mix_then_project)))
            ),
            "project_then_mix_nmi": records[1].get("cluster_nmi"),
            "mix_then_project_nmi": records[0].get("cluster_nmi"),
            "project_then_mix_nn": records[1].get("nearest_neighbor_label_agreement"),
            "mix_then_project_nn": records[0].get("nearest_neighbor_label_agreement"),
        }
    )
    return records


def _group_records(records: Sequence[Mapping[str, Any]], variant: str) -> list[Mapping[str, Any]]:
    return [row for row in records if row.get("variant") == variant]


def _variant_prefix(records: Sequence[Mapping[str, Any]], prefix: str) -> list[Mapping[str, Any]]:
    return [row for row in records if str(row.get("variant", "")).startswith(prefix)]


def _cell_variant_map(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row.get("cell_id")), str(row.get("variant"))): row
        for row in records
        if row.get("record_type") == "variant_metric"
    }


def _mean_delta_by_cell(
    records: Sequence[Mapping[str, Any]],
    *,
    left_variant: str,
    right_variant: str,
    metric: str,
) -> Optional[float]:
    by_key = _cell_variant_map(records)
    deltas: list[float] = []
    for cell_id, variant in list(by_key):
        if variant != left_variant:
            continue
        left = by_key.get((cell_id, left_variant), {})
        right = by_key.get((cell_id, right_variant), {})
        left_v = _safe_float(left.get(metric))
        right_v = _safe_float(right.get(metric))
        if left_v is not None and right_v is not None:
            deltas.append(left_v - right_v)
    return _mean(deltas)


def _status_from_thresholds(value: Optional[float], *, supported: float, partial: float) -> str:
    if value is None:
        return "UNTESTED"
    if value >= supported:
        return "SUPPORTED"
    if value >= partial:
        return "PARTIAL"
    return "UNSUPPORTED"


def summarize_question_results(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_dim_rks = [
        row
        for row in records
        if str(row.get("variant", "")).startswith("rks_dot_dim2048_")
        and row.get("comparison_target") == "exact_kernel_oracle"
    ]
    all_rks = [
        row
        for row in records
        if str(row.get("variant", "")).startswith("rks_dot_dim")
        and row.get("comparison_target") == "exact_kernel_oracle"
    ]
    rks_corr_2048 = _mean(row.get("kernel_corr") for row in max_dim_rks)
    rks_spearman_2048 = _mean(row.get("kernel_spearman") for row in max_dim_rks)
    rks_error_2048 = _mean(row.get("relative_frobenius_error") for row in max_dim_rks)
    rks_mae_2048 = _mean(row.get("kernel_mae") for row in max_dim_rks)
    rks_rmse_2048 = _mean(row.get("kernel_rmse") for row in max_dim_rks)
    rks_corr_all = _mean(row.get("kernel_corr") for row in all_rks)

    saved_track2 = _group_records(records, "saved_track2_rbf")
    saved_track2_dot = _group_records(records, "saved_track2_dot")
    track15 = _group_records(records, "track15_only")
    full = _group_records(records, "full_integrated")
    hadamard = _group_records(records, "hadamard_track2_track15")
    exact = _group_records(records, "exact_kernel_oracle")
    raw = _group_records(records, "raw_mean_cls_rbf")

    saved_track2_nmi = _mean(row.get("cluster_nmi") for row in saved_track2)
    saved_track2_dot_nmi = _mean(row.get("cluster_nmi") for row in saved_track2_dot)
    saved_track2_nn = _mean(row.get("nearest_neighbor_label_agreement") for row in saved_track2)
    saved_track2_dot_nn = _mean(
        row.get("nearest_neighbor_label_agreement") for row in saved_track2_dot
    )
    saved_track2_sep = _mean(row.get("same_different_similarity_separation") for row in saved_track2)
    track15_nmi = _mean(row.get("cluster_nmi") for row in track15)
    track15_nn = _mean(row.get("nearest_neighbor_label_agreement") for row in track15)
    full_nmi = _mean(row.get("cluster_nmi") for row in full)
    exact_nmi = _mean(row.get("cluster_nmi") for row in exact)
    raw_nmi = _mean(row.get("cluster_nmi") for row in raw)
    hadamard_nmi = _mean(row.get("cluster_nmi") for row in hadamard)
    hadamard_nn = _mean(row.get("nearest_neighbor_label_agreement") for row in hadamard)

    hadamard_minus_track15_nmi = _mean_delta_by_cell(
        records,
        left_variant="hadamard_track2_track15",
        right_variant="track15_only",
        metric="cluster_nmi",
    )
    full_minus_track15_nmi = _mean_delta_by_cell(
        records,
        left_variant="full_integrated",
        right_variant="track15_only",
        metric="cluster_nmi",
    )
    saved_track2_minus_track15_nmi = _mean_delta_by_cell(
        records,
        left_variant="saved_track2_rbf",
        right_variant="track15_only",
        metric="cluster_nmi",
    )
    saved_track2_dot_minus_track15_nmi = _mean_delta_by_cell(
        records,
        left_variant="saved_track2_dot",
        right_variant="track15_only",
        metric="cluster_nmi",
    )
    saved_track2_minus_track15_nn = _mean_delta_by_cell(
        records,
        left_variant="saved_track2_rbf",
        right_variant="track15_only",
        metric="nearest_neighbor_label_agreement",
    )
    hadamard_minus_track15_nn = _mean_delta_by_cell(
        records,
        left_variant="hadamard_track2_track15",
        right_variant="track15_only",
        metric="nearest_neighbor_label_agreement",
    )
    saved_track2_minus_raw_nmi = _mean_delta_by_cell(
        records,
        left_variant="saved_track2_rbf",
        right_variant="raw_mean_cls_rbf",
        metric="cluster_nmi",
    )
    saved_track2_dot_minus_raw_nmi = _mean_delta_by_cell(
        records,
        left_variant="saved_track2_dot",
        right_variant="raw_mean_cls_rbf",
        metric="cluster_nmi",
    )

    robustness_cells = defaultdict(list)
    for row in saved_track2:
        robustness_cells[str(row.get("kernel"))].append(row.get("cluster_nmi"))
    per_kernel_nmi = {kernel: _mean(values) for kernel, values in sorted(robustness_cells.items())}
    robust_kernel_pass_rate = _rate(
        value is not None and value >= 0.50 for value in per_kernel_nmi.values()
    )

    mixing_rows = [row for row in records if row.get("record_type") == "mixing_commutativity"]
    mixing_corr_2048 = _mean(
        row.get("kernel_corr")
        for row in mixing_rows
        if "dim2048_" in str(row.get("variant", ""))
    )
    mixing_spearman_2048 = _mean(
        row.get("kernel_spearman")
        for row in mixing_rows
        if "dim2048_" in str(row.get("variant", ""))
    )
    mixing_err_2048 = _mean(
        row.get("relative_frobenius_error")
        for row in mixing_rows
        if "dim2048_" in str(row.get("variant", ""))
    )

    return {
        "q1_rks_approximates_exact_kernel": {
            "status": _status_from_thresholds(rks_corr_2048, supported=0.85, partial=0.70),
            "mean_kernel_corr_dim2048": rks_corr_2048,
            "mean_kernel_spearman_dim2048": rks_spearman_2048,
            "mean_relative_frobenius_error_dim2048": rks_error_2048,
            "mean_kernel_mae_dim2048": rks_mae_2048,
            "mean_kernel_rmse_dim2048": rks_rmse_2048,
            "mean_kernel_corr_all_dims": rks_corr_all,
            "interpretation": (
                "RKS is treated as faithful when high-dimensional random features correlate "
                "strongly with the exact kernel upper triangle."
            ),
        },
        "q2_track2_preserves_planted_synthetic_structure": {
            "status": _status_from_thresholds(saved_track2_nmi, supported=0.50, partial=0.35),
            "mean_saved_track2_nmi": saved_track2_nmi,
            "mean_saved_track2_nn": saved_track2_nn,
            "mean_saved_track2_separation": saved_track2_sep,
            "per_kernel_nmi": per_kernel_nmi,
        },
        "q3_removing_track2_damages_recovery": {
            "status": (
                "SUPPORTED"
                if hadamard_minus_track15_nmi is not None and hadamard_minus_track15_nmi >= 0.03
                else "PARTIAL"
                if (
                    (hadamard_minus_track15_nmi is not None and hadamard_minus_track15_nmi >= 0.0)
                    or (
                        hadamard_minus_track15_nn is not None
                        and hadamard_minus_track15_nn >= 0.03
                    )
                )
                else "UNSUPPORTED"
            ),
            "mean_hadamard_minus_track15_nmi": hadamard_minus_track15_nmi,
            "mean_hadamard_minus_track15_nn": hadamard_minus_track15_nn,
            "mean_hadamard_nmi": hadamard_nmi,
            "mean_hadamard_nn": hadamard_nn,
            "mean_track15_nmi": track15_nmi,
            "mean_track15_nn": track15_nn,
            "mean_full_minus_track15_nmi": full_minus_track15_nmi,
            "interpretation": (
                "Positive NMI deltas mean Track 2 improves global cluster recovery. Positive NN deltas "
                "with negative NMI mean Track 2 sharpens local adjacency but fragments global clusters."
            ),
        },
        "q4_track2_adds_something_track15_cannot": {
            "status": (
                "SUPPORTED"
                if (
                    saved_track2_dot_minus_track15_nmi is not None
                    and saved_track2_dot_minus_track15_nmi >= 0.03
                )
                else "PARTIAL"
                if (
                    (
                        saved_track2_dot_minus_track15_nmi is not None
                        and saved_track2_dot_minus_track15_nmi >= -0.02
                    )
                    or (
                        saved_track2_minus_track15_nn is not None
                        and saved_track2_minus_track15_nn >= 0.02
                    )
                )
                else "UNSUPPORTED"
            ),
            "mean_saved_track2_minus_track15_nmi": saved_track2_minus_track15_nmi,
            "mean_saved_track2_dot_minus_track15_nmi": saved_track2_dot_minus_track15_nmi,
            "mean_saved_track2_minus_track15_nn": saved_track2_minus_track15_nn,
            "mean_saved_track2_nmi": saved_track2_nmi,
            "mean_saved_track2_dot_nmi": saved_track2_dot_nmi,
            "mean_saved_track2_nn": saved_track2_nn,
            "mean_saved_track2_dot_nn": saved_track2_dot_nn,
            "mean_track15_nmi": track15_nmi,
            "mean_track15_nn": track15_nn,
        },
        "q5_survives_seed_and_kernel_changes": {
            "status": "SUPPORTED" if robust_kernel_pass_rate >= 1.0 else "PARTIAL" if robust_kernel_pass_rate > 0 else "UNSUPPORTED",
            "kernel_pass_rate_at_nmi_ge_0_50": robust_kernel_pass_rate,
            "per_kernel_nmi": per_kernel_nmi,
        },
        "q6_project_then_mix_close_to_mix_then_project": {
            "status": _status_from_thresholds(mixing_corr_2048, supported=0.85, partial=0.65),
            "mean_kernel_corr_dim2048": mixing_corr_2048,
            "mean_kernel_spearman_dim2048": mixing_spearman_2048,
            "mean_relative_frobenius_error_dim2048": mixing_err_2048,
            "interpretation": (
                "This tests the claimed isomorphic communicativity. Low values mean observer "
                "projection order changes the geometry and must be treated as a real branch choice."
            ),
        },
        "q7_baselines_simple_geometry": {
            "status": (
                "SUPPORTED"
                if (
                    saved_track2_minus_raw_nmi is not None
                    and saved_track2_dot_minus_raw_nmi is not None
                    and saved_track2_minus_raw_nmi >= 0.0
                    and saved_track2_dot_minus_raw_nmi >= 0.0
                )
                else "PARTIAL"
                if (
                    saved_track2_dot_minus_raw_nmi is not None
                    and saved_track2_dot_minus_raw_nmi >= 0.0
                )
                else "UNSUPPORTED"
            ),
            "mean_saved_track2_minus_raw_cls_nmi": saved_track2_minus_raw_nmi,
            "mean_saved_track2_dot_minus_raw_cls_nmi": saved_track2_dot_minus_raw_nmi,
            "mean_raw_cls_nmi": raw_nmi,
            "mean_exact_oracle_nmi": exact_nmi,
            "mean_full_integrated_nmi": full_nmi,
            "mean_hadamard_nmi": hadamard_nmi,
        },
    }


def summarize_variants(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("record_type") == "variant_metric":
            grouped[str(row.get("variant"))].append(row)
    summary = []
    for variant, rows in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "n_records": len(rows),
                "mean_nmi": _mean(row.get("cluster_nmi") for row in rows),
                "median_nmi": _median(row.get("cluster_nmi") for row in rows),
                "mean_ari": _mean(row.get("cluster_ari") for row in rows),
                "mean_nn_agreement": _mean(row.get("nearest_neighbor_label_agreement") for row in rows),
                "mean_separation": _mean(
                    row.get("same_different_similarity_separation") for row in rows
                ),
                "mean_kernel_corr": _mean(row.get("kernel_corr") for row in rows),
                "mean_kernel_spearman": _mean(row.get("kernel_spearman") for row in rows),
                "mean_relative_frobenius_error": _mean(
                    row.get("relative_frobenius_error") for row in rows
                ),
                "mean_kernel_mae": _mean(row.get("kernel_mae") for row in rows),
                "mean_kernel_rmse": _mean(row.get("kernel_rmse") for row in rows),
            }
        )
    return summary


def build_artifact(
    *,
    synthetic_root: Path,
    kernels: Sequence[str],
    seeds: Sequence[int],
    rks_dims: Sequence[int],
    rks_seeds: Sequence[int],
    label_column: Optional[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    for kernel in kernels:
        for seed in seeds:
            cell = _load_cell(
                synthetic_root,
                kernel=kernel,
                seed=seed,
                label_column=label_column,
            )
            records.extend(build_cell_records(cell, rks_dims=rks_dims, rks_seeds=rks_seeds))
            cell_summaries.append(
                {
                    "cell_id": cell.cell_id,
                    "kernel": kernel,
                    "seed": seed,
                    "run_dir": str(cell.run_dir),
                    "label_column": cell.label_column,
                    "label_counts": dict(sorted(Counter(cell.labels).items())),
                    "track2_shape": list(cell.track2.shape),
                    "track15_shape": list(cell.track15.shape),
                    "full_integrated_shape": list(cell.full_integrated.shape),
                    "cls_per_bot_shape": list(cell.cls_per_bot.shape),
                }
            )

    question_results = summarize_question_results(records)
    supported = [
        key for key, value in question_results.items() if value.get("status") == "SUPPORTED"
    ]
    partial = [key for key, value in question_results.items() if value.get("status") == "PARTIAL"]
    unsupported = [
        key for key, value in question_results.items() if value.get("status") == "UNSUPPORTED"
    ]
    untested = [key for key, value in question_results.items() if value.get("status") == "UNTESTED"]

    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_root": str(synthetic_root),
        "kernels": list(kernels),
        "seeds": [int(seed) for seed in seeds],
        "rks_dims": [int(dim) for dim in rks_dims],
        "rks_seeds": [int(seed) for seed in rks_seeds],
        "label_column_requested": label_column,
        "claim_scope": "direct_synthetic_artifact_ablation_not_real_corpus_truth_claim",
        "foundation_audit_consumable": True,
        "safe_for_thesis_claim": False,
        "thresholds": {
            "rks_fidelity_supported_kernel_corr": 0.85,
            "rks_fidelity_partial_kernel_corr": 0.70,
            "track2_planted_structure_supported_nmi": 0.50,
            "track2_planted_structure_partial_nmi": 0.35,
            "track2_removal_damage_supported_delta_nmi": 0.03,
            "track2_removal_damage_partial_delta_nmi": 0.0,
            "track2_removal_damage_partial_delta_nn": 0.03,
            "track2_additive_supported_delta_nmi": 0.03,
            "track2_additive_partial_delta_nmi": 0.0,
            "track2_additive_partial_delta_nn": 0.02,
            "project_mix_supported_kernel_corr": 0.85,
            "project_mix_partial_kernel_corr": 0.65,
            "kernel_robustness_nmi_floor": 0.50,
        },
        "variants_evaluated": sorted(
            {
                str(row.get("variant"))
                for row in records
                if row.get("record_type") == "variant_metric"
            }
        ),
        "cell_summaries": cell_summaries,
        "question_results": question_results,
        "supported_questions": supported,
        "partial_questions": partial,
        "unsupported_questions": unsupported,
        "untested_questions": untested,
        "variant_summary": summarize_variants(records),
        "records": records,
    }


def _csv_fieldnames(records: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "record_type",
        "cell_id",
        "kernel",
        "seed",
        "variant",
        "kernel_mode",
        "label_column",
        "n_samples",
        "n_features",
        "comparison_target",
        "kernel_corr",
        "kernel_spearman",
        "relative_frobenius_error",
        "kernel_mae",
        "kernel_rmse",
        "diag_mean_abs_error",
        "same_different_similarity_separation",
        "pair_auc_same_vs_different",
        "nearest_neighbor_label_agreement",
        "cluster_nmi",
        "cluster_ari",
        "cluster_status",
        "project_then_mix_nmi",
        "mix_then_project_nmi",
        "project_then_mix_nn",
        "mix_then_project_nn",
    ]
    observed = {key for row in records for key in row.keys()}
    return preferred + sorted(observed - set(preferred))


def write_outputs(artifact: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "track2_necessity_ablation.json"
    csv_path = output_dir / "track2_necessity_ablation_records.csv"
    md_path = output_dir / "track2_necessity_ablation_summary.md"
    matrix_path = output_dir / "ablation_matrix.json"

    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = artifact.get("records") if isinstance(artifact.get("records"), list) else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(records), extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    question_results = artifact.get("question_results", {})
    q3 = question_results.get("q3_removing_track2_damages_recovery", {})
    q2 = question_results.get("q2_track2_preserves_planted_synthetic_structure", {})
    q4 = question_results.get("q4_track2_adds_something_track15_cannot", {})
    matrix = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": "ablation_matrix",
        "source_diagnostic_type": DIAGNOSTIC_TYPE,
        "source_artifact": str(json_path),
        "required_modes_present": True,
        "records": [
            {
                "ablation": "track2_removed",
                "component": "track2",
                "effect_pass": q3.get("status") == "SUPPORTED",
                "pass": q3.get("status") == "SUPPORTED",
                "thesis_safe": False,
                "mean_hadamard_minus_track15_nmi": q3.get("mean_hadamard_minus_track15_nmi"),
                "mean_full_minus_track15_nmi": q3.get("mean_full_minus_track15_nmi"),
                "interpretation": q3.get("interpretation"),
            },
            {
                "ablation": "track2_only_synthetic_structure",
                "component": "track2_foundation_signal",
                "effect_pass": q2.get("status") == "SUPPORTED",
                "pass": q2.get("status") == "SUPPORTED",
                "thesis_safe": False,
                "mean_saved_track2_nmi": q2.get("mean_saved_track2_nmi"),
                "mean_saved_track2_nn": q2.get("mean_saved_track2_nn"),
            },
            {
                "ablation": "track2_vs_track15_additive_value",
                "component": "track2_additive_signal",
                "effect_pass": q4.get("status") == "SUPPORTED",
                "pass": q4.get("status") == "SUPPORTED",
                "thesis_safe": False,
                "mean_saved_track2_minus_track15_nmi": q4.get(
                    "mean_saved_track2_minus_track15_nmi"
                ),
            },
        ],
    }
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track 2 Necessity Ablation",
        "",
        f"- Generated: `{artifact.get('generated_at')}`",
        f"- Cells: `{len(artifact.get('cell_summaries', []))}`",
        f"- Records: `{len(records)}`",
        f"- Supported: `{', '.join(artifact.get('supported_questions', [])) or 'none'}`",
        f"- Partial: `{', '.join(artifact.get('partial_questions', [])) or 'none'}`",
        f"- Unsupported: `{', '.join(artifact.get('unsupported_questions', [])) or 'none'}`",
        "",
        "## Question Results",
        "",
    ]
    for question, result in artifact.get("question_results", {}).items():
        lines.append(f"### {question}")
        lines.append("")
        lines.append(f"- Status: `{result.get('status')}`")
        for key, value in result.items():
            if key in {"status", "interpretation"}:
                continue
            lines.append(f"- {key}: `{value}`")
        if result.get("interpretation"):
            lines.append(f"- Interpretation: {result['interpretation']}")
        lines.append("")
    lines.extend(["## Top Variant Summary", ""])
    variant_summary = artifact.get("variant_summary", [])
    ranked = sorted(
        variant_summary,
        key=lambda row: (
            _safe_float(row.get("mean_nmi")) if _safe_float(row.get("mean_nmi")) is not None else -1.0
        ),
        reverse=True,
    )[:20]
    for row in ranked:
        lines.append(
            "- `{variant}` mean_nmi=`{mean_nmi}` mean_nn=`{mean_nn_agreement}` mean_corr=`{mean_kernel_corr}`".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "ablation_matrix": str(matrix_path),
    }


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / "track2_necessity_ablation" / f"current_{stamp}"


def _int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _str_list(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--kernels", default="rbf,matern,imq")
    parser.add_argument("--seeds", default="42,420,4200")
    parser.add_argument("--rks-dims", default="64,128,256,512,1024,2048")
    parser.add_argument("--rks-seeds", default="42,420,4200")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    artifact = build_artifact(
        synthetic_root=args.synthetic_root,
        kernels=_str_list(args.kernels),
        seeds=_int_list(args.seeds),
        rks_dims=_int_list(args.rks_dims),
        rks_seeds=_int_list(args.rks_seeds),
        label_column=args.label_column,
    )
    written = write_outputs(artifact, args.output_dir)
    print(
        json.dumps(
            {
                **written,
                "supported": artifact["supported_questions"],
                "partial": artifact["partial_questions"],
                "unsupported": artifact["unsupported_questions"],
                "untested": artifact["untested_questions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
