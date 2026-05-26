#!/usr/bin/env python3
"""Validate whether manifold geometry recovers source/outlet structure.

This is a lightweight, artifact-consuming diagnostic. It does not rerun the
pipeline. Given one or more completed run leaves, it loads article features and
metadata, filters to repeated source labels, and compares source cohesion against
deterministic shuffled-label nulls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_COLUMNS = ("source", "publication", "publisher", "outlet")
FEATURE_CANDIDATES = (
    "features.npy",
    "dirichlet_fused.npy",
    "checkpoints/batch/T3_dirichlet_fused.npy",
    "checkpoints/batch/T2_kernel_projections.npz",
    "checkpoints/batch/T1.5_spectral_state.npz",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _read_metadata(run_dir: Path) -> Tuple[pd.DataFrame, str]:
    candidates = (
        run_dir / "article_metadata.csv",
        run_dir / "MONOLITH_DATA.csv",
    )
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)
    json_path = run_dir / "article_metadata.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload), str(json_path)
        if isinstance(payload, dict):
            for key in ("articles", "metadata", "rows"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return pd.DataFrame(rows), str(json_path)
    raise FileNotFoundError(f"No article metadata found under {run_dir}")


def _select_source_column(df: pd.DataFrame) -> Optional[str]:
    for col in SOURCE_COLUMNS:
        if col not in df.columns:
            continue
        values = [
            str(v).strip()
            for v in df[col].tolist()
            if str(v).strip() and str(v).strip().lower() not in {"nan", "none", "unknown", "missing"}
        ]
        if len(set(values)) >= 2:
            return col
    return None


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


def _load_npz(path: Path) -> Tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=False)
    preferred = [key for key in data.files if str(key).startswith("z_")]
    for key in [*preferred, *data.files]:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[0] > 1:
            return arr.astype(np.float64, copy=False), key
    raise ValueError(f"No 2D feature array found in {path}")


def _resolve_feature_path(run_dir: Path, feature_path: Optional[Path]) -> Tuple[np.ndarray, str, str]:
    run_dir = Path(run_dir)
    if feature_path is not None:
        candidates = [feature_path if feature_path.is_absolute() else run_dir / feature_path]
    else:
        candidates = [run_dir / rel for rel in FEATURE_CANDIDATES]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        if path.suffix.lower() == ".npy":
            return _load_npy(path), str(path), ""
        if path.suffix.lower() == ".npz":
            arr, key = _load_npz(path)
            return arr, str(path), key
    raise FileNotFoundError(f"No feature matrix found under {run_dir}")


def _standardize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    z = (x - mean) / std
    return np.nan_to_num(z, copy=False)


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = _standardize_features(x)
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(dist2)


def _cohesion_metrics(x: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    labels = [str(v) for v in labels]
    n = len(labels)
    distances = _pairwise_distances(x)
    same: List[float] = []
    diff: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same.append(float(distances[i, j]))
            else:
                diff.append(float(distances[i, j]))

    same_mean = float(np.mean(same)) if same else None
    diff_mean = float(np.mean(diff)) if diff else None
    effect = (diff_mean - same_mean) if same_mean is not None and diff_mean is not None else None
    all_pair = np.asarray([*same, *diff], dtype=np.float64)
    scale = float(np.std(all_pair)) if all_pair.size else 0.0
    standardized_effect = (float(effect) / scale) if effect is not None and scale > 1e-12 else None
    ratio = (diff_mean / same_mean) if same_mean not in (None, 0.0) and diff_mean is not None else None

    nearest_hits = 0
    if n > 1:
        masked = distances.copy()
        np.fill_diagonal(masked, np.inf)
        nearest = np.argmin(masked, axis=1)
        nearest_hits = int(sum(labels[i] == labels[int(nearest[i])] for i in range(n)))
    source_counts = Counter(labels)
    expected_nn = (
        sum(count * (count - 1) for count in source_counts.values()) / float(n * (n - 1))
        if n > 1
        else None
    )
    nn_agreement = nearest_hits / float(n) if n else None
    return {
        "n_articles": n,
        "n_sources": len(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "same_source_pair_count": len(same),
        "different_source_pair_count": len(diff),
        "same_source_mean_distance": same_mean,
        "different_source_mean_distance": diff_mean,
        "source_distance_effect": effect,
        "standardized_source_distance_effect": standardized_effect,
        "different_over_same_distance_ratio": ratio,
        "nearest_neighbor_source_agreement": nn_agreement,
        "expected_random_neighbor_agreement": expected_nn,
        "nearest_neighbor_source_excess": (
            nn_agreement - expected_nn if nn_agreement is not None and expected_nn is not None else None
        ),
    }


def _null_distribution(
    x: np.ndarray,
    labels: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(list(labels), dtype=object)
    effects: List[float] = []
    nn_excess: List[float] = []
    for _ in range(int(permutations)):
        shuffled = labels_arr.copy()
        rng.shuffle(shuffled)
        metrics = _cohesion_metrics(x, shuffled.tolist())
        effect = metrics.get("source_distance_effect")
        excess = metrics.get("nearest_neighbor_source_excess")
        if effect is not None:
            effects.append(float(effect))
        if excess is not None:
            nn_excess.append(float(excess))
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "effect_mean": float(np.mean(effects)) if effects else None,
        "effect_std": float(np.std(effects)) if effects else None,
        "effect_values": effects,
        "nn_excess_mean": float(np.mean(nn_excess)) if nn_excess else None,
        "nn_excess_std": float(np.std(nn_excess)) if nn_excess else None,
        "nn_excess_values": nn_excess,
    }


def _right_tail_pvalue(observed: Optional[float], null_values: Sequence[float]) -> Optional[float]:
    if observed is None or not null_values:
        return None
    return (1.0 + sum(float(v) >= float(observed) for v in null_values)) / (1.0 + len(null_values))


def _filter_repeated_sources(
    x: np.ndarray,
    labels: Sequence[str],
    *,
    min_source_count: int,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    counts = Counter(str(v) for v in labels)
    keep = [idx for idx, value in enumerate(labels) if counts[str(value)] >= int(min_source_count)]
    return x[keep, :], [str(labels[idx]) for idx in keep], dict(sorted(counts.items()))


def validate_run_dir(
    run_dir: Path,
    *,
    feature_path: Optional[Path] = None,
    min_source_count: int = 3,
    min_sources: int = 3,
    min_articles: int = 18,
    permutations: int = 200,
    seed: int = 42,
    min_effect: float = 0.05,
    min_nn_excess: float = 0.05,
    max_pvalue: float = 0.05,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    metadata, metadata_path = _read_metadata(run_dir)
    source_column = _select_source_column(metadata)
    if source_column is None:
        return {
            "run_dir": str(run_dir),
            "status": "INSUFFICIENT_METADATA",
            "pass": False,
            "thesis_safe": False,
            "failure_reasons": ["missing_repeated_source_column"],
            "metadata_path": metadata_path,
        }

    features, resolved_feature_path, feature_key = _resolve_feature_path(run_dir, feature_path)
    n = min(len(metadata), int(features.shape[0]))
    metadata = metadata.iloc[:n].copy()
    features = features[:n, :]
    raw_labels = [str(v).strip() for v in metadata[source_column].tolist()]
    filtered_x, filtered_labels, raw_source_counts = _filter_repeated_sources(
        features,
        raw_labels,
        min_source_count=min_source_count,
    )
    filtered_counts = Counter(filtered_labels)
    reasons: List[str] = []
    if filtered_x.shape[0] < int(min_articles):
        reasons.append("insufficient_repeated_source_articles")
    if len(filtered_counts) < int(min_sources):
        reasons.append("insufficient_repeated_source_count")

    base_payload = {
        "run_dir": str(run_dir),
        "metadata_path": metadata_path,
        "feature_path": resolved_feature_path,
        "feature_key": feature_key,
        "source_column": source_column,
        "raw_article_count": int(n),
        "raw_source_count": int(len(raw_source_counts)),
        "raw_source_counts": raw_source_counts,
        "filtered_article_count": int(filtered_x.shape[0]),
        "filtered_source_count": int(len(filtered_counts)),
        "filtered_source_counts": dict(sorted(filtered_counts.items())),
        "thresholds": {
            "min_source_count": int(min_source_count),
            "min_sources": int(min_sources),
            "min_articles": int(min_articles),
            "min_effect": float(min_effect),
            "min_nn_excess": float(min_nn_excess),
            "max_pvalue": float(max_pvalue),
        },
    }
    if reasons:
        return {
            **base_payload,
            "status": "INSUFFICIENT_SOURCE_REPLICATION",
            "pass": False,
            "thesis_safe": False,
            "failure_reasons": reasons,
        }

    observed = _cohesion_metrics(filtered_x, filtered_labels)
    null = _null_distribution(filtered_x, filtered_labels, permutations=permutations, seed=seed)
    effect_p = _right_tail_pvalue(observed.get("source_distance_effect"), null.get("effect_values", []))
    nn_p = _right_tail_pvalue(observed.get("nearest_neighbor_source_excess"), null.get("nn_excess_values", []))
    pass_checks = {
        "effect_positive": (observed.get("source_distance_effect") or 0.0) >= float(min_effect),
        "effect_beats_null": effect_p is not None and effect_p <= float(max_pvalue),
        "nn_excess_positive": (observed.get("nearest_neighbor_source_excess") or 0.0) >= float(min_nn_excess),
        "nn_excess_beats_null": nn_p is not None and nn_p <= float(max_pvalue),
    }
    failure_reasons = [name for name, ok in pass_checks.items() if not ok]
    return {
        **base_payload,
        "status": "OK",
        "pass": not failure_reasons,
        "thesis_safe": not failure_reasons,
        "failure_reasons": failure_reasons,
        "observed": observed,
        "null": {
            key: value
            for key, value in null.items()
            if key not in {"effect_values", "nn_excess_values"}
        },
        "p_values": {
            "source_distance_effect": effect_p,
            "nearest_neighbor_source_excess": nn_p,
        },
        "pass_checks": pass_checks,
        "interpretation": (
            "Source-proxy validation passes only when same-source articles are closer than "
            "different-source articles and nearest-neighbor source agreement beats shuffled-label nulls. "
            "This validates editorial/source coherence, not political truth."
        ),
    }


def summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    pass_rows = [row for row in ok_rows if bool(row.get("pass"))]
    insufficient_rows = [row for row in rows if row.get("status") == "INSUFFICIENT_SOURCE_REPLICATION"]
    metadata_fail_rows = [row for row in rows if row.get("status") == "INSUFFICIENT_METADATA"]
    effects = [
        float((row.get("observed") or {}).get("source_distance_effect"))
        for row in ok_rows
        if (row.get("observed") or {}).get("source_distance_effect") is not None
    ]
    return {
        "schema_version": "1.0",
        "diagnostic_type": "source_proxy_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(rows),
        "ok_run_count": len(ok_rows),
        "passing_run_count": len(pass_rows),
        "insufficient_source_run_count": len(insufficient_rows),
        "insufficient_metadata_run_count": len(metadata_fail_rows),
        "pass_rate": len(pass_rows) / len(ok_rows) if ok_rows else 0.0,
        "mean_source_distance_effect": float(np.mean(effects)) if effects else None,
        "thesis_safe": bool(ok_rows) and len(pass_rows) == len(ok_rows) and not insufficient_rows and not metadata_fail_rows,
        "claim_boundary": {
            "source_proxy_validates_editorial_coherence_only": True,
            "not_a_bias_truth_label": True,
            "requires_repeated_sources": True,
        },
        "runs": list(rows),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--min-source-count", type=int, default=3)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--min-articles", type=int, default=18)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-effect", type=float, default=0.05)
    parser.add_argument("--min-nn-excess", type=float, default=0.05)
    parser.add_argument("--max-pvalue", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = [
        validate_run_dir(
            run_dir,
            feature_path=args.feature_path,
            min_source_count=args.min_source_count,
            min_sources=args.min_sources,
            min_articles=args.min_articles,
            permutations=args.permutations,
            seed=args.seed,
            min_effect=args.min_effect,
            min_nn_excess=args.min_nn_excess,
            max_pvalue=args.max_pvalue,
        )
        for run_dir in args.run_dir
    ]
    payload = summarize_results(rows)
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(payload.get("thesis_safe")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
