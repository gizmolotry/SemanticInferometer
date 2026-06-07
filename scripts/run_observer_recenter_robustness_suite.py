from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observer_local_recompute import (
    LOCAL_RECOMPUTE_DEFAULT_VARIANT,
    LOCAL_RECOMPUTE_MODE,
    LOCAL_RECOMPUTE_VARIANTS,
    load_primary_observer_payload,
)
from scripts.run_observer_recenter_meaning_probe import (
    DEFAULT_THRESHOLDS,
    _article_map,
    _clean_label,
    _json_safe,
    _load_json,
    _metadata_by_idx,
    _translated_anchor_view,
    aggregate_ideological_validation_suite,
    evaluate_label_geometry_tests,
    evaluate_observer_recenter,
    evaluate_run_dir,
    select_label_column,
)


ROBUSTNESS_SCHEMA_VERSION = "1.0"
DEFAULT_SYNTHETIC_ROOT = Path("outputs/experiments/runs/experiments_20260506_192553/synthetic")
DEFAULT_KERNELS = ("rbf", "matern", "imq")
DEFAULT_SEEDS = (42, 420, 4200)
LOCAL_VARIANT_BASELINES = {
    f"{LOCAL_RECOMPUTE_MODE}:{variant}": variant
    for variant in LOCAL_RECOMPUTE_VARIANTS
    if variant != LOCAL_RECOMPUTE_DEFAULT_VARIANT
}
SOURCE_PROXY_METRIC_BASELINE = f"{LOCAL_RECOMPUTE_MODE}:source_proxy_metric"
BASELINES = (
    "translation_only",
    "artifact_view",
    "raw_track2_pca",
    "cls_mean_pca",
    LOCAL_RECOMPUTE_MODE,
    *LOCAL_VARIANT_BASELINES.keys(),
    SOURCE_PROXY_METRIC_BASELINE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pca_project(features: np.ndarray, *, n_components: int = 2) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None, {"status": "DEGENERATE", "shape": list(arr.shape)}
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        return None, {"status": "SVD_FAILED", "error": str(exc), "shape": list(arr.shape)}
    k = int(min(max(n_components, 1), vt.shape[0]))
    projection = centered @ vt[:k].T
    if k < n_components:
        projection = np.pad(projection, ((0, 0), (0, n_components - k)), mode="constant")
    denom = float(np.sum(singular_values**2))
    evr = []
    if denom > 1e-12:
        evr = [float(v) for v in ((singular_values[:k] ** 2) / denom)]
    return projection[:, :n_components].astype(np.float64), {
        "status": "OK",
        "shape": list(arr.shape),
        "projection_shape": list(projection[:, :n_components].shape),
        "explained_variance_ratio": evr,
    }


def _matrix_to_anchor_articles(matrix: np.ndarray, anchor_idx: int) -> Dict[int, Dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {}
    anchor_idx = int(anchor_idx)
    if anchor_idx < 0 or anchor_idx >= arr.shape[0]:
        return {}
    shifted = arr[:, :2] - arr[anchor_idx : anchor_idx + 1, :2]
    return {
        int(idx): {"idx": int(idx), "x": float(row[0]), "y": float(row[1])}
        for idx, row in enumerate(shifted)
    }


def _label_representative_anchors(
    global_articles: Mapping[int, Mapping[str, Any]],
    metadata: Mapping[int, Mapping[str, Any]],
    label_col: Optional[str],
    label_counts: Mapping[str, int],
    min_label_count: int,
) -> List[Tuple[int, str]]:
    if not label_col:
        return []
    anchors: List[Tuple[int, str]] = []
    seen = set()
    for idx in sorted(global_articles):
        label = _clean_label(metadata.get(int(idx), {}).get(label_col))
        if not label or label in seen:
            continue
        if int(label_counts.get(label, 0) or 0) < int(min_label_count):
            continue
        anchors.append((int(idx), label))
        seen.add(label)
    return anchors


def _primary_gain(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("scale_normalized_label_gap_gain_over_translation")
    if value is None:
        value = row.get("label_gap_gain_over_translation")
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _summarize_observer_rows(
    observer_rows: Sequence[Mapping[str, Any]],
    *,
    label_col: Optional[str],
    label_basis: str,
    label_counts: Mapping[str, int],
    min_label_count: int,
) -> Dict[str, Any]:
    gains = [_primary_gain(row) for row in observer_rows]
    mean_gain = _mean(gains)
    pass_label = bool(
        mean_gain is not None
        and mean_gain >= float(DEFAULT_THRESHOLDS["min_label_gap_gain"])
    )
    suite = aggregate_ideological_validation_suite(
        observer_rows,
        label_column=label_col,
        label_basis=label_basis,
        label_counts=label_counts,
        min_label_count=min_label_count,
    )
    return {
        "status": "OK" if observer_rows else "NO_DATA",
        "observer_count": len(observer_rows),
        "label_contraction": {
            "status": "OK" if gains else "INSUFFICIENT_LABELS",
            "pass": pass_label,
            "primary_mean_label_gap_gain_over_translation": mean_gain,
            "min_label_gap_gain": DEFAULT_THRESHOLDS["min_label_gap_gain"],
            "n_observers_with_label_test": len([v for v in gains if v is not None]),
            "label_column": label_col,
            "label_basis": label_basis,
        },
        "ideological_validation_suite": suite,
        "safe_for_recenter_claim": bool(pass_label and suite.get("status") == "PASS"),
    }


def _evaluate_matrix_baseline(
    run_dir: Path,
    *,
    baseline_name: str,
    projection: Optional[np.ndarray],
    projection_diagnostics: Mapping[str, Any],
    min_label_count: int,
    preferred_label_column: Optional[str],
    label_mode: str,
) -> Dict[str, Any]:
    global_state = _load_json(run_dir / "MONOLITH.view_state.json")
    global_articles = _article_map(global_state)
    metadata, metadata_path = _metadata_by_idx(run_dir)
    label_col, label_counts, label_diagnostics = select_label_column(
        metadata,
        global_articles.keys(),
        min_label_count=min_label_count,
        preferred_label_column=preferred_label_column,
        label_mode=label_mode,
    )
    labels = {
        idx: _clean_label(row.get(label_col))
        for idx, row in metadata.items()
        if label_col and _clean_label(row.get(label_col))
    }
    anchors = _label_representative_anchors(
        global_articles,
        metadata,
        label_col,
        label_counts,
        min_label_count,
    )
    observer_rows: List[Dict[str, Any]] = []
    if projection is None:
        return {
            "baseline": baseline_name,
            "status": "NO_DATA",
            "metadata_path": metadata_path,
            "label_column": label_col,
            "label_basis": label_diagnostics["selected_basis"],
            "label_counts": label_counts,
            "projection_diagnostics": dict(projection_diagnostics),
            "observer_count": 0,
            "safe_for_recenter_claim": False,
            "observers": [],
        }
    for anchor_idx, anchor_label in anchors:
        if baseline_name == "translation_only":
            observer_articles = _translated_anchor_view(global_articles, anchor_idx)
        else:
            observer_articles = _matrix_to_anchor_articles(projection, anchor_idx)
        row = evaluate_observer_recenter(
            global_articles=global_articles,
            observer_articles=observer_articles,
            anchor_idx=anchor_idx,
            labels=labels,
            heldout_similarity=None,
        )
        row["baseline"] = baseline_name
        row["anchor_label"] = anchor_label
        row["label_column"] = label_col
        row["label_basis"] = label_diagnostics["selected_basis"]
        row["label_geometry_tests"] = evaluate_label_geometry_tests(
            global_articles=global_articles,
            observer_articles=observer_articles,
            anchor_idx=anchor_idx,
            labels=labels,
        )
        observer_rows.append(row)
    summary = _summarize_observer_rows(
        observer_rows,
        label_col=label_col,
        label_basis=label_diagnostics["selected_basis"],
        label_counts=label_counts,
        min_label_count=min_label_count,
    )
    return {
        "baseline": baseline_name,
        "metadata_path": metadata_path,
        "label_column": label_col,
        "label_basis": label_diagnostics["selected_basis"],
        "label_counts": label_counts,
        "projection_diagnostics": dict(projection_diagnostics),
        **summary,
        "observers": observer_rows,
    }


def _load_raw_track2_projection(run_dir: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    path = run_dir / "features.npy"
    if not path.exists():
        return None, {"status": "MISSING", "path": str(path)}
    try:
        features = np.load(path)
    except Exception as exc:
        return None, {"status": "LOAD_FAILED", "path": str(path), "error": str(exc)}
    projection, diag = _pca_project(features, n_components=2)
    diag["path"] = str(path)
    diag["basis"] = "features.npy_pca2"
    return projection, diag


def _load_cls_mean_projection(run_dir: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    payload, payload_path, error = load_primary_observer_payload(run_dir)
    if payload is None:
        return None, {"status": "MISSING", "error": error}
    cls = payload.get("cls_per_bot")
    try:
        import torch

        if torch.is_tensor(cls):
            cls = cls.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(cls, dtype=np.float64) if cls is not None else np.asarray([])
    if arr.ndim != 3:
        return None, {"status": "NO_CLS_PER_BOT", "payload_path": payload_path, "shape": list(arr.shape)}
    mean_cls = np.mean(arr, axis=1)
    projection, diag = _pca_project(mean_cls, n_components=2)
    diag["payload_path"] = payload_path
    diag["basis"] = "mean_cls_per_bot_pca2"
    return projection, diag


def _load_source_proxy_projection(
    run_dir: Path,
    *,
    preferred_label_column: Optional[str],
    min_label_count: int,
    label_mode: str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    global_state = _load_json(run_dir / "MONOLITH.view_state.json")
    global_articles = _article_map(global_state)
    metadata, metadata_path = _metadata_by_idx(run_dir)
    label_col, label_counts, label_diagnostics = select_label_column(
        metadata,
        global_articles.keys(),
        min_label_count=min_label_count,
        preferred_label_column=preferred_label_column,
        label_mode=label_mode,
    )
    if not label_col:
        return None, {
            "status": "NO_LABEL_COLUMN",
            "metadata_path": metadata_path,
            "label_diagnostics": label_diagnostics,
        }
    usable_labels = [
        label
        for label, count in sorted(label_counts.items())
        if int(count) >= int(min_label_count)
    ]
    if len(usable_labels) < 2:
        return None, {
            "status": "INSUFFICIENT_LABELS",
            "metadata_path": metadata_path,
            "label_column": label_col,
            "label_counts": label_counts,
        }
    label_to_pos = {
        label: (
            math.cos((2.0 * math.pi * idx) / float(len(usable_labels))),
            math.sin((2.0 * math.pi * idx) / float(len(usable_labels))),
        )
        for idx, label in enumerate(usable_labels)
    }
    rows: List[List[float]] = []
    for idx in sorted(global_articles):
        label = _clean_label(metadata.get(int(idx), {}).get(label_col))
        if label in label_to_pos:
            x, y = label_to_pos[label]
        else:
            x, y = 0.0, 0.0
        rows.append([float(x), float(y)])
    return np.asarray(rows, dtype=np.float64), {
        "status": "OK",
        "basis": "source_proxy_oracle_metric_circle",
        "metadata_path": metadata_path,
        "label_column": label_col,
        "label_counts": label_counts,
        "claim_boundary": "ablation_upper_bound_uses_registered_label_proxy_not_pure_local_recompute",
    }


def evaluate_baseline(run_dir: Path, baseline: str, *, min_label_count: int, preferred_label_column: Optional[str], label_mode: str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if baseline == SOURCE_PROXY_METRIC_BASELINE:
        projection, diag = _load_source_proxy_projection(
            run_dir,
            preferred_label_column=preferred_label_column,
            min_label_count=min_label_count,
            label_mode=label_mode,
        )
        result = _evaluate_matrix_baseline(
            run_dir,
            baseline_name=baseline,
            projection=projection,
            projection_diagnostics=diag,
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
        )
        result["local_recompute_variant"] = "source_proxy_metric"
        result["claim_boundary"] = "proxy_assisted_upper_bound_not_pure_local_recompute"
        result["proxy_oracle_baseline"] = True
        result["safe_for_recenter_claim"] = False
        return result
    if baseline == LOCAL_RECOMPUTE_MODE or baseline in LOCAL_VARIANT_BASELINES:
        local_variant = LOCAL_VARIANT_BASELINES.get(baseline, LOCAL_RECOMPUTE_DEFAULT_VARIANT)
        result = evaluate_run_dir(
            run_dir,
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
            recenter_mode=LOCAL_RECOMPUTE_MODE,
            local_recompute_variant=local_variant,
        )
        result["baseline"] = baseline
        result["local_recompute_variant"] = local_variant
        result["safe_for_recenter_claim"] = bool(
            (result.get("label_contraction") or {}).get("pass")
            and (result.get("ideological_validation_suite") or {}).get("status") == "PASS"
        )
        return result
    if baseline == "artifact_view":
        result = evaluate_run_dir(
            run_dir,
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
            recenter_mode="artifact_view",
        )
        result["baseline"] = "artifact_view"
        result["safe_for_recenter_claim"] = bool(
            (result.get("label_contraction") or {}).get("pass")
            and (result.get("ideological_validation_suite") or {}).get("status") == "PASS"
        )
        return result
    if baseline == "translation_only":
        global_state = _load_json(run_dir / "MONOLITH.view_state.json")
        global_articles = _article_map(global_state)
        projection = np.asarray([[row.get("x", 0.0), row.get("y", 0.0)] for _, row in sorted(global_articles.items())], dtype=np.float64)
        return _evaluate_matrix_baseline(
            run_dir,
            baseline_name=baseline,
            projection=projection,
            projection_diagnostics={"status": "OK", "basis": "global_view_translation_only"},
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
        )
    if baseline == "raw_track2_pca":
        projection, diag = _load_raw_track2_projection(run_dir)
        return _evaluate_matrix_baseline(
            run_dir,
            baseline_name=baseline,
            projection=projection,
            projection_diagnostics=diag,
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
        )
    if baseline == "cls_mean_pca":
        projection, diag = _load_cls_mean_projection(run_dir)
        return _evaluate_matrix_baseline(
            run_dir,
            baseline_name=baseline,
            projection=projection,
            projection_diagnostics=diag,
            min_label_count=min_label_count,
            preferred_label_column=preferred_label_column,
            label_mode=label_mode,
        )
    raise ValueError(f"Unsupported baseline={baseline!r}")


def _cell_path(synthetic_root: Path, kernel: str, seed: int) -> Path:
    return Path(synthetic_root) / f"{kernel}_seed{int(seed)}"


def _test_metric_mean(results: Sequence[Mapping[str, Any]], test_name: str, metric_key: str) -> Optional[float]:
    values: List[float] = []
    for result in results:
        suite = result.get("ideological_validation_suite") or {}
        rows = ((suite.get("tests") or {}).get(test_name) or {}).get("rows") or []
        for row in rows:
            value = row.get(metric_key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except Exception:
                continue
    return _mean(values)


def _anchor_failure_diagnostics(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize which recenter anchors fail strict validation most often."""

    by_anchor: Dict[str, Dict[str, Any]] = {}
    for result in results:
        for observer in result.get("observers", []) or []:
            if not isinstance(observer, Mapping):
                continue
            anchor_label = str(observer.get("anchor_label") or observer.get("anchor_idx") or "unknown")
            anchor = by_anchor.setdefault(
                anchor_label,
                {
                    "anchor_label": anchor_label,
                    "observer_count": 0,
                    "pass_count": 0,
                    "failed_test_counts": Counter(),
                    "primary_gains": [],
                    "kernels": set(),
                    "seeds": set(),
                },
            )
            anchor["observer_count"] += 1
            anchor["kernels"].add(str(result.get("kernel") or "unknown"))
            anchor["seeds"].add(str(result.get("seed") or "unknown"))
            gain = _primary_gain(observer)
            if gain is not None:
                anchor["primary_gains"].append(gain)
            tests = ((observer.get("label_geometry_tests") or {}).get("tests") or {})
            failed_any = False
            for test_name, test_payload in tests.items():
                if isinstance(test_payload, Mapping) and not bool(test_payload.get("pass")):
                    anchor["failed_test_counts"][str(test_name)] += 1
                    failed_any = True
            if not failed_any and tests:
                anchor["pass_count"] += 1

    rows: List[Dict[str, Any]] = []
    for anchor in by_anchor.values():
        observer_count = int(anchor["observer_count"])
        pass_count = int(anchor["pass_count"])
        failed = dict(sorted(anchor["failed_test_counts"].items()))
        rows.append(
            {
                "anchor_label": anchor["anchor_label"],
                "observer_count": observer_count,
                "pass_count": pass_count,
                "pass_rate": float(pass_count / observer_count) if observer_count else None,
                "mean_primary_label_gain": _mean(anchor["primary_gains"]),
                "failed_test_counts": failed,
                "kernels": sorted(anchor["kernels"]),
                "seeds": sorted(anchor["seeds"]),
            }
        )
    return sorted(rows, key=lambda row: (float(row["pass_rate"] or 0.0), row["anchor_label"]))


def aggregate_results(cell_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_baseline: Dict[str, List[Mapping[str, Any]]] = {}
    for cell in cell_results:
        for baseline in cell.get("baselines", []):
            by_baseline.setdefault(str(baseline.get("baseline")), []).append(baseline)
    aggregates: Dict[str, Any] = {}
    for baseline, rows in sorted(by_baseline.items()):
        pass_rows = [row for row in rows if bool(row.get("safe_for_recenter_claim"))]
        observer_counts = [int(row.get("observer_count") or 0) for row in rows]
        label_gains = [
            ((row.get("label_contraction") or {}).get("primary_mean_label_gap_gain_over_translation"))
            for row in rows
        ]
        suite_pass_counts = Counter((row.get("ideological_validation_suite") or {}).get("status", "NO_DATA") for row in rows)
        aggregates[baseline] = {
            "cell_count": len(rows),
            "pass_count": len(pass_rows),
            "pass_rate": float(len(pass_rows) / len(rows)) if rows else None,
            "mean_observer_count": _mean(observer_counts),
            "mean_primary_label_gain": _mean(label_gains),
            "mean_scale_normalized_all_pairs_gain": _test_metric_mean(
                rows,
                "all_pairs_separation",
                "scale_normalized_separation_gain_over_translation",
            ),
            "mean_silhouette_gain": _test_metric_mean(rows, "silhouette", "silhouette_gain_over_translation"),
            "mean_nmi_gain": _test_metric_mean(rows, "ari_nmi", "nmi_gain_over_translation"),
            "suite_status_counts": dict(sorted(suite_pass_counts.items())),
            "anchor_diagnostics": _anchor_failure_diagnostics(rows),
        }
    return aggregates


def run_suite(
    *,
    synthetic_root: Path,
    output_dir: Path,
    kernels: Sequence[str],
    seeds: Sequence[int],
    baselines: Sequence[str],
    min_label_count: int,
    preferred_label_column: Optional[str],
    label_mode: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_results: List[Dict[str, Any]] = []
    missing_cells: List[Dict[str, Any]] = []
    for kernel in kernels:
        for seed in seeds:
            run_dir = _cell_path(synthetic_root, kernel, int(seed))
            if not run_dir.exists():
                missing_cells.append({"kernel": kernel, "seed": int(seed), "run_dir": str(run_dir)})
                continue
            baseline_results = []
            for baseline in baselines:
                try:
                    result = evaluate_baseline(
                        run_dir,
                        baseline,
                        min_label_count=min_label_count,
                        preferred_label_column=preferred_label_column,
                        label_mode=label_mode,
                    )
                except Exception as exc:
                    result = {
                        "baseline": baseline,
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "safe_for_recenter_claim": False,
                    }
                result["kernel"] = kernel
                result["seed"] = int(seed)
                result["run_dir"] = str(run_dir)
                baseline_results.append(result)
            cell_results.append(
                {
                    "kernel": kernel,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "baselines": baseline_results,
                }
            )
    aggregate = aggregate_results(cell_results)
    local_summary = aggregate.get(LOCAL_RECOMPUTE_MODE, {})
    payload = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "summary_type": "observer_recenter_robustness_suite",
        "generated_at_utc": _utc_now(),
        "synthetic_root": str(Path(synthetic_root)),
        "kernels": list(kernels),
        "seeds": [int(seed) for seed in seeds],
        "baselines": list(baselines),
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "cell_count": len(cell_results),
        "missing_cells": missing_cells,
        "aggregate": aggregate,
        "robustness_pass": bool(
            not missing_cells
            and int(local_summary.get("pass_count") or 0) == len(kernels) * len(seeds)
        ),
        "local_recompute_variants": {
            LOCAL_RECOMPUTE_MODE: LOCAL_RECOMPUTE_DEFAULT_VARIANT,
            **LOCAL_VARIANT_BASELINES,
        },
        "claim_boundary": {
            "synthetic_label_claim": "controlled synthetic perspective_tag labels only",
            "real_corpus_label_claim": "unsafe without independent labels or defended proxy labels",
            "source_proxy_labels": "exploratory unless explicitly justified",
        },
        "cells": cell_results,
    }
    json_path = output_dir / "observer_recenter_robustness_suite.json"
    csv_path = output_dir / "observer_recenter_robustness_suite.csv"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "kernel",
            "seed",
            "baseline",
            "status",
            "safe_for_recenter_claim",
            "observer_count",
            "primary_mean_label_gain",
            "ideological_suite_status",
            "local_track_recompute_supported_count",
            "local_recompute_variant",
            "run_dir",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cell_results:
            for result in cell.get("baselines", []):
                writer.writerow(
                    {
                        "kernel": cell.get("kernel"),
                        "seed": cell.get("seed"),
                        "baseline": result.get("baseline"),
                        "status": result.get("status"),
                        "safe_for_recenter_claim": result.get("safe_for_recenter_claim"),
                        "observer_count": result.get("observer_count"),
                        "primary_mean_label_gain": (result.get("label_contraction") or {}).get(
                            "primary_mean_label_gap_gain_over_translation"
                        ),
                        "ideological_suite_status": (result.get("ideological_validation_suite") or {}).get("status"),
                        "local_track_recompute_supported_count": result.get("local_track_recompute_supported_count"),
                        "local_recompute_variant": result.get("local_recompute_variant"),
                        "run_dir": cell.get("run_dir"),
                    }
                )
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/observer_recenter_robustness_suite/robustness_20260528"))
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=BASELINES)
    parser.add_argument("--min-label-count", type=int, default=2)
    parser.add_argument("--label-column", dest="preferred_label_column", default="perspective_tag")
    parser.add_argument("--label-mode", choices=("auto", "ideological", "provenance", "diagnostic", "any"), default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    payload = run_suite(
        synthetic_root=args.synthetic_root,
        output_dir=args.output_dir,
        kernels=[str(k) for k in args.kernels],
        seeds=[int(seed) for seed in args.seeds],
        baselines=[str(baseline) for baseline in args.baselines],
        min_label_count=int(args.min_label_count),
        preferred_label_column=args.preferred_label_column,
        label_mode=args.label_mode,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "json": payload.get("artifacts", {}).get("json"),
                    "csv": payload.get("artifacts", {}).get("csv"),
                    "robustness_pass": payload.get("robustness_pass"),
                    "aggregate": payload.get("aggregate"),
                    "missing_cells": payload.get("missing_cells"),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
