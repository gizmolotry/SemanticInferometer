#!/usr/bin/env python3
"""Feed Track 1.5 gradient-probe features into the Track 4 action graph.

This is a contained follow-up to ``run_track15_gradient_probe.py``.  It does not
change production Track 4.  It asks a narrower engineering question:

If Track 2/content geometry is held fixed, does swapping the Track 1.5 shear
field from forward deltas to gradient sensitivity vectors improve Track 4
action separation against controls?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.track4_action_graph import ActionGraphConfig, run_action_graph  # noqa: E402


DEFAULT_FEATURES = (
    REPO_ROOT
    / "outputs"
    / "track15_gradient_probe"
    / "real_controls_n16_20260525"
    / "track15_gradient_probe_features.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "track15_gradient_probe" / "action_graph_n16_20260525"
DEFAULT_METHODS = ("forward_delta", "logit_gradient_delta", "anchor_gradient_delta")
DEFAULT_CORPORA = ("real", "constant", "shuffled", "random")


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [x for x in (_safe_float(value) for value in values) if x is not None]
    return float(np.mean(vals)) if vals else None


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = np.asarray([x for x in (_safe_float(value) for value in values) if x is not None], dtype=np.float64)
    return float(np.std(vals)) if vals.size else None


def _abs_log_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return float(abs(math.log(a / b)))


def _flatten(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError(f"Expected at least 2D features, got shape={x.shape}")
    return x.reshape(x.shape[0], -1)


def _observer_simplex(features: np.ndarray) -> np.ndarray:
    """Use per-observer feature norms as a soft observer-state simplex."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 1, x.shape[1])
    if x.ndim != 3:
        raise ValueError(f"Expected [N, B, H] features, got shape={x.shape}")
    norms = np.linalg.norm(x, axis=2)
    norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = np.clip(norms.sum(axis=1, keepdims=True), 1e-12, None)
    return (norms / row_sums).astype(np.float32)


def _density_from_embeddings(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Simple local density proxy from inverse mean kNN distance."""

    x = _flatten(embeddings).astype(np.float64)
    n_items = x.shape[0]
    if n_items <= 1:
        return np.ones((n_items,), dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    k_eff = max(1, min(int(k), n_items - 1))
    nearest = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_dist = np.mean(nearest, axis=1)
    density = 1.0 / (1.0 + mean_dist)
    return density.astype(np.float32)


def _load_features(path: Path) -> Dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    return {str(key): loaded[key] for key in loaded.files}


def _run_cell(
    *,
    feature_bank: Mapping[str, np.ndarray],
    method: str,
    corpus: str,
    null_corpus: str,
    config: ActionGraphConfig,
) -> Dict[str, Any]:
    content_key = f"forward_delta__{corpus}"
    stress_key = f"{method}__{corpus}"
    null_key = f"{method}__{null_corpus}"
    if content_key not in feature_bank:
        raise KeyError(f"Missing content feature key: {content_key}")
    if stress_key not in feature_bank:
        raise KeyError(f"Missing stress feature key: {stress_key}")

    content = feature_bank[content_key]
    stress = feature_bank[stress_key]
    embeddings = _flatten(content)
    metric_stress = _flatten(stress)
    density = _density_from_embeddings(embeddings, config.k_neighbors)
    simplex = _observer_simplex(stress)
    null_simplex = _observer_simplex(feature_bank[null_key]) if null_key in feature_bank else None

    result = run_action_graph(
        embeddings=torch.as_tensor(embeddings, dtype=torch.float32),
        track3_density=torch.as_tensor(density, dtype=torch.float32),
        metric_stress=torch.as_tensor(metric_stress, dtype=torch.float32),
        observer_simplex=torch.as_tensor(simplex, dtype=torch.float32),
        null_observer_simplex=(
            torch.as_tensor(null_simplex, dtype=torch.float32)
            if null_simplex is not None and null_simplex.shape[0] == simplex.shape[0]
            else None
        ),
        action_branch=config.action_mode,
        config=config,
    )
    summary = dict(result["summary"])
    records = list(result.get("records") or [])
    finite_actions = [
        _safe_float(row.get("action"))
        for row in records
        if row.get("reached") and _safe_float(row.get("action")) is not None
    ]
    unique_touched = sorted({zone for row in records for zone in row.get("unique_touched_zones", [])})
    return {
        "method": method,
        "corpus": corpus,
        "content_basis": "forward_delta",
        "stress_basis": method,
        "observer_state_basis": f"{method}_norm_simplex",
        "null_corpus": null_corpus if null_simplex is not None else None,
        "n_articles": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "stress_dim": int(metric_stress.shape[1]),
        "path_count": int(summary.get("path_count") or 0),
        "reached_count": int(summary.get("reached_count") or 0),
        "mean_action": _safe_float(summary.get("mean_action")),
        "median_action": _safe_float(summary.get("median_action")),
        "mean_metric": _mean(row.get("metric") for row in records),
        "mean_shear_penalty": _mean(row.get("shear_penalty") for row in records),
        "mean_observer_transport_penalty": _mean(row.get("observer_transport_penalty") for row in records),
        "mean_hysteresis_penalty": _safe_float(summary.get("mean_hysteresis_penalty")),
        "mean_null_hysteresis_penalty": _safe_float(summary.get("mean_null_hysteresis_penalty")),
        "mean_excess_hysteresis_penalty": _safe_float(summary.get("mean_excess_hysteresis_penalty")),
        "mean_calibrated_hysteresis_penalty": _safe_float(summary.get("mean_calibrated_hysteresis_penalty")),
        "terrain_zone_counts": summary.get("terrain_zone_counts", {}),
        "anchor_zones": summary.get("anchor_zones", []),
        "unique_touched_zones": unique_touched,
        "unique_touched_zone_count": int(len(unique_touched)),
    }


def _comparison_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_method = {str(row.get("method")): [] for row in rows}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    out: List[Dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        real = next((row for row in group if row.get("corpus") == "real"), None)
        if not real:
            continue
        for control in ("constant", "shuffled", "random"):
            ctrl = next((row for row in group if row.get("corpus") == control), None)
            if not ctrl:
                continue
            real_action = _safe_float(real.get("mean_action"))
            control_action = _safe_float(ctrl.get("mean_action"))
            real_hyst = _safe_float(real.get("mean_calibrated_hysteresis_penalty"))
            control_hyst = _safe_float(ctrl.get("mean_calibrated_hysteresis_penalty"))
            out.append(
                {
                    "method": method,
                    "control": control,
                    "real_mean_action": real_action,
                    "control_mean_action": control_action,
                    "action_ratio_real_over_control": (
                        float(real_action / control_action)
                        if real_action is not None and control_action not in {None, 0.0}
                        else None
                    ),
                    "action_abs_log_ratio": _abs_log_ratio(real_action, control_action),
                    "real_calibrated_hysteresis": real_hyst,
                    "control_calibrated_hysteresis": control_hyst,
                    "hysteresis_abs_log_ratio": _abs_log_ratio(real_hyst, control_hyst),
                }
            )
    return out


def _method_scores(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    methods = sorted({str(row.get("method")) for row in comparisons})
    scores: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        rows = [row for row in comparisons if row.get("method") == method]
        stochastic = [row for row in rows if row.get("control") in {"shuffled", "random"}]
        scores[method] = {
            "mean_action_abs_log_ratio_all_controls": _mean(row.get("action_abs_log_ratio") for row in rows),
            "mean_action_abs_log_ratio_stochastic_controls": _mean(
                row.get("action_abs_log_ratio") for row in stochastic
            ),
            "mean_hysteresis_abs_log_ratio_stochastic_controls": _mean(
                row.get("hysteresis_abs_log_ratio") for row in stochastic
            ),
            "action_ratios": {
                str(row.get("control")): row.get("action_ratio_real_over_control")
                for row in rows
            },
        }
    return scores


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, sort_keys=True)
            writer.writerow(out)


def run_probe(
    *,
    features_path: Path,
    output_dir: Path,
    methods: Sequence[str],
    corpora: Sequence[str],
    null_corpus: str,
    config: ActionGraphConfig,
) -> Dict[str, Any]:
    feature_bank = _load_features(features_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for method in methods:
        for corpus in corpora:
            print(f"[action-probe] method={method} corpus={corpus}")
            rows.append(
                _run_cell(
                    feature_bank=feature_bank,
                    method=str(method),
                    corpus=str(corpus),
                    null_corpus=str(null_corpus),
                    config=config,
                )
            )
    comparisons = _comparison_rows(rows)
    scores = _method_scores(comparisons)
    summary = {
        "schema_version": "1.0",
        "summary_type": "track15_gradient_action_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "action_graph_config": {
            "k_neighbors": int(config.k_neighbors),
            "action_mode": str(config.action_mode),
            "stress_weight": float(config.stress_weight),
            "shear_weight": float(config.shear_weight),
            "observer_transport_weight": float(config.observer_transport_weight),
            "hysteresis_weight": float(config.hysteresis_weight),
            "void_weight": float(config.void_weight),
            "curvature_weight": float(config.curvature_weight),
            "max_paths": int(config.max_paths),
            "target_count_per_anchor": int(config.target_count_per_anchor),
        },
        "rows": rows,
        "comparisons": comparisons,
        "method_scores": scores,
        "interpretation": {
            "controlled_geometry": "All cells use forward_delta as content geometry; only metric_stress and observer simplex are swapped.",
            "primary_metric": "mean_action_abs_log_ratio_stochastic_controls",
            "claim_boundary": "Diagnostic Track 4 force-field ablation only; not a production replacement by itself.",
        },
    }
    (output_dir / "track15_gradient_action_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "track15_gradient_action_rows.csv", rows)
    _write_csv(output_dir / "track15_gradient_action_comparisons.csv", comparisons)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--corpora", nargs="+", default=list(DEFAULT_CORPORA))
    parser.add_argument("--null-corpus", default="shuffled")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--action-mode", default="null_calibrated_hysteresis")
    parser.add_argument("--max-paths", type=int, default=12)
    parser.add_argument("--target-count-per-anchor", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = ActionGraphConfig(
        k_neighbors=int(args.k_neighbors),
        action_mode=str(args.action_mode),
        max_paths=int(args.max_paths),
        target_count_per_anchor=int(args.target_count_per_anchor),
    )
    summary = run_probe(
        features_path=Path(args.features),
        output_dir=Path(args.output_dir),
        methods=tuple(args.methods),
        corpora=tuple(args.corpora),
        null_corpus=str(args.null_corpus),
        config=config,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "method_scores": summary["method_scores"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
