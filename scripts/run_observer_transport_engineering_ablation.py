#!/usr/bin/env python3
"""Run engineering ablations for observer-slice transport.

This is an artifact-level search over concrete Track 4 engineering choices:
projection source, projection dimension, chart normalization, article-pair
selection, and action-field weights.  It does not change the validation target;
it asks which system changes improve real/control separation while preserving
the observer-slice transport effect.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observer_slice_transport import ObserverSliceTransportConfig  # noqa: E402
from scripts.run_observer_slice_transport_scale_suite import (  # noqa: E402
    DEFAULT_CONTROL_ROOT,
    DEFAULT_REAL_500,
    DEFAULT_SYNTHETIC_ROOT,
    _default_payloads,
    _fast_transport_summary,
    _infer_run_identity,
    _load_payload,
    _pairwise_sq_dists,
    _project_observer_slices,
    _safe_float,
    _select_article_pairs,
    _stress_density,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_transport_engineering_ablation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_transport_engineering_ablation" / "latest"


@dataclass(frozen=True)
class WeightProfile:
    name: str
    semantic_weight: float
    observer_switch_weight: float
    stress_weight: float
    density_weight: float


WEIGHT_PROFILES: tuple[WeightProfile, ...] = (
    WeightProfile("semantic_only", 1.0, 1.0, 0.0, 0.0),
    WeightProfile("default_action", 1.0, 1.0, 0.25, 0.15),
    WeightProfile("stress_forward", 1.0, 1.0, 0.75, 0.15),
    WeightProfile("density_forward", 1.0, 1.0, 0.25, 0.75),
    WeightProfile("stress_only", 1.0, 1.0, 1.0, 0.0),
    WeightProfile("density_only", 1.0, 1.0, 0.0, 1.0),
    WeightProfile("switch_heavy", 1.0, 1.5, 0.25, 0.15),
    WeightProfile("semantic_heavy", 1.5, 1.0, 0.25, 0.15),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _rate(flags: Iterable[Any]) -> float:
    vals = [bool(flag) for flag in flags]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _representative_payloads() -> list[Path]:
    paths: list[Path] = []
    if DEFAULT_SYNTHETIC_ROOT.exists():
        seed42 = sorted(DEFAULT_SYNTHETIC_ROOT.glob("*_seed42/observer_global.pt"))
        paths.extend(seed42[:3])
        if len(paths) < 3:
            for path in sorted(DEFAULT_SYNTHETIC_ROOT.glob("*_seed*/observer_global.pt")):
                if path not in paths:
                    paths.append(path)
                if len(paths) >= 3:
                    break
    for path in (
        DEFAULT_REAL_500,
        DEFAULT_CONTROL_ROOT / "real" / "observer_global.pt",
        DEFAULT_CONTROL_ROOT / "control_random" / "observer_global.pt",
        DEFAULT_CONTROL_ROOT / "control_shuffled" / "observer_global.pt",
    ):
        if path.exists():
            paths.append(path)
    return _dedupe_paths(paths)


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _payloads_for_mode(mode: str) -> list[Path]:
    if mode == "representative":
        return _representative_payloads()
    if mode == "defaults":
        return _default_payloads()
    if mode == "real_controls":
        return _dedupe_paths(
            [
                path
                for path in (
                    DEFAULT_REAL_500,
                    DEFAULT_CONTROL_ROOT / "real" / "observer_global.pt",
                    DEFAULT_CONTROL_ROOT / "control_random" / "observer_global.pt",
                    DEFAULT_CONTROL_ROOT / "control_shuffled" / "observer_global.pt",
                )
                if path.exists()
            ]
        )
    raise ValueError(f"unknown payload mode: {mode}")


def _article_indices(n_items: int, article_cap: Optional[int], random_seed: int) -> np.ndarray:
    if article_cap is None or int(article_cap) <= 0 or int(article_cap) >= n_items:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(int(random_seed))
    return np.sort(rng.choice(n_items, size=int(article_cap), replace=False)).astype(np.int64)


def _take_articles(slices: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(coords, dtype=np.float64)[indices] for name, coords in slices.items()}


def _raw_cls_slices(
    payload: Mapping[str, Any],
    *,
    projection_dim_cap: Optional[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cls = np.asarray(payload["cls_per_bot"], dtype=np.float64)
    if cls.ndim != 3:
        raise ValueError(f"cls_per_bot must have shape [N,B,H], got {cls.shape}")
    keep = cls.shape[-1] if projection_dim_cap is None else min(int(projection_dim_cap), int(cls.shape[-1]))
    arr = cls[:, :, :keep]
    return {f"observer_{idx}": arr[:, idx, :] for idx in range(arr.shape[1])}, {
        "projection_mode": "raw_cls_per_observer_dim_cap",
        "input_shape": list(cls.shape),
        "output_shape": list(arr.shape),
        "output_dim": int(keep),
    }


def _standardize(arr: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return (arr - mean) / np.where(std > eps, std, 1.0)


def _normalize_slices(
    slices: Mapping[str, np.ndarray],
    mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {name: np.asarray(coords, dtype=np.float64) for name, coords in slices.items()}
    if mode == "none":
        return {name: coords.copy() for name, coords in arrays.items()}, {"normalization_mode": "none"}
    if mode == "center_per_slice":
        return {
            name: coords - coords.mean(axis=0, keepdims=True)
            for name, coords in arrays.items()
        }, {"normalization_mode": mode}
    if mode == "zscore_per_slice":
        return {name: _standardize(coords) for name, coords in arrays.items()}, {"normalization_mode": mode}
    stacked = np.concatenate(list(arrays.values()), axis=0)
    if mode == "global_center":
        center = stacked.mean(axis=0, keepdims=True)
        return {name: coords - center for name, coords in arrays.items()}, {"normalization_mode": mode}
    if mode == "global_zscore":
        center = stacked.mean(axis=0, keepdims=True)
        scale = stacked.std(axis=0, keepdims=True)
        scale = np.where(scale > 1e-9, scale, 1.0)
        return {name: (coords - center) / scale for name, coords in arrays.items()}, {
            "normalization_mode": mode
        }
    raise ValueError(f"unknown normalization mode: {mode}")


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    total = 2166136261
    for char in text:
        total ^= ord(char)
        total = (total * 16777619) % (2**32)
    return int(total)


def _null_slices(
    slices: Mapping[str, np.ndarray],
    *,
    mode: str,
    random_seed: int,
) -> Optional[dict[str, np.ndarray]]:
    arrays = {name: np.asarray(coords, dtype=np.float64) for name, coords in slices.items()}
    if mode == "zero":
        return None
    if mode == "identical_mean_chart":
        mean_chart = np.mean(np.stack(list(arrays.values()), axis=0), axis=0)
        return {name: mean_chart.copy() for name in arrays}
    rng = np.random.default_rng(int(random_seed))
    if mode == "shared_article_shuffle":
        n_items = int(next(iter(arrays.values())).shape[0])
        perm = rng.permutation(n_items)
        return {name: coords[perm].copy() for name, coords in arrays.items()}
    if mode == "independent_article_shuffle":
        return {name: coords[rng.permutation(coords.shape[0])].copy() for name, coords in arrays.items()}
    if mode == "dimension_signflip_by_slice":
        out: dict[str, np.ndarray] = {}
        for name, coords in arrays.items():
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=coords.shape[1])
            out[name] = coords * signs.reshape(1, -1)
        return out
    raise ValueError(f"unknown null mode: {mode}")


def _apply_null_calibration(
    summary: Mapping[str, Any],
    null_summary: Optional[Mapping[str, Any]],
    *,
    null_mode: str,
) -> dict[str, Any]:
    raw_mean = _safe_float(summary.get("mean_holonomy_action"))
    raw_relative = _safe_float(summary.get("mean_relative_holonomy"))
    null_mean = _safe_float((null_summary or {}).get("mean_holonomy_action")) if null_summary else 0.0
    null_relative = _safe_float((null_summary or {}).get("mean_relative_holonomy")) if null_summary else 0.0
    calibrated = None if raw_mean is None or null_mean is None else float(raw_mean - null_mean)
    calibrated_relative = (
        None if raw_relative is None or null_relative is None else float(raw_relative - null_relative)
    )
    out = dict(summary)
    out.update(
        {
            "null_mode": null_mode,
            "mean_null_holonomy_action": null_mean,
            "mean_null_relative_holonomy": null_relative,
            "mean_calibrated_excess_holonomy_action": calibrated,
            "mean_calibrated_relative_holonomy": calibrated_relative,
            "null_record_count": (null_summary or {}).get("record_count", 0) if null_summary else 0,
        }
    )
    return out


def _add_pair(selected: list[tuple[int, int]], seen: set[tuple[int, int]], i: int, j: int) -> None:
    if i == j:
        return
    pair = (min(int(i), int(j)), max(int(i), int(j)))
    if pair not in seen:
        seen.add(pair)
        selected.append(pair)


def _select_pairs_by_mode(
    reference_coords: np.ndarray,
    *,
    pair_mode: str,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    pair_dim_cap: Optional[int],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    reference = np.asarray(reference_coords, dtype=np.float64)
    if pair_dim_cap is not None and int(pair_dim_cap) > 0:
        reference = reference[:, : min(int(pair_dim_cap), int(reference.shape[1]))]
    n_items = int(reference.shape[0])
    all_pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    if pair_mode == "mixed":
        pairs, meta = _select_article_pairs(
            reference,
            max_pairs=max_pairs,
            neighbor_count=neighbor_count,
            random_seed=random_seed,
        )
        return pairs, {**meta, "engineering_pair_mode": pair_mode, "pair_reference_dim_cap": pair_dim_cap}
    if len(all_pairs) <= int(max_pairs):
        return all_pairs, {
            "pair_mode": "all_pairs",
            "engineering_pair_mode": pair_mode,
            "candidate_pair_count": len(all_pairs),
            "pair_reference_dim_cap": pair_dim_cap,
        }
    rng = np.random.default_rng(int(random_seed))
    if pair_mode == "random":
        idx = rng.choice(len(all_pairs), size=min(len(all_pairs), int(max_pairs)), replace=False)
        return [all_pairs[int(raw)] for raw in np.asarray(idx).reshape(-1)], {
            "pair_mode": "random_pairs",
            "engineering_pair_mode": pair_mode,
            "candidate_pair_count": len(all_pairs),
            "random_seed": int(random_seed),
            "pair_reference_dim_cap": pair_dim_cap,
        }
    d2 = _pairwise_sq_dists(reference)
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    if pair_mode in {"nearest", "farthest"}:
        reverse = pair_mode == "farthest"
        for i in range(n_items):
            order = np.argsort(d2[i])
            if reverse:
                order = order[::-1]
            count = 0
            for raw_j in order:
                j = int(raw_j)
                if j == i:
                    continue
                _add_pair(selected, seen, i, j)
                count += 1
                if len(selected) >= int(max_pairs) or count >= max(1, int(neighbor_count)):
                    break
            if len(selected) >= int(max_pairs):
                break
        return selected[: int(max_pairs)], {
            "pair_mode": f"{pair_mode}_pairs",
            "engineering_pair_mode": pair_mode,
            "candidate_pair_count": len(all_pairs),
            "neighbor_count": int(neighbor_count),
            "pair_reference_dim_cap": pair_dim_cap,
        }
    if pair_mode == "distance_stratified":
        distances = np.sqrt(d2)
        flat = [(i, j, float(distances[i, j])) for i, j in all_pairs]
        flat.sort(key=lambda row: row[2])
        buckets = np.array_split(np.arange(len(flat)), max(1, min(8, int(max_pairs))))
        for bucket in buckets:
            if len(selected) >= int(max_pairs):
                break
            if len(bucket) == 0:
                continue
            size = max(1, int(max_pairs) // len(buckets))
            picks = rng.choice(bucket, size=min(size, len(bucket)), replace=False)
            for raw_idx in np.asarray(picks).reshape(-1):
                i, j, _ = flat[int(raw_idx)]
                _add_pair(selected, seen, i, j)
                if len(selected) >= int(max_pairs):
                    break
        while len(selected) < int(max_pairs):
            i, j = all_pairs[int(rng.integers(0, len(all_pairs)))]
            _add_pair(selected, seen, i, j)
            if len(seen) >= len(all_pairs):
                break
        return selected[: int(max_pairs)], {
            "pair_mode": "distance_stratified_pairs",
            "engineering_pair_mode": pair_mode,
            "candidate_pair_count": len(all_pairs),
            "random_seed": int(random_seed),
            "pair_reference_dim_cap": pair_dim_cap,
        }
    raise ValueError(f"unknown pair mode: {pair_mode}")


def _config_id(config: Mapping[str, Any]) -> str:
    return (
        f"{config['feature_source']}|dim={config['projection_dim_cap']}|"
        f"norm={config['normalization']}|pairs={config['pair_mode']}|"
        f"w={config['weight_profile']}|null={config['null_mode']}"
    )


def _make_configs(
    *,
    feature_sources: Sequence[str],
    projection_dim_caps: Sequence[int],
    normalizations: Sequence[str],
    pair_modes: Sequence[str],
    null_modes: Sequence[str],
    weight_profiles: Sequence[WeightProfile],
    max_configs: Optional[int],
    random_seed: int,
) -> list[dict[str, Any]]:
    configs = [
        {
            "feature_source": feature_source,
            "projection_dim_cap": int(dim),
            "normalization": normalization,
            "pair_mode": pair_mode,
            "null_mode": null_mode,
            "weight_profile": profile.name,
            "semantic_weight": profile.semantic_weight,
            "observer_switch_weight": profile.observer_switch_weight,
            "stress_weight": profile.stress_weight,
            "density_weight": profile.density_weight,
        }
        for feature_source, dim, normalization, pair_mode, null_mode, profile in itertools.product(
            feature_sources,
            projection_dim_caps,
            normalizations,
            pair_modes,
            null_modes,
            weight_profiles,
        )
    ]
    baseline = {
        "feature_source": "rks",
        "projection_dim_cap": 512,
        "normalization": "none",
        "pair_mode": "mixed",
        "null_mode": "zero",
        "weight_profile": "default_action",
        "semantic_weight": 1.0,
        "observer_switch_weight": 1.0,
        "stress_weight": 0.25,
        "density_weight": 0.15,
    }
    if baseline not in configs:
        configs.insert(0, baseline)
    if max_configs is not None and int(max_configs) > 0 and len(configs) > int(max_configs):
        rng = np.random.default_rng(int(random_seed))
        baseline_key = _config_id(baseline)
        keyed = {_config_id(config): config for config in configs}
        keys = [key for key in sorted(keyed) if key != baseline_key]
        keep_count = max(0, int(max_configs) - 1)
        sampled = rng.choice(keys, size=min(keep_count, len(keys)), replace=False)
        configs = [keyed[baseline_key], *[keyed[str(key)] for key in sorted(np.asarray(sampled).tolist())]]
    return configs


def _build_projection_cache_key(
    payload_path: Path,
    config: Mapping[str, Any],
    normalization: str,
    article_cap: Optional[int],
    random_seed: int,
) -> tuple[str, str, int, str, int, int]:
    return (
        str(payload_path.resolve()).lower(),
        str(config["feature_source"]),
        int(config["projection_dim_cap"]),
        str(normalization),
        int(article_cap or 0),
        int(random_seed),
    )


def _prepare_slices(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    normalization: str,
    article_cap: Optional[int],
    random_seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if config["feature_source"] == "rks":
        slices, projection = _project_observer_slices(
            payload,
            device=device,
            projection_dim_cap=int(config["projection_dim_cap"]),
        )
    elif config["feature_source"] == "raw_cls":
        slices, projection = _raw_cls_slices(
            payload,
            projection_dim_cap=int(config["projection_dim_cap"]),
        )
    else:
        raise ValueError(f"unknown feature source: {config['feature_source']}")
    n_items = int(next(iter(slices.values())).shape[0])
    density, stress, field_sources = _stress_density(payload, n_items)
    indices = _article_indices(n_items, article_cap, random_seed)
    slices = _take_articles(slices, indices)
    density = density[indices]
    stress = stress[indices]
    normalized, norm_meta = _normalize_slices(slices, normalization)
    metadata = {
        "projection": projection,
        "normalization": norm_meta,
        "field_sources": field_sources,
        "article_cap": article_cap,
        "article_indices": [int(idx) for idx in indices.tolist()],
        "article_count_before_cap": n_items,
        "article_count_after_cap": int(indices.size),
        "payload_path": str(payload_path),
    }
    return normalized, density, stress, metadata


def _run_one_payload_config(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    top_records: int,
    projection_cache: dict[tuple[str, str, int, str, int, int], tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]],
    pair_cache: dict[tuple[str, str, int, str, int, int, str, int, int, int], tuple[list[tuple[int, int]], dict[str, Any]]],
) -> dict[str, Any]:
    identity = _infer_run_identity(payload_path)
    proj_key = _build_projection_cache_key(
        payload_path,
        config,
        str(config["normalization"]),
        article_cap,
        random_seed,
    )
    if proj_key not in projection_cache:
        projection_cache[proj_key] = _prepare_slices(
            payload_path,
            payload,
            config,
            device=device,
            normalization=str(config["normalization"]),
            article_cap=article_cap,
            random_seed=random_seed,
        )
    slices, density, stress, metadata = projection_cache[proj_key]
    reference = np.mean(np.stack(list(slices.values()), axis=1), axis=1)
    pair_key = (
        *proj_key,
        str(config["pair_mode"]),
        int(max_pairs),
        int(neighbor_count),
        int(pair_dim_cap or 0),
    )
    if pair_key not in pair_cache:
        pair_cache[pair_key] = _select_pairs_by_mode(
            reference,
            pair_mode=str(config["pair_mode"]),
            max_pairs=int(max_pairs),
            neighbor_count=int(neighbor_count),
            random_seed=int(random_seed),
            pair_dim_cap=pair_dim_cap,
        )
    article_pairs, pair_meta = pair_cache[pair_key]
    action_config = ObserverSliceTransportConfig(
        semantic_weight=float(config["semantic_weight"]),
        observer_switch_weight=float(config["observer_switch_weight"]),
        stress_weight=float(config["stress_weight"]),
        density_weight=float(config["density_weight"]),
    )
    summary = _fast_transport_summary(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=action_config,
        top_records=int(top_records),
    )
    null_seed = _stable_seed(payload_path, _config_id(config), random_seed)
    null_chart = _null_slices(slices, mode=str(config["null_mode"]), random_seed=null_seed)
    null_summary = None
    if null_chart is not None:
        null_summary = _fast_transport_summary(
            null_chart,
            article_pairs=article_pairs,
            density=density,
            stress=stress,
            config=action_config,
            top_records=0,
        )
    summary = _apply_null_calibration(summary, null_summary, null_mode=str(config["null_mode"]))
    return {
        **identity,
        "config_id": _config_id(config),
        "config": dict(config),
        "n_articles": int(next(iter(slices.values())).shape[0]),
        "n_slices": len(slices),
        "pair_selection": pair_meta,
        "projection": metadata["projection"],
        "normalization": metadata["normalization"],
        "field_sources": metadata["field_sources"],
        "article_cap": metadata["article_cap"],
        "article_count_before_cap": metadata["article_count_before_cap"],
        "transport": summary,
        "null_transport": null_summary,
        "effect_supported": bool(
            summary.get("status") == "OK"
            and float(summary.get("mean_calibrated_excess_holonomy_action") or 0.0) > 0.0
            and int(summary.get("record_count") or 0) > 0
        ),
    }


def _aggregate_config(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_corpus: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_corpus[str(row.get("corpus") or "unknown")].append(row)

    def corpus_metric(corpus: str, metric: str) -> Optional[float]:
        def value(row: Mapping[str, Any]) -> Any:
            transport = row.get("transport") or {}
            if metric == "mean_excess_holonomy_action":
                return transport.get("mean_calibrated_excess_holonomy_action", transport.get(metric))
            if metric == "mean_relative_holonomy":
                return transport.get("mean_calibrated_relative_holonomy", transport.get(metric))
            return transport.get(metric)

        return _mean(value(row) for row in by_corpus.get(corpus, []))

    real_excess = corpus_metric("real", "mean_excess_holonomy_action")
    real_relative = corpus_metric("real", "mean_relative_holonomy")
    synthetic_excess = corpus_metric("synthetic", "mean_excess_holonomy_action")
    control_excess = _mean(
        corpus_metric(corpus, "mean_excess_holonomy_action")
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    control_relative = _mean(
        corpus_metric(corpus, "mean_relative_holonomy")
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    real_control_gap = (
        float(real_excess - control_excess)
        if real_excess is not None and control_excess is not None
        else None
    )
    real_relative_gap = (
        float(real_relative - control_relative)
        if real_relative is not None and control_relative is not None
        else None
    )
    ratio = (
        float(real_excess / max(control_excess, 1e-12))
        if real_excess is not None and control_excess is not None
        else None
    )
    engineering_score = None
    if real_control_gap is not None:
        engineering_score = real_control_gap
        if real_relative_gap is not None:
            engineering_score += 0.5 * real_relative_gap
        if synthetic_excess is not None:
            engineering_score += 0.05 * synthetic_excess
        if control_excess is not None:
            engineering_score -= 0.1 * control_excess
    first = rows[0] if rows else {}
    config = dict(first.get("config") or {})
    return {
        "config_id": first.get("config_id"),
        "config": config,
        "payload_count": len(rows),
        "effect_pass_rate": _rate(row.get("effect_supported") for row in rows),
        "real_mean_excess_holonomy_action": real_excess,
        "real_mean_relative_holonomy": real_relative,
        "control_mean_excess_holonomy_action": control_excess,
        "control_mean_relative_holonomy": control_relative,
        "synthetic_mean_excess_holonomy_action": synthetic_excess,
        "real_minus_control_mean_excess_holonomy_action": real_control_gap,
        "real_minus_control_mean_relative_holonomy": real_relative_gap,
        "real_to_control_excess_ratio": ratio,
        "engineering_score": engineering_score,
        "aggregate_by_corpus": {
            corpus: {
                "run_count": len(corpus_rows),
                "effect_pass_rate": _rate(row.get("effect_supported") for row in corpus_rows),
                "mean_excess_holonomy_action": _mean(
                    (row.get("transport") or {}).get(
                        "mean_calibrated_excess_holonomy_action",
                        (row.get("transport") or {}).get("mean_excess_holonomy_action"),
                    )
                    for row in corpus_rows
                ),
                "mean_relative_holonomy": _mean(
                    (row.get("transport") or {}).get(
                        "mean_calibrated_relative_holonomy",
                        (row.get("transport") or {}).get("mean_relative_holonomy"),
                    )
                    for row in corpus_rows
                ),
            }
            for corpus, corpus_rows in sorted(by_corpus.items())
        },
    }


def _summarize_configs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("config_id"))].append(row)
    summaries = [_aggregate_config(group_rows) for group_rows in grouped.values()]
    return sorted(
        summaries,
        key=lambda row: float(row.get("engineering_score") if row.get("engineering_score") is not None else -1e12),
        reverse=True,
    )


def _axis_summary(config_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    axes = ["feature_source", "projection_dim_cap", "normalization", "pair_mode", "weight_profile", "null_mode"]
    out: dict[str, Any] = {}
    for axis in axes:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for summary in config_summaries:
            config = summary.get("config") or {}
            groups[str(config.get(axis))].append(summary)
        out[axis] = {
            key: {
                "config_count": len(values),
                "mean_engineering_score": _mean(row.get("engineering_score") for row in values),
                "mean_real_control_gap": _mean(
                    row.get("real_minus_control_mean_excess_holonomy_action") for row in values
                ),
                "mean_real_relative_gap": _mean(
                    row.get("real_minus_control_mean_relative_holonomy") for row in values
                ),
                "mean_control_floor": _mean(row.get("control_mean_excess_holonomy_action") for row in values),
            }
            for key, values in sorted(groups.items())
        }
    return out


def _best_by_axis(config_summaries: Sequence[Mapping[str, Any]], axis: str) -> Optional[dict[str, Any]]:
    axis_rows = (_axis_summary(config_summaries).get(axis) or {})
    if not axis_rows:
        return None
    key, value = max(
        axis_rows.items(),
        key=lambda item: float(item[1].get("mean_engineering_score") or -1e12),
    )
    return {"axis": axis, "best_value": key, **value}


def _engineering_recommendations(config_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not config_summaries:
        return []
    baseline = next(
        (
            row
            for row in config_summaries
            if row.get("config_id") == "rks|dim=512|norm=none|pairs=mixed|w=default_action|null=zero"
        ),
        None,
    )
    baseline_score = _safe_float((baseline or {}).get("engineering_score")) or 0.0
    top = config_summaries[0]
    recs: list[dict[str, Any]] = [
        {
            "recommendation": "prioritize_top_configuration_for_next_core_patch",
            "reason": "highest engineering score in this ablation grid",
            "config_id": top.get("config_id"),
            "score": top.get("engineering_score"),
            "delta_vs_current_baseline": (
                float(top.get("engineering_score") - baseline_score)
                if top.get("engineering_score") is not None
                else None
            ),
            "system_change": top.get("config"),
        }
    ]
    for axis, message in (
        ("projection_dim_cap", "bottleneck Track 4 transport coordinates to this dimension before action scoring"),
        ("normalization", "normalize observer charts before transport instead of using raw chart coordinates"),
        ("feature_source", "route Track 4 through this observer feature source or add it as a hybrid channel"),
        ("pair_mode", "change path/pair proposal policy toward this sampler"),
        ("weight_profile", "change Track 4 action weights toward this field balance"),
        ("null_mode", "promote this null calibration into the primary Track 4 excess-action score"),
    ):
        best = _best_by_axis(config_summaries, axis)
        if best is not None:
            recs.append(
                {
                    "recommendation": f"axis_winner_{axis}",
                    "best_value": best.get("best_value"),
                    "mean_engineering_score": best.get("mean_engineering_score"),
                    "mean_real_control_gap": best.get("mean_real_control_gap"),
                    "mean_control_floor": best.get("mean_control_floor"),
                    "system_change": message,
                }
            )
    return recs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "stage",
        "config_id",
        "corpus",
        "kernel",
        "seed",
        "cell_id",
        "feature_source",
        "projection_dim_cap",
        "normalization",
        "pair_mode",
        "null_mode",
        "weight_profile",
        "n_articles",
        "record_count",
        "mean_excess_holonomy_action",
        "mean_calibrated_excess_holonomy_action",
        "mean_relative_holonomy",
        "mean_calibrated_relative_holonomy",
        "mean_null_holonomy_action",
        "positive_excess_rate",
        "payload_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            config = row.get("config") or {}
            transport = row.get("transport") or {}
            writer.writerow(
                {
                    "stage": row.get("stage"),
                    "config_id": row.get("config_id"),
                    "corpus": row.get("corpus"),
                    "kernel": row.get("kernel"),
                    "seed": row.get("seed"),
                    "cell_id": row.get("cell_id"),
                    "feature_source": config.get("feature_source"),
                    "projection_dim_cap": config.get("projection_dim_cap"),
                    "normalization": config.get("normalization"),
                    "pair_mode": config.get("pair_mode"),
                    "null_mode": config.get("null_mode"),
                    "weight_profile": config.get("weight_profile"),
                    "n_articles": row.get("n_articles"),
                    "record_count": transport.get("record_count"),
                    "mean_excess_holonomy_action": transport.get("mean_excess_holonomy_action"),
                    "mean_calibrated_excess_holonomy_action": transport.get(
                        "mean_calibrated_excess_holonomy_action"
                    ),
                    "mean_relative_holonomy": transport.get("mean_relative_holonomy"),
                    "mean_calibrated_relative_holonomy": transport.get("mean_calibrated_relative_holonomy"),
                    "mean_null_holonomy_action": transport.get("mean_null_holonomy_action"),
                    "positive_excess_rate": transport.get("positive_excess_rate"),
                    "payload_path": row.get("payload_path"),
                }
            )


def _run_grid(
    *,
    payload_paths: Sequence[Path],
    configs: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    top_records: int,
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload_cache: dict[str, dict[str, Any]] = {}
    projection_cache: dict[
        tuple[str, str, int, str, int, int],
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]],
    ] = {}
    pair_cache: dict[tuple[str, str, int, str, int, int, str, int, int, int], tuple[list[tuple[int, int]], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for payload_path in payload_paths:
        key = str(payload_path.resolve()).lower()
        try:
            payload_cache[key] = _load_payload(payload_path)
        except Exception as exc:
            failures.append({"stage": stage, "payload_path": str(payload_path), "error": str(exc)})
            continue
        for config in configs:
            try:
                row = _run_one_payload_config(
                    payload_path,
                    payload_cache[key],
                    config,
                    device=device,
                    max_pairs=max_pairs,
                    neighbor_count=neighbor_count,
                    random_seed=random_seed,
                    article_cap=article_cap,
                    pair_dim_cap=pair_dim_cap,
                    top_records=top_records,
                    projection_cache=projection_cache,
                    pair_cache=pair_cache,
                )
                row["stage"] = stage
                rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "stage": stage,
                        "payload_path": str(payload_path),
                        "config_id": _config_id(config),
                        "error": str(exc),
                    }
                )
    return rows, failures


def build_engineering_ablation(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    feature_sources: Sequence[str],
    projection_dim_caps: Sequence[int],
    normalizations: Sequence[str],
    pair_modes: Sequence[str],
    null_modes: Sequence[str],
    max_configs: Optional[int],
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    confirm_top: int,
    confirm_payload_paths: Sequence[Path],
    confirm_max_pairs: int,
    confirm_article_cap: Optional[int],
    device_name: str,
    top_records: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    configs = _make_configs(
        feature_sources=feature_sources,
        projection_dim_caps=projection_dim_caps,
        normalizations=normalizations,
        pair_modes=pair_modes,
        null_modes=null_modes,
        weight_profiles=WEIGHT_PROFILES,
        max_configs=max_configs,
        random_seed=random_seed,
    )
    stage1_rows, stage1_failures = _run_grid(
        payload_paths=payload_paths,
        configs=configs,
        device=device,
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        article_cap=article_cap,
        pair_dim_cap=pair_dim_cap,
        top_records=top_records,
        stage="screen",
    )
    stage1_summaries = _summarize_configs(stage1_rows)
    selected_stage2 = [row.get("config") for row in stage1_summaries[: max(0, int(confirm_top))]]
    selected_stage2 = [dict(config) for config in selected_stage2 if isinstance(config, Mapping)]
    stage2_rows: list[dict[str, Any]] = []
    stage2_failures: list[dict[str, Any]] = []
    stage2_summaries: list[dict[str, Any]] = []
    if selected_stage2 and confirm_payload_paths:
        stage2_rows, stage2_failures = _run_grid(
            payload_paths=confirm_payload_paths,
            configs=selected_stage2,
            device=device,
            max_pairs=confirm_max_pairs,
            neighbor_count=neighbor_count,
            random_seed=random_seed,
            article_cap=confirm_article_cap,
            pair_dim_cap=pair_dim_cap,
            top_records=top_records,
            stage="confirm",
        )
        stage2_summaries = _summarize_configs(stage2_rows)

    preferred_summaries = stage2_summaries or stage1_summaries
    rows = [*stage1_rows, *stage2_rows]
    failures = [*stage1_failures, *stage2_failures]
    csv_path = output_dir / "observer_transport_engineering_ablation_rows.csv"
    _write_csv(csv_path, rows)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "device": str(device),
        "config_space": {
            "feature_sources": list(feature_sources),
            "projection_dim_caps": [int(value) for value in projection_dim_caps],
            "normalizations": list(normalizations),
            "pair_modes": list(pair_modes),
            "null_modes": list(null_modes),
            "weight_profiles": [profile.name for profile in WEIGHT_PROFILES],
            "screen_config_count": len(configs),
        },
        "screen": {
            "payload_count": len(payload_paths),
            "row_count": len(stage1_rows),
            "failure_count": len(stage1_failures),
            "max_pairs": int(max_pairs),
            "article_cap": article_cap,
            "config_summaries": stage1_summaries,
            "axis_summary": _axis_summary(stage1_summaries),
        },
        "confirm": {
            "enabled": bool(selected_stage2 and confirm_payload_paths),
            "payload_count": len(confirm_payload_paths),
            "row_count": len(stage2_rows),
            "failure_count": len(stage2_failures),
            "max_pairs": int(confirm_max_pairs),
            "article_cap": confirm_article_cap,
            "selected_config_count": len(selected_stage2),
            "config_summaries": stage2_summaries,
            "axis_summary": _axis_summary(stage2_summaries),
        },
        "engineering_recommendations": _engineering_recommendations(preferred_summaries),
        "claim_boundary": {
            "safe_claim": "engineering ablation over saved observer-slice transport artifacts",
            "unsafe_claim": "changing thresholds or claim language instead of improving system internals",
        },
        "failures": failures,
        "artifacts": {
            "csv": str(csv_path),
        },
    }
    json_path = output_dir / "observer_transport_engineering_ablation.json"
    artifact["artifacts"]["json"] = str(json_path)
    json_path.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="representative")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-sources", nargs="+", default=["rks", "raw_cls"], choices=("rks", "raw_cls"))
    parser.add_argument("--projection-dim-caps", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument(
        "--normalizations",
        nargs="+",
        default=["none", "center_per_slice", "zscore_per_slice", "global_zscore"],
        choices=("none", "center_per_slice", "zscore_per_slice", "global_center", "global_zscore"),
    )
    parser.add_argument(
        "--pair-modes",
        nargs="+",
        default=["nearest", "mixed", "random", "farthest", "distance_stratified"],
        choices=("nearest", "mixed", "random", "farthest", "distance_stratified"),
    )
    parser.add_argument(
        "--null-modes",
        nargs="+",
        default=["zero", "independent_article_shuffle", "identical_mean_chart"],
        choices=(
            "zero",
            "shared_article_shuffle",
            "independent_article_shuffle",
            "dimension_signflip_by_slice",
            "identical_mean_chart",
        ),
    )
    parser.add_argument("--max-configs", type=int, default=384)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=220)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    parser.add_argument("--confirm-top", type=int, default=12)
    parser.add_argument("--confirm-payload-mode", choices=("representative", "defaults", "real_controls"), default="defaults")
    parser.add_argument("--confirm-max-pairs", type=int, default=512)
    parser.add_argument("--confirm-article-cap", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-records", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    confirm_payloads = _payloads_for_mode(str(args.confirm_payload_mode)) if int(args.confirm_top) > 0 else []
    artifact = build_engineering_ablation(
        payload_paths=payloads,
        output_dir=args.output_dir,
        feature_sources=[str(value) for value in args.feature_sources],
        projection_dim_caps=[int(value) for value in args.projection_dim_caps],
        normalizations=[str(value) for value in args.normalizations],
        pair_modes=[str(value) for value in args.pair_modes],
        null_modes=[str(value) for value in args.null_modes],
        max_configs=int(args.max_configs),
        max_pairs=int(args.max_pairs),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
        confirm_top=int(args.confirm_top),
        confirm_payload_paths=confirm_payloads,
        confirm_max_pairs=int(args.confirm_max_pairs),
        confirm_article_cap=int(args.confirm_article_cap) if int(args.confirm_article_cap) > 0 else None,
        device_name=str(args.device),
        top_records=int(args.top_records),
    )
    preferred = (artifact.get("confirm") or {}).get("config_summaries") or (artifact.get("screen") or {}).get(
        "config_summaries"
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "json": artifact.get("artifacts", {}).get("json"),
                    "csv": artifact.get("artifacts", {}).get("csv"),
                    "screen_rows": (artifact.get("screen") or {}).get("row_count"),
                    "confirm_rows": (artifact.get("confirm") or {}).get("row_count"),
                    "failure_count": len(artifact.get("failures") or []),
                    "top_config": preferred[0] if preferred else None,
                    "recommendations": artifact.get("engineering_recommendations", [])[:6],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if preferred else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
