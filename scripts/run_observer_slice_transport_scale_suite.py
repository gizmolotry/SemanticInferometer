#!/usr/bin/env python3
"""Scale observer-slice transport/holonomy over saved pipeline artifacts.

This uses saved ``observer_global.pt`` payloads to build one observer chart per
V-observer from the shared RKS basis.  It then measures whether semantic moves
and observer switches commute across those charts, with a translation/null
baseline.  The result is the large-corpus version of the property/theft
holonomy probe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observer_slice_transport import ObserverSliceTransportConfig  # noqa: E402


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_slice_transport_scale_suite"
DEFAULT_SYNTHETIC_ROOT = (
    ROOT / "outputs" / "experiments" / "runs" / "experiments_20260506_192553" / "synthetic"
)
DEFAULT_REAL_500 = (
    ROOT
    / "outputs"
    / "experiments"
    / "runs"
    / "experiments_20260403_173250"
    / "rbf"
    / "cls"
    / "real"
    / "observer_global.pt"
)
DEFAULT_CONTROL_ROOT = (
    ROOT / "outputs" / "experiments" / "runs" / "experiments_20260504_171454" / "matern" / "cls"
)


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


def _rate(flags: Iterable[Any]) -> float:
    vals = [bool(flag) for flag in flags]
    return float(sum(vals) / len(vals)) if vals else 0.0


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


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"observer payload root must be a dict: {path}")
    if payload.get("cls_per_bot") is None:
        raise ValueError(f"observer payload missing cls_per_bot: {path}")
    return payload


def _infer_run_identity(payload_path: Path) -> dict[str, Any]:
    parts = [part.lower() for part in payload_path.parts]
    corpus = "unknown"
    for candidate in ("real", "synthetic", "control_random", "control_shuffled", "control_constant"):
        if candidate in parts:
            corpus = candidate
            break
    kernel = None
    seed = None
    cell_id = payload_path.parent.name
    synthetic_match = re.match(r"(?P<kernel>[a-z0-9]+)_seed(?P<seed>\d+)$", cell_id)
    if synthetic_match:
        kernel = synthetic_match.group("kernel")
        seed = int(synthetic_match.group("seed"))
    else:
        for part in payload_path.parts:
            lower = part.lower()
            if lower in {"rbf", "matern", "imq", "laplacian", "rq"}:
                kernel = lower
                break
    return {
        "payload_path": str(payload_path),
        "run_dir": str(payload_path.parent),
        "cell_id": cell_id,
        "corpus": corpus,
        "kernel": kernel or "unknown",
        "seed": seed,
    }


def _default_payloads() -> list[Path]:
    paths: list[Path] = []
    if DEFAULT_SYNTHETIC_ROOT.exists():
        paths.extend(sorted(DEFAULT_SYNTHETIC_ROOT.glob("*_seed*/observer_global.pt")))
    for path in [
        DEFAULT_REAL_500,
        DEFAULT_CONTROL_ROOT / "real" / "observer_global.pt",
        DEFAULT_CONTROL_ROOT / "control_random" / "observer_global.pt",
        DEFAULT_CONTROL_ROOT / "control_shuffled" / "observer_global.pt",
    ]:
        if path.exists():
            paths.append(path)
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _basis_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _project_observer_slices(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    projection_dim_cap: Optional[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cls = torch.as_tensor(np.asarray(payload["cls_per_bot"]), device=device, dtype=torch.float32)
    if cls.ndim != 3:
        raise ValueError(f"cls_per_bot must have shape [N,B,H], got {tuple(cls.shape)}")
    basis = payload.get("rks_basis_state") if isinstance(payload.get("rks_basis_state"), dict) else {}
    omega_raw = basis.get("omega") if isinstance(basis, dict) else None
    bias_raw = basis.get("b") if isinstance(basis, dict) else None
    if omega_raw is None or bias_raw is None:
        # Fallback is still a valid chart diagnostic, but not a Track2/RKS one.
        arr = cls.detach().cpu().numpy().astype(np.float64)
        return {f"observer_{idx}": arr[:, idx, :] for idx in range(arr.shape[1])}, {
            "projection_mode": "raw_cls_no_rks_basis",
            "input_shape": list(arr.shape),
            "output_dim": int(arr.shape[-1]),
        }

    omega = _basis_tensor(omega_raw, device=device)
    bias = _basis_tensor(bias_raw, device=device)
    if omega.ndim != 2:
        raise ValueError(f"omega must be 2D, got {tuple(omega.shape)}")
    if omega.shape[0] != cls.shape[-1] and omega.shape[1] == cls.shape[-1]:
        omega = omega.T
    if omega.shape[0] != cls.shape[-1]:
        raise ValueError(f"omega input dim {omega.shape[0]} does not match cls dim {cls.shape[-1]}")
    if projection_dim_cap is not None:
        keep = min(int(projection_dim_cap), int(omega.shape[1]))
        omega = omega[:, :keep]
        bias = bias[:keep]
    sigma = _safe_float(basis.get("sigma")) if isinstance(basis, dict) else None
    if sigma is None or sigma <= 0.0:
        sigma_diag = basis.get("sigma_diagnostics") if isinstance(basis, dict) else {}
        sigma = _safe_float(sigma_diag.get("estimated_sigma")) if isinstance(sigma_diag, dict) else None
    sigma = float(sigma or 1.0)
    projected = torch.cos(cls @ (omega / sigma) + bias)
    projected = projected * math.sqrt(2.0 / float(projected.shape[-1]))
    arr = projected.detach().cpu().numpy().astype(np.float64)
    return {f"observer_{idx}": arr[:, idx, :] for idx in range(arr.shape[1])}, {
        "projection_mode": "shared_rks_basis_per_observer",
        "input_shape": list(cls.shape),
        "output_shape": list(arr.shape),
        "kernel_type": basis.get("kernel_type"),
        "basis_seed": basis.get("seed"),
        "sigma": sigma,
        "basis_hash": basis.get("hash"),
    }


def _normalize_unit_interval(values: np.ndarray, *, default: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    if hi - lo <= 1e-12:
        return np.full_like(arr, float(default), dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _stress_density(payload: Mapping[str, Any], n_items: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stress_source = "default_zero"
    density_source = "default_one"
    stress = np.zeros(n_items, dtype=np.float64)
    density = np.ones(n_items, dtype=np.float64)

    if payload.get("spectral_probe_magnitudes") is not None:
        probe = np.asarray(payload["spectral_probe_magnitudes"], dtype=np.float64)
        if probe.ndim == 2 and probe.shape[0] == n_items:
            stress = _normalize_unit_interval(np.linalg.norm(probe, axis=1), default=0.0)
            stress_source = "spectral_probe_magnitudes_norm"
    elif payload.get("d_spectral") is not None:
        raw = np.asarray(payload["d_spectral"], dtype=np.float64)
        if raw.shape[0] == n_items:
            stress = _normalize_unit_interval(np.abs(raw), default=0.0)
            stress_source = "d_spectral_abs"

    if payload.get("T3_topology") is not None and not isinstance(payload.get("T3_topology"), Mapping):
        raw = np.asarray(payload["T3_topology"], dtype=np.float64)
        if raw.shape[0] == n_items:
            density = 1.0 - _normalize_unit_interval(np.linalg.norm(raw.reshape(n_items, -1), axis=1), default=0.5)
            density_source = "T3_topology_inverse_norm"
    elif payload.get("dirichlet_curvature") is not None:
        curv = payload.get("dirichlet_curvature")
        if isinstance(curv, Mapping) and curv.get("participation_ratio") is not None:
            raw = np.asarray(curv["participation_ratio"], dtype=np.float64)
            if raw.shape[0] == n_items:
                density = _normalize_unit_interval(raw, default=0.5)
                density_source = "dirichlet_curvature_participation_ratio"

    return density, stress, {"density_source": density_source, "stress_source": stress_source}


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.sum(X * X, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * (X @ X.T), 0.0)


def _select_article_pairs(
    reference_coords: np.ndarray,
    *,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    n_items = int(reference_coords.shape[0])
    all_pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    if len(all_pairs) <= int(max_pairs):
        return all_pairs, {"pair_mode": "all_pairs", "candidate_pair_count": len(all_pairs)}

    d2 = _pairwise_sq_dists(reference_coords)
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_pair(i: int, j: int) -> None:
        if i == j:
            return
        pair = (min(i, j), max(i, j))
        if pair not in seen:
            seen.add(pair)
            selected.append(pair)

    for i in range(n_items):
        order = np.argsort(d2[i])
        neighbors = [int(j) for j in order if int(j) != i][: max(1, int(neighbor_count))]
        for j in neighbors:
            add_pair(i, j)
            if len(selected) >= max_pairs:
                return selected, {
                    "pair_mode": "nearest_neighbor_pairs",
                    "candidate_pair_count": len(all_pairs),
                    "neighbor_count": int(neighbor_count),
                }

    farthest_budget = max(0, int(max_pairs) // 5)
    farthest_added = 0
    for i in range(n_items):
        order = np.argsort(d2[i])[::-1]
        for j in order:
            if int(j) != i:
                before = len(selected)
                add_pair(i, int(j))
                farthest_added += int(len(selected) > before)
                break
        if farthest_added >= farthest_budget or len(selected) >= max_pairs:
            break

    rng = np.random.default_rng(random_seed)
    remaining = [pair for pair in all_pairs if pair not in seen]
    if remaining and len(selected) < max_pairs:
        idx = rng.choice(len(remaining), size=min(len(remaining), max_pairs - len(selected)), replace=False)
        for raw in np.asarray(idx).reshape(-1):
            selected.append(remaining[int(raw)])

    return selected[:max_pairs], {
        "pair_mode": "nearest_farthest_random_mix",
        "candidate_pair_count": len(all_pairs),
        "neighbor_count": int(neighbor_count),
        "random_seed": int(random_seed),
    }


def _compact_transport_summary(summary: Mapping[str, Any], *, top_records: int) -> dict[str, Any]:
    records = summary.get("records") if isinstance(summary.get("records"), list) else []
    null_records = summary.get("null_records") if isinstance(summary.get("null_records"), list) else []
    relative = [_safe_float(row.get("relative_holonomy")) for row in records if isinstance(row, Mapping)]
    relative = [value for value in relative if value is not None]
    holonomy = [_safe_float(row.get("holonomy_action")) for row in records if isinstance(row, Mapping)]
    holonomy = [value for value in holonomy if value is not None]
    null_holonomy = [
        _safe_float(row.get("holonomy_action")) for row in null_records if isinstance(row, Mapping)
    ]
    null_holonomy = [value for value in null_holonomy if value is not None]
    top = sorted(
        [row for row in records if isinstance(row, Mapping)],
        key=lambda row: float(row.get("holonomy_action") or 0.0),
        reverse=True,
    )[: int(top_records)]
    return {
        "summary_type": summary.get("summary_type"),
        "status": summary.get("status"),
        "slice_count": summary.get("slice_count"),
        "article_pair_count": summary.get("article_pair_count"),
        "slice_pair_count": summary.get("slice_pair_count"),
        "record_count": summary.get("record_count"),
        "null_record_count": len(null_records),
        "mean_holonomy_action": summary.get("mean_holonomy_action"),
        "max_holonomy_action": summary.get("max_holonomy_action"),
        "mean_null_holonomy_action": summary.get("mean_null_holonomy_action"),
        "mean_excess_holonomy_action": summary.get("mean_excess_holonomy_action"),
        "mean_relative_holonomy": _mean(relative),
        "positive_holonomy_rate": _rate(value > 1e-9 for value in holonomy),
        "positive_excess_rate": _rate(
            (value - (null_holonomy[idx] if idx < len(null_holonomy) else 0.0)) > 1e-9
            for idx, value in enumerate(holonomy)
        ),
        "top_records": top,
    }


def _semantic_action_vector(
    coords: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    density: np.ndarray,
    stress: np.ndarray,
    config: ObserverSliceTransportConfig,
) -> tuple[np.ndarray, np.ndarray]:
    deltas = coords[source_idx] - coords[target_idx]
    base = np.linalg.norm(deltas, axis=1)
    density_mid = 0.5 * (density[source_idx] + density[target_idx])
    stress_mid = 0.5 * (stress[source_idx] + stress[target_idx])
    action = (
        float(config.semantic_weight) * base
        + float(config.density_weight) * np.maximum(0.0, 1.0 - density_mid) * base
        + float(config.stress_weight) * np.maximum(0.0, stress_mid) * base
    )
    return action, base


def _observer_switch_vector(
    source_slice: np.ndarray,
    target_slice: np.ndarray,
    article_idx: np.ndarray,
    *,
    config: ObserverSliceTransportConfig,
) -> tuple[np.ndarray, np.ndarray]:
    deltas = source_slice[article_idx] - target_slice[article_idx]
    base = np.linalg.norm(deltas, axis=1)
    return float(config.observer_switch_weight) * base, base


def _fast_transport_summary(
    slices: Mapping[str, np.ndarray],
    *,
    article_pairs: Sequence[tuple[int, int]],
    density: np.ndarray,
    stress: np.ndarray,
    config: ObserverSliceTransportConfig,
    top_records: int,
) -> dict[str, Any]:
    names = list(slices)
    source_idx = np.asarray([pair[0] for pair in article_pairs], dtype=np.int64)
    target_idx = np.asarray([pair[1] for pair in article_pairs], dtype=np.int64)
    if source_idx.size == 0:
        return {
            "summary_type": "observer_slice_transport_summary",
            "status": "NO_RECORDS",
            "slice_count": len(names),
            "article_pair_count": 0,
            "slice_pair_count": 0,
            "record_count": 0,
            "null_record_count": 0,
            "mean_holonomy_action": None,
            "max_holonomy_action": None,
            "mean_null_holonomy_action": 0.0,
            "mean_excess_holonomy_action": None,
            "mean_relative_holonomy": None,
            "positive_holonomy_rate": 0.0,
            "positive_excess_rate": 0.0,
            "top_records": [],
        }

    record_count = 0
    holonomy_sum = 0.0
    relative_sum = 0.0
    positive_count = 0
    max_holonomy = 0.0
    top: list[dict[str, Any]] = []
    keep_top = max(0, int(top_records))

    for source_name in names:
        src_chart = slices[source_name]
        for target_name in names:
            if source_name == target_name:
                continue
            tgt_chart = slices[target_name]
            sem_first_sem, sem_move = _semantic_action_vector(
                src_chart,
                source_idx,
                target_idx,
                density=density,
                stress=stress,
                config=config,
            )
            sem_first_switch, sem_switch = _observer_switch_vector(
                src_chart,
                tgt_chart,
                target_idx,
                config=config,
            )
            obs_first_switch, obs_switch = _observer_switch_vector(
                src_chart,
                tgt_chart,
                source_idx,
                config=config,
            )
            obs_first_sem, obs_move = _semantic_action_vector(
                tgt_chart,
                source_idx,
                target_idx,
                density=density,
                stress=stress,
                config=config,
            )
            semantic_first = sem_first_sem + sem_first_switch
            observer_first = obs_first_switch + obs_first_sem
            gap = semantic_first - observer_first
            holonomy = np.abs(gap)
            route_min = np.maximum(np.minimum(semantic_first, observer_first), 1e-12)
            relative = holonomy / route_min

            record_count += int(holonomy.size)
            holonomy_sum += float(np.sum(holonomy))
            relative_sum += float(np.sum(relative))
            positive_count += int(np.sum(holonomy > 1e-9))
            if holonomy.size:
                max_holonomy = max(max_holonomy, float(np.max(holonomy)))

            if keep_top:
                local_keep = min(keep_top, int(holonomy.size))
                if local_keep:
                    local_idx = np.argpartition(holonomy, -local_keep)[-local_keep:]
                    for idx in local_idx:
                        idx_i = int(idx)
                        source = int(source_idx[idx_i])
                        target = int(target_idx[idx_i])
                        top.append(
                            {
                                "diagnostic_type": "observer_slice_transport_commutator",
                                "source_idx": source,
                                "target_idx": target,
                                "source_slice": source_name,
                                "target_slice": target_name,
                                "semantic_first_action": float(semantic_first[idx_i]),
                                "observer_first_action": float(observer_first[idx_i]),
                                "commutator_gap": float(gap[idx_i]),
                                "holonomy_action": float(holonomy[idx_i]),
                                "relative_holonomy": float(relative[idx_i]),
                                "semantic_first_components": {
                                    "semantic_move": float(sem_move[idx_i]),
                                    "observer_switch": float(sem_switch[idx_i]),
                                },
                                "observer_first_components": {
                                    "observer_switch": float(obs_switch[idx_i]),
                                    "semantic_move": float(obs_move[idx_i]),
                                },
                                "closed_loop": [
                                    {"article_idx": source, "slice": source_name},
                                    {"article_idx": target, "slice": source_name},
                                    {"article_idx": target, "slice": target_name},
                                    {"article_idx": source, "slice": target_name},
                                    {"article_idx": source, "slice": source_name},
                                ],
                            }
                        )
                    top = sorted(
                        top,
                        key=lambda row: float(row.get("holonomy_action") or 0.0),
                        reverse=True,
                    )[:keep_top]

    mean_holonomy = holonomy_sum / record_count if record_count else None
    mean_relative = relative_sum / record_count if record_count else None
    return {
        "summary_type": "observer_slice_transport_summary",
        "status": "OK" if record_count else "NO_RECORDS",
        "slice_count": len(names),
        "article_pair_count": len(article_pairs),
        "slice_pair_count": len(names) * max(0, len(names) - 1),
        "record_count": record_count,
        "null_record_count": record_count,
        "mean_holonomy_action": mean_holonomy,
        "max_holonomy_action": max_holonomy if record_count else None,
        "mean_null_holonomy_action": 0.0,
        "mean_excess_holonomy_action": mean_holonomy,
        "mean_relative_holonomy": mean_relative,
        "positive_holonomy_rate": positive_count / record_count if record_count else 0.0,
        "positive_excess_rate": positive_count / record_count if record_count else 0.0,
        "top_records": top,
    }


def run_payload_probe(
    payload_path: Path,
    *,
    max_pairs: int,
    neighbor_count: int,
    projection_dim_cap: Optional[int],
    device: torch.device,
    random_seed: int,
    top_records: int,
    config: ObserverSliceTransportConfig,
) -> dict[str, Any]:
    payload = _load_payload(payload_path)
    identity = _infer_run_identity(payload_path)
    slices, projection = _project_observer_slices(
        payload,
        device=device,
        projection_dim_cap=projection_dim_cap,
    )
    n_items = int(next(iter(slices.values())).shape[0])
    density, stress, field_sources = _stress_density(payload, n_items)
    reference = np.mean(np.stack(list(slices.values()), axis=1), axis=1)
    article_pairs, pair_meta = _select_article_pairs(
        reference,
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
    )
    compact = _fast_transport_summary(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=config,
        top_records=top_records,
    )
    min_records = min(int(max_pairs), len(article_pairs)) * max(1, (len(slices) * (len(slices) - 1)))
    effect_supported = bool(
        int(compact.get("record_count") or 0) >= max(1, min_records)
        and float(compact.get("mean_excess_holonomy_action") or 0.0) > 0.0
    )
    return {
        **identity,
        "status": compact.get("status"),
        "effect_supported": effect_supported,
        "n_articles": n_items,
        "n_slices": len(slices),
        "projection": projection,
        "field_sources": field_sources,
        "pair_selection": pair_meta,
        "transport": compact,
    }


def _aggregate_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    out: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        out[group] = {
            "run_count": len(group_rows),
            "effect_pass_rate": _rate(row.get("effect_supported") for row in group_rows),
            "mean_excess_holonomy_action": _mean(
                ((row.get("transport") or {}).get("mean_excess_holonomy_action")) for row in group_rows
            ),
            "mean_relative_holonomy": _mean(
                ((row.get("transport") or {}).get("mean_relative_holonomy")) for row in group_rows
            ),
            "mean_record_count": _mean(
                ((row.get("transport") or {}).get("record_count")) for row in group_rows
            ),
        }
    return out


def build_suite(
    payload_paths: Sequence[Path],
    *,
    output_dir: Path,
    max_pairs: int,
    neighbor_count: int,
    projection_dim_cap: Optional[int],
    device_name: str,
    random_seed: int,
    top_records: int,
    config: ObserverSliceTransportConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in payload_paths:
        try:
            rows.append(
                run_payload_probe(
                    path,
                    max_pairs=max_pairs,
                    neighbor_count=neighbor_count,
                    projection_dim_cap=projection_dim_cap,
                    device=device,
                    random_seed=random_seed,
                    top_records=top_records,
                    config=config,
                )
            )
        except Exception as exc:
            failures.append({"payload_path": str(path), "error": str(exc)})

    by_corpus = _aggregate_by(rows, "corpus")
    real_mean = (by_corpus.get("real") or {}).get("mean_excess_holonomy_action")
    control_values = [
        (by_corpus.get(key) or {}).get("mean_excess_holonomy_action")
        for key in ("control_random", "control_shuffled", "control_constant")
    ]
    control_mean = _mean(control_values)
    real_minus_control = (
        float(real_mean - control_mean)
        if real_mean is not None and control_mean is not None
        else None
    )
    transport_supported = bool(rows and _rate(row.get("effect_supported") for row in rows) >= 0.80)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "device": str(device),
        "payload_count": len(payload_paths),
        "successful_payload_count": len(rows),
        "failed_payload_count": len(failures),
        "config": {
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "projection_dim_cap": projection_dim_cap,
            "random_seed": int(random_seed),
            "top_records": int(top_records),
            "semantic_weight": float(config.semantic_weight),
            "observer_switch_weight": float(config.observer_switch_weight),
            "stress_weight": float(config.stress_weight),
            "density_weight": float(config.density_weight),
            "simplex_weight": float(config.simplex_weight),
        },
        "claim_boundary": {
            "safe_claim": (
                "observer-conditioned charts from saved pipeline artifacts exhibit "
                "null-calibrated noncommuting semantic/observer transport"
            ),
            "unsafe_claim": (
                "real-corpus ideological correctness or human-interpretable bias labels "
                "without independent validation"
            ),
            "controlled_synthetic_claim_safe": True,
            "real_corpus_claim_is_geometry_only": True,
        },
        "transport_supported": transport_supported,
        "real_minus_control_mean_excess_holonomy_action": real_minus_control,
        "real_vs_control_gap_supported": bool(real_minus_control is not None and real_minus_control > 0.0),
        "aggregate_by_corpus": by_corpus,
        "aggregate_by_kernel": _aggregate_by(rows, "kernel"),
        "runs": rows,
        "failures": failures,
    }
    artifact_path = output_dir / "observer_slice_transport_scale_suite.json"
    artifact_path.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact["artifact"] = str(artifact_path)
    artifact_path.write_text(json.dumps(_json_safe(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--no-defaults", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--projection-dim-cap", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--top-records", type=int, default=20)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--observer-switch-weight", type=float, default=1.0)
    parser.add_argument("--stress-weight", type=float, default=0.25)
    parser.add_argument("--density-weight", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "observer_slice_transport_scale_suite" / "latest",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_defaults else _default_payloads()
    payloads.extend(args.payload)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    config = ObserverSliceTransportConfig(
        semantic_weight=float(args.semantic_weight),
        observer_switch_weight=float(args.observer_switch_weight),
        stress_weight=float(args.stress_weight),
        density_weight=float(args.density_weight),
    )
    artifact = build_suite(
        payloads,
        output_dir=args.output_dir,
        max_pairs=int(args.max_pairs),
        neighbor_count=int(args.neighbor_count),
        projection_dim_cap=args.projection_dim_cap,
        device_name=str(args.device),
        random_seed=int(args.random_seed),
        top_records=int(args.top_records),
        config=config,
    )
    print(
        json.dumps(
            {
                "artifact": artifact.get("artifact"),
                "transport_supported": artifact.get("transport_supported"),
                "real_vs_control_gap_supported": artifact.get("real_vs_control_gap_supported"),
                "real_minus_control_mean_excess_holonomy_action": artifact.get(
                    "real_minus_control_mean_excess_holonomy_action"
                ),
                "successful_payload_count": artifact.get("successful_payload_count"),
                "failed_payload_count": artifact.get("failed_payload_count"),
                "aggregate_by_corpus": artifact.get("aggregate_by_corpus"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if artifact.get("transport_supported") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
