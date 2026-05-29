#!/usr/bin/env python3
"""Replay Track 4 as a least-action metric graph from existing artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.track4_action_graph import ActionGraphConfig, export_action_graph, run_action_graph


def _infer_action_mode(action_branch: str, explicit_mode: str | None = None) -> str:
    if explicit_mode is not None and str(explicit_mode).strip():
        return str(explicit_mode).strip()
    branch = str(action_branch or "baseline_raw_action").strip().lower()
    if branch == "null_calibrated_hysteresis":
        return "null_calibrated_hysteresis"
    if branch == "richer_walker_state":
        return "richer_state"
    if branch == "virtual_transition_states":
        return "virtual_transition_states"
    return "baseline"


def _default_null_observer_mode(action_mode: str, explicit_mode: str | None = None) -> str:
    if explicit_mode is not None and str(explicit_mode).strip():
        return str(explicit_mode).strip().lower()
    return "shuffled" if str(action_mode).strip().lower() in {"null_calibrated", "null_calibrated_hysteresis"} else "disabled"


def _default_virtual_steps(action_mode: str, explicit_steps: int | None = None) -> int:
    if explicit_steps is not None:
        return max(int(explicit_steps), 0)
    return 3 if str(action_mode).strip().lower() in {"virtual_transition", "virtual_transition_states"} else 0


def _load_observer(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"observer artifact not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"observer artifact must be a dict: {path}")
    return payload


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _select_embeddings(observer: Dict[str, Any], basis: str) -> np.ndarray:
    basis_norm = str(basis or "track2").strip().lower()
    candidates = {
        "track2": ("T2_kernels", "embeddings", "features"),
        "integrated": ("integrated_vectors", "embeddings", "features"),
        "logits_flat": ("T1_embeddings", "embeddings", "features"),
        "spectral": ("spectral_antagonism",),
    }.get(basis_norm)
    if not candidates:
        raise ValueError("basis must be one of: track2, integrated, logits_flat, spectral")
    for key in candidates:
        value = _to_numpy(observer.get(key))
        if value is not None and value.ndim == 2 and value.shape[0] > 0:
            return value.astype(np.float32, copy=False)
    raise ValueError(f"could not resolve embeddings for basis={basis!r}; tried {candidates}")


def _select_density(observer: Dict[str, Any]) -> np.ndarray | None:
    topology = observer.get("T3_topology")
    if isinstance(topology, dict):
        for key in ("dirichlet_fused_std", "fused_std", "density", "rho"):
            value = _to_numpy(topology.get(key))
            if value is not None:
                return value.astype(np.float32, copy=False)
    return None


def _select_stress(observer: Dict[str, Any]) -> np.ndarray | None:
    for key in ("spectral_antagonism", "spectral_probe_magnitudes", "spectral_u_axis"):
        value = _to_numpy(observer.get(key))
        if value is not None:
            return value.astype(np.float32, copy=False)
    spectral = observer.get("T1.5_spectral")
    if isinstance(spectral, dict):
        for key in ("antagonism", "probe_magnitudes", "u_axis"):
            value = _to_numpy(spectral.get(key))
            if value is not None:
                return value.astype(np.float32, copy=False)
    return None


def _select_observer_simplex(observer: Dict[str, Any]) -> np.ndarray | None:
    """Derive an [N, 8] V-observer simplex from existing artifacts."""

    for key in ("track3_weight_cold", "track3_observer_simplex", "observer_simplex"):
        value = _to_numpy(observer.get(key))
        if value is not None and value.ndim == 2:
            return value.astype(np.float32, copy=False)
    topology = observer.get("T3_topology")
    if isinstance(topology, dict):
        for key in ("observer_simplex", "final_weights", "cold_weights", "weight_simplex"):
            value = _to_numpy(topology.get(key))
            if value is not None and value.ndim == 2:
                return value.astype(np.float32, copy=False)
    spectral = observer.get("T1.5_spectral")
    if isinstance(spectral, dict):
        value = _to_numpy(spectral.get("probe_magnitudes"))
        if value is not None and value.ndim == 2 and value.shape[1] == 8:
            weights = np.exp(np.abs(value) - np.max(np.abs(value), axis=1, keepdims=True))
            return (weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-9, None)).astype(np.float32)
    value = _to_numpy(observer.get("spectral_probe_magnitudes"))
    if value is not None and value.ndim == 2 and value.shape[1] == 8:
        weights = np.exp(np.abs(value) - np.max(np.abs(value), axis=1, keepdims=True))
        return (weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-9, None)).astype(np.float32)
    cls_per_bot = _to_numpy(observer.get("cls_per_bot"))
    if cls_per_bot is not None and cls_per_bot.ndim == 3 and cls_per_bot.shape[1] == 8:
        magnitudes = np.linalg.norm(cls_per_bot, axis=-1)
        weights = np.exp(magnitudes - np.max(magnitudes, axis=1, keepdims=True))
        return (weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-9, None)).astype(np.float32)
    return None


def _apply_observer_state_mode(
    observer_simplex: np.ndarray | None,
    *,
    mode: str,
    seed: int,
) -> np.ndarray | None:
    """Apply replay-only observer-state ablations without changing artifacts."""

    mode_norm = str(mode or "artifact").strip().lower()
    if mode_norm == "artifact":
        return observer_simplex
    if mode_norm == "disabled":
        return None
    if mode_norm == "shuffled":
        if observer_simplex is None:
            return None
        rng = np.random.default_rng(int(seed))
        shuffled = np.asarray(observer_simplex, dtype=np.float32).copy()
        rng.shuffle(shuffled, axis=0)
        return shuffled.astype(np.float32, copy=False)
    raise ValueError("observer_state_mode must be one of: artifact, disabled, shuffled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-path", type=Path, required=True, help="Existing observer_*.pt artifact.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for least-action outputs.")
    parser.add_argument(
        "--action-branch",
        default="baseline_raw_action",
        help="Lab-only Track 4 engineering branch label threaded into manifests and summaries.",
    )
    parser.add_argument(
        "--action-mode",
        "--core-action-mode",
        dest="action_mode",
        default=None,
        choices=("baseline", "null_calibrated", "null_calibrated_hysteresis", "richer_state", "virtual_transition", "virtual_transition_states"),
        help="Core action mechanics to use. Defaults are inferred from --action-branch.",
    )
    parser.add_argument(
        "--null-calibrated-hysteresis",
        action="store_true",
        help="Compatibility flag: equivalent to --core-action-mode null_calibrated.",
    )
    parser.add_argument(
        "--richer-walker-state",
        action="store_true",
        help="Compatibility flag: equivalent to --core-action-mode richer_state.",
    )
    parser.add_argument(
        "--virtual-transition-states",
        action="store_true",
        help="Compatibility flag: equivalent to --core-action-mode virtual_transition.",
    )
    parser.add_argument(
        "--basis",
        default="track2",
        choices=("track2", "integrated", "logits_flat", "spectral"),
        help="Existing observer basis to traverse.",
    )
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--stress-weight", type=float, default=1.5)
    parser.add_argument("--shear-weight", type=float, default=1.0)
    parser.add_argument("--observer-transport-weight", type=float, default=1.0)
    parser.add_argument("--hysteresis-weight", type=float, default=0.5)
    parser.add_argument(
        "--observer-state-mode",
        default="artifact",
        choices=("artifact", "disabled", "shuffled"),
        help="Replay observer simplex as-is, remove it, or shuffle it across articles.",
    )
    parser.add_argument("--observer-shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--null-observer-state-mode",
        default=None,
        choices=("disabled", "artifact", "shuffled"),
        help="Observer simplex used as the null calibration baseline. Defaults to shuffled for null-calibrated mode.",
    )
    parser.add_argument(
        "--null-shuffle-seed",
        type=int,
        default=None,
        help="Seed for the shuffled null observer simplex. Defaults to --observer-shuffle-seed.",
    )
    parser.add_argument("--void-weight", type=float, default=1.0)
    parser.add_argument("--curvature-weight", type=float, default=0.75)
    parser.add_argument(
        "--use-virtual-transitions",
        action="store_true",
        help="Enable interpolated virtual transition nodes along existing graph edges.",
    )
    parser.add_argument(
        "--virtual-interpolation-steps",
        type=int,
        default=None,
        help="Number of virtual nodes to insert per existing graph edge. Inferred as 3 for virtual_transition_states.",
    )
    parser.add_argument("--work-bucket-count", type=int, default=6)
    parser.add_argument("--richer-state-memory-weight", type=float, default=0.05)
    parser.add_argument("--max-paths", type=int, default=12)
    parser.add_argument("--target-count-per-anchor", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observer = _load_observer(args.observer_path)
    embeddings = _select_embeddings(observer, args.basis)
    density = _select_density(observer)
    stress = _select_stress(observer)
    raw_observer_simplex = _select_observer_simplex(observer)
    observer_simplex = _apply_observer_state_mode(
        raw_observer_simplex,
        mode=args.observer_state_mode,
        seed=int(args.observer_shuffle_seed),
    )
    action_mode = _infer_action_mode(str(args.action_branch), args.action_mode)
    if args.null_calibrated_hysteresis:
        action_mode = "null_calibrated_hysteresis"
    if args.richer_walker_state:
        action_mode = "richer_state"
    if args.virtual_transition_states:
        action_mode = "virtual_transition_states"
    null_observer_state_mode = _default_null_observer_mode(action_mode, args.null_observer_state_mode)
    null_shuffle_seed = int(args.null_shuffle_seed) if args.null_shuffle_seed is not None else int(args.observer_shuffle_seed)
    null_observer_simplex = _apply_observer_state_mode(
        raw_observer_simplex,
        mode=null_observer_state_mode,
        seed=null_shuffle_seed,
    )
    virtual_interpolation_steps = _default_virtual_steps(action_mode, args.virtual_interpolation_steps)
    if args.use_virtual_transitions and args.virtual_interpolation_steps is None and virtual_interpolation_steps <= 0:
        virtual_interpolation_steps = 1
    use_virtual_transitions = bool(args.use_virtual_transitions or virtual_interpolation_steps > 0)
    config = ActionGraphConfig(
        k_neighbors=int(args.k_neighbors),
        action_mode=str(action_mode),
        stress_weight=float(args.stress_weight),
        shear_weight=float(args.shear_weight),
        observer_transport_weight=float(args.observer_transport_weight),
        hysteresis_weight=float(args.hysteresis_weight),
        void_weight=float(args.void_weight),
        curvature_weight=float(args.curvature_weight),
        max_paths=int(args.max_paths),
        target_count_per_anchor=int(args.target_count_per_anchor),
        use_virtual_transitions=use_virtual_transitions,
        virtual_interpolation_steps=int(virtual_interpolation_steps),
        work_bucket_count=int(args.work_bucket_count),
        richer_state_memory_weight=float(args.richer_state_memory_weight),
    )
    result = run_action_graph(
        embeddings,
        track3_density=density,
        metric_stress=stress,
        observer_simplex=observer_simplex,
        null_observer_simplex=null_observer_simplex,
        action_branch=str(args.action_branch),
        config=config,
    )
    paths = export_action_graph(result, args.output_dir)
    manifest = {
        "schema_version": "1.0",
        "runner": "run_track4_action_graph",
        "observer_path": str(args.observer_path),
        "action_branch": str(args.action_branch),
        "action_mode": str(action_mode),
        "basis": str(args.basis),
        "n_articles": int(embeddings.shape[0]),
        "density_available": density is not None,
        "stress_available": stress is not None,
        "observer_state_mode": str(args.observer_state_mode),
        "observer_shuffle_seed": int(args.observer_shuffle_seed),
        "null_observer_state_mode": str(null_observer_state_mode),
        "null_shuffle_seed": int(null_shuffle_seed),
        "observer_simplex_available": observer_simplex is not None,
        "null_observer_simplex_available": null_observer_simplex is not None,
        "raw_observer_simplex_available": raw_observer_simplex is not None,
        "observer_simplex_shape": list(observer_simplex.shape) if observer_simplex is not None else None,
        "null_observer_simplex_shape": list(null_observer_simplex.shape) if null_observer_simplex is not None else None,
        "virtual_transitions_enabled": bool(result["summary"].get("virtual_transitions_enabled")),
        "virtual_node_count": int(result["summary"].get("virtual_node_count") or 0),
        "virtual_interpolation_steps": int(virtual_interpolation_steps),
        "calibrated_hysteresis_terms": {
            "mean_hysteresis_penalty": result["summary"].get("mean_hysteresis_penalty"),
            "mean_null_hysteresis_penalty": result["summary"].get("mean_null_hysteresis_penalty"),
            "mean_excess_hysteresis_penalty": result["summary"].get("mean_excess_hysteresis_penalty"),
            "mean_positive_excess_hysteresis_penalty": result["summary"].get("mean_positive_excess_hysteresis_penalty"),
            "mean_calibrated_hysteresis_penalty": result["summary"].get("mean_calibrated_hysteresis_penalty"),
        },
        "outputs": paths,
        "summary": result["summary"],
    }
    manifest_path = Path(args.output_dir) / "track4_action_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=OK")
    print(f"action_branch={args.action_branch}")
    print(f"action_mode={action_mode}")
    print(f"basis={args.basis}")
    print(f"n_articles={embeddings.shape[0]}")
    print(f"virtual_node_count={result['summary'].get('virtual_node_count')}")
    print(f"summary={paths['summary']}")
    print(f"npz={paths['npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
