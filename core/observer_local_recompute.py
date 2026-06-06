from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


LOCAL_RECOMPUTE_MODE = "local_track_recompute"
LOCAL_RECOMPUTE_DEFAULT_VARIANT = "uniform_weighted_rks"
LOCAL_RECOMPUTE_VARIANTS = (
    LOCAL_RECOMPUTE_DEFAULT_VARIANT,
    "focus_weighted_rks",
    "local_tangent_pca",
    "cls_mean_pca",
    "graph_whitened_hybrid",
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _as_numpy(value: Any, *, dtype: Any = np.float64) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(value, dtype=dtype)
    except Exception:
        return None
    return arr if arr.size else None


def _robust_unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) <= 1e-12:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if abs(hi - lo) <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    out = (arr - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _pca_project(features: np.ndarray, *, n_components: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros((int(arr.shape[0]) if arr.ndim == 2 else 0, n_components), dtype=np.float64), {
            "status": "degenerate",
            "explained_variance_ratio": [],
        }
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    k = int(min(max(n_components, 1), vt.shape[0]))
    projected = centered @ vt[:k].T
    if k < n_components:
        projected = np.pad(projected, ((0, 0), (0, n_components - k)), mode="constant")
    denom = float(np.sum(singular_values**2))
    evr = []
    if denom > 1e-12:
        evr = [float(v) for v in ((singular_values[:k] ** 2) / denom)]
    return projected[:, :n_components].astype(np.float64), {
        "status": "ok",
        "explained_variance_ratio": evr,
        "singular_values": [float(v) for v in singular_values[:k]],
    }


def _standardize_columns(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    return (arr - mean) / np.clip(std, 1e-8, None)


def _kernel_pca_features(features: np.ndarray, *, n_components: int = 6, kernel: str = "rbf") -> np.ndarray:
    """Small deterministic nonlinear chart used only by local recenter ablations."""

    x = _standardize_columns(features)
    n_items = int(x.shape[0])
    if n_items < 3:
        return np.zeros((n_items, int(n_components)), dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    positive = dist2[dist2 > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-8)
    if kernel == "imq":
        affinity = 1.0 / np.sqrt(1.0 + dist2 / scale)
    else:
        affinity = np.exp(-dist2 / scale)
    h = np.eye(n_items, dtype=np.float64) - (np.ones((n_items, n_items), dtype=np.float64) / float(n_items))
    centered = h @ affinity @ h
    try:
        values, vectors = np.linalg.eigh(centered)
    except np.linalg.LinAlgError:
        return np.zeros((n_items, int(n_components)), dtype=np.float64)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    k = int(min(max(n_components, 1), vectors.shape[1]))
    coords = vectors[:, :k] * np.sqrt(np.clip(values[:k], 0.0, None))[None, :]
    if k < n_components:
        coords = np.pad(coords, ((0, 0), (0, int(n_components) - k)), mode="constant")
    return np.nan_to_num(coords[:, : int(n_components)], nan=0.0, posinf=0.0, neginf=0.0)


def _diffusion_features(features: np.ndarray, *, n_components: int = 6, k_neighbors: int = 8) -> np.ndarray:
    x = _standardize_columns(features)
    n_items = int(x.shape[0])
    if n_items < 3:
        return np.zeros((n_items, int(n_components)), dtype=np.float64)
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    positive = distances[distances > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-8)
    affinity = np.exp(-(distances * distances) / (scale * scale))
    np.fill_diagonal(affinity, 0.0)
    if 0 < int(k_neighbors) < n_items - 1:
        keep = np.zeros_like(affinity, dtype=bool)
        for row_idx in range(n_items):
            nn = np.argsort(distances[row_idx])[1 : int(k_neighbors) + 1]
            keep[row_idx, nn] = True
        keep = keep | keep.T
        affinity = np.where(keep, affinity, 0.0)
    degree = np.sum(affinity, axis=1, keepdims=True)
    transition = affinity / np.clip(degree, 1e-12, None)
    try:
        values, vectors = np.linalg.eig(transition)
    except np.linalg.LinAlgError:
        return np.zeros((n_items, int(n_components)), dtype=np.float64)
    order = np.argsort(np.real(values))[::-1]
    # Skip the stationary vector and use the next smooth graph coordinates.
    start = 1 if len(order) > 1 else 0
    selected = order[start : start + int(n_components)]
    coords = np.real(vectors[:, selected])
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, int(n_components) - coords.shape[1])), mode="constant")
    return np.nan_to_num(coords[:, : int(n_components)], nan=0.0, posinf=0.0, neginf=0.0)


def _restore_rks_basis(basis_state: Mapping[str, Any]) -> Optional[Any]:
    try:
        import torch
        from core.dirichlet_fusion import SharedRKSBasis
    except Exception:
        return None
    try:
        input_dim = int(basis_state.get("input_dim", 0) or 0)
        output_dim = int(basis_state.get("output_dim", 0) or 0)
        if input_dim <= 0 or output_dim <= 0:
            return None
        omega = basis_state.get("omega")
        b = basis_state.get("b")
        if omega is None or b is None:
            return None
        basis = SharedRKSBasis(
            input_dim=input_dim,
            output_dim=output_dim,
            seed=int(basis_state.get("seed", 0) or 0),
            kernel_type=str(basis_state.get("kernel_type", "rbf")),
            nu=float(basis_state.get("nu", 1.5)),
            roughness=int(basis_state.get("roughness", 3) or 3),
        )
        basis.omega.copy_(torch.as_tensor(omega, dtype=basis.omega.dtype))
        basis.b.copy_(torch.as_tensor(b, dtype=basis.b.dtype))
        sigma = basis_state.get("sigma")
        if sigma is not None:
            basis.set_sigma(float(sigma))
        return basis
    except Exception:
        return None


def _project_track2(local_tangent: np.ndarray, payload: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    basis_state = payload.get("rks_basis_state")
    if isinstance(basis_state, Mapping):
        basis = _restore_rks_basis(basis_state)
        if basis is not None:
            try:
                import torch

                with torch.no_grad():
                    projected = basis(torch.as_tensor(local_tangent, dtype=torch.float32))
                arr = projected.detach().cpu().numpy().astype(np.float64)
                return arr, {
                    "track2_basis": "shared_rks_basis",
                    "kernel_type": str(basis_state.get("kernel_type", "unknown")),
                    "rks_output_dim": int(arr.shape[1]),
                }
            except Exception as exc:
                return local_tangent, {
                    "track2_basis": "local_cls_tangent",
                    "rks_fallback_reason": f"{type(exc).__name__}: {exc}",
                }
    return local_tangent, {"track2_basis": "local_cls_tangent"}


def _focus_weights(payload: Mapping[str, Any], focus_idx: int, n_bots: int) -> Tuple[np.ndarray, str]:
    probe = _as_numpy(payload.get("spectral_probe_magnitudes"))
    if probe is not None and probe.ndim == 2 and focus_idx < probe.shape[0] and probe.shape[1] == n_bots:
        raw = np.abs(np.asarray(probe[focus_idx], dtype=np.float64))
        total = float(np.sum(raw))
        if math.isfinite(total) and total > 1e-12:
            return raw / total, "focus_spectral_probe_magnitudes"
    return np.ones(n_bots, dtype=np.float64) / float(max(n_bots, 1)), "uniform_fallback"


def _local_weights_for_variant(
    payload: Mapping[str, Any],
    focus_idx: int,
    n_bots: int,
    variant: str,
) -> Tuple[np.ndarray, str]:
    if variant == "uniform_weighted_rks":
        return (
            np.ones(n_bots, dtype=np.float64) / float(max(n_bots, 1)),
            "uniform_variant",
        )
    return _focus_weights(payload, focus_idx, n_bots)


def _track2_features_for_variant(
    *,
    cls: np.ndarray,
    local_tangent: np.ndarray,
    payload: Mapping[str, Any],
    focus_idx: int,
    variant: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if variant in {"focus_weighted_rks", "uniform_weighted_rks"}:
        features, diag = _project_track2(local_tangent, payload)
        diag = dict(diag)
        diag["variant_projection_policy"] = "rks_project_local_tangent"
        return features, diag
    if variant == "local_tangent_pca":
        return local_tangent, {
            "track2_basis": "local_cls_tangent",
            "variant_projection_policy": "skip_rks_use_weighted_local_tangent",
        }
    if variant == "cls_mean_pca":
        mean_cls = np.mean(np.asarray(cls, dtype=np.float64), axis=1)
        return mean_cls - mean_cls[int(focus_idx) : int(focus_idx) + 1], {
            "track2_basis": "mean_cls_per_bot",
            "variant_projection_policy": "skip_rks_use_mean_observer_cls",
        }
    if variant == "graph_whitened_hybrid":
        mean_cls = np.mean(np.asarray(cls, dtype=np.float64), axis=1)
        mean_cls = mean_cls - mean_cls[int(focus_idx) : int(focus_idx) + 1]
        rks_features, rks_diag = _project_track2(local_tangent, payload)
        spectral = _as_numpy(payload.get("spectral_probe_magnitudes"))
        blocks = [
            _kernel_pca_features(mean_cls, n_components=6, kernel="rbf"),
            _kernel_pca_features(mean_cls, n_components=6, kernel="imq"),
            _diffusion_features(mean_cls, n_components=6, k_neighbors=8),
            _kernel_pca_features(rks_features, n_components=6, kernel="rbf"),
            _diffusion_features(rks_features, n_components=6, k_neighbors=8),
        ]
        if spectral is not None and spectral.ndim == 2 and spectral.shape[0] == cls.shape[0]:
            spectral_delta = np.asarray(spectral, dtype=np.float64) - np.asarray(
                spectral[int(focus_idx) : int(focus_idx) + 1],
                dtype=np.float64,
            )
            blocks.append(_standardize_columns(spectral_delta))
        features = np.concatenate([_standardize_columns(block) for block in blocks], axis=1)
        return features, {
            "track2_basis": "graph_whitened_hybrid",
            "variant_projection_policy": "kernel_pca_plus_diffusion_over_mean_cls_and_rks",
            "rks_basis": rks_diag,
            "feature_blocks": len(blocks),
            "hybrid_feature_dim": int(features.shape[1]),
        }
    raise ValueError(f"Unsupported local recompute variant={variant!r}")


def _terrain_z(density: np.ndarray, stress: np.ndarray) -> np.ndarray:
    return np.clip(0.25 - (0.25 * density) + (0.75 * stress), 0.0, 1.0)


def _zones_from_density_stress(density: np.ndarray, stress: np.ndarray) -> List[str]:
    density = np.asarray(density, dtype=np.float64)
    stress = np.asarray(stress, dtype=np.float64)
    density_median = float(np.median(density[np.isfinite(density)])) if np.isfinite(density).any() else 0.5
    stress_median = float(np.median(stress[np.isfinite(stress)])) if np.isfinite(stress).any() else 0.5
    zones: List[str] = []
    for d_val, s_val in zip(density, stress):
        if d_val >= density_median and s_val < stress_median:
            zones.append("Bridge")
        elif d_val >= density_median and s_val >= stress_median:
            zones.append("Swamp")
        elif d_val < density_median and s_val < stress_median:
            zones.append("Tightrope")
        else:
            zones.append("Void")
    return zones


@dataclass(frozen=True)
class LocalObserverRecompute:
    focus_idx: int
    articles: List[Dict[str, Any]]
    track2_features: np.ndarray
    projection_xyz: np.ndarray
    density: np.ndarray
    stress: np.ndarray
    z_height: np.ndarray
    zones: List[str]
    diagnostics: Dict[str, Any]

    def article_map(self) -> Dict[int, Dict[str, Any]]:
        return {int(row["idx"]): dict(row) for row in self.articles}

    def to_public_dict(self) -> Dict[str, Any]:
        return _json_safe(
            {
                "mode": LOCAL_RECOMPUTE_MODE,
                "variant": self.diagnostics.get("variant"),
                "focus_idx": self.focus_idx,
                "article_count": len(self.articles),
                "track2_feature_shape": list(self.track2_features.shape),
                "projection_shape": list(self.projection_xyz.shape),
                "density_min": float(np.min(self.density)) if self.density.size else None,
                "density_max": float(np.max(self.density)) if self.density.size else None,
                "stress_min": float(np.min(self.stress)) if self.stress.size else None,
                "stress_max": float(np.max(self.stress)) if self.stress.size else None,
                "zone_counts": {
                    zone: int(sum(1 for value in self.zones if value == zone))
                    for zone in sorted(set(self.zones))
                },
                "diagnostics": self.diagnostics,
            }
        )


def load_primary_observer_payload(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    try:
        import torch
    except Exception as exc:
        return None, None, f"torch_unavailable: {exc}"
    run_dir = Path(run_dir)
    candidates = [run_dir / "observer_global.pt", *sorted(run_dir.glob("observer_*.pt"))]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("cls_per_bot") is not None:
            return payload, str(path), None
    return None, None, "no_observer_payload_with_cls_per_bot"


def compute_local_observer_recenter(
    payload: Mapping[str, Any],
    focus_idx: int,
    *,
    tau: float = 1.0,
    variant: str = LOCAL_RECOMPUTE_DEFAULT_VARIANT,
) -> Optional[LocalObserverRecompute]:
    variant = str(variant or LOCAL_RECOMPUTE_DEFAULT_VARIANT).strip().lower()
    if variant not in LOCAL_RECOMPUTE_VARIANTS:
        raise ValueError(f"Unsupported local recompute variant={variant!r}")
    cls = _as_numpy(payload.get("cls_per_bot"))
    if cls is None or cls.ndim != 3:
        return None
    n_articles, n_bots, hidden_dim = cls.shape
    focus_idx = int(focus_idx)
    if focus_idx < 0 or focus_idx >= n_articles:
        return None

    weights, weight_basis = _local_weights_for_variant(payload, focus_idx, n_bots, variant)
    focus_views = np.asarray(cls[focus_idx], dtype=np.float64)
    deltas = np.asarray(cls, dtype=np.float64) - focus_views[None, :, :]
    local_tangent = np.einsum("b,nbh->nh", weights, deltas)
    residual = deltas - local_tangent[:, None, :]
    residual_energy = np.sqrt(
        np.maximum(np.einsum("b,nbh,nbh->n", weights, residual, residual), 0.0)
        / float(max(hidden_dim, 1))
    )

    track2_features, track2_diag = _track2_features_for_variant(
        cls=np.asarray(cls, dtype=np.float64),
        local_tangent=local_tangent,
        payload=payload,
        focus_idx=focus_idx,
        variant=variant,
    )
    track2_features = np.asarray(track2_features, dtype=np.float64)
    if track2_features.ndim != 2 or track2_features.shape[0] != n_articles:
        return None
    track2_features = track2_features - track2_features[focus_idx : focus_idx + 1]

    projection, projection_diag = _pca_project(track2_features, n_components=3)
    if projection.shape[0] != n_articles:
        return None
    projection = projection - projection[focus_idx : focus_idx + 1]

    probe = _as_numpy(payload.get("spectral_probe_magnitudes"))
    if probe is not None and probe.ndim == 2 and probe.shape[0] == n_articles:
        probe_delta = np.linalg.norm(probe - probe[focus_idx : focus_idx + 1], axis=1)
    else:
        probe_delta = np.zeros(n_articles, dtype=np.float64)

    stress = np.clip(0.75 * _robust_unit(residual_energy) + 0.25 * _robust_unit(probe_delta), 0.0, 1.0)
    variance_scale = float(np.median(residual_energy[np.isfinite(residual_energy)]))
    if not math.isfinite(variance_scale) or variance_scale <= 1e-12:
        variance_scale = float(np.mean(residual_energy[np.isfinite(residual_energy)])) if np.isfinite(residual_energy).any() else 1.0
    if not math.isfinite(variance_scale) or variance_scale <= 1e-12:
        variance_scale = 1.0
    rho = 1.0 / (1.0 + (float(tau) * residual_energy / variance_scale))
    density = np.clip(np.nan_to_num(rho, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    z_height = _terrain_z(density, stress)
    projection[:, 2] = z_height
    zones = _zones_from_density_stress(density, stress)

    articles: List[Dict[str, Any]] = []
    for idx in range(n_articles):
        articles.append(
            {
                "idx": int(idx),
                "index": int(idx),
                "x": float(projection[idx, 0]),
                "y": float(projection[idx, 1]),
                "z": float(projection[idx, 2]),
                "density": float(density[idx]),
                "stress": float(stress[idx]),
                "z_height": float(z_height[idx]),
                "zone": zones[idx],
                "local_track2_x": float(projection[idx, 0]),
                "local_track2_y": float(projection[idx, 1]),
                "local_track2_z": float(projection[idx, 2]),
                "local_track15_stress": float(stress[idx]),
                "local_track3_density": float(density[idx]),
                "local_recompute_mode": LOCAL_RECOMPUTE_MODE,
                "local_recompute_variant": variant,
            }
        )

    diagnostics = {
        "status": "OK",
        "mode": LOCAL_RECOMPUTE_MODE,
        "variant": variant,
        "focus_idx": int(focus_idx),
        "n_articles": int(n_articles),
        "n_bots": int(n_bots),
        "hidden_dim": int(hidden_dim),
        "focus_weight_basis": weight_basis,
        "focus_weights": [float(v) for v in weights],
        "track2": track2_diag,
        "projection": projection_diag,
        "track15_stress_basis": "weighted_cls_residual_energy_plus_probe_delta",
        "track3_density_basis": "rho_i=1/(1+tau*local_residual_energy/median_residual_energy)",
        "tau": float(tau),
        "terrain_zone_basis": "local_density_median_x_local_stress_median",
    }
    return LocalObserverRecompute(
        focus_idx=int(focus_idx),
        articles=articles,
        track2_features=track2_features,
        projection_xyz=projection,
        density=density,
        stress=stress,
        z_height=z_height,
        zones=zones,
        diagnostics=diagnostics,
    )


def compute_local_observer_recenter_from_run(
    run_dir: Path,
    focus_idx: int,
    *,
    tau: float = 1.0,
    variant: str = LOCAL_RECOMPUTE_DEFAULT_VARIANT,
) -> Tuple[Optional[LocalObserverRecompute], Dict[str, Any]]:
    payload, payload_path, error = load_primary_observer_payload(Path(run_dir))
    if payload is None:
        return None, {
            "status": "NO_DATA",
            "mode": LOCAL_RECOMPUTE_MODE,
            "variant": variant,
            "error": error,
        }
    result = compute_local_observer_recenter(payload, int(focus_idx), tau=tau, variant=variant)
    if result is None:
        return None, {
            "status": "FAILED",
            "mode": LOCAL_RECOMPUTE_MODE,
            "variant": variant,
            "payload_path": payload_path,
            "error": "local_recompute_returned_none",
        }
    public = result.to_public_dict()
    public["payload_path"] = payload_path
    return result, public
