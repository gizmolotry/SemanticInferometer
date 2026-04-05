#!/usr/bin/env python3
"""
===============================================================================
WATERFALL ABLATION VISUALIZATION — 4-Panel Forensic Dashboard
===============================================================================
analysis/waterfall_viz.py

THE FORENSIC TIMELINE: Layer-by-layer signal visualization to diagnose where
NMI drops through the pipeline.

CHECKPOINT PANELS:
    CP-1 (Track 1)   : "The Raw Signal"      - Logits (24D → 2D PCA)
    CP-2 (Track 2)   : "The Flat Map"        - RKS Features (2D projection)
    CP-3 (Track 1.5) : "The Geometric Shape" - Spectral Dipoles (3D manifold)
    CP-4 (Synthesis) : "The World"           - Full Track 3 topology

If NMI drops from 0.7 to 0.1, the panel where points become unstructured
noise reveals the broken layer.

Author: Belief Transformer Project (ASTER v3.2)
===============================================================================
"""

from __future__ import annotations

import importlib.util
import io
import json
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if HAS_SKLEARN:
    try:
        from sklearn.manifold import TSNE
        HAS_TSNE = True
    except ImportError:
        TSNE = None
        HAS_TSNE = False
else:
    TSNE = None
    HAS_TSNE = False

HAS_UMAP = importlib.util.find_spec("umap") is not None


# =============================================================================
# FIGHTER JET PALETTE (consistent with MONOLITH_VIZ)
# =============================================================================
@dataclass
class WaterfallPalette:
    """Neon on void color scheme for waterfall panels."""
    # Background
    void: str = "#050505"
    grid_dark: str = "#0a0a0f"
    grid_line: str = "#1a1a2e"

    # Panel-specific colors
    cp1_primary: str = "#FF6B6B"     # Raw logits - coral red
    cp2_primary: str = "#4ECDC4"     # RKS flat - teal
    cp3_primary: str = "#9B59B6"     # Spectral - purple
    cp4_primary: str = "#F39C12"     # Synthesis - gold

    # Cluster colors (for ground truth labels)
    clusters: List[str] = field(default_factory=lambda: [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#9B59B6",  # Purple
        "#F39C12",  # Gold
        "#3498DB",  # Blue
        "#2ECC71",  # Green
        "#E74C3C",  # Dark red
        "#1ABC9C",  # Turquoise
        "#8E44AD",  # Dark purple
        "#F1C40F",  # Yellow
        "#E67E22",  # Orange
        "#95A5A6",  # Gray
    ])

    # Signal quality
    good_signal: str = "#00FF41"
    degraded_signal: str = "#FFD700"
    no_signal: str = "#FF2A00"

    # Text
    text_primary: str = "#e0e0e0"
    text_dim: str = "#888888"


PALETTE = WaterfallPalette()


# =============================================================================
# WATERFALL DATA LOADER
# =============================================================================
@dataclass
class WaterfallData:
    """Container for loaded checkpoint data.

    Accumulative Metric View (from the metric tensor):
    g_μν^total = (1/ρ) · δ_μν + ∇_μΦ∇_νΦ

    - CP-1: Logits only (Track 1 raw signal)
    - CP-2: Track 2 only (δ_μν - base geometry from RKS)
    - CP-3: Track 2 + Track 1.5 (adds ∇Φ∇Φ - observer shear)
    - CP-4: Full synthesis (adds 1/ρ - Track 3 inverse density)
    """
    # Core arrays
    t0_logits: Optional[np.ndarray] = None          # [N, 8, 3] or [N, 24] - Track 1 raw
    t1_embeddings: Optional[np.ndarray] = None      # [N, hidden_dim]
    t15_spectral: Optional[Dict] = None             # u_axis, evr, probe_magnitudes - Track 1.5
    t2_kernels: Optional[Dict] = None               # z_rbf, z_matern, etc. - Track 2
    t3_topology: Optional[Dict] = None              # bond_matrix, crack_matrix, blinker - Track 3
    t4_viz: Optional[Dict] = None                   # x_umap, x_tsne

    # Track 3 specific (inverse density / blinker variance)
    t3_blinker: Optional[np.ndarray] = None         # [N] or [N, D] - variance/entropy
    t3_density: Optional[np.ndarray] = None         # [N] local density ρ(x)

    # Ground truth
    labels: Optional[np.ndarray] = None             # [N] integer labels
    label_names: Optional[List[str]] = None         # Label string names

    # Metadata
    run_id: str = ""
    n_articles: int = 0
    checkpoint_dir: Optional[Path] = None
    contract_track_nmi: Dict[str, float] = field(default_factory=dict)
    contract_track_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Computed 2D projections for visualization
    cp1_2d: Optional[np.ndarray] = None             # [N, 2] PCA of logits (Track 1 only)
    cp2_2d: Optional[np.ndarray] = None             # [N, 2] Track 2 only (base geometry)
    cp3_2d: Optional[np.ndarray] = None             # [N, 2] Track 2 + Track 1.5 (+ observer shear)
    cp4_2d: Optional[np.ndarray] = None             # [N, 2] Full synthesis (+ Track 3 density)

    # Raw features for NMI (before 2D projection)
    cp1_raw: Optional[np.ndarray] = None            # [N, 24] logits flat
    cp2_raw: Optional[np.ndarray] = None            # [N, D] Track 2 features
    cp3_raw: Optional[np.ndarray] = None            # [N, D+8] Track 2 + Track 1.5 concatenated
    cp4_raw: Optional[np.ndarray] = None            # [N, D+8+1] Full metric with Track 3
    synthesis_features: Optional[np.ndarray] = None # [N, D] Final synthesis feature fallback


def load_waterfall_checkpoints(
    checkpoint_dir: Path,
    ground_truth: Optional[Dict[int, str]] = None,
) -> WaterfallData:
    """
    Load all checkpoint artifacts from a waterfall checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoints/{run_id}/ directory
        ground_truth: Optional dict mapping article index to label string

    Returns:
        WaterfallData with all loaded artifacts
    """
    checkpoint_dir = Path(checkpoint_dir)
    data = WaterfallData(
        run_id=checkpoint_dir.name,
        checkpoint_dir=checkpoint_dir,
    )

    # T0: Logit substrate
    t0_path = checkpoint_dir / "T0_substrate.npy"
    if t0_path.exists():
        data.t0_logits = np.load(t0_path)
        data.n_articles = data.t0_logits.shape[0]
        print(f"[Waterfall] Loaded T0 logits: {data.t0_logits.shape}")

    # T1: Embeddings
    t1_path = checkpoint_dir / "T1_embeddings_raw.npy"
    if t1_path.exists():
        data.t1_embeddings = np.load(t1_path)
        if data.n_articles == 0:
            data.n_articles = data.t1_embeddings.shape[0]
        print(f"[Waterfall] Loaded T1 embeddings: {data.t1_embeddings.shape}")

    # T1.5: Spectral state
    t15_path = checkpoint_dir / "T1.5_spectral_state.npz"
    if t15_path.exists():
        data.t15_spectral = dict(np.load(t15_path, allow_pickle=True))
        if data.n_articles == 0 and "probe_magnitudes" in data.t15_spectral:
            data.n_articles = data.t15_spectral["probe_magnitudes"].shape[0]
        print(f"[Waterfall] Loaded T1.5 spectral: {list(data.t15_spectral.keys())}")

    # T2: Kernel projections
    t2_path = checkpoint_dir / "T2_kernel_projections.npz"
    if t2_path.exists():
        data.t2_kernels = dict(np.load(t2_path, allow_pickle=True))
        if data.n_articles == 0:
            for key in ["z_rbf", "z_matern", "z_laplacian", "z_imq"]:
                if key in data.t2_kernels:
                    data.n_articles = data.t2_kernels[key].shape[0]
                    break
        print(f"[Waterfall] Loaded T2 kernels: {list(data.t2_kernels.keys())}")

    # =========================================================================
    # FALLBACK: Load from experiment directory (parent of checkpoints)
    # =========================================================================
    exp_dir = checkpoint_dir.parent.parent  # checkpoints/{run_id} -> {run_dir}

    # Fallback T2: dirichlet_fused.npy
    if data.t2_kernels is None:
        fused_path = exp_dir / "dirichlet_fused.npy"
        if fused_path.exists():
            fused = np.load(fused_path)
            data.t2_kernels = {"z_rbf": fused}  # Treat as RBF kernel output
            if data.n_articles == 0:
                data.n_articles = fused.shape[0]
            print(f"[Waterfall] Loaded T2 (fallback) from dirichlet_fused.npy: {fused.shape}")

    # Fallback T1.5: spectral_probe_magnitudes.npy + spectral_evr.npy
    if data.t15_spectral is None:
        spec_mags = exp_dir / "spectral_probe_magnitudes.npy"
        spec_evr = exp_dir / "spectral_evr.npy"
        if spec_mags.exists():
            probe_mags = np.load(spec_mags)
            evr = np.load(spec_evr) if spec_evr.exists() else np.ones(probe_mags.shape[0])
            data.t15_spectral = {
                "probe_magnitudes": probe_mags,
                "evr": evr,
            }
            if data.n_articles == 0:
                data.n_articles = probe_mags.shape[0]
            print(f"[Waterfall] Loaded T1.5 (fallback) from spectral_*.npy: {probe_mags.shape}")

    # Fallback T0: Try to load from observer_*.pt
    if data.t0_logits is None:
        observer_files = list(exp_dir.glob("observer_*.pt"))
        if observer_files:
            try:
                import torch
                obs_data = torch.load(observer_files[0], weights_only=False)
                # Look for logits in various places
                if "logits" in obs_data:
                    data.t0_logits = obs_data["logits"].detach().cpu().numpy() if hasattr(obs_data["logits"], "detach") else obs_data["logits"]
                    if data.n_articles == 0:
                        data.n_articles = data.t0_logits.shape[0]
                    print(f"[Waterfall] Loaded T0 (fallback) from observer_*.pt: {data.t0_logits.shape}")
            except Exception as e:
                print(f"[Waterfall] Could not load T0 from observer file: {e}")

    # =========================================================================
    # TRACK 3: Load blinker variance / density (for full metric synthesis)
    # =========================================================================
    # Track 3 = 1/ρ(x) in the metric tensor

    # Try dirichlet_fused_std.npy (blinker variance)
    blinker_path = exp_dir / "dirichlet_fused_std.npy"
    if blinker_path.exists():
        data.t3_blinker = np.load(blinker_path)
        print(f"[Waterfall] Loaded Track 3 blinker: {data.t3_blinker.shape}")

    # Try logit_confidence.npy as alternative density proxy
    if data.t3_blinker is None:
        conf_path = exp_dir / "logit_confidence.npy"
        if conf_path.exists():
            # Inverse confidence = uncertainty ~ 1/ρ
            conf = np.load(conf_path)
            data.t3_blinker = 1.0 - conf  # Convert confidence to uncertainty
            print(f"[Waterfall] Loaded Track 3 (from logit_confidence): {data.t3_blinker.shape}")

    synthesis_path = exp_dir / "features.npy"
    if synthesis_path.exists():
        data.synthesis_features = np.load(synthesis_path)
        print(f"[Waterfall] Loaded synthesis features: {data.synthesis_features.shape}")

    # T3: Topology
    t3_path = checkpoint_dir / "T3_topology.npz"
    if t3_path.exists():
        data.t3_topology = dict(np.load(t3_path, allow_pickle=True))
        print(f"[Waterfall] Loaded T3 topology: {list(data.t3_topology.keys())}")

    # T4: Viz coordinates
    t4_path = checkpoint_dir / "T4_viz_coords.npz"
    if t4_path.exists():
        data.t4_viz = dict(np.load(t4_path, allow_pickle=True))
        print(f"[Waterfall] Loaded T4 viz: {list(data.t4_viz.keys())}")

    # Load ground truth labels
    if ground_truth:
        label_set = sorted(set(ground_truth.values()))
        label_to_idx = {label: i for i, label in enumerate(label_set)}

        labels = []
        for i in range(data.n_articles):
            if i in ground_truth:
                labels.append(label_to_idx[ground_truth[i]])
            else:
                labels.append(-1)  # Unknown

        data.labels = np.array(labels)
        data.label_names = label_set
        print(f"[Waterfall] Loaded {len(label_set)} ground truth labels")

    validation_path = exp_dir / "validation.json"
    if validation_path.exists():
        try:
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            raw_track_nmi = payload.get("track_nmi", {})
            if isinstance(raw_track_nmi, dict):
                data.contract_track_nmi = {
                    str(k): float(v)
                    for k, v in raw_track_nmi.items()
                    if isinstance(v, (int, float))
                }
            raw_track_metrics = payload.get("track_metrics", {})
            if isinstance(raw_track_metrics, dict):
                data.contract_track_metrics = {
                    str(k): v
                    for k, v in raw_track_metrics.items()
                    if isinstance(v, dict)
                }
        except Exception as e:
            print(f"[Waterfall] Failed to load persisted track metrics: {e}")

    return data


def _override_with_contract_metrics(
    metrics: Dict[str, Any],
    data: WaterfallData,
    track_key: str,
) -> Dict[str, Any]:
    """
    Prefer persisted per-track NMI from validation.json when available so the
    dashboard and MONOLITH use the same ledger-backed values.
    """
    out = dict(metrics or {})
    contract = data.contract_track_metrics.get(track_key, {})
    if not isinstance(contract, dict):
        contract = {}
    contract_nmi = data.contract_track_nmi.get(track_key)
    if isinstance(contract_nmi, (int, float)):
        out["nmi"] = float(contract_nmi)
        if out["nmi"] > 0.5:
            out["signal_status"] = "GOOD"
            out["signal_color"] = PALETTE.good_signal
        elif out["nmi"] > 0.2:
            out["signal_status"] = "DEGRADED"
            out["signal_color"] = PALETTE.degraded_signal
        else:
            out["signal_status"] = "NOISE"
            out["signal_color"] = PALETTE.no_signal
    if isinstance(contract.get("ari"), (int, float)):
        out["ari"] = float(contract["ari"])
    if isinstance(contract.get("n_clusters"), int):
        out["n_clusters"] = int(contract["n_clusters"])
    if isinstance(contract.get("label_source"), str):
        out["label_source"] = contract["label_source"]
    return out


# =============================================================================
# PROJECTION COMPUTATION
# =============================================================================

def compute_nmi_on_raw_features(
    features: np.ndarray,
    labels: np.ndarray,
    name: str,
) -> Dict[str, float]:
    """
    Compute NMI directly on raw high-dimensional features (not just 2D projections).

    This is critical for debugging - NMI on raw features tells us if the signal
    is present BEFORE dimensionality reduction.
    """
    if not HAS_SKLEARN or features is None or labels is None:
        return {}

    from sklearn.metrics import adjusted_rand_score

    valid_mask = labels >= 0
    n_valid = valid_mask.sum()

    if n_valid < 10:
        return {"n_valid": int(n_valid), "status": "insufficient_labels"}

    # Use ground truth cluster count
    unique_labels = np.unique(labels[valid_mask])
    n_clusters = len(unique_labels)

    # Cluster on raw features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features)

    nmi = normalized_mutual_info_score(labels[valid_mask], pred_labels[valid_mask])
    ari = adjusted_rand_score(labels[valid_mask], pred_labels[valid_mask])

    print(f"[Waterfall] {name} RAW features NMI: {nmi:.4f}, ARI: {ari:.4f}")

    return {
        f"{name}_raw_nmi": float(nmi),
        f"{name}_raw_ari": float(ari),
        f"{name}_n_clusters": n_clusters,
        f"{name}_n_valid": int(n_valid),
    }


def compute_projections(data: WaterfallData, method: str = "pca") -> WaterfallData:
    """
    Compute 2D/3D projections for each checkpoint level.

    Args:
        data: WaterfallData with raw checkpoint arrays
        method: "pca", "tsne", or "umap"

    Returns:
        WaterfallData with cp1_2d, cp2_2d, cp3_2d, cp4_2d populated
    """
    if not HAS_SKLEARN:
        if method == "tsne":
            raise RuntimeError(
                "[Waterfall] --method tsne requested but scikit-learn is unavailable. "
                "Install scikit-learn or choose --method pca/umap."
            )
        print("[Waterfall] sklearn not available, skipping projections")
        return data

    # =========================================================================
    # BUILD ACCUMULATIVE FEATURE SETS
    # =========================================================================
    # CP-1: Logits only (Track 1 raw)
    # CP-2: Track 2 only (δ_μν base geometry)
    # CP-3: Track 2 + Track 1.5 (+ ∇Φ∇Φ observer shear)
    # CP-4: Track 2 + Track 1.5 + Track 3 (+ 1/ρ inverse density)

    # CP-1: Raw logits
    if data.t0_logits is not None:
        data.cp1_raw = data.t0_logits.reshape(data.n_articles, -1)  # [N, 24]

    # CP-2: Track 2 only (RKS features)
    if data.t2_kernels is not None:
        for key in ["z_rbf", "z_matern", "z_laplacian", "z_imq"]:
            if key in data.t2_kernels:
                data.cp2_raw = data.t2_kernels[key]  # [N, D]
                break

    # CP-3: Track 2 + Track 1.5 (concatenate)
    if data.cp2_raw is not None and data.t15_spectral is not None:
        probe_mags = data.t15_spectral.get("probe_magnitudes")
        if probe_mags is not None:
            # Concatenate: [N, D] + [N, 8] = [N, D+8]
            data.cp3_raw = np.concatenate([data.cp2_raw, probe_mags], axis=1)
            print(f"[Waterfall] CP3 = Track2 {data.cp2_raw.shape} + Track1.5 {probe_mags.shape} = {data.cp3_raw.shape}")

    # CP-4: Track 2 + Track 1.5 + Track 3 (full metric)
    if data.cp3_raw is not None and data.t3_blinker is not None:
        # Normalize blinker to [0, 1] range
        blinker = data.t3_blinker
        if blinker.ndim == 1:
            blinker = blinker.reshape(-1, 1)  # [N] -> [N, 1]
        elif blinker.ndim == 2 and blinker.shape[1] > 1:
            # If high-dim, reduce to mean
            blinker = blinker.mean(axis=1, keepdims=True)

        # Inverse density: 1/ρ ~ blinker variance (high variance = low density)
        # Scale to match feature magnitudes
        blinker_scaled = (blinker - blinker.min()) / (blinker.max() - blinker.min() + 1e-8)
        blinker_scaled = blinker_scaled * np.std(data.cp3_raw)

        data.cp4_raw = np.concatenate([data.cp3_raw, blinker_scaled], axis=1)
        print(f"[Waterfall] CP4 = CP3 {data.cp3_raw.shape} + Track3 {blinker_scaled.shape} = {data.cp4_raw.shape}")
    elif data.synthesis_features is not None:
        data.cp4_raw = data.synthesis_features
        print(f"[Waterfall] CP4 fallback = synthesis features {data.cp4_raw.shape}")

    # =========================================================================
    # COMPUTE NMI ON RAW ACCUMULATIVE FEATURES
    # =========================================================================
    print(f"\n[Waterfall] ═══════════════════════════════════════════════════════")
    print(f"[Waterfall] ACCUMULATIVE NMI (The Metric Tensor Build-Up)")
    print(f"[Waterfall] g_μν = (1/ρ)·δ_μν + ∇_μΦ∇_νΦ")
    print(f"[Waterfall] ═══════════════════════════════════════════════════════")

    if data.labels is not None:
        # CP-1: Track 1 only (raw logits)
        if data.cp1_raw is not None:
            compute_nmi_on_raw_features(data.cp1_raw, data.labels, "CP1_Track1_Logits")

        # CP-2: Track 2 only (base geometry δ_μν)
        if data.cp2_raw is not None:
            compute_nmi_on_raw_features(data.cp2_raw, data.labels, "CP2_Track2_BaseGeom")

        # CP-3: Track 2 + Track 1.5 (+ observer shear ∇Φ∇Φ)
        if data.cp3_raw is not None:
            compute_nmi_on_raw_features(data.cp3_raw, data.labels, "CP3_T2+T1.5_Shear")

        # CP-4: Full metric (+ Track 3 inverse density 1/ρ)
        if data.cp4_raw is not None:
            compute_nmi_on_raw_features(data.cp4_raw, data.labels, "CP4_FullMetric")

    print()

    # =========================================================================
    # 2D PROJECTIONS FOR VISUALIZATION
    # =========================================================================
    tsne_fallback_to_pca = False
    umap_fallback_to_pca = False
    if method == "tsne" and not HAS_TSNE:
        print(
            "[Waterfall] WARNING: --method tsne requested but TSNE dependency is unavailable; "
            "falling back to PCA explicitly."
        )
        tsne_fallback_to_pca = True
    if method == "umap" and not HAS_UMAP:
        print(
            "[Waterfall] WARNING: --method umap requested but umap dependency is unavailable; "
            "falling back to PCA explicitly."
        )
        umap_fallback_to_pca = True

    umap_mod = None

    def _project_2d(raw: np.ndarray) -> np.ndarray:
        nonlocal umap_mod, umap_fallback_to_pca
        if raw.shape[1] <= 2:
            return raw[:, :2]
        if method == "umap" and not umap_fallback_to_pca:
            try:
                if umap_mod is None:
                    import umap as _umap  # Lazy import to avoid heavy module import at test collection time.
                    umap_mod = _umap
                reducer = umap_mod.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(15, len(raw) - 1),
                )
                return reducer.fit_transform(raw)
            except Exception as e:
                print(f"[Waterfall] WARNING: UMAP projection unavailable ({e}); falling back to PCA.")
                umap_fallback_to_pca = True
        if method == "tsne" and not tsne_fallback_to_pca:
            n_samples = raw.shape[0]
            if n_samples < 2:
                return np.zeros((n_samples, 2), dtype=float)
            perplexity = max(1.0, min(30.0, float(n_samples - 1)))
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
            )
            return reducer.fit_transform(raw)
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(raw)

    # CP-1: Logits → 2D
    if data.cp1_raw is not None:
        if data.cp1_raw.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            data.cp1_2d = pca.fit_transform(data.cp1_raw)
            print(f"[Waterfall] CP1 projection: {data.cp1_2d.shape}, var explained: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            data.cp1_2d = data.cp1_raw[:, :2]

    # CP-2: Track 2 only (base geometry δ_μν) → 2D
    if data.cp2_raw is not None:
        data.cp2_2d = _project_2d(data.cp2_raw)
        print(f"[Waterfall] CP2 projection (Track 2 only): {data.cp2_2d.shape}")

    # CP-3: Track 2 + Track 1.5 (+ observer shear ∇Φ∇Φ) → 2D
    if data.cp3_raw is not None:
        data.cp3_2d = _project_2d(data.cp3_raw)
        print(f"[Waterfall] CP3 projection (Track 2 + 1.5): {data.cp3_2d.shape}")

    # CP-4: Full metric (Track 2 + 1.5 + 3) → 2D
    if data.cp4_raw is not None:
        data.cp4_2d = _project_2d(data.cp4_raw)
        print(f"[Waterfall] CP4 projection (Full Metric): {data.cp4_2d.shape}")
    elif data.t4_viz is not None and "x_umap" in data.t4_viz:
        # Fallback to pre-computed viz coords
        data.cp4_2d = data.t4_viz["x_umap"]
        print(f"[Waterfall] CP4 from T4 viz: {data.cp4_2d.shape}")

    return data


# =============================================================================
# SIGNAL QUALITY METRICS
# =============================================================================

def compute_signal_quality(
    coords_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_clusters_override: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute signal quality metrics for a 2D projection.

    IMPORTANT: Computes NMI separately for each checkpoint level to track
    where signal degrades through the pipeline.

    Returns:
        Dict with silhouette, cluster_separation, nmi, ari (if labels provided)
    """
    if not HAS_SKLEARN or coords_2d is None:
        return {"status": "unavailable"}

    metrics = {}
    n = len(coords_2d)

    # Determine number of clusters
    if n_clusters_override:
        n_clusters = n_clusters_override
    elif labels is not None:
        # Use ground truth cluster count
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels) if len(unique_labels) >= 2 else 4
    else:
        n_clusters = min(6, n // 3) if n >= 6 else 2

    # KMeans clustering on THIS checkpoint's features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(coords_2d)

    # Silhouette score (cluster quality without ground truth)
    if n > n_clusters:
        try:
            metrics["silhouette"] = float(silhouette_score(coords_2d, pred_labels))
        except:
            metrics["silhouette"] = 0.0

    # Cluster separation (inter-cluster distance / intra-cluster distance)
    cluster_centers = kmeans.cluster_centers_
    inter_dist = np.mean([
        np.linalg.norm(cluster_centers[i] - cluster_centers[j])
        for i in range(n_clusters)
        for j in range(i+1, n_clusters)
    ]) if n_clusters > 1 else 0.0

    intra_dist = np.mean([
        np.linalg.norm(coords_2d[pred_labels == k] - cluster_centers[k]).mean()
        for k in range(n_clusters)
        if (pred_labels == k).sum() > 0
    ])

    metrics["cluster_separation"] = inter_dist / (intra_dist + 1e-6)
    metrics["n_clusters"] = n_clusters

    # =========================================================================
    # NMI AND ARI - COMPUTED SEPARATELY FOR EACH CHECKPOINT
    # =========================================================================
    # This is the KEY metric - if NMI drops from CP-1 to CP-2, RKS is broken
    # If NMI drops from CP-2 to CP-3, Spectral is broken, etc.
    if labels is not None:
        valid_mask = labels >= 0
        n_valid = valid_mask.sum()
        metrics["n_valid_labels"] = int(n_valid)

        if n_valid >= 10:
            from sklearn.metrics import adjusted_rand_score

            # NMI: Normalized Mutual Information
            metrics["nmi"] = float(normalized_mutual_info_score(
                labels[valid_mask], pred_labels[valid_mask]
            ))

            # ARI: Adjusted Rand Index
            metrics["ari"] = float(adjusted_rand_score(
                labels[valid_mask], pred_labels[valid_mask]
            ))

            # Purity: fraction of correctly assigned samples
            from scipy.stats import mode
            purity = 0.0
            for k in range(n_clusters):
                cluster_mask = (pred_labels == k) & valid_mask
                if cluster_mask.sum() > 0:
                    most_common = mode(labels[cluster_mask], keepdims=False)[0]
                    purity += (labels[cluster_mask] == most_common).sum()
            metrics["purity"] = float(purity / n_valid)

    # Signal quality classification based on NMI (primary) or silhouette (fallback)
    nmi = metrics.get("nmi", None)
    silh = metrics.get("silhouette", 0.0)

    if nmi is not None:
        # NMI-based classification
        if nmi > 0.5:
            metrics["signal_status"] = "GOOD"
            metrics["signal_color"] = PALETTE.good_signal
        elif nmi > 0.2:
            metrics["signal_status"] = "DEGRADED"
            metrics["signal_color"] = PALETTE.degraded_signal
        else:
            metrics["signal_status"] = "NOISE"
            metrics["signal_color"] = PALETTE.no_signal
    else:
        # Silhouette-based fallback
        if silh > 0.5:
            metrics["signal_status"] = "GOOD"
            metrics["signal_color"] = PALETTE.good_signal
        elif silh > 0.2:
            metrics["signal_status"] = "DEGRADED"
            metrics["signal_color"] = PALETTE.degraded_signal
        else:
            metrics["signal_status"] = "NOISE"
            metrics["signal_color"] = PALETTE.no_signal

    return metrics


# =============================================================================
# VISUALIZATION GENERATION
# =============================================================================

def create_waterfall_dashboard(
    data: WaterfallData,
    output_path: Optional[Path] = None,
    title: str = "Waterfall Ablation Dashboard",
) -> Optional[go.Figure]:
    """
    Create 4-panel Plotly dashboard showing signal quality at each checkpoint.

    Args:
        data: WaterfallData with computed projections
        output_path: Optional path to save HTML
        title: Dashboard title

    Returns:
        Plotly Figure object
    """
    if not HAS_PLOTLY:
        print("[Waterfall] Plotly not available, cannot create dashboard")
        return None

    # Create 2x2 subplot grid
    # Accumulative metric view: g_μν = (1/ρ)·δ_μν + ∇_μΦ∇_νΦ
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "CP-1: Track 1 Only (Raw Logits)",
            "CP-2: Track 2 Only (δ_μν Base Geometry)",
            "CP-3: Track 2 + 1.5 (+ ∇Φ∇Φ Shear)",
            "CP-4: Full Metric (+ 1/ρ Density)",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Color mapping for labels
    if data.labels is not None:
        unique_labels = np.unique(data.labels[data.labels >= 0])
        color_map = {l: PALETTE.clusters[i % len(PALETTE.clusters)]
                     for i, l in enumerate(unique_labels)}
        colors = [color_map.get(l, PALETTE.text_dim) if l >= 0 else PALETTE.text_dim
                  for l in data.labels]
    else:
        colors = [PALETTE.cp1_primary] * data.n_articles

    # Hover text
    hover_texts = [f"Article {i}" for i in range(data.n_articles)]
    if data.label_names and data.labels is not None:
        hover_texts = [
            f"Article {i}<br>Label: {data.label_names[l] if l >= 0 else 'Unknown'}"
            for i, l in enumerate(data.labels)
        ]

    # =========================================================================
    # Panel 1: CP-1 Raw Logits
    # =========================================================================
    cp1_metrics = {"status": "NO DATA"}
    if data.cp1_2d is not None:
        cp1_metrics = compute_signal_quality(data.cp1_2d, data.labels)
        cp1_metrics = _override_with_contract_metrics(cp1_metrics, data, "T1")

        fig.add_trace(
            go.Scatter(
                x=data.cp1_2d[:, 0],
                y=data.cp1_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=1, color=PALETTE.void),
                ),
                text=hover_texts,
                hovertemplate="%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
                name="Logits",
            ),
            row=1, col=1
        )

        # Add signal quality annotation
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="x domain", yref="y domain",
            text=f"Signal: {cp1_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp1_metrics.get('silhouette', 0):.3f}<br>"
                 f"NMI: {cp1_metrics.get('nmi', 'N/A'):.3f}" if 'nmi' in cp1_metrics else
                 f"Signal: {cp1_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp1_metrics.get('silhouette', 0):.3f}",
            showarrow=False,
            font=dict(size=10, color=cp1_metrics.get('signal_color', PALETTE.text_dim)),
            bgcolor=PALETTE.void,
            bordercolor=cp1_metrics.get('signal_color', PALETTE.text_dim),
            borderwidth=1,
            row=1, col=1,
        )

    # =========================================================================
    # Panel 2: CP-2 RKS Features
    # =========================================================================
    cp2_metrics = {"status": "NO DATA"}
    if data.cp2_2d is not None:
        cp2_metrics = compute_signal_quality(data.cp2_2d, data.labels)
        cp2_metrics = _override_with_contract_metrics(cp2_metrics, data, "T2")

        fig.add_trace(
            go.Scatter(
                x=data.cp2_2d[:, 0],
                y=data.cp2_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=1, color=PALETTE.void),
                ),
                text=hover_texts,
                hovertemplate="%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
                name="RKS",
            ),
            row=1, col=2
        )

        fig.add_annotation(
            x=0.02, y=0.98,
            xref="x2 domain", yref="y2 domain",
            text=f"Signal: {cp2_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp2_metrics.get('silhouette', 0):.3f}<br>"
                 f"NMI: {cp2_metrics.get('nmi', 'N/A'):.3f}" if 'nmi' in cp2_metrics else
                 f"Signal: {cp2_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp2_metrics.get('silhouette', 0):.3f}",
            showarrow=False,
            font=dict(size=10, color=cp2_metrics.get('signal_color', PALETTE.text_dim)),
            bgcolor=PALETTE.void,
            bordercolor=cp2_metrics.get('signal_color', PALETTE.text_dim),
            borderwidth=1,
            row=1, col=2,
        )

    # =========================================================================
    # Panel 3: CP-3 Track 2 + Track 1.5 (Observer Shear)
    # =========================================================================
    cp3_metrics = {"status": "NO DATA"}
    if data.cp3_2d is not None:
        cp3_metrics = compute_signal_quality(data.cp3_2d, data.labels)
        cp3_metrics = _override_with_contract_metrics(cp3_metrics, data, "T1.5")

        fig.add_trace(
            go.Scatter(
                x=data.cp3_2d[:, 0],
                y=data.cp3_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=1, color=PALETTE.void),
                ),
                text=hover_texts,
                hovertemplate="%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
                name="T2+T1.5",
            ),
            row=2, col=1
        )

        fig.add_annotation(
            x=0.02, y=0.98,
            xref="x3 domain", yref="y3 domain",
            text=f"Signal: {cp3_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp3_metrics.get('silhouette', 0):.3f}<br>"
                 f"NMI: {cp3_metrics.get('nmi', 'N/A'):.3f}" if 'nmi' in cp3_metrics else
                 f"Signal: {cp3_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp3_metrics.get('silhouette', 0):.3f}",
            showarrow=False,
            font=dict(size=10, color=cp3_metrics.get('signal_color', PALETTE.text_dim)),
            bgcolor=PALETTE.void,
            bordercolor=cp3_metrics.get('signal_color', PALETTE.text_dim),
            borderwidth=1,
            row=2, col=1,
        )

    # =========================================================================
    # Panel 4: CP-4 Synthesis
    # =========================================================================
    cp4_metrics = {"status": "NO DATA"}
    if data.cp4_2d is not None:
        cp4_metrics = compute_signal_quality(data.cp4_2d, data.labels)
        cp4_metrics = _override_with_contract_metrics(cp4_metrics, data, "SYN")

        fig.add_trace(
            go.Scatter(
                x=data.cp4_2d[:, 0],
                y=data.cp4_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=1, color=PALETTE.void),
                ),
                text=hover_texts,
                hovertemplate="%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
                name="Synthesis",
            ),
            row=2, col=2
        )

        fig.add_annotation(
            x=0.02, y=0.98,
            xref="x4 domain", yref="y4 domain",
            text=f"Signal: {cp4_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp4_metrics.get('silhouette', 0):.3f}<br>"
                 f"NMI: {cp4_metrics.get('nmi', 'N/A'):.3f}" if 'nmi' in cp4_metrics else
                 f"Signal: {cp4_metrics.get('signal_status', 'N/A')}<br>"
                 f"Silh: {cp4_metrics.get('silhouette', 0):.3f}",
            showarrow=False,
            font=dict(size=10, color=cp4_metrics.get('signal_color', PALETTE.text_dim)),
            bgcolor=PALETTE.void,
            bordercolor=cp4_metrics.get('signal_color', PALETTE.text_dim),
            borderwidth=1,
            row=2, col=2,
        )

    # =========================================================================
    # Layout
    # =========================================================================
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Run: {data.run_id} | N={data.n_articles}</sub>",
            font=dict(size=18, color=PALETTE.text_primary),
            x=0.5,
        ),
        paper_bgcolor=PALETTE.void,
        plot_bgcolor=PALETTE.grid_dark,
        font=dict(color=PALETTE.text_primary),
        showlegend=False,
        height=900,
        width=1400,
    )

    # Update axes for all 4 2D plots
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                showgrid=True,
                gridcolor=PALETTE.grid_line,
                zeroline=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor=PALETTE.grid_line,
                zeroline=False,
                row=row,
                col=col,
            )

    # Save HTML
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f"[Waterfall] Dashboard saved: {output_path}")

    return fig


# =============================================================================
# DIAGNOSTIC REPORT
# =============================================================================

def generate_diagnostic_report(data: WaterfallData) -> str:
    """
    Generate human-readable diagnostic report identifying where signal degrades.

    CRITICAL: Shows NMI at EACH checkpoint level separately to identify
    exactly where the signal dies in the pipeline.

    Metric Tensor Accumulation:
    g_μν^total = (1/ρ)·δ_μν + ∇_μΦ∇_νΦ

    - CP-1: Track 1 only (raw logits)
    - CP-2: Track 2 only (δ_μν base geometry)
    - CP-3: Track 2 + Track 1.5 (+ ∇Φ∇Φ observer shear)
    - CP-4: Full metric (+ 1/ρ Track 3 inverse density)
    """
    lines = [
        "=" * 70,
        "WATERFALL ABLATION DIAGNOSTIC REPORT",
        "=" * 70,
        f"Run ID: {data.run_id}",
        f"N Articles: {data.n_articles}",
        f"Checkpoint Dir: {data.checkpoint_dir}",
        "",
        "METRIC TENSOR: g_μν = (1/ρ)·δ_μν + ∇_μΦ∇_νΦ",
        "",
        "-" * 70,
        "ACCUMULATIVE NMI (Track Build-Up)",
        "-" * 70,
        "",
        "  Checkpoint                       NMI      ARI      Silh     Status",
        "  " + "-" * 64,
    ]

    checkpoints = [
        ("CP-1: Track 1 (Logits)", data.cp1_2d),
        ("CP-2: Track 2 (δ_μν)", data.cp2_2d),
        ("CP-3: T2+T1.5 (+∇Φ∇Φ)", data.cp3_2d),
        ("CP-4: Full (+1/ρ)", data.cp4_2d),
    ]

    signal_history = []
    all_metrics = {}

    for name, coords in checkpoints:
        if coords is not None:
            metrics = compute_signal_quality(coords, data.labels)
            track_key = {
                "CP-1: Track 1 (Logits)": "T1",
                "CP-2: Track 2 (δ_μν)": "T2",
                "CP-3: T2+T1.5 (+∇Φ∇Φ)": "T1.5",
                "CP-4: Full (+1/ρ)": "SYN",
            }.get(name)
            if track_key is not None:
                metrics = _override_with_contract_metrics(metrics, data, track_key)
            status = metrics.get("signal_status", "N/A")
            silh = metrics.get("silhouette", 0.0)
            nmi = metrics.get("nmi", None)
            ari = metrics.get("ari", None)

            signal_history.append((name, status, silh, nmi, ari, metrics))
            all_metrics[name] = metrics

            nmi_str = f"{nmi:.4f}" if nmi is not None else "  N/A "
            ari_str = f"{ari:.4f}" if ari is not None else "  N/A "
            silh_str = f"{silh:.4f}"

            # Visual indicator
            if nmi is not None:
                if nmi > 0.5:
                    indicator = "✓"
                elif nmi > 0.2:
                    indicator = "⚡"
                else:
                    indicator = "✗"
            else:
                indicator = "?"

            lines.append(f"  {name:28s}  {nmi_str}   {ari_str}   {silh_str}   {status} {indicator}")
        else:
            lines.append(f"  {name:28s}    --       --       --      NO DATA")
            signal_history.append((name, "NO DATA", 0, None, None, {}))

    # NMI Delta Analysis
    lines.append("")
    lines.append("-" * 70)
    lines.append("NMI DELTA ANALYSIS (Where does signal die?)")
    lines.append("-" * 70)

    prev_nmi = None
    prev_name = None
    degradation_point = None
    biggest_drop = 0.0
    biggest_drop_location = None

    for name, status, silh, nmi, ari, metrics in signal_history:
        if nmi is not None and prev_nmi is not None:
            delta = nmi - prev_nmi
            delta_str = f"{delta:+.4f}"
            if delta < -0.1:
                lines.append(f"  {prev_name} → {name}: {delta_str}  ⚠️  SIGNAL DROP")
                if delta < biggest_drop:
                    biggest_drop = delta
                    biggest_drop_location = (prev_name, name)
            elif delta > 0.1:
                lines.append(f"  {prev_name} → {name}: {delta_str}  ✓ Signal improved")
            else:
                lines.append(f"  {prev_name} → {name}: {delta_str}  ~ Stable")
        prev_nmi = nmi
        prev_name = name

    # Identify degradation point
    lines.append("")
    lines.append("-" * 70)
    lines.append("DIAGNOSIS")
    lines.append("-" * 70)

    prev_status = None
    for name, status, silh, nmi, ari, metrics in signal_history:
        if prev_status in ["GOOD", "DEGRADED"] and status == "NOISE":
            degradation_point = name
            break
        prev_status = status

    if degradation_point:
        lines.append(f"  ⚠️  SIGNAL DEGRADATION DETECTED AT: {degradation_point}")
        lines.append("")
        lines.append("  Recommendation:")
        if "RKS" in degradation_point:
            lines.append("    - Check kernel bandwidth (sigma)")
            lines.append("    - Verify RKS dimension is sufficient")
            lines.append("    - Try different kernel type (IMQ, Matern)")
        elif "Spectral" in degradation_point:
            lines.append("    - Check EVR threshold (may be too strict)")
            lines.append("    - Verify probe pair gradients are meaningful")
            lines.append("    - Check gauge fixing reference probe")
        elif "Synthesis" in degradation_point:
            lines.append("    - Check Dirichlet alpha parameter")
            lines.append("    - Verify Track 3 variance isn't being baked into coordinates")
            lines.append("    - Check walker integration parameters")
    else:
        # Check overall signal
        final_status = signal_history[-1][1] if signal_history else "N/A"
        if final_status == "GOOD":
            lines.append("  ✓ Signal preserved through all checkpoints")
        elif final_status == "DEGRADED":
            lines.append("  ⚡ Signal degraded but still structured")
            lines.append("    - Consider tuning downstream parameters")
        elif final_status == "NOISE":
            lines.append("  ❌ Signal is noise from the start")
            lines.append("    - Check NLI model output (T0 logits)")
            lines.append("    - Verify input articles have distinguishable perspectives")
        else:
            lines.append("  ? Insufficient data for diagnosis")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_waterfall_analysis(
    checkpoint_dir: Path,
    output_dir: Optional[Path] = None,
    ground_truth: Optional[Dict[int, str]] = None,
    projection_method: str = "pca",
) -> Dict[str, Any]:
    """
    Run full waterfall ablation analysis on checkpoint data.

    Args:
        checkpoint_dir: Path to checkpoints/{run_id}/ directory
        output_dir: Where to save dashboard HTML and report
        ground_truth: Optional mapping of article index to label
        projection_method: "pca", "tsne", or "umap"

    Returns:
        Dict with metrics, report path, and dashboard path
    """
    checkpoint_dir = Path(checkpoint_dir)

    if output_dir is None:
        output_dir = checkpoint_dir / "waterfall_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("WATERFALL ABLATION ANALYSIS")
    print(f"{'='*60}")

    # Load checkpoints
    data = load_waterfall_checkpoints(checkpoint_dir, ground_truth)

    if data.n_articles == 0:
        print("[Waterfall] No checkpoint data found!")
        return {"status": "failed", "error": "No checkpoint data"}

    # Compute projections
    data = compute_projections(data, method=projection_method)

    # Generate dashboard
    dashboard_path = output_dir / "waterfall_dashboard.html"
    fig = create_waterfall_dashboard(
        data,
        output_path=dashboard_path,
        title=f"Waterfall Ablation: {checkpoint_dir.name}",
    )

    # Generate diagnostic report
    report = generate_diagnostic_report(data)
    print(report)

    report_path = output_dir / "waterfall_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Waterfall] Report saved: {report_path}")

    # Compute aggregate metrics
    metrics = {
        "run_id": data.run_id,
        "n_articles": data.n_articles,
        "checkpoints_available": {
            "T0_logits": data.t0_logits is not None,
            "T1_embeddings": data.t1_embeddings is not None,
            "T1.5_spectral": data.t15_spectral is not None,
            "T2_kernels": data.t2_kernels is not None,
            "T3_topology": data.t3_topology is not None,
            "T4_viz": data.t4_viz is not None,
        },
        "projections_computed": {
            "cp1_2d": data.cp1_2d is not None,
            "cp2_2d": data.cp2_2d is not None,
            "cp3_2d": data.cp3_2d is not None,
            "cp4_2d": data.cp4_2d is not None,
        },
        "track_nmi": data.contract_track_nmi,
        "track_metrics": data.contract_track_metrics,
    }

    # Save metrics
    metrics_path = output_dir / "waterfall_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "status": "success",
        "dashboard_path": str(dashboard_path),
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Waterfall Ablation Analysis - 4-Panel Forensic Dashboard"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to checkpoints/{run_id}/ directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for dashboard and report"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=Path,
        default=None,
        help="Path to ground truth JSON file (mapping index to label)"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="Projection method for 2D visualization"
    )

    args = parser.parse_args()

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)
            # Convert string keys to int if needed
            ground_truth = {int(k): v for k, v in ground_truth.items()}

    result = run_waterfall_analysis(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        ground_truth=ground_truth,
        projection_method=args.method,
    )

    print(f"\nResult: {result['status']}")
    if result['status'] == 'success':
        print(f"Dashboard: {result['dashboard_path']}")
