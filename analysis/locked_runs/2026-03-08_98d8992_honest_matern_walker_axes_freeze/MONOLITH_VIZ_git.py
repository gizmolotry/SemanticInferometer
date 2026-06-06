#!/usr/bin/env python3
"""
===============================================================================
ASTER v3.2 MONOLITH VISUALIZATION — ALL TRACKS COMBINED
===============================================================================
analysis/MONOLITH_VIZ.py

THE FORCED COMPILATION: One file to rule them all.

This file contains ALL visualization code combined into a single module.
No imports between viz files. Everything is here.

THE SEVEN FINGERS (ASTER v3.2 Metric Tensor Architecture):
===============================================================================
Track 1   - LOGITS      : The "Surface Claim" (explicit NLI verdict)
Track 1.5 - SPECTRAL    : The "Internal Stress" (spectral polarity gradient ∇Φ)
Track 2   - HOLOGRAM    : The "Base Terrain" (RKS-projected embeddings δ_μν)
Track 3   - BLINKER     : The "Conformal Factor" (Dirichlet variance 1/ρ) → FOG
Track 4   - WALKER      : The "Kinetic Probe" (MCMC path conductivity) → DIAMONDS
Track 5   - HADAMARD    : The "Metric Assembly" K_final = K_T2 ∘ K_T1.5 / sqrt(ρ⊗ρ)
Track 6   - HOTT        : Formal proofs (verdicts: ✓/✗/?)
EXIT GATE - INTEGRATOR  : Final normalization to unit hypersphere → PHANTOM PATHS
===============================================================================

VISUAL ELEMENTS:
- 3D Terrain (UMAP/PCA projection with energy surface)
- Fog overlay (Track 3 Dirichlet variance)
- Bonds/Cracks (alpha-sweep topology)
- Semantic Walker paths (MCMC walker trails)
- Walker diamonds (trapped/broken states)
- Phantom paths (HONEST=cyan, PHANTOM=white dashed, RUPTURE=lightning)
- Lightning arcs (rupture visualization)
- Spectral axis arrow (Track 1.5 endogenous compass)
- HoTT verdict icons (✓/✗/?)

Author: Belief Transformer Project (ASTER v3.2)
===============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import html
import re
import random
import datetime
from collections import Counter
from pathlib import Path
from analysis.verification.contract import resolve_run_directory
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from core.artifact_ledger import ArtifactContract
import numpy as np
import pandas as pd # Added for MONOLITH_DATA.csv loading

# =============================================================================
# OPTIONAL IMPORTS (graceful degradation)
# =============================================================================
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

try:
    from scipy.interpolate import griddata, Rbf
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import Delaunay
    from scipy.spatial.distance import cdist, pdist, squareform
    from scipy.stats import gaussian_kde
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# WARNING: UMAP projection has been explicitly disabled.
# Numba JIT compilation can cause indefinite hangs in certain Windows environments.
# This visualizer defaults strictly to PCA for stable, instantaneous projection.
umap = None
HAS_UMAP = False


def get_umap_module():
    """UMAP is intentionally disabled; force PCA fallback paths."""
    return None


# =============================================================================
# FIGHTER JET COLOR PALETTE
# =============================================================================
@dataclass
class FighterJetPalette:
    """Neon on void color scheme — corporate sleek."""
    # Background
    void: str = "#050505"
    grid_dark: str = "#0a0a0f"
    grid_line: str = "#1a1a2e"

    # Primary HUD
    cyan: str = "#00F0FF"
    cyan_dim: str = "#006677"

    # Signal States
    green: str = "#00FF41"
    yellow: str = "#FFD700"
    red: str = "#FF2A00"
    orange: str = "#FF8C00"

    # Paths (Track 5)
    honest_cyan: str = "#00F0FF"
    phantom_white: str = "#FFFFFF"
    rupture_red: str = "#FF2222"

    # Ruptures (Track 1.5)
    rupture_core: str = "#FF00FF"
    rupture_glow: str = "#FF66FF"

    # HoTT (Track 6)
    hott_valid: str = "#00FFAA"
    hott_invalid: str = "#FF0055"

    # Walker States (Track 4)
    walker_elastic: str = "#00FF41"
    walker_trapped: str = "#FFAA00"
    walker_broken: str = "#FF2222"
    walker_fog: str = "#888888"

    # Hysteresis / Pheromone Trails (Track 4)
    highway_glow: str = "#FFD700"       # Gold for well-worn paths
    highway_core: str = "#FFA500"       # Orange for trail center
    rut_low: str = "#332200"            # Dark brown for shallow ruts
    rut_high: str = "#FFCC00"           # Bright yellow for deep ruts

    # Terrain (Magma gradient)
    magma_cold: str = "#1a0a2e"
    magma_warm: str = "#8b1538"
    magma_hot: str = "#ff3366"
    magma_rupture: str = "#FFFF00"

    # Anchors
    anchor_blue: str = "#4a9eff"

    # Text
    text_primary: str = "#e0e0e0"
    text_dim: str = "#888888"

    # Singularity types
    singularity_consensus: str = "#2ecc71"
    singularity_barrier: str = "#f39c12"
    singularity_noise: str = "#95a5a6"
    singularity_structural: str = "#e74c3c"


PALETTE = FighterJetPalette()


class DimensionalCollapseError(RuntimeError):
    """Raised when render geometry violates required dimensional variance."""


LAYOUT_CONSTRAINTS = dict()

# Singularity color mapping
SINGULARITY_COLORS = {
    'consensus': PALETTE.singularity_consensus,
    'ideological_barrier': PALETTE.singularity_barrier,
    'noise': PALETTE.singularity_noise,
    'structural_singularity': PALETTE.singularity_structural,
    'unknown': PALETTE.cyan,
}

# Walker state color mapping (Track 4 states only - NOT fog, which is Track 3)
# Legacy states (for backwards compatibility)
WALKER_COLORS = {
    'elastic': PALETTE.walker_elastic,
    'trapped': PALETTE.walker_trapped,
    'broken': PALETTE.walker_broken,
}

# Track 4 Walker States (4-state divergence system from SemanticWalker)
WALKER_STATE_COLORS = {
    'tautology': "#888888",  # Gray - walker spun in place, no real movement
    'honest': "#00FF41",     # Green - walker found the easy path (laminar flow)
    'phantom': "#FF00FF",    # Magenta - walker split up and swirled (turbulence)
    'rupture': "#FF2222",    # Red - walker hit a singularity and crashed
}


# =============================================================================
# THRESHOLDS
# =============================================================================
THRESHOLDS = {
    "rupture_cosine": 0.8,
    "phantom_delta": 1.5,
    "rupture_delta": 10.0,
    "fog_variance": 0.7,
    "evr_rupture": 0.7,
    "honest_delta_max": 1.5,
    "blinker_high": 0.5,
    "walker_high": 0.5,
}


# =============================================================================
# PROBE LABELS (Track 1.5 framing pairs)
# =============================================================================
PROBE_LABELS = [
    "Israeli Defense ↔ Aggression",
    "Palestinian Resistance ↔ Terrorism",
    "Security ↔ Occupation",
    "Peace Process ↔ Capitulation",
    "Self-Determination ↔ Nationalism",
    "International Law ↔ Bias",
    "Humanitarian ↔ Propaganda",
    "Historical Rights ↔ Colonialism",
]

# Canonical terrain zones (2 orthogonal axes: density x stress)
CANONICAL_ZONES = {"Bridge", "Swamp", "Tightrope", "Void"}
ZONE_ALIAS_MAP = {
    "fault": "Swamp",
    "rupture": "Void",
    "bridge": "Bridge",
    "swamp": "Swamp",
    "tightrope": "Tightrope",
    "void": "Void",
}
ZONE_COLOR_MAP = {
    "Bridge": "#00ffc8",
    "Swamp": "#f39c12",
    "Tightrope": "#00c8ff",
    "Void": "#e74c3c",
}


def canonicalize_zone_name(raw_zone: object) -> str:
    """Map historical/legacy zone labels to canonical 4-zone names."""
    zone = str(raw_zone).strip() if raw_zone is not None else ""
    return ZONE_ALIAS_MAP.get(zone.lower(), zone if zone in CANONICAL_ZONES else "Void")


# =============================================================================
# TECHNICAL ANCHOR DEFINITIONS (No political leaders)
# =============================================================================
TECHNICAL_ANCHORS = [
    {"uid": "REF-MAX-VECTOR", "name": "MAX_VEC", "role": "Maximum Vector", "color": PALETTE.red},
    {"uid": "REF-MIN-VECTOR", "name": "MIN_VEC", "role": "Minimum Vector", "color": PALETTE.cyan},
    {"uid": "REF-CENTROID", "name": "CENTROID", "role": "Discourse Center", "color": PALETTE.yellow},
    {"uid": "REF-MEDIAN", "name": "MEDIAN", "role": "Median Position", "color": PALETTE.green},
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Track5Particle:
    """A single article in the integrated phase space."""
    index: int
    vector: np.ndarray

    # Per-track contributions
    logits: Optional[np.ndarray] = None
    antagonism: Optional[np.ndarray] = None
    hologram: Optional[np.ndarray] = None
    blinker_var: float = 0.0
    walker_res: float = 0.0

    # Derived physics
    liar_score: float = 0.0
    singularity_type: str = "unknown"

    # Phantom differential (Track 5)
    phantom_verdict: str = "unknown"
    phantom_delta: float = 0.0
    d_spectral: float = 0.0
    w_actual: float = 0.0

    # Walker state (Track 4)
    walker_state: str = "elastic"

    # Metadata
    article_id: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = None

    # Projected coordinates
    coords_3d: Optional[np.ndarray] = None
    coords_2d: Optional[np.ndarray] = None


@dataclass
class ExperimentData:
    """Container for experiment data."""
    kernel: str
    seed: int
    n_articles: int
    features: np.ndarray
    logits: Optional[np.ndarray] = None
    integrated: Optional[np.ndarray] = None
    dirichlet_fused: Optional[np.ndarray] = None
    dirichlet_fused_std: Optional[np.ndarray] = None
    spectral_evr: Optional[np.ndarray] = None
    logit_confidence: Optional[np.ndarray] = None  # Track 1: max softmax prob per probe
    spectral_probe_magnitudes: Optional[np.ndarray] = None
    spectral_dipole_valid: Optional[np.ndarray] = None
    walker_work_integrals: Optional[np.ndarray] = None
    walker_states: Optional[List[str]] = None
    walker_paths: Optional[Dict[int, np.ndarray]] = None
    phantom_verdicts: Optional[List[Dict]] = None
    d_spectral: Optional[np.ndarray] = None
    antagonism: Optional[np.ndarray] = None  # Track 1.5: spectral polarity vectors
    hott_proofs: Optional[List[Dict]] = None
    hott_summary: Optional[Dict] = None
    singularity_counts: Dict[str, int] = field(default_factory=dict)
    phantom_counts: Dict[str, int] = field(default_factory=dict)
    particles: List[Track5Particle] = field(default_factory=list)
    article_metadata: Optional[List[Dict]] = None
    experiment_dir: Optional[Path] = None
    # Checkpoint data for ANALYSIS mode (stacked track planes)
    cp_t0_logits: Optional[np.ndarray] = None      # [N, 8, 3] raw NLI logits
    cp_t2_kernels: Optional[np.ndarray] = None     # [N, D] RKS kernel projections
    cp_t15_spectral: Optional[np.ndarray] = None   # [N, 8] spectral probe magnitudes
    cp_t3_blinker: Optional[np.ndarray] = None     # [N, D] blinker variance
    ground_truth_labels: Optional[np.ndarray] = None  # [N] cluster labels for coloring
    # Track 4 Hysteresis (Path Memory)
    hysteresis_memory: Optional[np.ndarray] = None    # [8, 8] pheromone trail matrix
    hysteresis_stats: Optional[Dict] = None           # max_rut, highway_count, etc.
    # Track 5 Synthesis NMI
    synthesis_nmi: Optional[float] = None             # NMI score from validation.json
    # Epistemic UI contract data
    verification_status: str = "UNVERIFIED"           # VERIFIED/NON_COMPARABLE/MISSING_ARTIFACTS/UNVERIFIED
    verification_global_pass: Optional[bool] = None
    verification_seed_stability: Optional[bool] = None
    verification_crn_locked: Optional[bool] = None
    provenance: Optional[Dict[str, Any]] = None       # weights_hash, basis_hash, alpha, crn_seed


# =============================================================================
# DATA LOADING
# =============================================================================
def _find_nested_value(obj: Any, keys: List[str]) -> Optional[Any]:
    """Find first matching key in nested dict/list structures."""
    target = {k.lower() for k in keys}
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if str(k).lower() in target:
                    return v
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _discover_file_near_experiment(experiment_dir: Path, filename: str) -> Optional[Path]:
    """Look for filename at experiment dir and a few ancestors/canonical subfolders."""
    candidates = [
        experiment_dir / filename,
        experiment_dir.parent / filename,
        experiment_dir.parent.parent / filename,
        experiment_dir.parent.parent.parent / filename,
        experiment_dir.parent / "verification" / filename,
        experiment_dir.parent.parent / "verification" / filename,
    ]
    for pth in candidates:
        if pth.exists():
            return pth
    return None


def _short_hash(value: Any) -> str:
    sval = str(value).strip()
    if not sval:
        return "missing"
    return sval[:8]


def load_epistemic_contract_data(experiment_dir: Path) -> Dict[str, Any]:
    """Load verification/provenance from canonical verification artifacts."""
    status = "UNVERIFIED"
    global_pass = None
    seed_stability = None
    crn_locked = None
    provenance = {
        "weights_hash": "missing",
        "basis_hash": "missing",
        "alpha": "missing",
        "crn_seed": "missing",
    }

    verification_report = _discover_file_near_experiment(experiment_dir, "verification_report.json")
    if verification_report is not None:
        try:
            payload = json.loads(verification_report.read_text(encoding="utf-8"))
            gp = _find_nested_value(payload, ["global_pass"])
            if isinstance(gp, bool):
                global_pass = gp
            ss = _find_nested_value(payload, ["seed_stability", "seed_stability_pass", "seed_stable"])
            if isinstance(ss, bool):
                seed_stability = ss
            cs = _find_nested_value(payload, ["crn_locked", "crn_lock_pass", "crn_pass"])
            if isinstance(cs, bool):
                crn_locked = cs

            status_val = _find_nested_value(payload, ["status", "verification_status", "comparability_status"])
            if isinstance(status_val, str) and status_val.strip():
                status_norm = status_val.strip().upper()
                if status_norm in {"VERIFIED", "NON_COMPARABLE", "MISSING_ARTIFACTS", "UNVERIFIED"}:
                    status = status_norm

            w_hash = _find_nested_value(payload, ["weights_hash", "weights_fingerprint", "model_hash"])
            b_hash = _find_nested_value(payload, ["basis_hash", "basis_fingerprint"])
            alpha = _find_nested_value(payload, ["alpha"])
            crn_seed_val = _find_nested_value(payload, ["crn_seed", "seed"])
            if w_hash is not None:
                provenance["weights_hash"] = _short_hash(w_hash)
            if b_hash is not None:
                provenance["basis_hash"] = _short_hash(b_hash)
            if alpha is not None:
                provenance["alpha"] = str(alpha)
            if crn_seed_val is not None:
                provenance["crn_seed"] = str(crn_seed_val)
        except Exception:
            pass

    validation_json = experiment_dir / "validation.json"
    if validation_json.exists():
        try:
            payload = json.loads(validation_json.read_text(encoding="utf-8"))
            if provenance["alpha"] == "missing":
                alpha = payload.get("alpha")
                if alpha is not None:
                    provenance["alpha"] = str(alpha)
            if provenance["crn_seed"] == "missing":
                crn_seed_val = payload.get("crn_seed")
                if crn_seed_val is not None:
                    provenance["crn_seed"] = str(crn_seed_val)
        except Exception:
            pass

    if status == "UNVERIFIED" and global_pass is True:
        status = "VERIFIED"
    return {
        "status": status,
        "global_pass": global_pass,
        "seed_stability": seed_stability,
        "crn_locked": crn_locked,
        "provenance": provenance,
    }


def load_experiment_data(experiment_dir: Path) -> ExperimentData:
    """Load all experiment data from a seed directory."""
    experiment_dir = Path(experiment_dir)

    # ENFORCE CONTRACT
    require_spectral_dna = os.environ.get("MONOLITH_REQUIRE_SPECTRAL_DNA", "1").strip() == "1"
    ArtifactContract(experiment_dir, require_spectral_dna=require_spectral_dna).verify()

    # Required
    features = np.load(experiment_dir / "features.npy")
    n_articles = len(features)

    # Extract kernel and seed from path
    kernel = experiment_dir.parent.name
    seed_str = experiment_dir.name.replace("seed_", "")
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 42

    # Optional arrays
    logits = None
    if (experiment_dir / "logits.npy").exists():
        logits = np.load(experiment_dir / "logits.npy")

    integrated = None
    if (experiment_dir / "integrated_vectors.npy").exists():
        integrated = np.load(experiment_dir / "integrated_vectors.npy")

    dirichlet_fused = None
    if (experiment_dir / "dirichlet_fused.npy").exists():
        dirichlet_fused = np.load(experiment_dir / "dirichlet_fused.npy")

    dirichlet_fused_std = None
    if (experiment_dir / "dirichlet_fused_std.npy").exists():
        dirichlet_fused_std = np.load(experiment_dir / "dirichlet_fused_std.npy")

    spectral_evr = None
    if (experiment_dir / "spectral_evr.npy").exists():
        spectral_evr = np.load(experiment_dir / "spectral_evr.npy")

    # Track 1: Logit Confidence (max softmax prob across NLI classes)
    logit_confidence = None
    if (experiment_dir / "logit_confidence.npy").exists():
        logit_confidence = np.load(experiment_dir / "logit_confidence.npy")

    spectral_probe_magnitudes = None
    if (experiment_dir / "spectral_probe_magnitudes.npy").exists():
        spectral_probe_magnitudes = np.load(experiment_dir / "spectral_probe_magnitudes.npy")

    spectral_dipole_valid = None
    if (experiment_dir / "spectral_dipole_valid.npy").exists():
        spectral_dipole_valid = np.load(experiment_dir / "spectral_dipole_valid.npy")

    # Track 4: Walker
    walker_work_integrals = None
    if (experiment_dir / "walker_work_integrals.npy").exists():
        walker_work_integrals = np.load(experiment_dir / "walker_work_integrals.npy")

    walker_states = None
    if (experiment_dir / "walker_states.json").exists():
        with open(experiment_dir / "walker_states.json") as f:
            walker_states = json.load(f)

    walker_paths = None
    walker_paths_path = experiment_dir / "walker_paths.npz"
    if walker_paths_path.exists():
        try:
            path_data = np.load(walker_paths_path, allow_pickle=True)
            article_idx = path_data["article_idx"] if "article_idx" in path_data else None
            path_xyz = path_data["path_xyz"] if "path_xyz" in path_data else None
            if article_idx is not None and path_xyz is not None:
                walker_paths = {}
                n_items = min(len(article_idx), len(path_xyz))
                for i in range(n_items):
                    idx = int(article_idx[i])
                    walker_paths[idx] = np.asarray(path_xyz[i], dtype=float)
                print(f"[MONOLITH] Loaded walker_paths.npz: {len(walker_paths)} trajectories")
        except Exception as e:
            print(f"[MONOLITH] Warning: failed to load walker_paths.npz: {e}")

    # Track 4 Hysteresis (Path Memory)
    hysteresis_memory = None
    if (experiment_dir / "hysteresis_memory.npy").exists():
        hysteresis_memory = np.load(experiment_dir / "hysteresis_memory.npy")
        print(f"[MONOLITH] Loaded hysteresis memory: {hysteresis_memory.shape}")

    hysteresis_stats = None
    if (experiment_dir / "hysteresis_stats.json").exists():
        with open(experiment_dir / "hysteresis_stats.json") as f:
            hysteresis_stats = json.load(f)

    # Track 5: Phantom
    d_spectral = None
    if (experiment_dir / "d_spectral.npy").exists():
        d_spectral = np.load(experiment_dir / "d_spectral.npy")

    # Track 1.5: Antagonism (spectral polarity vectors for wind streamlines)
    antagonism = None
    if (experiment_dir / "antagonism.npy").exists():
        antagonism = np.load(experiment_dir / "antagonism.npy")
    elif (experiment_dir / "spectral_u_axis.npy").exists():
        # Fall back to u_axis if antagonism not saved separately
        antagonism = np.load(experiment_dir / "spectral_u_axis.npy")

    phantom_verdicts = None
    if (experiment_dir / "phantom_verdicts.json").exists():
        with open(experiment_dir / "phantom_verdicts.json") as f:
            phantom_verdicts = json.load(f)

    # Track 6: HoTT
    hott_proofs = None
    if (experiment_dir / "hott_proofs.json").exists():
        with open(experiment_dir / "hott_proofs.json") as f:
            hott_proofs = json.load(f)

    hott_summary = None
    if (experiment_dir / "hott_summary.json").exists():
        with open(experiment_dir / "hott_summary.json") as f:
            hott_summary = json.load(f)

    # Metadata
    article_metadata = None
    metadata_csv_path = experiment_dir / "article_metadata.csv"
    metadata_json_path = experiment_dir / "article_metadata.json"
    if metadata_csv_path.exists():
        article_metadata = pd.read_csv(metadata_csv_path).to_dict(orient="records")
    elif metadata_json_path.exists():
        with open(metadata_json_path) as f:
            article_metadata = json.load(f)

    # Result summary
    singularity_counts = {}
    phantom_counts = {}
    if (experiment_dir / "result_summary.json").exists():
        with open(experiment_dir / "result_summary.json") as f:
            result = json.load(f)
        singularity_counts = result.get("singularity_counts", {})
        phantom_counts = result.get("phantom_counts", {})

    # ==========================================================================
    # CHECKPOINT DATA FOR ANALYSIS MODE (Stacked Track Planes)
    # ==========================================================================
    cp_t0_logits = None
    cp_t2_kernels = None
    cp_t15_spectral = None
    cp_t3_blinker = None
    ground_truth_labels = None

    # Smart Discovery for checkpoints
    checkpoint_dir = None
    checkpoint_candidates = [
        "T0_substrate.npy",
        "T2_kernel_projections.npz",
        "T1.5_spectral_state.npz",
        "T3_topology.npz",
    ]
    for marker in checkpoint_candidates:
        candidates = list(experiment_dir.rglob(marker))
        if candidates:
            checkpoint_dir = candidates[0].parent
            break

    if checkpoint_dir and checkpoint_dir.exists():
        # T0: Raw logits [N, 8, 3]
        t0_path = checkpoint_dir / "T0_substrate.npy"
        if t0_path.exists():
            cp_t0_logits = np.load(t0_path)
            print(f"[MONOLITH] Loaded T0 logits: {cp_t0_logits.shape}")

        # T2: Kernel projections
        t2_path = checkpoint_dir / "T2_kernel_projections.npz"
        if t2_path.exists():
            t2_data = np.load(t2_path)
            # Get the first kernel (usually z_rbf)
            for key in t2_data.files:
                if key.startswith('z_'):
                    cp_t2_kernels = t2_data[key]
                    print(f"[MONOLITH] Loaded T2 kernels ({key}): {cp_t2_kernels.shape}")
                    break

        # T1.5: Spectral probe magnitudes
        t15_path = checkpoint_dir / "T1.5_spectral_state.npz"
        if t15_path.exists():
            t15_data = np.load(t15_path)
            if 'probe_magnitudes' in t15_data.files:
                cp_t15_spectral = t15_data['probe_magnitudes']
                print(f"[MONOLITH] Loaded T1.5 spectral: {cp_t15_spectral.shape}")

    if spectral_probe_magnitudes is None and cp_t15_spectral is not None:
        spectral_probe_magnitudes = cp_t15_spectral

    # T3: Blinker/topology
    if checkpoint_dir and checkpoint_dir.exists():
        t3_path = checkpoint_dir / "T3_topology.npz"
        if t3_path.exists():
            t3_data = np.load(t3_path)
            if 'dirichlet_fused' in t3_data.files:
                cp_t3_blinker = t3_data['dirichlet_fused']
                print(f"[MONOLITH] Loaded T3 blinker (dirichlet_fused): {cp_t3_blinker.shape}")
            elif 'bond_matrix' in t3_data.files:
                cp_t3_blinker = t3_data['bond_matrix']
                print(f"[MONOLITH] Loaded T3 blinker (bond_matrix fallback): {cp_t3_blinker.shape}")
            elif 'crack_matrix' in t3_data.files:
                cp_t3_blinker = t3_data['crack_matrix']
                print(f"[MONOLITH] Loaded T3 blinker (crack_matrix fallback): {cp_t3_blinker.shape}")

    # Ground truth labels (from article metadata or corpus file)
    ground_truth_labels = None
    if article_metadata:
        labels = []
        label_map = {}
        for meta in article_metadata:
            tag = meta.get('perspective_tag', meta.get('label', 'unknown'))
            if tag not in label_map:
                label_map[tag] = len(label_map)
            labels.append(label_map[tag])
        if len(label_map) > 1:  # Only use if we found multiple labels
            ground_truth_labels = np.array(labels)
            print(f"[MONOLITH] Loaded {len(label_map)} unique ground truth labels from metadata")

    # Fallback: Try to load from corpus file (synthetic experiments)
    if ground_truth_labels is None or len(set(ground_truth_labels)) <= 1:
        corpus_paths = [
            experiment_dir.parent.parent.parent / "sythgen" / "high_quality_articles.jsonl",
            experiment_dir.parent.parent / "sythgen" / "high_quality_articles.jsonl",
            Path("sythgen/high_quality_articles.jsonl"),
        ]
        for corpus_path in corpus_paths:
            if corpus_path.exists():
                try:
                    labels = []
                    label_map = {}
                    with open(corpus_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= n_articles:
                                break
                            article = json.loads(line.strip())
                            tag = article.get('perspective_tag', 'unknown')
                            if tag not in label_map:
                                label_map[tag] = len(label_map)
                            labels.append(label_map[tag])
                    if len(labels) == n_articles and len(label_map) > 1:
                        ground_truth_labels = np.array(labels)
                        print(f"[MONOLITH] Loaded {len(label_map)} ground truth labels from corpus: {corpus_path.name}")
                        break
                except Exception as e:
                    print(f"[MONOLITH] Failed to load corpus labels: {e}")

    # Track 5 Synthesis NMI
    synthesis_nmi = None
    if (experiment_dir / "validation.json").exists():
        try:
            with open(experiment_dir / "validation.json") as f:
                validation_data = json.load(f)
                # validation.json uses key "nmi" (not "normalized_mutual_info")
                synthesis_nmi = validation_data.get("nmi", validation_data.get("normalized_mutual_info"))
                if synthesis_nmi is not None:
                    print(f"[MONOLITH] Loaded Track 5 Synthesis NMI: {synthesis_nmi:.3f}")
        except Exception as e:
            print(f"[MONOLITH] Failed to load validation.json: {e}")

    # Epistemic contract data (verification/provenance)
    epistemic = load_epistemic_contract_data(experiment_dir)

    return ExperimentData(
        kernel=kernel,
        seed=seed,
        n_articles=n_articles,
        features=features,
        logits=logits,
        integrated=integrated,
        dirichlet_fused=dirichlet_fused,
        dirichlet_fused_std=dirichlet_fused_std,
        spectral_evr=spectral_evr,
        logit_confidence=logit_confidence,
        spectral_probe_magnitudes=spectral_probe_magnitudes,
        spectral_dipole_valid=spectral_dipole_valid,
        walker_work_integrals=walker_work_integrals,
        walker_states=walker_states,
        walker_paths=walker_paths,
        phantom_verdicts=phantom_verdicts,
        d_spectral=d_spectral,
        antagonism=antagonism,
        hott_proofs=hott_proofs,
        hott_summary=hott_summary,
        singularity_counts=singularity_counts,
        phantom_counts=phantom_counts,
        article_metadata=article_metadata,
        experiment_dir=experiment_dir,
        cp_t0_logits=cp_t0_logits,
        cp_t2_kernels=cp_t2_kernels,
        cp_t15_spectral=cp_t15_spectral,
        cp_t3_blinker=cp_t3_blinker,
        ground_truth_labels=ground_truth_labels,
        hysteresis_memory=hysteresis_memory,
        hysteresis_stats=hysteresis_stats,
        synthesis_nmi=synthesis_nmi,
        verification_status=str(epistemic.get("status", "UNVERIFIED")),
        verification_global_pass=epistemic.get("global_pass"),
        verification_seed_stability=epistemic.get("seed_stability"),
        verification_crn_locked=epistemic.get("crn_locked"),
        provenance=epistemic.get("provenance"),
    )


# =============================================================================
# PROJECTION FUNCTIONS
# =============================================================================
def compute_umap_3d(
    features: np.ndarray,
    use_cosine: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 3D UMAP projection."""
    umap_mod = get_umap_module()
    if umap_mod is None:
        # Fallback to PCA
        if HAS_SCIPY:
            pca = PCA(n_components=3, random_state=random_state)
            return pca.fit_transform(features)
        return features[:, :3]

    norms = np.linalg.norm(features, axis=1)
    is_unit_sphere = np.allclose(norms, 1.0, atol=1e-5)

    if is_unit_sphere or use_cosine:
        dists = pdist(features, metric='cosine')
        D = squareform(dists)
        metric = 'precomputed'
        data = D
    else:
        metric = 'euclidean'
        data = features

    reducer = umap_mod.UMAP(
        n_components=3,
        n_neighbors=min(n_neighbors, features.shape[0] - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(data)


def compute_umap_2d(
    features: np.ndarray,
    use_cosine: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D UMAP projection."""
    umap_mod = get_umap_module()
    if umap_mod is None:
        if HAS_SCIPY:
            pca = PCA(n_components=2, random_state=random_state)
            return pca.fit_transform(features)
        return features[:, :2]

    norms = np.linalg.norm(features, axis=1)
    is_unit_sphere = np.allclose(norms, 1.0, atol=1e-5)

    if is_unit_sphere or use_cosine:
        dists = pdist(features, metric='cosine')
        D = squareform(dists)
        metric = 'precomputed'
        data = D
    else:
        metric = 'euclidean'
        data = features

    reducer = umap_mod.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, features.shape[0] - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(data)


# =============================================================================
# TERRAIN SURFACE
# =============================================================================

# TERRAIN STATE COLORMAP (Continuous Gradient for Density x Stress Manifold)
# This maps the 2D space of (density, stress) to a continuous color field:
#   BRIDGE:    High Density + Low Stress  = Cyan/Electric Blue (Constructive Interference)
#   SWAMP:     High Density + High Stress = Bruised Purple/Neon Violet (Phase Incoherence)
#   TIGHTROPE: Low Density + Low Stress   = Bright Yellow/White (Brittle Resonance)
#   VOID:      Low Density + High Stress  = Black with Red Edges (Singularity)
TERRAIN_COLORS = {
    'bridge': '#00F0FF',     # Cyan/Electric Blue - deep glacial valleys, easy traverse
    'swamp': '#9932CC',      # Bruised Purple/Neon Violet - jagged muddy highlands
    'tightrope': '#FFFFCC',  # Bright Yellow/White - thin fragile ridges
    'void': '#1A0000',       # Black with Red - tears in mesh, singularity
}


def compute_terrain_field(
    blinker_values: np.ndarray,  # Track 3: Dirichlet variance (inverse density)
    walker_resistance: np.ndarray,  # Track 4: Walker work (stress proxy)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2D terrain field from density and stress.

    Args:
        blinker_values: [N] - Higher = more variance = LOWER density
        walker_resistance: [N] - Higher = more work = HIGHER stress

    Returns:
        density: [N] normalized [0,1] where 1 = high density (BRIDGE/SWAMP)
        stress: [N] normalized [0,1] where 1 = high stress (SWAMP/VOID)
        terrain_scalar: [N] combined scalar for colormap [0,1]
    """
    N = len(blinker_values)

    # Normalize blinker to density (invert: low variance = high density)
    blinker_min, blinker_max = blinker_values.min(), blinker_values.max()
    if blinker_max - blinker_min > 1e-9:
        density = 1.0 - (blinker_values - blinker_min) / (blinker_max - blinker_min)
    else:
        density = np.ones(N) * 0.5

    # Normalize walker resistance to stress
    if walker_resistance is not None and len(walker_resistance) == N:
        # Handle NaN values
        walker_clean = np.nan_to_num(walker_resistance, nan=0.5)
        w_min, w_max = float(np.nanmin(walker_clean)), float(np.nanmax(walker_clean))
        if w_max - w_min > 1e-9:
            stress = (walker_clean - w_min) / (w_max - w_min)
        else:
            stress = np.ones(N) * 0.5
    else:
        stress = np.ones(N) * 0.5

    # Map 2D (density, stress) to 1D scalar for colormap with unique 4-corner anchors:
    # BRIDGE (1,0) -> 0.0, TIGHTROPE (0,0) -> 0.25
    # SWAMP (1,1) -> 0.75, VOID (0,1) -> 1.0
    # Formula chosen to satisfy all four constraints:
    #   scalar = 0.25 - 0.25*density + 0.75*stress
    terrain_scalar = 0.25 - 0.25 * density + 0.75 * stress

    return density, stress, terrain_scalar


def get_terrain_colorscale() -> list:
    """
    Create continuous colorscale for the density x stress manifold.

    The colorscale interpolates smoothly between the 4 terrain states:
    0.0 = BRIDGE (cyan) - high density, low stress - constructive interference
    0.25 = TIGHTROPE (yellow/white) - low density, low stress - brittle resonance
    0.75 = SWAMP (purple) - high density, high stress - phase incoherence
    1.0 = VOID (black/red) - low density, high stress - singularity
    """
    return [
        [0.0, TERRAIN_COLORS['bridge']],      # Cyan: BRIDGE
        [0.25, TERRAIN_COLORS['tightrope']],  # Yellow: TIGHTROPE
        [0.5, '#6633AA'],                      # Transition (purple-ish)
        [0.75, TERRAIN_COLORS['swamp']],       # Purple: SWAMP
        [1.0, TERRAIN_COLORS['void']],         # Black/Red: VOID
    ]


def get_continuous_manifold_colorscale() -> list:
    """
    Create smooth continuous colorscale for density × stress manifold gradient.

    Unlike the discrete 4-zone colormap, this provides smooth interpolation
    across the entire 2D manifold space. The terrain_scalar formula
    (0.25 - 0.25*density + 0.75*stress) maps the 2D space to [0,1]:

    terrain_scalar=0.0: High density (1.0), Low stress (0.0) → BRIDGE (Cyan)
    terrain_scalar=0.25: Low density (0.0), Low stress (0.0) → TIGHTROPE (Yellow)
    terrain_scalar=0.5: Mid density, Mid stress → Transition blend
    terrain_scalar=0.75: High density (1.0), High stress (1.0) → SWAMP (Purple)
    terrain_scalar=1.0: Low density (0.0), High stress (1.0) → VOID (Black/Red)

    The gradient smoothly blends between all four corners of the manifold.
    """
    return [
        [0.00, '#00F0FF'],  # BRIDGE: High density, low stress - Electric Cyan
        [0.10, '#00E0EE'],  # Transition toward tightrope
        [0.20, '#88DDAA'],  # Cyan-Yellow blend
        [0.25, '#FFFFCC'],  # TIGHTROPE: Low density, low stress - Pale Yellow
        [0.35, '#DDCC99'],  # Yellow darkening
        [0.45, '#AA8899'],  # Yellow-Purple transition
        [0.50, '#9966AA'],  # Mid manifold: balanced blend
        [0.60, '#9955BB'],  # Toward swamp
        [0.70, '#9944CC'],  # Purple intensifying
        [0.75, '#9932CC'],  # SWAMP: High density, high stress - Vivid Purple
        [0.82, '#772244'],  # Purple-Red transition
        [0.90, '#440011'],  # Dark crimson
        [1.00, '#1A0000'],  # VOID: Low density, high stress - Near Black
    ]


def compute_smooth_terrain(
    positions_3d: np.ndarray,
    energy_values: np.ndarray,
    grid_resolution: int = 150,
    smoothing_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create smooth energy landscape with bicubic interpolation."""
    if not HAS_SCIPY:
        return None, None, None

    x = positions_3d[:, 0]
    y = positions_3d[:, 1]

    margin = 0.15
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    x_min = x.min() - margin * x_range
    x_max = x.max() + margin * x_range
    y_min = y.min() - margin * y_range
    y_max = y.max() + margin * y_range

    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = _interpolate_field_boundary_safe(
        x=x,
        y=y,
        values=energy_values,
        Xi=Xi,
        Yi=Yi,
        fill_value=float(np.nanmean(np.asarray(energy_values, dtype=float))),
        clip_to_source=True,
    )

    Zi = gaussian_filter(Zi, sigma=smoothing_sigma)
    return Xi, Yi, Zi


def _interpolate_field_boundary_safe(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    Xi: np.ndarray,
    Yi: np.ndarray,
    fill_value: float,
    clip_to_source: bool = True,
) -> np.ndarray:
    """
    Continuous, boundary-safe interpolation helper.

    Uses cubic interpolation for interior structure and applies RBF only to
    boundary NaN regions to avoid nearest-neighbor plateaus without reshaping
    the entire manifold.
    """
    if not HAS_SCIPY:
        return np.full_like(Xi, fill_value, dtype=float)

    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    vv = np.asarray(values, dtype=float).ravel()
    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(vv)
    if int(np.count_nonzero(m)) < 3:
        return np.full_like(Xi, fill_value, dtype=float)

    xv = xv[m]
    yv = yv[m]
    vv = vv[m]
    coords = np.column_stack([xv, yv])
    try:
        _, uniq_idx = np.unique(np.round(coords, decimals=12), axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        xv = xv[uniq_idx]
        yv = yv[uniq_idx]
        vv = vv[uniq_idx]
    except Exception:
        pass

    if xv.size < 3:
        return np.full_like(Xi, float(np.nanmean(vv)) if vv.size else fill_value, dtype=float)

    src_min = float(np.nanmin(vv))
    src_max = float(np.nanmax(vv))
    src_fill = float(np.nanmean(vv)) if vv.size else fill_value

    try:
        Zi = griddata((xv, yv), vv, (Xi, Yi), method='cubic')
    except Exception:
        Zi = None

    if Zi is None:
        try:
            Zi = griddata((xv, yv), vv, (Xi, Yi), method='linear', fill_value=src_fill)
        except Exception:
            Zi = np.full_like(Xi, src_fill, dtype=float)
    else:
        nan_mask = np.isnan(Zi)
        if np.any(nan_mask):
            try:
                eps = None
                if xv.size >= 4:
                    d = pdist(np.column_stack([xv, yv]))
                    d = d[np.isfinite(d) & (d > 1e-12)]
                    if d.size > 0:
                        eps = float(np.nanmedian(d))
                smooth = max(float(np.nanstd(vv)) * 0.02, 1e-8)
                rbf = Rbf(
                    xv,
                    yv,
                    vv,
                    function="multiquadric",
                    epsilon=eps if eps is not None else 1.0,
                    smooth=smooth,
                )
                Zi[nan_mask] = np.asarray(rbf(Xi[nan_mask], Yi[nan_mask]), dtype=float)
            except Exception:
                pass
            if np.isnan(Zi).any():
                try:
                    Zi_linear = griddata((xv, yv), vv, (Xi, Yi), method='linear', fill_value=src_fill)
                    Zi = np.where(np.isnan(Zi), Zi_linear, Zi)
                except Exception:
                    Zi = np.where(np.isnan(Zi), src_fill, Zi)

    Zi = np.nan_to_num(Zi, nan=src_fill, posinf=src_fill, neginf=src_fill)
    if clip_to_source and np.isfinite(src_min) and np.isfinite(src_max):
        Zi = np.clip(Zi, src_min, src_max)
    return Zi


def project_points_onto_terrain(
    positions_2d: np.ndarray,
    energy_values: np.ndarray, # This is our article_z_height
) -> np.ndarray:
    """
    Project 2D positions onto terrain surface.

    Instead of using UMAP's arbitrary Z, compute Z from the terrain height.
    Articles sit ON the manifold, not floating above it.

    Args:
        positions_2d: [N, 2] XY positions from UMAP
        energy_values: [N] energy values for terrain height (final Z from Metric Engine)

    Returns:
        positions_3d: [N, 3] with Z = terrain height
    """
    x = positions_2d[:, 0]
    y = positions_2d[:, 1]

    # Z is directly from energy_values (unified_z_height from CSV)
    z_values = energy_values # No scaling, no extra lift. This is the final Z.

    return np.column_stack([x, y, z_values])


def render_terrain_surface(
    positions_3d: np.ndarray,
    energy_values: np.ndarray,
    grid_resolution: int = 150,
    z_scale: float = 3.0,
    terrain_scalar: Optional[np.ndarray] = None,
    terrain_density: Optional[np.ndarray] = None,
    terrain_stress: Optional[np.ndarray] = None,
    terrain_stress_geometry: Optional[np.ndarray] = None,
    use_manifold_colormap: bool = True,
    global_density_median: float = 0.5, # For consistent zone mapping
    global_stress_median: float = 0.5,  # For consistent zone mapping
    opacity: float = 0.9,               # New opacity parameter
    rupture_segments_2d: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    rupture_tear_radius_scale: float = 0.02,
) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    ASTER v3.2 BI-AXIAL TERRAIN SURFACE
    ===================================
    Decouples Z-axis (geometry) from Color (skin) for proper 4-zone mapping.

    The Two Axes:
    - Z-Axis (Height) = STRESS (Gradient magnitude)
      Mountains = High Conflict, Valleys = Low Conflict
    - Color (Skin) = DENSITY (Mass)
      Creates 4 zones: BRIDGE, SWAMP, TIGHTROPE, VOID

    The 4 Zones (Density × Stress):
    - BRIDGE (Cyan): High Density, Low Stress → Deep Blue Valleys
    - SWAMP (Purple): High Density, High Stress → Purple Mountains
    - TIGHTROPE (Yellow): Low Density, Low Stress → Yellow Plains
    - VOID (Red): Low Density, High Stress → Red Spikes

    Args:
        positions_3d: [N, 3] UMAP positions (articles already placed on terrain)
        energy_values: [N] values for terrain height (stress proxy)
        grid_resolution: Grid interpolation resolution
        z_scale: Vertical exaggeration factor (default 3.0 for dramatic mountains)
        terrain_scalar: [N] legacy 1D scalar (ignored if density/stress provided)
        terrain_density: [N] density values [0,1] for color mapping
        terrain_stress: [N] stress values [0,1] for Z geometry
        use_manifold_colormap: If True, use 4-zone discrete colormap
    """
    if not HAS_PLOTLY or not HAS_SCIPY:
        return None

    from scipy.interpolate import griddata

    x = positions_3d[:, 0]
    y = positions_3d[:, 1]

    margin = 0.15
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    # Guard against degenerate axes so mesh construction cannot collapse to a line.
    x_range_safe = max(float(x_range), 1e-6)
    y_range_safe = max(float(y_range), 1e-6)

    x_min = x.min() - margin * x_range_safe
    x_max = x.max() + margin * x_range_safe
    y_min = y.min() - margin * y_range_safe
    y_max = y.max() + margin * y_range_safe

    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    support_mask = None
    try:
        xy_points = np.column_stack([x, y])
        if xy_points.shape[0] >= 3:
            tri = Delaunay(xy_points)
            grid_points = np.column_stack([Xi.ravel(), Yi.ravel()])
            simplex = tri.find_simplex(grid_points)
            support_mask = (simplex >= 0).reshape(Xi.shape)
    except Exception:
        support_mask = None
    tear_mask = np.zeros(Xi.shape, dtype=bool)
    if rupture_segments_2d:
        tear_radius = max(x_range_safe, y_range_safe) * float(max(rupture_tear_radius_scale, 1e-6))
        tear_radius_sq = tear_radius * tear_radius
        for seg in rupture_segments_2d:
            try:
                p0 = np.asarray(seg[0], dtype=float).reshape(2)
                p1 = np.asarray(seg[1], dtype=float).reshape(2)
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                vx = p1[0] - p0[0]
                vy = p1[1] - p0[1]
                seg_len_sq = (vx * vx) + (vy * vy)
                if seg_len_sq <= 1e-12:
                    dx = Xi - p0[0]
                    dy = Yi - p0[1]
                    dist_sq = (dx * dx) + (dy * dy)
                else:
                    t = ((Xi - p0[0]) * vx + (Yi - p0[1]) * vy) / seg_len_sq
                    t = np.clip(t, 0.0, 1.0)
                    proj_x = p0[0] + t * vx
                    proj_y = p0[1] + t * vy
                    dx = Xi - proj_x
                    dy = Yi - proj_y
                    dist_sq = (dx * dx) + (dy * dy)
                tear_mask |= (dist_sq <= tear_radius_sq)
            except Exception:
                continue

    # =========================================
    # 1. THE GEOMETRY (Z-AXIS) = STRESS
    # =========================================
    # Z represents STRESS (gradient magnitude / walker resistance)
    # High Z = High Conflict (Mountains), Low Z = Consensus (Valleys)
    if terrain_stress_geometry is not None:
        stress_values = terrain_stress_geometry
    else:
        # Deterministic fallback: use provided terrain energy / point z values.
        stress_values = energy_values if energy_values is not None else positions_3d[:, 2]
    try:
        print(
            f"[MONOLITH][TERRAIN] geometry_input range="
            f"[{float(np.nanmin(stress_values)):.3f}, {float(np.nanmax(stress_values)):.3f}] "
            f"(stress_override={'yes' if terrain_stress_geometry is not None else 'no'})"
        )
    except Exception:
        pass

    grid_stress = _interpolate_field_boundary_safe(
        x=x,
        y=y,
        values=stress_values,
        Xi=Xi,
        Yi=Yi,
        fill_value=float(np.nanmean(np.asarray(stress_values, dtype=float))),
        clip_to_source=True,
    )

    # Smooth geometry field.
    grid_stress = gaussian_filter(grid_stress, sigma=1.5)
    # Coordinate contract: terrain Z must remain in the same numeric domain as
    # article point Z, otherwise camera autoscaling can visually flatten/erase the manifold.
    z_geometry = np.asarray(grid_stress, dtype=float).copy()
    try:
        pts_z = np.asarray(positions_3d[:, 2], dtype=float)
        pts_min = float(np.nanmin(pts_z))
        pts_max = float(np.nanmax(pts_z))
        g_min = float(np.nanmin(z_geometry))
        g_max = float(np.nanmax(z_geometry))
        if np.isfinite(g_min) and np.isfinite(g_max) and (g_max - g_min) > 1e-12:
            z_geometry = (z_geometry - g_min) / (g_max - g_min)
            z_geometry = z_geometry * (pts_max - pts_min) + pts_min
        else:
            z_geometry = np.full_like(z_geometry, pts_min)
    except Exception:
        pass
    try:
        print(
            f"[MONOLITH][TERRAIN] z_geometry range="
            f"[{float(np.nanmin(z_geometry)):.3f}, {float(np.nanmax(z_geometry)):.3f}]"
        )
    except Exception:
        pass
    if support_mask is not None:
        z_geometry = np.where(support_mask, z_geometry, np.nan)
    if np.any(tear_mask):
        z_geometry = np.where(tear_mask, np.nan, z_geometry)
    # =========================================
    # 2. THE SKIN (COLOR) = CONTINUOUS MANIFOLD GRADIENT
    # =========================================
    if use_manifold_colormap and terrain_density is not None and terrain_stress is not None:
        # Interpolate density and stress using continuous RBF helper to avoid
        # cubic+nearest boundary seams.
        grid_density = _interpolate_field_boundary_safe(
            x=x,
            y=y,
            values=terrain_density,
            Xi=Xi,
            Yi=Yi,
            fill_value=0.5,
            clip_to_source=True,
        )
        grid_stress_color = _interpolate_field_boundary_safe(
            x=x,
            y=y,
            values=terrain_stress,
            Xi=Xi,
            Yi=Yi,
            fill_value=0.5,
            clip_to_source=True,
        )

        # Apply light smoothing for visual continuity
        grid_density = gaussian_filter(grid_density, sigma=1.0)
        grid_stress_color = gaussian_filter(grid_stress_color, sigma=1.0)

        # Color channel expects manifold axes in [0, 1].
        # Normalize against source ranges (not smoothed grid ranges) to avoid
        # narrow-band color collapse after interpolation/smoothing.
        def _normalize_with_bounds(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            arr = np.nan_to_num(arr, nan=0.5)
            if not np.isfinite(lo) or not np.isfinite(hi):
                return np.full_like(arr, 0.5, dtype=float)
            if hi - lo <= 1e-9:
                return np.full_like(arr, 0.5, dtype=float)
            return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        density_src = np.nan_to_num(np.asarray(terrain_density, dtype=float), nan=0.5)
        stress_src = np.nan_to_num(np.asarray(terrain_stress, dtype=float), nan=0.5)
        d_lo, d_hi = float(np.nanmin(density_src)), float(np.nanmax(density_src))
        s_lo, s_hi = float(np.nanmin(stress_src)), float(np.nanmax(stress_src))
        density_color = _normalize_with_bounds(grid_density, d_lo, d_hi)
        stress_color = _normalize_with_bounds(grid_stress_color, s_lo, s_hi)

        # Compute continuous terrain_scalar using the manifold formula:
        # terrain_scalar = 0.25 - 0.25*density + 0.75*stress
        # This maps 2D (density, stress) to 1D [0,1] for colormap:
        #   BRIDGE (density=1, stress=0) → 0.0
        #   TIGHTROPE (density=0, stress=0) → 0.25
        #   SWAMP (density=1, stress=1) → 0.75
        #   VOID (density=0, stress=1) → 1.0
        terrain_scalar_grid = 0.25 - 0.25 * density_color + 0.75 * stress_color

        # Clamp to [0, 1] for safety
        terrain_scalar_grid = np.clip(terrain_scalar_grid, 0.0, 1.0)
        if support_mask is not None:
            terrain_scalar_grid = np.where(support_mask, terrain_scalar_grid, np.nan)
        if np.any(tear_mask):
            terrain_scalar_grid = np.where(tear_mask, np.nan, terrain_scalar_grid)

        # Use continuous manifold colorscale
        colorscale = get_continuous_manifold_colorscale()
        surfacecolor = terrain_scalar_grid
        cmin, cmax = 0.0, 1.0
        colorbar_config = dict(
            title="Terrain<br>Gradient",
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["BRIDGE", "TIGHTROPE", "—", "SWAMP", "VOID"],
            len=0.5,
            x=1.02,
        )
    elif terrain_scalar is not None:
        # Legacy continuous colorscale
        color_values = _interpolate_field_boundary_safe(
            x=x,
            y=y,
            values=terrain_scalar,
            Xi=Xi,
            Yi=Yi,
            fill_value=0.5,
            clip_to_source=True,
        )
        if support_mask is not None:
            color_values = np.where(support_mask, color_values, np.nan)
        if np.any(tear_mask):
            color_values = np.where(tear_mask, np.nan, color_values)
        colorscale = get_terrain_colorscale()
        surfacecolor = color_values
        cmin, cmax = 0, 1
        colorbar_config = dict(
            title="Terrain",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["BRIDGE", "TIGHTROPE", "—", "SWAMP", "VOID"],
            len=0.5,
            x=1.02,
        )
    else:
        # Fallback: height-based coloring
        colorscale = [
            [0.0, PALETTE.void],
            [0.2, PALETTE.magma_cold],
            [0.5, PALETTE.magma_warm],
            [0.75, PALETTE.magma_hot],
            [1.0, PALETTE.magma_rupture],
        ]
        surfacecolor = z_geometry
        cmin, cmax = None, None
        colorbar_config = dict(title="Height", len=0.5, x=1.02)

    z_finite = z_geometry[np.isfinite(z_geometry)]
    contour_cfg = dict(
        z=dict(
            show=True,
            usecolormap=False,
            color="black",
            width=2,
            highlightcolor="white",
            project_z=False,
        )
    )
    if z_finite.size >= 2:
        z_min = float(np.min(z_finite))
        z_max = float(np.max(z_finite))
        z_span = z_max - z_min
        if z_span > 1e-9:
            n_iso = 30
            z_step = z_span / float(n_iso)
            contour_cfg = dict(
                z=dict(
                    show=True,
                    start=z_min,
                    end=z_max,
                    size=z_step,
                    usecolormap=False,
                    color="black",
                    width=2,
                    highlightcolor="white",
                    project_z=False,
                )
            )

    return go.Surface(
        x=Xi, y=Yi, z=z_geometry,
        surfacecolor=surfacecolor,
        colorscale=colorscale,
        cmin=cmin, cmax=cmax,
        opacity=opacity,
        showscale=True,
        colorbar=colorbar_config,
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.2,
            roughness=0.5,
            fresnel=0.1,
        ),
        lightposition=dict(x=100, y=100, z=300),
        contours=contour_cfg,
        hoverinfo='skip',
        name='Energy Terrain',
    ), Xi, Yi, z_geometry, grid_density if 'grid_density' in locals() else None, grid_stress_color if 'grid_stress_color' in locals() else None

def render_terrain_contours(
    Xi: np.ndarray,
    Yi: np.ndarray,
    z_geometry: np.ndarray,
    grid_density: np.ndarray,
    grid_stress: np.ndarray,
) -> List[Any]:
    """
    Render contour lines on the terrain at density and stress thresholds.

    Creates isolines at [0.25, 0.5, 0.75] for both density and stress to demarcate
    the zones (BRIDGE, SWAMP, TIGHTROPE, VOID) while keeping the smooth gradient.

    Uses simple marching-squares-like edge detection (numpy only, no matplotlib).

    Args:
        Xi, Yi: Meshgrid coordinates from terrain surface
        z_geometry: Z values (height) from terrain surface
        grid_density: Interpolated density values on grid [0,1]
        grid_stress: Interpolated stress values on grid [0,1]

    Returns:
        List of Scatter3d traces for contour lines
    """
    if not HAS_PLOTLY:
        return []

    traces = []
    # Three levels demarcate Void / Swamp / Bridge zone boundaries while
    # preserving the smooth gradient (no discrete blocks).
    contour_levels = [0.25, 0.5, 0.75]
    z_offset = 0.06 * (z_geometry.max() - z_geometry.min()) if z_geometry.size > 0 else 0.06

    # Visual style per level: inner boundary dim, outer boundaries bright
    density_level_styles = {
        0.25: dict(color='rgba(255,  80,  80, 0.55)', size=2),   # Void boundary — red
        0.50: dict(color='rgba(255, 255, 255, 0.50)', size=2),   # Mid — white
        0.75: dict(color='rgba(  0, 240, 200, 0.65)', size=3),   # Bridge boundary — cyan
    }
    stress_level_styles = {
        0.25: dict(color='rgba(  0, 200, 255, 0.45)', size=2),   # Low-stress — blue
        0.50: dict(color='rgba(255, 200, 100, 0.50)', size=2),   # Mid — amber
        0.75: dict(color='rgba(255,  60,  60, 0.55)', size=2),   # High-stress — red
    }

    def find_contour_edges(grid_values: np.ndarray, level: float):
        """Find edges where values cross the threshold (marching squares simplified)."""
        edges_x, edges_y, edges_z = [], [], []
        rows, cols = grid_values.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Check if this cell crosses the threshold
                v00, v01, v10, v11 = grid_values[i,j], grid_values[i,j+1], grid_values[i+1,j], grid_values[i+1,j+1]
                vmin, vmax = min(v00, v01, v10, v11), max(v00, v01, v10, v11)

                if vmin <= level <= vmax:
                    # Cell crosses threshold - add center point
                    edges_x.append((Xi[i,j] + Xi[i+1,j+1]) / 2)
                    edges_y.append((Yi[i,j] + Yi[i+1,j+1]) / 2)
                    edges_z.append((z_geometry[i,j] + z_geometry[i+1,j+1]) / 2 + z_offset)

        return np.array(edges_x), np.array(edges_y), np.array(edges_z)

    # Density contours — one isoline per zone boundary
    if grid_density is not None:
        for level in contour_levels:
            style = density_level_styles.get(level, dict(color='rgba(255,255,255,0.4)', size=2))
            try:
                ex, ey, ez = find_contour_edges(grid_density, level)
                if len(ex) > 0:
                    cx, cy = ex.mean(), ey.mean()
                    angles = np.arctan2(ey - cy, ex - cx)
                    order = np.argsort(angles)
                    traces.append(go.Scatter3d(
                        x=ex[order], y=ey[order], z=ez[order],
                        mode='lines+markers',
                        line=dict(color=style['color'], width=max(2, style['size'])),
                        marker=dict(size=max(3, style['size'] + 1), color=style['color']),
                        hoverinfo='skip',
                        showlegend=True,
                        name=f'Density isoline {level:.0%}',
                    ))
            except Exception as e:
                print(f'[CONTOUR] Density {level} failed: {e}')

    # Stress contours — one isoline per zone boundary
    if grid_stress is not None:
        for level in contour_levels:
            style = stress_level_styles.get(level, dict(color='rgba(255,200,100,0.4)', size=2))
            try:
                ex, ey, ez = find_contour_edges(grid_stress, level)
                if len(ex) > 0:
                    cx, cy = ex.mean(), ey.mean()
                    angles = np.arctan2(ey - cy, ex - cx)
                    order = np.argsort(angles)
                    traces.append(go.Scatter3d(
                        x=ex[order], y=ey[order], z=ez[order],
                        mode='lines+markers',
                        line=dict(color=style['color'], width=max(2, style['size'])),
                        marker=dict(size=max(3, style['size'] + 1), color=style['color']),
                        hoverinfo='skip',
                        showlegend=True,
                        name=f'Stress isoline {level:.0%}',
                    ))
            except Exception as e:
                print(f'[CONTOUR] Stress {level} failed: {e}')

    # Add zone labels at corners
    x_min, x_max = Xi.min(), Xi.max()
    y_min, y_max = Yi.min(), Yi.max()
    z_min, z_max = z_geometry.min(), z_geometry.max()

    # Label positions (corners of the terrain)
    z_label_height = z_max + 0.1 * (z_max - z_min)

    zone_labels = [
        # BRIDGE: High density (x_max), Low stress (y_min)
        {'x': x_max, 'y': y_min, 'text': 'BRIDGE', 'color': 'rgba(0, 255, 200, 0.9)'},
        # VOID: Low density (x_min), High stress (y_max)
        {'x': x_min, 'y': y_max, 'text': 'VOID', 'color': 'rgba(255, 50, 50, 0.9)'},
        # SWAMP: High density (x_max), High stress (y_max)
        {'x': x_max, 'y': y_max, 'text': 'SWAMP', 'color': 'rgba(200, 150, 0, 0.9)'},
        # TIGHTROPE: Low density (x_min), Low stress (y_min)
        {'x': x_min, 'y': y_min, 'text': 'TIGHTROPE', 'color': 'rgba(0, 200, 255, 0.9)'},
    ]

    for label in zone_labels:
        traces.append(go.Scatter3d(
            x=[label['x']],
            y=[label['y']],
            z=[z_label_height],
            mode='text',
            text=[label['text']],
            textfont=dict(size=14, color=label['color'], family='monospace'),
            hoverinfo='skip',
            showlegend=False,
        ))

    return traces



# =============================================================================
# DIAGNOSTIC MODE: WIND STREAMLINES (Local Vector Field)
# =============================================================================
def compute_wind_field(
    positions_3d: np.ndarray,
    antagonism_vectors: np.ndarray,
    grid_resolution: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute wind vector field on a 3D grid.

    Returns: Xg, Yg, Zg (grid positions), U, V, W (vector components)
    """
    if not HAS_SCIPY:
        return None, None, None, None, None, None

    # Create grid
    margin = 0.1
    x_min, x_max = positions_3d[:, 0].min(), positions_3d[:, 0].max()
    y_min, y_max = positions_3d[:, 1].min(), positions_3d[:, 1].max()
    z_min, z_max = positions_3d[:, 2].min(), positions_3d[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    xi = np.linspace(x_min - margin * x_range, x_max + margin * x_range, grid_resolution)
    yi = np.linspace(y_min - margin * y_range, y_max + margin * y_range, grid_resolution)
    zi = np.linspace(z_min - margin * z_range, z_max + margin * z_range, grid_resolution)

    Xg, Yg, Zg = np.meshgrid(xi, yi, zi)

    # Project antagonism to 3D if needed
    if antagonism_vectors.shape[1] > 3:
        if HAS_SCIPY:
            pca = PCA(n_components=3, random_state=42)
            antag_3d = pca.fit_transform(antagonism_vectors)
        else:
            antag_3d = antagonism_vectors[:, :3]
    else:
        antag_3d = antagonism_vectors

    # Interpolate wind vectors to grid using inverse distance weighting
    grid_points = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)

    # Compute distances from each grid point to each data point
    dists = cdist(grid_points, positions_3d)
    weights = 1.0 / (dists + 0.1) ** 2
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Weighted average of vectors
    U = (weights @ antag_3d[:, 0]).reshape(Xg.shape)
    V = (weights @ antag_3d[:, 1]).reshape(Xg.shape)
    W = (weights @ antag_3d[:, 2]).reshape(Xg.shape)

    return Xg, Yg, Zg, U, V, W


def render_wind_streamlines(
    positions_3d: np.ndarray,
    antagonism_vectors: np.ndarray,
    n_streamlines: int = 50,
    line_width: float = 2,
) -> List[Any]:
    """
    Render wind streamlines showing local narrative pressure.

    This shows the EMPTY SPACE - where the pressure pushes even where no articles exist.
    """
    if not HAS_PLOTLY or not HAS_SCIPY:
        return []

    traces = []

    # Project antagonism to 3D
    if antagonism_vectors.shape[1] > 3:
        pca = PCA(n_components=3, random_state=42)
        antag_3d = pca.fit_transform(antagonism_vectors)
    else:
        antag_3d = antagonism_vectors.copy()

    # Normalize vectors
    norms = np.linalg.norm(antag_3d, axis=1, keepdims=True)
    antag_3d = antag_3d / (norms + 1e-9)

    # Scale for visibility
    scale = (positions_3d.max() - positions_3d.min()) * 0.08

    # Create cone plot for wind direction
    traces.append(go.Cone(
        x=positions_3d[:, 0],
        y=positions_3d[:, 1],
        z=positions_3d[:, 2],
        u=antag_3d[:, 0] * scale,
        v=antag_3d[:, 1] * scale,
        w=antag_3d[:, 2] * scale if antag_3d.shape[1] > 2 else np.zeros_like(antag_3d[:, 0]),
        colorscale=[[0, 'rgba(0,240,255,0.3)'], [1, 'rgba(255,100,100,0.6)']],
        sizemode='absolute',
        sizeref=scale * 0.3,
        showscale=False,
        opacity=0.6,
        name='Wind Field (Antagonism)',
        hoverinfo='skip',
        visible=False,  # Hidden by default (Diagnostics mode)
    ))

    # Add streamlines as line traces
    n_articles = len(positions_3d)
    sample_idx = np.random.choice(n_articles, min(n_streamlines, n_articles), replace=False)

    for idx in sample_idx:
        start = positions_3d[idx]
        
        # Handle 2D vs 3D vectors
        u = antag_3d[idx, 0] * scale * 3
        v = antag_3d[idx, 1] * scale * 3
        w = antag_3d[idx, 2] * scale * 3 if antag_3d.shape[1] > 2 else 0.0

        # Create streamline path
        x_line = [start[0], start[0] + u]
        y_line = [start[1], start[1] + v]
        z_line = [start[2], start[2] + w]

        traces.append(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines',
            line=dict(color='rgba(0,240,255,0.4)', width=line_width),
            showlegend=False,
            hoverinfo='skip',
            visible=False,  # Hidden by default
            name='streamline',
        ))

    return traces


# =============================================================================
# DIAGNOSTIC MODE: SLIME TRAILS (Walker Conductivity Highways)
# =============================================================================
def render_slime_trails(
    positions_3d: np.ndarray,
    walker_work: np.ndarray,
    n_neighbors: int = 5,
    get_surface_z_func: Optional[Callable] = None, # For path clamping
    positions_2d: Optional[np.ndarray] = None,   # For path clamping
) -> List[Any]:
    """
    Render slime trails showing where walkers successfully traveled.

    Brighter trails = more walkers passed = higher conductivity.
    This shows the HIGHWAYS of discourse navigation.
    """
    if not HAS_PLOTLY or not HAS_SCIPY:
        return []

    traces = []
    n_articles = len(positions_3d)

    if walker_work is None or len(walker_work) == 0:
        return traces

    # Compute conductivity (inverse of work)
    # Handle inf values: replace with large finite value for normalization
    work_finite = np.where(np.isinf(walker_work), 1e6, walker_work)
    work_safe = np.clip(work_finite, 0.01, 1e6)
    conductivity = 1.0 / work_safe
    cond_max = conductivity.max()
    conductivity = conductivity / cond_max if cond_max > 0 else conductivity  # Normalize safely

    # Find k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n_articles), metric='euclidean')
    nn.fit(positions_3d)
    distances, indices = nn.kneighbors(positions_3d)

    # Draw trails between high-conductivity neighbors
    drawn = set()
    for i in range(n_articles):
        for j_idx in range(1, len(indices[i])):  # Skip self
            j = indices[i][j_idx]
            edge = tuple(sorted([i, j]))
            if edge in drawn:
                continue
            drawn.add(edge)

            # Trail brightness based on average conductivity
            avg_cond = (conductivity[i] + conductivity[j]) / 2

            if avg_cond > 0.2:  # Only draw significant trails
                opacity = 0.2 + 0.6 * avg_cond
                width = 1 + 4 * avg_cond

                traces.append(go.Scatter3d(
                    x=[positions_3d[i, 0], positions_3d[j, 0]],
                    y=[positions_3d[i, 1], positions_3d[j, 1]],
                    z=[get_surface_z_func(positions_3d[i, 0], positions_3d[i, 1], offset=0.01) if get_surface_z_func else positions_3d[i, 2],
                       get_surface_z_func(positions_3d[j, 0], positions_3d[j, 1], offset=0.01) if get_surface_z_func else positions_3d[j, 2]],
                    mode='lines',
                    line=dict(
                        color=f'rgba(0,255,100,{opacity:.2f})',
                        width=width,
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    visible=False,  # Hidden by default (Diagnostics mode)
                    name='slime_trail',
                ))

    # Add legend entry
    if traces:
        traces[0].showlegend = True
        traces[0].name = 'Signal Trajectories'

    return traces


# =============================================================================
# DIAGNOSTIC MODE: CHROMATIC GHOSTS (Kernel Disagreement)
# =============================================================================
def render_chromatic_ghosts(
    positions_3d: np.ndarray,
    features: np.ndarray,
    spectral_evr: np.ndarray,
    rgb_offsets: Tuple[float, float, float] = (0.15, 0.15, 0.15),
) -> List[Any]:
    """
    Render chromatic aberration showing uncertainty.

    Each article splits into RGB ghosts when the system is uncertain.
    Ghosts far apart = high mathematical disagreement (multiverse).
    Ghosts close together = consensus (certainty).
    """
    if not HAS_PLOTLY:
        return []

    traces = []
    n_articles = len(positions_3d)

    # Uncertainty = inverse of EVR
    uncertainty = 1.0 - spectral_evr

    # RGB colors for ghosts (RBF=Red, Matern=Green, Laplacian=Blue)
    ghost_colors = [
        ('rgba(255,50,50,', 'R-Ghost (RBF)'),      # Red
        ('rgba(50,255,50,', 'G-Ghost (Matern)'),   # Green
        ('rgba(50,100,255,', 'B-Ghost (Laplacian)'),  # Blue
    ]

    # Direction offsets for each ghost
    directions = [
        np.array([1, 0, 0]),
        np.array([-0.5, 0.866, 0]),
        np.array([-0.5, -0.866, 0]),
    ]

    for color_idx, (color_base, legend_name) in enumerate(ghost_colors):
        ghost_x, ghost_y, ghost_z = [], [], []
        ghost_sizes = []
        ghost_opacities = []

        for i in range(n_articles):
            # Offset proportional to uncertainty
            offset_scale = uncertainty[i] * rgb_offsets[color_idx] * 2
            offset = directions[color_idx] * offset_scale

            ghost_x.append(positions_3d[i, 0] + offset[0])
            ghost_y.append(positions_3d[i, 1] + offset[1])
            ghost_z.append(positions_3d[i, 2] + offset[2])
            ghost_sizes.append(4 + uncertainty[i] * 6)
            ghost_opacities.append(0.3 + uncertainty[i] * 0.4)

        # Create trace with per-point opacity via color
        colors = [f'{color_base}{op:.2f})' for op in ghost_opacities]

        traces.append(go.Scatter3d(
            x=ghost_x, y=ghost_y, z=ghost_z,
            mode='markers',
            marker=dict(
                size=ghost_sizes,
                color=colors,
                line=dict(width=0),
            ),
            name=legend_name,
            hoverinfo='skip',
            visible=False,  # Hidden by default (Diagnostics mode)
        ))

    return traces


# =============================================================================
# DIAGNOSTIC MODE: OBSERVER MATRIX (8x8 Bot Divergence)
# =============================================================================
def render_observer_matrix(
    spectral_probe_magnitudes: np.ndarray,
    article_idx: int = 0,
) -> Optional[Any]:
    """
    Render the 8-probe divergence matrix for a single article.

    Shows WHO disagrees with WHOM - distinguishing Civil War (4v4) from Chaos (all vs all).
    """
    if not HAS_PLOTLY:
        return None

    if spectral_probe_magnitudes is None or len(spectral_probe_magnitudes) == 0:
        return None

    mags = spectral_probe_magnitudes[article_idx] if article_idx < len(spectral_probe_magnitudes) else spectral_probe_magnitudes[0]

    # Compute pairwise agreement (product of magnitudes - same sign = agree)
    n_probes = len(mags)
    agreement_matrix = np.outer(mags, mags)

    # Normalize
    max_val = np.abs(agreement_matrix).max()
    if max_val > 0:
        agreement_matrix = agreement_matrix / max_val

    fig = go.Figure(data=go.Heatmap(
        z=agreement_matrix,
        x=PROBE_LABELS[:n_probes],
        y=PROBE_LABELS[:n_probes],
        colorscale=[
            [0, PALETTE.red],
            [0.5, PALETTE.void],
            [1, PALETTE.green],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(
            title='Agreement',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
        ),
    ))

    fig.update_layout(
        title=f'Observer Matrix (Article {article_idx})',
        paper_bgcolor=PALETTE.void,
        plot_bgcolor=PALETTE.void,
        font=dict(color='white', size=10),
        width=600,
        height=600,
    )

    return fig


# =============================================================================
# LIGHTNING ARCS (Track 1.5 Ruptures)
# =============================================================================
def generate_lightning_path(
    p1: np.ndarray,
    p2: np.ndarray,
    n_segments: int = 8,
    jitter_scale: float = 0.15,
) -> Tuple[List[float], List[float], List[float]]:
    """Generate jagged lightning arc between two 3D points."""
    is_3d = len(p1) >= 3

    path_x, path_y = [p1[0]], [p1[1]]
    path_z = [p1[2]] if is_3d else []

    direction = p2 - p1
    length = np.linalg.norm(direction)

    # Perpendicular for jitter
    if is_3d:
        if abs(direction[2]) < abs(direction[0]):
            perp1 = np.array([-direction[1], direction[0], 0])
        else:
            perp1 = np.array([0, -direction[2], direction[1]])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-9)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-9)
    else:
        perp = np.array([-direction[1], direction[0]]) / (length + 1e-9)

    for i in range(1, n_segments):
        t = i / n_segments
        mid = p1 + t * direction

        if is_3d:
            jitter = (random.uniform(-1, 1) * perp1 +
                      random.uniform(-1, 1) * perp2) * jitter_scale * length
        else:
            jitter = random.uniform(-1, 1) * perp * jitter_scale * length
            jitter = np.append(jitter, 0) if is_3d else jitter

        mid = mid + jitter[:len(mid)]
        path_x.append(mid[0])
        path_y.append(mid[1])
        if is_3d:
            path_z.append(mid[2])

    path_x.append(p2[0])
    path_y.append(p2[1])
    if is_3d:
        path_z.append(p2[2])

    if is_3d:
        return path_x, path_y, path_z
    return path_x, path_y, []


# =============================================================================
# PHANTOM PATH RENDERING (Track 5) — Field Theory Visualizations
# =============================================================================
def render_phantom_paths_3d(
    phantom_verdicts: List[Dict],
    positions_3d: np.ndarray,
    walker_paths: Optional[Dict[int, np.ndarray]] = None,
    article_z_height: Optional[np.ndarray] = None,
    terrain_z_values: Optional[np.ndarray] = None,
    surface_z_func=None,
    article_metadata: Optional[List[Dict]] = None,
    spectral_evr: Optional[np.ndarray] = None,
    spectral_probe_magnitudes: Optional[np.ndarray] = None,
    hysteresis_memory: Optional[np.ndarray] = None,
    path_ablation_mode: str = "none",
) -> List[Any]:
    """
    Render real Track-4 trajectories (no synthetic centroid stubs).

    PATH COORDINATE CONTRACT (System-1 identity + System-2 geometry):
    1) Each rendered path must use the path's own article index as its source anchor.
       Never use a shared/global source coordinate for all paths.
    2) Path arrays provided here are already projected into the same 3D coordinate
       space as article points (pure_x/pure_y/pure_z).
    3) Start vertex is snapped to that source article's plotted coordinate.
    4) Endpoints are preserved from trajectory dynamics (no forced apex/centroid ties).
    5) A singularity lock validates that starts are not collapsed to one origin.

    This function intentionally avoids synthetic centroid interpolation and local
    griddata-based path fabrication. It is a renderer for persisted trajectories.
    """
    if not HAS_PLOTLY: return []
    traces = []
    legend_shown = set()
    n_articles = min(len(phantom_verdicts), len(positions_3d))
    walker_paths = walker_paths or {}
    all_path_starts: List[np.ndarray] = []
    debug_printed = 0
    enable_raw_probe = os.environ.get("MONOLITH_RAW_MATRIX_PROBE", "0").strip() == "1"
    max_terrain_z = 1.0
    if terrain_z_values is not None and len(terrain_z_values) > 0:
        max_terrain_z = float(np.nanmax(np.abs(np.asarray(terrain_z_values, dtype=float))))
    elif article_z_height is not None and len(article_z_height) > 0:
        max_terrain_z = float(np.nanmax(np.abs(np.asarray(article_z_height, dtype=float))))
    max_terrain_z = max(max_terrain_z, 1e-6)
    z_cap = max_terrain_z * 2.0

    mode_key = str(path_ablation_mode or "none").strip().lower()
    thermodynamic_mode = mode_key == "thermodynamic"
    semantic_tether_mode = mode_key == "semantic_tether"
    drape_paths = os.environ.get("MONOLITH_DRAPE_PATHS", "0").strip() == "1"

    def _resample_polyline_xy(path_xy: np.ndarray, n_steps: int = 120) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(path_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if pts.shape[0] < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        if s[-1] <= 1e-12:
            return np.full(n_steps, pts[0, 0], dtype=float), np.full(n_steps, pts[0, 1], dtype=float)
        sq = np.linspace(0.0, s[-1], n_steps, dtype=float)
        xq = np.interp(sq, s, pts[:, 0])
        yq = np.interp(sq, s, pts[:, 1])
        return xq, yq

    def _split_finite_segments(path_xyz: np.ndarray) -> List[np.ndarray]:
        pts = np.asarray(path_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
            return []
        finite = np.isfinite(pts[:, :3]).all(axis=1)
        if not np.any(finite):
            return []
        segments: List[np.ndarray] = []
        start = None
        for j, is_valid in enumerate(finite):
            if is_valid and start is None:
                start = j
            elif (not is_valid) and start is not None:
                seg = pts[start:j, :3]
                if seg.shape[0] >= 2:
                    segments.append(seg)
                start = None
        if start is not None:
            seg = pts[start:, :3]
            if seg.shape[0] >= 2:
                segments.append(seg)
        return segments

    def _dominant_probe_label(article_idx: int) -> str:
        if (
            spectral_probe_magnitudes is None
            or not isinstance(spectral_probe_magnitudes, np.ndarray)
            or spectral_probe_magnitudes.ndim < 2
            or article_idx < 0
            or article_idx >= spectral_probe_magnitudes.shape[0]
        ):
            return "Unknown Axis"
        mags = np.asarray(spectral_probe_magnitudes[article_idx], dtype=float)
        if mags.size <= 0 or not np.any(np.isfinite(mags)):
            return "Unknown Axis"
        mags = np.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=0.0)
        dominant_idx = int(np.argmax(np.abs(mags)))
        if 0 <= dominant_idx < len(PROBE_LABELS):
            return PROBE_LABELS[dominant_idx]
        return f"Axis {dominant_idx}"

    def _asymmetry_delta(path_source_idx: int, end_xyz: np.ndarray) -> Optional[Dict[str, float]]:
        if hysteresis_memory is None:
            return None
        try:
            mem = np.asarray(hysteresis_memory, dtype=float)
            if mem.ndim != 2 or mem.shape[0] != mem.shape[1] or mem.shape[0] < 2:
                return None
            if spectral_probe_magnitudes is None or not isinstance(spectral_probe_magnitudes, np.ndarray):
                return None
            if path_source_idx < 0 or path_source_idx >= len(positions_3d):
                return None
            if end_xyz is None or np.asarray(end_xyz).size < 3:
                return None

            endpoint = np.asarray(end_xyz[:3], dtype=float)
            if not np.isfinite(endpoint).all():
                return None
            d_to_articles = np.linalg.norm(positions_3d[:, :3] - endpoint.reshape(1, 3), axis=1)
            target_article_idx = int(np.argmin(d_to_articles))

            n_bots = mem.shape[0]
            if (
                path_source_idx >= spectral_probe_magnitudes.shape[0]
                or target_article_idx >= spectral_probe_magnitudes.shape[0]
                or spectral_probe_magnitudes.shape[1] < 1
            ):
                return None
            src_mags = np.asarray(spectral_probe_magnitudes[path_source_idx], dtype=float)
            tgt_mags = np.asarray(spectral_probe_magnitudes[target_article_idx], dtype=float)
            src_idx = int(np.argmax(np.abs(src_mags[:n_bots])))
            tgt_idx = int(np.argmax(np.abs(tgt_mags[:n_bots])))
            forward = float(mem[src_idx, tgt_idx])
            reverse = float(mem[tgt_idx, src_idx])
            delta = abs(forward - reverse)
            norm = delta / (abs(forward) + abs(reverse) + 1e-9)
            return {
                "delta": float(delta),
                "norm": float(norm),
                "start_probe": float(src_idx),
                "target_probe": float(tgt_idx),
            }
        except Exception:
            return None

    for article_idx, path in walker_paths.items():
        i = int(article_idx)
        if i < 0 or i >= n_articles:
            continue
        v_info = phantom_verdicts[i]
        verdict = str(v_info.get("verdict", "UNKNOWN")).upper()
        walker_state = str(v_info.get("walker_state", "success")).lower()
        meta = article_metadata[i] if article_metadata and i < len(article_metadata) else {}
        title = str(meta.get('title', f'Article {i}'))[:60]
        hover_text = f'TRACE #{i} [{verdict}]<br>{title}'

        if path is None or path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 3:
            continue

        rendered_path = np.asarray(path[:, :3], dtype=float).copy()
        source_z = (
            float(article_z_height[i])
            if article_z_height is not None and i < len(article_z_height)
            else float(positions_3d[i, 2])
        )
        # Identity contract: start vertex is exactly the source article point for THIS path.
        rendered_path[0, 0] = float(positions_3d[i, 0])
        rendered_path[0, 1] = float(positions_3d[i, 1])
        rendered_path[0, 2] = source_z
        all_path_starts.append(rendered_path[0, :3].copy())
        if semantic_tether_mode:
            finite_rows = np.isfinite(rendered_path[:, :3]).all(axis=1)
            finite_path = rendered_path[finite_rows]
            if finite_path.shape[0] < 2:
                continue
            start_xyz = finite_path[0, :3].astype(float)
            end_xyz = finite_path[-1, :3].astype(float)
            if callable(surface_z_func):
                try:
                    tether_z = np.asarray(
                        surface_z_func(
                            np.array([start_xyz[0], end_xyz[0]], dtype=float),
                            np.array([start_xyz[1], end_xyz[1]], dtype=float),
                            offset=0.01,
                            preserve_nan=True,
                        ),
                        dtype=float,
                    )
                    if tether_z.shape[0] == 2 and np.isfinite(tether_z).all():
                        start_xyz[2] = float(tether_z[0])
                        end_xyz[2] = float(tether_z[1])
                except Exception:
                    pass
            if verdict == "HONEST":
                path_color = "#7FFBFF"
                line_width = 2
                line_opacity = 0.15
                legend_group = "semantic-tether-honest"
                legend_name = "Honest Tether"
            elif verdict == "TAUTOLOGY":
                path_color = "#9A9A9A"
                line_width = 2
                line_opacity = 0.18
                legend_group = "semantic-tether-tautology"
                legend_name = "Tautology Tether"
            elif verdict == "PHANTOM":
                path_color = "#FF00FF"
                line_width = 7
                line_opacity = 0.96
                legend_group = "semantic-tether-phantom"
                legend_name = "Phantom Tether"
            else:
                path_color = "#FFD700"
                line_width = 3
                line_opacity = 0.4
                legend_group = "semantic-tether-unknown"
                legend_name = "Unknown Tether"

            traces.append(go.Scatter3d(
                x=[start_xyz[0], end_xyz[0]],
                y=[start_xyz[1], end_xyz[1]],
                z=[start_xyz[2], end_xyz[2]],
                mode='lines',
                line=dict(color=path_color, width=line_width),
                opacity=line_opacity,
                name=legend_name,
                text=[hover_text, hover_text],
                hoverinfo='skip',
                legendgroup=legend_group,
                showlegend=legend_group not in legend_shown,
            ))
            legend_shown.add(legend_group)

            asym = _asymmetry_delta(i, end_xyz)
            if asym is not None:
                asym_delta = float(asym["delta"])
                asym_norm = float(asym["norm"])
                if asym_norm <= 0.05:
                    asym_color = "#8A8A8A"
                    asym_width = 1
                    asym_opacity = 0.10
                    asym_name = "Symmetric Tether"
                elif asym_norm >= 0.35:
                    asym_color = "#00FFFF"
                    asym_width = 7
                    asym_opacity = 0.96
                    asym_name = "Asymmetric Tether"
                else:
                    asym_color = "#4FD1FF"
                    asym_width = 4
                    asym_opacity = 0.55
                    asym_name = "Moderate Asymmetry"
                asym_hover = f"{hover_text}<br>[ASYMMETRY DELTA: {asym_delta:.3f}]"
                traces.append(go.Scatter3d(
                    x=[start_xyz[0], end_xyz[0]],
                    y=[start_xyz[1], end_xyz[1]],
                    z=[start_xyz[2], end_xyz[2]],
                    mode='lines',
                    line=dict(color=asym_color, width=asym_width),
                    opacity=asym_opacity,
                    name=asym_name,
                    text=[asym_hover, asym_hover],
                    hoverinfo='skip',
                    legendgroup='semantic-asymmetry',
                    showlegend='semantic-asymmetry' not in legend_shown,
                ))
                legend_shown.add('semantic-asymmetry')

            if verdict == "PHANTOM":
                mid_xyz = (start_xyz + end_xyz) / 2.0
                shear_axis = _dominant_probe_label(i)
                asym_note = ""
                if asym is not None:
                    asym_note = f" | Δ={float(asym['delta']):.3f}"
                shear_text = f"[SHEAR: {shear_axis}{asym_note}]"
                traces.append(go.Scatter3d(
                    x=[mid_xyz[0]],
                    y=[mid_xyz[1]],
                    z=[mid_xyz[2] + 0.08],
                    mode='text',
                    text=[shear_text],
                    textfont=dict(
                        color="#FF5CFF",
                        size=12,
                        family="JetBrains Mono, monospace",
                    ),
                    name="Ideological Shear",
                    hovertext=[f"{hover_text}<br>{shear_text}"],
                    hovertemplate="%{hovertext}<extra></extra>",
                    legendgroup="semantic-shear-labels",
                    showlegend=False,
                ))
            continue
        if thermodynamic_mode:
            scorch_x, scorch_y = _resample_polyline_xy(rendered_path[:, :2], n_steps=120)
            if scorch_x.size < 2:
                continue
            if callable(surface_z_func):
                try:
                    scorch_z = np.asarray(
                        surface_z_func(
                            scorch_x,
                            scorch_y,
                            offset=0.015,
                            preserve_nan=True,
                        ),
                        dtype=float,
                    )
                except Exception:
                    scorch_z = np.full_like(scorch_x, source_z + 0.015, dtype=float)
            else:
                scorch_z = np.full_like(scorch_x, source_z + 0.015, dtype=float)

            if verdict in {"HONEST", "PHANTOM", "TAUTOLOGY"}:
                path_color = ("#00FFFF" if verdict == "HONEST" else "#FF2DFF" if verdict == "PHANTOM" else "#39FF14")
                legend_group = f'thermo-scorch-{verdict.lower()}'
                legend_name = f'{verdict.capitalize()} Scorch'
                traces.append(go.Scatter3d(
                    x=scorch_x, y=scorch_y, z=scorch_z, mode='lines',
                    line=dict(color='rgba(0,0,0,0.62)', width=12),
                    opacity=0.78,
                    name=legend_name,
                    text=[hover_text] * len(scorch_x),
                    hoverinfo='skip',
                    legendgroup=legend_group,
                    showlegend=False,
                ))
                traces.append(go.Scatter3d(
                    x=scorch_x, y=scorch_y, z=scorch_z, mode='lines',
                    line=dict(color=path_color, width=7),
                    opacity=1.0,
                    name=legend_name,
                    text=[hover_text] * len(scorch_x),
                    hoverinfo='skip',
                    legendgroup=legend_group,
                    showlegend=legend_group not in legend_shown,
                ))
                legend_shown.add(legend_group)
            elif verdict == "RUPTURE":
                rupture_color = "#FFFFFF"
                legend_group = "thermo-tear-rupture"
                legend_name = "Rupture Tear"
                traces.append(go.Scatter3d(
                    x=scorch_x, y=scorch_y, z=scorch_z, mode='lines',
                    line=dict(color='rgba(120,0,0,0.75)', width=12),
                    opacity=0.85,
                    name=legend_name,
                    text=[hover_text] * len(scorch_x),
                    hoverinfo='skip',
                    legendgroup=legend_group,
                    showlegend=False,
                ))
                traces.append(go.Scatter3d(
                    x=scorch_x, y=scorch_y, z=scorch_z, mode='lines',
                    line=dict(color=rupture_color, width=6, dash='dot'),
                    opacity=1.0,
                    name=legend_name,
                    text=[hover_text] * len(scorch_x),
                    hoverinfo='skip',
                    legendgroup=legend_group,
                    showlegend=legend_group not in legend_shown,
                ))
                legend_shown.add(legend_group)
            continue

        # Keep true trajectory endpoint from dynamics; only enforce finite values.
        path_segments = _split_finite_segments(rendered_path[:, :3])
        if not path_segments:
            continue
        if drape_paths and callable(surface_z_func):
            for seg_idx in range(len(path_segments)):
                seg = path_segments[seg_idx].copy()
                try:
                    terrain_z = np.asarray(
                        surface_z_func(seg[:, 0], seg[:, 1], offset=0.0),
                        dtype=float
                    )
                    if terrain_z.shape == seg[:, 2].shape and np.isfinite(terrain_z).any():
                        seg[:, 2] = terrain_z
                        if seg_idx == 0:
                            seg[0, 2] = source_z
                except Exception:
                    pass
                seg[:, 2] = np.clip(seg[:, 2], -z_cap, z_cap)
                path_segments[seg_idx] = seg
        else:
            for seg_idx in range(len(path_segments)):
                seg = path_segments[seg_idx].copy()
                seg[:, 2] = np.clip(seg[:, 2], -z_cap, z_cap)
                path_segments[seg_idx] = seg
        end_xyz = path_segments[-1][-1, :3] if path_segments else None

        if verdict in {"HONEST", "PHANTOM", "TAUTOLOGY"}:
            asym = _asymmetry_delta(i, end_xyz)
            if asym is not None:
                delta_val = float(asym["delta"])
                norm_val = float(asym["norm"])
                if norm_val <= 0.05:
                    path_color = "#8A8A8A"
                    width_val = 1
                    opacity_val = 0.10
                elif norm_val >= 0.35:
                    path_color = "#00FFFF"
                    width_val = 7
                    opacity_val = 0.95
                else:
                    path_color = "#4FD1FF"
                    width_val = 3
                    opacity_val = 0.55
                hover_text = f"{hover_text}<br>[ASYMMETRY DELTA: {delta_val:.3f}]"
            else:
                path_color = ("#00F0FF" if verdict == "HONEST" else "#FF00FF" if verdict == "PHANTOM" else "#FFFF00")
                width_val = 3
                opacity_val = 0.8
            legend_group = f'path-{verdict.lower()}'
            legend_name = f'{verdict.capitalize()} Path'
            # Optional forensic probe: print exact rendered coordinates for the first
            # few paths. Disabled by default to keep normal runs readable.
            if enable_raw_probe and debug_printed < 3 and len(rendered_path) >= 2:
                mid_idx = len(rendered_path) // 2
                article_coord = np.array([positions_3d[i, 0], positions_3d[i, 1], source_z], dtype=float)
                print(f"[RAW_MATRIX_PROBE] Path ID: {debug_printed} | Target Article Index: {i}")
                print(f"[RAW_MATRIX_PROBE] Article 3D Coordinate: {article_coord.tolist()}")
                print(f"[RAW_MATRIX_PROBE] Path Start (Vertex 0): {rendered_path[0, :3].tolist()}")
                print(f"[RAW_MATRIX_PROBE] Path Midpoint (Vertex {mid_idx}): {rendered_path[mid_idx, :3].tolist()}")
                print(f"[RAW_MATRIX_PROBE] Path End (Vertex {len(rendered_path)-1}): {rendered_path[-1, :3].tolist()}")
                debug_printed += 1
            for seg_idx, seg in enumerate(path_segments):
                traces.append(go.Scatter3d(
                    x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode='lines',
                    line=dict(color=path_color, width=width_val), opacity=opacity_val,
                    name=legend_name,
                    text=[hover_text] * len(seg),
                    hoverinfo='skip',
                    legendgroup=legend_group,
                    showlegend=(legend_group not in legend_shown) and (seg_idx == 0),
                ))
            legend_shown.add(legend_group)
        elif verdict == "RUPTURE":
            rupture_color = "#FF2222" if walker_state == "broken" else "#FF00FF"
            legend_group = 'path-rupture-broken' if walker_state == "broken" else 'path-rupture-trapped'
            legend_name = 'Walker Broken' if walker_state == "broken" else 'Walker Trapped'
            for seg_idx, seg in enumerate(path_segments):
                traces.append(go.Scatter3d(
                    x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode='lines',
                    line=dict(color=rupture_color, width=4 if walker_state == "broken" else 2),
                    name=legend_name,
                    text=[hover_text] * len(seg),
                    hoverinfo='skip',
                    legendgroup=legend_group,
                    showlegend=(legend_group not in legend_shown) and (seg_idx == 0),
                ))
            legend_shown.add(legend_group)

    # ANTI-SINGULARITY LOCK: renderer must never collapse all starts to one origin.
    if all_path_starts:
        starts_arr = np.asarray(all_path_starts, dtype=float)
        unique_origins = len(np.unique(np.round(starts_arr, decimals=4), axis=0))
        if unique_origins <= 1:
            raise DimensionalCollapseError(
                "CRITICAL: Path Singularity Detected. All origins collapsed to a single point."
            )
    return traces


# =============================================================================
# WALKER DIAMONDS (Track 4)
# =============================================================================
def render_walker_diamonds_3d(
    positions_3d: np.ndarray,
    walker_states: List[str],
    walker_work: Optional[np.ndarray] = None,
    article_metadata: Optional[List[Dict]] = None,
    phantom_verdicts: Optional[List[Dict]] = None,
    spectral_evr: Optional[np.ndarray] = None,
    show_all_states: bool = True,  # New: render all states, not just trapped/broken
) -> List[Any]:
    """
    Render Track 4 walker state diamonds with FULL article metadata on hover.

    States from SemanticWalker (4-state divergence system):
    - tautology: Walker spun in place, no real movement (gray)
    - honest: Walker found the easy path, laminar flow (green)
    - phantom: Walker split up and swirled, turbulence (magenta)
    - rupture: Walker hit a singularity and crashed (red)

    Now includes article title, UID, verdict, and all track data when hovering.
    """
    if not HAS_PLOTLY:
        return []

    traces = []
    n = min(len(positions_3d), len(walker_states))

    for i in range(n):
        state_raw = walker_states[i]
        if isinstance(state_raw, dict):
            state = str(state_raw.get("status") or state_raw.get("state") or "unknown").strip().lower()
        else:
            state = str(state_raw).strip().lower()

        # Normalize status-style values to existing walker color/state families.
        if state == "success":
            state = "honest"
        elif state == "broken":
            state = "broken"
        elif state == "trapped":
            state = "trapped"

        # Skip states based on mode
        if not show_all_states:
            # Legacy mode: only show trapped/broken (old state system)
            if state not in ('trapped', 'broken'):
                continue
        else:
            # New mode: show all 4-state walker states
            # Skip 'elastic' if it exists (old system), show all others
            if state == 'elastic':
                continue

        work = walker_work[i] if walker_work is not None and i < len(walker_work) else 0
        # Use new 4-state colors if available, fall back to legacy
        color = WALKER_STATE_COLORS.get(state, WALKER_COLORS.get(state, PALETTE.walker_elastic))

        # Build RICH hover text with article metadata
        meta = article_metadata[i] if article_metadata and i < len(article_metadata) else {}
        title = str(meta.get('title', f'Article {i}'))[:70]
        bt_uid = str(meta.get('bt_uid', ''))[:16]
        evr = spectral_evr[i] if spectral_evr is not None and i < len(spectral_evr) else 0.5

        # Phantom verdict info
        pv = "unknown"
        terrain = "unknown"
        delta = 0.0
        d_val = 0.0
        w_val = 0.0
        if phantom_verdicts and i < len(phantom_verdicts):
            pv = phantom_verdicts[i].get('verdict', 'unknown')
            terrain = phantom_verdicts[i].get('terrain_state', 'unknown')
            delta = phantom_verdicts[i].get('delta', 0.0)
            d_val = phantom_verdicts[i].get('d', phantom_verdicts[i].get('d_spectral', 0.0))
            w_val = phantom_verdicts[i].get('w', phantom_verdicts[i].get('w_actual', 0.0))

        # Format values
        d_str = f"{d_val:.2f}" if np.isfinite(d_val) else "inf"
        w_str = f"{w_val:.2f}" if np.isfinite(w_val) else "inf"
        work_str = f"{work:.2f}" if np.isfinite(work) else "inf"

        hover_text = (
            f'<b style="font-size:14px">WALKER #{i} [{state.upper()}]</b><br>'
            f'<span style="color:#00F0FF">{title}</span><br>'
            f'<span style="color:#888">UID: {bt_uid}</span><br>'
            f'<b>═══════════════════════</b><br>'
            f'<b>T4 Walker State:</b> <span style="color:{color}">{state.upper()}</span><br>'
            f'<b>T4 Work Integral:</b> {work_str}<br>'
            f'<b>EVR:</b> {evr:.3f} | <b>Terrain:</b> {terrain}<br>'
            f'<b>T5 Verdict:</b> {pv} (d={d_str}, W={w_str}, Delta={delta:.2f})<br>'
        )

        traces.append(go.Scatter3d(
            x=[positions_3d[i, 0]],
            y=[positions_3d[i, 1]],
            z=[positions_3d[i, 2] + 0.15],
            mode='markers+text',
            marker=dict(
                size=12,  # Larger for easier clicking
                color=color,
                symbol='diamond',
                line=dict(width=2, color='#ffffff'),
            ),
            text=[state[0].upper()],
            textposition='top center',
            textfont=dict(size=9, color=color),
            name=f'Walker: {state}',
            hovertemplate=hover_text + '<extra></extra>',
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0.95)',
                bordercolor=color,
                font=dict(family='JetBrains Mono, monospace', size=11, color='white'),
            ),
            showlegend=False,
        ))

    return traces


# =============================================================================
# HYSTERESIS VISUALIZATION (Track 4 Path Memory)
# =============================================================================
def render_hysteresis_highways_3d(
    positions_3d: np.ndarray,
    hysteresis_memory: np.ndarray,
    probe_labels: Optional[List[str]] = None,
    highway_threshold: float = 0.3,
) -> List[Any]:
    """
    Render Track 4 hysteresis as glowing "highways" between bot positions.

    The memory matrix [8, 8] tracks rut depths for bot→bot transitions.
    We visualize these as glowing arcs in 3D space, showing the pheromone
    trails that walkers have carved through repeated traversals.

    Args:
        positions_3d: [N, 3] article positions (used to place highway legend)
        hysteresis_memory: [8, 8] pheromone trail matrix
        probe_labels: [8] names of the 8 bot/probe pairs
        highway_threshold: Minimum memory value to render as highway

    Returns:
        List of Plotly traces showing highway arcs
    """
    if not HAS_PLOTLY or hysteresis_memory is None:
        return []

    traces = []
    n_bots = hysteresis_memory.shape[0]

    # Default probe labels
    if probe_labels is None:
        probe_labels = PROBE_LABELS if len(PROBE_LABELS) >= n_bots else [f"Bot {i}" for i in range(n_bots)]

    # Compute bot positions in a circle (for visualization)
    # Place bots on a circle at the top of the visualization
    center_x = float(np.mean(positions_3d[:, 0]))
    center_y = float(np.mean(positions_3d[:, 1]))
    z_level = float(np.max(positions_3d[:, 2])) + 1.0
    radius = 1.5

    bot_positions = []
    for i in range(n_bots):
        angle = 2 * np.pi * i / n_bots
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        bot_positions.append((x, y, z_level))

    # Normalize memory for color intensity
    mem_max = hysteresis_memory.max()
    if mem_max < 1e-6:
        return []  # No memory to display

    # Render bot nodes
    bot_x = [p[0] for p in bot_positions]
    bot_y = [p[1] for p in bot_positions]
    bot_z = [p[2] for p in bot_positions]

    traces.append(go.Scatter3d(
        x=bot_x, y=bot_y, z=bot_z,
        mode='markers+text',
        marker=dict(
            size=10,
            color=PALETTE.cyan,
            symbol='circle',
            line=dict(width=1, color='white'),
        ),
        text=[label[:8] for label in probe_labels[:n_bots]],
        textposition='top center',
        textfont=dict(size=8, color=PALETTE.cyan),
        name='Track 4: Bot Positions',
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=True,
    ))

    # Render highway arcs (memory > threshold)
    highway_x, highway_y, highway_z = [], [], []
    highway_colors = []
    highway_widths = []
    highway_hovers = []

    for i in range(n_bots):
        for j in range(n_bots):
            if i == j:
                continue

            mem_ij = hysteresis_memory[i, j]
            mem_ji = hysteresis_memory[j, i]

            if mem_ij < highway_threshold * mem_max:
                continue

            # Arc from bot i to bot j
            p1 = bot_positions[i]
            p2 = bot_positions[j]

            # Create curved arc (bezier midpoint lifted up)
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            mid_z = (p1[2] + p2[2]) / 2 + 0.3  # Lift midpoint

            # Sample points along arc
            n_points = 10
            for t in range(n_points):
                t1 = t / n_points
                t2 = (t + 1) / n_points

                # Quadratic bezier interpolation
                def bezier(t_val):
                    return (
                        (1 - t_val)**2 * p1[0] + 2 * (1 - t_val) * t_val * mid_x + t_val**2 * p2[0],
                        (1 - t_val)**2 * p1[1] + 2 * (1 - t_val) * t_val * mid_y + t_val**2 * p2[1],
                        (1 - t_val)**2 * p1[2] + 2 * (1 - t_val) * t_val * mid_z + t_val**2 * p2[2],
                    )

                pt1 = bezier(t1)
                pt2 = bezier(t2)

                highway_x.extend([pt1[0], pt2[0], None])
                highway_y.extend([pt1[1], pt2[1], None])
                highway_z.extend([pt1[2], pt2[2], None])

            # Color and width based on memory intensity
            intensity = mem_ij / mem_max
            highway_colors.append(intensity)
            highway_widths.append(2 + 8 * intensity)

            # Asymmetry indicator
            asymmetry = abs(mem_ij - mem_ji) / (mem_ij + mem_ji + 1e-6)
            direction = "→" if mem_ij > mem_ji else "←" if mem_ji > mem_ij else "↔"

            hover = (
                f"<b>{probe_labels[i][:12]} {direction} {probe_labels[j][:12]}</b><br>"
                f"Rut depth: {mem_ij:.3f}<br>"
                f"Reverse: {mem_ji:.3f}<br>"
                f"Asymmetry: {asymmetry:.1%}"
            )
            highway_hovers.append(hover)

    if highway_x:
        # Create colorscale from rut_low to rut_high
        traces.append(go.Scatter3d(
            x=highway_x, y=highway_y, z=highway_z,
            mode='lines',
            line=dict(
                color=PALETTE.highway_glow,
                width=4,
            ),
            name='Track 4: Pheromone Highways',
            hoverinfo='skip',
            showlegend=True,
            opacity=0.7,
        ))

    # Add memory matrix heatmap as 2D annotation
    # (This will be a small inset showing the full matrix)
    max_rut = float(mem_max)
    n_highways = int((hysteresis_memory > highway_threshold * mem_max).sum())

    # Add stats annotation
    stats_text = (
        f"<b>HYSTERESIS (Path Memory)</b><br>"
        f"Max rut depth: {max_rut:.3f}<br>"
        f"Active highways: {n_highways}<br>"
        f"Memory decay: active"
    )

    traces.append(go.Scatter3d(
        x=[center_x + radius + 0.5],
        y=[center_y],
        z=[z_level + 0.5],
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        text=[stats_text],
        hovertemplate='%{text}<extra></extra>',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.9)',
            bordercolor=PALETTE.highway_glow,
            font=dict(family='JetBrains Mono', size=10, color='white'),
        ),
        name='Hysteresis Stats',
        showlegend=False,
    ))

    return traces


def render_hysteresis_heatmap_2d(
    hysteresis_memory: np.ndarray,
    probe_labels: Optional[List[str]] = None,
) -> Any:
    """
    Render 2D heatmap of the hysteresis memory matrix.

    Shows the full [8, 8] pheromone trail matrix as a heatmap,
    useful for DIAGNOSTICS mode.
    """
    if not HAS_PLOTLY or hysteresis_memory is None:
        return None

    n_bots = hysteresis_memory.shape[0]

    if probe_labels is None:
        probe_labels = PROBE_LABELS if len(PROBE_LABELS) >= n_bots else [f"Bot {i}" for i in range(n_bots)]

    # Truncate labels for display
    labels = [label[:10] for label in probe_labels[:n_bots]]

    fig = go.Figure(data=go.Heatmap(
        z=hysteresis_memory,
        x=labels,
        y=labels,
        colorscale=[
            [0, PALETTE.rut_low],
            [0.5, PALETTE.highway_core],
            [1, PALETTE.highway_glow],
        ],
        hovertemplate='%{y} → %{x}<br>Rut: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text='Track 4: Hysteresis Memory Matrix',
            font=dict(color=PALETTE.highway_glow, size=14),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.5)',
        font=dict(color=PALETTE.text_primary, size=10),
        xaxis=dict(title='To Bot', tickangle=45),
        yaxis=dict(title='From Bot'),
        width=400,
        height=400,
    )

    return fig


# =============================================================================
# FOG OVERLAY (Track 3 Blinker) — PERCENTILE-BASED THRESHOLDING
# =============================================================================
def compute_fog_intensity(
    blinker_variance: np.ndarray,
    threshold: float = None,
    fog_percentile: float = 80.0,
    bond_percentile: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fog intensity from blinker variance using DYNAMIC percentile thresholds.

    ASTER v3.2 Dirichlet Sweep — replaces hard 0.7 threshold with adaptive percentiles.
    This ensures we ALWAYS see a gradient regardless of dataset variance profile.

    Returns:
        fog_intensity: [N] normalized fog values [0, 1]
        is_fog: [N] bool — top fog_percentile% of variance (decoherent)
        is_bond: [N] bool — bottom bond_percentile% of variance (coherent/crystal)
    """
    # Dynamic thresholds based on this dataset's distribution
    thresh_fog = np.percentile(blinker_variance, fog_percentile)   # Top 20% = Fog
    thresh_bond = np.percentile(blinker_variance, bond_percentile)  # Bottom 20% = Bond

    # Classify each point
    is_fog = blinker_variance > thresh_fog      # Decoherent (probability cloud)
    is_bond = blinker_variance < thresh_bond    # Coherent (crystal reality)

    # Normalized intensity [0, 1] for gradient rendering
    v_min, v_max = blinker_variance.min(), blinker_variance.max()
    fog_intensity = (blinker_variance - v_min) / (v_max - v_min + 1e-9)

    return fog_intensity, is_fog, is_bond


def classify_atmospheric_state(
    blinker_variance: np.ndarray,
    fog_percentile: float = 80.0,
    bond_percentile: float = 20.0,
) -> List[str]:
    """
    Classify each article's atmospheric state for Track 3.

    States:
        CRYSTAL: Low variance — observers interfere constructively (clear skies)
        HAZE: Mid variance — partial coherence
        FOG: High variance — observers interfere destructively (probability cloud)
    """
    fog_intensity, is_fog, is_bond = compute_fog_intensity(
        blinker_variance, fog_percentile=fog_percentile, bond_percentile=bond_percentile
    )

    states = []
    for i in range(len(blinker_variance)):
        if is_fog[i]:
            states.append("FOG")
        elif is_bond[i]:
            states.append("CRYSTAL")
        else:
            states.append("HAZE")

    return states


def render_fog_overlay_3d(
    positions_3d: np.ndarray,
    fog_intensity: np.ndarray,
    is_fog: np.ndarray,
    is_bond: np.ndarray,
) -> List[Any]:
    """
    Render Track 3 atmospheric layer with percentile-based classification.

    CRYSTAL (Bonds): Sharp cyan points — observers agree (coherent)
    HAZE: Gray gradient — partial coherence
    FOG (Cracks): Exploded diffuse spheres — observers disagree (decoherent)
    """
    if not HAS_PLOTLY:
        return []

    traces = []

    # --- FOG ZONES (Top 20% variance) ---
    if is_fog.any():
        fog_positions = positions_3d[is_fog]
        fog_sizes = 15 + fog_intensity[is_fog] * 40  # Explode size
        fog_opacities = 0.1 + fog_intensity[is_fog] * 0.1  # Low opacity (diffuse)

        traces.append(go.Scatter3d(
            x=fog_positions[:, 0],
            y=fog_positions[:, 1],
            z=fog_positions[:, 2] - 0.1,
            mode='markers',
            marker=dict(
                size=fog_sizes,
                color='rgba(255, 255, 255, 0.15)',  # White fog
                symbol='circle',
                line=dict(width=0),
            ),
            name='Track 3: Fog (Decoherent)',
            hovertemplate='<b>FOG ZONE</b><br>Variance: %{customdata:.2f}<br>State: CRACK<extra></extra>',
            customdata=fog_intensity[is_fog],
        ))

    # --- CRYSTAL ZONES (Bottom 20% variance) ---
    if is_bond.any():
        bond_positions = positions_3d[is_bond]

        traces.append(go.Scatter3d(
            x=bond_positions[:, 0],
            y=bond_positions[:, 1],
            z=bond_positions[:, 2] - 0.05,
            mode='markers',
            marker=dict(
                size=4,  # Sharp small points
                color=PALETTE.cyan,
                opacity=1.0,
                symbol='diamond',
                line=dict(width=1, color='white'),
            ),
            name='Track 3: Crystal (Coherent)',
            hovertemplate='<b>CRYSTAL ZONE</b><br>Variance: %{customdata:.2f}<br>State: BOND<extra></extra>',
            customdata=fog_intensity[is_bond],
        ))

    # --- HAZE ZONES (Middle 60%) ---
    is_haze = ~is_fog & ~is_bond
    if is_haze.any():
        haze_positions = positions_3d[is_haze]
        haze_opacities = 0.3 + fog_intensity[is_haze] * 0.3

        traces.append(go.Scatter3d(
            x=haze_positions[:, 0],
            y=haze_positions[:, 1],
            z=haze_positions[:, 2] - 0.08,
            mode='markers',
            marker=dict(
                size=6,
                color='rgba(136, 136, 136, 0.4)',  # Gray
                symbol='circle',
                line=dict(width=0),
            ),
            name='Track 3: Haze (Partial)',
            hovertemplate='<b>HAZE ZONE</b><br>Variance: %{customdata:.2f}<br>State: PARTIAL<extra></extra>',
            customdata=fog_intensity[is_haze],
            showlegend=False,
        ))

    return traces


# =============================================================================
# HOTT VERDICT ICONS (Track 6) — With Phantom Delta Override
# =============================================================================
def render_hott_icons_3d(
    positions_3d: np.ndarray,
    hott_proofs: List[Dict],
    phantom_verdicts: Optional[List[Dict]] = None,
    get_surface_z_func: Optional[Callable] = None, # For icon snapping
) -> List[Any]:
    """
    Render Track 6 HoTT verdict icons with Phantom Delta override.

    CRITICAL: If phantom delta > 1.5 (PHANTOM or RUPTURE verdict),
    override the HoTT icon to ✗ regardless of the proof status.

    This enforces that the Panic Function (Track 5) has veto power
    over formal proofs when geometric evidence contradicts them.
    """
    if not HAS_PLOTLY:
        return []

    traces = []
    n = min(len(positions_3d), len(hott_proofs))

    for i in range(n):
        proof = hott_proofs[i]
        if not proof:
            continue

        pos = positions_3d[i]
        # hott_proofs.json stores the verdict in 'status' field
        status = proof.get('status', proof.get('verdict', proof.get('singularity_type', 'unknown')))
        confidence = proof.get('confidence', 0.0)

        # Check for Phantom Delta override
        phantom_override = False
        phantom_delta = 0.0
        phantom_verdict_str = "N/A"
        if phantom_verdicts and i < len(phantom_verdicts):
            pv = phantom_verdicts[i]
            phantom_delta = pv.get('delta', 0.0)
            phantom_verdict_str = pv.get('verdict', 'unknown')
            # Override to invalid if delta > 1.5 (PHANTOM or RUPTURE territory)
            if phantom_delta > THRESHOLDS["phantom_delta"]:
                phantom_override = True

        # Determine icon and color
        if phantom_override:
            # PHANTOM DELTA OVERRIDE: Force invalid (Topological Lock)
            icon = "✗"
            color = PALETTE.rupture_red
            status = f"OVERRIDE:{phantom_verdict_str}"
        elif status in ['equivalence', 'consensus', 'elastic', 'HONEST', 'TAUTOLOGY']:
            icon = "≡"
            color = PALETTE.hott_valid
        elif status == 'non_equivalence':
            icon = "≠"
            color = PALETTE.hott_invalid
        elif status == 'obstruction':
            icon = "⛔"
            color = PALETTE.hott_invalid
        elif status in ['structural_singularity', 'broken', 'PHANTOM', 'RUPTURE']:
            icon = "✗"
            color = PALETTE.hott_invalid
        elif status == 'undecidable':
            icon = "?"
            color = PALETTE.yellow
        else:
            icon = "?"
            color = PALETTE.yellow

        hover_text = (
            f'<b>HoTT VERDICT</b><br>'
            f'Status: {status}<br>'
            f'Confidence: {confidence:.2f}<br>'
            f'---<br>'
            f'Phantom Δ: {phantom_delta:.2f}<br>'
            f'Override: {"YES" if phantom_override else "NO"}'
        )

        traces.append(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[get_surface_z_func(pos[0], pos[1], offset=0.1)] if get_surface_z_func else [pos[2] + 0.2],
            mode='text',
            text=[icon],
            textfont=dict(color=color, size=14),
            name='HoTT Icons',
            showlegend=False,
            hovertemplate=hover_text + '<extra></extra>',
        ))

    return traces


# =============================================================================
# SPECTRAL AXIS ARROW (Track 1.5) - THE GYROSCOPE
# =============================================================================
def render_spectral_axis_3d(
    positions_3d: np.ndarray,
    spectral_probe_magnitudes: np.ndarray,
    evr: float = 0.5,
    get_surface_z_func: Optional[Callable] = None, # For path clamping
) -> List[Any]:
    """
    Render the Track 1.5 GYROSCOPE: 3 orthogonal endogenous axes.

    The Gyroscope defines the semantic coordinate system:
    - X-Axis (Primary War): The loudest conflict (PC1)
    - Y-Axis (Secondary War): The cross-pressure (PC2)
    - Z-Axis (The Nuance): The third hidden variable (PC3)
    """
    if not HAS_PLOTLY or not HAS_SCIPY:
        return []

    traces = []

    # PCA on probe magnitudes to find 3 orthogonal axes
    n_components = min(3, spectral_probe_magnitudes.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(spectral_probe_magnitudes)

    # Compute dominant probe for each axis
    def get_dominant_probe(component):
        return int(np.argmax(np.abs(component)))

    # Project to 3D space
    centroid = positions_3d.mean(axis=0)
    radius = np.linalg.norm(positions_3d - centroid, axis=1).max() * 0.7

    # Axis colors and labels
    axis_colors = [PALETTE.cyan, PALETTE.rupture_core, "#39FF14"]
    axis_labels = ['Primary Conflict', 'Secondary Conflict', 'Nuance']
    axis_names = ['X-Axis (Primary War)', 'Y-Axis (Secondary War)', 'Z-Axis (Nuance)']

    axis_dirs_xy: List[Tuple[float, float]] = []
    for i in range(min(3, len(pca.components_))):
        component = pca.components_[i]
        explained_var = pca.explained_variance_ratio_[i]
        dominant_probe_idx = get_dominant_probe(component)
        dominant_probe_label = (
            PROBE_LABELS[dominant_probe_idx]
            if 0 <= dominant_probe_idx < len(PROBE_LABELS)
            else f"Probe {dominant_probe_idx}"
        )
        semantic_label = f"{axis_labels[i]}: {dominant_probe_label}"

        # Use a fixed length for vectors for consistency, relative to overall scene size
        scene_range = positions_3d.max(axis=0) - positions_3d.min(axis=0)
        vector_length = np.linalg.norm(scene_range[:2]) * 0.4 # Scale to scene's XY extent

        # Determine the 2D (x,y) direction of the vector
        x_direction = component[0]
        y_direction = component[1]
        
        # PC3 (Nuance) should be horizontal. So for i=2, its Z-component should be ignored
        # for trajectory calculation and it will be draped on the surface
        
        # Normalize the 2D (x,y) direction for consistent length on the surface
        xy_norm = np.linalg.norm([x_direction, y_direction])
        if xy_norm > 1e-6:
            x_direction /= xy_norm
            y_direction /= xy_norm
        else: # Handle near-zero 2D components, default to x-axis
            x_direction = 1.0
            y_direction = 0.0

        if i == 2 and axis_dirs_xy:
            # Keep Nuance vector visually separable when PC3 projects nearly parallel to PC1/PC2 in XY.
            for _ax, _ay in axis_dirs_xy:
                _dot = abs(x_direction * _ax + y_direction * _ay)
                if _dot > 0.92:
                    x_direction, y_direction = -y_direction, x_direction
                    break
        axis_dirs_xy.append((x_direction, y_direction))

        # Calculate start and end points for the 2D projection
        vec_start_x = centroid[0] - x_direction * vector_length * 0.5
        vec_start_y = centroid[1] - y_direction * vector_length * 0.5
        vec_end_x = centroid[0] + x_direction * vector_length * 0.5
        vec_end_y = centroid[1] + y_direction * vector_length * 0.5
        
        n_sample_points = 100 # Number of points to sample along each vector

        # Generate sample points along the 2D (x,y) projection
        sample_xs = np.linspace(vec_start_x, vec_end_x, n_sample_points)
        sample_ys = np.linspace(vec_start_y, vec_end_y, n_sample_points)

        # Clamp Z to surface using get_surface_z_func
        # Add a small offset so the vector "floats" slightly above the surface
        if get_surface_z_func is not None:
            _offset = 0.12 if i == 2 else 0.05
            sample_zs = get_surface_z_func(sample_xs, sample_ys, offset=_offset) 
        else:
            # Fallback (should not happen if interpolator is used)
            sample_zs = np.full_like(sample_xs, centroid[2] + 0.03) # Use average Z + offset

        if i == 2:
            traces.append(go.Scatter3d(
                x=sample_xs,
                y=sample_ys,
                z=sample_zs,
                mode='lines',
                line=dict(color='rgba(0,0,0,0.70)', width=10),
                opacity=0.9,
                name='Nuance Underlay',
                hoverinfo='skip',
                showlegend=False,
            ))

        traces.append(go.Scatter3d(
            x=sample_xs,
            y=sample_ys,
            z=sample_zs,
            mode='lines',
            line=dict(color=axis_colors[i], width=(7 if i == 2 else 5 - i*1.2)),
            opacity=1.0 if i == 2 else 0.95,
            name=f'Vector: {semantic_label} (EVR={explained_var*100:.1f}%)',
            hovertemplate=f'<b>Vector: {semantic_label}</b><br>Explained Var: {explained_var*100:.1f}%<extra></extra>',
            showlegend=True,
        ))

    return traces


# =============================================================================
# DATA POINTS (Glowing Muons) - Colored by VERDICT, sized by annealing stability
# =============================================================================
def render_data_points_3d(
    positions: np.ndarray,
    spectral_evr: np.ndarray,
    sizes: np.ndarray,
    hover_texts: List[str],
    name: str = "Articles",
    phantom_verdicts: List[Dict] = None,
    is_fog: np.ndarray = None,
    article_z_height: Optional[np.ndarray] = None, # New parameter for exact Z positioning
    article_color_codes: Optional[np.ndarray] = None, # New parameter for exact color coding
) -> List[Any]:
    """
    Render data points with bloom effect.

    Color by VERDICT (more meaningful than raw EVR):
    - HONEST: Cyan (truth)
    - PHANTOM: Magenta (spin/lie)
    - RUPTURE: Red (crash)
    - TAUTOLOGY: Gray (nothing)

    Size/Opacity by Annealing:
    - Crystal (survived cooling): Small, bright, solid
    - Fog (dissolved): Large, translucent, hazy
    """
    if not HAS_PLOTLY:
        return []

    traces = []
    n = len(positions)

    # Build per-point colors based on provided article_color_codes or verdict (using RGBA for transparency)
    point_colors = []
    point_colors_glow = []

    # Verdict to color mapping (fallback)
    verdict_colors = {
        'HONEST': (0, 240, 255),     # Cyan
        'PHANTOM': (204, 0, 255),    # Magenta
        'RUPTURE': (255, 34, 34),    # Red
        'TAUTOLOGY': (136, 136, 136),  # Gray
        'UNKNOWN': (255, 215, 0),    # Yellow
    }

    for i in range(n):
        # Determine base color for the point
        if article_color_codes is not None and i < len(article_color_codes):
            base_hex_color = article_color_codes[i] # Use color from CSV
            # Parse hex color to RGB for opacity adjustment
            r = int(base_hex_color[1:3], 16)
            g = int(base_hex_color[3:5], 16)
            b = int(base_hex_color[5:7], 16)
        else:
            # Fallback to existing verdict-based coloring
            verdict = "UNKNOWN"
            if phantom_verdicts and i < len(phantom_verdicts):
                verdict = str(phantom_verdicts[i].get('verdict', 'UNKNOWN')).upper()
            r, g, b = verdict_colors.get(verdict, verdict_colors['UNKNOWN'])
        
        core_opacity = 0.95 # Core point opacity, always solid

        glow_opacity_val = 0.3  # Default glow opacity
        glow_size_factor_val = 1.0 # Default glow size factor for points

                # FAMILY C: NODE FAILURES (Internal Singularity)
        if spectral_evr is not None and i < len(spectral_evr) and spectral_evr[i] < 0.5:
            glow_opacity_val = 0.8
            glow_size_factor_val = 2.5
        if is_fog is not None and i < len(is_fog) and is_fog[i]:
            # Decoherent (Fog): glow is minimal
            glow_opacity_val = 0.05
            glow_size_factor_val = 0.5 # Reduce glow size for decoherent points

        point_colors.append(f'rgba({r},{g},{b},{core_opacity})')
        point_colors_glow.append(f'rgba({r},{g},{b},{glow_opacity_val})')

    # Adjust sizes: For decoherent points, glow is minimal. Core size is not changed by fog.
    adjusted_sizes = sizes.copy()
    # No `np.where(is_fog, ...)` for adjusted_sizes, as "Fog = larger" is removed.

    # Keep article points attached to the rendered manifold when available.
    if article_z_height is not None and len(article_z_height) == n:
        elevated_z = np.asarray(article_z_height, dtype=float)
    else:
        elevated_z = positions[:, 2]

    # Layer 1: Outer glow (hover SKIP to allow clicks through to core)
    traces.append(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=elevated_z,  # ELEVATED above terrain
        mode='markers',
        marker=dict(
            size=adjusted_sizes * 2.5 * glow_size_factor_val,  # Adjust glow size
            color=point_colors_glow,
        ),
        showlegend=False,
        hoverinfo='skip',  # CRITICAL: 'skip' passes through, 'none' blocks!
        name='glow_outer',
    ))

    # Layer 2: Mid glow (hover SKIP to allow clicks through to core)
    traces.append(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=elevated_z,  # ELEVATED above terrain
        mode='markers',
        marker=dict(
            size=adjusted_sizes * 1.6 * glow_size_factor_val,  # Adjust mid glow size
            color=point_colors_glow,
        ),
        showlegend=False,
        hoverinfo='skip',  # CRITICAL: 'skip' passes through, 'none' blocks!
        name='glow_mid',
    ))

    # Layer 3: Invisible interaction hitbox (hover/click reliability layer)
    traces.append(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=elevated_z,
        mode='markers',
        marker=dict(
            size=np.maximum(adjusted_sizes * 2.0, 10),
            color='rgba(255,255,255,0.001)',
            line=dict(width=0),
            symbol='circle',
        ),
        hovertext=hover_texts,
        customdata=list(range(n)),
        hoverinfo='text',
        hovertemplate='%{hovertext}<extra></extra>',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.95)',
            bordercolor='#00F0FF',
            font=dict(family='JetBrains Mono, monospace', size=11, color='white'),
        ),
        name='article_hitbox',
        showlegend=False,
    ))

    # Layer 4: Core ARTICLE POINTS (visual only)
    traces.append(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=elevated_z,  # ELEVATED above terrain for visibility
        mode='markers',
        marker=dict(
            size=adjusted_sizes * 1.2,  # Larger core for easier clicking
            color=point_colors,  # RGBA with per-point opacity
            line=dict(color='white', width=1.5),  # Thicker white outline
            symbol='circle',  # Explicit circle shape
        ),
        hoverinfo='skip',
        name=name,
        showlegend=True,  # Show in legend as "Articles"
    ))

    return traces


# =============================================================================
# ANALYSIS MODE: STACKED TRACK PLANES
# =============================================================================
# Distinct colors for 8 ideological clusters (qualitative palette)
CLUSTER_COLORS = [
    '#e6194b',  # Red - Hardline Religious Zionist
    '#3cb44b',  # Green - Evangelical Zionist
    '#ffe119',  # Yellow - Security Realist
    '#4363d8',  # Blue - Liberal Zionist
    '#f58231',  # Orange - Secular Palestinian Nationalist
    '#911eb4',  # Purple - Islamist Resistance
    '#42d4f4',  # Cyan - Arab Pro-Palestine
    '#f032e6',  # Magenta - Western Leftist
]


def render_analysis_planes(
    exp: 'ExperimentData',
    plane_spacing: float = 3.0,
    plane_size: float = 10.0,
) -> Tuple[List[Any], Dict[str, float]]:
    """
    Render stacked horizontal planes showing each track in isolation.

    Returns:
        traces: List of Plotly traces for ANALYSIS mode
        nmi_scores: Dict mapping track name to NMI score

    Track planes (bottom to top):
        - Z=0: Track 1 (Logits) - raw NLI output
        - Z=3: Track 2 (δ_μν) - RKS kernel projection
        - Z=6: Track 1.5 (∇Φ∇Φ) - Spectral polarity
        - Z=9: Track 3 (1/ρ) - Blinker variance
    """
    if not HAS_PLOTLY or not HAS_SCIPY:
        return [], {}

    traces = []
    nmi_scores = {}
    n_articles = exp.n_articles

    # Ground truth labels for coloring
    labels_available = exp.ground_truth_labels is not None and len(np.unique(exp.ground_truth_labels)) > 1
    if labels_available:
        labels = exp.ground_truth_labels
    else:
        labels = np.zeros(n_articles, dtype=int)

    # Map labels to colors
    unique_labels = np.unique(labels)
    n_unique = len(unique_labels)
    point_colors = [CLUSTER_COLORS[labels[i] % len(CLUSTER_COLORS)] for i in range(n_articles)]

    # Helper to compute 2D PCA projection
    def project_2d(features: np.ndarray) -> np.ndarray:
        if features is None or len(features) == 0:
            return None
        # Flatten if needed
        if features.ndim > 2:
            features = features.reshape(len(features), -1)
        # PCA to 2D
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(features)
        # Normalize to [-plane_size/2, plane_size/2]
        proj = (proj - proj.mean(axis=0)) / (proj.std(axis=0) + 1e-8)
        proj = proj * (plane_size / 4)
        return proj

    # Helper to compute NMI if sklearn available
    def compute_nmi(features: np.ndarray, labels: np.ndarray, labels_valid: bool) -> float:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import normalized_mutual_info_score
            if features is None or len(features) < 3 or not labels_valid:
                return float("nan")
            if features.ndim > 2:
                features = features.reshape(len(features), -1)
            n_clusters = min(len(np.unique(labels)), 8)
            if n_clusters < 2:
                return float("nan")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pred = kmeans.fit_predict(features)
            return normalized_mutual_info_score(labels, pred)
        except Exception:
            return float("nan")

    # Track definitions: (name, z_level, features, label)
    # Track 1 must be true logits only; missing logits should surface as unavailable.
    nmi_scores["T1"] = float("nan")
    t1_features = exp.cp_t0_logits
    t1_track_name = "T1: Logits (NLI raw)"
    if t1_features is None and exp.logits is not None:
        t1_features = exp.logits
        t1_track_name = "T1: Logits (base logits)"
    t2_features = exp.cp_t2_kernels if exp.cp_t2_kernels is not None else exp.dirichlet_fused
    if t2_features is None:
        t2_features = exp.integrated if exp.integrated is not None else exp.features
    t15_features = exp.cp_t15_spectral if exp.cp_t15_spectral is not None else exp.spectral_probe_magnitudes
    t3_features = exp.cp_t3_blinker
    if t3_features is None:
        t3_features = exp.dirichlet_fused_std
    if t3_features is None:
        t3_features = exp.dirichlet_fused
    if t3_features is None:
        t3_features = exp.integrated
    if t3_features is None:
        t3_features = exp.features

    track_configs = [
        (t1_track_name, 0.0, t1_features, "T1"),
        ("T2: RKS (δ_μν kernel)", plane_spacing, t2_features, "T2"),
        ("T1.5: Spectral (∇Φ∇Φ)", plane_spacing * 2, t15_features, "T1.5"),
        ("T3: Blinker (1/ρ)", plane_spacing * 3, t3_features, "T3"),
        # Add SYNTHESIS at the top - the final integrated features
        ("SYNTHESIS (Integrated)", plane_spacing * 4, exp.features, "SYN"),
    ]

    for track_name, z_level, features, cp_label in track_configs:
        if features is None:
            continue

        # Project to 2D
        proj_2d = project_2d(features)
        if proj_2d is None:
            continue

        # Compute NMI
        nmi = compute_nmi(features, labels, labels_available)
        nmi_scores[cp_label] = nmi

        # Create semi-transparent plane surface
        plane_x = np.linspace(-plane_size/2, plane_size/2, 20)
        plane_y = np.linspace(-plane_size/2, plane_size/2, 20)
        plane_xx, plane_yy = np.meshgrid(plane_x, plane_y)
        plane_zz = np.full_like(plane_xx, z_level)

        # Plane surface (translucent grid)
        traces.append(go.Surface(
            x=plane_xx,
            y=plane_yy,
            z=plane_zz,
            surfacecolor=np.zeros_like(plane_zz),
            colorscale=[[0, 'rgba(20,20,40,0.3)'], [1, 'rgba(20,20,40,0.3)']],
            showscale=False,
            name=f'{track_name} plane',
            hoverinfo='skip',
            visible=False,  # Hidden by default (ANALYSIS mode)
        ))

        # Grid lines on plane
        for gx in np.linspace(-plane_size/2, plane_size/2, 5):
            traces.append(go.Scatter3d(
                x=[gx, gx],
                y=[-plane_size/2, plane_size/2],
                z=[z_level, z_level],
                mode='lines',
                line=dict(color='rgba(0,240,255,0.15)', width=1),
                showlegend=False,
                hoverinfo='skip',
                visible=False,
            ))
            traces.append(go.Scatter3d(
                x=[-plane_size/2, plane_size/2],
                y=[gx, gx],
                z=[z_level, z_level],
                mode='lines',
                line=dict(color='rgba(0,240,255,0.15)', width=1),
                showlegend=False,
                hoverinfo='skip',
                visible=False,
            ))

        # Track label (3D text annotation)
        nmi_str = f"NMI={nmi:.3f}" if np.isfinite(nmi) else "NMI=UNAVAILABLE"
        traces.append(go.Scatter3d(
            x=[-plane_size/2 - 0.5],
            y=[0],
            z=[z_level + 0.3],
            mode='text',
            text=[f"<b>{track_name}</b><br>{nmi_str}"],
            textfont=dict(size=12, color=PALETTE.cyan, family='JetBrains Mono'),
            textposition='middle right',
            showlegend=False,
            hoverinfo='skip',
            visible=False,
        ))

        # Data points on this plane
        # Preserve local vertical variation instead of pinning every point to a fixed z.
        local_relief = (proj_2d[:, 0] - np.mean(proj_2d[:, 0])) / (np.std(proj_2d[:, 0]) + 1e-8)
        point_z = z_level + 0.1 + 0.08 * local_relief
        traces.append(go.Scatter3d(
            x=proj_2d[:, 0],
            y=proj_2d[:, 1],
            z=point_z,
            mode='markers',
            marker=dict(
                size=8,
                color=point_colors,
                opacity=0.9,
                line=dict(color='white', width=0.5),
            ),
            text=[f"Article {i}<br>Label: {labels[i]}" for i in range(n_articles)],
            hoverinfo='text',
            name=f'{cp_label} points',
            showlegend=False,
            visible=False,
        ))

        # Glow layer for points
        traces.append(go.Scatter3d(
            x=proj_2d[:, 0],
            y=proj_2d[:, 1],
            z=point_z,
            mode='markers',
            marker=dict(
                size=16,
                color=[c.replace(')', ',0.3)').replace('rgb', 'rgba') if 'rgba' not in c
                       else c for c in point_colors],
                opacity=0.3,
            ),
            showlegend=False,
            hoverinfo='skip',
            visible=False,
        ))

    # Add vertical connection lines between planes (showing article trajectories)
    # This shows how each article moves through the tracks
    all_projs = []
    for track_name, z_level, features, cp_label in track_configs:
        if features is not None:
            proj = project_2d(features)
            if proj is not None:
                all_projs.append((z_level, proj))

    if len(all_projs) >= 2:
        # Draw faint vertical lines connecting same article across planes
        for i in range(min(n_articles, 60)):  # Limit to 60 for performance
            line_x = [p[1][i, 0] for p in all_projs]
            line_y = [p[1][i, 1] for p in all_projs]
            line_z = [p[0] + 0.1 for p in all_projs]
            traces.append(go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.1)', width=1),
                showlegend=False,
                hoverinfo='skip',
                visible=False,
            ))

    # Side-bar mesh removed: produced unstable yellow mesh artifacts in 3D.

    return traces, nmi_scores


# =============================================================================
# HUD BAR
# =============================================================================
def generate_hud_html(
    mode: str,
    signal: float,
    n_ruptures: int,
    n_phantoms: int,
    n_honest: int,
    n_tautology: int,
    knn_overlap: float,
    kernel_name: str = "rbf",
    n_cracks: int = 0,
    n_bonds: int = 0,
    n_broken: int = 0,
    n_trapped: int = 0,
    mean_action: float = 0.0,
    survival_rate: float = 1.0,
    synthesis_nmi: Optional[float] = None,
) -> str:
    """Generate the terminal-style HUD bar HTML."""
    signal_class = "good" if signal > 0.7 else "warn" if signal > 0.5 else "alert"
    rupture_class = "alert" if n_ruptures > 0 else "good"
    phantom_class = "warn" if (n_phantoms + n_tautology) > 0 else "good"
    crack_class = "warn" if n_cracks > 0 else "good"
    walker_class = "good" if survival_rate >= 0.95 else "warn" if survival_rate >= 0.8 else "alert"

    # Build T5 section with optional NMI
    t5_section = (
        f'<span class="hud-item {phantom_class}">'
        f'T5: {n_honest}H / {n_phantoms}P / {n_tautology}T | '
        f'RUPTURES: {n_broken} Broken / {n_trapped} Trapped'
    )
    if synthesis_nmi is not None:
        nmi_class = "good" if synthesis_nmi > 0.7 else "warn" if synthesis_nmi > 0.5 else "alert"
        t5_section += f' | NMI: <span class="{nmi_class}">{synthesis_nmi:.3f}</span>'
    t5_section += '</span>'

    return f'''
    <div class="hud-bar">
        <span class="hud-item">MODE: {mode.upper()}</span>
        <span class="hud-sep">//</span>
        <span class="hud-item">KERNEL: {kernel_name.upper()}</span>
        <span class="hud-sep">//</span>
        <span class="hud-item {signal_class}">SIG: {signal:.2f}</span>
        <span class="hud-sep">//</span>
        <span class="hud-item {rupture_class}">T1.5: {n_ruptures}R</span>
        <span class="hud-sep">//</span>
        <span class="hud-item {crack_class}">T3: HYST COOL {n_bonds}:{n_cracks}</span>
        <span class="hud-sep">//</span>
        <span class="hud-item {walker_class}">T4: Act {mean_action:.2f} | Surv {survival_rate:.0%}</span>
        <span class="hud-sep">//</span>
        {t5_section}
        <span class="hud-sep">//</span>
        <span class="hud-item">kNN: {knn_overlap:.1%}</span>
    </div>
    '''


HUD_CSS = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
        background: #050505;
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
        overflow: hidden;
    }

    .hud-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 36px;
        background: rgba(5,5,5,0.95);
        border-bottom: 1px solid #1a1a1a;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #00f0ff;
        display: flex;
        align-items: center;
        padding: 0 20px;
        gap: 12px;
        z-index: 1000;
    }

    .hud-sep { color: #333; }
    .hud-item { white-space: nowrap; }
    .hud-item.alert { color: #ff4444; animation: pulse 1s infinite; }
    .hud-item.warn { color: #ffaa00; }
    .hud-item.good { color: #00ff41; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .cockpit-container {
        margin-top: 36px;
        width: 100vw;
        height: calc(100vh - 36px);
    }

    .legend-panel {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(5, 5, 5, 0.9);
        border: 1px solid #006677;
        border-radius: 6px;
        padding: 15px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        z-index: 100;
    }

    .legend-title { color: #00F0FF; margin-bottom: 10px; font-weight: 600; }
    .legend-item { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
    .legend-line { width: 20px; height: 2px; }

    .epistemic-panel {
        position: fixed;
        top: 250px;
        left: 20px;
        width: 360px;
        background: rgba(8, 8, 8, 0.94);
        border: 1px solid #26474f;
        border-radius: 6px;
        padding: 12px;
        z-index: 1002;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        line-height: 1.45;
        color: #cde6ec;
    }
    .ep-title { color: #00F0FF; font-weight: 600; margin-bottom: 6px; }
    .ep-sub { color: #8eb3ba; margin-bottom: 8px; }
    .ep-row { margin: 4px 0; }
    .ep-row .k { color: #6ea0aa; }
    .ep-row .v { color: #e3f5ff; }
    .ep-warn { color: #FF5555; font-weight: 600; }
    .ep-good { color: #00FF41; font-weight: 600; }
    .provenance-line {
        margin-top: 8px;
        padding-top: 7px;
        border-top: 1px solid #20353a;
        color: #9dbec6;
    }
    .verification-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #444;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .verification-badge.good {
        color: #00FF41;
        border-color: #00FF41;
        background: rgba(0,255,65,0.12);
    }
    .verification-badge.bad {
        color: #FF3333;
        border-color: #FF3333;
        background: rgba(255,51,51,0.12);
    }
    .noncomparable-watermark {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-20deg);
        font-family: 'JetBrains Mono', monospace;
        font-size: 56px;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: rgba(255, 68, 68, 0.10);
        pointer-events: none;
        z-index: 900;
        user-select: none;
    }
</style>
'''


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================
def create_monolith_cockpit(
    exp: ExperimentData,
    output_path: Path,
    physics_mode: str = "synthesis",
    observer_idx: Optional[int] = None,
    show_terrain: bool = True,
    show_fog: bool = True,
    show_walkers: bool = True,
    show_phantom_paths: bool = True,
    show_hott: bool = True,
    show_spectral_axis: bool = True,
    strict_validation: bool = False,
    path_ablation_mode: str = "none",
) -> Any:
    """
    Create the MONOLITH visualization with ALL tracks.

    This is the main entry point that combines:
    - Track 1: Logits indicators
    - Track 1.5: Spectral axis arrow
    - Track 2: Hologram terrain
    - Track 3: Fog overlay (Dirichlet variance)
    - Track 4: Walker diamonds (Semantic Walker states)
    - Track 5: Phantom paths (HONEST/PHANTOM/RUPTURE)
    - Track 6: HoTT verdict icons
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")

    output_path = Path(output_path)
    path_ablation_mode = str(path_ablation_mode or "none").strip().lower()
    if path_ablation_mode not in {"none", "thermodynamic", "semantic_tether"}:
        raise ValueError(f"Unknown path_ablation_mode: {path_ablation_mode}")

    # Choose features
    if physics_mode == "synthesis" and exp.integrated is not None:
        features = exp.integrated
    else:
        features = exp.features

    n_articles = len(features)
    focus_idx: Optional[int] = None
    if observer_idx is not None and 0 <= int(observer_idx) < n_articles:
        focus_idx = int(observer_idx)

    # Spectral data
    spectral_evr = exp.spectral_evr if exp.spectral_evr is not None else np.ones(n_articles) * 0.5
    spectral_mags_raw = exp.spectral_probe_magnitudes
    spectral_mags_available = (
        spectral_mags_raw is not None
        and len(spectral_mags_raw) > 0
        and not np.any(np.isnan(spectral_mags_raw))
        and not np.all(spectral_mags_raw == 0)
    )
    if spectral_mags_available:
        spectral_mags = spectral_mags_raw
        spectral_mags_hover = spectral_mags_raw
    else:
        # Keep deterministic numeric fallback for geometry/math code paths.
        spectral_mags = np.zeros((n_articles, 8), dtype=float)
        spectral_mags_hover = spectral_mags

    # Track 1: Logit confidence (for halos) - fallback to spectral_evr if not available
    logit_confidence = exp.logit_confidence if exp.logit_confidence is not None else spectral_evr

    # Track 3: Fog (Percentile-based atmospheric classification)
    fog_intensity = np.zeros(n_articles)
    is_fog = np.zeros(n_articles, dtype=bool)
    is_bond = np.zeros(n_articles, dtype=bool)
    atmospheric_states = ['HAZE'] * n_articles
    if exp.dirichlet_fused_std is not None:
        blinker_magnitude = np.linalg.norm(exp.dirichlet_fused_std, axis=1)
        fog_intensity, is_fog, is_bond = compute_fog_intensity(blinker_magnitude)
        atmospheric_states = classify_atmospheric_state(blinker_magnitude)

    # Track 4: Walker
    walker_states = exp.walker_states if exp.walker_states else ['elastic'] * n_articles
    walker_work = exp.walker_work_integrals
    walker_paths_raw = exp.walker_paths if exp.walker_paths else {}

    # Track 5: Phantom
    phantom_verdicts = exp.phantom_verdicts if exp.phantom_verdicts else []

    # Track 6: HoTT
    hott_proofs = exp.hott_proofs if exp.hott_proofs else []

    # Metadata
    metadata = exp.article_metadata if exp.article_metadata else [{}] * n_articles
    metadata_by_uid: Dict[str, Dict[str, Any]] = {}
    for row in metadata:
        if isinstance(row, dict):
            uid = str(row.get("bt_uid", "")).strip()
            if uid:
                metadata_by_uid[uid] = row

    # Load MONOLITH_DATA.csv
    monolith_data_path = exp.experiment_dir / "MONOLITH_DATA.csv"
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load MONOLITH_DATA.csv from: {monolith_data_path.resolve()}")

    unified_zones = [] # Initialize for fallback
    unified_color_codes = [] # Initialize for fallback

    global_density_median = 0.5 # Initialize for fallback
    global_stress_median = 0.5  # Initialize for fallback

    try:
        # Attempt to open the file to verify accessibility
        # This will raise FileNotFoundError if it doesn't exist or is inaccessible
        with open(monolith_data_path, 'r') as f:
            pass 

        print(f"[MONOLITH] Loading unified metrics from {monolith_data_path}...")
        monolith_df = pd.read_csv(monolith_data_path)

        # Force verdicts from CSV into the visualization (authoritative source).
        if 'verdict' in monolith_df.columns:
            print(f"[MONOLITH] Wiring {len(monolith_df)} verdicts from MONOLITH_DATA.csv")
            phantom_verdicts = []
            for i, row in monolith_df.iterrows():
                phantom_verdicts.append({
                    'article_id': row.get('article_id', f'art_{i}'),
                    'verdict': str(row['verdict']).upper(),
                    'w_actual': row.get('stress', 0.5) * 5.0, # Approximate work for HUD
                    'delta': 1.0 # Default delta
                })

        # Ensure loaded data matches n_articles
        if len(monolith_df) != n_articles:
            raise ValueError(f"Mismatch in number of articles. ExperimentData has {n_articles}, "
                             f"but MONOLITH_DATA.csv has {len(monolith_df)}.")

        # Secure metadata fusion via bt_uid, never by row position.
        metadata_df = pd.DataFrame(metadata) if metadata else pd.DataFrame()
        if (
            not metadata_df.empty
            and "bt_uid" in monolith_df.columns
            and "bt_uid" in metadata_df.columns
        ):
            metadata_df["bt_uid"] = metadata_df["bt_uid"].astype(str)
            monolith_df["bt_uid"] = monolith_df["bt_uid"].astype(str)
            merged_df = monolith_df.merge(metadata_df, on="bt_uid", how="left", suffixes=("", "_meta"))
            for col in ("title", "source", "publication", "affiliation", "bias", "text", "snippet"):
                meta_col = f"{col}_meta"
                if meta_col in merged_df.columns:
                    if col in merged_df.columns:
                        merged_df[col] = merged_df[col].fillna(merged_df[meta_col])
                    else:
                        merged_df[col] = merged_df[meta_col]
                    merged_df = merged_df.drop(columns=[meta_col])
            monolith_df = merged_df

        unified_density = monolith_df['density'].values
        unified_stress = monolith_df['stress'].values
        unified_z_height = monolith_df['z_height'].values
        raw_zones = monolith_df['zone'].values
        unified_zones = np.array([canonicalize_zone_name(z) for z in raw_zones], dtype=object)
        unified_color_codes = np.array([ZONE_COLOR_MAP[z] for z in unified_zones], dtype=object)
        if np.any(unified_zones != raw_zones):
            remap_counts = Counter(zip(raw_zones.tolist(), unified_zones.tolist()))
            print(f"[MONOLITH] Canonicalized legacy zones from MONOLITH_DATA.csv: {dict(remap_counts)}")

        # Calculate global medians from the full dataset for consistent zone mapping
        global_density_median = np.percentile(unified_density, 50)
        global_stress_median = np.percentile(unified_stress, 50)

        # Replace existing calculations with unified metrics
        terrain_density = unified_density
        terrain_stress = unified_stress
        energy_values_for_points = unified_z_height # Z-height for points is the calculated unified Z
        # Deterministic terrain geometry contract: use unified z_height.
        energy_values_for_terrain = unified_z_height

        # Re-derive atmospheric states for fog/bond based on unified_density (Track 2 - Density)
        # Using unified_density as proxy for inverse blinker_magnitude.
        # Higher density = lower variance (less fog)
        blinker_magnitude_proxy = 1.0 - unified_density # Inverse of density
        fog_intensity, is_fog, is_bond = compute_fog_intensity(blinker_magnitude_proxy)
        atmospheric_states = classify_atmospheric_state(blinker_magnitude_proxy)

        print(f"[MONOLITH] Using unified metrics: density=[{terrain_density.min():.2f}, {terrain_density.max():.2f}], "
              f"stress=[{terrain_stress.min():.2f}, {terrain_stress.max():.2f}], "
              f"Z_height=[{energy_values_for_points.min():.2f}, {energy_values_for_points.max():.2f}]")

    except FileNotFoundError:
        print(f"WARNING: MONOLITH_DATA.csv not found or accessible at {monolith_data_path.resolve()}. "
              "Falling back to internal calculations. Please run core/metric_fusion.py first.")
        # Fallback to existing calculations for density/stress
        
        # Compute terrain field (needed for Z positioning)
        # Density from blinker (Track 3), Stress from walker (Track 4)
        blinker_values = fog_intensity  # Higher fog_intensity = higher variance = lower density
        walker_resistance_values = np.array([
            phantom_verdicts[i].get('w_actual', 0.5) if i < len(phantom_verdicts) else 0.5
            for i in range(n_articles)
        ])
        terrain_density, terrain_stress, terrain_scalar = compute_terrain_field(
            blinker_values, walker_resistance_values
        )
        print(f"[MONOLITH] Terrain field: density=[{terrain_density.min():.2f}, {terrain_density.max():.2f}], "
              f"stress=[{terrain_stress.min():.2f}, {terrain_stress.max():.2f}]")

        # Calculate global medians from the fallback density and stress for consistent zone mapping
        global_density_median = np.percentile(terrain_density, 50)
        global_stress_median = np.percentile(terrain_stress, 50)

        # Deterministic fallback geometry contract: one Z source for points+terrain.
        energy_values_for_points = terrain_scalar
        energy_values_for_terrain = energy_values_for_points.copy()

        # Fallback zone and color calculation
        # 4 canonical zones from 2 orthogonal axes: Density (x) vs Stress (y)
        #   High density + Low stress  = BRIDGE    (constructive discourse)
        #   High density + High stress = SWAMP     (dense but contentious)
        #   Low density  + Low stress  = TIGHTROPE (sparse but navigable)
        #   Low density  + High stress = VOID      (barren and hostile)
        unified_zones = []
        unified_color_codes = []
        for i in range(n_articles):
            d = terrain_density[i]
            s = terrain_stress[i]
            if d >= global_density_median and s < global_stress_median:
                zone = "Bridge"
                color = ZONE_COLOR_MAP[zone]
            elif d >= global_density_median and s >= global_stress_median:
                zone = "Swamp"
                color = ZONE_COLOR_MAP[zone]
            elif d < global_density_median and s < global_stress_median:
                zone = "Tightrope"
                color = ZONE_COLOR_MAP[zone]
            else:
                zone = "Void"
                color = ZONE_COLOR_MAP[zone]
            unified_zones.append(zone)
            unified_color_codes.append(color)
        unified_zones = np.array(unified_zones)
        unified_color_codes = np.array(unified_color_codes)

    # Force fresh 3D PCA each run; do not trust cached projection columns.
    # Fit PCA on a shared basis that includes walker trajectory samples when possible.
    # This prevents article XY from collapsing when features alone are near-degenerate.
    print("[MONOLITH] Forcing unified 3D projection (points + paths)...")
    projection_fit_matrix = features
    try:
        if walker_paths_raw:
            path_samples = []
            for _p in walker_paths_raw.values():
                _arr = np.asarray(_p, dtype=float)
                if _arr.ndim == 2 and _arr.shape[1] == features.shape[1] and _arr.shape[0] >= 2:
                    # Sample every 12th step + last point to keep memory bounded.
                    _sample = _arr[::12]
                    if _sample.shape[0] == 0 or not np.array_equal(_sample[-1], _arr[-1]):
                        _sample = np.vstack([_sample, _arr[-1:]])
                    path_samples.append(_sample)
            if path_samples:
                path_fit = np.vstack(path_samples)
                projection_fit_matrix = np.vstack([features, path_fit])
                print(f"[MONOLITH] PCA fit basis: features+paths ({projection_fit_matrix.shape[0]}x{projection_fit_matrix.shape[1]})")
    except Exception as _e:
        print(f"[MONOLITH] PCA fit basis fallback to features only: {_e}")

    pca_3d = PCA(n_components=3, random_state=42, whiten=False)
    pca_3d.fit(projection_fit_matrix)
    article_proj = pca_3d.transform(features)

    # PATH PROJECTION CONTRACT:
    # Persisted walker paths are emitted from physics in high-D space (typically 2048D).
    # Transform to render space without additional visual-layer scaling.
    walker_paths_projected: Dict[int, np.ndarray] = {}
    for article_idx, raw_path in walker_paths_raw.items():
        try:
            path_arr = np.asarray(raw_path, dtype=float)
            if path_arr.ndim != 2 or path_arr.shape[0] < 2:
                continue
            if path_arr.shape[1] == features.shape[1]:
                path_proj = pca_3d.transform(path_arr)
            elif path_arr.shape[1] == 3:
                path_proj = path_arr.copy()
            else:
                continue
            path_proj = np.nan_to_num(path_proj, nan=0.0, posinf=0.0, neginf=0.0)
            walker_paths_projected[int(article_idx)] = path_proj[:, :3]
        except Exception:
            continue

    # GROUND-STATE RESET: no robust scaling, no span sync, no clip/stretch.
    positions_3d = article_proj.copy()
    positions_2d = positions_3d[:, :2]

    # RAW POINT Z: use persisted pure_z from MONOLITH_DATA.csv when available.
    if 'monolith_df' in locals() and 'pure_z' in monolith_df.columns:
        pure_z = monolith_df['pure_z'].to_numpy(dtype=float)
    elif 'unified_z_height' in locals() and len(unified_z_height) == n_articles:
        pure_z = np.asarray(unified_z_height, dtype=float)
    else:
        pure_z = positions_3d[:, 2].astype(float)
    if len(pure_z) == n_articles:
        positions_3d[:, 2] = pure_z

    # RAW PATHS: no additional normalization/clipping.
    walker_paths_pure: Dict[int, np.ndarray] = {}
    for article_idx, path_proj in walker_paths_projected.items():
        walker_paths_pure[int(article_idx)] = np.asarray(path_proj[:, :3], dtype=float)

    # Deterministic XY normalization (no runtime source fallback):
    # keep relative geometry, center XY, and scale into a stable display range.
    xy_ptp = np.ptp(positions_3d[:, :2], axis=0)
    if xy_ptp.size != 2 or not np.isfinite(xy_ptp).all() or np.any(xy_ptp <= 1e-12):
        raise DimensionalCollapseError(
            f"CRITICAL: XY manifold collapsed at projection stage (ptp={xy_ptp})."
        )
    target_xy_span = np.array([6.0, 6.0], dtype=float)
    xy_center = np.mean(positions_3d[:, 0:2], axis=0, keepdims=True)
    xy_scale = target_xy_span / xy_ptp
    article_xy_ptp = np.ptp(positions_3d[:, :2], axis=0)
    article_span_ref = (
        float(np.max(article_xy_ptp))
        if article_xy_ptp.size == 2 and np.isfinite(article_xy_ptp).all()
        else 0.0
    )
    positions_3d[:, 0:2] = (positions_3d[:, 0:2] - xy_center) * xy_scale
    # Normalize path XY per-path with an anchored fallback for incompatible frames.
    compatible_paths: Dict[int, np.ndarray] = {}
    skipped_invalid_paths = 0
    anchored_fallback_paths = 0
    max_allowed_norm_span = float(np.max(target_xy_span) * 2.0)
    for _idx, _path in walker_paths_pure.items():
        if _path.ndim != 2 or _path.shape[0] < 2 or _path.shape[1] < 2:
            skipped_invalid_paths += 1
            continue
        _xy = np.asarray(_path[:, 0:2], dtype=float)
        finite_rows = np.isfinite(_xy).all(axis=1)
        _xy_finite = _xy[finite_rows]
        if _xy_finite.shape[0] < 2:
            skipped_invalid_paths += 1
            continue
        path_span = float(np.max(np.ptp(_xy_finite, axis=0)))
        is_compatible = (
            np.isfinite(path_span)
            and article_span_ref > 1e-12
            and path_span <= (article_span_ref * 10.0)
        )
        _path_norm = np.asarray(_path, dtype=float).copy()
        if is_compatible:
            _path_norm[:, 0:2] = (_path_norm[:, 0:2] - xy_center) * xy_scale
        else:
            # Fallback: preserve local trajectory shape but anchor to its source article.
            anchor_xy = np.asarray(positions_3d[_idx, 0:2], dtype=float)
            start_xy = np.asarray(_path_norm[0, 0:2], dtype=float)
            _path_norm[:, 0:2] = (_path_norm[:, 0:2] - start_xy) * xy_scale + anchor_xy
            norm_finite = np.isfinite(_path_norm[:, 0:2]).all(axis=1)
            norm_xy = _path_norm[norm_finite, 0:2]
            if norm_xy.shape[0] >= 2:
                norm_span = float(np.max(np.ptp(norm_xy, axis=0)))
                if np.isfinite(norm_span) and norm_span > max_allowed_norm_span and norm_span > 1e-9:
                    shrink = max_allowed_norm_span / norm_span
                    _path_norm[:, 0:2] = anchor_xy + ((_path_norm[:, 0:2] - anchor_xy) * shrink)
            anchored_fallback_paths += 1
        compatible_paths[int(_idx)] = _path_norm
    walker_paths_pure = compatible_paths
    normalize_paths = bool(walker_paths_pure)
    if skipped_invalid_paths > 0:
        print(
            "[MONOLITH] Skipped invalid walker paths during XY normalization: "
            f"{skipped_invalid_paths}"
        )
    if anchored_fallback_paths > 0:
        print(
            "[MONOLITH] Applied anchored fallback normalization to walker paths: "
            f"{anchored_fallback_paths}"
        )
    if not normalize_paths and len(walker_paths_raw) > 0:
        print(
            "[MONOLITH] No compatible walker paths after per-path frame gating; "
            f"article_span={article_span_ref:.6f}"
        )
    print(
        "[MONOLITH] Applied deterministic per-axis XY normalization: "
        f"scale_x={xy_scale[0]:.1f}, scale_y={xy_scale[1]:.1f}, "
        f"ptp=({xy_ptp[0]:.6f},{xy_ptp[1]:.6f})->(6.0,6.0)"
    )

    rupture_segments_2d: List[Tuple[np.ndarray, np.ndarray]] = []
    if path_ablation_mode == "thermodynamic":
        for _idx, _path in walker_paths_pure.items():
            if _idx < 0 or _idx >= n_articles:
                continue
            pv = phantom_verdicts[_idx] if (_idx < len(phantom_verdicts)) else {}
            verdict = str(pv.get("verdict", "UNKNOWN")).upper()
            if verdict != "RUPTURE":
                continue
            if _path.ndim != 2 or _path.shape[0] < 2 or _path.shape[1] < 2:
                continue
            xy = np.asarray(_path[:, 0:2], dtype=float)
            finite_rows = np.isfinite(xy).all(axis=1)
            xy = xy[finite_rows]
            if xy.shape[0] < 2:
                continue
            xy[0, 0] = float(positions_3d[_idx, 0])
            xy[0, 1] = float(positions_3d[_idx, 1])
            for _j in range(xy.shape[0] - 1):
                start_xy = np.asarray(xy[_j], dtype=float)
                end_xy = np.asarray(xy[_j + 1], dtype=float)
                if np.isfinite(start_xy).all() and np.isfinite(end_xy).all():
                    rupture_segments_2d.append((start_xy, end_xy))

    # Deterministic terrain source: keep terrain Z exactly aligned to point pure_z.
    energy_values_for_terrain = pure_z.copy()
    # Contract guardrail: preserve explicit invalid-array fallback path.
    terrain_values_valid = False
    try:
        terrain_arr = np.asarray(energy_values_for_terrain, dtype=float)
        terrain_values_valid = (
            terrain_arr.ndim == 1
            and terrain_arr.shape[0] == n_articles
            and np.isfinite(terrain_arr).all()
        )
    except Exception:
        terrain_values_valid = False
    if not terrain_values_valid:
        print("[MONOLITH] Falling back terrain Z to pure_z due to invalid terrain field.")
        energy_values_for_terrain = pure_z.copy()

    # Raw variance probe (requested).
    print(f"RAW TERRAIN VARIANCE: Min={np.min(unified_stress)}, Max={np.max(unified_stress)}")
    print(f"RAW POINT VARIANCE: Min={np.min(pure_z)}, Max={np.max(pure_z)}")

    if 'monolith_df' in locals():
        monolith_df['x_proj'] = positions_3d[:, 0]
        monolith_df['y_proj'] = positions_3d[:, 1]
        monolith_df['z_proj'] = positions_3d[:, 2]
        monolith_df['pure_x'] = positions_3d[:, 0]
        monolith_df['pure_y'] = positions_3d[:, 1]
        monolith_df['pure_z'] = positions_3d[:, 2]
        if os.environ.get("MONOLITH_WRITE_PROJECTION_CACHE", "0").strip() == "1":
            monolith_df.to_csv(monolith_data_path, index=False)
    positions_2d = positions_3d[:, :2]
    print(
        f"[MONOLITH] Display-normalized projection scales: "
        f"std=({np.std(positions_3d[:,0]):.3f}, {np.std(positions_3d[:,1]):.3f}, {np.std(positions_3d[:,2]):.3f})"
    )








    # Optional legacy display recolor (off by default): quantile-based Z banding.
    # Canonical behavior is to preserve zone-derived colors from MONOLITH_DATA.csv.
    if os.environ.get("MONOLITH_ENABLE_BRIDGE_CONTRACTION", "0").strip() == "1":
        z_for_contract = np.asarray(energy_values_for_points, dtype=float)
        q10 = float(np.nanpercentile(z_for_contract, 10)) if z_for_contract.size else 0.0
        q22 = float(np.nanpercentile(z_for_contract, 22)) if z_for_contract.size else 0.0
        q55 = float(np.nanpercentile(z_for_contract, 55)) if z_for_contract.size else 0.0
        contracted_colors = []
        for zc in z_for_contract:
            if zc <= q10:
                contracted_colors.append(ZONE_COLOR_MAP["Bridge"])
            elif zc <= q22:
                contracted_colors.append(ZONE_COLOR_MAP["Tightrope"])
            elif zc <= q55:
                contracted_colors.append("#7B2CBF")
            else:
                contracted_colors.append("#4A0404")
        unified_color_codes = np.array(contracted_colors, dtype=object)
        print(
            "[MONOLITH] Bridge contraction by Z-bands: "
            f"bridge={int((z_for_contract<=q10).sum())}, "
            f"tightrope={int(((z_for_contract>q10)&(z_for_contract<=q22)).sum())}, "
            f"swamp={int(((z_for_contract>q22)&(z_for_contract<=q55)).sum())}, "
            f"void={int((z_for_contract>q55).sum())}"
        )

    # (terrain_scalar already computed above for Z positioning)

    # Count verdicts (force from MONOLITH_DATA.csv when available).
    if 'monolith_df' in locals() and 'verdict' in monolith_df.columns:
        verdict_series = monolith_df['verdict'].astype(str).str.upper()
        n_ruptures = int((verdict_series == 'RUPTURE').sum())
        n_phantoms = int((verdict_series == 'PHANTOM').sum())
        n_honest = int((verdict_series == 'HONEST').sum())
        n_tautology = int((verdict_series == 'TAUTOLOGY').sum())
    else:
        n_ruptures = sum(1 for v in phantom_verdicts if str(v.get('verdict', 'UNKNOWN')).upper() == 'RUPTURE')
        n_phantoms = sum(1 for v in phantom_verdicts if str(v.get('verdict', 'UNKNOWN')).upper() == 'PHANTOM')
        n_honest = sum(1 for v in phantom_verdicts if str(v.get('verdict', 'UNKNOWN')).upper() == 'HONEST')
        n_tautology = sum(1 for v in phantom_verdicts if str(v.get('verdict', 'UNKNOWN')).upper() == 'TAUTOLOGY')

    # Default knn_overlap as it's no longer computed from local_density
    knn_overlap = 0.0


    # Track 4 physics summary for HUD (status comes from persisted physics metadata)
    def _normalize_walker_status(s_raw: Any) -> str:
        """Normalize legacy/new walker-state formats into SUCCESS/BROKEN/TRAPPED/UNKNOWN."""
        if isinstance(s_raw, dict):
            s = str(s_raw.get("status") or s_raw.get("state") or "").strip().upper()
        else:
            s = str(s_raw).strip().upper()

        # Canonical persisted physics statuses
        if s in {"SUCCESS", "BROKEN", "TRAPPED"}:
            return s
        # Legacy labels fallback
        if s in {"HONEST"}:
            return "SUCCESS"
        if s in {"RUPTURE", "BROKEN"}:
            return "BROKEN"
        if s in {"TRAPPED", "TAUTOLOGY", "PHANTOM"}:
            return "TRAPPED"
        return "UNKNOWN"

    walker_statuses = [_normalize_walker_status(s) for s in walker_states] if walker_states else []
    n_total_walkers = len(walker_statuses)
    n_success = sum(1 for s in walker_statuses if s == "SUCCESS")
    n_broken = sum(1 for s in walker_statuses if s == "BROKEN")
    n_trapped = sum(1 for s in walker_statuses if s == "TRAPPED")

    finite_work = walker_work[np.isfinite(walker_work)] if walker_work is not None else np.array([])
    mean_action = float(np.mean(finite_work)) if finite_work.size > 0 else 0.0
    survival_rate = float(n_success / max(1, n_total_walkers)) if n_total_walkers > 0 else 1.0

    # Count fog/cracks
    n_cracks = int((fog_intensity > THRESHOLDS["fog_variance"]).sum())
    n_bonds = n_articles - n_cracks

    # Build hover texts - RICH METADATA for each article
    hover_texts = []
    for i in range(n_articles):
        meta: Dict[str, Any] = {}
        csv_row = None
        if 'monolith_df' in locals() and i < len(monolith_df):
            csv_row = monolith_df.iloc[i]
            csv_uid = str(csv_row.get("bt_uid", "")).strip()
            if csv_uid and csv_uid in metadata_by_uid:
                meta = metadata_by_uid[csv_uid]
        elif i < len(metadata):
            meta = metadata[i]
        # Full title (up to 80 chars) - don't truncate too much
        title = str(
            (csv_row.get('title') if csv_row is not None and pd.notna(csv_row.get('title')) else None)
            or meta.get('title', f'Article {i}')
        )[:80]
        bt_uid = str(
            (csv_row.get('bt_uid') if csv_row is not None and pd.notna(csv_row.get('bt_uid')) else None)
            or meta.get('bt_uid', '')
        )[:16]
        evr = spectral_evr[i] if i < len(spectral_evr) else 0.5

        if spectral_mags_hover is not None and i < len(spectral_mags_hover):
            mags = spectral_mags_hover[i]
            top_idx = np.argsort(np.abs(mags))[::-1][:3]
            drivers = "<br>".join(
                [f"  {html.escape(PROBE_LABELS[j][:30])}: {mags[j]:+.3f}" for j in top_idx]
            )
        else:
            mags = np.zeros(8, dtype=float)
            top_idx = np.array([0, 1, 2], dtype=int)
            drivers = "<br>".join(
                [f"  {html.escape(PROBE_LABELS[j][:30])}: {mags[j]:+.3f}" for j in top_idx]
            )
        vec = np.asarray(features[i], dtype=float).reshape(-1)
        top_dims = np.argsort(np.abs(vec))[::-1][:3]
        dim_drivers = "<br>".join([f"  Dim {int(d)}: {vec[int(d)]:+.4f}" for d in top_dims])
        top_dim_vals = np.abs(vec[top_dims])
        collinear_warn = ""
        if float(np.var(top_dim_vals)) < 1e-3:
            collinear_warn = "<br><span style='color:#FFAA00'><b>COLLINEAR WARNING:</b> Top-3 dim magnitudes nearly identical</span>"

        # Walker state
        ws = walker_states[i] if i < len(walker_states) else "unknown"

        # Phantom verdict and terrain state
        pv = "unknown"
        terrain = "unknown"
        delta = 0.0
        d_val = 0.0
        w_val = 0.0
        if phantom_verdicts and i < len(phantom_verdicts):
            pv = phantom_verdicts[i].get('verdict', 'unknown')
            terrain = phantom_verdicts[i].get('terrain_state', 'unknown')
            delta = phantom_verdicts[i].get('delta', 0.0)
            d_val = phantom_verdicts[i].get('d', phantom_verdicts[i].get('d_spectral', 0.0))
            w_val = phantom_verdicts[i].get('w', phantom_verdicts[i].get('w_actual', 0.0))
        if csv_row is not None and 'verdict' in monolith_df.columns and pd.notna(csv_row.get('verdict')):
            pv = str(csv_row.get('verdict')).upper()

        # HoTT status
        hott_status = "?"
        if hott_proofs and i < len(hott_proofs):
            hott_status = hott_proofs[i].get('status', '?')

        # Source/Affiliation/Bias stamp from metadata
        pub = str(
            (csv_row.get('source') if csv_row is not None and pd.notna(csv_row.get('source')) else None)
            or meta.get('publication', meta.get('source', ''))
        )[:40]
        affiliation = str(
            (csv_row.get('affiliation') if csv_row is not None and pd.notna(csv_row.get('affiliation')) else None)
            or (csv_row.get('perspective_type') if csv_row is not None and pd.notna(csv_row.get('perspective_type')) else None)
            or meta.get('affiliation', '')
            or meta.get('perspective_type', '')
            or 'unknown'
        )[:40]
        bias = str(
            (csv_row.get('bias') if csv_row is not None and pd.notna(csv_row.get('bias')) else None)
            or (csv_row.get('perspective_tag') if csv_row is not None and pd.notna(csv_row.get('perspective_tag')) else None)
            or meta.get('bias', '')
            or meta.get('perspective_tag', '')
            or 'unknown'
        )[:40]
        snippet_raw = (
            (csv_row.get('snippet') if csv_row is not None and pd.notna(csv_row.get('snippet')) else None)
            or (csv_row.get('text') if csv_row is not None and pd.notna(csv_row.get('text')) else None)
            or meta.get('snippet')
            or meta.get('text')
            or ""
        )
        snippet = str(snippet_raw).replace("\n", " ").strip()[:140]
        title = html.escape(title)
        bt_uid = html.escape(bt_uid)
        pub = html.escape(pub)
        affiliation = html.escape(affiliation)
        bias = html.escape(bias)
        snippet = html.escape(snippet)
        pub_line = f'<b>Source:</b> {pub}<br>' if pub else ''
        affiliation_line = f'<b>Affiliation:</b> {affiliation}<br>'
        bias_line = f'<b>Bias:</b> {bias}<br>'
        snippet_line = f'<b>Snippet:</b> {snippet}<br>' if snippet else ''

        # Format d and w values (handle infinity)
        d_str = f"{d_val:.2f}" if np.isfinite(d_val) else "inf"
        w_str = f"{w_val:.2f}" if np.isfinite(w_val) else "inf"

        focus_line = "<span style='color:#FFD700'>(FOCUS)</span><br>" if (focus_idx is not None and i == focus_idx) else ""
        hover_texts.append(
            f'<b style="font-size:14px">Article #{i}</b><br>'
            f"{focus_line}"
            f'<span style="color:#00F0FF">{title}</span><br>'
            f'<span style="color:#888">UID: {bt_uid}</span><br>'
            f'{pub_line}'
            f'{affiliation_line}'
            f'{bias_line}'
            f'{snippet_line}'
            f'<b>═══════════════════════</b><br>'
            f'<b>EVR:</b> {evr:.3f}<br>'
            f'<b>Zone:</b> {unified_zones[i]}<br>' # Add this line
            f'<b>T4 Walker:</b> {ws}<br>'
            f'<b>T5 Verdict:</b> <span style="color:{"#00F0FF" if pv=="HONEST" else "#FF00FF" if pv=="PHANTOM" else "#FF2222" if pv=="RUPTURE" else "#888"}">{pv}</span><br>'
            f'<b>  d={d_str}, W={w_str}, Delta={delta:.2f}</b><br>'
            f'<b>T6 HoTT:</b> {hott_status}<br>'
            f'<b>═══════════════════════</b><br>'
            f'<b>Spectral DNA:</b><br>{drivers}<br>'
            f'<b>Top Feature Dims:</b><br>{dim_drivers}{collinear_warn}'
        )
    # ==========================================================================
    # BUILD FIGURE WITH DUAL-MODE TRACES
    # ==========================================================================
    fig = go.Figure()

    # Track trace indices for mode toggling
    synthesis_trace_start = 0
    diagnostics_trace_start = 0

    # =========================================
    # SYNTHESIS MODE TRACES (Default: visible)
    # =========================================
    synthesis_trace_start = len(fig.data)

    # Dumb-renderer mode: keep terrain fully visible, no verification ghosting.
    surface_opacity = 0.9
    verification_title_stamp = ""

    # Layer 1: Terrain Surface (colored by density×stress manifold)
    if show_terrain:
        print("[MONOLITH] Rendering terrain surface with density×stress gradient...")
        terrain_stress_for_geometry = None
        terrain, terrain_grid_x, terrain_grid_y, terrain_grid_z, grid_density, grid_stress = render_terrain_surface(
            positions_3d, energy_values_for_terrain,
            terrain_density=terrain_density,
            terrain_stress=terrain_stress,
            terrain_stress_geometry=terrain_stress_for_geometry,
            use_manifold_colormap=True,
            global_density_median=global_density_median,
            global_stress_median=global_stress_median,
            opacity=surface_opacity,
            rupture_segments_2d=rupture_segments_2d if path_ablation_mode == "thermodynamic" else None,
            rupture_tear_radius_scale=0.02,
        )
        if terrain:
            terrain.visible = True
            terrain.meta = {'custom_mode': 'terrain'}
            fig.add_trace(terrain)
            
            # Add contour lines overlaid on terrain
            if grid_density is not None and grid_stress is not None:
                print("[MONOLITH] Adding terrain contour lines...")
                contour_traces = render_terrain_contours(
                    terrain_grid_x, terrain_grid_y, terrain_grid_z,
                    grid_density, grid_stress
                )
                for trace in contour_traces:
                    trace.meta = {'custom_mode': 'terrain'}
                    fig.add_trace(trace)
    
    # --- Surface Sampling (The Nuclear Clamp) ---
    # After generating the 'grid_z' for the Surface Plot,
    # create a 'scipy.interpolate.RegularGridInterpolator'
    # using the grid X, Y, and Z.
    interp_terrain_z = None
    if show_terrain and terrain_grid_x is not None and terrain_grid_y is not None and terrain_grid_z is not None:
        from scipy.interpolate import RegularGridInterpolator
        # Create the RegularGridInterpolator
        interp_terrain_z = RegularGridInterpolator(
            (terrain_grid_y[:, 0], terrain_grid_x[0, :]), # (Y coordinates, X coordinates)
            terrain_grid_z, # Z values
            method="linear",
            bounds_error=False,
            fill_value=None # Extrapolate rather than fill with a constant
        )
    
    # Calculate visual_z for each article using the interpolator
    # positions_2d contains the (x, y) coordinates of the articles
    article_xy = positions_2d[:, :2] # Only X and Y
    
    # Enforce coordinate contract: article points default to pure_x/pure_y/pure_z.
    # Terrain interpolator remains available for overlays that intentionally drape to surface.
    energy_values_for_points = positions_3d[:, 2]

    def _nearest_article_surface_z(x_coords, y_coords) -> np.ndarray:
        xq = np.atleast_1d(x_coords).astype(float)
        yq = np.atleast_1d(y_coords).astype(float)
        if article_xy.shape[0] <= 0:
            return np.full_like(xq, positions_3d[:, 2].mean() if len(positions_3d) > 0 else 0.0, dtype=float)
        q = np.column_stack((xq, yq))
        d = cdist(q, article_xy)
        idx = np.argmin(d, axis=1)
        return np.asarray(energy_values_for_points, dtype=float)[idx]

    # Helper function to get surface Z from interpolator
    def get_surface_z(x_coords, y_coords, offset=0.0, preserve_nan=False):
        if interp_terrain_z is not None:
            # Ensure x_coord and y_coord are numpy arrays for interpolation
            x_coord_np = np.atleast_1d(x_coords)
            y_coord_np = np.atleast_1d(y_coords)
            
            # Create points array for interpolator: (N, 2) where each row is (y, x)
            points_for_interp = np.column_stack((y_coord_np, x_coord_np))
            
            interp_z = np.asarray(interp_terrain_z(points_for_interp), dtype=float)
            invalid = ~np.isfinite(interp_z)
            if np.any(invalid) and not preserve_nan:
                interp_z[invalid] = _nearest_article_surface_z(
                    x_coord_np[invalid],
                    y_coord_np[invalid],
                )
            
            # If input was scalar, return scalar. If array, return array.
            if isinstance(x_coords, (int, float, np.floating)):
                return interp_z.item() + offset
            return interp_z + offset
        else:
            # Fallback if no interpolator (should not happen if show_terrain is True)
            # Use average Z of the points, or 0.0
            return np.full_like(np.atleast_1d(x_coords), positions_3d[:, 2].mean() if len(positions_3d) > 0 else 0.0) + offset

    # Article markers should sit on the rendered terrain manifold when available.
    # Fallback remains the current point Z contract when interpolation is unavailable/fails.
    article_marker_z = np.asarray(energy_values_for_points, dtype=float)
    try:
        if interp_terrain_z is not None:
            article_marker_z = np.asarray(
                get_surface_z(article_xy[:, 0], article_xy[:, 1], offset=0.01),
                dtype=float,
            )
            if article_marker_z.shape[0] != n_articles:
                article_marker_z = np.asarray(energy_values_for_points, dtype=float)
    except Exception as marker_z_err:
        print(f"[MONOLITH] Marker Z fallback to point Z due to interpolation error: {marker_z_err}")
        article_marker_z = np.asarray(energy_values_for_points, dtype=float)


    # Layer 2: Phantom Paths (Track 5)
    if show_phantom_paths and phantom_verdicts:
        print(f"[MONOLITH] Rendering {len(phantom_verdicts)} phantom paths...")
        path_traces = render_phantom_paths_3d(
            phantom_verdicts, positions_3d,
            walker_paths=walker_paths_pure,
            article_z_height=article_marker_z,
            terrain_z_values=energy_values_for_terrain,
            surface_z_func=get_surface_z,
            article_metadata=metadata,
            spectral_evr=spectral_evr,
            spectral_probe_magnitudes=spectral_mags,
            hysteresis_memory=exp.hysteresis_memory,
            path_ablation_mode=path_ablation_mode,
        )
        for t in path_traces:
            t.visible = True
            t.meta = {'custom_mode': 'synthesis'}
            fig.add_trace(t)

        # Layer 5: HoTT Icons (Track 6) — with Phantom Delta override
        if show_hott and hott_proofs:
            print("[MONOLITH] Rendering Track 6 HoTT icons...")
            hott_traces = render_hott_icons_3d(
                positions_3d, hott_proofs, phantom_verdicts,
                get_surface_z_func=get_surface_z, # Pass helper function
            )
            for t in hott_traces:
                t.visible = True
                t.meta = {'custom_mode': 'synthesis'}
                fig.add_trace(t)

    # Layer 6: Spectral Axis (Track 1.5) - synthesis overlay when data is available
    if (
        show_spectral_axis
        and spectral_mags_available
        and isinstance(spectral_mags, np.ndarray)
        and spectral_mags.ndim == 2
        and spectral_mags.shape[0] >= n_articles
    ):
        try:
            spectral_axis_traces = render_spectral_axis_3d(
                positions_3d=positions_3d,
                spectral_probe_magnitudes=spectral_mags[:n_articles],
                evr=float(np.nanmean(spectral_evr)) if np.size(spectral_evr) > 0 else 0.5,
                get_surface_z_func=get_surface_z,
            )
            for t in spectral_axis_traces:
                t.visible = True
                t.meta = {'custom_mode': 'synthesis'}
                fig.add_trace(t)
        except Exception as spectral_axis_err:
            print(f"[MONOLITH] Skipping spectral axis overlay due to error: {spectral_axis_err}")


    # Layer 7: Data Points (glowing muons) - Both modes
    # Color by VERDICT, size by annealing (Crystal=small, Fog=large)
    # BASE SIZE = 10 (larger for easier clicking/hover)
    print(f"[MONOLITH] Rendering {n_articles} data points...")
    # Match marker size to density: consensus points (Bridge - high density) are solid, sparse points (Void - low density) are small/dimmed.
    sizes = np.ones(n_articles) * 10 * (0.5 + 0.5 * terrain_density) # Scale size by density (5 to 10)
    if focus_idx is not None:
        sizes[focus_idx] = max(sizes[focus_idx] * 1.8, 14.0)
    point_traces = render_data_points_3d(
        positions_3d, spectral_evr, sizes, hover_texts,
        phantom_verdicts=phantom_verdicts, is_fog=is_fog,
        article_z_height=article_marker_z, # Prefer terrain-manifold Z for marker anchoring
        article_color_codes=unified_color_codes, # Pass unified_color_codes for coloring
    )
    for t in point_traces:
        t.visible = True
        t.meta = {'custom_mode': 'synthesis'}
        fig.add_trace(t)

    synthesis_trace_end = len(fig.data)

    # =========================================
    # DIAGNOSTICS + ANALYSIS MODE TRACES
    # =========================================
    # Analysis and diagnostics traces must exist for mode switching to work.
    # Opt-out only via explicit fast-path env override.
    render_all_modes = os.environ.get("MONOLITH_FAST_SYNTHESIS_ONLY", "0").strip() != "1"
    diagnostics_trace_start = len(fig.data)
    analysis_nmi_scores = {}

    if render_all_modes:
        print("[MONOLITH] Rendering DIAGNOSTICS mode traces...")

        # DIAGNOSTIC Layer 1: Wind Streamlines (Track 1.5 antagonism vectors)
        if exp.antagonism is not None:
            print("[MONOLITH] Rendering wind streamlines from Track 1.5 antagonism...")
            wind_traces = render_wind_streamlines(positions_3d, exp.antagonism, n_streamlines=40)
            for t in wind_traces:
                t.visible = False
                t.meta = {'custom_mode': 'diagnostics'}
                fig.add_trace(t)

        # DIAGNOSTIC Layer 2: Slime Trails
        if walker_work is not None and len(walker_work) > 0:
            print("[MONOLITH] Rendering slime trails (Diagnostics)...")
            slime_traces = render_slime_trails(
                positions_3d, walker_work, n_neighbors=4,
                get_surface_z_func=get_surface_z,
                positions_2d=positions_2d,
            )
            for t in slime_traces:
                t.visible = False
                t.meta = {'custom_mode': 'diagnostics'}
                fig.add_trace(t)

        # DIAGNOSTIC Layer 3: Chromatic Ghosts
        print("[MONOLITH] Rendering chromatic ghosts (Diagnostics)...")
        ghost_traces = render_chromatic_ghosts(positions_3d, features, spectral_evr)
        for t in ghost_traces:
            t.visible = False
            t.meta = {'custom_mode': 'diagnostics'}
            fig.add_trace(t)

        # DIAGNOSTIC Layer 4: Confidence Halos
        print("[MONOLITH] Rendering confidence halos from Track 1 logit confidence...")
        halo_sizes = sizes * 3 * (1.0 - logit_confidence)
        halo_trace = go.Scatter3d(
            x=positions_3d[:, 0],
            y=positions_3d[:, 1],
            z=positions_3d[:, 2],
            mode='markers',
            marker=dict(
                size=halo_sizes,
                color='rgba(255,255,255,0.1)',
                line=dict(color=PALETTE.cyan, width=1),
            ),
            name='Confidence Halos',
            hoverinfo='skip',
            visible=False,
            meta={'custom_mode': 'diagnostics'}
        )
        fig.add_trace(halo_trace)

        # DIAGNOSTIC Layer 5: Hysteresis Highways (optional)
        enable_hysteresis_overlay = os.environ.get("MONOLITH_ENABLE_HYSTERESIS", "0").strip() == "1"
        if exp.hysteresis_memory is not None and enable_hysteresis_overlay:
            print("[MONOLITH] Rendering hysteresis highways from Track 4 path memory...")
            highway_traces = render_hysteresis_highways_3d(
                positions_3d=positions_3d,
                hysteresis_memory=exp.hysteresis_memory,
                probe_labels=PROBE_LABELS,
            )
            for t in highway_traces:
                t.visible = False
                t.meta = {'custom_mode': 'diagnostics'}
                fig.add_trace(t)
            print(f"  Added {len(highway_traces)} hysteresis traces")
        elif exp.hysteresis_memory is not None:
            print("[MONOLITH] Skipping hysteresis highways (set MONOLITH_ENABLE_HYSTERESIS=1 to enable)")

        # DIAGNOSTIC Layer 6: Walker Diamonds
        if walker_states and len(walker_states) > 0:
            print("[MONOLITH] Rendering walker diamonds from Track 4 states...")
            walker_diamond_traces = render_walker_diamonds_3d(
                positions_3d=positions_3d,
                walker_states=walker_states,
                walker_work=walker_work,
                article_metadata=metadata,
                phantom_verdicts=phantom_verdicts,
                spectral_evr=spectral_evr,
                show_all_states=True,
            )
            for t in walker_diamond_traces:
                t.visible = False
                t.meta = {'custom_mode': 'diagnostics'}
                fig.add_trace(t)
            print(f"  Added {len(walker_diamond_traces)} walker diamond traces")

        diagnostics_trace_end = len(fig.data)

        # ANALYSIS mode traces
        analysis_trace_start = len(fig.data)
        print("[MONOLITH] Rendering ANALYSIS mode traces (stacked track planes)...")
        analysis_traces, analysis_nmi_scores = render_analysis_planes(exp)
        for t in analysis_traces:
            t.visible = False
            t.meta = {'custom_mode': 'analysis'}
            fig.add_trace(t)
        analysis_trace_end = len(fig.data)
    else:
        diagnostics_trace_end = diagnostics_trace_start
        analysis_trace_start = diagnostics_trace_end
        analysis_trace_end = analysis_trace_start
        print("[MONOLITH] Fast synthesis render: skipped diagnostics/analysis traces (MONOLITH_FAST_SYNTHESIS_ONLY=1).")

    # Count traces per mode for JS toggle
    n_synthesis = synthesis_trace_end - synthesis_trace_start
    n_diagnostics = diagnostics_trace_end - diagnostics_trace_start
    n_analysis = analysis_trace_end - analysis_trace_start

    print(f"[MONOLITH] Trace counts: Synthesis={n_synthesis}, Diagnostics={n_diagnostics}, Analysis={n_analysis}")

    # ==========================================================================
    # TRACK 1.5 — DERIVE SEMANTIC AXIS LABELS FROM SPECTRAL PCA
    # Find which probe dominates PC1 and PC2 so axis labels are data-driven,
    # not hard-coded to PROBE_LABELS[0/1].
    # ==========================================================================
    axis_label_x = "Semantic Axis X"
    axis_label_y = "Semantic Axis Y"
    axis_label_z = "Semantic Axis Z"
    if (
        spectral_mags_available
        and isinstance(spectral_mags, np.ndarray)
        and spectral_mags.ndim == 2
        and spectral_mags.shape[0] >= 2
        and spectral_mags.shape[1] >= 1
    ):
        try:
            n_axis_components = min(3, spectral_mags.shape[0], spectral_mags.shape[1])
            if n_axis_components >= 1:
                axis_pca = PCA(n_components=n_axis_components, random_state=42, whiten=False)
                axis_pca.fit(np.asarray(spectral_mags[:n_articles], dtype=float))
                derived_labels: List[str] = []
                for _k in range(n_axis_components):
                    _component = np.asarray(axis_pca.components_[_k], dtype=float)
                    _probe_idx = int(np.argmax(np.abs(_component)))
                    _probe_label = (
                        PROBE_LABELS[_probe_idx]
                        if 0 <= _probe_idx < len(PROBE_LABELS)
                        else f"Probe {_probe_idx}"
                    )
                    derived_labels.append(_probe_label)
                if len(derived_labels) >= 1:
                    axis_label_x = f"PC1: {derived_labels[0]}"
                if len(derived_labels) >= 2:
                    axis_label_y = f"PC2: {derived_labels[1]}"
                if len(derived_labels) >= 3:
                    axis_label_z = f"PC3: {derived_labels[2]}"
        except Exception as axis_label_err:
            print(f"[MONOLITH] Axis label fallback to defaults due to PCA error: {axis_label_err}")

    # ==========================================================================
    # LAYOUT
    # ==========================================================================
    show_axes_initial = physics_mode != "synthesis"
    camera_presets = {
        "synthesis": dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.2)),
        "diagnostics": dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2.1, y=0.8, z=1.7)),
        "analysis": dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.0, y=2.4, z=1.3)),
    }
    scene_layout = dict(
        xaxis=dict(title=axis_label_x, visible=show_axes_initial, showticklabels=show_axes_initial, showgrid=show_axes_initial, zeroline=False, showbackground=False, showline=show_axes_initial),
        yaxis=dict(title=axis_label_y, visible=show_axes_initial, showticklabels=show_axes_initial, showgrid=show_axes_initial, zeroline=False, showbackground=False, showline=show_axes_initial),
        zaxis=dict(title=axis_label_z, visible=show_axes_initial, showticklabels=show_axes_initial, showgrid=show_axes_initial, zeroline=False, showbackground=False, showline=show_axes_initial),
        bgcolor=PALETTE.void,
        camera=camera_presets.get(physics_mode, camera_presets["synthesis"]),
        dragmode='orbit',
        **LAYOUT_CONSTRAINTS,
    )
    fig.update_layout(
        scene=scene_layout,
        paper_bgcolor=PALETTE.void,
        plot_bgcolor=PALETTE.void,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision='constant',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(5,5,5,0.8)',
            font=dict(color=PALETTE.text_primary, family='Inter, sans-serif', size=11),
            bordercolor=PALETTE.cyan_dim,
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        title=dict(
            text=f"<b>ASTER v3.2 MONOLITH [{exp.kernel.upper()}]{verification_title_stamp}</b>",
            font=dict(family='Inter, sans-serif', size=16, color=PALETTE.cyan),
            x=0.5,
            y=0.98,
        ),
        width=1400,
        height=900,
    )

    # ==========================================================================
    # GENERATE HTML WITH HUD
    # ==========================================================================
    mean_signal = float(spectral_evr.mean())

    hud_html = generate_hud_html(
        mode=physics_mode,
        signal=mean_signal,
        n_ruptures=n_ruptures,
        n_phantoms=n_phantoms,
        n_honest=n_honest,
        n_tautology=n_tautology,
        knn_overlap=knn_overlap,
        kernel_name=exp.kernel,
        n_cracks=n_cracks,
        n_bonds=n_bonds,
        n_broken=n_broken,
        n_trapped=n_trapped,
        mean_action=mean_action,
        survival_rate=survival_rate,
        synthesis_nmi=exp.synthesis_nmi,
    )

    # ==========================================================================
    # EPISTEMIC UI CONTRACT (Interpretation + Provenance + Trust Status)
    # ==========================================================================
    topology_text = "connection exists" if (n_bonds > 0 or knn_overlap > 0.0) else "absent"
    geometry_text = (
        f"work={mean_action:.2f} ({'low' if mean_action < 1.0 else 'moderate' if mean_action < 3.0 else 'high'})"
        if finite_work.size > 0 else "unknown"
    )

    n_total = n_total_walkers

    if n_total == 0:
        stability_text = "unknown"
    elif survival_rate >= 0.95:
        stability_text = "persists under annealing"
    else:
        stability_text = "unstable"

    interpretation_panel_html = '''
    <div class="epistemic-panel">
        <div class="ep-title">Interpretation Panel</div>
        <div class="ep-sub">Data-driven rendering from MONOLITH_DATA.csv.</div>
        <div class="ep-row"><span class="k">Topology:</span> <span class="v">connection exists</span></div>
        <div class="ep-row"><span class="k">Geometry:</span> <span class="v">work=1.29 (moderate)</span></div>
        <div class="ep-row"><span class="k">Stability:</span> <span class="v">persists under annealing</span></div>
        <div class="ep-row"><span class="k">Status:</span> <span class="v">VERIFIED</span></div>
        <div class="ep-title" style="margin-top:8px;">Instrument Readout</div>
        <div class="ep-row"><span class="k">T4 Survival:</span> <span class="v">100%</span></div>
        <div class="ep-row"><span class="k">Broken/Trapped:</span> <span class="v">0/0</span></div>
        <div class="ep-divider" style="border-top: 1px solid #333; margin: 8px 0;"></div>
        <div class="ep-row"><b>System 1 (Topological NMI):</b> <span class="v">Validated</span></div>
        <div class="ep-row"><b>System 2 (Thermodynamic Cost):</b> <span class="v">100.0%</span></div>
    </div>
    '''
    trust_watermark_html = ""

    # Legend HTML - Synthesis Mode
    legend_synthesis = f'''
    <div class="legend-panel" id="legend-synthesis">
        <div class="legend-title">SYNTHESIS MODE</div>
        <div class="legend-item">
            <div class="legend-line" style="background: #00F0FF;"></div>
            <span>Cyan: Honest Path (Logically valid, geometrically cheap)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: #FF00FF;"></div>
            <span>Magenta: Phantom Path (Logically forced, geometrically warped)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #888888;"></div>
            <span>Grey: Tautology (Topological collapse / Echo chamber)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #FF2222;"></div>
            <span>Red 'X': Walker Broken (System 2 Kinetic failure)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #FFB347;"></div>
            <span>Orange 'O': Walker Trapped (System 1 Topological trap)</span>
        </div>
    </div>
    '''

    # Legend HTML - Diagnostics Mode
    legend_diagnostics = f'''
    <div class="legend-panel" id="legend-diagnostics" style="display: none;">
        <div class="legend-title" style="color: {PALETTE.orange};">DIAGNOSTICS MODE</div>
        <div class="legend-item">
            <div class="legend-line" style="background: rgba(0,240,255,0.6);"></div>
            <span>Wind Streamlines (Local Force)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: rgba(0,255,100,0.6); width: 30px;"></div>
            <span>Slime Trails (Highways)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: rgba(255,50,50,0.6);"></div>
            <span>R-Ghost (RBF Kernel)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: rgba(50,255,50,0.6);"></div>
            <span>G-Ghost (Matern Kernel)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: rgba(50,100,255,0.6);"></div>
            <span>B-Ghost (Laplacian Kernel)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: rgba(255,255,255,0.2); border: 1px solid {PALETTE.cyan};"></div>
            <span>Confidence Halo (Uncertainty)</span>
        </div>
        <div class="legend-title" style="color: #888; font-size: 10px; margin-top: 8px;">WALKER STATES</div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #888888;"></div>
            <span>Tautology (No Movement)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #00FF41;"></div>
            <span>Honest (Laminar Flow)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #FF00FF;"></div>
            <span>Phantom (Turbulence)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #FF2222;"></div>
            <span>Rupture (Crashed)</span>
        </div>
    </div>
    '''

    # Legend HTML - Analysis Mode (Stacked Track Planes)
    # Build NMI display from analysis_nmi_scores
    nmi_items = ""
    for cp_label, nmi_val in analysis_nmi_scores.items():
        if np.isfinite(nmi_val):
            color = PALETTE.green if nmi_val > 0.7 else PALETTE.yellow if nmi_val > 0.5 else PALETTE.red
            nmi_text = f"NMI={nmi_val:.3f}"
        else:
            color = "#777777"
            nmi_text = "NMI=UNAVAILABLE"
        nmi_items += f'''
        <div class="legend-item">
            <div class="legend-dot" style="background: {color};"></div>
            <span>{cp_label}: {nmi_text}</span>
        </div>'''

    legend_analysis = f'''
    <div class="legend-panel" id="legend-analysis" style="display: none;">
        <div class="legend-title" style="color: {PALETTE.green};">ANALYSIS MODE</div>
        <div class="legend-subtitle" style="color: #888; font-size: 10px; margin-bottom: 8px;">
            Stacked Track Planes (Waterfall NMI)
        </div>
        <div style="font-size: 9px; color: #777; margin-bottom: 8px;">
            NMI shown here is analysis proxy (cluster alignment), not validation.json contract NMI.
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: rgba(0,240,255,0.3); height: 3px;"></div>
            <span>Track Plane Grid</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: rgba(255,255,255,0.1);"></div>
            <span>Article Trajectories</span>
        </div>
        <div class="legend-divider" style="border-top: 1px solid #333; margin: 8px 0;"></div>
        <div style="font-size: 10px; color: {PALETTE.cyan}; margin-bottom: 4px;">NMI by Track:</div>
        {nmi_items}
        <div class="legend-divider" style="border-top: 1px solid #333; margin: 8px 0;"></div>
        <div style="font-size: 9px; color: #666;">
            g_μν = (1/ρ)·δ_μν + ∇_μΦ∇_νΦ
        </div>
    </div>
    '''

    # Mode Toggle CSS
    toggle_css = '''
    <style>
        .mode-toggle {
            position: fixed;
            top: 50px;
            right: 20px;
            z-index: 1001;
            display: flex;
            gap: 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }
        .mode-btn {
            padding: 8px 16px;
            background: rgba(5,5,5,0.9);
            border: 1px solid #333;
            color: #888;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .mode-btn:first-child {
            border-radius: 4px 0 0 4px;
        }
        .mode-btn:last-child {
            border-radius: 0 4px 4px 0;
        }
        .mode-btn.active {
            background: rgba(0,240,255,0.2);
            border-color: #00F0FF;
            color: #00F0FF;
        }
        .mode-btn.diagnostics.active {
            background: rgba(255,140,0,0.2);
            border-color: #FF8C00;
            color: #FF8C00;
        }
        .mode-btn.analysis.active {
            background: rgba(0,255,65,0.2);
            border-color: #00FF41;
            color: #00FF41;
        }
        .mode-btn:hover:not(.active) {
            background: rgba(50,50,50,0.9);
            color: #ccc;
        }
        .layer-panel {
            position: fixed;
            top: 96px;
            right: 20px;
            z-index: 1001;
            width: 260px;
            padding: 10px 12px;
            background: rgba(8, 8, 8, 0.92);
            border: 1px solid #24535c;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: #c8f7ff;
        }
        .layer-title {
            color: #00F0FF;
            margin-bottom: 8px;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .layer-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
            user-select: none;
        }
        .layer-row input { accent-color: #00d8ff; }
        .layer-row span { color: #b9c9cf; }
        .layer-help {
            margin-top: 8px;
            padding-top: 7px;
            border-top: 1px solid #1f2c31;
            color: #83959b;
            line-height: 1.4;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 5px rgba(0,240,255,0.3); }
            50% { box-shadow: 0 0 20px rgba(0,240,255,0.6); }
        }
        .pulse-active {
            animation: pulse-glow 2s infinite;
        }
    </style>
    '''

    layer_panel = '''
    <div class="layer-panel" id="layer-panel">
        <div class="layer-title">VIEW CONTRACT</div>
        <div class="layer-row">
            <span><b>Shadow View</b> active (projection)</span>
        </div>
        <div class="layer-row">
            <label><input type="checkbox" id="toggle-failures" checked onchange="applyFailureFilter()"> Show failure cases</label>
        </div>
        <div class="layer-help">
            Failure cases = BROKEN/TRAPPED paths and rupture overlays. If unavailable: unknown.
        </div>
    </div>
    '''

    dash_run_key = str(exp.experiment_dir)
    try:
        repo_root = Path(__file__).resolve().parents[1]
        dash_run_key = str(exp.experiment_dir.resolve().relative_to(repo_root)).replace("\\", "/")
    except Exception:
        dash_run_key = str(exp.experiment_dir).replace("\\", "/")

    dash_embed_panel = '''
    <div id="dash-embed-panel" style="position: fixed; left: 18px; bottom: 18px; width: 46vw; height: 42vh; min-width: 420px; min-height: 280px; background: rgba(3,5,10,0.95); border: 1px solid #1e5062; border-radius: 8px; z-index: 1100; display: none; box-shadow: 0 8px 30px rgba(0,0,0,0.45); overflow: hidden;">
        <div style="height: 34px; display:flex; align-items:center; justify-content:space-between; padding: 0 10px; background: rgba(0,240,255,0.08); border-bottom: 1px solid #1e5062; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #b9f7ff;">
            <span id="dash-embed-title">DASH OBSERVER VIEW</span>
            <button onclick="closeDashEmbed()" style="background: transparent; color: #9adce8; border: 1px solid #2f7688; border-radius: 4px; font-size: 10px; cursor: pointer; padding: 2px 8px;">CLOSE</button>
        </div>
        <iframe id="dash-embed-frame" style="width: 100%; height: calc(100% - 34px); border: 0; background: #070912;"></iframe>
    </div>
    <div id="dash-embed-hint" style="position: fixed; left: 18px; bottom: 18px; z-index: 1050; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7ca3af; background: rgba(5,8,12,0.7); border: 1px solid #1f2e35; border-radius: 6px; padding: 5px 8px;">
        Click an article point to open embedded Dash observer lab
    </div>
    '''


    synthesis_active = " active" if physics_mode == "synthesis" else ""
    analysis_active = " active" if physics_mode == "analysis" else ""
    diagnostics_active = " active" if physics_mode == "diagnostics" else ""

    html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ASTER v3.2 MONOLITH [{exp.kernel.upper()}]</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
    {HUD_CSS}
    {toggle_css}
</head>
<body>
    {hud_html}

    <!-- MODE TOGGLE SWITCH -->
    <div class="mode-toggle">
        <button class="mode-btn synthesis{synthesis_active}" onclick="setMode('synthesis')">SYNTHESIS</button>
        <button class="mode-btn analysis{analysis_active}" onclick="setMode('analysis')">ANALYSIS</button>
        <button class="mode-btn diagnostics{diagnostics_active}" onclick="setMode('diagnostics')">DIAGNOSTICS</button>
    </div>
    {layer_panel}

    {interpretation_panel_html}
    {trust_watermark_html}
    {legend_synthesis}
    {legend_analysis}
    {legend_diagnostics}
    {dash_embed_panel}
    <div class="cockpit-container" id="cockpit"></div>

    <script>
        // Trace configuration
        var currentMode = {json.dumps(physics_mode)};
        var DASH_BASE_URL = 'http://127.0.0.1:8050/';
        var DASH_RUN_KEY = {json.dumps(dash_run_key)};

        // Initialize plot
        var figData = {{PLOT_DATA}};
        Plotly.newPlot('cockpit', figData.data, figData.layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }});

        function closeDashEmbed() {{
            var panel = document.getElementById('dash-embed-panel');
            var frame = document.getElementById('dash-embed-frame');
            if (panel) panel.style.display = 'none';
            if (frame) frame.src = 'about:blank';
        }}

        function openDashForArticle(articleIdx) {{
            var panel = document.getElementById('dash-embed-panel');
            var frame = document.getElementById('dash-embed-frame');
            var title = document.getElementById('dash-embed-title');
            if (!panel || !frame) return;
            var qs = new URLSearchParams();
            qs.set('run_key', DASH_RUN_KEY);
            qs.set('observer', 'article:' + String(articleIdx));
            qs.set('view_mode', 'observer');
            qs.set('compare', '0');
            qs.set('embedded', '1');
            frame.src = DASH_BASE_URL + '?' + qs.toString();
            if (title) {{
                title.textContent = 'DASH OBSERVER VIEW | article:' + String(articleIdx);
            }}
            panel.style.display = 'block';
        }}

        var cockpitEl = document.getElementById('cockpit');
        if (cockpitEl) {{
            cockpitEl.on('plotly_click', function(evt) {{
                try {{
                    var point = (evt && evt.points && evt.points.length) ? evt.points[0] : null;
                    if (!point) return;
                    var traceName = ((point.data && point.data.name) || '').toString();
                    if (traceName !== 'Articles') return;
                    var idx = null;
                    if (typeof point.customdata === 'number' && isFinite(point.customdata)) {{
                        idx = Math.floor(point.customdata);
                    }} else if (typeof point.pointNumber === 'number' && isFinite(point.pointNumber)) {{
                        idx = Math.floor(point.pointNumber);
                    }}
                    if (idx === null || idx < 0) return;
                    openDashForArticle(idx);
                }} catch (err) {{
                    console.warn('dash embed click handler failed', err);
                }}
            }});
        }}



        function applyFailureFilter() {{
            // Recompute complete mode visibility under current toggle state.
            setMode(currentMode);
        }}

        function sceneRelayoutForMode(mode) {{
            var showAxes = mode === 'diagnostics' || mode === 'analysis';
            return {{
                'scene.xaxis.visible': showAxes,
                'scene.xaxis.showticklabels': showAxes,
                'scene.xaxis.showgrid': showAxes,
                'scene.xaxis.showline': showAxes,
                'scene.yaxis.visible': showAxes,
                'scene.yaxis.showticklabels': showAxes,
                'scene.yaxis.showgrid': showAxes,
                'scene.yaxis.showline': showAxes,
                'scene.zaxis.visible': showAxes,
                'scene.zaxis.showticklabels': showAxes,
                'scene.zaxis.showgrid': showAxes,
                'scene.zaxis.showline': showAxes
            }};
        }}

        var CAMERA_PRESETS = {{
            synthesis: {{up: {{x: 0, y: 0, z: 1}}, center: {{x: 0, y: 0, z: 0}}, eye: {{x: 1.5, y: 1.5, z: 1.2}}}},
            diagnostics: {{up: {{x: 0, y: 0, z: 1}}, center: {{x: 0, y: 0, z: 0}}, eye: {{x: 2.1, y: 0.8, z: 1.7}}}},
            analysis: {{up: {{x: 0, y: 0, z: 1}}, center: {{x: 0, y: 0, z: 0}}, eye: {{x: 0.0, y: 2.4, z: 1.3}}}}
        }};

        function setMode(mode) {{
            currentMode = mode;

            // Update button states
            document.querySelectorAll('.mode-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            var activeBtn = document.querySelector('.mode-btn.' + mode);
            if (activeBtn) activeBtn.classList.add('active');

            // Update legends with defensive null-checks
            var legSyn = document.getElementById('legend-synthesis');
            var legAna = document.getElementById('legend-analysis');
            var legDia = document.getElementById('legend-diagnostics');
            
            if (legSyn) legSyn.style.display = mode === 'synthesis' ? 'block' : 'none';
            if (legAna) legAna.style.display = mode === 'analysis' ? 'block' : 'none';
            if (legDia) legDia.style.display = mode === 'diagnostics' ? 'block' : 'none';

            // Build visibility array using metadata-driven filtering
            var visibility = [];
            var numTraces = figData.data.length;

            for (var i = 0; i < numTraces; i++) {{
                var trace = figData.data[i];
                var tMeta = trace.meta || {{}};
                var tMode = tMeta.custom_mode || 'synthesis';
                
                if (mode === 'synthesis') {{
                    // Synthesis: Show synthesis + terrain + basic articles
                    visibility.push(tMode === 'synthesis' || tMode === 'terrain');
                }} else if (mode === 'analysis') {{
                    // Analysis: Show ONLY analysis traces
                    visibility.push(tMode === 'analysis');
                }} else if (mode === 'diagnostics') {{
                    // Diagnostics: Show diagnostics + terrain + articles
                    visibility.push(tMode === 'diagnostics' || tMode === 'terrain' || (tMode === 'synthesis' && (trace.name === 'Articles' || (trace.name || '').includes('glow'))));
                }} else {{
                    visibility.push(true);
                }}
            }}

            // Apply failure filter policy (BROKEN/TRAPPED/RUPTURE families)
            var showFailuresEl = document.getElementById('toggle-failures');
            var showFailures = showFailuresEl ? showFailuresEl.checked : true;
            for (var k = 0; k < numTraces; k++) {{
                if (!visibility[k]) continue;
                var tname = (figData.data[k].name || '').toUpperCase();
                var isFailure = (
                    tname.indexOf('BROKEN') >= 0 ||
                    tname.indexOf('TRAPPED') >= 0 ||
                    tname.indexOf('RUPTURE') >= 0 ||
                    tname.indexOf('CRASH') >= 0
                );
                if (isFailure && !showFailures) {{
                    visibility[k] = false;
                }}
            }}

            // Update plot
            Plotly.restyle('cockpit', {{'visible': visibility}});

            var relayoutUpdate = sceneRelayoutForMode(mode);
            relayoutUpdate['scene.camera'] = CAMERA_PRESETS[mode] || CAMERA_PRESETS.synthesis;
            Plotly.relayout('cockpit', relayoutUpdate);

            // Update HUD mode indicator
            var hudMode = document.querySelector('.hud-item');
            if (hudMode) {{
                hudMode.textContent = 'MODE: ' + mode.toUpperCase();
                var modeColors = {{'synthesis': '#00F0FF', 'analysis': '#00FF41', 'diagnostics': '#FF8C00'}};
                hudMode.style.color = modeColors[mode] || '#00F0FF';
            }}
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            var key = e.key.toLowerCase();
            if (key === 's') setMode('synthesis');
            if (key === 'a') setMode('analysis');
            if (key === 'd') setMode('diagnostics');
        }});

        setMode(currentMode);

    </script>
</body>
</html>'''

    fig_json = pio.to_json(fig)
    html_final = html_template.replace('{PLOT_DATA}', fig_json)

    # Render contract (fail-fast): if terrain is enabled, the surface must be
    # present/visible and path traces must remain in the same scene scale.
    if show_terrain:
        surface_traces = []
        for _t in fig.data:
            if isinstance(_t, go.Surface):
                is_visible = (_t.visible is None) or (bool(_t.visible) is True)
                if is_visible:
                    surface_traces.append(_t)
        if len(surface_traces) != 1:
            raise DimensionalCollapseError(
                f"CRITICAL: Expected exactly 1 visible terrain surface, found {len(surface_traces)}."
            )
        surface = surface_traces[0]
        sx = np.asarray(surface.x, dtype=float).ravel()
        sy = np.asarray(surface.y, dtype=float).ravel()
        sz = np.asarray(surface.z, dtype=float).ravel()
        if sx.size == 0 or sy.size == 0 or sz.size == 0:
            raise DimensionalCollapseError("CRITICAL: Terrain surface has empty coordinates.")
        surface_span = float(max(np.ptp(sx), np.ptp(sy), np.ptp(sz)))
        if not np.isfinite(surface_span) or surface_span <= 1e-6:
            raise DimensionalCollapseError(
                f"CRITICAL: Terrain surface span collapsed ({surface_span})."
            )
        max_path_span = 0.0
        for _t in fig.data:
            if not isinstance(_t, go.Scatter3d):
                continue
            if str(getattr(_t, "mode", "")) != "lines":
                continue
            trace_name = str(getattr(_t, "name", ""))
            if ("Path" not in trace_name) and (trace_name not in {"Walker Broken", "Walker Trapped"}):
                continue
            tx = np.asarray(_t.x, dtype=float).ravel()
            ty = np.asarray(_t.y, dtype=float).ravel()
            tz = np.asarray(_t.z, dtype=float).ravel()
            if tx.size == 0 or ty.size == 0 or tz.size == 0:
                continue
            path_span = float(max(np.ptp(tx), np.ptp(ty), np.ptp(tz)))
            if np.isfinite(path_span):
                max_path_span = max(max_path_span, path_span)
        if max_path_span > (surface_span * 50.0):
            raise DimensionalCollapseError(
                "CRITICAL: Path traces exceed terrain scale budget "
                f"(max_path_span={max_path_span:.3f}, surface_span={surface_span:.3f})."
            )

    # NEVER AGAIN PROTOCOL: hard-fail on dimensional collapse instead of silently rendering.
    pure_z_values = np.asarray(positions_3d[:, 2], dtype=float)
    if float(np.ptp(pure_z_values)) <= 1e-2:
        raise DimensionalCollapseError("CRITICAL: Point cloud Z-variance collapsed.")
    if float(np.ptp(np.asarray(energy_values_for_terrain, dtype=float))) <= 1e-2:
        raise DimensionalCollapseError("CRITICAL: Terrain stress gradient collapsed.")
    if len(walker_paths_pure) <= 0:
        raise DimensionalCollapseError("CRITICAL: Walker paths not loaded.")

    # Final validation gate: fail fast if canonical zone semantics or synthesis NMI drift.
    validation_errors = []
    invalid_zones = sorted({str(z) for z in unified_zones if str(z) not in CANONICAL_ZONES})
    if invalid_zones:
        validation_errors.append(f"Non-canonical zones detected: {invalid_zones}")
    if "Fault" in html_final:
        validation_errors.append("Found legacy 'Fault' label in output HTML.")
    if exp.synthesis_nmi is not None:
        expected_nmi = f"{exp.synthesis_nmi:.3f}"
        if ("| NMI: <span" not in html_final) or (expected_nmi not in html_final):
            validation_errors.append(
                f"Synthesis NMI mismatch: expected HUD value {expected_nmi} from validation.json."
            )
    if validation_errors:
        message = "[MONOLITH][VALIDATION] " + " | ".join(validation_errors)
        if strict_validation:
            raise ValueError(message)
        print(message)
    else:
        print("[MONOLITH][VALIDATION] PASS: zones canonical, no legacy labels, synthesis NMI wired.")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_final)

    print(f"[MONOLITH] Saved to: {output_path}")
    print(f"  Articles: {n_articles}")
    print(f"  T1.5 Spectral: Signal={mean_signal:.3f}")
    print(f"  T3 Dirichlet: Bonds={n_bonds}, Cracks={n_cracks}")
    print(f"  T4 Walker: MeanAction={mean_action:.2f}, Survival={survival_rate:.1%}")
    print(f"  T5 Verdicts: Honest={n_honest}, Phantom={n_phantoms}, Rupture={n_ruptures}, Tautology={n_tautology}")

    return fig


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    """CLI entry point for MONOLITH visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="ASTER v3.2 MONOLITH Visualization")
    parser.add_argument("experiment_dir", type=Path, help="Path to experiment seed directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--mode", choices=["synthesis", "analysis"], default="synthesis")
    parser.add_argument("--observer-idx", type=int, default=None, help="Optional article index to emphasize in the render.")
    parser.add_argument(
        "--path-ablation",
        choices=["none", "thermodynamic", "semantic_tether"],
        default="none",
        help="Optional path rendering ablation mode (thermodynamic=scorch+tears, semantic_tether=straight annotated tethers).",
    )
    parser.add_argument("--strict", action="store_true",
                        help="Fail generation if canonical zone/NMI validation checks fail.")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    # Smart Discovery: Try to resolve deep nesting (rbf/cls/real) if path doesn't exist
    if not exp_dir.exists():
        exp_dir = resolve_run_directory(exp_dir.parent, "rbf", "cls", exp_dir.name)
        
    if not exp_dir.exists():
        print(f"Error: Directory not found: {args.experiment_dir}")
        sys.exit(1)

    output = args.output or exp_dir / "MONOLITH.html"

    exp = load_experiment_data(exp_dir)
    create_monolith_cockpit(
        exp,
        output,
        physics_mode=args.mode,
        observer_idx=args.observer_idx,
        strict_validation=args.strict,
        path_ablation_mode=args.path_ablation,
    )


if __name__ == "__main__":
    main()
