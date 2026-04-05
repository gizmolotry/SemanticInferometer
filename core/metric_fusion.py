"""
metric_fusion.py - Implements the Track 3 Unified Metric Tensor logic.

Fuses Track 1.5 (Gradients) and Track 2 (Density) to generate unified metrics
for visualization, including density, stress, z_height, zones, and color codes.

CRITICAL 'NO BUTTERFLY' CONSTRAINTS:
1. NON-DESTRUCTIVE: Only appends new columns.
2. PRESERVE FLOW: Does not alter existing function signatures.
3. COMPATIBILITY: Output 'MONOLITH_DATA.csv' is a superset of the input metadata.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List
import json
import argparse
from .thermo_config import ThermodynamicConfig
from .artifact_ledger import ArtifactContract


def _canonicalize_track5_verdict(raw_verdict: object) -> str:
    text = str(raw_verdict or "").strip().upper()
    if text in {"TYPE_2_RUPTURE", "TYPE 2 RUPTURE", "TRAPPED", "FAILED"}:
        return "TAUTOLOGY"
    if text in {"TYPE_1_RUPTURE", "TYPE 1 RUPTURE", "RUPTURE", "BROKEN"}:
        return "PHANTOM"
    if text in {"HONEST", "PHANTOM", "TAUTOLOGY"}:
        return text
    return "UNKNOWN"


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return np.full(arr.shape, 0.5, dtype=float)
    lo = float(np.percentile(finite, 10.0))
    hi = float(np.percentile(finite, 90.0))
    if hi <= lo + 1e-12:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo + 1e-12:
        return np.full(arr.shape, 0.5, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _compute_density_field(
    embeddings: np.ndarray,
    *,
    knn_k: int,
    epsilon: float,
) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=float)
    n = int(arr.shape[0])
    if n <= 1:
        raise ValueError("Need at least 2 samples to compute a density field.")

    effective_k = int(max(1, min(int(knn_k), n - 1)))
    print(f"Calculating density with KNN (requested_k={knn_k}, effective_k={effective_k})...")

    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(arr)
    distances, _ = nn.kneighbors(arr)

    neighbor_distances = np.asarray(distances[:, 1 : effective_k + 1], dtype=float)
    neighbor_distances = np.nan_to_num(neighbor_distances, nan=np.inf, posinf=np.inf, neginf=0.0)
    mean_neighbor_distance = np.mean(neighbor_distances, axis=1)
    density = 1.0 / (mean_neighbor_distance + epsilon)
    density = _robust_unit_interval(density)

    # Never silently flatten the manifold on small/degenerate runs. If the local
    # KNN estimate collapses, fall back to inverse mean distance over the full
    # point cloud before giving up.
    if n >= 3 and float(np.ptp(density)) <= 1e-9:
        pairwise = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
        finite_mask = np.isfinite(pairwise)
        diag_mask = np.eye(n, dtype=bool)
        pairwise = np.where(diag_mask, np.nan, pairwise)
        pairwise = np.where(finite_mask, pairwise, np.nan)
        mean_pairwise_distance = np.nanmean(pairwise, axis=1)
        if np.isfinite(mean_pairwise_distance).any():
            fallback_density = 1.0 / (np.nan_to_num(mean_pairwise_distance, nan=np.nanmax(mean_pairwise_distance)) + epsilon)
            density = _robust_unit_interval(fallback_density)
            print("  [WARN] Local KNN density collapsed; using inverse mean pairwise distance fallback.")

    if n >= 3 and float(np.ptp(density)) <= 1e-9:
        raise ValueError(
            "Density field collapsed after adaptive KNN and pairwise-distance fallback; refusing to emit flat manifold metrics."
        )
    return density.astype(float)


def calculate_unified_metric(
    embeddings_path: Path,
    gradients_path: Path,
    metadata_path: Path,
    output_path: Path,
    knn_k: int = 20,
    z_stress_factor: float = 1.2, # Retained for signature compatibility
    z_density_factor: float = 0.8, # Retained for signature compatibility
) -> pd.DataFrame:
    """
    Fuses Track 1.5 (Gradients) and Track 2 (Density) to calculate unified metrics.
    """
    # 0. ENFORCE CONTRACT
    ArtifactContract(embeddings_path.parent).verify()

    print(f"Loading data: {embeddings_path}, {gradients_path}, {metadata_path}")
    thermo_config = ThermodynamicConfig()

    # 1. Load 'embeddings.npy', 'gradients.npy', and 'articles.csv'
    embeddings = np.load(embeddings_path)
    gradients = np.load(gradients_path)
    if Path(metadata_path).suffix == ".json":
        metadata_df = pd.read_json(metadata_path)
    else:
        metadata_df = pd.read_csv(metadata_path)

    # Ensure embeddings and gradients match metadata length
    if len(embeddings) != len(metadata_df) or len(gradients) != len(metadata_df):
        raise ValueError(
            "Mismatch in number of articles between embeddings, gradients, and metadata."
            f"Embeddings: {len(embeddings)}, Gradients: {len(gradients)}, Metadata: {len(metadata_df)}"
        )

    # 2. Calculate DENSITY (rho) using adaptive KNN. Never silently replace the
    # manifold with a uniform density field just because the corpus is small.
    density = _compute_density_field(
        embeddings,
        knn_k=knn_k,
        epsilon=float(thermo_config.density_clamp_min),
    )

    # 3. Calculate STRESS as a varying article-level scalar.
    # `spectral_u_axis.npy` can be unit-normalized, which makes its L2 norm
    # effectively constant and collapses the manifold skin. Prefer the existing
    # spectral probe magnitudes when available.
    print("Calculating stress...")
    probe_magnitudes_path = embeddings_path.parent / "spectral_probe_magnitudes.npy"
    raw_stress = None
    if probe_magnitudes_path.exists():
        try:
            probe_magnitudes = np.load(probe_magnitudes_path)
            if (
                isinstance(probe_magnitudes, np.ndarray)
                and probe_magnitudes.ndim == 2
                and probe_magnitudes.shape[0] == len(metadata_df)
            ):
                raw_stress = np.linalg.norm(np.asarray(probe_magnitudes, dtype=float), axis=1).astype(float)
                print(f"  [OK] Using spectral probe magnitudes from {probe_magnitudes_path.name} for stress.")
        except Exception as probe_err:
            print(f"  [WARN] Failed to load spectral probe magnitudes for stress: {probe_err}")
            raw_stress = None
    if raw_stress is None:
        raw_stress = np.linalg.norm(np.asarray(gradients, dtype=float), axis=1).astype(float)
        print("  [WARN] Falling back to gradient-vector norm for stress.")
    stress = _robust_unit_interval(raw_stress)

    # 4. Calculate Z_HEIGHT from soft-floored log-density potential:
    #    Z = -log(rho + epsilon_z), preserving raw potential scale.
    print("Calculating Z_HEIGHT...")
    epsilon_z = thermo_config.epsilon_z
    z_potential = -np.log(density + epsilon_z)
    z_height = z_potential
    if len(z_height) >= 3 and float(np.ptp(z_height)) <= 1e-9:
        raise ValueError("Z-height collapsed after density computation; refusing to emit flat manifold metrics.")

    # 5. Calculate ZONES (Bridge/Swamp/Tightrope/Void) using ABSOLUTE THRESHOLDS
    print("Classifying zones (Bridge/Swamp/Tightrope/Void)...")
    # ASTER v3.2 Strict Physical Thresholds (No percentiles)
    density_thresh = 0.5
    stress_thresh = 0.5

    zones = []
    color_codes = [] 
    for i in range(len(metadata_df)):
        if density[i] >= density_thresh and stress[i] < stress_thresh:
            zones.append("Bridge")
            color_codes.append("#00F0FF")
        elif density[i] >= density_thresh and stress[i] >= stress_thresh:
            zones.append("Swamp")
            color_codes.append("#9932CC")
        elif density[i] < density_thresh and stress[i] < stress_thresh:
            zones.append("Tightrope")
            color_codes.append("#FFFFCC")
        else: 
            zones.append("Void")
            color_codes.append("#FF0000")

    # 5.5. Calculate VERDICTS (ASTER v3.2 Strict Physical Handoff)
    print("Classifying verdicts (Honest/Phantom/Rupture/Tautology)...")
    
    # Try to load walker work integrals and states for actual physics
    walker_work_path = embeddings_path.parent / "walker_work_integrals.npy"
    walker_states_path = embeddings_path.parent / "walker_states.json"
    phantom_verdicts_path = embeddings_path.parent / "phantom_verdicts.json"
    
    if walker_work_path.exists():
        w_actual = np.load(walker_work_path)
    else:
        print(f"  [WARN] {walker_work_path.name} missing. Using zero-work fallback.")
        w_actual = np.zeros(len(metadata_df))
    
    walker_states = None
    if walker_states_path.exists():
        with open(walker_states_path, 'r') as f:
            walker_states = json.load(f)
        print(f"  [OK] Using actual walker states from {walker_states_path.name}")
    else:
        print(f"  [WARN] walker_states.json missing. Falling back to work-only classification.")

    phantom_verdicts = None
    if phantom_verdicts_path.exists():
        with open(phantom_verdicts_path, 'r') as f:
            phantom_verdicts = json.load(f)
        if not isinstance(phantom_verdicts, list) or len(phantom_verdicts) != len(metadata_df):
            print(f"  [WARN] phantom_verdicts.json shape mismatch. Ignoring verdict ledger.")
            phantom_verdicts = None
        else:
            print(f"  [OK] Using Track 5 verdict ledger from {phantom_verdicts_path.name}")

    # Divergence Ratio Logic (Panic Function)
    d_spectral = None
    d_spectral_path = embeddings_path.parent / "d_spectral.npy"
    if d_spectral_path.exists():
        d_spectral = np.asarray(np.load(d_spectral_path), dtype=float).reshape(-1)
        if d_spectral.shape[0] != len(metadata_df):
            print(
                f"  [WARN] {d_spectral_path.name} length mismatch "
                f"({d_spectral.shape[0]} vs {len(metadata_df)}). Ignoring persisted spectral distance."
            )
            d_spectral = None
        else:
            print(f"  [OK] Using persisted Track 1.5 spectral distance from {d_spectral_path.name}")

    if d_spectral is None and phantom_verdicts is not None:
        extracted = [
            float(payload.get("d_spectral", payload.get("d", np.nan)))
            for payload in phantom_verdicts
        ]
        d_candidate = np.asarray(extracted, dtype=float).reshape(-1)
        if d_candidate.shape[0] == len(metadata_df) and np.isfinite(d_candidate).any():
            d_spectral = d_candidate
            print("  [OK] Using Track 5 verdict ledger for persisted spectral distance.")

    if d_spectral is None:
        raise FileNotFoundError(
            "CRITICAL ERROR: Missing canonical Track 1.5 spectral distance "
            f"({d_spectral_path.name} or phantom_verdicts.json with d_spectral)."
        )

    d_spectral = np.clip(np.asarray(d_spectral, dtype=float), 0.1, None)

    # 1) Calculate absolute curvature penalty for surviving walkers.
    valid_indices = []
    for i in range(len(metadata_df)):
        raw_state = walker_states[i] if walker_states is not None else "UNKNOWN"
        if isinstance(raw_state, dict):
            state = _canonicalize_track5_verdict(raw_state.get("label", raw_state.get("status", "UNKNOWN")))
        else:
            state = _canonicalize_track5_verdict(raw_state)
        if state in {"HONEST", "PHANTOM", "TAUTOLOGY"}:
            valid_indices.append(i)

    delta_values = []
    for i in valid_indices:
        d = d_spectral[i] + 1e-8
        delta_values.append(w_actual[i] / d)

    # 2) Fit 1D K-Means to find natural energetic states.
    delta_array = np.array(delta_values).reshape(-1, 1)
    sorted_centers = np.array([1.0, 5.0, 10.0], dtype=float)
    min_stable_samples = 10
    threshold_mode = "kmeans_3cluster"
    if len(delta_array) >= min_stable_samples:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(delta_array)
        sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
        threshold_tautology = sorted_centers[0] + (sorted_centers[1] - sorted_centers[0]) / 2.0
        threshold_honest = sorted_centers[1] + (sorted_centers[2] - sorted_centers[1]) / 2.0
    else:
        # Sparse-state fallback: use stable calibrated thresholds to avoid
        # run-local quantile drift on tiny/failed runs.
        threshold_tautology, threshold_honest = 1.0, 10.0
        threshold_mode = "sparse_fixed_calibrated"

    verdicts = []
    output_delta = []
    output_d_spectral = []
    output_w_actual = []
    for i in range(len(metadata_df)):
        if phantom_verdicts is not None:
            verdict_payload = phantom_verdicts[i]
            verdict = _canonicalize_track5_verdict(verdict_payload.get("verdict"))
            if verdict != "UNKNOWN":
                verdicts.append(verdict)
                output_delta.append(float(verdict_payload.get("delta", w_actual[i] / (d_spectral[i] + 1e-8))))
                output_d_spectral.append(float(verdict_payload.get("d_spectral", verdict_payload.get("d", d_spectral[i]))))
                output_w_actual.append(float(verdict_payload.get("w_actual", verdict_payload.get("w", w_actual[i]))))
                continue

        raw_state = walker_states[i] if walker_states is not None else "UNKNOWN"
        if isinstance(raw_state, dict):
            state = _canonicalize_track5_verdict(raw_state.get("label", raw_state.get("status", "UNKNOWN")))
        else:
            state = _canonicalize_track5_verdict(raw_state)

        delta = w_actual[i] / (d_spectral[i] + 1e-8)
        if state in {"HONEST", "PHANTOM", "TAUTOLOGY"}:
            verdicts.append(state)
            output_delta.append(float(delta))
            output_d_spectral.append(float(d_spectral[i]))
            output_w_actual.append(float(w_actual[i]))
            continue

        if delta < threshold_tautology:
            verdicts.append("TAUTOLOGY")
        elif delta <= threshold_honest:
            verdicts.append("HONEST")
        else:
            verdicts.append("PHANTOM")
        output_delta.append(float(delta))
        output_d_spectral.append(float(d_spectral[i]))
        output_w_actual.append(float(w_actual[i]))

    # 6. Save the result as 'MONOLITH_DATA.csv' with new columns:
    #    'density', 'stress', 'z_height', 'zone', 'color_code', 'verdict'.
    print(f"Appending new columns and saving to {output_path}...")
    metadata_df['density'] = density
    metadata_df['stress'] = stress
    metadata_df['z_height'] = z_height
    metadata_df['zone'] = zones
    metadata_df['color_code'] = color_codes
    metadata_df['verdict'] = verdicts
    metadata_df['delta'] = output_delta
    metadata_df['d_spectral'] = output_d_spectral
    metadata_df['w_actual'] = output_w_actual

    metadata_df.to_csv(output_path, index=False)
    print("Unified metric calculation complete and saved.")
    print(f"LEARNED CENTROIDS: {sorted_centers}")
    print(
        f"THRESHOLDS: Tautology < {threshold_tautology:.2f} | "
        f"Honest <= {threshold_honest:.2f} | Phantom > {threshold_honest:.2f} "
        f"(mode={threshold_mode})"
    )

    return metadata_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse metrics for MONOLITH visualization.")
    parser.add_argument("experiment_dir", type=str, help="Path to the experiment directory (e.g., experiments_20260212_180018/rbf/real/seed_43)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)

    # Define paths
    embeddings_path = exp_dir / "features.npy"
    gradients_path = exp_dir / "spectral_u_axis.npy"
    
    # Prioritize 'articles_with_sources.csv' if it exists for richer metadata
    articles_with_sources_path = exp_dir / "articles_with_sources.csv"
    article_metadata_csv = exp_dir / "article_metadata.csv"
    
    if articles_with_sources_path.exists():
        metadata_path = articles_with_sources_path
        print(f"Using rich metadata from: {metadata_path}")
    elif article_metadata_csv.exists():
        metadata_path = article_metadata_csv
        print(f"Using metadata from: {metadata_path}")
    else: # Fallback to original metadata logic
        metadata_path = exp_dir / "article_metadata.json" # Fallback to original metadata file
        if metadata_path.exists() and metadata_path.suffix == '.json':
            print(f"Converting {metadata_path} to temporary CSV for processing...")
            with open(metadata_path, 'r') as f:
                metadata_list = json.load(f)
            temp_csv_path = exp_dir / "articles.csv"
            pd.DataFrame(metadata_list).to_csv(temp_csv_path, index=False)
            metadata_path = temp_csv_path
        elif not metadata_path.exists():
            print(f"Error: {metadata_path} not found.")
            exit(1)

    output_file = exp_dir / "MONOLITH_DATA.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running calculate_unified_metric for {exp_dir.name}...")
    result_df = calculate_unified_metric(
        embeddings_path=embeddings_path,
        gradients_path=gradients_path,
        metadata_path=metadata_path,
        output_path=output_file,
    )
    print(f"Generated MONOLITH_DATA.csv with {len(result_df.columns)} columns.")
    print(result_df.head())
    print(result_df["verdict"].value_counts())
