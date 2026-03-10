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
    if text in {"TYPE_1_RUPTURE", "TYPE_2_RUPTURE", "RUPTURE"}:
        return "RUPTURE"
    if text in {"HONEST", "PHANTOM", "TAUTOLOGY"}:
        return text
    return "UNKNOWN"


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
    metadata_df = pd.read_csv(metadata_path)

    # Ensure embeddings and gradients match metadata length
    if len(embeddings) != len(metadata_df) or len(gradients) != len(metadata_df):
        raise ValueError(
            "Mismatch in number of articles between embeddings, gradients, and metadata."
            f"Embeddings: {len(embeddings)}, Gradients: {len(gradients)}, Metadata: {len(metadata_df)}"
        )

    # 2. Calculate DENSITY (rho) using KNN (k=20). Normalize 0-1.
    print(f"Calculating density with KNN (k={knn_k})...")
    if len(embeddings) > knn_k:
        nn = NearestNeighbors(n_neighbors=knn_k + 1, metric='euclidean')
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        # Density is inversely proportional to average distance to k-th neighbor
        # We take the distance to the k-th neighbor (knn_k index, as 0th is self)
        k_distances = distances[:, knn_k]
        density = 1.0 / (k_distances + thermo_config.density_clamp_min)  # Add epsilon to prevent division by zero
        density = (
            (density - density.min()) /
            (density.max() - density.min() + thermo_config.density_clamp_min)
        )  # Normalize 0-1
    else:
        print(f"Warning: Not enough samples ({len(embeddings)}) for KNN k={knn_k}. Assigning uniform density.")
        density = np.ones(len(embeddings)) * 0.5 # Default to mid-density

    # 3. Calculate STRESS (grad_norm) as the raw L2 norm of gradient vectors.
    # NOTE: Do not normalize/compress this value in fusion; keep raw thermodynamic scale.
    print("Calculating stress (L2 norm of gradients)...")
    raw_stress = np.linalg.norm(gradients, axis=1).astype(float)
    stress = raw_stress

    # 4. Calculate Z_HEIGHT from soft-floored log-density potential:
    #    Z = -log(rho + epsilon_z), preserving raw potential scale.
    print("Calculating Z_HEIGHT...")
    epsilon_z = thermo_config.epsilon_z
    z_potential = -np.log(density + epsilon_z)
    z_height = z_potential

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
    
    if not walker_work_path.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Physics payload missing (work). {walker_work_path.name}")
    
    w_actual = np.load(walker_work_path)
    
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
    centroid_2d = np.mean(embeddings[:, :2], axis=0)
    d_spectral = np.linalg.norm(embeddings[:, :2] - centroid_2d, axis=1)
    d_spectral = np.clip(d_spectral, 0.1, None)

    rupture_states = {
        "BROKEN",
        "TRAPPED",
        "RUPTURE",
        "TYPE_1_RUPTURE",
        "TYPE_2_RUPTURE",
        "FAILED",
    }

    # 1) Calculate absolute curvature penalty for surviving walkers.
    valid_indices = []
    for i in range(len(metadata_df)):
        raw_state = walker_states[i] if walker_states is not None else "UNKNOWN"
        if isinstance(raw_state, dict):
            state = str(raw_state.get("status", "UNKNOWN")).upper()
        else:
            state = str(raw_state).upper()
        if state not in rupture_states and state != "UNKNOWN":
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
    for i in range(len(metadata_df)):
        if phantom_verdicts is not None:
            verdict = _canonicalize_track5_verdict(phantom_verdicts[i].get("verdict"))
            if verdict != "UNKNOWN":
                verdicts.append(verdict)
                continue

        raw_state = walker_states[i] if walker_states is not None else "UNKNOWN"
        if isinstance(raw_state, dict):
            state = str(raw_state.get("label", raw_state.get("status", "UNKNOWN"))).upper()
        else:
            state = str(raw_state).upper()

        if state in {"BROKEN", "TYPE_1_RUPTURE"}:
            verdicts.append("TYPE_1_RUPTURE") # Kinetic crash
            continue
        if state in {"TRAPPED", "TYPE_2_RUPTURE", "RUPTURE", "FAILED"}:
            verdicts.append("TYPE_2_RUPTURE") # Topological/behavioral failure
            continue

        delta = w_actual[i] / (d_spectral[i] + 1e-8)
        if delta < threshold_tautology:
            verdicts.append("TAUTOLOGY")
        elif delta <= threshold_honest:
            verdicts.append("HONEST")
        else:
            verdicts.append("PHANTOM")

    # 6. Save the result as 'MONOLITH_DATA.csv' with new columns:
    #    'density', 'stress', 'z_height', 'zone', 'color_code', 'verdict'.
    print(f"Appending new columns and saving to {output_path}...")
    metadata_df['density'] = density
    metadata_df['stress'] = stress
    metadata_df['z_height'] = z_height
    metadata_df['zone'] = zones
    metadata_df['color_code'] = color_codes
    metadata_df['verdict'] = verdicts

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
