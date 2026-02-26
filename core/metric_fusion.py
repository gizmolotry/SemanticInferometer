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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Tuple, List
import json
import argparse
from .thermo_config import ThermodynamicConfig

def calculate_unified_metric(
    embeddings_path: Path,
    gradients_path: Path,
    metadata_path: Path,
    output_path: Path,
    knn_k: int = 20,
    z_stress_factor: float = 1.2, # Updated factor
    z_density_factor: float = 0.8, # Updated factor
) -> pd.DataFrame:
    """
    Fuses Track 1.5 (Gradients) and Track 2 (Density) to calculate unified metrics.

    Args:
        embeddings_path: Path to the .npy file containing embeddings (e.g., CLS features).
        gradients_path: Path to the .npy file containing gradient vectors (Track 1.5).
        metadata_path: Path to the .csv file containing article metadata.
        output_path: Path to save the resulting MONOLITH_DATA.csv.
        knn_k: Number of neighbors for KNN density calculation.
        z_stress_factor: Multiplier for stress in Z_HEIGHT calculation.
        z_density_factor: Multiplier for (1-density) in Z_HEIGHT calculation.

    Returns:
        A pandas DataFrame with the original metadata and appended unified metric columns.
    """
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

    # 3. Calculate STRESS (grad_norm) as the L2 norm of the gradient vectors.
    # Use robust scaling + power-law stretch so high-friction extremes are visible.
    print("Calculating stress (L2 norm of gradients)...")
    raw_stress = np.linalg.norm(gradients, axis=1).astype(float)
    stress_scaler = RobustScaler()
    stress_robust = stress_scaler.fit_transform(raw_stress.reshape(-1, 1)).reshape(-1)
    # Shift to non-negative before power transform.
    stress_shifted = stress_robust - stress_robust.min()
    stress_power = np.power(stress_shifted + thermo_config.density_clamp_min, 0.75)
    stress = (
        (stress_power - stress_power.min()) /
        (stress_power.max() - stress_power.min() + thermo_config.density_clamp_min)
    )

    # 4. Calculate Z_HEIGHT from clamped log-density potential:
    #    Z = -log(rho + epsilon), then min-max scale to [0, max_z].
    print("Calculating Z_HEIGHT...")
    epsilon = thermo_config.density_clamp_min
    z_potential = -np.log(np.clip(density, epsilon, None))
    max_z = float(z_stress_factor + z_density_factor)
    z_scaler = MinMaxScaler(feature_range=(0.0, max_z))
    z_height = z_scaler.fit_transform(z_potential.reshape(-1, 1)).reshape(-1)

    # 5. Calculate ZONES (Bridge/Swamp/Tightrope/Void) using DYNAMIC MEDIAN THRESHOLDS
    print("Classifying zones (Bridge/Swamp/Tightrope/Void)...")
    # Dynamic median thresholds to force even distribution
    density_median = np.percentile(density, 50)
    stress_median = np.percentile(stress, 50)

    zones = []
    color_codes = [] # Also generate color codes based on zone for convenience
    for i in range(len(metadata_df)):
        if density[i] >= density_median and stress[i] < stress_median:
            zones.append("Bridge")
            color_codes.append("#00F0FF") # Cyan
        elif density[i] >= density_median and stress[i] >= stress_median:
            zones.append("Swamp")
            color_codes.append("#9932CC") # Purple
        elif density[i] < density_median and stress[i] < stress_median:
            zones.append("Tightrope")
            color_codes.append("#FFFFCC") # Yellow
        else: # density[i] < density_median and stress[i] >= stress_median
            zones.append("Void")
            color_codes.append("#FF0000") # Red (or #1A0000 for dark void)

    # 6. Save the result as 'MONOLITH_DATA.csv' with new columns:
    #    'density', 'stress', 'z_height', 'zone', 'color_code'.
    print(f"Appending new columns and saving to {output_path}...")
    metadata_df['density'] = density
    metadata_df['stress'] = stress
    metadata_df['z_height'] = z_height
    metadata_df['zone'] = zones
    metadata_df['color_code'] = color_codes

    metadata_df.to_csv(output_path, index=False)
    print("Unified metric calculation complete and saved.")

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
    if articles_with_sources_path.exists():
        metadata_path = articles_with_sources_path
        print(f"Using rich metadata from: {metadata_path}")
    else: # Fallback to original metadata logic if articles_with_sources.csv does not exist
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
