"""
analysis/ablation.py - The Judge for Titan Protocol Thesis Pipeline.

Calculates Clustering Quality (NMI) across three evolutionary stages to compare
Physics-Informed Manifold (Stage 3) clustering against raw Euclidean geometry (Stage 1).
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from typing import Dict, List, Tuple, Any

# --- Ground Truth Mapping ---
def get_tribe(publication: str) -> int:
    """
    Maps publication names to tribe labels (0: RED, 1: BLUE, 2: GRAY).
    """
    if not isinstance(publication, str):
        return 2 # GRAY TEAM for non-string (e.g., NaN) inputs

    publication = publication.lower()

    red_team = [
        "al jazeera", "electronic intifada", "haaretz", "guardian",
        "intercept", "mondoweiss", "middle east eye", "mintpress news"
    ]
    blue_team = [
        "arutz sheva", "jpost", " israel hayom", "fox news", "kohelet policy forum",
        "breitbart", "daily wire", "townhall"
    ]

    if any(p in publication for p in red_team):
        return 0  # RED TEAM
    elif any(p in publication for p in blue_team):
        return 1  # BLUE TEAM
    else:
        return 2  # GRAY TEAM (e.g., Reuters, BBC, CNN, NYT)

def calculate_nmi_score(X: np.ndarray, y: np.ndarray, n_clusters: int = 3) -> float:
    """
    Calculates NMI score using KMeans clustering.
    """
    if len(X) < n_clusters:
        return 0.0 # Cannot cluster with fewer samples than clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X)
    return normalized_mutual_info_score(y, labels_pred)

def run_ablation_analysis(data_path: Path) -> Dict[str, Any]:
    """
    Runs the ablation analysis across three stages and reports NMI scores.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load verification report once from nearest parent experiment directory.
    verify_report_path = None
    for parent in data_path.parents:
        candidate = parent / "verification_report.json"
        if candidate.exists():
            verify_report_path = candidate
            break
    verify_status = "UNKNOWN"
    is_verified = False
    fail_reasons = []
    
    if verify_report_path is not None:
        try:
            with open(verify_report_path, "r") as f:
                vdata = json.load(f)
            
            # Find the status for the current layer
            # data_path structure: .../exp_dir/kernel/channel/real/MONOLITH_DATA.csv
            # Supports both layout A (.../<kernel>/<channel>/<corpus>/MONOLITH_DATA.csv)
            # and layout B (.../<group>/<seed_dir>/MONOLITH_DATA.csv) via fallback matching.
            current_layer_id = f"{data_path.parents[2].name}/{data_path.parents[1].name}"
            
            layer_entry = next((l for l in vdata.get("layers", []) if l.get("layer_id") == current_layer_id), None)
            if layer_entry:
                verify_status = layer_entry.get("status")
                is_verified = (verify_status == "VERIFIED")
                fail_reasons = layer_entry.get("fail_reasons", [])
            else:
                fallback_key = f"{data_path.parents[1].name}/{data_path.parents[0].name.split('_seed')[0]}"
                layer_entry = next((l for l in vdata.get("layers", []) if l.get("layer_id") == fallback_key), None)
                if layer_entry:
                    verify_status = layer_entry.get("status")
                    is_verified = (verify_status == "VERIFIED")
                    fail_reasons = layer_entry.get("fail_reasons", [])
                else:
                    verify_status = "LAYER_NOT_FOUND_IN_REPORT"
        except Exception as e:
            verify_status = f"ERROR_READING_REPORT: {str(e)}"

    print("\n" + "="*70)
    print(f"TITAN PROTOCOL ABLATION ANALYSIS: {data_path.parents[2].name}/{data_path.parents[1].name}")
    print(f"Verification Status: {verify_status}")
    if not is_verified:
        print(f"WARNING: LAYER INTEGRITY NOT VERIFIED.")
        for reason in fail_reasons:
            print(f"  - {reason}")
    print("="*70)

    df = pd.read_csv(data_path)

    # Filter out rows with NaN in critical columns
    df_clean = df.dropna(subset=['x_proj', 'y_proj', 'z_height', 'stress', 'density'])

    if df_clean.empty:
        print("Warning: Cleaned DataFrame is empty after dropping NaNs. Cannot perform analysis.")
        return {
            "stage_1_nmi": 0.0,
            "stage_2_nmi": 0.0,
            "stage_3_nmi": 0.0,
            "delta_nmi": 0.0,
            "retained_percentage": 0.0,
            "verification_status": verify_status,
            "is_verified": is_verified
        }

    # Define Ground Truth
    if 'publication' in df_clean.columns:
        df_clean['tribe_label'] = df_clean['publication'].apply(get_tribe)
    elif 'source' in df_clean.columns:
        df_clean['tribe_label'] = df_clean['source'].apply(get_tribe)
    else:
        print("Warning: 'publication' or 'source' column not found for ground truth. Assigning dummy tribe labels.")
        df_clean['tribe_label'] = np.random.randint(0, 3, size=len(df_clean))

    y = df_clean['tribe_label'].values

    x_coords = df_clean['x_proj'].values
    y_coords = df_clean['y_proj'].values
    z_coords = df_clean['z_height'].values
    stress_values = df_clean['stress'].values
    density_values = df_clean['density'].values

    # Combine into a 3D coordinate array
    X_baseline = np.column_stack([x_coords, y_coords, z_coords])

    # 3. Compute Stage 1 NMI (The Baseline)
    nmi_stage_1 = calculate_nmi_score(X_baseline, y)

    # 4. Compute Stage 2 NMI (The Stress Test)
    X_weighted = np.copy(X_baseline)
    X_weighted[:, 0] = x_coords * (1 + stress_values * 5.0)
    X_weighted[:, 1] = y_coords * (1 + stress_values * 5.0)
    X_weighted[:, 2] = z_coords * (1 + stress_values * 5.0)
    nmi_stage_2 = calculate_nmi_score(X_weighted, y)

    # 5. Compute Stage 3 NMI (The Reality Check)
    df_stage_3 = df_clean[df_clean['density'] >= 0.2]

    if df_stage_3.empty:
        print("Warning: Stage 3 DataFrame is empty after filtering by density. Cannot perform analysis.")
        nmi_stage_3 = 0.0
        retained_percentage = 0.0
    else:
        y_stage_3 = df_stage_3['tribe_label'].values
        x_coords_stage_3 = df_stage_3['x_proj'].values
        y_coords_stage_3 = df_stage_3['y_proj'].values
        z_coords_stage_3 = df_stage_3['z_height'].values

        X_stage_3 = np.column_stack([x_coords_stage_3, y_coords_stage_3, z_coords_stage_3])
        nmi_stage_3 = calculate_nmi_score(X_stage_3, y_stage_3)
        retained_percentage = len(df_stage_3) / len(df_clean) * 100

    # 6. Reporting
    comparability_label = f"[{verify_status}]"
    
    print(f"\n--- Ablation Analysis Results (NMI Scores) {comparability_label} ---")
    print(f"Stage 1 (Baseline - Raw Euclidean): {nmi_stage_1:.3f}")
    print(f"Stage 2 (Stress Test - Weighted Coords): {nmi_stage_2:.3f}")
    print(f"Stage 3 (Reality Check - Density Filtered): {nmi_stage_3:.3f}")
    print(f"Delta (Stage 3 vs Stage 1): {nmi_stage_3 - nmi_stage_1:.3f}")
    print(f"Data Retained in Stage 3: {retained_percentage:.1f}%")

    return {
        "stage_1_nmi": nmi_stage_1,
        "stage_2_nmi": nmi_stage_2,
        "stage_3_nmi": nmi_stage_3,
        "delta_nmi": nmi_stage_3 - nmi_stage_1,
        "retained_percentage": retained_percentage,
        "verification_status": verify_status,
        "is_verified": is_verified,
        "comparability_label": comparability_label
    }

if __name__ == "__main__":
    # Standard entry point
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to MONOLITH_DATA.csv")
    args = parser.parse_args()
    
    if args.data_path.exists():
        run_ablation_analysis(args.data_path)
    else:
        print(f"Error: Path not found: {args.data_path}")
