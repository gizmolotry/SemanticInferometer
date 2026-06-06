import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
import json

def run_eigen_audit(experiment_dir):
    exp_path = Path(experiment_dir)
    features_path = exp_path / "features.npy"
    metadata_path = exp_path / "article_metadata.csv"
    
    if not features_path.exists() or not metadata_path.exists():
        print(f"Error: Missing files in {experiment_dir}")
        return

    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)
    
    print(f"--- Eigen Audit for {experiment_dir} ---")
    print(f"Features shape: {features.shape}")
    
    # Task 1: PCA Component Decomposition
    n_components = min(10, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    print("\n[Task 1: PCA Decomposition]")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Singular Values: {pca.singular_values_}")
    
    # Extract top 5 features (dimensions) for PC1 and PC3
    def get_top_dims(component, top_n=5):
        abs_comp = np.abs(component)
        top_indices = np.argsort(abs_comp)[-top_n:][::-1]
        return [(idx, component[idx]) for idx in top_indices]

    pc1_top = get_top_dims(pca.components_[0])
    print("\nTop 5 Dimensions contributing to PC1:")
    for idx, val in pc1_top:
        print(f"  Dim {idx}: {val:.4f}")

    if n_components >= 3:
        pc3_top = get_top_dims(pca.components_[2])
        print("\nTop 5 Dimensions contributing to PC3:")
        for idx, val in pc3_top:
            print(f"  Dim {idx}: {val:.4f}")

    # Task 2: Distribution Check
    print("\n[Task 2: Distribution Check]")
    mean_val = np.mean(features)
    var_val = np.var(features)
    std_val = np.std(features)
    print(f"Global Mean: {mean_val:.6f}")
    print(f"Global Variance: {var_val:.6f}")
    print(f"Global Std Dev: {std_val:.6f}")
    
    # Check for clipping/normalization
    max_val = np.max(features)
    min_val = np.min(features)
    print(f"Range: [{min_val:.4f}, {max_val:.4f}]")
    
    # Check \"Sphericalness\" (Ratio of variances of top components)
    if len(pca.explained_variance_ratio_) > 1:
        sphericalness = pca.explained_variance_ratio_[1] / pca.explained_variance_ratio_[0]
        print(f"Sphericalness (EVR2/EVR1): {sphericalness:.4f}")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_eigen_audit(sys.argv[1])
    else:
        run_eigen_audit("outputs/honest_matern")
