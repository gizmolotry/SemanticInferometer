import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import sys

def run_features_only_audit(features_path):
    features_path = Path(features_path)
    if not features_path.exists():
        print(f"Error: Missing {features_path}")
        return

    features = np.load(features_path)
    
    print(f"--- Eigen Audit for {features_path} ---")
    print(f"Features shape: {features.shape}")
    
    # Task 1: PCA Component Decomposition
    n_components = min(10, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    print("\n[Task 1: PCA Decomposition]")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
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
    print(f"Global Mean: {np.mean(features):.6f}")
    print(f"Global Variance: {np.var(features):.6f}")
    print(f"Range: [{np.min(features):.4f}, {np.max(features):.4f}]")
    
    if len(pca.explained_variance_ratio_) > 1:
        sphericalness = pca.explained_variance_ratio_[1] / pca.explained_variance_ratio_[0]
        print(f"Sphericalness (EVR2/EVR1): {sphericalness:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_features_only_audit(sys.argv[1])
    else:
        run_features_only_audit("outputs/honest_matern/features.npy")
