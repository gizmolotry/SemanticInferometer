"""
Quick test: Check if UMAP is producing valid output
"""

import torch
import numpy as np
from umap import UMAP

# Load one observer file
filepath = "enhanced_observer_42.pt"
data = torch.load(filepath, map_location='cpu')

# Get features
features = data.get('embeddings') or data.get('features')
print(f"Loaded features: {features.shape}")
print(f"Data type: {features.dtype}")

# Convert to numpy
X = features.numpy()[:2000]  # Sample 2000
print(f"\nSample: {X.shape}")

# Check for issues
print(f"\nData health check:")
print(f"  Has NaN: {np.isnan(X).any()}")
print(f"  Has Inf: {np.isinf(X).any()}")
print(f"  Mean: {X.mean():.4f}")
print(f"  Std: {X.std():.4f}")
print(f"  Min: {X.min():.4f}")
print(f"  Max: {X.max():.4f}")

# Compute UMAP
print(f"\nComputing UMAP...")
reducer = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    random_state=42
)

X_2d = reducer.fit_transform(X)

print(f"\nUMAP output: {X_2d.shape}")
print(f"  Has NaN: {np.isnan(X_2d).any()}")
print(f"  Has Inf: {np.isinf(X_2d).any()}")
print(f"  X range: [{X_2d[:, 0].min():.2f}, {X_2d[:, 0].max():.2f}]")
print(f"  Y range: [{X_2d[:, 1].min():.2f}, {X_2d[:, 1].max():.2f}]")

if np.allclose(X_2d, X_2d[0]):
    print(f"\n✗ PROBLEM: All points are the same!")
elif X_2d[:, 0].std() < 0.01 and X_2d[:, 1].std() < 0.01:
    print(f"\n✗ PROBLEM: Points have very low variance!")
else:
    print(f"\n✓ UMAP projection looks good!")
    print(f"  X std: {X_2d[:, 0].std():.2f}")
    print(f"  Y std: {X_2d[:, 1].std():.2f}")

# Sample some points
print(f"\nFirst 5 points:")
for i in range(5):
    print(f"  Point {i}: ({X_2d[i, 0]:.2f}, {X_2d[i, 1]:.2f})")
