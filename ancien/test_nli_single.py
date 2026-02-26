"""
Debug NLI Extraction - Simplified Test
"""

import torch
import json
import sys
sys.path.insert(0, '.')

from core.nli_extraction import MultiFramingNLIExtractor

print("="*70)
print("NLI EXTRACTION DEBUG")
print("="*70)

# Load first 3 articles
articles = []
with open('data/scraped_articles.jsonl', encoding='utf-8') as f:
    for i in range(3):
        articles.append(json.loads(f.readline()))

print(f"\nTest articles:")
for i, art in enumerate(articles):
    content_len = len(art.get('content', ''))
    print(f"  [{i}] {art.get('publisher', '?')[:20]}: {content_len} chars")

# Initialize extractor
print("\nInitializing NLI extractor...")
extractor = MultiFramingNLIExtractor()

# Extract for each article
print("\n" + "="*70)
print("EXTRACTION RESULTS")
print("="*70)

all_features = []

for i, article in enumerate(articles):
    print(f"\nArticle {i}:")
    text = article.get('content', '')
    
    # Extract
    features = extractor.extract_framings(text)
    all_features.append(features)
    
    print(f"  Output shape: {features.shape}")
    print(f"  Output mean: {features.mean():.6f}")
    print(f"  Output std: {features.std():.6f}")
    print(f"  Output norm: {features.norm():.2f}")

# Stack all features
print("\n" + "="*70)
print("CROSS-ARTICLE ANALYSIS")
print("="*70)

all_feats = torch.stack(all_features)
print(f"\nStacked shape: {all_feats.shape}")

# Flatten if needed
if len(all_feats.shape) == 3:
    # [num_articles, num_framings, embedding_dim]
    flat_features = all_feats.reshape(-1, all_feats.shape[-1])
    print(f"Flattened to: {flat_features.shape}")
else:
    flat_features = all_feats

# Compute pairwise similarities
print("\nComputing pairwise similarities...")
flat_norm = flat_features / (flat_features.norm(dim=1, keepdim=True) + 1e-8)
sim_matrix = flat_norm @ flat_norm.T

print(f"\nSimilarity statistics:")
print(f"  Mean: {sim_matrix.mean():.6f}")
print(f"  Std:  {sim_matrix.std():.6f}")
print(f"  Min:  {sim_matrix.min():.6f}")
print(f"  Max:  {sim_matrix.max():.6f}")

# Show a sample
print(f"\nSample 5x5 similarity matrix:")
print(sim_matrix[:5, :5])

# Get off-diagonal similarities
mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
off_diag_sims = sim_matrix[mask]
print(f"\nOff-diagonal similarities:")
print(f"  Mean: {off_diag_sims.mean():.6f}")
print(f"  Std:  {off_diag_sims.std():.6f}")

# Diagnosis
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

std = sim_matrix.std().item()

if std > 0.1:
    print("✓ GOOD: Features are diverse!")
    print(f"  Similarity std = {std:.4f} > 0.1")
elif std > 0.05:
    print("⚠ MODERATE: Some diversity but could be better")
    print(f"  Similarity std = {std:.4f} (target: >0.1)")
else:
    print("✗ COLLAPSED: Features are too similar!")
    print(f"  Similarity std = {std:.4f} << 0.1")
    print("\nThis means:")
    print("  - All articles look nearly identical to the model")
    print("  - Attention will be forced uniform")
    print("  - Need to fix feature extraction")

print("\n" + "="*70)