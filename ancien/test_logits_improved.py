"""
Test Logits + Improved Queries

Tests if using NLI logits + asymmetric queries solves the embedding collapse.
"""

import torch

# Import the new logits-based extractor
from core.nli_extraction_logits import MultiFramingNLIExtractorLogits

print("="*70)
print("LOGITS + IMPROVED QUERIES TEST")
print("="*70)

# Test with improved queries
print("\nInitializing with improved queries...")
extractor = MultiFramingNLIExtractorLogits(
    queries_config='config/framing_queries_improved.yaml'
)

# Biased test articles
articles = [
    {
        'title': 'Pro-Israel Article',
        'content': """
        Hamas terrorists launched unprovoked rocket attacks targeting innocent Israeli civilians.
        Israel has every right to defend itself against these barbaric attacks.
        The IDF goes to extraordinary lengths to avoid harming civilians, warning them before strikes.
        Palestinian casualties are tragic but unavoidable given Hamas's use of human shields.
        """
    },
    {
        'title': 'Pro-Palestinian Article',
        'content': """
        Israeli occupation forces conducted brutal attacks on defenseless Palestinian civilians.
        The resistance fighters are engaged in legitimate armed struggle against colonial occupation.
        Israel deliberately targets hospitals, schools, and residential areas in war crimes.
        The root cause of all violence is Israel's illegal 50-year military occupation.
        """
    },
    {
        'title': 'Neutral Article',
        'content': """
        Fighting between Israeli forces and Palestinian groups continued today.
        Both sides reported casualties. The UN called for de-escalation.
        International observers expressed concern about civilian deaths.
        """
    }
]

print(f"\nTest articles:")
for i, art in enumerate(articles):
    print(f"  [{i}] {art['title']}")

# Extract
print("\nExtracting features...")
features = extractor.extract_batch(articles, show_progress=False)

print(f"\nFeature shape: {features.shape}")
print(f"  Articles: {features.shape[0]}")
print(f"  Dimensions: {features.shape[1]} ({extractor.n_framings} framings × 3 logits)")

# Analyze individual framings
print("\n" + "="*70)
print("FRAMING ANALYSIS")
print("="*70)

framing_names = [
    "Justify Israeli military",
    "Condemn Israeli military",
    "Justify Palestinian resistance",
    "Condemn Palestinian violence"
]

for i, article in enumerate(articles):
    print(f"\n### {article['title']} ###")
    for j, name in enumerate(framing_names[:4]):  # First 4 framings
        logits = features[i, j*3:(j+1)*3]
        print(f"\n{name}:")
        print(f"  Contradict: {logits[0]:.3f}")
        print(f"  Neutral:    {logits[1]:.3f}")
        print(f"  Entail:     {logits[2]:.3f}")

# Cross-article similarity
print("\n" + "="*70)
print("CROSS-ARTICLE SIMILARITY")
print("="*70)

feat_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
sim_matrix = feat_norm @ feat_norm.T

print(f"\nSimilarity matrix:")
print(sim_matrix)

print(f"\nPairwise similarities:")
print(f"  Pro-Israel vs Pro-Palestinian: {sim_matrix[0, 1]:.4f}")
print(f"  Pro-Israel vs Neutral: {sim_matrix[0, 2]:.4f}")
print(f"  Pro-Palestinian vs Neutral: {sim_matrix[1, 2]:.4f}")

# Diagnosis
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

sim_opposite = sim_matrix[0, 1].item()

if sim_opposite < 0.5:
    print(f"✓✓✓ EXCELLENT: Opposite articles very different ({sim_opposite:.3f})")
    print("    The logits + improved queries approach WORKS!")
elif sim_opposite < 0.7:
    print(f"✓✓ GOOD: Opposite articles distinguishable ({sim_opposite:.3f})")
    print("    Significant improvement over CLS embeddings")
elif sim_opposite < 0.85:
    print(f"✓ MODERATE: Some distinction ({sim_opposite:.3f})")
    print("    Better than before, but still room for improvement")
else:
    print(f"✗ STILL COLLAPSED: Too similar ({sim_opposite:.3f})")
    print("    Need further fixes")

print(f"\nSimilarity std across all pairs: {sim_matrix.std():.4f}")
print(f"Target: >0.15 for good diversity")

print("\n" + "="*70)
