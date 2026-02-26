"""
NLI Extraction Diagnostic - Find the Collapse
"""

import torch
import json
import sys
sys.path.insert(0, '.')

from core.nli_extraction import MultiFramingNLIExtractor

print("="*70)
print("NLI EXTRACTION DIAGNOSTIC")
print("="*70)

# Load 3 test articles
articles = []
with open('data/scraped_articles.jsonl', encoding='utf-8') as f:
    for i in range(3):
        articles.append(json.loads(f.readline()))

print(f"\nTest articles:")
for i, art in enumerate(articles):
    print(f"  [{i}] {len(art.get('content', ''))} chars")

# Initialize
print("\nInitializing extractor...")
extractor = MultiFramingNLIExtractor()

print(f"  Number of framings: {extractor.n_framings}")
print(f"  Embedding dim per framing: {extractor.embedding_dim}")
print(f"  Total concatenated dim: {extractor.n_framings * extractor.embedding_dim}")

# Test extraction
print("\n" + "="*70)
print("TESTING INDIVIDUAL FRAMING EXTRACTIONS")
print("="*70)

article_0_text = articles[0].get('content', '')

print(f"\nExtracting all {extractor.n_framings} framings for Article 0...")
print(f"Framing queries:")
for i, query in enumerate(extractor.framing_queries):
    print(f"  [{i}] {query[:60]}...")

individual_embeddings = []
for i, query in enumerate(extractor.framing_queries):
    emb = extractor.extract_single_framing(article_0_text, query)
    individual_embeddings.append(emb)
    print(f"\nFraming {i}:")
    print(f"  Shape: {emb.shape}")
    print(f"  Mean: {emb.mean():.6f}")
    print(f"  Std: {emb.std():.6f}")
    print(f"  Norm: {emb.norm():.2f}")

# Check similarity between framings
print("\n" + "="*70)
print("FRAMING SIMILARITY (within Article 0)")
print("="*70)

print("\nPairwise cosine similarity between framings:")
for i in range(len(individual_embeddings)):
    for j in range(i+1, len(individual_embeddings)):
        emb_i_norm = individual_embeddings[i] / individual_embeddings[i].norm()
        emb_j_norm = individual_embeddings[j] / individual_embeddings[j].norm()
        sim = (emb_i_norm @ emb_j_norm).item()
        print(f"  Framing {i} vs {j}: {sim:.4f}")

# Now test concatenated version
print("\n" + "="*70)
print("CONCATENATED MULTI-FRAMING EXTRACTION")
print("="*70)

all_features = []
for i, article in enumerate(articles):
    text = article.get('content', '')
    feat = extractor.extract_multi_framing(text)
    all_features.append(feat)
    print(f"\nArticle {i}:")
    print(f"  Shape: {feat.shape}")
    print(f"  Mean: {feat.mean():.6f}")
    print(f"  Std: {feat.std():.6f}")

# Compare across articles
print("\n" + "="*70)
print("CROSS-ARTICLE SIMILARITY")
print("="*70)

all_feats = torch.stack(all_features)  # [3, 8192]
print(f"\nStacked features: {all_feats.shape}")

# Normalize and compute similarity
all_feats_norm = all_feats / (all_feats.norm(dim=1, keepdim=True) + 1e-8)
sim_matrix = all_feats_norm @ all_feats_norm.T

print(f"\nSimilarity matrix:")
print(sim_matrix)

print(f"\nSimilarity statistics:")
print(f"  Mean: {sim_matrix.mean():.6f}")
print(f"  Std:  {sim_matrix.std():.6f}")

# Diagnosis
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

std = sim_matrix.std().item()

if std > 0.1:
    print("✓ DIVERSE: Articles produce different features")
elif std > 0.05:
    print("⚠ MODERATE: Some diversity, needs improvement")
else:
    print("✗ COLLAPSED: All articles look the same!")
    
print(f"\nSimilarity std: {std:.6f}")
print(f"Target: >0.1 for good diversity")

# Check if individual framings are diverse but concatenation collapses
print("\n" + "="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

print("\nHypothesis 1: Individual framings are too similar")
print("  Check the 'FRAMING SIMILARITY' section above")
print("  If all pairs show >0.95 similarity → framings extract same info")

print("\nHypothesis 2: Articles are genuinely similar in content")  
print("  If articles about same topic → high similarity is expected")
print("  Test with very different articles to verify")

print("\nHypothesis 3: Model normalization/pooling is too aggressive")
print("  The CLS token might lose fine-grained differences")
print("  Solution: Use mean pooling or multiple layers")

print("\n" + "="*70)