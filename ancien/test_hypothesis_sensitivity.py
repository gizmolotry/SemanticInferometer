"""
Test if hypothesis affects the embedding
"""

import torch
import sys
sys.path.insert(0, '.')

from core.nli_extraction import MultiFramingNLIExtractor

print("="*70)
print("HYPOTHESIS SENSITIVITY TEST")
print("="*70)

# Initialize
extractor = MultiFramingNLIExtractor()

# Test article
article_text = "Israel launched airstrikes on Gaza. Hamas fired rockets at Israel. Many civilians died."

# Three VERY different hypotheses
hypotheses = [
    "This is about ice cream",
    "This is about war and conflict",
    "This is about birthday parties"
]

print(f"\nArticle: {article_text}")
print(f"\nTesting with 3 very different hypotheses:\n")

embeddings = []
for i, hyp in enumerate(hypotheses):
    print(f"[{i}] Hypothesis: {hyp}")
    emb = extractor.extract_single_framing(article_text, hyp)
    embeddings.append(emb)
    print(f"    Embedding norm: {emb.norm():.4f}")
    print(f"    Embedding mean: {emb.mean():.6f}")

# Compare embeddings
print("\n" + "="*70)
print("PAIRWISE SIMILARITIES")
print("="*70)

for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        emb_i_norm = embeddings[i] / embeddings[i].norm()
        emb_j_norm = embeddings[j] / embeddings[j].norm()
        sim = (emb_i_norm @ emb_j_norm).item()
        print(f"Hyp {i} vs Hyp {j}: {sim:.6f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

max_sim = max(
    (embeddings[i] / embeddings[i].norm() @ embeddings[j] / embeddings[j].norm()).item()
    for i in range(len(embeddings))
    for j in range(i+1, len(embeddings))
)

if max_sim > 0.999:
    print("✗ BROKEN: All hypotheses produce identical embeddings")
    print("  The model is NOT processing the hypothesis")
    print("\nPossible causes:")
    print("  1. Tokenization is wrong")
    print("  2. Model needs to be fine-tuned for NLI")
    print("  3. Using wrong model (need -mnli version)")
elif max_sim > 0.95:
    print("⚠ WEAK: Hypotheses produce very similar embeddings")
    print("  The model barely uses the hypothesis")
else:
    print("✓ WORKING: Different hypotheses produce different embeddings")

print(f"\nMax similarity: {max_sim:.6f}")
print("Target: <0.95 for good hypothesis sensitivity")

print("\n" + "="*70)