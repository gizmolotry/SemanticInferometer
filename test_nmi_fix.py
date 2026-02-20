#!/usr/bin/env python3
"""Quick test to verify NMI fix - targets 0.7+ NMI."""
import sys
import json
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

def test_nmi():
    print("=" * 60)
    print("TESTING NMI FIX")
    print("=" * 60)

    # Load synthetic corpus
    corpus_path = Path("sythgen/high_quality_articles.jsonl")
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        return

    articles = []
    ground_truth = {}
    with open(corpus_path) as f:
        for line in f:
            art = json.loads(line)
            articles.append(art)
            # Ground truth is in 'perspective_tag' field
            if 'perspective_tag' in art:
                ground_truth[len(articles)-1] = art['perspective_tag']

    print(f"Loaded {len(articles)} articles")
    print(f"Ground truth labels: {len(ground_truth)}")

    # Use subset for speed (10 articles for quick testing)
    n_test = min(10, len(articles))
    articles = articles[:n_test]

    # Import pipeline
    from core.complete_pipeline import initialize_full_pipeline, BeliefTransformerPipeline

    print("\nInitializing pipeline with NMI=0.7 settings...")
    components = initialize_full_pipeline(
        random_seed=43,
        device="cuda",
        kernel_type="rbf",
        use_cls_tokens=True,
        use_dirichlet_fusion=True,
        dirichlet_rks_dim=512,
        dirichlet_n_observers=10,
        dirichlet_alpha=1.0,           # Mode B settings
        dirichlet_hidden_dim=1536,
        mix_in_rkhs=True,              # CRITICAL for NMI
        geometry_mode="rks",
        normalize_features=True,
    )

    pipeline = BeliefTransformerPipeline(
        components=components,
        random_seed=43,
    )

    print("\nRunning pipeline...")
    result = pipeline.process_month(articles, month_name="test")

    features = result['features']
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features std: {features.std():.6f}")
    print(f"Features norm mean: {np.linalg.norm(features, axis=1).mean():.4f}")

    # Check if features are collapsed
    if features.std() < 0.01:
        print("\n[ERROR] Features collapsed (std < 0.01) - kernel collapse!")
        return

    # Cluster and compute NMI
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    # Get ground truth for test articles
    true_labels = []
    valid_indices = []
    for i in range(n_test):
        if i in ground_truth:
            true_labels.append(ground_truth[i])
            valid_indices.append(i)

    if len(valid_indices) < 5:
        print(f"\n[WARNING] Only {len(valid_indices)} articles with ground truth")
        return

    # Map labels to integers
    label_set = list(set(true_labels))
    label_map = {l: i for i, l in enumerate(label_set)}
    y_true = np.array([label_map[l] for l in true_labels])

    # Cluster
    n_clusters = len(label_set)
    kmeans = KMeans(n_clusters=n_clusters, random_state=43, n_init=10)
    X_valid = features[valid_indices]
    y_pred = kmeans.fit_predict(X_valid)

    # Compute metrics
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"RESULTS: NMI = {nmi:.3f}, ARI = {ari:.3f}")
    print("=" * 60)

    if nmi >= 0.5:
        print("SUCCESS! NMI >= 0.5")
    elif nmi >= 0.3:
        print("PARTIAL - NMI >= 0.3 but below target")
    else:
        print("FAILED - NMI < 0.3, kernel likely collapsed")

    return nmi, ari

if __name__ == "__main__":
    test_nmi()
