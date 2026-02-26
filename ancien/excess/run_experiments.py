"""
Master Experimental Runner

Coordinates the complete Belief Transformer experimental pipeline:
1. Real corpus → multi-observer
2. Control corpus → multi-observer  
3. Temporal slices → multi-observer
4. Procrustes comparison across all conditions
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import torch

# ---------------------------------------------------------------------
# Project root & core import fix
# ---------------------------------------------------------------------
# ROOT = ...\belief-transformer\V3
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

from core.complete_pipeline import run_multi_observer_experiment
from core.provenance import build_metadata_indices


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_article(raw, idx: int | None = None):
    """
    Take a raw JSONL row and make sure it has the fields the pipeline expects:
    - source      (string)
    - timestamp   (float)
    - author      (string, optional)
    Everything else (content, title, url, control_class, etc.) is preserved.
    """
    a = dict(raw)  # shallow copy so we don't mutate the original

    # ----- SOURCE -----
    if "source" not in a:
        a["source"] = (
            raw.get("publisher")
            or raw.get("source_name")
            or raw.get("source")
            or "unknown"
        )

    # ----- TIMESTAMP -----
    ts = raw.get("timestamp") or raw.get("published_at") or raw.get("fetched_at")
    t_val = None

    if isinstance(ts, (int, float)):
        t_val = float(ts)
    elif isinstance(ts, str) and ts:
        # Handle ISO strings like "2025-11-05T17:18:06.111010" or "...Z"
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            t_val = dt.timestamp()
        except Exception:
            t_val = None

    if t_val is None:
        # Fallback: use row index as a monotonic pseudo-timestamp
        t_val = float(idx if idx is not None else 0.0)

    a["timestamp"] = t_val

    # ----- AUTHOR (optional, but provenance can use it) -----
    if "author" not in a:
        a["author"] = raw.get("author") or raw.get("byline") or "unknown_author"

    return a


def load_articles(relative_path: str):
    """
    Load articles from a JSONL file, ALWAYS resolving relative to project ROOT
    (...\\V3), and normalize metadata fields.
    """
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Could not find article file at: {path}")

    articles = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            article = normalize_article(raw, idx=idx)
            articles.append(article)

    print(f"Loaded {len(articles)} articles from {path}")
    return articles


# ---------------------------------------------------------------------
# 1. Real corpus
# ---------------------------------------------------------------------
def run_real_corpus_experiment(seeds=[42, 43, 44, 45, 46]):
    """Run experiment on real Gaza/Israel corpus."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: REAL CORPUS")
    print("=" * 70)

    # Load real articles (ROOT-aware)
    articles = load_articles("data/scraped_articles.jsonl")

    # Run multi-observer
    results = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=True,
        use_framing_rope=True,
        device="cuda",
        nli_model_name="microsoft/deberta-v2-xlarge-mnli",
        use_rks=True,
        rks_dim=512,
        rks_sigma=1.0
    )

    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tag + save results
    for seed, result in results.items():
        result["corpus_type"] = "real"
        result["corpus_name"] = "gaza_israel"
        out_path = OUTPUT_DIR / f"real_observer_{seed}.pt"
        torch.save(result, out_path)

    print("\n✓ Real corpus experiment complete")
    return results


# ---------------------------------------------------------------------
# 2. Control corpus
# ---------------------------------------------------------------------
def run_control_corpus_experiment(seeds=[42, 43, 44, 45, 46]):
    """Run experiment on control corpus."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CONTROL CORPUS")
    print("=" * 70)

    control_path = DATA_DIR / "control_corpus.jsonl"
    if not control_path.exists():
        print(f"✗ Control corpus not found at {control_path}")
        print("  Run make_control_corpus.py first.")
        return None

    articles = load_articles("data/control_corpus.jsonl")

    # Count by class
    from collections import Counter

    classes = Counter(a.get("control_class", "unknown") for a in articles)
    print("\nControl classes:")
    for cls, count in classes.items():
        print(f"  {cls:12s}: {count:4d}")

    # Run multi-observer
    results = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=True,
        use_framing_rope=True,
        device="cuda",
        nli_model_name="microsoft/deberta-v2-xlarge-mnli",
        use_rks=True,
        rks_dim=512,
        rks_sigma=1.0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tag + save results
    for seed, result in results.items():
        result["corpus_type"] = "control"
        result["corpus_name"] = "three_class"
        out_path = OUTPUT_DIR / f"control_observer_{seed}.pt"
        torch.save(result, out_path)

    print("\n✓ Control corpus experiment complete")
    return results


# ---------------------------------------------------------------------
# 3. Temporal slices
# ---------------------------------------------------------------------
def run_temporal_experiment(slice_name, seeds=[42, 43, 44, 45, 46]):
    """Run experiment on a specific temporal slice."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: TEMPORAL SLICE - {slice_name}")
    print("=" * 70)

    temporal_path = DATA_DIR / "temporal" / f"{slice_name}.jsonl"
    if not temporal_path.exists():
        print(f"✗ Temporal slice not found: {temporal_path}")
        print("  Run scrape_temporal.py first.")
        return None

    # Use relative path for loader, but it resolves against ROOT
    articles = load_articles(str(temporal_path.relative_to(ROOT)))
    print(f"Loaded {len(articles)} articles from {slice_name}")

    results = run_multi_observer_experiment(
        articles=articles,
        seeds=seeds,
        use_gru=True,
        use_framing_rope=True,
        device="cuda",
        nli_model_name="microsoft/deberta-v2-xlarge-mnli",
        use_rks=True,
        rks_dim=512,
        rks_sigma=1.0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed, result in results.items():
        result["corpus_type"] = "temporal"
        result["corpus_name"] = slice_name
        out_path = OUTPUT_DIR / f"temporal_{slice_name}_observer_{seed}.pt"
        torch.save(result, out_path)

    print(f"\n✓ Temporal experiment complete: {slice_name}")
    return results


def run_all_temporal_experiments(seeds=[42, 43, 44, 45, 46]):
    """Run experiments on all temporal slices."""
    from scrape_temporal import define_temporal_slices

    slices = define_temporal_slices()
    results = {}
    for slice_config in slices:
        slice_results = run_temporal_experiment(slice_config["name"], seeds=seeds)
        if slice_results:
            results[slice_config["name"]] = slice_results
    return results


# ---------------------------------------------------------------------
# 4. Cross-experiment comparison
# ---------------------------------------------------------------------
def compare_all_experiments():
    """Run Procrustes comparison across all experiments."""
    print("\n" + "=" * 70)
    print("CROSS-EXPERIMENT COMPARISON")
    print("=" * 70)

    from compare_observers import (
        load_all_observers,
        pairwise_procrustes_comparison,
        identify_high_divergence_articles,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use ROOT/outputs instead of CWD
    all_files = list(OUTPUT_DIR.glob("*observer_*.pt"))

    print(f"\nFound {len(all_files)} observer files:")

    by_corpus = {}
    for fpath in all_files:
        obs = torch.load(fpath)
        corpus_type = obs.get("corpus_type", "unknown")
        corpus_name = obs.get("corpus_name", "unknown")
        key = f"{corpus_type}_{corpus_name}"

        by_corpus.setdefault(key, []).append(fpath)

    for key, files in by_corpus.items():
        print(f"  {key:30s}: {len(files)} files")

    import pandas as pd

    all_comparisons = []

    for corpus_key, files in by_corpus.items():
        print(f"\n{'=' * 70}")
        print(f"Analyzing: {corpus_key}")
        print(f"{'=' * 70}")

        observers = {}
        for fpath in files:
            seed = int(fpath.stem.split("_")[-1])
            observers[seed] = torch.load(fpath)

        if len(observers) < 2:
            print("  ⚠ Need at least 2 observers for comparison")
            continue

        df = pairwise_procrustes_comparison(observers, mode="umap")
        df["corpus_type"] = corpus_key
        all_comparisons.append(df)

    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        out_csv = OUTPUT_DIR / "all_procrustes_comparisons.csv"
        combined.to_csv(out_csv, index=False)
        print(f"\n→ Saved combined comparisons to {out_csv}")

        print("\n" + "=" * 70)
        print("RESIDUAL SUMMARY")
        print("=" * 70)

        summary = combined.groupby("corpus_type")["residual_mean"].agg(
            ["mean", "std", "min", "max"]
        )
        print("\n" + summary.to_string())


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Master experimental runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "real", "control", "temporal", "compare"],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Observer seeds",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MASTER EXPERIMENTAL RUNNER")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Seeds: {args.seeds}")

    if args.mode in ("real", "all"):
        run_real_corpus_experiment(seeds=args.seeds)

    if args.mode in ("control", "all"):
        run_control_corpus_experiment(seeds=args.seeds)

    if args.mode in ("temporal", "all"):
        run_all_temporal_experiments(seeds=args.seeds)

    if args.mode in ("compare", "all"):
        compare_all_experiments()

    print("\n" + "=" * 70)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()