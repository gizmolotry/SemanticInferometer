#!/usr/bin/env python3
"""
QUICK INTEGRATION - Your Scraper → Belief Transformer

Converts your JSONL format to pipeline format and RUNS IT.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add belief transformer to path
sys.path.insert(0, str(Path(__file__).parent / 'belief_transformer'))

from core.complete_pipeline import run_multi_observer_experiment
from umap import UMAP
import matplotlib.pyplot as plt
import torch


def load_scraped_articles(jsonl_path: str):
    """Load articles from your scraper's JSONL output."""
    articles = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    article = json.loads(line)
                    articles.append(article)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping line {line_num}, invalid JSON: {e}")
                    continue
        
        print(f"✓ Loaded {len(articles)} articles from {jsonl_path}")
        return articles
        
    except FileNotFoundError:
        print(f"✗ File not found: {jsonl_path}")
        print(f"  Current directory: {Path('.').absolute()}")
        raise
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        raise


def convert_to_pipeline_format(scraped_articles):
    """
    Convert your scraper format to Belief Transformer format.
    
    YOUR FORMAT → PIPELINE FORMAT:
    - content → text
    - publisher → source  
    - published_at → timestamp
    - (keeps: url, author, themes, locations, sentiment_score)
    """
    pipeline_articles = []
    
    for article in scraped_articles:
        # Convert published_at to timestamp
        pub_date = article.get('published_at')
        if pub_date:
            try:
                # Try parsing ISO format
                if 'T' in pub_date:
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                else:
                    # Just date string like "2025-10-28"
                    dt = datetime.strptime(pub_date, "%Y-%m-%d")
                timestamp = dt.timestamp()
            except:
                timestamp = datetime.now().timestamp()
        else:
            timestamp = datetime.now().timestamp()
        
        # Convert to pipeline format
        pipeline_article = {
            'text': article.get('content', ''),
            'source': article.get('publisher', 'Unknown'),
            'timestamp': timestamp,
            'url': article.get('url', ''),
            'author': article.get('author'),
            
            # Preserve GDELT metadata (optional but cool)
            'themes': article.get('themes', []),
            'locations': article.get('locations', []),
            'sentiment_score': article.get('sentiment_score'),
        }
        
        # Skip if no text
        if not pipeline_article['text']:
            continue
        
        pipeline_articles.append(pipeline_article)
    
    print(f"✓ Converted {len(pipeline_articles)} articles to pipeline format")
    return pipeline_articles


def run_quick_test(articles, num_articles=100):
    """Run a quick test with first N articles."""
    
    print(f"\n{'='*70}")
    print(f"QUICK TEST: First {num_articles} articles")
    print(f"{'='*70}\n")
    
    # Take subset
    test_articles = articles[:num_articles]
    
    print(f"Running 3 observers (seeds 42, 43, 44)...")
    print(f"This will take ~5-10 minutes on CPU, ~1-2 min on GPU\n")
    
    # Run multi-observer experiment
    results = run_multi_observer_experiment(
        articles=test_articles,
        seeds=[42, 43, 44],
        use_gru=False,  # Disable for quick test
        use_framing_rope=False  # Disable for quick test
    )
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    
    # Compare observers
    print("Observer attention patterns:")
    for seed in [42, 43, 44]:
        mean_attn = results[seed]['attention_matrix'].mean().item()
        print(f"  Observer {seed}: {mean_attn:.6f}")
    
    # Calculate differences
    print("\nPairwise differences:")
    for seed1, seed2 in [(42, 43), (42, 44), (43, 44)]:
        diff = (results[seed1]['attention_matrix'] - results[seed2]['attention_matrix']).abs().mean().item()
        print(f"  Observer {seed1} vs {seed2}: {diff:.6f}")
    
    if diff > 0.001:
        print("\n✓ OBSERVERS DIFFER - THESIS SUPPORTED!")
    else:
        print("\n⚠ Observers look similar - might need more articles")
    
    # Generate figure
    print("\nGenerating UMAP visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, seed in enumerate([42, 43, 44]):
        coords = UMAP(random_state=0).fit_transform(
            results[seed]['article_tokens'].cpu().numpy()
        )
        
        # Color by source
        sources = [m['source'] for m in results[seed]['metadata']]
        unique_sources = list(set(sources))
        colors = [unique_sources.index(s) for s in sources]
        
        scatter = axes[i].scatter(coords[:, 0], coords[:, 1], c=colors, 
                                 cmap='tab10', alpha=0.6, s=30)
        axes[i].set_title(f'Observer {seed}', fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/quick_test_three_observers.png', dpi=300)
    print("✓ Saved: outputs/quick_test_three_observers.png")
    
    print(f"\n{'='*70}")
    print("QUICK TEST COMPLETE")
    print(f"{'='*70}\n")
    
    return results


def run_full_experiment(articles):
    """Run full experiment with all features."""
    
    print(f"\n{'='*70}")
    print(f"FULL EXPERIMENT: All {len(articles)} articles")
    print(f"{'='*70}\n")
    
    print(f"Running 5 observers with GRU + RoPE...")
    print(f"This will take ~15-30 minutes\n")
    
    # Run with all features
    results = run_multi_observer_experiment(
        articles=articles,
        seeds=[42, 43, 44, 45, 46],
        use_gru=True,
        use_framing_rope=True
    )
    
    # Generate full figure
    print("\nGenerating full UMAP visualization...")
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i, seed in enumerate([42, 43, 44, 45, 46]):
        coords = UMAP(random_state=0).fit_transform(
            results[seed]['article_tokens'].cpu().numpy()
        )
        
        sources = [m['source'] for m in results[seed]['metadata']]
        unique_sources = list(set(sources))
        colors = [unique_sources.index(s) for s in sources]
        
        axes[i].scatter(coords[:, 0], coords[:, 1], c=colors, 
                       cmap='tab20', alpha=0.6, s=20)
        axes[i].set_title(f'Observer {seed}', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/full_five_observers.png', dpi=300)
    print("✓ Saved: outputs/full_five_observers.png")
    
    print(f"\n{'='*70}")
    print("FULL EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")
    
    return results


def main():
    print("\n" + "="*70)
    print("YOUR SCRAPER → BELIEF TRANSFORMER")
    print("="*70)
    
    # 1. Load your scraped data
    print("\n[1/3] Loading scraped articles...")
    scraped = load_scraped_articles('historical_202510-202511_20251105_114558.jsonl')
    
    # Show sample
    if scraped:
        sample = scraped[0]
        print(f"\nSample article:")
        print(f"  Publisher: {sample.get('publisher')}")
        print(f"  Title: {sample.get('title', '')[:60]}...")
        print(f"  Themes: {sample.get('themes', [])[:3]}")
        print(f"  Sentiment: {sample.get('sentiment_score')}")
    
    # 2. Convert format
    print(f"\n[2/3] Converting to pipeline format...")
    articles = convert_to_pipeline_format(scraped)
    
    # 3. Choose what to run
    print(f"\n[3/3] Choose experiment:")
    print(f"  1. Quick test (100 articles, ~5 min)")
    print(f"  2. Full experiment (all {len(articles)} articles, ~20 min)")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        results = run_quick_test(articles, num_articles=100)
    elif choice == "2":
        results = run_full_experiment(articles)
    else:
        print("Invalid choice")
        return
    
    print("\n✓ DONE!")
    print("\nNext steps:")
    print("  1. Look at the UMAP figure")
    print("  2. If observers differ → you have your result")
    print("  3. Scale up to more months of data")
    print("  4. Write paper")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()