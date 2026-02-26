#!/usr/bin/env python3
"""
Example usage patterns for the Belief Transformer Ingestion Pipeline.
"""

import asyncio
from pathlib import Path
from ingest import run_ingestion
import json


async def example_basic():
    """Basic usage - fetch last 24 hours."""
    print("=" * 60)
    print("Example 1: Basic Fetch")
    print("=" * 60)
    
    articles = await run_ingestion(
        lookback_hours=24,
        max_articles=100  # Limit for demo
    )
    
    print(f"\n✓ Fetched {len(articles)} articles")
    
    if articles:
        # Show sample article
        sample = articles[0]
        print(f"\nSample article:")
        print(f"  Title: {sample['title']}")
        print(f"  Publisher: {sample['publisher']}")
        print(f"  Words: {sample['word_count']}")
        print(f"  Method: {sample['extraction_method']}")


async def example_filtered():
    """Fetch with custom filtering."""
    print("\n" + "=" * 60)
    print("Example 2: Filtered Fetch (Reuters only)")
    print("=" * 60)
    
    # Fetch all articles
    articles = await run_ingestion(
        lookback_hours=48,
        max_articles=500
    )
    
    # Filter for Reuters
    reuters_articles = [
        a for a in articles 
        if 'reuters' in a.get('publisher', '').lower()
    ]
    
    print(f"\n✓ Found {len(reuters_articles)} Reuters articles")
    
    # Analyze by section
    sections = {}
    for article in reuters_articles:
        section = article.get('section', 'unknown')
        sections[section] = sections.get(section, 0) + 1
    
    print("\nArticles by section:")
    for section, count in sorted(sections.items(), key=lambda x: x[1], reverse=True):
        print(f"  {section}: {count}")


async def example_custom_processing():
    """Fetch and do custom processing."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Processing")
    print("=" * 60)
    
    articles = await run_ingestion(
        lookback_hours=24,
        max_articles=200
    )
    
    # Find longest articles
    longest = sorted(articles, key=lambda a: a['word_count'], reverse=True)[:5]
    
    print(f"\nTop 5 longest articles:")
    for i, article in enumerate(longest, 1):
        print(f"\n{i}. {article['title'][:60]}...")
        print(f"   {article['word_count']} words | {article['publisher']}")
    
    # Find articles with specific keywords
    keywords = ['climate', 'election', 'economy']
    
    print(f"\nArticles mentioning: {', '.join(keywords)}")
    for keyword in keywords:
        matching = [
            a for a in articles
            if keyword.lower() in a['content'].lower()
        ]
        print(f"  {keyword}: {len(matching)} articles")


async def example_export_formats():
    """Export to different formats."""
    print("\n" + "=" * 60)
    print("Example 4: Export Formats")
    print("=" * 60)
    
    articles = await run_ingestion(
        lookback_hours=12,
        max_articles=50
    )
    
    # Export as single JSON file
    json_path = Path("data/processed/articles_export.json")
    with open(json_path, 'w') as f:
        json.dump(articles, f, indent=2)
    print(f"\n✓ Exported to JSON: {json_path}")
    
    # Export as CSV (simplified)
    import csv
    csv_path = Path("data/processed/articles_export.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'publisher', 'url', 'word_count', 'published_at'])
        for a in articles:
            writer.writerow([
                a['title'],
                a['publisher'],
                a['url'],
                a['word_count'],
                a.get('published_at', '')
            ])
    print(f"✓ Exported to CSV: {csv_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total articles: {len(articles)}")
    print(f"  Total words: {sum(a['word_count'] for a in articles):,}")
    print(f"  Avg words/article: {sum(a['word_count'] for a in articles) / len(articles):.0f}")
    
    publishers = {}
    for a in articles:
        pub = a['publisher']
        publishers[pub] = publishers.get(pub, 0) + 1
    
    print(f"\nTop publishers:")
    for pub, count in sorted(publishers.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pub}: {count}")


async def example_incremental():
    """Incremental fetching (avoid re-processing)."""
    print("\n" + "=" * 60)
    print("Example 5: Incremental Fetching")
    print("=" * 60)
    
    # First fetch (will use dedup cache)
    print("\nFirst fetch (creates dedup cache)...")
    articles1 = await run_ingestion(
        lookback_hours=6,
        max_articles=100
    )
    print(f"✓ Fetched {len(articles1)} articles")
    
    # Second fetch (should skip duplicates)
    print("\nSecond fetch (same time window - should skip duplicates)...")
    articles2 = await run_ingestion(
        lookback_hours=6,
        max_articles=100
    )
    print(f"✓ Fetched {len(articles2)} new articles")
    
    print(f"\nDeduplication working: {len(articles1)} first, {len(articles2)} second")


async def run_examples():
    """Run all examples."""
    print("\n🎯 Belief Transformer Ingestion - Usage Examples\n")
    
    try:
        await example_basic()
        await example_filtered()
        await example_custom_processing()
        await example_export_formats()
        await example_incremental()
        
        print("\n" + "=" * 60)
        print("✅ All examples complete!")
        print("=" * 60)
        print("\nCheck data/processed/ for output files")
        print("See README.md for more usage patterns")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_examples())
