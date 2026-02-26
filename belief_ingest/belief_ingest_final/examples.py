#!/usr/bin/env python3
"""
Example usage script for GDELT Historical Mode.
Demonstrates common usage patterns.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path


async def example_1_quick_test():
    """Example 1: Quick test with 1 week of data."""
    print("\n" + "=" * 60)
    print("Example 1: Quick Test (1 Week)")
    print("=" * 60)
    
    from ingest import run_ingestion
    
    # Fetch last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nFetching articles from {start_date.date()} to {end_date.date()}...")
    print("This should take 5-10 minutes...\n")
    
    articles = await run_ingestion(
        historical=True,
        start_date=start_date,
        end_date=end_date,
        max_articles=500  # Limit for quick test
    )
    
    print(f"\n✓ Fetched {len(articles)} articles")
    
    if articles:
        # Show sample
        sample = articles[0]
        print(f"\nSample article:")
        print(f"  Title: {sample['title'][:60]}...")
        print(f"  Publisher: {sample['publisher']}")
        print(f"  Themes: {sample.get('themes', [])[:3]}")
        print(f"  Locations: {sample.get('locations', [])[:3]}")
        print(f"  Sentiment: {sample.get('sentiment_score')}")
        print(f"  Snippets: {len(sample['snippets'])} segments")


async def example_2_one_month():
    """Example 2: Fetch 1 month of data."""
    print("\n" + "=" * 60)
    print("Example 2: One Month Fetch")
    print("=" * 60)
    
    from ingest import run_ingestion
    
    print("\nFetching last 1 month of articles...")
    print("This should take 15-30 minutes...\n")
    
    articles = await run_ingestion(
        historical=True,
        months=1,
        max_articles=5000
    )
    
    print(f"\n✓ Fetched {len(articles)} articles")
    
    # Analyze by publisher
    publishers = {}
    for article in articles:
        pub = article['publisher']
        publishers[pub] = publishers.get(pub, 0) + 1
    
    print(f"\nTop 10 publishers:")
    for pub, count in sorted(publishers.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pub}: {count} articles")


async def example_3_acquisition_study():
    """Example 3: Acquisition study (pre + post windows)."""
    print("\n" + "=" * 60)
    print("Example 3: Acquisition Study")
    print("=" * 60)
    
    from ingest import run_ingestion
    from config import ACQUISITION_TARGETS
    
    # Use Irish Independent as example
    target = ACQUISITION_TARGETS['irish_independent']
    
    print(f"\nAcquisition: {target['name']}")
    print(f"Date: {target['acquisition_date']}")
    
    # Pre-acquisition window
    print(f"\n1. Fetching PRE-acquisition data...")
    print(f"   {target['pre_start']} to {target['pre_end']}")
    
    pre_start = datetime.strptime(target['pre_start'], "%Y-%m-%d")
    pre_end = datetime.strptime(target['pre_end'], "%Y-%m-%d")
    
    pre_articles = await run_ingestion(
        historical=True,
        start_date=pre_start,
        end_date=pre_end,
        max_articles=10000,  # Limit for example
        output_dir=Path("data/processed/irish_pre")
    )
    
    print(f"   ✓ Fetched {len(pre_articles)} pre-acquisition articles")
    
    # Post-acquisition window
    print(f"\n2. Fetching POST-acquisition data...")
    print(f"   {target['post_start']} to {target['post_end']}")
    
    post_start = datetime.strptime(target['post_start'], "%Y-%m-%d")
    post_end = datetime.strptime(target['post_end'], "%Y-%m-%d")
    
    post_articles = await run_ingestion(
        historical=True,
        start_date=post_start,
        end_date=post_end,
        max_articles=10000,  # Limit for example
        output_dir=Path("data/processed/irish_post")
    )
    
    print(f"   ✓ Fetched {len(post_articles)} post-acquisition articles")
    
    print(f"\n✓ Acquisition study complete!")
    print(f"\nNext steps:")
    print(f"  1. Copy to V2: cp data/processed/irish_* ../../V2/data/raw/")
    print(f"  2. Run Belief Transformer to compute θ fingerprints")
    print(f"  3. Analyze θ distribution shift: pre vs post")


async def example_4_lenient_mode():
    """Example 4: Lenient filtering for higher recall."""
    print("\n" + "=" * 60)
    print("Example 4: Lenient Filtering Mode")
    print("=" * 60)
    
    from ingest import run_ingestion
    
    print("\nFetching with LENIENT filtering (more articles)...")
    print("This uses: theme OR location OR URL keyword\n")
    
    articles = await run_ingestion(
        historical=True,
        months=1,
        max_articles=5000,
        lenient=True  # Lenient mode
    )
    
    print(f"\n✓ Fetched {len(articles)} articles (lenient mode)")
    print(f"\nNote: Lenient mode typically yields 2x more articles than strict mode")


async def example_5_metadata_analysis():
    """Example 5: Analyze GDELT metadata."""
    print("\n" + "=" * 60)
    print("Example 5: GDELT Metadata Analysis")
    print("=" * 60)
    
    from ingest import run_ingestion
    
    print("\nFetching articles with metadata...")
    
    articles = await run_ingestion(
        historical=True,
        months=0.25,  # 1 week
        max_articles=500
    )
    
    print(f"\n✓ Fetched {len(articles)} articles")
    
    # Analyze themes
    theme_counts = {}
    for article in articles:
        for theme in article.get('themes', []):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    print(f"\nTop 10 GDELT themes:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {theme}: {count} articles")
    
    # Analyze locations
    location_counts = {}
    for article in articles:
        for location in article.get('locations', []):
            location_counts[location] = location_counts.get(location, 0) + 1
    
    print(f"\nTop 10 locations:")
    for loc, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {loc}: {count} articles")
    
    # Analyze sentiment
    sentiments = [a.get('sentiment_score') for a in articles if a.get('sentiment_score')]
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        print(f"\nAverage sentiment score: {avg_sentiment:.2f}")
        print(f"(GDELT tone: negative = conflict/violence, positive = cooperation)")


async def run_examples():
    """Run all examples."""
    print("\n🎯 GDELT Historical Mode - Usage Examples\n")
    
    print("Select an example to run:")
    print("  1. Quick test (1 week, ~5-10 min)")
    print("  2. One month fetch (~15-30 min)")
    print("  3. Acquisition study (pre + post windows)")
    print("  4. Lenient filtering mode")
    print("  5. GDELT metadata analysis")
    print("  0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == "1":
        await example_1_quick_test()
    elif choice == "2":
        await example_2_one_month()
    elif choice == "3":
        await example_3_acquisition_study()
    elif choice == "4":
        await example_4_lenient_mode()
    elif choice == "5":
        await example_5_metadata_analysis()
    elif choice == "0":
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice")
        return
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nCheck data/processed/ for output files")
    print("See README.md for more usage patterns")


if __name__ == "__main__":
    try:
        asyncio.run(run_examples())
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()