"""
Temporal Scraper

Extends the existing scraper to gather articles from multiple time periods.
Allows comparing same observers across different temporal slices.
"""

import sys
sys.path.append('.')

from belief_ingest.scraper import scrape_period
from datetime import datetime, timedelta
import json
from pathlib import Path


def define_temporal_slices():
    """Define time periods for comparison."""
    return [
        {
            'name': 'Oct_2023',
            'start': datetime(2023, 10, 7),   # Start of current conflict
            'end': datetime(2023, 10, 31),
            'description': 'October 2023 - Conflict onset'
        },
        {
            'name': 'Nov_2023',
            'start': datetime(2023, 11, 1),
            'end': datetime(2023, 11, 30),
            'description': 'November 2023 - Escalation'
        },
        {
            'name': 'Dec_2023',
            'start': datetime(2023, 12, 1),
            'end': datetime(2023, 12, 31),
            'description': 'December 2023 - Ceasefire attempts'
        },
        {
            'name': 'Jan_2024',
            'start': datetime(2024, 1, 1),
            'end': datetime(2024, 1, 31),
            'description': 'January 2024'
        },
        {
            'name': 'Feb_2024',
            'start': datetime(2024, 2, 1),
            'end': datetime(2024, 2, 29),
            'description': 'February 2024'
        },
        {
            'name': 'Mar_2024',
            'start': datetime(2024, 3, 1),
            'end': datetime(2024, 3, 31),
            'description': 'March 2024 - Ramadan'
        }
    ]


def scrape_temporal_slice(slice_config, target_articles=500, output_dir='data/temporal'):
    """Scrape one temporal slice."""
    print("\n" + "="*70)
    print(f"SCRAPING: {slice_config['name']}")
    print(f"Period: {slice_config['start'].date()} to {slice_config['end'].date()}")
    print(f"Description: {slice_config['description']}")
    print("="*70)
    
    # Use your existing scraper
    articles = scrape_period(
        start_date=slice_config['start'],
        end_date=slice_config['end'],
        target_count=target_articles,
        keywords=['Gaza', 'Israel', 'Hamas', 'Palestine']
    )
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{slice_config['name']}.jsonl"
    
    with open(output_path, 'w') as f:
        for article in articles:
            # Add temporal metadata
            article['temporal_slice'] = slice_config['name']
            article['slice_start'] = slice_config['start'].isoformat()
            article['slice_end'] = slice_config['end'].isoformat()
            f.write(json.dumps(article) + '\n')
    
    print(f"\n→ Saved {len(articles)} articles to {output_path}")
    
    return articles


def scrape_all_temporal_slices(target_per_slice=500):
    """Scrape all defined temporal slices."""
    print("="*70)
    print("TEMPORAL SCRAPING PIPELINE")
    print("="*70)
    
    slices = define_temporal_slices()
    
    print(f"\nDefined {len(slices)} temporal slices:")
    for s in slices:
        print(f"  - {s['name']:12s}: {s['description']}")
    
    print(f"\nTarget: {target_per_slice} articles per slice")
    print(f"Total target: {len(slices) * target_per_slice} articles")
    
    results = {}
    
    for slice_config in slices:
        try:
            articles = scrape_temporal_slice(
                slice_config,
                target_articles=target_per_slice
            )
            results[slice_config['name']] = len(articles)
        except Exception as e:
            print(f"✗ Error scraping {slice_config['name']}: {e}")
            results[slice_config['name']] = 0
    
    # Summary
    print("\n" + "="*70)
    print("SCRAPING SUMMARY")
    print("="*70)
    
    total = sum(results.values())
    print(f"\nTotal articles scraped: {total}")
    print("\nPer slice:")
    for name, count in results.items():
        status = "✓" if count >= target_per_slice * 0.8 else "⚠"
        print(f"  {status} {name:12s}: {count:4d} articles")


def load_temporal_slice(slice_name, data_dir='data/temporal'):
    """Load articles from a specific temporal slice."""
    filepath = Path(data_dir) / f"{slice_name}.jsonl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Temporal slice not found: {filepath}")
    
    articles = []
    with open(filepath) as f:
        for line in f:
            articles.append(json.loads(line))
    
    return articles


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Temporal scraper for Belief Transformer')
    parser.add_argument('--target', type=int, default=500, 
                       help='Target articles per slice')
    parser.add_argument('--slice', type=str, default=None,
                       help='Scrape specific slice only (e.g., Oct_2023)')
    
    args = parser.parse_args()
    
    if args.slice:
        # Scrape specific slice
        slices = define_temporal_slices()
        slice_config = next((s for s in slices if s['name'] == args.slice), None)
        if slice_config is None:
            print(f"✗ Unknown slice: {args.slice}")
            print(f"Available: {[s['name'] for s in slices]}")
            return
        
        scrape_temporal_slice(slice_config, target_articles=args.target)
    else:
        # Scrape all
        scrape_all_temporal_slices(target_per_slice=args.target)


if __name__ == "__main__":
    main()
