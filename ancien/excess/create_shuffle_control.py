"""
Length-Matched Shuffle Control

THE MOST CRITICAL ABLATION

This tests whether variance is driven by:
- Length differences (bad - means variance is shallow)
- Semantic content (good - means variance is deep)

Method:
1. Take real articles
2. Shuffle tokens WITHIN each article
3. Preserves: length, vocabulary distribution
4. Destroys: semantic relationships, word order
5. Run same observers
6. Compare variance

If Real >> Shuffle → semantics matter
If Real ≈ Shuffle → it's just length
"""

import json
import random
from pathlib import Path
from datetime import datetime


def convert_date_to_timestamp(date_string):
    """Convert ISO date string to Unix timestamp."""
    try:
        if isinstance(date_string, (int, float)):
            return date_string
        dt = datetime.fromisoformat(str(date_string).replace('Z', '+00:00'))
        return int(dt.timestamp())
    except:
        return 0  # Default if conversion fails


def create_length_matched_shuffle_control(
    input_file='data/scraped_articles.jsonl',
    output_file='data/control_length_matched_shuffle.jsonl',
    seed=42
):
    """
    Create length-matched shuffle control.
    
    For each article:
    - Split into tokens
    - Shuffle tokens randomly
    - Reconstruct as text
    - Preserve metadata
    """
    print("="*70)
    print("LENGTH-MATCHED SHUFFLE CONTROL")
    print("="*70)
    
    random.seed(seed)
    
    # Load articles
    print(f"\nLoading articles from {input_file}...")
    articles = []
    with open(input_file, encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    
    print(f"  Loaded {len(articles)} articles")
    
    # Create shuffled versions
    print("\nShuffling tokens within each article...")
    shuffled_articles = []
    
    for i, article in enumerate(articles):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(articles)}...")
        
        # Get text from article - your articles use 'content' not 'text'
        text = article.get('content', '')
        
        if not text:
            print(f"Warning: No content for article {i}, skipping")
            continue
        
        # Split into tokens (simple whitespace split)
        tokens = text.split()
        
        # Shuffle
        shuffled_tokens = tokens.copy()
        random.shuffle(shuffled_tokens)
        
        # Reconstruct
        shuffled_text = ' '.join(shuffled_tokens)
        
        # Preserve metadata
        shuffled_article = {
            'content': shuffled_text,  # Use 'content' to match your format
            'url': article.get('url', ''),
            'publisher': article.get('publisher', 'unknown'),
            'source': article.get('publisher', 'unknown'),  # Add for provenance compatibility
            'source_type': article.get('source_type', 'unknown'),
            'published_at': article.get('published_at', ''),
            'timestamp': convert_date_to_timestamp(article.get('published_at', 0)),  # Convert to numeric
            'title': article.get('title', ''),
            'original_length': len(tokens),
            'control_type': 'length_matched_shuffle'
        }
        
        shuffled_articles.append(shuffled_article)
    
    # Save
    print(f"\nSaving to {output_file}...")
    Path(output_file).parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in shuffled_articles:
            f.write(json.dumps(article) + '\n')
    
    print(f"  ✓ Saved {len(shuffled_articles)} shuffled articles")
    
    # Stats
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    # Calculate original lengths
    original_lengths = [len(a.get('content', '').split()) for a in articles]
    shuffled_lengths = [a['original_length'] for a in shuffled_articles]
    
    print(f"\nOriginal articles:")
    print(f"  Mean length: {sum(original_lengths)/len(original_lengths):.1f} tokens")
    print(f"  Min/Max: {min(original_lengths)}/{max(original_lengths)}")
    
    print(f"\nShuffled articles:")
    print(f"  Mean length: {sum(shuffled_lengths)/len(shuffled_lengths):.1f} tokens")
    print(f"  Min/Max: {min(shuffled_lengths)}/{max(shuffled_lengths)}")
    
    print("\n" + "="*70)
    print("✓ CONTROL GENERATION COMPLETE")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Run: python run_diverse_experiments.py --corpus shuffle")
    print("2. Run: python compare_real_shuffle_random.py")
    
    return shuffled_articles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/scraped_articles.jsonl')
    parser.add_argument('--output', default='data/control_length_matched_shuffle.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_length_matched_shuffle_control(
        input_file=args.input,
        output_file=args.output,
        seed=args.seed
    )