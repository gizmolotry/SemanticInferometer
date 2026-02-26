"""
Control Corpus Generator

Creates three types of control corpora:
1. Class A (Constant): Identical article text
2. Class B (Shuffled): Same tokens, random order
3. Class C (Random): Random vocabulary soup

Plus Gaussian mode for pure architectural baseline.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict


def make_constant_corpus(n_articles=500, seed=42) -> List[Dict]:
    """Class A: Same article repeated n times."""
    np.random.seed(seed)
    
    base_text = """
    The ongoing conflict in the Gaza Strip continues to evolve as international 
    observers monitor the humanitarian situation. Multiple stakeholders have 
    expressed concerns about civilian casualties and infrastructure damage. 
    Diplomatic efforts remain ongoing as regional tensions persist.
    """
    
    articles = []
    for i in range(n_articles):
        articles.append({
            'text': base_text.strip(),
            'source': 'ControlConstant',
            'timestamp': 1704067200 + i * 3600,  # Jan 2024 + hourly
            'url': f'https://control.constant/{i}',
            'control_class': 'constant',
            'control_index': i
        })
    
    return articles


def make_shuffled_corpus(n_articles=500, seed=42) -> List[Dict]:
    """Class B: Same vocabulary, shuffled word order."""
    np.random.seed(seed)
    
    base_text = """
    The ongoing conflict in the Gaza Strip continues to evolve as international 
    observers monitor the humanitarian situation. Multiple stakeholders have 
    expressed concerns about civilian casualties and infrastructure damage. 
    Diplomatic efforts remain ongoing as regional tensions persist.
    """
    
    tokens = base_text.strip().split()
    
    articles = []
    for i in range(n_articles):
        shuffled_tokens = np.random.permutation(tokens)
        articles.append({
            'text': ' '.join(shuffled_tokens),
            'source': 'ControlShuffled',
            'timestamp': 1704067200 + i * 3600,
            'url': f'https://control.shuffled/{i}',
            'control_class': 'shuffled',
            'control_index': i
        })
    
    return articles


def make_random_corpus(n_articles=500, seed=42) -> List[Dict]:
    """Class C: Random vocabulary soup."""
    np.random.seed(seed)
    
    # Common English words
    vocab = [
        'the', 'and', 'is', 'to', 'in', 'of', 'a', 'that', 'it', 'with',
        'as', 'for', 'on', 'was', 'are', 'by', 'this', 'from', 'or', 'at',
        'conflict', 'situation', 'international', 'humanitarian', 'civilian',
        'ongoing', 'regional', 'diplomatic', 'security', 'military', 'peace',
        'violence', 'tensions', 'crisis', 'refugees', 'aid', 'negotiations'
    ]
    
    # Same length as base text (~50 tokens)
    n_tokens = 50
    
    articles = []
    for i in range(n_articles):
        random_tokens = np.random.choice(vocab, size=n_tokens, replace=True)
        articles.append({
            'text': ' '.join(random_tokens),
            'source': 'ControlRandom',
            'timestamp': 1704067200 + i * 3600,
            'url': f'https://control.random/{i}',
            'control_class': 'random',
            'control_index': i
        })
    
    return articles


def make_three_class_control(n_per_class=500, seed=42) -> List[Dict]:
    """Generate all three control classes."""
    print(f"Generating three-class control corpus...")
    print(f"  Articles per class: {n_per_class}")
    print(f"  Seed: {seed}")
    
    constant = make_constant_corpus(n_per_class, seed)
    shuffled = make_shuffled_corpus(n_per_class, seed)
    random_corp = make_random_corpus(n_per_class, seed)
    
    all_articles = constant + shuffled + random_corp
    
    print(f"\nGenerated:")
    print(f"  Constant:  {len(constant)} articles")
    print(f"  Shuffled:  {len(shuffled)} articles")
    print(f"  Random:    {len(random_corp)} articles")
    print(f"  Total:     {len(all_articles)} articles")
    
    return all_articles


def save_control_corpus(articles: List[Dict], output_path='data/control_corpus.jsonl'):
    """Save control corpus to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for article in articles:
            f.write(json.dumps(article) + '\n')
    
    print(f"\n→ Saved {len(articles)} articles to {output_path}")


def main():
    print("="*70)
    print("CONTROL CORPUS GENERATOR")
    print("="*70)
    
    # Generate control corpus
    control_articles = make_three_class_control(n_per_class=500, seed=42)
    
    # Save
    save_control_corpus(control_articles, 'data/control_corpus.jsonl')
    
    # Stats
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)
    
    from collections import Counter
    classes = Counter(a['control_class'] for a in control_articles)
    sources = Counter(a['source'] for a in control_articles)
    
    print("\nBy control class:")
    for cls, count in classes.items():
        print(f"  {cls:12s}: {count:4d} articles")
    
    print("\nBy source:")
    for src, count in sources.items():
        print(f"  {src:20s}: {count:4d} articles")
    
    # Sample
    print("\n" + "="*70)
    print("SAMPLES")
    print("="*70)
    
    for cls in ['constant', 'shuffled', 'random']:
        article = next(a for a in control_articles if a['control_class'] == cls)
        print(f"\n{cls.upper()}:")
        print(f"  Text: {article['text'][:100]}...")
        print(f"  Source: {article['source']}")


if __name__ == "__main__":
    main()
