"""
Run Diverse Observer Experiments - FIXED
"""

import json
import sys
sys.path.append('.')

from complete_pipeline_diverse_FIXED import run_diverse_observer_experiment


def load_articles(filepath):
    """Load articles from JSONL with UTF-8 encoding."""
    articles = []
    with open(filepath, encoding='utf-8') as f:  # FIXED: UTF-8
        for line in f:
            articles.append(json.loads(line))
    return articles


def run_real_corpus(mode='diverse'):
    print("="*70)
    print(f"EXPERIMENT: REAL CORPUS ({mode.upper()} MODE)")
    print("="*70)
    
    articles = load_articles('data/scraped_articles.jsonl')
    print(f"\nLoaded {len(articles)} articles")
    
    results = run_diverse_observer_experiment(
        articles=articles,
        observer_mode=mode,
        use_gru=True,
        use_framing_rope=True,
        device='cuda',
        nli_model_name='microsoft/deberta-v2-xlarge-mnli'
    )
    
    print("\nReal corpus complete")
    return results


def run_shuffle_corpus(mode='diverse'):
    print("\n" + "="*70)
    print(f"EXPERIMENT: SHUFFLE CONTROL ({mode.upper()} MODE)")
    print("="*70)
    
    try:
        articles = load_articles('data/control_length_matched_shuffle.jsonl')
        print(f"\nLoaded {len(articles)} shuffled articles")
    except FileNotFoundError:
        print("\nControl not found! Run: python create_shuffle_control.py")
        return None
    
    results = run_diverse_observer_experiment(
        articles=articles,
        observer_mode=mode,
        use_gru=True,
        use_framing_rope=True,
        device='cuda',
        nli_model_name='microsoft/deberta-v2-xlarge-mnli'
    )
    
    print("\nShuffle control complete")
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='diverse', choices=['baseline', 'diverse', 'extreme'])
    parser.add_argument('--corpus', default='real', choices=['real', 'shuffle', 'both'])
    
    args = parser.parse_args()
    
    if args.corpus in ['real', 'both']:
        run_real_corpus(mode=args.mode)
    
    if args.corpus in ['shuffle', 'both']:
        run_shuffle_corpus(mode=args.mode)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. python inspect_divergent_articles.py")
    print("2. python compare_real_shuffle_random.py")


if __name__ == "__main__":
    main()
