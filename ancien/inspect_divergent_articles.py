"""
High-Divergence Article Inspector (REMADE)

Automated + interactive inspection of articles where observers most disagree.

Features:
- Automatic spam/quality heuristics
- Text preview generation
- Source pattern analysis
- Optional interactive labeling
- Export for manual review
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import re


def load_articles(filepath='data/scraped_articles.jsonl'):
    """Load all articles with proper encoding."""
    articles = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def compute_spam_heuristics(article):
    """
    Compute heuristic scores for spam detection.
    
    Returns dict of features that suggest low quality.
    """
    # Get text from article - your articles use 'content'
    text = article.get('content', '')
    
    if not text:
        # Return default features if no text found
        return {
            'length': 0,
            'very_short': True,
            'very_long': False,
            'repetitive': False,
            'has_bullets': False,
            'has_multiple_urls': False,
            'has_pipes': False,
            'excessive_caps': False,
            'is_biztoc': False,
            'is_aggregator': False,
            'spam_score': 1.0  # Max spam if no text
        }
    
    features = {}
    
    # Length features
    features['length'] = len(text)
    features['very_short'] = len(text) < 200  # Too short
    features['very_long'] = len(text) > 5000  # Suspiciously long
    
    # Repetition
    words = text.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        features['repetitive'] = unique_ratio < 0.5
    else:
        features['repetitive'] = False
    
    # Special characters (listicles, aggregators)
    features['has_bullets'] = bool(re.search(r'[•·▪▫○●]', text))
    features['has_multiple_urls'] = text.count('http') > 3
    features['has_pipes'] = text.count('|') > 5  # Common in aggregators
    
    # All caps (often spam)
    if len(text) > 50:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        features['excessive_caps'] = caps_ratio > 0.3
    else:
        features['excessive_caps'] = False
    
    # Source patterns - your articles use 'publisher'
    source = article.get('publisher', '').lower()
    features['is_biztoc'] = 'biztoc' in source
    features['is_aggregator'] = any(x in source for x in ['biztoc', 'yahoo', 'msn', 'google'])
    
    # Spam score (0-1)
    spam_indicators = [
        features['very_short'],
        features['very_long'],
        features['repetitive'],
        features['has_bullets'],
        features['has_multiple_urls'],
        features['has_pipes'],
        features['excessive_caps'],
        features['is_biztoc']
    ]
    
    features['spam_score'] = sum(spam_indicators) / len(spam_indicators)
    
    return features


def analyze_divergent_articles(n=100, auto_label=True):
    """
    Analyze high-divergence articles.
    
    Parameters
    ----------
    n : int
        Number of top articles to analyze
    auto_label : bool
        If True, use heuristics to auto-label spam
    """
    print("="*70)
    print("DIVERGENT ARTICLE ANALYZER")
    print("="*70)
    
    # Load divergence scores
    try:
        divergence_df = pd.read_csv('outputs/high_divergence_articles.csv')
    except FileNotFoundError:
        print("✗ high_divergence_articles.csv not found!")
        print("  Run: python compare_observers.py first")
        return
    
    # Load articles
    print(f"\nLoading articles...")
    try:
        articles = load_articles('data/scraped_articles.jsonl')
    except FileNotFoundError:
        print("✗ data/scraped_articles.jsonl not found!")
        return
    
    print(f"  Loaded {len(articles)} articles")
    
    # Get top N
    top_divergent = divergence_df.head(n).copy()
    
    print(f"\nAnalyzing top {n} most divergent articles...")
    
    # Analyze each
    analysis_results = []
    
    for i, row in top_divergent.iterrows():
        idx = int(row['index'])
        divergence = float(row['divergence'])
        source = row['source']
        
        article = articles[idx]
        
        # Compute features
        features = compute_spam_heuristics(article)
        
        # Auto-label if enabled
        if auto_label:
            if features['spam_score'] > 0.5:
                auto_category = 'SPAM'
            elif features['is_biztoc']:
                auto_category = 'AGGREGATOR'
            elif features['very_short']:
                auto_category = 'TOO_SHORT'
            else:
                auto_category = 'UNKNOWN'
        else:
            auto_category = 'UNKNOWN'
        
        # Text preview - your articles use 'content'
        text = article.get('content', '')
        preview = text[:300].replace('\n', ' ') if text else 'NO TEXT FOUND'
        
        analysis_results.append({
            'index': idx,
            'divergence': divergence,
            'publisher': article.get('publisher', 'unknown'),
            'auto_category': auto_category,
            'spam_score': features['spam_score'],
            'length': features['length'],
            'repetitive': features['repetitive'],
            'is_biztoc': features['is_biztoc'],
            'is_aggregator': features['is_aggregator'],
            'preview': preview,
            'url': article.get('url', '')
        })
    
    results_df = pd.DataFrame(analysis_results)
    
    # Summary statistics
    print("\n" + "="*70)
    print("AUTOMATIC ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nAuto-labeled categories:")
    category_counts = results_df['auto_category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    print(f"\nSpam score distribution:")
    print(f"  Mean: {results_df['spam_score'].mean():.3f}")
    print(f"  High spam (>0.5): {(results_df['spam_score'] > 0.5).sum()} ({(results_df['spam_score'] > 0.5).mean()*100:.1f}%)")
    
    print(f"\nPublisher breakdown:")
    publisher_counts = results_df['publisher'].value_counts().head(10)
    for publisher, count in publisher_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {publisher}: {count} ({pct:.1f}%)")
    
    print(f"\nLength statistics:")
    print(f"  Mean: {results_df['length'].mean():.0f} chars")
    print(f"  Median: {results_df['length'].median():.0f} chars")
    print(f"  Very short (<200): {results_df['length'].lt(200).sum()}")
    print(f"  Very long (>5000): {results_df['length'].gt(5000).sum()}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    spam_pct = (results_df['spam_score'] > 0.5).mean() * 100
    biztoc_pct = results_df['is_biztoc'].mean() * 100
    aggregator_pct = results_df['is_aggregator'].mean() * 100
    
    if spam_pct > 60:
        print(f"\n✗✗ MAJOR PROBLEM: {spam_pct:.1f}% high spam score")
        print("  Divergence is driven by low-quality content.")
        print("  Need to filter spam sources before making claims.")
    elif aggregator_pct > 70:
        print(f"\n✗ PROBLEM: {aggregator_pct:.1f}% from aggregators")
        print("  Divergence may reflect aggregator formatting, not semantics.")
        print("  Consider excluding aggregator sources.")
    elif spam_pct > 40:
        print(f"\n⚠ MODERATE CONCERN: {spam_pct:.1f}% high spam score")
        print("  Mix of quality and spam content.")
        print("  Results need qualification about data quality.")
    else:
        print(f"\n✓ ACCEPTABLE: {spam_pct:.1f}% high spam score")
        print("  Most divergent articles appear legitimate.")
        print("  Divergence likely reflects genuine semantic variance.")
    
    # Save detailed results
    output_file = 'outputs/divergent_articles_analysis.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n→ Saved detailed analysis to {output_file}")
    
    # Generate human-readable report
    report_file = 'outputs/divergent_articles_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HIGH-DIVERGENCE ARTICLES: DETAILED REPORT\n")
        f.write("="*70 + "\n\n")
        
        for i, row in results_df.head(20).iterrows():
            f.write(f"\n{'='*70}\n")
            f.write(f"ARTICLE #{i+1}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Index: {row['index']}\n")
            f.write(f"Divergence: {row['divergence']:.2f}\n")
            f.write(f"Publisher: {row['publisher']}\n")
            f.write(f"Auto Category: {row['auto_category']}\n")
            f.write(f"Spam Score: {row['spam_score']:.3f}\n")
            f.write(f"Length: {row['length']} chars\n")
            f.write(f"URL: {row['url']}\n")
            f.write(f"\nPreview:\n{row['preview']}\n")
    
    print(f"→ Saved human-readable report to {report_file}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Review: outputs/divergent_articles_report.txt")
    print("2. Open: outputs/divergent_articles_analysis.csv (in Excel)")
    print("3. If spam > 50%: Filter biztoc.com and re-run experiments")
    
    return results_df


def interactive_review(df, start_idx=0):
    """
    Interactive review of articles after automatic analysis.
    
    Allows manual correction of auto-labels.
    """
    print("\n" + "="*70)
    print("INTERACTIVE REVIEW")
    print("="*70)
    
    print("\nInstructions:")
    print("  [S] Spam/low-quality")
    print("  [C] Contested/ambiguous topic")
    print("  [F] Formatting/structural issue")
    print("  [L] Legitimate article")
    print("  [K] Keep auto-label")
    print("  [Q] Quit")
    
    df = df.copy()
    df['manual_category'] = df['auto_category']
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        print("\n" + "="*70)
        print(f"Article {i+1}/{len(df)}")
        print("="*70)
        print(f"Index: {row['index']}")
        print(f"Divergence: {row['divergence']:.2f}")
        print(f"Publisher: {row['publisher']}")
        print(f"Auto-label: {row['auto_category']} (spam score: {row['spam_score']:.3f})")
        print(f"\nPreview:\n{row['preview'][:400]}")
        print("-"*70)
        
        while True:
            choice = input("\nCategory [S/C/F/L/K/Q]: ").strip().upper()
            
            if choice == 'Q':
                print("\nSaving and quitting...")
                df.to_csv('outputs/divergent_articles_manual.csv', index=False, encoding='utf-8')
                return df
            elif choice == 'K':
                break
            elif choice == 'S':
                df.at[i, 'manual_category'] = 'SPAM'
                break
            elif choice == 'C':
                df.at[i, 'manual_category'] = 'CONTESTED'
                break
            elif choice == 'F':
                df.at[i, 'manual_category'] = 'FORMATTING'
                break
            elif choice == 'L':
                df.at[i, 'manual_category'] = 'LEGITIMATE'
                break
            else:
                print("Invalid. Use S, C, F, L, K, or Q")
    
    # Save
    df.to_csv('outputs/divergent_articles_manual.csv', index=False, encoding='utf-8')
    print(f"\n→ Saved manual labels to outputs/divergent_articles_manual.csv")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze high-divergence articles')
    parser.add_argument('--n', type=int, default=100, help='Number of articles to analyze')
    parser.add_argument('--mode', choices=['auto', 'interactive'], default='auto',
                       help='auto: automatic analysis only, interactive: manual review after')
    
    args = parser.parse_args()
    
    # Run automatic analysis
    results_df = analyze_divergent_articles(n=args.n, auto_label=True)
    
    # Interactive review if requested
    if args.mode == 'interactive' and results_df is not None:
        print("\n" + "="*70)
        print("Starting interactive review...")
        print("="*70)
        interactive_review(results_df)
