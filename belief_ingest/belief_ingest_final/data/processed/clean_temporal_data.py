"""
Clean and filter temporal Gaza/Israel article batches

Controls:
1. TOPIC: Only Gaza/Israel articles (filter by keywords)
2. LENGTH: Standardize article lengths (min/max word count)
3. QUALITY: Remove nonsense, duplicates, non-English

Output: Clean, controlled corpus ready for pipeline
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import hashlib


class TemporalDataCleaner:
    """Clean temporal article batches with topic and length controls"""
    
    def __init__(self,
                 min_words: int = 100,
                 max_words: int = 2000,
                 require_keywords: List[str] = None,
                 min_keyword_density: float = 0.001):
        """
        Parameters:
        -----------
        min_words : Minimum article length
        max_words : Maximum article length
        require_keywords : Keywords that MUST appear (Gaza/Israel related)
        min_keyword_density : Minimum fraction of keywords in text
        """
        
        self.min_words = min_words
        self.max_words = max_words
        
        # Default: Gaza/Israel keywords
        if require_keywords is None:
            self.keywords = [
                'gaza', 'israel', 'israeli', 'palestine', 'palestinian',
                'hamas', 'idf', 'west bank', 'netanyahu', 'tel aviv',
                'jerusalem', 'rafah', 'khan younis', 'hebron', 'nablus'
            ]
        else:
            self.keywords = [k.lower() for k in require_keywords]
        
        self.min_keyword_density = min_keyword_density
        
        # Statistics
        self.stats = {
            'total_articles': 0,
            'too_short': 0,
            'too_long': 0,
            'no_keywords': 0,
            'duplicates': 0,
            'non_english': 0,
            'nonsense': 0,
            'kept': 0
        }
        
        self.seen_hashes = set()
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def has_required_keywords(self, text: str) -> tuple:
        """
        Check if text contains required keywords
        
        Returns:
            (has_keywords: bool, density: float, matched_keywords: list)
        """
        text_lower = text.lower()
        
        # Count keyword occurrences
        matched = []
        total_keyword_words = 0
        
        for keyword in self.keywords:
            count = text_lower.count(keyword)
            if count > 0:
                matched.append((keyword, count))
                total_keyword_words += count * len(keyword.split())
        
        # Calculate density
        total_words = self.count_words(text)
        density = total_keyword_words / total_words if total_words > 0 else 0
        
        has_keywords = len(matched) > 0 and density >= self.min_keyword_density
        
        return has_keywords, density, matched
    
    def is_duplicate(self, article: dict) -> bool:
        """Check if article is duplicate (by content hash)"""
        content = article.get('content', '') + article.get('title', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def is_nonsense(self, text: str) -> bool:
        """
        Detect nonsense/malformed articles
        
        Heuristics:
        - Too many special characters
        - Too many repeated words
        - No proper sentences
        """
        
        # Check 1: Excessive special characters
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        total_chars = len(text)
        if total_chars > 0 and special_chars / total_chars > 0.3:
            return True
        
        # Check 2: Excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            if most_common_count / len(words) > 0.2:  # Same word >20% of text
                return True
        
        # Check 3: No proper sentences
        sentences = re.split(r'[.!?]+', text)
        proper_sentences = [s for s in sentences if len(s.split()) >= 5]
        if len(proper_sentences) < 2:
            return True
        
        return False
    
    def is_likely_english(self, text: str) -> bool:
        """
        Simple English detection
        
        Checks for common English words
        """
        english_indicators = [
            'the', 'and', 'is', 'to', 'in', 'of', 'a', 'for',
            'said', 'that', 'has', 'have', 'was', 'were'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for word in english_indicators if word in text_lower.split())
        
        return matches >= 5  # At least 5 common English words
    
    def clean_article(self, article: dict) -> Optional[dict]:
        """
        Clean and validate single article
        
        Returns:
            Cleaned article dict or None if rejected
        """
        
        self.stats['total_articles'] += 1
        
        # Get content
        content = article.get('content', '')
        title = article.get('title', '')
        full_text = f"{title} {content}"
        
        # Check 1: Length
        word_count = self.count_words(content)
        
        if word_count < self.min_words:
            self.stats['too_short'] += 1
            return None
        
        if word_count > self.max_words:
            self.stats['too_long'] += 1
            return None
        
        # Check 2: Keywords (topic control)
        has_keywords, density, matched = self.has_required_keywords(full_text)
        
        if not has_keywords:
            self.stats['no_keywords'] += 1
            return None
        
        # Check 3: Duplicates
        if self.is_duplicate(article):
            self.stats['duplicates'] += 1
            return None
        
        # Check 4: English
        if not self.is_likely_english(full_text):
            self.stats['non_english'] += 1
            return None
        
        # Check 5: Nonsense
        if self.is_nonsense(full_text):
            self.stats['nonsense'] += 1
            return None
        
        # Passed all checks!
        self.stats['kept'] += 1
        
        # Add metadata
        article['word_count'] = word_count
        article['keyword_density'] = density
        article['matched_keywords'] = [k for k, _ in matched]
        
        return article
    
    def clean_batch(self, input_path: Path, output_path: Path) -> dict:
        """
        Clean single batch file
        
        Returns:
            Statistics dict
        """
        
        print(f"\nProcessing: {input_path.name}")
        print(f"  Size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        cleaned_articles = []
        
        # Read and clean
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    article = json.loads(line)
                    cleaned = self.clean_article(article)
                    
                    if cleaned is not None:
                        cleaned_articles.append(cleaned)
                    
                    if line_num % 1000 == 0:
                        print(f"    Processed {line_num} articles, kept {len(cleaned_articles)}")
                
                except json.JSONDecodeError:
                    print(f"    [ERROR] JSON error on line {line_num}")
                    continue
                except Exception as e:
                    print(f"    [ERROR] Error on line {line_num}: {e}")
                    continue
        
        # Write cleaned batch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in cleaned_articles:
                f.write(json.dumps(article) + '\n')
        
        print(f"  [OK] Wrote {len(cleaned_articles)} articles to {output_path.name}")
        
        return {
            'input_file': input_path.name,
            'output_file': output_path.name,
            'kept': len(cleaned_articles)
        }
    
    def print_summary(self):
        """Print cleaning statistics"""
        
        print("\n" + "="*70)
        print("CLEANING SUMMARY")
        print("="*70)
        
        total = self.stats['total_articles']
        
        print(f"Total articles processed: {total}")
        print(f"\nRejected:")
        print(f"  - Too short (<{self.min_words} words): {self.stats['too_short']} ({self.stats['too_short']/total*100:.1f}%)")
        print(f"  - Too long (>{self.max_words} words): {self.stats['too_long']} ({self.stats['too_long']/total*100:.1f}%)")
        print(f"  - No keywords (topic): {self.stats['no_keywords']} ({self.stats['no_keywords']/total*100:.1f}%)")
        print(f"  - Duplicates: {self.stats['duplicates']} ({self.stats['duplicates']/total*100:.1f}%)")
        print(f"  - Non-English: {self.stats['non_english']} ({self.stats['non_english']/total*100:.1f}%)")
        print(f"  - Nonsense: {self.stats['nonsense']} ({self.stats['nonsense']/total*100:.1f}%)")
        
        print(f"\n[OK] Kept: {self.stats['kept']} ({self.stats['kept']/total*100:.1f}%)")
        print("="*70)


def clean_all_batches(input_dir: Path,
                      output_dir: Path,
                      pattern: str = "batch_*.jsonl",
                      **cleaner_kwargs):
    """
    Clean all batch files in directory
    
    Parameters:
    -----------
    input_dir : Directory with raw batches
    output_dir : Directory for cleaned batches
    pattern : File pattern to match
    **cleaner_kwargs : Arguments for TemporalDataCleaner
    """
    
    # Find batch files
    batch_files = sorted(input_dir.glob(pattern))
    
    if not batch_files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir}")
    
    print("="*70)
    print("TEMPORAL DATA CLEANING")
    print("="*70)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(batch_files)} batch files")
    
    # Create cleaner
    cleaner = TemporalDataCleaner(**cleaner_kwargs)
    
    print(f"\nCleaning parameters:")
    print(f"  - Word count: {cleaner.min_words} - {cleaner.max_words}")
    print(f"  - Keywords: {', '.join(cleaner.keywords[:10])}...")
    print(f"  - Min keyword density: {cleaner.min_keyword_density:.4f}")
    
    # Clean each batch
    batch_results = []
    
    for batch_file in batch_files:
        output_file = output_dir / f"cleaned_{batch_file.name}"
        result = cleaner.clean_batch(batch_file, output_file)
        batch_results.append(result)
    
    # Summary
    cleaner.print_summary()
    
    print("\nPer-batch results:")
    for result in batch_results:
        print(f"  {result['input_file']} -> {result['output_file']}: {result['kept']} articles")
    
    print(f"\n[OK] Cleaned batches saved to: {output_dir}")
    
    return batch_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean temporal Gaza/Israel article batches"
    )
    
    repo_root = Path(__file__).resolve().parents[4]

    parser.add_argument('--input_dir', type=str,
                       default=str(repo_root / 'belief_ingest' / 'belief_ingest_final' / 'data' / 'processed'),
                       help='Directory with raw batch files')
    parser.add_argument('--output_dir', type=str,
                       default=str(repo_root / 'data' / 'temporal_cleaned'),
                       help='Directory for cleaned batches')
    parser.add_argument('--pattern', type=str, default='batch_*.jsonl',
                       help='File pattern to match')
    parser.add_argument('--min_words', type=int, default=100,
                       help='Minimum article length (words)')
    parser.add_argument('--max_words', type=int, default=2000,
                       help='Maximum article length (words)')
    parser.add_argument('--min_keyword_density', type=float, default=0.001,
                       help='Minimum keyword density (fraction)')
    
    args = parser.parse_args()
    
    clean_all_batches(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        pattern=args.pattern,
        min_words=args.min_words,
        max_words=args.max_words,
        min_keyword_density=args.min_keyword_density
    )
