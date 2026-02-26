"""
Utility functions for the ingestion pipeline.
"""

import hashlib
import re
from datetime import datetime
from typing import Set, Optional
from pathlib import Path
import xxhash
import structlog

logger = structlog.get_logger()


class URLDeduplicator:
    """Fast URL deduplication using xxhash and bloom-like set."""
    
    def __init__(self, cache_file: Path, max_size: int = 100000):
        self.cache_file = cache_file
        self.max_size = max_size
        self.seen: Set[str] = set()
        self._load_cache()
    
    def _load_cache(self):
        """Load previously seen URLs from cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.seen = set(line.strip() for line in f if line.strip())
                logger.info("dedup_cache_loaded", count=len(self.seen))
            except Exception as e:
                logger.warning("dedup_cache_load_failed", error=str(e))
    
    def _save_cache(self):
        """Persist seen URLs to disk."""
        try:
            # Keep only recent URLs if we exceed max size
            if len(self.seen) > self.max_size:
                self.seen = set(list(self.seen)[-self.max_size:])
            
            with open(self.cache_file, 'w') as f:
                f.write('\n'.join(self.seen))
            logger.debug("dedup_cache_saved", count=len(self.seen))
        except Exception as e:
            logger.warning("dedup_cache_save_failed", error=str(e))
    
    def is_duplicate(self, url: str) -> bool:
        """Check if URL has been seen before."""
        url_hash = self._hash_url(url)
        return url_hash in self.seen
    
    def mark_seen(self, url: str):
        """Mark URL as seen."""
        url_hash = self._hash_url(url)
        self.seen.add(url_hash)
        
        # Periodically save to disk
        if len(self.seen) % 1000 == 0:
            self._save_cache()
    
    def _hash_url(self, url: str) -> str:
        """Fast URL hashing using xxhash."""
        # Normalize URL first
        url = self._normalize_url(url)
        return xxhash.xxh64(url.encode()).hexdigest()
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for deduplication."""
        url = url.lower().strip()
        # Remove tracking parameters
        url = re.sub(r'[?&](utm_|fbclid|gclid|ref_|source=|campaign=)[^&]*', '', url)
        # Remove trailing slashes
        url = url.rstrip('/')
        # Remove www
        url = url.replace('://www.', '://')
        return url
    
    def close(self):
        """Save cache and cleanup."""
        self._save_cache()


def generate_article_id(url: str, title: str, published_at: Optional[str] = None) -> str:
    """
    Generate deterministic article ID from URL + title + date.
    Uses SHA256 for collision resistance.
    """
    components = [url, title]
    if published_at:
        components.append(published_at)
    
    content = '|'.join(components).encode('utf-8')
    return hashlib.sha256(content).hexdigest()[:16]


def normalize_url(url: str) -> str:
    """Clean and normalize URL."""
    url = url.strip()
    
    # Remove AMP and mobile variants
    url = re.sub(r'/amp[/.]?', '/', url)
    url = re.sub(r'm\.', '', url)
    
    # Remove tracking
    url = re.sub(r'[?&](utm_|fbclid|gclid|ref_|source=|campaign=)[^&]*', '', url)
    
    # Clean up
    url = url.rstrip('/')
    url = url.replace(' ', '%20')
    
    return url


def extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    match = re.search(r'://([^/]+)', url)
    if match:
        domain = match.group(1)
        # Remove www
        domain = re.sub(r'^www\.', '', domain)
        return domain
    return ""


def clean_text(text: str) -> str:
    """Clean article text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common boilerplate
    boilerplate_patterns = [
        r'Sign up for our newsletter',
        r'Subscribe to.*?newsletter',
        r'Follow us on Twitter',
        r'Like us on Facebook',
        r'Advertisement',
        r'ADVERTISEMENT',
        r'Sponsored Content',
        r'This article was amended on',
        r'This story has been updated',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def segment_text(text: str, max_words: int = 100) -> list[str]:
    """
    Segment article text into chunks for Belief Transformer.
    Preserves sentence boundaries.
    """
    if not text:
        return []
    
    # Split into sentences (rough)
    sentences = re.split(r'[.!?]\s+', text)
    
    segments = []
    current_segment = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = sentence.split()
        sentence_word_count = len(words)
        
        if current_word_count + sentence_word_count <= max_words:
            current_segment.append(sentence)
            current_word_count += sentence_word_count
        else:
            # Finish current segment
            if current_segment:
                segments.append(' '.join(current_segment) + '.')
            
            # Start new segment
            current_segment = [sentence]
            current_word_count = sentence_word_count
    
    # Add final segment
    if current_segment:
        segments.append(' '.join(current_segment) + '.')
    
    return segments


def parse_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parse various date formats to ISO 8601.
    Returns None if parsing fails.
    """
    if not date_str:
        return None
    
    from dateutil import parser
    
    try:
        dt = parser.parse(date_str)
        return dt.isoformat()
    except Exception:
        return None


class MetricsCollector:
    """Simple metrics collector for monitoring."""
    
    def __init__(self):
        self.counters = {
            "articles_fetched": 0,
            "articles_success": 0,
            "articles_failed": 0,
            "extraction_trafilatura": 0,
            "extraction_newspaper": 0,
            "extraction_readability": 0,
            "extraction_playwright": 0,
            "extraction_failed": 0,
            "urls_duplicate": 0,
            "urls_too_short": 0,
        }
        self.timers = {
            "total_fetch_time": 0.0,
            "total_extraction_time": 0.0,
        }
        self.start_time = datetime.now()
    
    def increment(self, counter: str, value: int = 1):
        """Increment a counter."""
        if counter in self.counters:
            self.counters[counter] += value
    
    def add_time(self, timer: str, seconds: float):
        """Add time to a timer."""
        if timer in self.timers:
            self.timers[timer] += seconds
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        success_rate = 0.0
        if self.counters["articles_fetched"] > 0:
            success_rate = self.counters["articles_success"] / self.counters["articles_fetched"]
        
        playwright_rate = 0.0
        total_extractions = sum(self.counters[f"extraction_{m}"] for m in 
                               ["trafilatura", "newspaper", "readability", "playwright"])
        if total_extractions > 0:
            playwright_rate = self.counters["extraction_playwright"] / total_extractions
        
        avg_fetch_time = 0.0
        if self.counters["articles_fetched"] > 0:
            avg_fetch_time = self.timers["total_fetch_time"] / self.counters["articles_fetched"]
        
        return {
            "runtime_seconds": runtime,
            "counters": self.counters.copy(),
            "rates": {
                "success_rate": success_rate,
                "playwright_usage_rate": playwright_rate,
                "articles_per_second": self.counters["articles_success"] / runtime if runtime > 0 else 0,
            },
            "timing": {
                "avg_fetch_time_seconds": avg_fetch_time,
            }
        }
    
    def log_stats(self):
        """Log current statistics."""
        stats = self.get_stats()
        logger.info("metrics", **stats)
        return stats


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped (not a news article)."""
    skip_patterns = [
        r'/video/',
        r'/videos/',
        r'/gallery/',
        r'/live/',
        r'/liveblog/',
        r'/podcast/',
        r'/audio/',
        r'/interactive/',
        r'/graphics/',
        r'/opinion/',  # Optional: skip opinion pieces
        r'/sponsored/',
        r'/advertisement/',
    ]
    
    url_lower = url.lower()
    for pattern in skip_patterns:
        if re.search(pattern, url_lower):
            return True
    
    return False
