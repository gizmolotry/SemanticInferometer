"""
Utility helpers for Belief Transformer ingestion pipeline.
"""

import xxhash
import re
from pathlib import Path
from typing import Set, Optional, List
from datetime import datetime
from urllib.parse import urlparse
import structlog

logger = structlog.get_logger()


class URLDeduplicator:
    """
    Fast URL deduplication using xxhash and disk-backed cache.
    """
    
    def __init__(self, cache_file: Path, max_size: int = 100000):
        self.cache_file = cache_file
        self.max_size = max_size
        self.seen_urls: Set[str] = set()
        
        # Load existing cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.seen_urls = set(line.strip() for line in f)
                logger.info("loaded_url_cache", count=len(self.seen_urls))
            except Exception as e:
                logger.warning("cache_load_failed", error=str(e))
    
    def is_duplicate(self, url: str) -> bool:
        """Check if URL has been seen before."""
        url_hash = self._hash_url(url)
        return url_hash in self.seen_urls
    
    def mark_seen(self, url: str):
        """Mark URL as seen."""
        url_hash = self._hash_url(url)
        self.seen_urls.add(url_hash)
        
        # Write to cache file (append mode)
        try:
            with open(self.cache_file, 'a') as f:
                f.write(f"{url_hash}\n")
        except Exception as e:
            logger.warning("cache_write_failed", error=str(e))
        
        # Trim cache if too large
        if len(self.seen_urls) > self.max_size:
            self._trim_cache()
    
    @staticmethod
    def _hash_url(url: str) -> str:
        """Fast URL hashing."""
        return xxhash.xxh64(url.encode()).hexdigest()
    
    def _trim_cache(self):
        """Trim cache to max size."""
        logger.info("trimming_cache", old_size=len(self.seen_urls))
        
        # Keep most recent entries
        recent = list(self.seen_urls)[-self.max_size:]
        self.seen_urls = set(recent)
        
        # Rewrite cache file
        try:
            with open(self.cache_file, 'w') as f:
                for url_hash in recent:
                    f.write(f"{url_hash}\n")
        except Exception as e:
            logger.error("cache_trim_failed", error=str(e))
    
    def close(self):
        """Cleanup (cache is auto-saved)."""
        pass


def generate_article_id(url: str, title: str, published_at: Optional[str] = None) -> str:
    """
    Generate unique article ID from URL + title + date.
    Returns 16-character hex string.
    """
    # Combine URL and title
    content = f"{url}|{title}"
    if published_at:
        content += f"|{published_at}"
    
    # Hash to 64-bit
    return xxhash.xxh64(content.encode()).hexdigest()


def should_skip_url(url: str) -> bool:
    """
    Check if URL should be skipped (videos, images, etc.).
    """
    skip_patterns = [
        r'/video/',
        r'/videos/',
        r'/gallery/',
        r'/photo/',
        r'/image/',
        r'/podcast/',
        r'/audio/',
        r'\.mp4$',
        r'\.mp3$',
        r'\.jpg$',
        r'\.png$',
        r'\.pdf$',
        r'/live-blog',
        r'/liveblog',
    ]
    
    url_lower = url.lower()
    
    for pattern in skip_patterns:
        if re.search(pattern, url_lower):
            return True
    
    return False


def clean_text(text: str) -> str:
    """
    Clean extracted text.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def segment_text(text: str, segment_size: int = 100) -> List[str]:
    """
    Segment text into chunks for Belief Transformer.
    
    Args:
        text: Full article text
        segment_size: Words per segment
    
    Returns:
        List of text segments
    """
    words = text.split()
    
    segments = []
    for i in range(0, len(words), segment_size):
        segment = ' '.join(words[i:i + segment_size])
        segments.append(segment)
    
    return segments


def extract_domain(url: str) -> str:
    """
    Extract clean domain from URL.
    
    Example: https://www.reuters.com/world/... -> reuters.com
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www.
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except:
        return ""


def parse_date(date_string: Optional[str]) -> Optional[datetime]:
    """
    Parse various date formats to datetime.
    """
    if not date_string:
        return None
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %Z",  # RSS format
        "%a, %d %b %Y %H:%M:%S %z",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except:
            continue
    
    return None


class MetricsCollector:
    """
    Collect and report pipeline metrics.
    """
    
    def __init__(self):
        self.metrics = {}
        self.timings = {}
    
    def increment(self, key: str, amount: int = 1):
        """Increment a counter metric."""
        self.metrics[key] = self.metrics.get(key, 0) + amount
    
    def add_time(self, key: str, seconds: float):
        """Add timing data."""
        if key not in self.timings:
            self.timings[key] = []
        self.timings[key].append(seconds)
    
    def log_stats(self):
        """Log collected statistics."""
        logger.info("pipeline_metrics", metrics=self.metrics)
        
        # Calculate timing averages
        if self.timings:
            avg_timings = {
                key: sum(times) / len(times)
                for key, times in self.timings.items()
            }
            logger.info("pipeline_timings", avg_timings=avg_timings)
