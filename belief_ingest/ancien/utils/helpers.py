"""
Utility helpers for ingestion pipeline.
"""

import xxhash
import re
from typing import Set, Optional
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()


class URLDeduplicator:
    """
    Efficient URL deduplication using xxhash and file-backed cache.
    """
    
    def __init__(self, cache_file: Path, max_size: int = 100000):
        self.cache_file = cache_file
        self.max_size = max_size
        self.seen_hashes: Set[str] = set()
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load seen URLs from cache file."""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                for line in f:
                    url_hash = line.strip()
                    if url_hash:
                        self.seen_hashes.add(url_hash)
            
            logger.info("dedup_cache_loaded", size=len(self.seen_hashes))
        except Exception as e:
            logger.warning("dedup_cache_load_failed", error=str(e))
    
    def is_duplicate(self, url: str) -> bool:
        """Check if URL has been seen before."""
        url_hash = xxhash.xxh64(url).hexdigest()
        return url_hash in self.seen_hashes
    
    def mark_seen(self, url: str):
        """Mark URL as seen."""
        url_hash = xxhash.xxh64(url).hexdigest()
        self.seen_hashes.add(url_hash)
        
        # Periodically flush to disk
        if len(self.seen_hashes) % 1000 == 0:
            self._flush_cache()
        
        # Prevent unlimited growth
        if len(self.seen_hashes) > self.max_size:
            self._trim_cache()
    
    def _flush_cache(self):
        """Write cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                for url_hash in self.seen_hashes:
                    f.write(f"{url_hash}\n")
        except Exception as e:
            logger.warning("dedup_cache_flush_failed", error=str(e))
    
    def _trim_cache(self):
        """Trim cache to max size by removing oldest entries."""
        # Simple approach: keep most recent half
        keep_size = self.max_size // 2
        self.seen_hashes = set(list(self.seen_hashes)[-keep_size:])
        logger.info("dedup_cache_trimmed", new_size=len(self.seen_hashes))
    
    def close(self):
        """Flush and close cache."""
        self._flush_cache()


def generate_article_id(url: str, title: str, published_at: Optional[str] = None) -> str:
    """
    Generate unique article ID from URL + title + date.
    """
    # Combine fields
    parts = [url, title or ""]
    if published_at:
        parts.append(published_at)
    
    combined = "|".join(parts)
    
    # Hash to create ID
    return xxhash.xxh64(combined).hexdigest()


def should_skip_url(url: str) -> bool:
    """
    Check if URL should be skipped (non-article pages).
    """
    url_lower = url.lower()
    
    # Skip common non-article patterns
    skip_patterns = [
        '/video/', '/videos/', '/gallery/', '/galleries/',
        '/live/', '/liveblog/', '/podcast/', '/podcasts/',
        '/rss/', '/feed/', '/search/', '/tag/', '/category/',
        '/author/', '/opinion/', '/editorial/',
        '/interactive/', '/graphic/',
        'youtube.com', 'twitter.com', 'facebook.com',
        'instagram.com', 'tiktok.com'
    ]
    
    for pattern in skip_patterns:
        if pattern in url_lower:
            return True
    
    return False


def clean_text(text: str) -> str:
    """
    Clean article text - remove boilerplate, extra whitespace, etc.
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common boilerplate patterns
    boilerplate = [
        r'Click here to.*',
        r'Subscribe to.*',
        r'Sign up for.*',
        r'Read more:.*',
        r'Related articles:.*',
        r'Advertisement\s*',
        r'\[.*?\]',  # Remove [brackets]
    ]
    
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Trim
    text = text.strip()
    
    return text


def segment_text(text: str, segment_size: int = 100) -> list:
    """
    Segment text into ~100 word chunks for Belief Transformer.
    
    Args:
        text: Full article text
        segment_size: Target words per segment
    
    Returns:
        List of text segments
    """
    if not text:
        return []
    
    words = text.split()
    segments = []
    
    for i in range(0, len(words), segment_size):
        segment = ' '.join(words[i:i + segment_size])
        segments.append(segment)
    
    return segments


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www.
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except:
        return ""


def parse_date(date_str: str) -> Optional[str]:
    """
    Parse various date formats into ISO format.
    """
    if not date_str:
        return None
    
    # Common date formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.isoformat()
        except:
            continue
    
    return None


class MetricsCollector:
    """
    Simple metrics collector for monitoring pipeline performance.
    """
    
    def __init__(self):
        self.counters = {}
        self.timers = {}
    
    def increment(self, metric: str, amount: int = 1):
        """Increment a counter."""
        self.counters[metric] = self.counters.get(metric, 0) + amount
    
    def add_time(self, metric: str, duration: float):
        """Add a timing measurement."""
        if metric not in self.timers:
            self.timers[metric] = []
        self.timers[metric].append(duration)
    
    def get_stats(self) -> dict:
        """Get statistics summary."""
        stats = {
            "counters": self.counters.copy()
        }
        
        # Calculate timer statistics
        for metric, times in self.timers.items():
            if times:
                stats[f"{metric}_avg"] = sum(times) / len(times)
                stats[f"{metric}_total"] = sum(times)
        
        return stats
    
    def log_stats(self):
        """Log current statistics."""
        stats = self.get_stats()
        
        logger.info("pipeline_metrics", **stats)
        
        # Calculate derived metrics
        counters = stats.get("counters", {})
        
        fetched = counters.get("articles_fetched", 0)
        success = counters.get("articles_success", 0)
        
        if fetched > 0:
            success_rate = success / fetched
            logger.info("success_rate", rate=success_rate)
        
        # Extraction method breakdown
        extraction_methods = {
            k: v for k, v in counters.items()
            if k.startswith("extraction_")
        }
        
        if extraction_methods:
            logger.info("extraction_methods", **extraction_methods)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def estimate_processing_time(num_articles: int, articles_per_second: float = 3.0) -> str:
    """Estimate processing time for given number of articles."""
    seconds = num_articles / articles_per_second
    
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    else:
        return f"{seconds / 3600:.1f} hours"
