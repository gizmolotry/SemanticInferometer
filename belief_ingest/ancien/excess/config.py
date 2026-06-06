"""
Production configuration for Belief Transformer ingestion pipeline.
Edit this file to add/remove sources, tune performance, etc.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TOPIC FILTERING (🎯 Gaza/Israel)
# ============================================================================
TOPIC_FILTER_ENABLED = True
TOPIC_KEYWORDS = [
    # Primary keywords
    'gaza', 'israel', 'israeli', 'hamas', 'palestinian',
    
    # Leaders & organizations
    'netanyahu', 'idf', 'hezbollah', 'fatah',
    
    # Locations
    'tel aviv', 'jerusalem', 'west bank', 'rafah', 'khan younis',
    'gaza strip', 'gaza city', 'lebanon border',
    
    # Conflict terms
    'hostage', 'ceasefire', 'military operation', 'rocket attack',
    'airstrike', 'tunnel', 'border crossing',
    
    # Regional context
    'middle east conflict', 'iran proxy', 'abraham accords'
]

# ============================================================================
# PERFORMANCE
# ============================================================================
MAX_CONCURRENT_FEEDS = 50  # Parallel RSS feed fetches
MAX_CONCURRENT_ARTICLES = 100  # Parallel article extractions
MAX_CONCURRENT_PLAYWRIGHT = 5  # Expensive, keep low
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
BATCH_SIZE = 500  # Articles per output file

# ============================================================================
# DEDUPLICATION
# ============================================================================
DEDUP_CACHE_SIZE = 100000  # URLs to remember
DEDUP_CACHE_FILE = CACHE_DIR / "seen_urls.txt"
SIMILARITY_THRESHOLD = 0.95  # For near-duplicate detection

# ============================================================================
# EXTRACTION
# ============================================================================
MIN_ARTICLE_LENGTH = 200  # characters
MAX_ARTICLE_LENGTH = 50000  # characters
SEGMENT_SIZE = 100  # words per snippet (for Belief Transformer)

# Extraction strategy priorities
EXTRACTION_STRATEGIES = [
    "trafilatura",  # Fast, accurate
    "newspaper3k",  # Good fallback
    "readability",  # Another fallback
    "playwright"    # Last resort (slow)
]

# When to use Playwright (expensive)
PLAYWRIGHT_DOMAINS = [
    "ft.com",  # Paywalled
    "economist.com",
    "wsj.com",
    "bloomberg.com",
    "nytimes.com"  # Heavy JS
]

# ============================================================================
# RSS FEEDS (Curated for Gaza/Israel coverage)
# ============================================================================

RSS_FEEDS = {
    # === Middle East Focused ===
    "timesofisrael": "https://www.timesofisrael.com/feed/",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "aljazeera_middleeast": "https://www.aljazeera.com/xml/rss/middle-east.xml",
    
    # === Major Outlets - World News (Will have Gaza coverage) ===
    "reuters_world": "https://www.reuters.com/rssfeed/world",
    "ap_topnews": "https://rss.ap.org/",
    "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_middleeast": "http://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "guardian_world": "https://www.theguardian.com/world/rss",
    "nytimes_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    
    # === US Mainstream ===
    "npr_news": "https://feeds.npr.org/1001/rss.xml",
    "pbs_news": "https://www.pbs.org/newshour/feeds/rss/headlines",
    "cnn_world": "http://rss.cnn.com/rss/cnn_world.rss",
    
    # === US Left-Leaning ===
    "wapo_world": "https://feeds.washingtonpost.com/rss/world",
    "msnbc_news": "https://www.msnbc.com/feeds/latest",
    "huffpost_politics": "https://www.huffpost.com/section/politics/feed",
    "vox_world": "https://www.vox.com/rss/world/index.xml",
    
    # === US Right-Leaning ===
    "foxnews_world": "https://moxie.foxnews.com/google-publisher/world.xml",
    "nypost_news": "https://nypost.com/news/feed/",
    "washingtonexaminer": "https://www.washingtonexaminer.com/feed",
    "thehill_news": "https://thehill.com/news/feed/",
    
    # === UK ===
    "telegraph_news": "https://www.telegraph.co.uk/news/rss.xml",
    "independent": "https://www.independent.co.uk/news/rss",
    
    # === Europe ===
    "france24_english": "https://www.france24.com/en/rss",
    "dw_topstories": "https://rss.dw.com/rdf/rss-en-all",
    "euronews": "https://www.euronews.com/rss",
    
    # === Financial (Often cover geopolitical events) ===
    "ft_world": "https://www.ft.com/world?format=rss",
    "wsj_world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "economist": "https://www.economist.com/international/rss.xml",
}

# ============================================================================
# GDELT INTEGRATION
# ============================================================================
GDELT_ENABLED = True
GDELT_LOOKBACK_HOURS = 24  # How far back to pull
GDELT_MAX_ARTICLES = 5000  # Per fetch
GDELT_UPDATE_INTERVAL = 900  # seconds (15 min)

# GDELT CSV endpoints (no API key needed!)
GDELT_ENDPOINTS = {
    "last15min": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
    "master": "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt",
}

# Filter GDELT by these domains (empty = all)
GDELT_DOMAIN_FILTER = []  # e.g., ["reuters.com", "bbc.com"]

# ============================================================================
# OUTPUT FORMAT (Belief Transformer Compatible)
# ============================================================================
OUTPUT_FORMAT = "jsonl"  # or "json", "parquet"
OUTPUT_SCHEMA = {
    "id": "required",  # Generated hash
    "url": "required",
    "title": "required",
    "content": "required",  # Full text
    "snippets": "required",  # Segmented for transformer
    "publisher": "required",
    "section": "optional",
    "author": "optional",
    "published_at": "required",  # ISO format
    "fetched_at": "required",
    "language": "optional",
    "country": "optional",
    "topic": "optional",
    "source_type": "required",  # "rss", "gdelt", "manual"
    "extraction_method": "required",  # "trafilatura", "playwright", etc.
    "word_count": "required",
    "char_count": "required",
}

# ============================================================================
# MONITORING
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "json"  # json or text
METRICS_ENABLED = True
METRICS_INTERVAL = 60  # seconds

# Alerts (log warnings when thresholds hit)
ALERT_THRESHOLDS = {
    "extraction_failure_rate": 0.3,  # 30% failures
    "playwright_usage_rate": 0.2,    # 20% using expensive fallback
    "avg_fetch_time": 10.0,          # seconds
}

# ============================================================================
# USER AGENT
# ============================================================================
USER_AGENT = "BeliefTransformer/1.0 (+https://github.com/yourorg/belief-transformer) Academic Research"
