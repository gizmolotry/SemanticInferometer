"""
Production configuration for Belief Transformer ingestion pipeline.
Now includes HISTORICAL MODE for GDELT archives.
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
# HISTORICAL MODE (GDELT Archives)
# ============================================================================
HISTORICAL_ENABLED = True

# GDELT filtering modes
HISTORICAL_STRICT_MODE = True  # True = theme AND (location OR URL)
                               # False = theme OR location OR URL

# GDELT themes to filter by
GDELT_THEMES = {
    # Conflict
    'ARMED_CONFLICT', 'TERROR', 'MILITARY', 'MILITARY_ACTION',
    'WAR', 'ATTACK', 'VIOLENCE', 'CEASEFIRE',
    
    # Humanitarian
    'HUMANITARIAN', 'REFUGEE', 'CASUALTIES', 'HUMAN_RIGHTS',
    
    # Political
    'PEACE_TALKS', 'DIPLOMACY', 'SANCTIONS', 'PROTEST',
    
    # Regional
    'MIDDLE_EAST', 'ARAB_ISRAELI_CONFLICT'
}

# GDELT locations to filter by
GDELT_LOCATIONS = {
    'gaza', 'gaza strip', 'gaza city', 'israel', 'israeli',
    'tel aviv', 'jerusalem', 'west bank', 'rafah', 'khan younis',
    'palestine', 'palestinian'
}

# URL keywords for GDELT filtering
GDELT_URL_KEYWORDS = {
    'gaza', 'israel', 'hamas', 'palestinian', 'netanyahu',
    'idf', 'middle-east', 'mideast'
}

# Historical fetch defaults
HISTORICAL_DEFAULT_MONTHS = 12
HISTORICAL_MAX_ARTICLES = 50000
HISTORICAL_CACHE_ENABLED = True

# ============================================================================
# PERFORMANCE
# ============================================================================
MAX_CONCURRENT_FEEDS = 50  # Parallel RSS feed fetches
MAX_CONCURRENT_ARTICLES = 100  # Parallel article extractions
MAX_CONCURRENT_PLAYWRIGHT = 5  # Expensive, keep low
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
BATCH_SIZE = 500  # Articles per batch during extraction

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
    "nytimes.com",
    "timesofisrael.com"  # Heavy JS
]

# ============================================================================
# RSS FEEDS (For RECENT mode)
# ============================================================================

RSS_FEEDS = {
    # === Middle East Focused ===
    "timesofisrael": "https://www.timesofisrael.com/feed/",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "aljazeera_middleeast": "https://www.aljazeera.com/xml/rss/middle-east.xml",
    
    # === Major Outlets - World News ===
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
    
    # === Ireland (for acquisition studies) ===
    "irishtimes": "https://www.irishtimes.com/cmlink/news-1.1319192",
    "independent_ie": "https://www.independent.ie/rss/",
    "rte_news": "https://www.rte.ie/rss/news.xml",
    
    # === Financial ===
    "ft_world": "https://www.ft.com/world?format=rss",
    "wsj_world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "economist": "https://www.economist.com/international/rss.xml",
    
    # === Additional International ===
    "politico_eu": "https://www.politico.eu/feed/",
    "axios_world": "https://www.axios.com/feeds/world.rss",
}

# ============================================================================
# GDELT RECENT MODE (For RSS + recent GDELT)
# ============================================================================
GDELT_ENABLED = True
GDELT_LOOKBACK_HOURS = 24
GDELT_MAX_ARTICLES = 5000
GDELT_UPDATE_INTERVAL = 900  # 15 minutes

# GDELT CSV endpoints
GDELT_ENDPOINTS = {
    "last15min": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
    "master": "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt",
}

# Filter by domains (empty = all)
GDELT_DOMAIN_FILTER = []

# ============================================================================
# OUTPUT FORMAT (Belief Transformer Compatible)
# ============================================================================
OUTPUT_FORMAT = "jsonl"
OUTPUT_SCHEMA = {
    "id": "required",
    "url": "required",
    "title": "required",
    "content": "required",
    "snippets": "required",
    "publisher": "required",
    "section": "optional",
    "author": "optional",
    "published_at": "required",
    "fetched_at": "required",
    "source_type": "required",
    "extraction_method": "required",
    "word_count": "required",
    "char_count": "required",
    
    # GDELT-specific fields (for historical mode)
    "themes": "optional",
    "locations": "optional",
    "sentiment_score": "optional",
}

# ============================================================================
# MONITORING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "json"
METRICS_ENABLED = True
METRICS_INTERVAL = 60  # seconds

ALERT_THRESHOLDS = {
    "extraction_failure_rate": 0.3,
    "playwright_usage_rate": 0.2,
    "avg_fetch_time": 10.0,
}

# ============================================================================
# USER AGENT
# ============================================================================
USER_AGENT = "BeliefTransformer/1.0 (+https://github.com/yourorg/belief-transformer) Academic Research"

# ============================================================================
# ACQUISITION STUDY PRESETS
# ============================================================================
# For your validation studies
ACQUISITION_TARGETS = {
    "irish_independent": {
        "name": "Irish Independent → Mediahuis",
        "acquisition_date": "2019-07-01",
        "pre_start": "2018-01-01",
        "pre_end": "2019-06-30",
        "post_start": "2019-07-01",
        "post_end": "2020-12-31",
    },
    "politico": {
        "name": "POLITICO → Axel Springer",
        "acquisition_date": "2021-01-01",
        "pre_start": "2019-07-01",
        "pre_end": "2020-12-31",
        "post_start": "2021-01-01",
        "post_end": "2022-06-30",
    },
    "financial_times": {
        "name": "Financial Times → Nikkei",
        "acquisition_date": "2015-07-01",
        "pre_start": "2014-01-01",
        "pre_end": "2015-06-30",
        "post_start": "2015-07-01",
        "post_end": "2016-12-31",
    },
}