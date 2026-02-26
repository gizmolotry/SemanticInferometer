"""
RSS feed fetcher for RECENT mode.
This is a placeholder - for HISTORICAL mode, use gdelt_historical_fetcher.py instead.
"""

import asyncio
from typing import List, Dict
import structlog

logger = structlog.get_logger()


async def fetch_rss_articles(lookback_hours: int) -> List[Dict]:
    """
    Fetch articles from RSS feeds for recent mode.
    
    This is a placeholder. For historical mode, use the GDELT historical fetcher.
    
    Args:
        lookback_hours: Hours to look back
    
    Returns:
        List of article metadata dicts
    """
    logger.warning("rss_fetcher_placeholder",
                  message="RSS fetcher not implemented - use historical mode instead")
    
    # Return empty list - historical mode doesn't need RSS
    return []


if __name__ == "__main__":
    print("RSS Fetcher - Placeholder")
    print("For historical Gaza/Israel articles, use:")
    print("  python -m ingest --historical --months 12")
