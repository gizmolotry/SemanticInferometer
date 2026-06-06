"""
GDELT recent fetcher for RECENT mode.
This is a placeholder - for HISTORICAL mode, use gdelt_historical_fetcher.py instead.
"""

import asyncio
from typing import List, Dict
import structlog

logger = structlog.get_logger()


async def fetch_gdelt_articles(lookback_hours: int) -> List[Dict]:
    """
    Fetch recent articles from GDELT for recent mode.
    
    This is a placeholder. For historical mode, use the GDELT historical fetcher.
    
    Args:
        lookback_hours: Hours to look back
    
    Returns:
        List of article metadata dicts
    """
    logger.warning("gdelt_recent_fetcher_placeholder",
                  message="Recent GDELT fetcher not implemented - use historical mode instead")
    
    # Return empty list - historical mode is the primary mode
    return []


if __name__ == "__main__":
    print("GDELT Recent Fetcher - Placeholder")
    print("For historical Gaza/Israel articles, use:")
    print("  python -m ingest --historical --months 12")
