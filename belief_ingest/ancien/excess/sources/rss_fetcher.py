"""
RSS feed fetching with async concurrency and retry logic.
"""

import asyncio
import feedparser
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import structlog
from aiohttp_retry import RetryClient, ExponentialRetry

from config import (
    RSS_FEEDS, MAX_CONCURRENT_FEEDS, REQUEST_TIMEOUT, 
    RETRY_ATTEMPTS, USER_AGENT
)
from utils.helpers import normalize_url, extract_domain, parse_date

logger = structlog.get_logger()


class RSSFetcher:
    """Async RSS feed fetcher with robust error handling."""
    
    def __init__(self):
        self.feeds = RSS_FEEDS
        self.session: Optional[aiohttp.ClientSession] = None
        self.retry_options = ExponentialRetry(attempts=RETRY_ATTEMPTS)
    
    async def __aenter__(self):
        """Setup async context."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async context."""
        if self.session:
            await self.session.close()
    
    async def fetch_all_feeds(self, lookback_hours: int = 24) -> List[Dict]:
        """
        Fetch all RSS feeds concurrently.
        
        Args:
            lookback_hours: Only return articles published within this window
        
        Returns:
            List of article metadata dicts
        """
        logger.info("fetching_rss_feeds", feed_count=len(self.feeds))
        
        # Create tasks for all feeds
        tasks = [
            self._fetch_feed(feed_name, feed_url, lookback_hours)
            for feed_name, feed_url in self.feeds.items()
        ]
        
        # Run with concurrency limit
        articles = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FEEDS)
        
        async def limited_fetch(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[limited_fetch(task) for task in tasks],
            return_exceptions=True
        )
        
        # Collect successful results
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                logger.warning("feed_fetch_exception", error=str(result))
        
        logger.info("rss_fetch_complete", 
                   total_articles=len(articles),
                   unique_urls=len(set(a['url'] for a in articles)))
        
        return articles
    
    async def _fetch_feed(self, feed_name: str, feed_url: str, 
                         lookback_hours: int) -> List[Dict]:
        """
        Fetch single RSS feed with retry logic.
        
        Returns:
            List of article metadata from this feed
        """
        try:
            logger.debug("fetching_feed", name=feed_name, url=feed_url)
            
            retry_client = RetryClient(
                client_session=self.session,
                retry_options=self.retry_options
            )
            
            async with retry_client.get(feed_url) as response:
                if response.status != 200:
                    logger.warning("feed_fetch_failed", 
                                 name=feed_name,
                                 status=response.status)
                    return []
                
                content = await response.text()
                
                # Parse RSS/Atom feed
                feed = feedparser.parse(content)
                
                if feed.bozo:
                    logger.warning("feed_parse_error",
                                 name=feed_name,
                                 error=str(feed.bozo_exception))
                
                # Extract articles
                articles = self._parse_feed_entries(
                    feed, feed_name, lookback_hours
                )
                
                logger.info("feed_fetched",
                          name=feed_name,
                          article_count=len(articles))
                
                return articles
                
        except asyncio.TimeoutError:
            logger.warning("feed_timeout", name=feed_name, url=feed_url)
            return []
        except Exception as e:
            logger.warning("feed_error",
                         name=feed_name,
                         url=feed_url,
                         error=str(e))
            return []
    
    def _parse_feed_entries(self, feed, feed_name: str, 
                           lookback_hours: int) -> List[Dict]:
        """
        Extract article metadata from feed entries.
        
        Returns:
            List of normalized article metadata
        """
        articles = []
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        for entry in feed.entries:
            try:
                # Extract URL
                url = entry.get('link') or entry.get('id')
                if not url:
                    continue
                
                url = normalize_url(url)
                
                # Extract publish date
                published_at = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        pub_dt = datetime(*entry.published_parsed[:6])
                        published_at = pub_dt.isoformat()
                        
                        # Skip old articles
                        if pub_dt < cutoff_time:
                            continue
                    except Exception:
                        pass
                
                # Extract metadata
                article = {
                    'url': url,
                    'title': entry.get('title', '').strip(),
                    'summary': entry.get('summary', '').strip(),
                    'published_at': published_at,
                    'author': self._extract_author(entry),
                    'publisher': extract_domain(url),
                    'section': feed_name.split('_')[0] if '_' in feed_name else None,
                    'source_type': 'rss',
                    'feed_name': feed_name,
                    'fetched_at': datetime.now().isoformat(),
                }
                
                # Extract categories/tags
                if hasattr(entry, 'tags'):
                    article['tags'] = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
                
                articles.append(article)
                
            except Exception as e:
                logger.debug("entry_parse_error", error=str(e))
                continue
        
        return articles
    
    @staticmethod
    def _extract_author(entry) -> Optional[str]:
        """Extract author from feed entry."""
        # Try different author fields
        if hasattr(entry, 'author') and entry.author:
            return entry.author.strip()
        
        if hasattr(entry, 'authors') and entry.authors:
            return ', '.join(a.get('name', '').strip() for a in entry.authors if a.get('name'))
        
        if hasattr(entry, 'author_detail') and entry.author_detail:
            return entry.author_detail.get('name', '').strip()
        
        return None


async def fetch_rss_articles(lookback_hours: int = 24) -> List[Dict]:
    """
    Convenience function to fetch all RSS articles.
    
    Args:
        lookback_hours: Only return articles from last N hours
    
    Returns:
        List of article metadata
    """
    async with RSSFetcher() as fetcher:
        return await fetcher.fetch_all_feeds(lookback_hours)


# For testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    async def test():
        articles = await fetch_rss_articles(lookback_hours=48)
        print(f"\nFetched {len(articles)} articles")
        
        if articles:
            print("\nSample article:")
            import json
            print(json.dumps(articles[0], indent=2))
    
    asyncio.run(test())
