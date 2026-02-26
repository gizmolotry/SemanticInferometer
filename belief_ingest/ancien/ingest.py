"""
Main ingestion orchestrator with HISTORICAL and RECENT modes.

HISTORICAL MODE (GDELT):
    python -m ingest --historical --months 12
    python -m ingest --historical --start 2024-01-01 --end 2024-12-31

RECENT MODE (RSS + GDELT):
    python -m ingest --hours 24
    python -m ingest --hours 48 --max 1000

Or as a module:
    from ingest import run_ingestion
    await run_ingestion(lookback_hours=24)
"""

import asyncio
import argparse
import jsonlines
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import structlog

from config import (
    PROCESSED_DIR, BATCH_SIZE, MAX_CONCURRENT_ARTICLES,
    LOG_LEVEL, METRICS_ENABLED
)

# Import fetchers
try:
    from sources.rss_fetcher import fetch_rss_articles
    from sources.gdelt_fetcher import fetch_gdelt_articles
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("rss_fetcher_not_available")

# Import historical fetcher
try:
    from gdelt_historical_fetcher import fetch_historical_articles
    HISTORICAL_AVAILABLE = True
except ImportError:
    HISTORICAL_AVAILABLE = False

from content_extractor import ContentExtractor
from utils.helpers import (
    URLDeduplicator, generate_article_id, should_skip_url, MetricsCollector
)

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


class IngestionPipeline:
    """
    Main ingestion pipeline orchestrator.
    
    Supports two modes:
    1. HISTORICAL: Fetch from GDELT archives (12+ months back)
    2. RECENT: Fetch from RSS + recent GDELT (last 24-48 hours)
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.deduplicator = URLDeduplicator(
            cache_file=Path("data/cache/seen_urls.txt"),
            max_size=100000
        )
        
        self.metrics = MetricsCollector()
        self.extractor: ContentExtractor = None
    
    async def run_historical(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        months: float = None,
        max_articles: int = None,
        lenient: bool = False
    ) -> List[Dict]:
        """
        Run HISTORICAL mode - fetch from GDELT archives.
        
        Args:
            start_date: Start date
            end_date: End date (default: today)
            months: Alternative to start_date (e.g., months=0.25 for 1 week, months=12 for 12 months)
            max_articles: Limit total articles
            lenient: Use lenient filtering (more recall, less precision)
        
        Returns:
            List of processed articles
        """
        if not HISTORICAL_AVAILABLE:
            logger.error("historical_fetcher_not_available")
            raise ImportError("gdelt_historical_fetcher not found")
        
        start_time = datetime.now()
        
        # Calculate dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None and months:
            start_date = end_date - timedelta(days=int(months * 30))
        
        logger.info("historical_mode_start",
                   start_date=start_date.isoformat() if start_date else None,
                   end_date=end_date.isoformat(),
                   months=months,
                   lenient=lenient)
        
        # Step 1: Fetch from GDELT
        article_metas = await fetch_historical_articles(
            start_date=start_date,
            end_date=end_date,
            months=months,
            max_articles=max_articles,
            lenient=lenient
        )
        
        logger.info("gdelt_fetch_complete", total_urls=len(article_metas))
        
        # Step 2: Deduplicate
        article_metas = self._deduplicate(article_metas)
        
        # Step 3: Extract content
        articles = await self._extract_all_content(article_metas)
        
        # Step 4: Write output
        date_range = f"{start_date.strftime('%Y%m') if start_date else ''}-{end_date.strftime('%Y%m')}"
        output_path = await self._write_output(articles, prefix=f"historical_{date_range}")
        
        # Step 5: Report
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("historical_mode_complete",
                   duration_seconds=duration,
                   articles_processed=len(articles),
                   output_file=str(output_path))
        
        self.metrics.log_stats()
        
        return articles
    
    async def run_recent(
        self,
        lookback_hours: int = 24,
        max_articles: int = None
    ) -> List[Dict]:
        """
        Run RECENT mode - fetch from RSS + recent GDELT.
        
        Args:
            lookback_hours: How far back to fetch
            max_articles: Limit total articles
        
        Returns:
            List of processed articles
        """
        start_time = datetime.now()
        
        logger.info("recent_mode_start",
                   lookback_hours=lookback_hours,
                   max_articles=max_articles)
        
        # Step 1: Fetch URLs from all sources
        article_metas = await self._fetch_all_sources(lookback_hours)
        
        # Step 2: Deduplicate
        article_metas = self._deduplicate(article_metas)
        
        # Limit if requested
        if max_articles:
            article_metas = article_metas[:max_articles]
        
        logger.info("urls_collected",
                   total=len(article_metas),
                   sources={
                       'rss': sum(1 for a in article_metas if a.get('source_type') == 'rss'),
                       'gdelt': sum(1 for a in article_metas if a.get('source_type') == 'gdelt'),
                   })
        
        # Step 3: Extract content
        articles = await self._extract_all_content(article_metas)
        
        # Step 4: Write output
        output_path = await self._write_output(articles)
        
        # Step 5: Report
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("recent_mode_complete",
                   duration_seconds=duration,
                   articles_processed=len(articles),
                   articles_per_second=len(articles) / duration if duration > 0 else 0,
                   output_file=str(output_path))
        
        self.metrics.log_stats()
        
        return articles
    
    async def _fetch_all_sources(self, lookback_hours: int) -> List[Dict]:
        """Fetch article metadata from RSS + recent GDELT."""
        if not RSS_AVAILABLE:
            logger.warning("rss_not_available_skipping")
            return []
        
        logger.info("fetching_sources")
        
        # Launch RSS and GDELT fetches in parallel
        tasks = [fetch_rss_articles(lookback_hours)]
        
        # Add recent GDELT if available
        try:
            tasks.append(fetch_gdelt_articles(lookback_hours))
        except:
            pass
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_articles.extend(result)
                source = "rss" if i == 0 else "gdelt"
                logger.info(f"{source}_articles_fetched", count=len(result))
            else:
                logger.error("fetch_failed", error=str(result))
        
        return all_articles
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate and unwanted URLs."""
        unique_articles = []
        
        for article in articles:
            url = article.get('url')
            if not url:
                continue
            
            # Skip non-article URLs
            if should_skip_url(url):
                self.metrics.increment('urls_skipped')
                continue
            
            # Check deduplication
            if self.deduplicator.is_duplicate(url):
                self.metrics.increment('urls_duplicate')
                continue
            
            self.deduplicator.mark_seen(url)
            unique_articles.append(article)
        
        logger.info("deduplication_complete",
                   input=len(articles),
                   output=len(unique_articles),
                   duplicates=len(articles) - len(unique_articles))
        
        return unique_articles
    
    async def _extract_all_content(self, article_metas: List[Dict]) -> List[Dict]:
        """Extract full content from all articles."""
        logger.info("extracting_content", total=len(article_metas))
        
        articles = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_ARTICLES)
        
        async with ContentExtractor() as extractor:
            self.extractor = extractor
            
            async def extract_with_limit(meta):
                async with semaphore:
                    return await self._extract_single(meta)
            
            # Process in batches
            for i in range(0, len(article_metas), BATCH_SIZE):
                batch = article_metas[i:i + BATCH_SIZE]
                
                tasks = [extract_with_limit(meta) for meta in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect successful extractions
                for result in results:
                    if isinstance(result, dict):
                        articles.append(result)
                    elif isinstance(result, Exception):
                        logger.debug("extraction_exception", error=str(result))
                
                # Log progress
                logger.info("batch_complete",
                           batch_num=i // BATCH_SIZE + 1,
                           processed=min(i + BATCH_SIZE, len(article_metas)),
                           total=len(article_metas),
                           success_count=len(articles))
        
        logger.info("extraction_complete",
                   success=len(articles),
                   failed=len(article_metas) - len(articles),
                   success_rate=len(articles) / len(article_metas) if article_metas else 0)
        
        return articles
    
    async def _extract_single(self, meta: Dict) -> Optional[Dict]:
        """Extract single article and track metrics."""
        import time
        start = time.time()
        
        try:
            self.metrics.increment('articles_fetched')
            
            article = await self.extractor.extract(meta)
            
            if article:
                # Generate article ID
                article['id'] = generate_article_id(
                    article['url'],
                    article['title'],
                    article.get('published_at')
                )
                
                # Track extraction method
                method = article.get('extraction_method', 'unknown')
                self.metrics.increment(f'extraction_{method}')
                
                self.metrics.increment('articles_success')
                
                elapsed = time.time() - start
                self.metrics.add_time('total_extraction_time', elapsed)
                
                return article
            else:
                self.metrics.increment('articles_failed')
                return None
                
        except Exception as e:
            logger.warning("extraction_error",
                         url=meta.get('url'),
                         error=str(e))
            self.metrics.increment('articles_failed')
            return None
    
    async def _write_output(self, articles: List[Dict], prefix: str = "articles") -> Path:
        """Write articles to JSONL output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{prefix}_{timestamp}.jsonl"
        
        try:
            with jsonlines.open(output_path, mode='w') as writer:
                for article in articles:
                    writer.write(article)
            
            logger.info("output_written",
                       path=str(output_path),
                       articles=len(articles))
            
            return output_path
            
        except Exception as e:
            logger.error("output_write_failed", error=str(e))
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        self.deduplicator.close()


async def run_ingestion(
    # Historical mode parameters
    historical: bool = False,
    start_date: datetime = None,
    end_date: datetime = None,
    months: float = None,
    lenient: bool = False,
    
    # Recent mode parameters
    lookback_hours: int = 24,
    
    # Common parameters
    max_articles: int = None,
    output_dir: Path = None
) -> List[Dict]:
    """
    Main entry point for ingestion pipeline.
    
    Args:
        historical: Use historical mode (GDELT archives)
        start_date: Historical mode - start date
        end_date: Historical mode - end date
        months: Historical mode - number of months back (can be decimal)
        lenient: Historical mode - use lenient filtering
        lookback_hours: Recent mode - hours to look back
        max_articles: Limit total articles
        output_dir: Output directory
    
    Returns:
        List of processed articles
    """
    pipeline = IngestionPipeline(output_dir)
    
    try:
        if historical:
            # HISTORICAL MODE
            articles = await pipeline.run_historical(
                start_date=start_date,
                end_date=end_date,
                months=months,
                max_articles=max_articles,
                lenient=lenient
            )
        else:
            # RECENT MODE
            articles = await pipeline.run_recent(
                lookback_hours=lookback_hours,
                max_articles=max_articles
            )
        
        return articles
    finally:
        pipeline.cleanup()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Belief Transformer Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recent mode (last 24 hours)
  python -m ingest --hours 24
  
  # Historical mode (last 12 months)
  python -m ingest --historical --months 12
  
  # Historical mode (specific date range)
  python -m ingest --historical --start 2024-01-01 --end 2024-12-31
  
  # Lenient filtering (more articles)
  python -m ingest --historical --months 6 --lenient
  
  # Limit articles
  python -m ingest --hours 48 --max 1000
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--historical",
        action="store_true",
        help="Use historical mode (fetch from GDELT archives)"
    )
    mode_group.add_argument(
        "--hours",
        type=int,
        help="Recent mode: hours to look back (default: 24)"
    )
    
    # Historical mode parameters
    hist_group = parser.add_argument_group("Historical mode options")
    hist_group.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    hist_group.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today)"
    )
    hist_group.add_argument(
        "--months",
        type=float,
        help="Number of months back from end date (can be decimal, e.g., 0.25 for 1 week)"
    )
    hist_group.add_argument(
        "--lenient",
        action="store_true",
        help="Use lenient filtering (more recall, less precision)"
    )
    
    # Common parameters
    parser.add_argument(
        "--max",
        type=int,
        help="Maximum articles to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start:
        try:
            start_date = datetime.strptime(args.start, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid start date format. Use YYYY-MM-DD")
            return
    
    if args.end:
        try:
            end_date = datetime.strptime(args.end, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid end date format. Use YYYY-MM-DD")
            return
    
    # Determine mode
    if args.historical:
        mode = "historical"
        lookback_hours = None
    elif args.hours:
        mode = "recent"
        lookback_hours = args.hours
    else:
        # Default to recent mode with 24 hours
        mode = "recent"
        lookback_hours = 24
    
    # Set log level
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run pipeline
    output_dir = Path(args.output) if args.output else None
    
    print(f"\n🚀 Starting {mode.upper()} mode...")
    
    articles = asyncio.run(
        run_ingestion(
            historical=(mode == "historical"),
            start_date=start_date,
            end_date=end_date,
            months=args.months,
            lenient=args.lenient,
            lookback_hours=lookback_hours,
            max_articles=args.max,
            output_dir=output_dir
        )
    )
    
    print(f"\n✓ Successfully processed {len(articles)} articles")
    
    if mode == "historical":
        print(f"\n📊 Historical Summary:")
        if start_date and end_date:
            print(f"  Date range: {start_date.date()} to {end_date.date()}")
        elif args.months:
            print(f"  Period: Last {args.months} months")
        print(f"  Filtering: {'Lenient' if args.lenient else 'Strict'}")
    else:
        print(f"\n📊 Recent Summary:")
        print(f"  Lookback: {lookback_hours} hours")
    
    print(f"  Articles: {len(articles)}")


if __name__ == "__main__":
    main()
