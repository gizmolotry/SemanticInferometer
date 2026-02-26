"""
Main ingestion orchestrator - coordinates RSS, GDELT, extraction, and output.

Usage:
    python -m belief_ingest.ingest --hours 24 --output data/output.jsonl
    
Or as a module:
    from belief_ingest.ingest import run_ingestion
    await run_ingestion(lookback_hours=24)
"""

import asyncio
import argparse
import jsonlines
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import structlog

from config import (
    PROCESSED_DIR, BATCH_SIZE, MAX_CONCURRENT_ARTICLES,
    LOG_LEVEL, METRICS_ENABLED, METRICS_INTERVAL
)
from sources.rss_fetcher import fetch_rss_articles
from sources.gdelt_fetcher import fetch_gdelt_articles
from extractors.content_extractor import ContentExtractor
from utils.helpers import (
    URLDeduplicator, generate_article_id, should_skip_url, MetricsCollector
)

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if True else structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


class IngestionPipeline:
    """
    Main ingestion pipeline orchestrator.
    
    Workflow:
    1. Fetch URLs from RSS feeds + GDELT
    2. Deduplicate URLs
    3. Extract full content with fallback strategies
    4. Generate Belief Fingerprint format
    5. Write to output files
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
    
    async def run(self, lookback_hours: int = 24, 
                  max_articles: int = None) -> List[Dict]:
        """
        Run full ingestion pipeline.
        
        Args:
            lookback_hours: How far back to fetch articles
            max_articles: Limit total articles (None = no limit)
        
        Returns:
            List of processed articles with content
        """
        start_time = datetime.now()
        logger.info("pipeline_start", 
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
        
        # Step 3: Extract full content
        articles = await self._extract_all_content(article_metas)
        
        # Step 4: Write output
        output_path = await self._write_output(articles)
        
        # Step 5: Report metrics
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("pipeline_complete",
                   duration_seconds=duration,
                   articles_processed=len(articles),
                   articles_per_second=len(articles) / duration if duration > 0 else 0,
                   output_file=str(output_path))
        
        self.metrics.log_stats()
        
        return articles
    
    async def _fetch_all_sources(self, lookback_hours: int) -> List[Dict]:
        """Fetch article metadata from all sources concurrently."""
        logger.info("fetching_sources")
        
        # Launch RSS and GDELT fetches in parallel
        rss_task = fetch_rss_articles(lookback_hours)
        gdelt_task = fetch_gdelt_articles(lookback_hours)
        
        results = await asyncio.gather(
            rss_task,
            gdelt_task,
            return_exceptions=True
        )
        
        # Collect results
        all_articles = []
        
        if isinstance(results[0], list):
            all_articles.extend(results[0])
            logger.info("rss_articles_fetched", count=len(results[0]))
        else:
            logger.error("rss_fetch_failed", error=str(results[0]))
        
        if isinstance(results[1], list):
            all_articles.extend(results[1])
            logger.info("gdelt_articles_fetched", count=len(results[1]))
        else:
            logger.error("gdelt_fetch_failed", error=str(results[1]))
        
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
        """Extract full content from all articles with concurrency control."""
        logger.info("extracting_content", total=len(article_metas))
        
        articles = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_ARTICLES)
        
        async with ContentExtractor() as extractor:
            self.extractor = extractor
            
            async def extract_with_limit(meta):
                async with semaphore:
                    return await self._extract_single(meta)
            
            # Process in batches for progress tracking
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
    
    async def _extract_single(self, meta: Dict) -> Dict:
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
                self.metrics.increment('extraction_failed')
                return None
                
        except Exception as e:
            logger.warning("extraction_error", 
                         url=meta.get('url'),
                         error=str(e))
            self.metrics.increment('articles_failed')
            return None
    
    async def _write_output(self, articles: List[Dict]) -> Path:
        """Write articles to output file in JSONL format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"articles_{timestamp}.jsonl"
        
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


async def run_ingestion(lookback_hours: int = 24, 
                       max_articles: int = None,
                       output_dir: Path = None) -> List[Dict]:
    """
    Main entry point for ingestion pipeline.
    
    Args:
        lookback_hours: How far back to fetch articles
        max_articles: Limit total articles (None = no limit)
        output_dir: Where to write output (default: config.PROCESSED_DIR)
    
    Returns:
        List of processed articles
    """
    pipeline = IngestionPipeline(output_dir)
    
    try:
        articles = await pipeline.run(
            lookback_hours=lookback_hours,
            max_articles=max_articles
        )
        return articles
    finally:
        pipeline.cleanup()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Belief Transformer Ingestion Pipeline"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours to look back for articles (default: 24)"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum articles to process (default: unlimited)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run pipeline
    output_dir = Path(args.output) if args.output else None
    
    articles = asyncio.run(
        run_ingestion(
            lookback_hours=args.hours,
            max_articles=args.max,
            output_dir=output_dir
        )
    )
    
    print(f"\n✓ Successfully processed {len(articles)} articles")


if __name__ == "__main__":
    main()
