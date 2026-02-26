#!/usr/bin/env python3
"""
PRODUCTION 12-MONTH SCRAPER FOR BELIEF TRANSFORMER

This script is optimized for scraping 12 months of Gaza/Israel news coverage from GDELT.

Features:
- Automatic checkpoint/resume system
- Progress tracking with estimated time remaining
- Batch processing to avoid memory issues
- Automatic retry on failures
- Detailed logging

Usage:
    python scrape_12_months.py
    
    # Custom date range:
    python scrape_12_months.py --start 2024-01-01 --end 2024-12-31
    
    # Resume from checkpoint:
    python scrape_12_months.py --resume
"""

import asyncio
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import structlog
import time

from gdelt_historical_fetcher import GDELTHistoricalFetcher
from content_extractor import ContentExtractor
from utils.helpers import URLDeduplicator, generate_article_id, should_skip_url
from config import PROCESSED_DIR

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


class CheckpointManager:
    """Manage scraping checkpoints for resumability."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, state: dict):
        """Save checkpoint state."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("checkpoint_saved", date=state.get('current_date'))
    
    def load(self) -> dict:
        """Load checkpoint state if it exists."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            logger.info("checkpoint_loaded", date=state.get('current_date'))
            return state
        except Exception as e:
            logger.error("checkpoint_load_failed", error=str(e))
            return None
    
    def clear(self):
        """Clear checkpoint."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        logger.info("checkpoint_cleared")


class ProductionScraper:
    """
    Production-grade 12-month scraper with checkpointing and progress tracking.
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        output_dir: Path = None,
        batch_days: int = 7,
        checkpoint_file: Path = None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir or PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_days = batch_days
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_file or Path("data/cache/checkpoint.json")
        )
        
        self.deduplicator = URLDeduplicator(
            cache_file=Path("data/cache/seen_urls_12mo.txt"),
            max_size=200000  # Larger cache for 12 months
        )
        
        self.stats = {
            'articles_fetched': 0,
            'articles_extracted': 0,
            'articles_failed': 0,
            'batches_complete': 0,
            'total_batches': 0,
        }
        
        self.start_time = None
    
    async def run(self, resume: bool = False):
        """
        Run the 12-month scrape.
        
        Args:
            resume: If True, resume from checkpoint
        """
        self.start_time = time.time()
        
        # Check for checkpoint
        checkpoint = None
        if resume:
            checkpoint = self.checkpoint_manager.load()
        
        # Determine starting point
        if checkpoint:
            current_date = datetime.fromisoformat(checkpoint['current_date'])
            self.stats = checkpoint.get('stats', self.stats)
            logger.info("resuming_from_checkpoint", current_date=current_date.isoformat())
        else:
            current_date = self.start_date
            logger.info("starting_fresh_scrape",
                       start_date=self.start_date.isoformat(),
                       end_date=self.end_date.isoformat())
        
        # Calculate total batches
        total_days = (self.end_date - self.start_date).days
        self.stats['total_batches'] = (total_days // self.batch_days) + 1
        
        logger.info("scrape_plan",
                   total_days=total_days,
                   batch_days=self.batch_days,
                   total_batches=self.stats['total_batches'])
        
        # Process in batches
        batch_num = checkpoint.get('batch_num', 0) if checkpoint else 0
        
        while current_date < self.end_date:
            batch_num += 1
            
            # Calculate batch end date
            batch_end = min(
                current_date + timedelta(days=self.batch_days),
                self.end_date
            )
            
            logger.info("batch_start",
                       batch_num=batch_num,
                       total_batches=self.stats['total_batches'],
                       start=current_date.date(),
                       end=batch_end.date())
            
            try:
                # Process this batch
                await self._process_batch(current_date, batch_end, batch_num)
                
                self.stats['batches_complete'] += 1
                
                # Save checkpoint
                self.checkpoint_manager.save({
                    'current_date': batch_end.isoformat(),
                    'batch_num': batch_num,
                    'stats': self.stats,
                })
                
                # Log progress
                self._log_progress(batch_num)
                
            except Exception as e:
                logger.error("batch_failed",
                           batch_num=batch_num,
                           error=str(e))
                
                # Save checkpoint anyway
                self.checkpoint_manager.save({
                    'current_date': current_date.isoformat(),
                    'batch_num': batch_num,
                    'stats': self.stats,
                })
                
                # Continue to next batch
                pass
            
            # Move to next batch
            current_date = batch_end
            
            # Brief pause between batches
            await asyncio.sleep(1)
        
        # Complete
        self._log_completion()
        self.checkpoint_manager.clear()
        self.deduplicator.close()
    
    async def _process_batch(self, start: datetime, end: datetime, batch_num: int):
        """Process a single batch of dates."""
        batch_start_time = time.time()
        
        # Step 1: Fetch article metadata from GDELT
        async with GDELTHistoricalFetcher(strict_mode=True) as fetcher:
            article_metas = await fetcher.fetch_historical(
                start_date=start,
                end_date=end,
                max_articles=None  # No limit per batch
            )
        
        logger.info("batch_fetch_complete",
                   batch_num=batch_num,
                   articles_fetched=len(article_metas))
        
        self.stats['articles_fetched'] += len(article_metas)
        
        if not article_metas:
            logger.warning("no_articles_in_batch", batch_num=batch_num)
            return
        
        # Step 2: Deduplicate
        unique_metas = []
        for meta in article_metas:
            url = meta.get('url')
            if not url or should_skip_url(url):
                continue
            
            if not self.deduplicator.is_duplicate(url):
                self.deduplicator.mark_seen(url)
                unique_metas.append(meta)
        
        logger.info("batch_deduplicated",
                   batch_num=batch_num,
                   unique_articles=len(unique_metas))
        
        if not unique_metas:
            return
        
        # Step 3: Extract content
        articles = []
        
        async with ContentExtractor() as extractor:
            # Process in smaller chunks to avoid memory issues
            chunk_size = 100
            
            for i in range(0, len(unique_metas), chunk_size):
                chunk = unique_metas[i:i + chunk_size]
                
                # Extract chunk in parallel
                tasks = [extractor.extract(meta) for meta in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if result and not isinstance(result, Exception):
                        # Generate ID
                        result['id'] = generate_article_id(
                            result['url'],
                            result['title'],
                            result.get('published_at')
                        )
                        articles.append(result)
                        self.stats['articles_extracted'] += 1
                    else:
                        self.stats['articles_failed'] += 1
                
                # Brief pause between chunks
                await asyncio.sleep(0.5)
        
        logger.info("batch_extraction_complete",
                   batch_num=batch_num,
                   articles_extracted=len(articles))
        
        # Step 4: Write batch output
        if articles:
            output_file = self.output_dir / f"batch_{batch_num:04d}_{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}.jsonl"
            
            import jsonlines
            with jsonlines.open(output_file, mode='w') as writer:
                for article in articles:
                    writer.write(article)
            
            logger.info("batch_written",
                       batch_num=batch_num,
                       file=str(output_file),
                       articles=len(articles))
        
        batch_duration = time.time() - batch_start_time
        logger.info("batch_complete",
                   batch_num=batch_num,
                   duration_seconds=batch_duration)
    
    def _log_progress(self, batch_num: int):
        """Log progress with time estimates."""
        elapsed = time.time() - self.start_time
        
        # Estimate time remaining
        batches_done = self.stats['batches_complete']
        batches_total = self.stats['total_batches']
        
        if batches_done > 0:
            avg_batch_time = elapsed / batches_done
            batches_remaining = batches_total - batches_done
            est_remaining = avg_batch_time * batches_remaining
            
            # Format time remaining
            hours_remaining = int(est_remaining // 3600)
            minutes_remaining = int((est_remaining % 3600) // 60)
            
            logger.info("progress_update",
                       batches_complete=batches_done,
                       batches_total=batches_total,
                       percent_complete=f"{(batches_done / batches_total * 100):.1f}%",
                       articles_fetched=self.stats['articles_fetched'],
                       articles_extracted=self.stats['articles_extracted'],
                       articles_failed=self.stats['articles_failed'],
                       elapsed_hours=f"{elapsed / 3600:.1f}h",
                       est_remaining=f"{hours_remaining}h {minutes_remaining}m")
    
    def _log_completion(self):
        """Log completion summary."""
        elapsed = time.time() - self.start_time
        
        logger.info("scrape_complete",
                   total_batches=self.stats['batches_complete'],
                   articles_fetched=self.stats['articles_fetched'],
                   articles_extracted=self.stats['articles_extracted'],
                   articles_failed=self.stats['articles_failed'],
                   success_rate=f"{self.stats['articles_extracted'] / max(self.stats['articles_fetched'], 1) * 100:.1f}%",
                   total_time=f"{elapsed / 3600:.1f}h")
        
        print("\n" + "=" * 60)
        print("✓ 12-MONTH SCRAPE COMPLETE")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  Batches processed: {self.stats['batches_complete']}")
        print(f"  Articles fetched: {self.stats['articles_fetched']}")
        print(f"  Articles extracted: {self.stats['articles_extracted']}")
        print(f"  Articles failed: {self.stats['articles_failed']}")
        print(f"  Success rate: {self.stats['articles_extracted'] / max(self.stats['articles_fetched'], 1) * 100:.1f}%")
        print(f"  Total time: {elapsed / 3600:.1f} hours")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Merge batch files: cat {self.output_dir}/batch_*.jsonl > articles_12mo.jsonl")
        print(f"  2. Copy to Belief Transformer: cp articles_12mo.jsonl ../../V2/data/raw/")
        print(f"  3. Run Belief Transformer analysis")


async def main():
    parser = argparse.ArgumentParser(
        description="Production 12-month scraper for Belief Transformer"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD), default: 12 months ago"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD), default: today"
    )
    
    parser.add_argument(
        "--batch-days",
        type=int,
        default=7,
        help="Days per batch (default: 7)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: data/processed)"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=365)  # 12 months
    
    output_dir = Path(args.output) if args.output else None
    
    # Create scraper
    scraper = ProductionScraper(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        batch_days=args.batch_days
    )
    
    # Run
    print("\n🚀 Starting 12-month production scrape...")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    print(f"   Batch size: {args.batch_days} days")
    if args.resume:
        print("   Mode: RESUME from checkpoint")
    print()
    
    await scraper.run(resume=args.resume)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Scrape interrupted by user")
        print("   Progress saved to checkpoint")
        print("   Resume with: python scrape_12_months.py --resume")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
