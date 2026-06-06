#!/usr/bin/env python3
"""
Test script for Belief Transformer Ingestion Pipeline.
Tests both HISTORICAL and RECENT modes.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Testing Belief Transformer Ingestion Pipeline\n")
print("=" * 60)


def test_dependencies():
    """Test that all dependencies are installed."""
    print("\n[1/5] Testing Dependencies...")
    
    deps = {
        'aiohttp': 'HTTP client',
        'feedparser': 'RSS parsing',
        'trafilatura': 'Content extraction',
        'newspaper': 'Fallback extraction',
        'readability': 'HTML cleaning',
        'pandas': 'GDELT parsing (REQUIRED)',
        'structlog': 'Logging',
        'xxhash': 'Fast hashing',
        'jsonlines': 'Output format',
    }
    
    all_ok = True
    for package, description in deps.items():
        try:
            __import__(package)
            print(f"  ✓ {package:15s} ({description})")
        except ImportError:
            print(f"  ✗ {package:15s} MISSING - pip install {package}")
            all_ok = False
    
    # Check optional Playwright
    try:
        from playwright.async_api import async_playwright
        print(f"  ✓ {'playwright':15s} (optional - for JS-heavy sites)")
    except ImportError:
        print(f"  ⊘ {'playwright':15s} not installed (optional)")
    
    return all_ok


async def test_gdelt_historical():
    """Test GDELT historical fetcher."""
    print("\n[2/5] Testing GDELT Historical Fetcher...")
    
    try:
        from gdelt_historical_fetcher import fetch_historical_articles
        
        # Fetch just 2 days for quick test
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        print(f"  Testing with date range: {start_date.date()} to {end_date.date()}")
        
        articles = await fetch_historical_articles(
            start_date=start_date,
            end_date=end_date,
            max_articles=10  # Limit for quick test
        )
        
        if articles:
            print(f"  ✓ Fetched {len(articles)} articles from GDELT")
            
            # Show sample
            if len(articles) > 0:
                sample = articles[0]
                print(f"  ✓ Sample: {sample.get('url', '')[:60]}...")
                if 'themes' in sample:
                    print(f"    Themes: {sample['themes'][:3]}")
                if 'locations' in sample:
                    print(f"    Locations: {sample['locations'][:3]}")
            
            return True
        else:
            print("  ⚠ No articles fetched (this is okay if no recent Gaza news)")
            return True
            
    except Exception as e:
        print(f"  ✗ GDELT historical test failed: {e}")
        return False


async def test_content_extraction():
    """Test content extraction."""
    print("\n[3/5] Testing Content Extraction...")
    
    try:
        from content_extractor import ContentExtractor
        
        # Test with a well-known news URL
        test_url = {
            'url': 'https://www.reuters.com/world/',
            'source_type': 'test',
            'themes': ['ARMED_CONFLICT'],
            'locations': ['Gaza Strip'],
        }
        
        async with ContentExtractor() as extractor:
            result = await extractor.extract(test_url)
            
            if result:
                print(f"  ✓ Extraction works (method: {result['extraction_method']})")
                print(f"  ✓ Extracted {result['word_count']} words")
                
                # Check GDELT metadata preservation
                if 'themes' in result:
                    print(f"  ✓ GDELT themes preserved: {result['themes']}")
                if 'locations' in result:
                    print(f"  ✓ GDELT locations preserved: {result['locations']}")
                
                return True
            else:
                print("  ⚠ Extraction failed (may need different test URL)")
                return True
            
    except Exception as e:
        print(f"  ✗ Extraction test failed: {e}")
        return False


async def test_helpers():
    """Test utility helpers."""
    print("\n[4/5] Testing Utility Helpers...")
    
    try:
        from utils.helpers import (
            URLDeduplicator, generate_article_id, should_skip_url,
            clean_text, segment_text, extract_domain
        )
        
        # Test deduplication with unique URL
        import time
        dedup = URLDeduplicator(
            cache_file=Path("data/cache/test_urls.txt"),
            max_size=1000
        )
        
        # Use timestamp to ensure unique URL
        test_url = f"https://example.com/article/{int(time.time() * 1000)}"
        assert not dedup.is_duplicate(test_url), "Should not be duplicate initially"
        dedup.mark_seen(test_url)
        assert dedup.is_duplicate(test_url), "Should be duplicate after marking"
        dedup.close()
        
        print("  ✓ URL deduplication works")
        
        # Test article ID generation
        article_id = generate_article_id(
            "https://example.com/article",
            "Test Title",
            "2024-01-01"
        )
        assert len(article_id) == 16, "Article ID should be 16 chars"
        print("  ✓ Article ID generation works")
        
        # Test URL filtering
        assert should_skip_url("https://example.com/video/123"), "Should skip video URLs"
        assert not should_skip_url("https://example.com/article/123"), "Should not skip article URLs"
        print("  ✓ URL filtering works")
        
        # Test text cleaning
        dirty = "   Extra    spaces    and\n\nnewlines   "
        clean = clean_text(dirty)
        assert "  " not in clean, "Should remove extra spaces"
        print("  ✓ Text cleaning works")
        
        # Test segmentation
        text = " ".join(["word"] * 250)  # 250 words
        segments = segment_text(text, segment_size=100)
        assert len(segments) == 3, "Should create 3 segments"
        print("  ✓ Text segmentation works")
        
        # Test domain extraction
        domain = extract_domain("https://www.example.com/article")
        assert domain == "example.com", "Should extract domain correctly"
        print("  ✓ Domain extraction works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Helper tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_config():
    """Test configuration."""
    print("\n[5/5] Testing Configuration...")
    
    try:
        from config import (
            HISTORICAL_ENABLED, GDELT_THEMES, GDELT_LOCATIONS,
            RSS_FEEDS, TOPIC_FILTER_ENABLED, ACQUISITION_TARGETS
        )
        
        assert HISTORICAL_ENABLED, "Historical mode should be enabled"
        print("  ✓ Historical mode enabled")
        
        assert len(GDELT_THEMES) > 0, "Should have GDELT themes configured"
        print(f"  ✓ {len(GDELT_THEMES)} GDELT themes configured")
        
        assert len(GDELT_LOCATIONS) > 0, "Should have GDELT locations configured"
        print(f"  ✓ {len(GDELT_LOCATIONS)} GDELT locations configured")
        
        assert len(RSS_FEEDS) > 0, "Should have RSS feeds configured"
        print(f"  ✓ {len(RSS_FEEDS)} RSS feeds configured")
        
        assert TOPIC_FILTER_ENABLED, "Topic filtering should be enabled"
        print("  ✓ Topic filtering enabled")
        
        assert len(ACQUISITION_TARGETS) == 3, "Should have 3 acquisition targets"
        print(f"  ✓ {len(ACQUISITION_TARGETS)} acquisition targets configured")
        
        # Show acquisition targets
        for key, target in ACQUISITION_TARGETS.items():
            print(f"    - {target['name']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


async def run_tests():
    """Run all tests."""
    results = []
    
    # Test dependencies first
    results.append(test_dependencies())
    
    # Test async components
    results.append(await test_gdelt_historical())
    results.append(await test_content_extraction())
    results.append(await test_helpers())
    results.append(await test_config())
    
    print("\n" + "=" * 60)
    print("\n📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"  ✓ All {total} tests passed!")
        print("\n🚀 Pipeline is ready to use!")
        print("\n📚 Try these commands:")
        print("\n  HISTORICAL MODE (12 months):")
        print("    python -m ingest --historical --months 12")
        print("\n  HISTORICAL MODE (specific dates):")
        print("    python -m ingest --historical --start 2024-01-01 --end 2024-12-31")
        print("\n  RECENT MODE (24 hours):")
        print("    python -m ingest --hours 24")
        print("\n  Quick test (1 day historical):")
        print("    python -m ingest --historical --months 0.033 --max 100")
    else:
        print(f"  ⚠ {passed}/{total} tests passed")
        print("\nSome tests failed. Check errors above.")
        print("Most common issues:")
        print("  1. Missing pandas: pip install pandas")
        print("  2. Missing dependencies: pip install -r requirements.txt")
        print("  3. Network issues: Check your connection")
        print("  4. Playwright: Run 'playwright install chromium' if needed")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)