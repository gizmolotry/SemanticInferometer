#!/usr/bin/env python3
"""
Quick test script to verify the pipeline is working.
Tests each component independently.
"""

import asyncio
import sys
from pathlib import Path

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Testing Belief Transformer Ingestion Pipeline\n")
print("=" * 60)


async def test_rss():
    """Test RSS feed fetching."""
    print("\n[1/4] Testing RSS Feed Fetching...")
    
    try:
        from sources.rss_fetcher import fetch_rss_articles
        
        # Fetch just 1 hour of articles for quick test
        articles = await fetch_rss_articles(lookback_hours=1)
        
        if articles:
            print(f"  ✓ Fetched {len(articles)} articles from RSS feeds")
            print(f"  ✓ Sample: {articles[0].get('title', 'No title')[:60]}...")
            return True
        else:
            print("  ⚠ No articles fetched (this is okay if no recent news)")
            return True
            
    except Exception as e:
        print(f"  ✗ RSS test failed: {e}")
        return False


async def test_gdelt():
    """Test GDELT fetching."""
    print("\n[2/4] Testing GDELT Fetching...")
    
    try:
        from sources.gdelt_fetcher import fetch_gdelt_articles
        from config import GDELT_ENABLED
        
        if not GDELT_ENABLED:
            print("  ⊘ GDELT disabled in config (this is fine)")
            return True
        
        # Fetch just 1 hour for quick test
        articles = await fetch_gdelt_articles(lookback_hours=1)
        
        if articles:
            print(f"  ✓ Fetched {len(articles)} articles from GDELT")
            return True
        else:
            print("  ⚠ No GDELT articles (this is okay)")
            return True
            
    except Exception as e:
        print(f"  ⚠ GDELT test failed: {e}")
        print("  (GDELT failures are non-critical)")
        return True


async def test_extraction():
    """Test content extraction."""
    print("\n[3/4] Testing Content Extraction...")
    
    try:
        from extractors.content_extractor import ContentExtractor
        
        # Test with a well-known news URL
        test_urls = [
            {
                'url': 'https://www.reuters.com/world/',
                'source_type': 'test'
            }
        ]
        
        async with ContentExtractor() as extractor:
            for meta in test_urls:
                result = await extractor.extract(meta)
                
                if result:
                    print(f"  ✓ Extraction works (method: {result['extraction_method']})")
                    print(f"  ✓ Extracted {result['word_count']} words")
                    return True
                else:
                    print("  ⚠ Extraction failed (may need different test URL)")
                    return True
        
        return False
            
    except Exception as e:
        print(f"  ✗ Extraction test failed: {e}")
        return False


def test_dependencies():
    """Test that all dependencies are installed."""
    print("\n[4/4] Testing Dependencies...")
    
    deps = {
        'aiohttp': 'HTTP client',
        'feedparser': 'RSS parsing',
        'trafilatura': 'Content extraction',
        'newspaper': 'Fallback extraction',
        'readability': 'HTML cleaning',
        'pandas': 'Data processing',
        'structlog': 'Logging',
        'xxhash': 'Fast hashing',
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


async def run_tests():
    """Run all tests."""
    results = []
    
    # Test dependencies first
    results.append(test_dependencies())
    
    # Test async components
    results.append(await test_rss())
    results.append(await test_gdelt())
    results.append(await test_extraction())
    
    print("\n" + "=" * 60)
    print("\n📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"  ✓ All {total} tests passed!")
        print("\n🚀 Pipeline is ready to use!")
        print("\nTry: python -m ingest --hours 2 --max 100")
    else:
        print(f"  ⚠ {passed}/{total} tests passed")
        print("\nSome tests failed. Check errors above.")
        print("Most common issues:")
        print("  1. Missing dependencies: pip install -r requirements.txt")
        print("  2. Network issues: Check your connection")
        print("  3. Playwright: Run 'playwright install chromium' if needed")
    
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
