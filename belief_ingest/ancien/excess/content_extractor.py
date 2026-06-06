"""
Multi-strategy content extraction with graceful fallbacks and topic filtering.

Priority order:
1. trafilatura (fast, accurate)
2. newspaper3k (good fallback)
3. readability-lxml (HTML cleaning)
4. Playwright (last resort for JS-heavy sites)

Now includes Gaza/Israel topic filtering at extraction stage.
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Tuple
from datetime import datetime
import structlog

import trafilatura
from newspaper import Article
from readability import Document
from bs4 import BeautifulSoup

from config import (
    MIN_ARTICLE_LENGTH, MAX_ARTICLE_LENGTH, EXTRACTION_STRATEGIES,
    PLAYWRIGHT_DOMAINS, REQUEST_TIMEOUT, USER_AGENT,
    TOPIC_FILTER_ENABLED, TOPIC_KEYWORDS
)
from utils.helpers import clean_text, segment_text, extract_domain, parse_date

logger = structlog.get_logger()


class ContentExtractor:
    """
    Robust content extraction with multiple strategies.
    Tries fast methods first, falls back to expensive methods.
    Includes Gaza/Israel topic filtering.
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.playwright_browser = None
        self.playwright_context = None
    
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
        
        if self.playwright_context:
            await self.playwright_context.close()
        
        if self.playwright_browser:
            await self.playwright_browser.close()
    
    async def extract(self, article_meta: Dict) -> Optional[Dict]:
        """
        Extract full article content using best available method.
        Filters for Gaza/Israel topics.
        
        Args:
            article_meta: Metadata from RSS/GDELT (must include 'url')
        
        Returns:
            Complete article dict with content, or None if extraction failed or filtered
        """
        url = article_meta.get('url')
        if not url:
            logger.warning("no_url_provided")
            return None
        
        domain = extract_domain(url)
        
        # Determine extraction strategy
        strategies = self._get_strategies(domain)
        
        for strategy in strategies:
            try:
                logger.debug("trying_extraction", url=url, strategy=strategy)
                
                if strategy == "trafilatura":
                    result = await self._extract_trafilatura(url, article_meta)
                elif strategy == "newspaper3k":
                    result = await self._extract_newspaper(url, article_meta)
                elif strategy == "readability":
                    result = await self._extract_readability(url, article_meta)
                elif strategy == "playwright":
                    result = await self._extract_playwright(url, article_meta)
                else:
                    continue
                
                if result and self._validate_content(result):
                    # 🎯 TOPIC FILTERING - Filter for Gaza/Israel
                    if not self._matches_topic(result):
                        logger.debug("article_filtered_by_topic", 
                                   url=url,
                                   title=result.get('title', '')[:50])
                        return None
                    
                    result['extraction_method'] = strategy
                    logger.debug("extraction_success", url=url, strategy=strategy)
                    return result
                    
            except Exception as e:
                logger.debug("extraction_failed", 
                           url=url, 
                           strategy=strategy, 
                           error=str(e))
                continue
        
        logger.warning("all_extraction_methods_failed", url=url)
        return None
    
    def _get_strategies(self, domain: str) -> list:
        """Determine extraction strategy order based on domain."""
        # For known JS-heavy/paywalled sites, go straight to Playwright
        if any(pw_domain in domain for pw_domain in PLAYWRIGHT_DOMAINS):
            return ["playwright"]
        
        # Otherwise try fast methods first
        return EXTRACTION_STRATEGIES
    
    async def _extract_trafilatura(self, url: str, meta: Dict) -> Optional[Dict]:
        """
        Extract using trafilatura (recommended method).
        Fast, accurate, handles most news sites.
        """
        # Fetch HTML
        async with self.session.get(url) as response:
            if response.status != 200:
                return None
            html = await response.text()
        
        # Extract with trafilatura
        extracted = trafilatura.extract(
            html,
            favor_recall=True,
            include_comments=False,
            include_tables=False,
            output_format='json',
            with_metadata=True
        )
        
        if not extracted:
            return None
        
        # Parse JSON result
        import json
        data = json.loads(extracted)
        
        content = data.get('text', '')
        if not content:
            return None
        
        # Clean content
        content = clean_text(content)
        
        # Build result
        result = {
            **meta,
            'content': content,
            'title': data.get('title') or meta.get('title'),
            'author': data.get('author') or meta.get('author'),
            'published_at': data.get('date') or meta.get('published_at'),
            'word_count': len(content.split()),
            'char_count': len(content),
            'snippets': segment_text(content),
        }
        
        return result
    
    async def _extract_newspaper(self, url: str, meta: Dict) -> Optional[Dict]:
        """
        Extract using newspaper3k library.
        Good fallback, handles many sites.
        """
        # Fetch HTML
        async with self.session.get(url) as response:
            if response.status != 200:
                return None
            html = await response.text()
        
        # Parse with newspaper3k
        article = Article(url)
        article.download(input_html=html)
        article.parse()
        
        content = article.text
        if not content:
            return None
        
        content = clean_text(content)
        
        # newspaper3k also does NLP (can be slow)
        try:
            article.nlp()
        except Exception:
            pass
        
        # Build result
        result = {
            **meta,
            'content': content,
            'title': article.title or meta.get('title'),
            'author': ', '.join(article.authors) if article.authors else meta.get('author'),
            'published_at': article.publish_date.isoformat() if article.publish_date else meta.get('published_at'),
            'word_count': len(content.split()),
            'char_count': len(content),
            'snippets': segment_text(content),
        }
        
        if article.keywords:
            result['keywords'] = article.keywords[:10]
        
        if article.summary:
            result['summary'] = article.summary
        
        return result
    
    async def _extract_readability(self, url: str, meta: Dict) -> Optional[Dict]:
        """
        Extract using readability-lxml.
        Good for cleaning messy HTML.
        """
        # Fetch HTML
        async with self.session.get(url) as response:
            if response.status != 200:
                return None
            html = await response.text()
        
        # Parse with readability
        doc = Document(html)
        
        # Get cleaned HTML
        clean_html = doc.summary()
        
        # Extract text from HTML
        soup = BeautifulSoup(clean_html, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)
        
        if not content:
            return None
        
        content = clean_text(content)
        
        # Build result
        result = {
            **meta,
            'content': content,
            'title': doc.title() or meta.get('title'),
            'word_count': len(content.split()),
            'char_count': len(content),
            'snippets': segment_text(content),
        }
        
        return result
    
    async def _extract_playwright(self, url: str, meta: Dict) -> Optional[Dict]:
        """
        Extract using Playwright (last resort).
        Slow but handles JS-rendered content and paywalls.
        """
        # Lazy-load Playwright
        if not self.playwright_browser:
            await self._init_playwright()
        
        try:
            page = await self.playwright_context.new_page()
            
            # Navigate with timeout
            await page.goto(url, timeout=REQUEST_TIMEOUT * 1000, wait_until='networkidle')
            
            # Wait for content to load
            await page.wait_for_timeout(2000)
            
            # Get page HTML
            html = await page.content()
            
            # Close page
            await page.close()
            
            # Now try trafilatura on the rendered HTML
            extracted = trafilatura.extract(
                html,
                favor_recall=True,
                include_comments=False,
                output_format='json',
                with_metadata=True
            )
            
            if not extracted:
                return None
            
            import json
            data = json.loads(extracted)
            
            content = data.get('text', '')
            if not content:
                return None
            
            content = clean_text(content)
            
            # Build result
            result = {
                **meta,
                'content': content,
                'title': data.get('title') or meta.get('title'),
                'author': data.get('author') or meta.get('author'),
                'published_at': data.get('date') or meta.get('published_at'),
                'word_count': len(content.split()),
                'char_count': len(content),
                'snippets': segment_text(content),
            }
            
            return result
            
        except Exception as e:
            logger.warning("playwright_extraction_failed", url=url, error=str(e))
            return None
    
    async def _init_playwright(self):
        """Initialize Playwright browser (lazy loaded)."""
        try:
            from playwright.async_api import async_playwright
            
            logger.info("initializing_playwright")
            
            playwright = await async_playwright().start()
            self.playwright_browser = await playwright.chromium.launch(
                headless=True,
                args=['--disable-dev-shm-usage']
            )
            self.playwright_context = await self.playwright_browser.new_context(
                user_agent=USER_AGENT
            )
            
            logger.info("playwright_initialized")
            
        except Exception as e:
            logger.error("playwright_init_failed", error=str(e))
            raise
    
    @staticmethod
    def _validate_content(article: Dict) -> bool:
        """Validate extracted content meets quality thresholds."""
        content = article.get('content', '')
        
        if not content:
            return False
        
        # Check length
        char_count = len(content)
        if char_count < MIN_ARTICLE_LENGTH:
            logger.debug("content_too_short", chars=char_count)
            return False
        
        if char_count > MAX_ARTICLE_LENGTH:
            logger.debug("content_too_long", chars=char_count)
            return False
        
        # Check has title
        if not article.get('title'):
            logger.debug("no_title")
            return False
        
        # Check word count
        word_count = len(content.split())
        if word_count < 50:
            logger.debug("too_few_words", words=word_count)
            return False
        
        return True
    
    @staticmethod
    def _matches_topic(article: Dict) -> bool:
        """
        Check if article matches Gaza/Israel topic.
        
        Returns:
            True if article matches topic (or filtering disabled)
            False if article should be filtered out
        """
        # If filtering disabled, accept all articles
        if not TOPIC_FILTER_ENABLED:
            return True
        
        content = article.get('content', '').lower()
        title = article.get('title', '').lower()
        
        # Check if any keyword matches
        matches = any(
            keyword.lower() in content or keyword.lower() in title 
            for keyword in TOPIC_KEYWORDS
        )
        
        return matches


# For testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    async def test():
        test_urls = [
            {
                'url': 'https://www.timesofisrael.com/liveblog_entry/netanyahu-hamas-must-release-hostages-by-monday-or-face-consequences/',
                'source_type': 'test'
            },
            {
                'url': 'https://www.bbc.com/news/world-middle-east',
                'source_type': 'test'
            }
        ]
        
        async with ContentExtractor() as extractor:
            for meta in test_urls:
                print(f"\nTesting: {meta['url']}")
                result = await extractor.extract(meta)
                
                if result:
                    print(f"✓ Success: {result['title'][:60]}...")
                    print(f"  Method: {result['extraction_method']}")
                    print(f"  Words: {result['word_count']}")
                    print(f"  Gaza/Israel: {'Yes' if 'gaza' in result['content'].lower() or 'israel' in result['content'].lower() else 'No'}")
                else:
                    print("✗ Failed or filtered")
    
    asyncio.run(test())
