"""
GDELT Project data fetcher - no API keys required!
Pulls from GDELT 2.0 event database via public CSV files.
"""

import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import structlog
from io import StringIO
import zipfile
from pathlib import Path

from config import (
    GDELT_ENABLED, GDELT_LOOKBACK_HOURS, GDELT_MAX_ARTICLES,
    GDELT_ENDPOINTS, GDELT_DOMAIN_FILTER, CACHE_DIR,
    REQUEST_TIMEOUT, USER_AGENT
)
from utils.helpers import normalize_url, extract_domain, parse_date

logger = structlog.get_logger()


class GDELTFetcher:
    """
    Fetch news articles from GDELT Project.
    
    GDELT monitors news outlets worldwide and provides:
    - Article URLs
    - Publish times
    - Source domains
    - Geographic info
    - Themes/tones
    
    All data is public and free to access.
    """
    
    def __init__(self):
        self.enabled = GDELT_ENABLED
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = CACHE_DIR / "gdelt"
        self.cache_dir.mkdir(exist_ok=True)
    
    async def __aenter__(self):
        """Setup async context."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT * 2)  # GDELT can be slow
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async context."""
        if self.session:
            await self.session.close()
    
    async def fetch_recent_articles(self, lookback_hours: int = None) -> List[Dict]:
        """
        Fetch recent articles from GDELT.
        
        Args:
            lookback_hours: How far back to look (default: config value)
        
        Returns:
            List of article metadata
        """
        if not self.enabled:
            logger.info("gdelt_disabled")
            return []
        
        lookback_hours = lookback_hours or GDELT_LOOKBACK_HOURS
        
        logger.info("fetching_gdelt", lookback_hours=lookback_hours)
        
        try:
            # Get list of recent data files
            file_urls = await self._get_recent_file_urls(lookback_hours)
            
            if not file_urls:
                logger.warning("no_gdelt_files_found")
                return []
            
            logger.info("gdelt_files_found", count=len(file_urls))
            
            # Fetch and parse files
            articles = []
            for file_url in file_urls[:20]:  # Limit to avoid overwhelming
                try:
                    file_articles = await self._fetch_and_parse_file(file_url)
                    articles.extend(file_articles)
                    
                    if len(articles) >= GDELT_MAX_ARTICLES:
                        break
                        
                except Exception as e:
                    logger.warning("gdelt_file_error", 
                                 url=file_url, 
                                 error=str(e))
                    continue
            
            # Deduplicate by URL
            unique_articles = self._deduplicate_articles(articles)
            
            logger.info("gdelt_fetch_complete",
                       total=len(unique_articles),
                       domains=len(set(a['publisher'] for a in unique_articles)))
            
            return unique_articles[:GDELT_MAX_ARTICLES]
            
        except Exception as e:
            logger.error("gdelt_fetch_failed", error=str(e))
            return []
    
    async def _get_recent_file_urls(self, lookback_hours: int) -> List[str]:
        """
        Get list of GDELT file URLs for recent time window.
        
        GDELT updates every 15 minutes with a new ZIP file.
        """
        try:
            # Get master file list
            async with self.session.get(GDELT_ENDPOINTS['master']) as response:
                if response.status != 200:
                    logger.warning("gdelt_master_list_failed", status=response.status)
                    return []
                
                content = await response.text()
                
                # Parse file list (format: size hash url)
                lines = content.strip().split('\n')
                
                # Filter for GKG (Global Knowledge Graph) files which have article URLs
                # and for recent files based on timestamp in filename
                cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
                
                file_urls = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        url = parts[2]
                        
                        # We want the GKG files (contain article data)
                        if '.gkg.csv.zip' in url:
                            # Extract timestamp from filename
                            # Format: ...YYYYMMDDHHMMSS.gkg.csv.zip
                            try:
                                timestamp_str = url.split('/')[-1].split('.')[0]
                                file_time = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                                
                                if file_time >= cutoff_time:
                                    file_urls.append(url)
                            except Exception:
                                continue
                
                # Sort by timestamp (most recent first)
                file_urls.sort(reverse=True)
                
                return file_urls
                
        except Exception as e:
            logger.error("gdelt_master_list_error", error=str(e))
            return []
    
    async def _fetch_and_parse_file(self, file_url: str) -> List[Dict]:
        """
        Fetch and parse a GDELT GKG CSV file.
        
        GKG files contain rich metadata about news articles.
        """
        try:
            # Download ZIP file
            async with self.session.get(file_url) as response:
                if response.status != 200:
                    return []
                
                zip_content = await response.read()
            
            # Extract CSV from ZIP
            import io
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # Get first (only) file in ZIP
                csv_filename = zf.namelist()[0]
                csv_content = zf.read(csv_filename).decode('utf-8', errors='ignore')
            
            # Parse CSV
            # GDELT GKG format is tab-separated
            # Key columns: DATE, SourceCommonName, DocumentIdentifier, Themes, Locations, Tone
            df = pd.read_csv(
                StringIO(csv_content),
                sep='\t',
                on_bad_lines='skip',
                low_memory=False,
                usecols=['DATE', 'SourceCommonName', 'DocumentIdentifier', 'Themes', 'Locations', 'Tone'],
                dtype=str
            )
            
            articles = []
            for _, row in df.iterrows():
                try:
                    url = row.get('DocumentIdentifier', '').strip()
                    if not url or not url.startswith('http'):
                        continue
                    
                    url = normalize_url(url)
                    domain = extract_domain(url)
                    
                    # Apply domain filter if configured
                    if GDELT_DOMAIN_FILTER and domain not in GDELT_DOMAIN_FILTER:
                        continue
                    
                    # Parse date
                    date_str = row.get('DATE', '')
                    published_at = None
                    if date_str:
                        try:
                            # Format: YYYYMMDDHHMMSS
                            dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                            published_at = dt.isoformat()
                        except Exception:
                            pass
                    
                    # Extract metadata
                    article = {
                        'url': url,
                        'title': None,  # GDELT doesn't provide titles
                        'summary': None,
                        'published_at': published_at,
                        'author': None,
                        'publisher': domain,
                        'source_name': row.get('SourceCommonName', '').strip(),
                        'section': None,
                        'source_type': 'gdelt',
                        'fetched_at': datetime.now().isoformat(),
                    }
                    
                    # Parse themes (semicolon-separated)
                    themes_str = row.get('Themes', '')
                    if themes_str and pd.notna(themes_str):
                        article['themes'] = [t.strip() for t in themes_str.split(';') if t.strip()]
                    
                    # Parse locations (semicolon-separated)
                    locations_str = row.get('Locations', '')
                    if locations_str and pd.notna(locations_str):
                        article['locations'] = [l.strip() for l in locations_str.split(';') if l.strip()][:3]
                    
                    # Parse tone (sentiment score)
                    tone_str = row.get('Tone', '')
                    if tone_str and pd.notna(tone_str):
                        try:
                            tone_parts = tone_str.split(',')
                            if tone_parts:
                                article['sentiment_score'] = float(tone_parts[0])
                        except Exception:
                            pass
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.debug("gdelt_row_parse_error", error=str(e))
                    continue
            
            return articles
            
        except Exception as e:
            logger.warning("gdelt_file_parse_error", url=file_url, error=str(e))
            return []
    
    @staticmethod
    def _deduplicate_articles(articles: List[Dict]) -> List[Dict]:
        """Remove duplicate URLs."""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article['url']
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles


async def fetch_gdelt_articles(lookback_hours: int = None) -> List[Dict]:
    """
    Convenience function to fetch GDELT articles.
    
    Args:
        lookback_hours: How far back to look
    
    Returns:
        List of article metadata
    """
    async with GDELTFetcher() as fetcher:
        return await fetcher.fetch_recent_articles(lookback_hours)


# For testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    async def test():
        articles = await fetch_gdelt_articles(lookback_hours=2)
        print(f"\nFetched {len(articles)} articles from GDELT")
        
        if articles:
            print("\nSample article:")
            import json
            print(json.dumps(articles[0], indent=2))
            
            # Show domain distribution
            domains = {}
            for a in articles:
                d = a.get('publisher', 'unknown')
                domains[d] = domains.get(d, 0) + 1
            
            print("\nTop domains:")
            for d, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {d}: {count}")
    
    asyncio.run(test())
