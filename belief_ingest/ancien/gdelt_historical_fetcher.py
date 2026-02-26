"""
GDELT Historical Fetcher for Belief Transformer
Fetches articles from GDELT's public dataset with Gaza/Israel filtering.

Features:
- Historical data (12+ months back)
- Theme-based filtering (ARMED_CONFLICT, TERROR, MILITARY, etc.)
- Location filtering (Gaza Strip, Israel, etc.)
- URL keyword filtering
- Strict/lenient modes
- No API key required
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import structlog
from pathlib import Path
import io

logger = structlog.get_logger()

# GDELT GKG (Global Knowledge Graph) columns we care about
GKG_COLUMNS = [
    'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
    'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
    'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
    'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
    'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
    'AllNames', 'Amounts', 'TranslationInfo', 'Extras'
]

# Gaza/Israel themes to filter by
GAZA_THEMES = {
    # Conflict
    'ARMED_CONFLICT', 'TERROR', 'MILITARY', 'MILITARY_ACTION',
    'WAR', 'ATTACK', 'VIOLENCE', 'CEASEFIRE',
    
    # Humanitarian
    'HUMANITARIAN', 'REFUGEE', 'CASUALTIES', 'HUMAN_RIGHTS',
    
    # Political
    'PEACE_TALKS', 'DIPLOMACY', 'SANCTIONS', 'PROTEST',
    
    # Regional
    'MIDDLE_EAST', 'ARAB_ISRAELI_CONFLICT'
}

# Gaza/Israel locations
GAZA_LOCATIONS = {
    'gaza', 'gaza strip', 'gaza city', 'israel', 'israeli',
    'tel aviv', 'jerusalem', 'west bank', 'rafah', 'khan younis',
    'palestine', 'palestinian'
}

# URL keywords
GAZA_URL_KEYWORDS = {
    'gaza', 'israel', 'hamas', 'palestinian', 'netanyahu',
    'idf', 'middle-east', 'mideast'
}


class GDELTHistoricalFetcher:
    """
    Fetch historical articles from GDELT with Gaza/Israel filtering.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, require theme AND (location OR URL).
                        If False, require theme OR location OR URL.
        """
        self.strict_mode = strict_mode
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = Path("data/cache/gdelt")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        max_articles: int = None
    ) -> List[Dict]:
        """
        Fetch historical articles from GDELT.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            max_articles: Maximum articles to fetch (None = unlimited)
        
        Returns:
            List of article metadata dicts
        """
        logger.info("gdelt_historical_fetch_start",
                   start_date=start_date.isoformat(),
                   end_date=end_date.isoformat(),
                   strict_mode=self.strict_mode)
        
        articles = []
        current_date = start_date
        
        while current_date <= end_date:
            if max_articles and len(articles) >= max_articles:
                logger.info("max_articles_reached", count=len(articles))
                break
            
            # Fetch one day's worth of articles
            day_articles = await self._fetch_day(current_date)
            articles.extend(day_articles)
            
            logger.info("gdelt_day_complete",
                       date=current_date.strftime("%Y-%m-%d"),
                       articles_today=len(day_articles),
                       total_articles=len(articles))
            
            current_date += timedelta(days=1)
            
            # Brief delay to avoid overwhelming GDELT
            await asyncio.sleep(0.1)
        
        # Truncate if over max
        if max_articles and len(articles) > max_articles:
            articles = articles[:max_articles]
        
        logger.info("gdelt_historical_fetch_complete",
                   total_articles=len(articles),
                   date_range=f"{start_date.date()} to {end_date.date()}")
        
        return articles
    
    async def _fetch_day(self, date: datetime) -> List[Dict]:
        """Fetch all articles for a single day."""
        articles = []
        
        # GDELT updates every 15 minutes, so there are 96 files per day
        # Format: YYYYMMDDHHMMSS
        for hour in range(0, 24):
            for minute in [0, 15, 30, 45]:
                timestamp = date.replace(hour=hour, minute=minute, second=0)
                
                # Check cache first
                cached = self._read_cache(timestamp)
                if cached is not None:
                    articles.extend(cached)
                    continue
                
                # Fetch from GDELT
                try:
                    file_articles = await self._fetch_15min_file(timestamp)
                    articles.extend(file_articles)
                    
                    # Cache the results
                    self._write_cache(timestamp, file_articles)
                    
                    # Brief delay
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    logger.debug("gdelt_file_fetch_failed",
                               timestamp=timestamp.isoformat(),
                               error=str(e))
                    continue
        
        return articles
    
    async def _fetch_15min_file(self, timestamp: datetime) -> List[Dict]:
        """Fetch a single 15-minute GKG file from GDELT."""
        # Format: http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip
        date_str = timestamp.strftime("%Y%m%d%H%M%S")
        url = f"http://data.gdeltproject.org/gdeltv2/{date_str}.gkg.csv.zip"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                # Read zip content
                zip_content = await response.read()
                
                # Extract CSV from zip
                import zipfile
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                    csv_filename = f"{date_str}.gkg.csv"
                    with zf.open(csv_filename) as csv_file:
                        # Read CSV with pandas
                        df = pd.read_csv(
                            csv_file,
                            sep='\t',
                            header=None,
                            names=GKG_COLUMNS,
                            low_memory=False,
                            on_bad_lines='skip'
                        )
                
                # Filter for Gaza/Israel articles
                articles = self._filter_articles(df)
                
                return articles
                
        except Exception as e:
            logger.debug("gdelt_file_parse_error",
                        url=url,
                        error=str(e))
            return []
    
    def _filter_articles(self, df: pd.DataFrame) -> List[Dict]:
        """Filter GDELT data for Gaza/Israel articles."""
        articles = []
        
        for idx, row in df.iterrows():
            # Extract fields
            url = str(row.get('DocumentIdentifier', ''))
            if not url or url == 'nan':
                continue
            
            themes = str(row.get('V2Themes', '') or row.get('Themes', ''))
            locations = str(row.get('V2Locations', '') or row.get('Locations', ''))
            tone = str(row.get('V2Tone', ''))
            date_str = str(row.get('DATE', ''))
            source_name = str(row.get('SourceCommonName', ''))
            
            # Check if matches Gaza/Israel topic
            if not self._matches_topic(url, themes, locations):
                continue
            
            # Parse date
            try:
                published_at = datetime.strptime(date_str, "%Y%m%d%H%M%S")
            except:
                published_at = None
            
            # Parse tone (format: "tone,positive,negative,polarity,activity_ref_density,self_ref_density,word_count")
            sentiment_score = None
            if tone and tone != 'nan':
                try:
                    tone_parts = tone.split(',')
                    if len(tone_parts) > 0:
                        sentiment_score = float(tone_parts[0])
                except:
                    pass
            
            # Extract domain
            from urllib.parse import urlparse
            try:
                domain = urlparse(url).netloc
                if domain.startswith('www.'):
                    domain = domain[4:]
            except:
                domain = source_name
            
            # Build article metadata
            article = {
                'url': url,
                'publisher': domain or source_name,
                'source_type': 'gdelt',
                'published_at': published_at.isoformat() if published_at else None,
                'themes': self._parse_themes(themes),
                'locations': self._parse_locations(locations),
                'sentiment_score': sentiment_score,
            }
            
            articles.append(article)
        
        return articles
    
    def _matches_topic(self, url: str, themes: str, locations: str) -> bool:
        """Check if article matches Gaza/Israel topic."""
        url_lower = url.lower()
        themes_lower = themes.lower()
        locations_lower = locations.lower()
        
        # Check theme match
        theme_match = any(
            theme.lower() in themes_lower
            for theme in GAZA_THEMES
        )
        
        # Check location match
        location_match = any(
            loc in locations_lower
            for loc in GAZA_LOCATIONS
        )
        
        # Check URL keyword match
        url_match = any(
            keyword in url_lower
            for keyword in GAZA_URL_KEYWORDS
        )
        
        if self.strict_mode:
            # Strict: theme AND (location OR URL)
            return theme_match and (location_match or url_match)
        else:
            # Lenient: theme OR location OR URL
            return theme_match or location_match or url_match
    
    def _parse_themes(self, themes_str: str) -> List[str]:
        """Parse GDELT themes string into list."""
        if not themes_str or themes_str == 'nan':
            return []
        
        # Themes are semicolon-separated
        themes = themes_str.split(';')
        
        # Filter for Gaza-relevant themes
        relevant = [
            t.strip() for t in themes
            if any(gaza_theme.lower() in t.lower() for gaza_theme in GAZA_THEMES)
        ]
        
        return relevant[:10]  # Limit to 10 themes
    
    def _parse_locations(self, locations_str: str) -> List[str]:
        """Parse GDELT locations string into list."""
        if not locations_str or locations_str == 'nan':
            return []
        
        # Locations format: type#name#country#adm1#lat#long;...
        locations = []
        for loc in locations_str.split(';'):
            parts = loc.split('#')
            if len(parts) >= 2:
                location_name = parts[1].strip()
                if location_name and any(
                    gaza_loc in location_name.lower()
                    for gaza_loc in GAZA_LOCATIONS
                ):
                    locations.append(location_name)
        
        return locations[:10]  # Limit to 10 locations
    
    def _read_cache(self, timestamp: datetime) -> Optional[List[Dict]]:
        """Read cached GDELT data if available."""
        cache_file = self.cache_dir / f"{timestamp.strftime('%Y%m%d%H%M%S')}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            import json
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _write_cache(self, timestamp: datetime, articles: List[Dict]):
        """Write GDELT data to cache."""
        cache_file = self.cache_dir / f"{timestamp.strftime('%Y%m%d%H%M%S')}.json"
        
        try:
            import json
            with open(cache_file, 'w') as f:
                json.dump(articles, f)
        except Exception as e:
            logger.debug("cache_write_failed", error=str(e))


async def fetch_historical_articles(
    start_date: datetime = None,
    end_date: datetime = None,
    months: float = None,
    max_articles: int = None,
    lenient: bool = False
) -> List[Dict]:
    """
    Convenience function to fetch historical GDELT articles.
    
    Args:
        start_date: Start date (if None and months provided, calculated from end_date)
        end_date: End date (if None, uses today)
        months: Number of months back from end_date (can be decimal, e.g., 0.25 for 1 week)
        max_articles: Maximum articles to fetch
        lenient: If True, use lenient filtering (more recall, less precision)
    
    Returns:
        List of article metadata
    """
    # Calculate dates
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        if months:
            start_date = end_date - timedelta(days=int(months * 30))
        else:
            start_date = end_date - timedelta(days=30)  # Default: 1 month
    
    async with GDELTHistoricalFetcher(strict_mode=not lenient) as fetcher:
        articles = await fetcher.fetch_historical(
            start_date=start_date,
            end_date=end_date,
            max_articles=max_articles
        )
    
    return articles


# For testing
if __name__ == "__main__":
    async def test():
        print("Testing GDELT Historical Fetcher")
        print("=" * 60)
        
        # Test: Fetch last 7 days
        end = datetime.now()
        start = end - timedelta(days=7)
        
        print(f"\nFetching articles from {start.date()} to {end.date()}...")
        
        articles = await fetch_historical_articles(
            start_date=start,
            end_date=end,
            max_articles=100
        )
        
        print(f"\n✓ Fetched {len(articles)} articles")
        
        if articles:
            print("\nSample article:")
            sample = articles[0]
            print(f"  URL: {sample['url']}")
            print(f"  Publisher: {sample['publisher']}")
            print(f"  Themes: {sample.get('themes', [])}")
            print(f"  Locations: {sample.get('locations', [])}")
            print(f"  Sentiment: {sample.get('sentiment_score')}")
    
    asyncio.run(test())
