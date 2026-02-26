# Belief Transformer - 12-Month Gaza/Israel News Scraper

Production-ready scraper for collecting 12 months of Gaza/Israel news coverage from GDELT's historical archives.

## Features

✅ **12-month historical data** from GDELT archives  
✅ **Automatic checkpointing** - resume from interruptions  
✅ **Progress tracking** with time estimates  
✅ **Batch processing** to avoid memory issues  
✅ **Multi-strategy content extraction** (trafilatura, newspaper3k, readability, playwright)  
✅ **Smart deduplication** using xxhash  
✅ **GDELT metadata preservation** (themes, locations, sentiment scores)  

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (REQUIRED)
pip install aiohttp feedparser trafilatura newspaper3k beautifulsoup4 lxml readability-lxml pandas xxhash structlog jsonlines python-dateutil aiohttp-retry --break-system-packages

# Optional (for JS-heavy sites like NYT, WSJ)
pip install playwright --break-system-packages
playwright install chromium
```

### 2. Run 12-Month Scrape

```bash
# Default: Last 12 months
python scrape_12_months.py

# Custom date range
python scrape_12_months.py --start 2024-01-01 --end 2024-12-31

# Resume from checkpoint (if interrupted)
python scrape_12_months.py --resume
```

### 3. Expected Output

```
data/processed/
├── batch_0001_20240101-20240107.jsonl
├── batch_0002_20240108-20240114.jsonl
├── batch_0003_20240115-20240121.jsonl
...
└── batch_0052_20241220-20241226.jsonl
```

Each batch contains articles in JSONL format with:
- Full article content
- GDELT themes (ARMED_CONFLICT, TERROR, etc.)
- GDELT locations (Gaza Strip, Israel, etc.)
- Sentiment scores
- Text segmentation for Belief Transformer

## Usage Patterns

### Quick Test (1 Week)

```bash
python scrape_12_months.py --start 2024-12-15 --end 2024-12-21 --batch-days 1
```

### Full 12 Months (Production)

```bash
# Start the scrape
python scrape_12_months.py

# If it gets interrupted (power loss, network issues, etc.)
python scrape_12_months.py --resume

# Merge all batches into single file
cat data/processed/batch_*.jsonl > articles_12mo.jsonl

# Copy to Belief Transformer V2
cp articles_12mo.jsonl ../V2/data/raw/
```

### Acquisition Study Windows

```bash
# Irish Independent: Pre-acquisition (Jan 2018 - Jun 2019)
python scrape_12_months.py --start 2018-01-01 --end 2019-06-30 --output data/processed/irish_pre

# Irish Independent: Post-acquisition (Jul 2019 - Dec 2020)
python scrape_12_months.py --start 2019-07-01 --end 2020-12-31 --output data/processed/irish_post
```

## Configuration

Edit `config.py` to customize:

- **Topic filtering**: Keywords, themes, locations
- **Extraction strategies**: trafilatura → newspaper3k → readability → playwright
- **Performance tuning**: Batch sizes, concurrency limits, timeouts
- **GDELT filtering**: Strict vs lenient mode

### Strict vs Lenient Filtering

**Strict mode** (default): `theme AND (location OR URL keyword)`
- Higher precision, fewer articles
- Recommended for focused analysis

**Lenient mode**: `theme OR location OR URL keyword`  
- Higher recall, more articles
- Useful for exploratory analysis

To enable lenient mode, edit `config.py`:
```python
HISTORICAL_STRICT_MODE = False
```

## Performance

**Expected throughput**: ~50-100 articles/minute (depending on network)

**12-month scrape estimates**:
- **Strict filtering**: ~20,000-40,000 articles, ~4-8 hours
- **Lenient filtering**: ~40,000-80,000 articles, ~8-16 hours

**Batch processing**: 7-day batches by default
- Prevents memory issues
- Enables checkpointing
- Adjust with `--batch-days` flag

## Checkpoint System

The scraper automatically saves progress after each batch to `data/cache/checkpoint.json`.

If interrupted:
```bash
python scrape_12_months.py --resume
```

The scraper will:
1. Load the checkpoint
2. Resume from the last completed batch
3. Preserve all statistics

## Output Format

Each article is a JSON object with:

```json
{
  "id": "a1b2c3d4e5f6g7h8",
  "url": "https://www.reuters.com/...",
  "title": "Article Title",
  "content": "Full article text...",
  "snippets": ["segment 1...", "segment 2...", ...],
  "publisher": "reuters.com",
  "author": "John Doe",
  "published_at": "2024-01-15T10:30:00",
  "fetched_at": "2024-12-21T13:45:00",
  "source_type": "gdelt",
  "extraction_method": "trafilatura",
  "word_count": 842,
  "char_count": 5234,
  "themes": ["ARMED_CONFLICT", "MIDDLE_EAST", "HUMANITARIAN"],
  "locations": ["Gaza Strip", "Israel"],
  "sentiment_score": -12.5
}
```

## Troubleshooting

### Missing pandas

```bash
pip install pandas --break-system-packages
```

### Extraction failures

Check extraction method distribution:
```bash
cat data/processed/batch_*.jsonl | jq '.extraction_method' | sort | uniq -c
```

If many failures, install playwright:
```bash
pip install playwright --break-system-packages
playwright install chromium
```

### Memory issues

Reduce batch size:
```bash
python scrape_12_months.py --batch-days 3
```

### Network timeouts

Edit `config.py`:
```python
REQUEST_TIMEOUT = 60  # Increase from 30
RETRY_ATTEMPTS = 5    # Increase from 3
```

## Integration with Belief Transformer

1. **Scrape the data**:
   ```bash
   python scrape_12_months.py
   cat data/processed/batch_*.jsonl > articles_12mo.jsonl
   ```

2. **Copy to Belief Transformer**:
   ```bash
   cp articles_12mo.jsonl ../V2/data/raw/gaza_israel_12mo.jsonl
   ```

3. **Compute θ fingerprints**:
   ```bash
   cd ../V2
   python -m belief_transformer.compute_embeddings \
       --input data/raw/gaza_israel_12mo.jsonl \
       --output data/embeddings/
   ```

4. **Analyze stance distributions**:
   ```python
   from belief_transformer import BeliefTransformer
   
   bt = BeliefTransformer()
   bt.load_articles("data/raw/gaza_israel_12mo.jsonl")
   bt.compute_belief_space()
   bt.analyze_publisher_stances()
   ```

## Advanced Usage

### Testing the Pipeline

```bash
# Test dependencies
python test.py

# Test GDELT fetch
python -c "
from gdelt_historical_fetcher import fetch_historical_articles
from datetime import datetime, timedelta
import asyncio

async def test():
    end = datetime.now()
    start = end - timedelta(days=2)
    articles = await fetch_historical_articles(start, end, max_articles=10)
    print(f'Fetched {len(articles)} articles')

asyncio.run(test())
"
```

### Analyzing Output

```bash
# Count articles per publisher
cat articles_12mo.jsonl | jq -r '.publisher' | sort | uniq -c | sort -rn | head -20

# Average sentiment by month
cat articles_12mo.jsonl | jq -r '[.published_at[:7], .sentiment_score] | @tsv' | awk '{sum[$1]+=$2; count[$1]++} END {for (m in sum) print m, sum[m]/count[m]}' | sort

# Theme distribution
cat articles_12mo.jsonl | jq -r '.themes[]' | sort | uniq -c | sort -rn | head -20
```

## File Structure

```
belief_ingest/
├── scrape_12_months.py          # Main production scraper
├── gdelt_historical_fetcher.py  # GDELT API wrapper
├── content_extractor.py         # Multi-strategy extraction
├── config.py                    # Configuration
├── ingest.py                    # General ingestion pipeline
├── test.py                      # Test suite
├── examples.py                  # Usage examples
├── requirements.txt             # Dependencies
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Utilities
└── data/
    ├── cache/                   # URL cache, checkpoints
    ├── processed/               # Output batches
    └── logs/                    # Logs
```

## Support

For issues or questions:
1. Check test results: `python test.py`
2. Review logs in `data/logs/`
3. Check checkpoint: `cat data/cache/checkpoint.json`

## License

MIT License - Academic Research Use
