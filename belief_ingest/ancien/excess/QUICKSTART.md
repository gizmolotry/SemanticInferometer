# Quick Start Guide

Get the Belief Transformer Ingestion Pipeline running in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Test Everything Works

```bash
python test.py
```

You should see all tests pass. If not:
- Check you have Python 3.9+
- Ensure you have internet connection
- Install any missing dependencies

## Step 3: Run Your First Fetch

Fetch the last 2 hours of news (small test):

```bash
python -m ingest --hours 2 --max 100
```

This will:
- Fetch from 50+ RSS feeds
- Extract full article content
- Save to `data/processed/articles_TIMESTAMP.jsonl`

## Step 4: Check Output

```bash
# View the output file
ls -lh data/processed/

# Preview first article
head -n 1 data/processed/articles_*.jsonl | python -m json.tool
```

## Step 5: Scale Up

Now fetch 24 hours of news (thousands of articles):

```bash
python -m ingest --hours 24
```

This typically takes 30-60 minutes and fetches 10,000-20,000 articles.

## What's Next?

### Run Continuously

Add to crontab to run every hour:

```bash
crontab -e

# Add this line:
0 * * * * cd /path/to/belief_ingest && python -m ingest --hours 2
```

### Customize Sources

Edit `config.py` to:
- Add/remove RSS feeds
- Adjust concurrency limits
- Filter by domains

### Process the Data

```python
import jsonlines

# Read articles
with jsonlines.open('data/processed/articles_TIMESTAMP.jsonl') as f:
    articles = list(f)

# Now feed into your Belief Transformer!
for article in articles:
    # article['snippets'] are already segmented at ~100 words
    for snippet in article['snippets']:
        process_with_belief_transformer(snippet)
```

## Common Commands

```bash
# Fetch last 24 hours
python -m ingest --hours 24

# Limit to 1000 articles (for testing)
python -m ingest --hours 48 --max 1000

# Debug mode (verbose logging)
python -m ingest --hours 6 --debug

# Custom output directory
python -m ingest --hours 24 --output /my/data/path
```

## Troubleshooting

**Problem**: No articles fetched  
**Solution**: Some RSS feeds may be temporarily down. Try again later or disable GDELT in config.py

**Problem**: Extraction is slow  
**Solution**: Reduce `MAX_CONCURRENT_ARTICLES` in config.py from 100 to 50

**Problem**: Running out of disk space  
**Solution**: Old output files in `data/processed/` can be deleted after processing

## Getting Help

- Read the full README.md
- Check examples.py for usage patterns
- Run test.py to diagnose issues

---

**You're ready to go!** 🚀

The pipeline will give you clean, structured news articles at scale with no API keys required.
