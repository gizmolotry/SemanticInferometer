#!/bin/bash

echo "======================================================================"
echo "  Belief Transformer - 12-Month Scraper Setup"
echo "======================================================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
echo "✓ Python 3 found"
echo ""

# Install core dependencies
echo "[2/5] Installing core dependencies..."
pip install --break-system-packages \
    aiohttp \
    feedparser \
    trafilatura \
    newspaper3k \
    beautifulsoup4 \
    lxml \
    readability-lxml \
    pandas \
    xxhash \
    structlog \
    jsonlines \
    python-dateutil \
    aiohttp-retry || { echo "❌ Dependency installation failed"; exit 1; }

echo "✓ Core dependencies installed"
echo ""

# Optional: Install Playwright
echo "[3/5] Install Playwright? (for JS-heavy sites like NYT, WSJ)"
echo "   This is optional but recommended. Install? (y/n)"
read -r install_playwright

if [[ "$install_playwright" =~ ^[Yy]$ ]]; then
    echo "Installing Playwright..."
    pip install --break-system-packages playwright
    playwright install chromium
    echo "✓ Playwright installed"
else
    echo "⊘ Skipping Playwright (can install later if needed)"
fi
echo ""

# Create directory structure
echo "[4/5] Creating directory structure..."
mkdir -p data/{raw,processed,cache} logs utils sources
echo "✓ Directories created"
echo ""

# Run tests
echo "[5/5] Running test suite..."
python3 test.py

echo ""
echo "======================================================================"
echo "  Setup Complete!"
echo "======================================================================"
echo ""
echo "Quick Start:"
echo ""
echo "  # Test with 1 week (5-10 minutes):"
echo "  python3 scrape_12_months.py --start 2024-12-15 --end 2024-12-21"
echo ""
echo "  # Full 12-month scrape (4-8 hours):"
echo "  python3 scrape_12_months.py"
echo ""
echo "  # Resume if interrupted:"
echo "  python3 scrape_12_months.py --resume"
echo ""
echo "See README.md for more usage patterns."
echo ""
