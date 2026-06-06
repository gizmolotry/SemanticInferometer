#!/bin/bash
#
# Setup script for Belief Transformer Ingestion Pipeline
#

set -e

echo "🚀 Setting up Belief Transformer Ingestion Pipeline"
echo "=================================================="
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo
    read -p "Create virtual environment? (recommended) [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        echo "✓ Virtual environment created and activated"
    fi
fi

# Install dependencies
echo
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "✓ Dependencies installed"

# Ask about Playwright
echo
read -p "Install Playwright for JS-heavy sites? (optional) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Playwright browsers..."
    playwright install chromium
    echo "✓ Playwright installed"
else
    echo "⊘ Skipping Playwright (can install later with: playwright install chromium)"
fi

# Create directories
echo
echo "Creating data directories..."
mkdir -p data/raw data/processed data/cache logs
echo "✓ Directories created"

# Run tests
echo
echo "Running tests..."
python3 test.py

echo
echo "=================================================="
echo "✅ Setup complete!"
echo
echo "Quick start:"
echo "  python -m ingest --hours 24 --max 100"
echo
echo "Full documentation:"
echo "  cat README.md"
echo
