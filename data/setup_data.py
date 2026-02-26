"""
Setup data files for run_full_experiment_suite.py

This script ensures your temporal_ALL.preprocessed file is available
with the filename that run_experiments.py expects.
"""

import shutil
from pathlib import Path
import sys

def setup_data():
    """Setup data files."""
    
    data_dir = Path("data")
    
    # Your preprocessed file
    source_file = data_dir / "temporal_ALL.preprocessed"
    
    # What run_experiments.py looks for (for "real" corpus)
    target_file = data_dir / "scraped_articles.jsonl"
    
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        print(f"\nExpected location: {source_file.absolute()}")
        print(f"\nDid you merge your batch files yet?")
        print(f"Run: python merge_batches.py --input-dir data/temporal_cleaned --output data/temporal_ALL.preprocessed")
        sys.exit(1)
    
    # Copy (not symlink for Windows compatibility)
    print(f"Copying {source_file.name} -> {target_file.name}...")
    shutil.copy2(source_file, target_file)
    
    size_mb = target_file.stat().st_size / 1024 / 1024
    
    print(f"✅ Setup complete!")
    print(f"   File: {target_file}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"\nNow run:")
    print(f"   python run_full_experiment_suite.py --limit 500 --seeds 42 420 4200")


if __name__ == "__main__":
    setup_data()
