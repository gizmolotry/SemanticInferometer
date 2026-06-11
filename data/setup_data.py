"""
Setup data files for run_full_experiment_suite.py.

This script ensures your temporal_ALL.preprocessed file is available with the
filename that run_experiments.py expects for the real corpus.
"""

from pathlib import Path
import shutil
import sys


def setup_data() -> None:
    """Copy the preprocessed temporal corpus into the expected real-corpus path."""
    data_dir = Path("data")
    source_file = data_dir / "temporal_ALL.preprocessed"
    target_file = data_dir / "scraped_articles.jsonl"

    if not source_file.exists():
        print(f"[MISSING] Source file not found: {source_file}")
        print(f"\nExpected location: {source_file.absolute()}")
        print("\nDid you merge your batch files yet?")
        print(
            "Run: python merge_batches.py --input-dir data/temporal_cleaned "
            "--output data/temporal_ALL.preprocessed"
        )
        sys.exit(1)

    print(f"Copying {source_file.name} -> {target_file.name}...")
    shutil.copy2(source_file, target_file)

    size_mb = target_file.stat().st_size / 1024 / 1024
    print("[OK] Setup complete!")
    print(f"   File: {target_file}")
    print(f"   Size: {size_mb:.2f} MB")
    print("\nNow run:")
    print("   python run_full_experiment_suite.py --limit 500 --seeds 42 420 4200")


if __name__ == "__main__":
    setup_data()
