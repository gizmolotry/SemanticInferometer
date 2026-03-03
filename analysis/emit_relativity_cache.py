#!/usr/bin/env python3
"""
Emit Relativity Cache for Type 2 Dissonance Calculation.

Iterates through experiment runs, calculates global vs local survival rates,
and emits state_{idx}.json and delta_{idx}.json in relativity_cache/.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Constants mirroring isolated_dash_prototype.py
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT

def discover_artifact_roots() -> List[Path]:
    patterns = (
        "experiments_*/synthetic",
        "experiments/experiments_*/synthetic",
        "outputs/experiments/runs/experiments_*/synthetic",
        "outputs/experiments/runs/*/synthetic",
        "outputs/experiments/*/synthetic",
    )
    candidate_roots: List[Path] = []
    for pattern in patterns:
        for synthetic in ROOT.glob(pattern):
            if synthetic.exists() and synthetic.is_dir():
                candidate_roots.append(synthetic)
    return candidate_roots

def process_run(run_dir: Path):
    print(f"[RELATIVITY] Processing {run_dir.name}...")
    
    data_csv = run_dir / "MONOLITH_DATA.csv"
    if not data_csv.exists():
        print(f"  [SKIP] MONOLITH_DATA.csv not found.")
        return

    # 1. Load data and calculate global mean
    articles = []
    try:
        with data_csv.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                articles.append(row)
    except Exception as e:
        print(f"  [ERROR] Failed to read CSV: {e}")
        return

    if not articles:
        print(f"  [SKIP] No articles found.")
        return

    # Metric extraction (assuming 'survival_rate' or 'honesty' or similar exists)
    def get_survival(row):
        try:
            total = float(row.get("n_total", 0))
            if total > 0:
                broken = float(row.get("n_broken", 0))
                trapped = float(row.get("n_trapped", 0))
                return max(0.0, 1.0 - ((broken + trapped) / total))
        except:
            pass
        return 0.0

    survivals = [get_survival(a) for a in articles]
    global_survival_mean = sum(survivals) / len(survivals) if survivals else 0.0

    # 2. Prepare relativity_cache
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)

    # 3. Emit state and delta for each observer
    for i, article in enumerate(articles):
        idx = article.get("index", str(i))
        local_survival = survivals[i]
        delta_survival = local_survival - global_survival_mean
        
        dissonance = abs(delta_survival) > 0.15 
        
        state_blob = {
            "observer_id": int(idx),
            "articles": [article],
            "metrics": {
                "local_survival_rate": local_survival,
                "global_survival_mean": global_survival_mean,
                "d_survival_pct": delta_survival * 100.0,
                "type2_dissonance": dissonance,
            },
            "provenance": {
                "run_id": run_dir.name,
                "type": "ARTICLE_OBSERVER"
            }
        }
        
        delta_blob = {
            "observer_id": int(idx),
            "null_observer_equivalence": local_survival > 0.9,
            "path_flip_delta": 0.0,
            "metrics_delta": {
                "survival_diff": delta_survival
            },
            "axis_delta": 0.0
        }
        
        with (rel_dir / f"state_{idx}.json").open("w") as f:
            json.dump(state_blob, f, indent=2)
        with (rel_dir / f"delta_{idx}.json").open("w") as f:
            json.dump(delta_blob, f, indent=2)

    print(f"  [OK] Emitted {len(articles)} relativity states. Global Mean: {global_survival_mean:.4f}")

def main():
    roots = discover_artifact_roots()
    if not roots:
        print("[RELATIVITY] No artifact roots found.")
        return

    for root in roots:
        for run_dir in root.iterdir():
            if run_dir.is_dir() and any(run_dir.glob("*.html")):
                process_run(run_dir)

if __name__ == "__main__":
    main()
