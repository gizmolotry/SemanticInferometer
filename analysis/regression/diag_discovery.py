import os
from pathlib import Path

exp_dir = Path("../experiments_20260225_034558")
print(f"Checking exp_dir: {exp_dir.resolve()}")

groups = [d for d in exp_dir.iterdir() if d.is_dir() and d.name not in ["ablation", "alpha_sweep", "alignment", "viz_output"]]
print(f"Found groups: {[g.name for g in groups]}")

for group in groups:
    print(f"Checking group: {group.name}")
    for seed_dir in group.iterdir():
        print(f"  Checking seed_dir: {seed_dir.name} (is_dir: {seed_dir.is_dir()})")
        observer_files = list(seed_dir.glob("observer_*.pt"))
        print(f"    Found {len(observer_files)} observer files")
