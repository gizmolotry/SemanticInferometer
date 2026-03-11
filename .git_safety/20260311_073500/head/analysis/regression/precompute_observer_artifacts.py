#!/usr/bin/env python3
"""
Precompute observer/article artifact files for Dash browsing.

Creates observer_N/MONOLITH.html files and writes observer_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def _load_article_indices(monolith_csv: Path) -> List[int]:
    if not monolith_csv.exists():
        raise FileNotFoundError(f"Missing MONOLITH_DATA.csv: {monolith_csv}")
    indices: List[int] = []
    with monolith_csv.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row_i, row in enumerate(reader):
            raw = row.get("index", "")
            try:
                idx = int(raw)
            except Exception:
                idx = row_i
            indices.append(idx)
    return indices


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "exists"
        dst.unlink()
    _ensure_parent(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def _render_focused(run_dir: Path, output_path: Path, observer_idx: int, strict: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "analysis" / "MONOLITH_VIZ.py"),
        str(run_dir),
        "--output",
        str(output_path),
        "--observer-idx",
        str(observer_idx),
    ]
    if strict:
        cmd.append("--strict")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute observer artifacts and manifest for Dash.")
    parser.add_argument("run_dir", type=Path, help="Path to run folder containing MONOLITH_DATA.csv")
    parser.add_argument("--variant", default="MONOLITH.html", help="Variant filename to materialize")
    parser.add_argument(
        "--mode",
        choices=["link", "copy", "focused"],
        default="link",
        help="link/copy reuses existing global artifact; focused renders one file per observer index.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing observer files.")
    parser.add_argument("--strict", action="store_true", help="Pass strict validation to focused renders.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    indices = _load_article_indices(monolith_csv)
    variant_name = str(args.variant)
    global_variant = run_dir / variant_name

    if not global_variant.exists() and args.mode in {"link", "copy"}:
        raise FileNotFoundError(
            f"Global artifact missing: {global_variant}. Generate it first or use --mode focused."
        )

    records = []
    for idx in indices:
        out_path = run_dir / f"observer_{idx}" / variant_name
        if args.mode == "focused":
            _ensure_parent(out_path)
            if out_path.exists() and not args.overwrite:
                action = "exists"
            else:
                _render_focused(run_dir, out_path, idx, args.strict)
                action = "rendered"
        elif args.mode == "copy":
            action = _link_or_copy(global_variant, out_path, args.overwrite)
            if action == "hardlink":
                action = "copy"
        else:
            action = _link_or_copy(global_variant, out_path, args.overwrite)

        records.append(
            {
                "idx": idx,
                "value": f"article:{idx}",
                "relative_path": str(out_path.relative_to(run_dir)).replace("\\", "/"),
                "exists": out_path.exists(),
                "action": action,
            }
        )

    found = sum(1 for r in records if r["exists"])
    total = len(records)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "variant": variant_name,
        "mode": args.mode,
        "n_articles": total,
        "coverage": {
            "found": found,
            "total": total,
            "pct": (float(found) / float(total)) if total else 0.0,
        },
        "observers": records,
    }

    manifest_path = run_dir / "observer_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[observer-precompute] run={run_dir}")
    print(f"[observer-precompute] mode={args.mode} variant={variant_name}")
    print(f"[observer-precompute] coverage={found}/{total}")
    print(f"[observer-precompute] manifest={manifest_path}")


if __name__ == "__main__":
    main()
