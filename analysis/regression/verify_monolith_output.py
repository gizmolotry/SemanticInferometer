#!/usr/bin/env python3
"""Fast verification for generated MONOLITH HTML outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CANONICAL_ZONES = {"Bridge", "Swamp", "Tightrope", "Void"}


def extract_hud_nmi(html_text: str) -> float | None:
    m = re.search(r"T5:\s*\d+H/\d+P/\d+R\s*\|\s*NMI:\s*<span[^>]*>([0-9]+\.[0-9]+)</span>", html_text)
    if not m:
        return None
    return float(m.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MONOLITH HTML output integrity.")
    parser.add_argument("experiment_dir", type=Path, help="Experiment seed directory containing validation.json")
    parser.add_argument("html_path", type=Path, help="Generated MONOLITH HTML to verify")
    args = parser.parse_args()

    validation_path = args.experiment_dir / "validation.json"
    if not validation_path.exists():
        print(f"[FAIL] Missing validation.json: {validation_path}")
        return 1
    if not args.html_path.exists():
        print(f"[FAIL] Missing HTML output: {args.html_path}")
        return 1

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected_nmi = validation.get("nmi", validation.get("normalized_mutual_info"))
    html_text = args.html_path.read_text(encoding="utf-8", errors="replace")

    failures: list[str] = []

    if "Fault" in html_text:
        failures.append("Found legacy zone label 'Fault' in HTML.")

    hud_nmi = extract_hud_nmi(html_text)
    if expected_nmi is not None:
        if hud_nmi is None:
            failures.append("HUD T5 NMI not found in HTML.")
        elif abs(hud_nmi - round(float(expected_nmi), 3)) > 1e-9:
            failures.append(f"HUD T5 NMI mismatch: html={hud_nmi:.3f}, validation={float(expected_nmi):.3f}.")

    zone_hits = {z: (z in html_text) for z in CANONICAL_ZONES}
    if not any(zone_hits.values()):
        failures.append("None of the canonical zone labels were detected in HTML.")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        return 1

    print("[PASS] MONOLITH HTML verification passed.")
    if hud_nmi is not None:
        print(f"[PASS] HUD T5 NMI: {hud_nmi:.3f}")
    print("[PASS] No legacy 'Fault' label detected.")
    print("[PASS] Canonical zone labels detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

