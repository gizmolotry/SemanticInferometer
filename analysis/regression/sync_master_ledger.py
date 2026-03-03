#!/usr/bin/env python3
"""
Sync Master Ledger and Consolidate Epistemic Contracts.

1. Consolidates baseline_meta.json and verification_report.json into EPISTEMIC_CONTRACT.json.
2. Builds a unified MASTER_LEDGER.json for cross-run analysis.
"""

import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT

def discover_run_dirs() -> List[Path]:
    patterns = (
        "experiments_*/synthetic/*",
        "experiments/experiments_*/synthetic/*",
        "outputs/experiments/runs/experiments_*/synthetic/*",
    )
    run_dirs = []
    for pattern in patterns:
        for run_dir in ROOT.glob(pattern):
            if run_dir.is_dir() and any(run_dir.glob("*.html")):
                run_dirs.append(run_dir)
    return run_dirs

def consolidate_contract(run_dir: Path) -> dict:
    meta_path = run_dir / "baseline_meta.json"
    report_path = run_dir / "verification_report.json"
    contract_path = run_dir / "EPISTEMIC_CONTRACT.json"
    
    meta = {}
    if meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
            
    report = {}
    if report_path.exists():
        with report_path.open("r") as f:
            report = json.load(f)

    summary_path = run_dir.parent / "synthetic_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                summary = json.load(f)
                for res in summary.get("results", []):
                    if res.get("run_key") == run_dir.name:
                        meta["nmi"] = res.get("nmi", meta.get("nmi", 0.0))
                        meta["ari"] = res.get("ari", meta.get("ari", 0.0))
                        break
        except Exception:
            pass
            
    rel_dir = run_dir / "relativity_cache"
    type2_stats = {"n_dissonant": 0, "total_observers": 0}
    if rel_dir.exists():
        states = list(rel_dir.glob("state_*.json"))
        type2_stats["total_observers"] = len(states)
        for s_path in states:
            with s_path.open("r") as f:
                s_data = json.load(f)
                if s_data.get("metrics", {}).get("type2_dissonance"):
                    type2_stats["n_dissonant"] += 1

    contract = {
        "provenance": meta,
        "verification": report,
        "relativity": type2_stats,
        "consensus": {
            "global_pass": report.get("global_pass", False),
            "type2_robustness": 1.0 - (type2_stats["n_dissonant"] / type2_stats["total_observers"]) if type2_stats["total_observers"] > 0 else 1.0
        }
    }
    
    with contract_path.open("w") as f:
        json.dump(contract, f, indent=2)
        
    return contract

def main():
    run_dirs = discover_run_dirs()
    if not run_dirs:
        print("[LEDGER] No run directories found.")
        return

    master_ledger = {}
    
    for run_dir in run_dirs:
        print(f"[LEDGER] Syncing {run_dir.name}...")
        contract = consolidate_contract(run_dir)
        
        master_ledger[run_dir.name] = {
            "type1": contract["consensus"]["global_pass"],
            "type2": contract["consensus"]["type2_robustness"],
            "nmi": contract["provenance"].get("nmi", 0.0),
            "ari": contract["provenance"].get("ari", 0.0),
            "path": str(run_dir.relative_to(ROOT))
        }

    ledger_path = ROOT / "analysis" / "MASTER_LEDGER.json"
    with ledger_path.open("w") as f:
        json.dump(master_ledger, f, indent=2)
        
    print(f"[LEDGER] Master Ledger synced to {ledger_path}")

if __name__ == "__main__":
    main()
