#!/usr/bin/env python3
"""
run_full_experiment_suite.py

ONE COMMAND TO RULE THEM ALL

Runs all experiments (Real + 3 Controls) × (N seeds)
Outputs to a timestamped folder with clear structure
Optionally runs an NLI minimal-pair probe after each corpus (synthetic + optional corpus perturbations)
Tracks variance at each pipeline stage (if supported by run_experiments.py)
Generates comparison report automatically

Usage:
    python run_full_experiment_suite.py --limit 500
    python run_full_experiment_suite.py --limit 500 --probe --probe-hypotheses hypotheses.json

Output structure:
    experiments_YYYYMMDD_HHMMSS/
        real/
            observer_42.pt
            observer_43.pt
            ...
            nli_probe_results.json           (if --probe)
        control_constant/
        control_shuffled/
        control_random/
        experiment_manifest.json
        comparison_results.json            (produced by compare_controls.py, if it saves)
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys


# -----------------------------
# Directory / orchestration
# -----------------------------

def create_experiment_directory() -> Path:
    """Create timestamped experiment directory with subfolders for each corpus."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments_{timestamp}")
    exp_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (exp_dir / "real").mkdir(exist_ok=True)
    (exp_dir / "control_constant").mkdir(exist_ok=True)
    (exp_dir / "control_shuffled").mkdir(exist_ok=True)
    (exp_dir / "control_random").mkdir(exist_ok=True)

    return exp_dir


def run_single_corpus(
    corpus: str,
    seeds: List[int],
    limit: int,
    output_dir: Path,
    track_variance: bool = True
) -> Dict:
    """Run experiments for one corpus via run_experiments.py, then move per-seed output files into output_dir."""
    
    # Fix Windows console encoding for Unicode characters
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print(f"\n{'='*80}")
    print(f"RUNNING: {corpus.upper()}")
    print(f"{'='*80}")

    # Build command - run_experiments.py already has UTF-8 fix built in
    cmd = [
        sys.executable,
        "run_experiments.py",
        "--corpus", corpus,
        "--mode", "enhanced",
        "--seeds",
    ] + [str(s) for s in seeds] + [
        "--limit", str(limit),
        "--output-root", str(output_dir),
    ]

    if track_variance:
        cmd.append("--track-variance")
    
    # Set UTF-8 environment for subprocess
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    print(f"Command: {' '.join(cmd)}")

    # Run experiment with UTF-8 encoding to handle Unicode output
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)

    if result.returncode != 0:
        print(f"❌ FAILED: {corpus}")
        print(f"STDERR: {result.stderr}")
        return {
            "corpus": corpus,
            "status": "failed",
            "error": result.stderr,
            "stdout": result.stdout,
            "returncode": result.returncode,
        }

    print(f"✅ COMPLETED: {corpus}")

    # Files are saved by run_experiments.py in output_dir
    # Check multiple possible naming patterns
    moved_files: List[str] = []
    missing: List[str] = []

    for seed in seeds:
        # Try multiple patterns
        patterns = [
            f"seed_{seed}.pt",  # NEW pattern (what's actually created)
            f"{corpus}_enhanced_mode_observer_{seed}.pt",  # OLD pattern
            f"observer_{seed}.pt",  # Already renamed
        ]
        
        found = False
        for pattern in patterns:
            source_file = output_dir / pattern
            if source_file.exists():
                # Rename to standard format if needed
                dest_file = output_dir / f"observer_{seed}.pt"
                if source_file != dest_file:
                    source_file.rename(dest_file)
                    print(f"  Renamed: {pattern} → observer_{seed}.pt")
                else:
                    print(f"  Found: {pattern}")
                moved_files.append(str(dest_file))
                found = True
                break
        
        if not found:
            missing.append(f"seed_{seed}.pt (or variants)")

    if missing:
        print(f"⚠ Missing expected output files for {corpus}: {len(missing)}")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... (+{len(missing) - 10} more)")

    return {
        "corpus": corpus,
        "status": "success",
        "seeds": seeds,
        "files": moved_files,
        "missing_files": missing,
        "output_dir": str(output_dir),
        "stdout": result.stdout,
        "returncode": result.returncode,
    }


# -----------------------------
# Probe integration (Option B + optional shuffle probes)
# -----------------------------

def _first_existing(path_candidates: List[Path]) -> Optional[Path]:
    for p in path_candidates:
        if p.exists():
            return p
    return None


def _detect_premises_jsonl(output_dir: Path) -> Optional[Path]:
    """
    Best-effort auto-detection of a JSONL file containing raw premises/texts inside a corpus output folder.
    If none found, the probe will still run synthetic minimal pairs (if entities provided).
    """
    candidates = [
        output_dir / "premises.jsonl",
        output_dir / "articles.jsonl",
        output_dir / "corpus.jsonl",
        output_dir / "texts.jsonl",
        output_dir / "data.jsonl",
    ]
    return _first_existing(candidates)


def run_nli_probe_for_corpus(
    corpus: str,
    output_dir: Path,
    probe_script: str,
    probe_model: str,
    probe_hypotheses_path: Path,
    probe_entities: str,
    probe_n_synth: int,
    probe_corpus_jsonl: Optional[Path],
    probe_corpus_field: str,
    probe_corpus_limit: int,
    probe_max_length: int,
    probe_batch_size: int,
) -> Dict:
    """
    Runs nli_probe.py (or compatible script) and writes results into output_dir/nli_probe_results.json.
    This is intentionally subprocess-based so you can drop it into an existing repo without refactoring imports.
    """
    out_path = output_dir / "nli_probe_results.json"

    if not Path(probe_script).exists():
        raise FileNotFoundError(
            f"Probe script not found: {probe_script}. "
            f"Expected a file path relative to the repo root or an absolute path."
        )

    cmd = [
        sys.executable,
        probe_script,
        "--model", probe_model,
        "--hypotheses", str(probe_hypotheses_path),
        "--out", str(out_path),
        "--n-synth", str(probe_n_synth),
        "--max-length", str(probe_max_length),
        "--batch-size", str(probe_batch_size),
    ]

    if probe_entities.strip():
        cmd.extend(["--entities", probe_entities.strip()])

    if probe_corpus_jsonl is not None:
        cmd.extend([
            "--corpus-jsonl", str(probe_corpus_jsonl),
            "--corpus-field", probe_corpus_field,
            "--corpus-limit", str(probe_corpus_limit),
        ])

    print(f"\n{'-'*80}")
    print(f"NLI PROBE: {corpus}")
    print(f"{'-'*80}")
    print(f"Probe command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode != 0:
        print(f"❌ PROBE FAILED: {corpus}")
        print(f"STDERR: {result.stderr}")
        return {
            "status": "failed",
            "corpus": corpus,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output": str(out_path),
        }

    print(f"✅ PROBE COMPLETED: {corpus}")
    # Keep stdout in the log (useful summary)
    if result.stdout.strip():
        print(result.stdout)

    return {
        "status": "success",
        "corpus": corpus,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": str(out_path),
        "premises_jsonl_used": str(probe_corpus_jsonl) if probe_corpus_jsonl else None,
    }


# -----------------------------
# Reporting
# -----------------------------

def save_manifest(exp_dir: Path, results: List[Dict], config: Dict):
    """Save experiment manifest."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "experiments": results,
        "directory_structure": {
            "real": str(exp_dir / "real"),
            "control_constant": str(exp_dir / "control_constant"),
            "control_shuffled": str(exp_dir / "control_shuffled"),
            "control_random": str(exp_dir / "control_random"),
        }
    }

    manifest_path = exp_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Saved manifest: {manifest_path}")


def run_comparison(exp_dir: Path, seeds: List[int]) -> bool:
    """Run comparison analysis on all experiments via compare_controls.py."""

    print(f"\n{'='*80}")
    print("RUNNING COMPARISON ANALYSIS")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        "compare_controls.py",
        "--data-dir", str(exp_dir),
        "--seeds",
    ] + [str(s) for s in seeds]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if result.returncode != 0:
        print(f"❌ Comparison failed: {result.stderr}")
        return False

    print("✅ Comparison complete")
    if result.stdout.strip():
        print(result.stdout)

    return True


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run complete experiment suite: Real + 3 Controls"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of articles per corpus (default: 500)"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds to run (default: 42 43 44)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing experiment directory (e.g., experiments_20260105_215516)"
    )

    parser.add_argument(
        "--no-variance-tracking",
        action="store_true",
        help="Disable variance tracking (if run_experiments.py supports it)"
    )

    # ---- Probe flags (ALWAYS ON with defaults) ----
    parser.add_argument(
        "--probe-script",
        default="nli_probe.py",
        help="Path to probe script (default: nli_probe.py)"
    )
    parser.add_argument(
        "--probe-model",
        default="roberta-large-mnli",
        help="HF model name for probe (default: roberta-large-mnli)"
    )
    parser.add_argument(
        "--probe-hypotheses",
        default="probe_hypotheses.json",  # DEFAULT to file in root
        help="Path to JSON file containing hypothesis strings (default: probe_hypotheses.json)"
    )
    parser.add_argument(
        "--probe-entities",
        default="Israel,Palestine,Gaza,Hamas,IDF,West Bank,Netanyahu,Fatah",
        help="Comma-separated entities for synthetic minimal pairs (default: Gaza/Israel entities)"
    )
    parser.add_argument(
        "--probe-n-synth",
        type=int,
        default=50,
        help="Pairs per synthetic transform type (default: 50)"
    )
    parser.add_argument(
        "--probe-corpus-jsonl",
        default="",
        help="Optional JSONL file of premises to run shuffle probes on. If omitted, auto-detect within each corpus output dir."
    )
    parser.add_argument(
        "--probe-corpus-field",
        default="text",
        help="Field name in JSONL for premise text (default: text)"
    )
    parser.add_argument(
        "--probe-corpus-limit",
        type=int,
        default=200,
        help="How many premises to sample for shuffle probes (default: 200)"
    )
    parser.add_argument(
        "--probe-max-length",
        type=int,
        default=256,
        help="Tokenizer max_length for probe (default: 256)"
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=16,
        help="Batch size for probe (default: 16)"
    )
    parser.add_argument(
        "--probe-nonfatal",
        action="store_true",
        default=True,  # DEFAULT: Continue even if probe fails
        help="Continue suite even if probe fails (default: True, use --no-probe-nonfatal to make fatal)"
    )
    parser.add_argument(
        "--no-probe-nonfatal",
        action="store_false",
        dest="probe_nonfatal",
        help="Make probe failures fatal (stop suite)"
    )

    args = parser.parse_args()

    # Validate probe files (always try to use probe)
    probe_hyp_path = Path(args.probe_hypotheses)
    probe_script_path = Path(args.probe_script)
    
    probe_enabled = True
    if not probe_hyp_path.exists():
        print(f"⚠️  WARNING: Probe hypotheses not found: {probe_hyp_path}")
        print(f"⚠️  Probe will be DISABLED")
        probe_enabled = False
    elif not probe_script_path.exists():
        print(f"⚠️  WARNING: Probe script not found: {probe_script_path}")
        print(f"⚠️  Probe will be DISABLED")
        probe_enabled = False

    # Create or resume experiment directory
    if args.resume:
        exp_dir = Path(args.resume)
        if not exp_dir.exists():
            raise SystemExit(f"ERROR: Resume directory not found: {exp_dir}")
        print(f"\n{'='*80}")
        print("🔄 RESUMING EXPERIMENT SUITE")
        print(f"{'='*80}")
        print(f"Resuming from: {exp_dir.absolute()}")
    else:
        exp_dir = create_experiment_directory()
        print(f"\n{'='*80}")
        print("FULL EXPERIMENT SUITE")
        print(f"{'='*80}")
        print(f"Output directory: {exp_dir.absolute()}")

    # Print config
    print(f"Articles per corpus: {args.limit}")
    print(f"Seeds: {args.seeds}")
    print(f"Variance tracking: {not args.no_variance_tracking}")
    if probe_enabled:
        print(f"Probe: ENABLED")
        print(f"  Script: {args.probe_script}")
        print(f"  Model: {args.probe_model}")
        print(f"  Hypotheses: {probe_hyp_path}")
        print(f"  Entities: {args.probe_entities or '(auto-detect)'}")
        print(f"  Non-fatal: {args.probe_nonfatal} (suite will {'continue' if args.probe_nonfatal else 'stop'} on probe failure)")
    else:
        print(f"Probe: DISABLED (missing files)")
    print(f"{'='*80}")

    corpora = ["real", "control_constant", "control_shuffled", "control_random"]
    results: List[Dict] = []
    
    # 🔍 CHECKPOINT DETECTION - Skip already completed corpora
    completed_corpora = set()
    seeds_to_run = {}  # corpus -> list of remaining seeds
    
    if args.resume:
        print(f"\n{'='*80}")
        print("🔍 SCANNING FOR CHECKPOINTS")
        print(f"{'='*80}")
        
        for corpus in corpora:
            corpus_dir = exp_dir / corpus
            if not corpus_dir.exists():
                seeds_to_run[corpus] = args.seeds
                print(f"  {corpus}: NOT STARTED (will run all seeds: {args.seeds})")
                continue
            
            # Check which seeds are done
            completed_seeds = []
            missing_seeds = []
            
            for seed in args.seeds:
                patterns = [f"seed_{seed}.pt", f"observer_{seed}.pt"]
                found = any((corpus_dir / p).exists() for p in patterns)
                if found:
                    completed_seeds.append(seed)
                else:
                    missing_seeds.append(seed)
            
            if missing_seeds:
                seeds_to_run[corpus] = missing_seeds
                print(f"  {corpus}: PARTIAL ({len(completed_seeds)}/{len(args.seeds)} seeds done)")
                print(f"    ✅ Completed: {completed_seeds}")
                print(f"    ⏳ Remaining: {missing_seeds}")
            else:
                completed_corpora.add(corpus)
                print(f"  {corpus}: ✅ COMPLETE (all {len(args.seeds)} seeds done)")
        
        print(f"{'='*80}")
        
        if completed_corpora == set(corpora):
            print("\n✅ ALL CORPORA COMPLETE! Nothing to resume.")
            print("Run comparison analysis or start a new experiment.")
            return
    else:
        # Fresh run - all seeds for all corpora
        for corpus in corpora:
            seeds_to_run[corpus] = args.seeds

    for corpus in corpora:
        # DEBUG: Confirm we're entering loop iteration
        print(f"\n[DEBUG] Loop iteration for corpus: {corpus}")
        
        # Skip if already complete
        if corpus in completed_corpora:
            print(f"\n{'='*80}")
            print(f"SKIPPING: {corpus.upper()} (already complete)")
            print(f"{'='*80}")
            continue
        
        # Get remaining seeds for this corpus
        remaining_seeds = seeds_to_run.get(corpus, args.seeds)
        if not remaining_seeds:
            continue
        
        output_dir = exp_dir / corpus.replace("_", "_")

        result = run_single_corpus(
            corpus=corpus,
            seeds=remaining_seeds,  # Use remaining seeds only!
            limit=args.limit,
            output_dir=output_dir,
            track_variance=not args.no_variance_tracking
        )

        # Stop early on failure
        if result["status"] == "failed":
            results.append(result)
            print(f"\n❌ Experiment failed: {corpus}")
            print("Stopping experiment suite.")
            break

        # Probe (OPTIONAL - only if hypotheses provided)
        if probe_enabled:
            try:
                if args.probe_corpus_jsonl:
                    premises_path = Path(args.probe_corpus_jsonl)
                    if not premises_path.exists():
                        raise FileNotFoundError(f"--probe-corpus-jsonl not found: {premises_path}")
                    probe_corpus_jsonl = premises_path
                else:
                    probe_corpus_jsonl = _detect_premises_jsonl(output_dir)

                probe_result = run_nli_probe_for_corpus(
                    corpus=corpus,
                    output_dir=output_dir,
                    probe_script=args.probe_script,
                    probe_model=args.probe_model,
                    probe_hypotheses_path=probe_hyp_path,
                    probe_entities=args.probe_entities,
                    probe_n_synth=args.probe_n_synth,
                    probe_corpus_jsonl=probe_corpus_jsonl,
                    probe_corpus_field=args.probe_corpus_field,
                    probe_corpus_limit=args.probe_corpus_limit,
                    probe_max_length=args.probe_max_length,
                    probe_batch_size=args.probe_batch_size,
                )
                result["probe"] = probe_result

                # If probe failed and nonfatal is off, stop suite
                if probe_result.get("status") != "success" and not args.probe_nonfatal:
                    results.append(result)
                    print("\n❌ Probe failed and --probe-nonfatal is not set.")
                    print("Stopping experiment suite.")
                    break

            except Exception as e:
                # Treat probe exceptions as failures unless nonfatal
                result["probe"] = {
                    "status": "failed",
                    "corpus": corpus,
                    "error": str(e),
                }
                if not args.probe_nonfatal:
                    results.append(result)
                    print(f"\n❌ Probe exception: {e}")
                    print("Stopping experiment suite.")
                    break
                else:
                    print(f"⚠ Probe exception (nonfatal): {e}")

        results.append(result)
        
        # Progress update
        completed_so_far = len([r for r in results if r.get("status") == "success"])
        total_remaining = len(corpora) - len(completed_corpora)
        print(f"\n{'='*80}")
        print(f"📊 PROGRESS: {completed_so_far}/{total_remaining} corpora completed in this run")
        print(f"{'='*80}\n")
        sys.stdout.flush()  # Force output to appear immediately
        
        # DEBUG: Confirm we're continuing
        print(f"[DEBUG] About to continue to next corpus...")
        sys.stdout.flush()

    # DEBUG: Reached end of loop
    print(f"\n[DEBUG] Exited main loop, processing {len(results)} results")
    sys.stdout.flush()
    
    # Save manifest
    config = {
        "limit": args.limit,
        "seeds": args.seeds,
        "variance_tracking": not args.no_variance_tracking,
        "probe": {
            "enabled": True,  # FIXED: Always enabled now
            "script": args.probe_script,
            "model": args.probe_model,
            "hypotheses": str(probe_hyp_path),
            "entities": args.probe_entities,
            "n_synth": args.probe_n_synth,
            "corpus_jsonl": args.probe_corpus_jsonl or "auto",
            "corpus_field": args.probe_corpus_field,
            "corpus_limit": args.probe_corpus_limit,
            "max_length": args.probe_max_length,
            "batch_size": args.probe_batch_size,
            "nonfatal": args.probe_nonfatal,
        }
    }
    save_manifest(exp_dir, results, config)

    # Only run comparison if all experiments succeeded
    all_success = all(r.get("status") == "success" for r in results if r.get("corpus") in corpora)
    if all_success:
        run_comparison(exp_dir, args.seeds)
    else:
        print("\n⚠ Skipping comparison due to earlier failures.")

    # Summary
    print(f"\n{'='*80}")
    print("SUITE COMPLETE")
    print(f"{'='*80}")
    print("Results summary:")
    for r in results:
        status_emoji = "✅" if r.get("status") == "success" else "❌"
        print(f"  {status_emoji} {r.get('corpus')}: {r.get('status')}")

        if "probe" in r:
            p = r["probe"]
            p_emoji = "✅" if p.get("status") == "success" else "❌"
            print(f"      Probe: {p_emoji} {p.get('status')}")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print(f"1. View results: cd {exp_dir}")
    print(f"2. Check variance: cat */variance_tracking.json (if present)")
    print(f"3. View comparison: cat comparison_results.json (if produced)")
    print(f"4. View probe results: cat */nli_probe_results.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()