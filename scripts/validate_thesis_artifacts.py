#!/usr/bin/env python3
"""Validate reviewer-facing thesis evidence plus minimal canonical documentation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.verification.thesis_evidence import SUMMARY_FILES, write_thesis_evidence


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_paths(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not is_nonempty_file(path):
            errors.append(f"Missing or empty: {path.as_posix()}")
    return errors


def warn_missing_optional(paths: Iterable[Path]) -> list[str]:
    warnings: list[str] = []
    for path in paths:
        if not is_nonempty_file(path):
            warnings.append(f"Optional legacy artifact missing: {path.as_posix()}")
    return warnings


def validate_summary_contracts(
    output_dir: Path,
    *,
    required_claim_ids: Iterable[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    loaded: dict[str, dict] = {}
    required_claim_set = {str(claim_id) for claim_id in (required_claim_ids or [])}
    for file_name in SUMMARY_FILES:
        path = output_dir / file_name
        if not is_nonempty_file(path):
            errors.append(f"Missing or empty thesis evidence summary: {path.as_posix()}")
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            errors.append(f"Unreadable JSON summary {path.as_posix()}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"Summary must be a JSON object: {path.as_posix()}")
            continue
        loaded[file_name] = payload

    claim_matrix = loaded.get("claim_matrix.json") or {}
    summary = loaded.get("scientific_validation_summary.json") or {}
    claims = claim_matrix.get("claims")
    if isinstance(claims, list):
        if required_claim_set:
            claims_by_id = {str(claim.get("claim_id")): claim for claim in claims}
            missing_claims = sorted(required_claim_set - set(claims_by_id))
            if missing_claims:
                errors.append(f"Required thesis claim(s) missing: {', '.join(missing_claims)}")
            unsafe_required = [
                claim_id
                for claim_id in sorted(required_claim_set & set(claims_by_id))
                if not bool((claims_by_id.get(claim_id) or {}).get("thesis_safe"))
            ]
            if unsafe_required:
                errors.append(f"Required thesis claim(s) unsafe: {', '.join(unsafe_required)}")
        else:
            unsafe = [claim.get("claim_id") for claim in claims if not bool(claim.get("thesis_safe"))]
            if unsafe:
                errors.append(f"Unsafe thesis claims present: {', '.join(map(str, unsafe))}")
    if isinstance(summary, dict):
        if not required_claim_set:
            freeze = summary.get("canonical_freeze") or {}
            if not bool(freeze.get("thesis_safe")):
                errors.append("Canonical freeze / coverage is not thesis-safe")
            failure_modes = summary.get("failure_modes") or {}
            if int(failure_modes.get("count", 0) or 0) > 0:
                errors.append(
                    f"Scientific validation reports {failure_modes.get('count')} failure mode(s)"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate required thesis/release artifacts for a release folder."
    )
    parser.add_argument(
        "--release-dir",
        default="thesis_release/2026-02-26",
        help="Release directory to validate (default: %(default)s)",
    )
    parser.add_argument(
        "--no-registry-sync",
        action="store_true",
        help="Skip automatic RESULTS.md registry refresh before validation.",
    )
    parser.add_argument(
        "--runs-dir",
        default="outputs/experiments/runs",
        help="Experiment runs directory for thesis evidence discovery.",
    )
    parser.add_argument(
        "--methods",
        default="METHODS.md",
        help="Canonical methods document path.",
    )
    parser.add_argument(
        "--results-md",
        default="RESULTS.md",
        help="Results document used for run-to-claim traceability checks.",
    )
    parser.add_argument(
        "--evidence-output-dir",
        default="outputs/thesis_validation/latest",
        help="Where to emit the thesis evidence summary bundle.",
    )
    parser.add_argument(
        "--run-id-allowlist",
        default=None,
        help="Optional newline-delimited run-id allowlist for focused evidence validation.",
    )
    parser.add_argument(
        "--control-metric-basis",
        choices=["auto", "direct", "comprehensive"],
        default="auto",
        help="Control-metric basis selection passed through to the thesis evidence builder.",
    )
    parser.add_argument(
        "--required-claim",
        action="append",
        default=None,
        help=(
            "Validate only the named claim(s) as thesis-safe. Repeatable. "
            "When omitted, the validator keeps the strict release-wide claim gate."
        ),
    )
    args = parser.parse_args()

    if not args.no_registry_sync:
        builder = Path("scripts/build_results_registry.py")
        if builder.exists():
            print("Syncing RESULTS.md registry from manifests...")
            res = subprocess.run([sys.executable, str(builder)], capture_output=True, text=True)
            if res.returncode != 0:
                print("VALIDATION FAILED")
                print("- Registry sync failed")
                if res.stdout.strip():
                    print(res.stdout.strip())
                if res.stderr.strip():
                    print(res.stderr.strip())
                return 1
            if res.stdout.strip():
                print(res.stdout.strip())

    release_dir = Path(args.release_dir)

    required = [
        Path(".gitignore"),
        Path(args.methods),
        Path(args.results_md),
    ]
    if release_dir.exists():
        required.append(release_dir / "README.md")

    errors = validate_paths(required)

    optional_legacy = [
        Path("MAINTAINABILITY_PLAN.md"),
        Path("notes/physarum_walk_update_report.txt"),
        Path("notes/test_write_shell.txt"),
    ]
    warnings = warn_missing_optional(optional_legacy)

    root_should_be_absent = [
        Path("physarum_walk_update_report.txt"),
        Path("test_write_shell.txt"),
    ]
    for root_file in root_should_be_absent:
        if root_file.exists():
            warnings.append(
                f"Legacy artifact still lives in repository root: {root_file.as_posix()}"
            )

    evidence_output_dir = Path(args.evidence_output_dir)
    try:
        write_thesis_evidence(
            runs_dir=Path(args.runs_dir),
            methods_path=Path(args.methods),
            results_path=Path(args.results_md),
            out_dir=evidence_output_dir,
            run_id_allowlist_path=Path(args.run_id_allowlist) if args.run_id_allowlist else None,
            control_metric_basis=args.control_metric_basis,
        )
        errors.extend(
            validate_summary_contracts(
                evidence_output_dir,
                required_claim_ids=args.required_claim,
            )
        )
    except Exception as exc:
        errors.append(f"Unexpected thesis evidence validation failure: {exc}")

    if errors:
        print("VALIDATION FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("VALIDATION PASSED")
    for path in required:
        print(f"- OK: {path.as_posix()}")
    for warning in warnings:
        print(f"- WARN: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
