#!/usr/bin/env python3
"""Build reviewer-facing thesis evidence summaries from completed experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.verification.thesis_evidence import build_thesis_evidence, write_thesis_evidence


def _load_path_list(path: Path) -> list[Path]:
    items: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Path(line))
    return items


def _focused_selection_is_valid(summary: dict) -> tuple[bool, str]:
    selection = summary.get("input_selection") if isinstance(summary, dict) else None
    if not isinstance(selection, dict) or not bool(selection.get("focused_filter_active", False)):
        return True, "focused filter inactive"
    selected = int(selection.get("selected_manifest_count") or 0)
    missing_run_ids = selection.get("missing_requested_run_ids") or []
    missing_manifest_paths = selection.get("missing_requested_manifest_paths") or []
    if selected <= 0:
        return False, "focused evidence selected zero manifests"
    if missing_run_ids:
        return False, f"focused evidence missing requested run ids: {missing_run_ids}"
    if missing_manifest_paths:
        return False, f"focused evidence missing requested manifest paths: {missing_manifest_paths}"
    return True, "focused evidence selection valid"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="outputs/experiments/runs")
    parser.add_argument(
        "--suite-manifest",
        action="append",
        default=None,
        help="Optional manifest path to include in a focused proof bundle. Repeatable.",
    )
    parser.add_argument(
        "--suite-manifest-list",
        default=None,
        help="Optional newline-delimited file of suite manifest paths for a focused proof bundle.",
    )
    parser.add_argument(
        "--synthetic-manifest",
        action="append",
        default=None,
        help="Optional synthetic manifest path to include in a focused proof bundle. Repeatable.",
    )
    parser.add_argument(
        "--synthetic-manifest-list",
        default=None,
        help="Optional newline-delimited file of synthetic manifest paths for a focused proof bundle.",
    )
    parser.add_argument(
        "--run-id-allowlist",
        default=None,
        help="Optional newline-delimited file of run ids to include in a focused proof bundle.",
    )
    parser.add_argument("--methods-path", "--methods", dest="methods_path", default="METHODS.md")
    parser.add_argument("--results-path", "--results-md", dest="results_path", default="RESULTS.md")
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default="outputs/thesis_validation/latest")
    parser.add_argument(
        "--control-metric-basis",
        choices=["auto", "direct", "comprehensive"],
        default="auto",
        help="Read control_metrics basis snapshots when available.",
    )
    parser.add_argument(
        "--print-bundle",
        action="store_true",
        help="Print the scientific_validation_summary payload after writing files.",
    )
    args = parser.parse_args()

    suite_manifest_paths = [Path(item) for item in (args.suite_manifest or [])]
    if args.suite_manifest_list:
        suite_manifest_paths.extend(_load_path_list(Path(args.suite_manifest_list)))

    synthetic_manifest_paths = [Path(item) for item in (args.synthetic_manifest or [])]
    if args.synthetic_manifest_list:
        synthetic_manifest_paths.extend(_load_path_list(Path(args.synthetic_manifest_list)))

    run_id_allowlist_path = Path(args.run_id_allowlist) if args.run_id_allowlist else None

    written = write_thesis_evidence(
        runs_dir=Path(args.runs_dir),
        methods_path=Path(args.methods_path),
        results_path=Path(args.results_path),
        out_dir=Path(args.output_dir),
        run_id_allowlist_path=run_id_allowlist_path,
        suite_manifest_paths=suite_manifest_paths or None,
        synthetic_manifest_paths=synthetic_manifest_paths or None,
        control_metric_basis=args.control_metric_basis,
    )
    payloads = build_thesis_evidence(
        runs_dir=Path(args.runs_dir),
        methods_path=Path(args.methods_path),
        results_path=Path(args.results_path),
        run_id_allowlist_path=run_id_allowlist_path,
        suite_manifest_paths=suite_manifest_paths or None,
        synthetic_manifest_paths=synthetic_manifest_paths or None,
        control_metric_basis=args.control_metric_basis,
    )
    focused_ok, focused_reason = _focused_selection_is_valid(payloads.get("scientific_validation_summary") or {})

    print("Built thesis evidence bundle:")
    for key, target in written.items():
        print(f"- {key}: {target}")
    if args.print_bundle:
        print(json.dumps(payloads["scientific_validation_summary"], indent=2))
    if not focused_ok:
        print(f"[THESIS_EVIDENCE][FAIL] {focused_reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
