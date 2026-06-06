#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.verification.open_hypothesis_suite import (  # noqa: E402
    OpenHypothesisConfig,
    run_open_hypothesis_suite,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the nine remaining publication hypotheses from existing ledger artifacts.",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "thesis_validation" / "open_hypothesis_suite_latest",
    )
    parser.add_argument("--claim-matrix", type=Path)
    parser.add_argument("--recentering-summary", type=Path)
    parser.add_argument("--track4-observer-state-summary", type=Path)
    parser.add_argument("--property-theft-transport-summary", type=Path)
    parser.add_argument("--independent-label-summary", type=Path)
    parser.add_argument("--prompt-invariance-summary", type=Path)
    parser.add_argument("--track3-density-validation", type=Path)
    parser.add_argument("--visualization-validation", type=Path)
    parser.add_argument("--min-real-scale-property-records", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_open_hypothesis_suite(
        OpenHypothesisConfig(
            repo_root=args.repo_root.resolve(),
            output_dir=args.output_dir,
            claim_matrix_path=args.claim_matrix,
            recentering_summary_path=args.recentering_summary,
            track4_observer_state_summary_path=args.track4_observer_state_summary,
            property_theft_transport_summary_path=args.property_theft_transport_summary,
            independent_label_summary_path=args.independent_label_summary,
            prompt_invariance_summary_path=args.prompt_invariance_summary,
            track3_density_validation_path=args.track3_density_validation,
            visualization_validation_path=args.visualization_validation,
            min_real_scale_property_records=args.min_real_scale_property_records,
        )
    )
    print(json.dumps(payload["status_counts"], indent=2, sort_keys=True))
    print(args.output_dir / "open_hypothesis_matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
