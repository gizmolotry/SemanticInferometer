#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_proxy_validation import (  # noqa: E402
    summarize_results,
    validate_run_dir,
)


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
    except Exception:
        pass
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_validation(
    *,
    run_dirs: Sequence[Path],
    output_dir: Path,
    permutations: int = 100,
    max_pvalue: float = 0.10,
    allow_proxy_as_independent: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        validate_run_dir(
            run_dir,
            permutations=int(permutations),
            max_pvalue=float(max_pvalue),
        )
        for run_dir in run_dirs
    ]
    source_summary = summarize_results(rows)
    proxy_pass = bool(source_summary.get("thesis_safe"))
    independent = bool(proxy_pass and allow_proxy_as_independent)
    payload = {
        "schema_version": "1.0",
        "summary_type": "independent_label_validation",
        "independent_labels_validated": independent,
        "unit_of_analysis": "article",
        "accuracy_gate_pass": False,
        "agreement_gate_pass": independent,
        "validation_basis": (
            "source_proxy_promoted_for_exact_gate_ablation"
            if allow_proxy_as_independent
            else "source_proxy_not_promoted_to_independent_truth"
        ),
        "source_proxy_summary": source_summary,
        "failure_reasons": [] if independent else [
            "source_proxy_not_allowed_as_independent_label"
            if proxy_pass
            else "source_proxy_validation_failed"
        ],
        "claim_boundary": {
            "safe_claim": "article-level repeated-source/proxy labels form a registered agreement target",
            "unsafe_claim": "source/proxy labels prove real ideological truth without human/audited labels",
        },
    }
    out = output_dir / "independent_label_validation_summary.json"
    out.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["artifact"] = str(out)
    out.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "independent_label_validation" / "latest",
    )
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--max-pvalue", type=float, default=0.10)
    parser.add_argument(
        "--allow-proxy-as-independent",
        action="store_true",
        help="Promote a passing source/proxy agreement diagnostic to the exact open-hypothesis external gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_validation(
        run_dirs=[Path(p) for p in args.run_dir],
        output_dir=args.output_dir,
        permutations=int(args.permutations),
        max_pvalue=float(args.max_pvalue),
        allow_proxy_as_independent=bool(args.allow_proxy_as_independent),
    )
    print(
        json.dumps(
            {
                "independent_labels_validated": payload["independent_labels_validated"],
                "agreement_gate_pass": payload["agreement_gate_pass"],
                "failure_reasons": payload["failure_reasons"],
                "artifact": payload["artifact"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(payload["independent_labels_validated"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
