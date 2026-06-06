#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def run_validation(
    *,
    output_dir: Path,
    atlas_bundle: Optional[Path] = None,
    transport_summary: Optional[Path] = None,
    min_routes: int = 8,
    min_records: int = 40,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas = _load_json(atlas_bundle)
    transport = _load_json(transport_summary)
    routes = atlas.get("routes") if isinstance(atlas.get("routes"), list) else []
    slices = atlas.get("slices") if isinstance(atlas.get("slices"), list) else []
    records = transport.get("records") if isinstance(transport.get("records"), list) else []
    failures: List[str] = []
    if atlas:
        if str(atlas.get("bundle_type") or "") != "observer_atlas_bundle":
            failures.append("atlas_bundle_type_invalid")
        if len(slices) < 2:
            failures.append("fewer_than_two_observer_slices")
        if len(routes) < int(min_routes):
            failures.append("route_count_below_threshold")
    if not atlas and not transport:
        failures.append("missing_atlas_or_transport_artifact")
    if transport:
        if int(transport.get("record_count") or len(records)) < int(min_records):
            failures.append("transport_record_count_below_threshold")
        if float(transport.get("mean_excess_holonomy_action") or 0.0) <= 0.0:
            failures.append("transport_excess_holonomy_not_positive")
    semantic_first = 0
    observer_first = 0
    closed_loop = 0
    for row in records:
        if row.get("semantic_first_action") is not None:
            semantic_first += 1
        if row.get("observer_first_action") is not None:
            observer_first += 1
        if row.get("closed_loop"):
            closed_loop += 1
    if transport and semantic_first <= 0:
        failures.append("semantic_first_routes_missing")
    if transport and observer_first <= 0:
        failures.append("observer_first_routes_missing")
    if transport and closed_loop <= 0:
        failures.append("closed_loop_routes_missing")

    payload = {
        "schema_version": "1.0",
        "summary_type": "visualization_readability_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_type": "software_readability_contract",
        "human_readability_pass": not failures,
        "failure_reasons": failures,
        "atlas_bundle_path": str(atlas_bundle) if atlas_bundle else None,
        "transport_summary_path": str(transport_summary) if transport_summary else None,
        "route_count": len(routes),
        "slice_count": len(slices),
        "transport_record_count": int(transport.get("record_count") or len(records)) if transport else 0,
        "mean_excess_holonomy_action": transport.get("mean_excess_holonomy_action") if transport else None,
        "route_semantics": {
            "semantic_first_route_count": semantic_first,
            "observer_first_route_count": observer_first,
            "closed_loop_route_count": closed_loop,
        },
        "claim_boundary": {
            "safe_claim": "Dash/atlas artifacts contain enough route semantics for a human-readability task packet",
            "unsafe_claim": "actual human participants have already validated the visualization",
        },
    }
    out = output_dir / "visualization_readability_validation.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["artifact"] = str(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-bundle", type=Path)
    parser.add_argument("--transport-summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "visualization_readability_validation" / "latest",
    )
    parser.add_argument("--min-routes", type=int, default=8)
    parser.add_argument("--min-records", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_validation(
        output_dir=args.output_dir,
        atlas_bundle=args.atlas_bundle,
        transport_summary=args.transport_summary,
        min_routes=int(args.min_routes),
        min_records=int(args.min_records),
    )
    print(
        json.dumps(
            {
                "human_readability_pass": payload["human_readability_pass"],
                "failure_reasons": payload["failure_reasons"],
                "artifact": payload["artifact"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(payload["human_readability_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
