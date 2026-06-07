#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observer_slice_transport import (  # noqa: E402
    ObserverSliceTransportConfig,
    summarize_observer_slice_transport,
    write_observer_slice_transport_summary,
)


CONCEPTS = (
    "property",
    "theft",
    "rent",
    "extraction",
    "wage_labor",
    "exploitation",
    "ownership",
    "exclusion",
    "contract",
    "coercion",
    "market",
    "violence",
    "land_title",
    "enclosure",
    "profit",
    "surplus",
    "debt",
    "discipline",
    "commons",
    "stewardship",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _slice_coordinates() -> Dict[str, np.ndarray]:
    n = len(CONCEPTS)
    pair_id = np.arange(n, dtype=np.float64) // 2.0
    side = np.asarray([0.0 if idx % 2 == 0 else 1.0 for idx in range(n)], dtype=np.float64)
    slow = pair_id / max(float(pair_id.max()), 1.0)
    # Four observer charts deliberately disagree about whether the paired terms
    # are close semantic neighbors or separated by legal/moral barriers.
    return {
        "anarchist": np.column_stack([slow, 0.12 * side]),
        "liberal_property": np.column_stack([slow, 1.85 * side + 0.10 * slow]),
        "legalist": np.column_stack([slow, 1.20 * side + 0.45 * np.sin(pair_id)]),
        "marxist": np.column_stack([slow, 0.45 * side + 0.75 * slow * side]),
    }


def _null_slices(slices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    base = np.asarray(slices["legalist"], dtype=np.float64)
    return {name: base.copy() for name in slices}


def run_probe(*, output_dir: Path, max_pairs: int = 12) -> Dict[str, Any]:
    slices = _slice_coordinates()
    article_pairs = [(idx, idx + 1) for idx in range(0, min(len(CONCEPTS), int(max_pairs) * 2), 2)]
    density = np.linspace(0.85, 0.35, len(CONCEPTS), dtype=np.float64)
    stress = np.asarray([0.15 if idx % 2 == 0 else 0.75 for idx in range(len(CONCEPTS))], dtype=np.float64)
    summary = summarize_observer_slice_transport(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        null_slices=_null_slices(slices),
        config=ObserverSliceTransportConfig(
            semantic_weight=1.0,
            observer_switch_weight=1.0,
            stress_weight=0.35,
            density_weight=0.20,
        ),
    )
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "probe_name": "property_theft_scaled_observer_slice_transport",
            "concepts": list(CONCEPTS),
            "record_scale_pass": bool(
                int(summary.get("record_count") or 0) >= 40
                and float(summary.get("mean_excess_holonomy_action") or 0.0) > 0.0
            ),
            "claim_boundary": {
                "safe_claim": "scaled observer-slice transport expresses chart-dependent holonomy on a controlled semantic atlas",
                "unsafe_claim": "this is real-corpus ideological accuracy",
            },
        }
    )
    paths = write_observer_slice_transport_summary(summary, output_dir)
    summary["artifacts"] = paths
    Path(paths["summary"]).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "track4_observer_slice_transport_probe" / "property_theft_scaled_latest",
    )
    parser.add_argument("--max-pairs", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_probe(output_dir=args.output_dir, max_pairs=int(args.max_pairs))
    print(
        json.dumps(
            {
                "record_count": payload.get("record_count"),
                "mean_excess_holonomy_action": payload.get("mean_excess_holonomy_action"),
                "record_scale_pass": payload.get("record_scale_pass"),
                "artifact": payload.get("artifacts", {}).get("summary"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(payload.get("record_scale_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
