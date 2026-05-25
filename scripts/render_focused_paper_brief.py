#!/usr/bin/env python3
"""Render a concise Markdown brief from a focused paper review packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_DIR = REPO_ROOT / "outputs" / "review_packets" / "focused_paper_evidence_20260522_MAX10"
DEFAULT_OUT_PATH = REPO_ROOT / "paper" / "focused_paper_claim_brief_20260522.md"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _claim_line(claim_id: str, claim_points: Dict[str, Any]) -> str:
    payload = claim_points.get(claim_id) if isinstance(claim_points.get(claim_id), dict) else {}
    return (
        f"- `{claim_id}`: point=`{_fmt(payload.get('point_estimate'))}`, "
        f"direction=`{payload.get('effect_direction') or 'n/a'}`"
    )


def render_brief(packet_dir: Path = DEFAULT_PACKET_DIR, out_path: Path = DEFAULT_OUT_PATH) -> Path:
    digest_path = Path(packet_dir) / "review_digest.json"
    digest = _load_json(digest_path)
    claim_points = digest.get("claim_points") if isinstance(digest.get("claim_points"), dict) else {}
    track4 = digest.get("track4_observer_state_action") if isinstance(digest.get("track4_observer_state_action"), dict) else {}
    track4_claim = track4.get("claim_evaluation") if isinstance(track4.get("claim_evaluation"), dict) else {}
    terrain = digest.get("track4_terrain_semantics") if isinstance(digest.get("track4_terrain_semantics"), dict) else {}
    terrain_boundary = terrain.get("claim_boundary") if isinstance(terrain.get("claim_boundary"), dict) else {}
    soft_terrain = terrain.get("soft_terrain") if isinstance(terrain.get("soft_terrain"), dict) else {}
    variance = digest.get("variance_separation") if isinstance(digest.get("variance_separation"), dict) else {}
    semantic = digest.get("semantic_signal_interpretation") if isinstance(digest.get("semantic_signal_interpretation"), dict) else {}

    lines = [
        "# Focused Paper Claim Brief",
        "",
        "This brief is generated from the focused evidence packet. It is a paper-scope map, not a claim that every historical thesis artifact is safe.",
        "",
        "## Scope",
        "",
        f"- Source packet: `{packet_dir}`",
        f"- Publication ready: `{digest.get('publication_ready')}`",
        f"- Publication scope: `{digest.get('publication_scope')}`",
        f"- Semantic status: `{semantic.get('status')}`",
        f"- Paper profile status: `{semantic.get('paper_profile_status')}`",
        f"- Scope warning: {semantic.get('paper_profile_scope_warning') or 'n/a'}",
        "",
    ]
    if not bool(digest.get("publication_ready")):
        lines.extend(
            [
                "## Publication Warning",
                "",
                "- `publication_ready` is false. Treat this packet as diagnostic until blocked core claims are resolved.",
                "",
            ]
        )
    blocked_core_claims = [str(claim_id) for claim_id in digest.get("blocked_core_claims", []) or []]
    if blocked_core_claims:
        lines.extend(["## Blocked Core Claims", ""])
        lines.extend(_claim_line(claim_id, claim_points) for claim_id in blocked_core_claims)
        lines.append("")
    lines.extend(["## Core Supported Claims", ""])
    core_claims = [str(claim_id) for claim_id in digest.get("core_claims", []) or []]
    lines.extend(_claim_line(claim_id, claim_points) for claim_id in core_claims)
    supported_exploratory_claims = [
        str(claim_id) for claim_id in digest.get("supported_exploratory_claims", []) or []
    ]
    if supported_exploratory_claims:
        lines.extend(["", "## Supported Exploratory Claims", ""])
        lines.extend(_claim_line(claim_id, claim_points) for claim_id in supported_exploratory_claims)
    lines.extend([
        "",
        "## Unsafe Or Non-Core Claims",
        "",
    ])
    unsafe_claims = [str(claim_id) for claim_id in digest.get("unsafe_claims", []) or []]
    lines.extend(_claim_line(claim_id, claim_points) for claim_id in unsafe_claims)
    lines.extend([
        "",
        "## Track 4 Boundary",
        "",
        f"- Observer-state action safe: `{track4_claim.get('safe_for_thesis_claim')}`",
        f"- Observer-state action ratio: `{_fmt(track4_claim.get('point_estimate'))}`",
        f"- Hysteresis metric: `{track4_claim.get('hysteresis_gate_metric') or 'n/a'}`",
        f"- Selected source: `{track4.get('source_path') or 'n/a'}`",
        f"- Soft terrain work coupling supported: `{terrain_boundary.get('soft_terrain_work_coupling_supported')}`",
        f"- Matched soft terrain specificity supported: `{terrain_boundary.get('matched_soft_terrain_specificity_supported')}`",
        f"- Pooled soft terrain specificity supported: `{terrain_boundary.get('pooled_soft_terrain_specificity_supported')}`",
        f"- Matched soft terrain cells: `{_fmt(soft_terrain.get('supporting_cell_count'))}` / `{_fmt(soft_terrain.get('usable_matched_cell_count'))}`",
        f"- Targeted event-pair status: `{terrain_boundary.get('targeted_event_pair_status') or 'n/a'}`",
        "- Classic terrain traversal/work claims remain outside the core profile unless their own gates pass.",
        "",
        "## Variance Separation",
        "",
        f"- Primary basis: `{variance.get('primary_basis') or 'n/a'}`",
        f"- Thesis safe: `{variance.get('thesis_safe')}`",
        f"- Mean abs log ratio: `{_fmt(variance.get('mean_primary_abs_log_ratio'))}`",
        f"- Direction: `{variance.get('effect_direction') or 'n/a'}`",
        "",
        "## Packet Files",
        "",
    ])
    lines.extend(f"- `{filename}`" for filename in digest.get("packet_files", []) or [])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_path = render_brief(packet_dir=args.packet_dir, out_path=args.out)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
