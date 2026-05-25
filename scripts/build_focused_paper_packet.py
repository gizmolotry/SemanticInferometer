#!/usr/bin/env python3
"""Build a compact review packet from a focused thesis-evidence bundle.

The packet is intentionally capped at 10 files so it can be handed to other
agents/reviewers without dragging along historical experiment debris.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "outputs" / "thesis_validation" / "focused_paper_evidence_20260522"
DEFAULT_TRACK4_BASIS_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "track4_focused_basis_validation"
    / "matched500_repair3_20260522"
    / "track4_focused_basis_validation_summary.json"
)
DEFAULT_TRACK4_TERRAIN_DIAGNOSTICS_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "track4_terrain_semantics_diagnostics"
    / "matched500_soft_terrain_20260525"
    / "track4_terrain_semantics_diagnostics_summary.json"
)
DEFAULT_PACKET_ROOT = REPO_ROOT / "outputs" / "review_packets"
CORE_PACKET_FILES = (
    "scientific_validation_summary.json",
    "claim_matrix.json",
    "paper_claim_profile.json",
    "unsafe_claim_strategy.json",
    "variance_separation_summary.json",
    "ablation_matrix.json",
    "observer_relativity_summary.json",
    "track4_observer_state_action_summary.json",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_load_json(path: Path) -> Dict[str, Any]:
    return _load_json(path) if path.exists() else {}


def _claim_points(claim_matrix: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    points: Dict[str, Dict[str, Any]] = {}
    for claim in claim_matrix.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id:
            continue
        points[claim_id] = {
            "thesis_safe": bool(claim.get("thesis_safe")),
            "pass": bool(claim.get("pass")),
            "point_estimate": claim.get("point_estimate"),
            "effect_direction": claim.get("effect_direction"),
            "claim_scope": claim.get("claim_scope"),
        }
    return points


def build_review_digest(
    *,
    evidence_dir: Path,
    copied_files: Sequence[Path],
    track4_basis_summary: Optional[Path] = None,
    track4_terrain_diagnostics_summary: Optional[Path] = None,
) -> Dict[str, Any]:
    scientific = _safe_load_json(evidence_dir / "scientific_validation_summary.json")
    claim_matrix = _safe_load_json(evidence_dir / "claim_matrix.json")
    profile = _safe_load_json(evidence_dir / "paper_claim_profile.json")
    track4 = _safe_load_json(evidence_dir / "track4_observer_state_action_summary.json")
    variance = _safe_load_json(evidence_dir / "variance_separation_summary.json")
    terrain_diagnostics = (
        _safe_load_json(Path(track4_terrain_diagnostics_summary))
        if track4_terrain_diagnostics_summary is not None and Path(track4_terrain_diagnostics_summary).exists()
        else {}
    )
    terrain_soft = {}
    if track4_terrain_diagnostics_summary is not None and Path(track4_terrain_diagnostics_summary).exists():
        terrain_soft = _safe_load_json(Path(track4_terrain_diagnostics_summary).parent / "soft_terrain_work_coupling.json")
    claims = claim_matrix.get("claims", []) if isinstance(claim_matrix.get("claims"), list) else []
    track4_claim = track4.get("claim_evaluation") if isinstance(track4.get("claim_evaluation"), dict) else {}
    core_claims = [
        row.get("claim_id")
        for row in profile.get("core_claims", []) or []
        if isinstance(row, dict)
    ]
    core_claim_set = {str(claim_id) for claim_id in core_claims if claim_id}
    return {
        "packet_schema_version": "1.0",
        "packet_type": "focused_paper_evidence",
        "source_bundle": str(evidence_dir),
        "input_selection": scientific.get("input_selection"),
        "semantic_signal_interpretation": scientific.get("semantic_signal_interpretation"),
        "publication_ready": bool(profile.get("publication_ready", False)),
        "publication_scope": "focused_core_claim_profile",
        "core_claims": core_claims,
        "blocked_core_claims": [
            row.get("claim_id")
            for row in profile.get("blocked_core_claims", []) or []
            if isinstance(row, dict)
        ],
        "supported_exploratory_claims": [
            str(claim.get("claim_id"))
            for claim in claims
            if (
                isinstance(claim, dict)
                and bool(claim.get("pass"))
                and bool(claim.get("thesis_safe"))
                and bool(str(claim.get("claim_id") or ""))
                and str(claim.get("claim_id") or "") not in core_claim_set
            )
        ],
        "unsafe_claims": [
            claim.get("claim_id")
            for claim in claims
            if isinstance(claim, dict) and not bool(claim.get("thesis_safe"))
        ],
        "claim_points": _claim_points(claim_matrix),
        "track4_observer_state_action": {
            "source_path": track4.get("source_path"),
            "selection_policy": track4.get("selection_policy"),
            "candidate_inventory": track4.get("candidate_inventory"),
            "claim_evaluation": track4_claim,
            "scoped_supported_findings": track4.get("scoped_supported_findings", []),
        },
        "variance_separation": {
            "primary_basis": variance.get("primary_basis"),
            "pass": variance.get("pass"),
            "thesis_safe": variance.get("thesis_safe"),
            "mean_primary_abs_log_ratio": variance.get("mean_primary_abs_log_ratio"),
            "effect_direction": variance.get("effect_direction"),
        },
        "track4_basis_summary": str(track4_basis_summary) if track4_basis_summary else None,
        "track4_terrain_semantics": {
            "source_path": str(track4_terrain_diagnostics_summary)
            if track4_terrain_diagnostics_summary and Path(track4_terrain_diagnostics_summary).exists()
            else None,
            "claim_boundary": terrain_diagnostics.get("claim_boundary", {}),
            "soft_terrain": {
                "evidence_basis": terrain_soft.get("evidence_basis"),
                "real_soft_work_coupling_supported": terrain_soft.get("real_soft_work_coupling_supported"),
                "pooled_soft_terrain_specificity_supported": terrain_soft.get("pooled_soft_terrain_specificity_supported"),
                "matched_soft_terrain_specificity_supported": terrain_soft.get("matched_soft_terrain_specificity_supported"),
                "matched_cell_pass_rate": (
                    (terrain_soft.get("matched_cell_specificity") or {}).get("matched_cell_pass_rate")
                    if isinstance(terrain_soft.get("matched_cell_specificity"), dict)
                    else None
                ),
                "supporting_cell_count": (
                    (terrain_soft.get("matched_cell_specificity") or {}).get("supporting_cell_count")
                    if isinstance(terrain_soft.get("matched_cell_specificity"), dict)
                    else None
                ),
                "usable_matched_cell_count": (
                    (terrain_soft.get("matched_cell_specificity") or {}).get("usable_matched_cell_count")
                    if isinstance(terrain_soft.get("matched_cell_specificity"), dict)
                    else None
                ),
                "median_excess_corr_real_minus_control": (
                    (terrain_soft.get("matched_cell_specificity") or {}).get("median_excess_corr_real_minus_control")
                    if isinstance(terrain_soft.get("matched_cell_specificity"), dict)
                    else None
                ),
            },
            "interpretation": (
                "Digest-only Track 4 terrain diagnostic. This does not promote hard terrain ontology "
                "into the core paper profile and does not consume an extra packet file."
            ),
        },
        "packet_files": [path.name for path in copied_files],
    }


def _copy_existing(paths: Iterable[Path], out_dir: Path) -> List[Path]:
    copied: List[Path] = []
    for path in paths:
        if not path.exists():
            continue
        target = out_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _missing_paths(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if not path.exists()]


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_packet(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    out_dir: Optional[Path] = None,
    track4_basis_summary: Optional[Path] = DEFAULT_TRACK4_BASIS_SUMMARY,
    track4_terrain_diagnostics_summary: Optional[Path] = DEFAULT_TRACK4_TERRAIN_DIAGNOSTICS_SUMMARY,
    max_files: int = 10,
    allow_missing_core_files: bool = False,
) -> Dict[str, Any]:
    evidence_dir = Path(evidence_dir)
    if out_dir is None:
        out_dir = DEFAULT_PACKET_ROOT / f"{evidence_dir.name}_MAX{max_files}"
    out_dir = Path(out_dir)

    source_files = [evidence_dir / filename for filename in CORE_PACKET_FILES]
    missing_core = _missing_paths(source_files)
    if missing_core and not allow_missing_core_files:
        names = ", ".join(str(path) for path in missing_core)
        raise FileNotFoundError(f"Focused evidence bundle is missing required packet files: {names}")
    if track4_basis_summary is not None and Path(track4_basis_summary).exists():
        source_files.append(Path(track4_basis_summary))
    existing_source_files = [path for path in source_files if path.exists()]
    predicted_file_count = len(existing_source_files) + 1
    if predicted_file_count > max_files:
        names = ", ".join([*(path.name for path in existing_source_files), "review_digest.json"])
        raise RuntimeError(f"Packet has {predicted_file_count} files, above max_files={max_files}: {names}")

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_existing(source_files, out_dir)
    digest = build_review_digest(
        evidence_dir=evidence_dir,
        copied_files=copied,
        track4_basis_summary=track4_basis_summary if track4_basis_summary and Path(track4_basis_summary).exists() else None,
        track4_terrain_diagnostics_summary=(
            track4_terrain_diagnostics_summary
            if track4_terrain_diagnostics_summary and Path(track4_terrain_diagnostics_summary).exists()
            else None
        ),
    )
    digest["missing_core_packet_files"] = [path.name for path in missing_core]
    digest_path = out_dir / "review_digest.json"
    copied_with_digest = [*copied, digest_path]
    digest["packet_files"] = [path.name for path in copied_with_digest]
    digest["packet_file_count"] = len(copied_with_digest)
    digest["packet_file_fingerprints"] = [_file_fingerprint(path) for path in copied]
    digest_tmp_path = digest_path.with_suffix(digest_path.suffix + ".tmp")
    digest_tmp_path.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")
    digest_tmp_path.replace(digest_path)
    return {
        "packet_dir": str(out_dir),
        "file_count": len(copied_with_digest),
        "files": [path.name for path in copied_with_digest],
        "digest_path": str(digest_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--track4-basis-summary", type=Path, default=DEFAULT_TRACK4_BASIS_SUMMARY)
    parser.add_argument("--no-track4-basis-summary", action="store_true")
    parser.add_argument(
        "--track4-terrain-diagnostics-summary",
        type=Path,
        default=DEFAULT_TRACK4_TERRAIN_DIAGNOSTICS_SUMMARY,
    )
    parser.add_argument("--no-track4-terrain-diagnostics", action="store_true")
    parser.add_argument(
        "--allow-missing-core-files",
        action="store_true",
        help="Build a diagnostic packet even if required focused evidence files are absent.",
    )
    parser.add_argument("--max-files", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_packet(
        evidence_dir=args.evidence_dir,
        out_dir=args.out_dir,
        track4_basis_summary=None if args.no_track4_basis_summary else args.track4_basis_summary,
        track4_terrain_diagnostics_summary=(
            None if args.no_track4_terrain_diagnostics else args.track4_terrain_diagnostics_summary
        ),
        max_files=args.max_files,
        allow_missing_core_files=bool(args.allow_missing_core_files),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
