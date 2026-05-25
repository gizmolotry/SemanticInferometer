#!/usr/bin/env python3
"""Build a compact Track 4 diagnostics packet for external review.

This copies only small, claim-bearing files. It deliberately avoids copying
heavy per-run artifacts such as ``cyclic_paths.npz`` and observer tensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "track4_terrain_semantics_diagnostics"
    / "matched500_soft_terrain_20260525"
)
DEFAULT_NOTE = REPO_ROOT / "notes" / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "review_packets"

PACKET_FILES = (
    "track4_terrain_semantics_diagnostics_summary.json",
    "terrain_construct_validity.json",
    "real_vs_control_terrain_specificity.json",
    "soft_terrain_work_coupling.json",
    "walker_sensitivity_matrix.json",
    "work_decomposition_summary.json",
    "targeted_event_pair_results.json",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def _copy_if_exists(source: Path, target: Path) -> Dict[str, str]:
    if not source.exists():
        return {"name": source.name, "status": "missing", "source": str(source), "packet_path": ""}
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"name": source.name, "status": "copied", "source": str(source), "packet_path": str(target)}


def packet_file_count(packet_dir: Path) -> int:
    return sum(1 for path in Path(packet_dir).rglob("*") if path.is_file())


def build_packet(
    *,
    diagnostic_dir: Path,
    out_dir: Path,
    note_path: Path = DEFAULT_NOTE,
    max_files: int = 10,
) -> Dict[str, Any]:
    diagnostic_dir = Path(diagnostic_dir).resolve()
    out_dir = Path(out_dir).resolve()
    note_path = Path(note_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []
    for file_name in PACKET_FILES:
        copied.append(_copy_if_exists(diagnostic_dir / file_name, out_dir / "evidence" / file_name))
    copied.append(_copy_if_exists(note_path, out_dir / note_path.name))

    summary = _load_json(diagnostic_dir / "track4_terrain_semantics_diagnostics_summary.json")
    construct = _load_json(diagnostic_dir / "terrain_construct_validity.json")
    specificity = _load_json(diagnostic_dir / "real_vs_control_terrain_specificity.json")
    contrast = _load_json(diagnostic_dir / "terrain_contrast_matrix.json")
    work = _load_json(diagnostic_dir / "work_decomposition_summary.json")
    walker = _load_json(diagnostic_dir / "walker_sensitivity_matrix.json")
    soft = _load_json(diagnostic_dir / "soft_terrain_work_coupling.json")
    targeted = _load_json(diagnostic_dir / "targeted_event_pair_results.json")
    claim_boundary = summary.get("claim_boundary", {}) if isinstance(summary, dict) else {}
    top_contrast = (contrast.get("ranked_contrasts") or [{}])[0] if isinstance(contrast, dict) else {}
    matched = soft.get("matched_cell_specificity", {}) if isinstance(soft.get("matched_cell_specificity"), dict) else {}

    readme = f"""# Track 4 Terrain Semantics Diagnostic Packet

Generated: {datetime.now(timezone.utc).isoformat()}

This packet is intentionally small. It contains only claim-bearing Track 4
diagnostic summaries, not heavy route tensors or model artifacts.

## Claim Boundary

- Terrain construct distinctness: `{_fmt(claim_boundary.get("terrain_construct_distinctness"))}`
- Walker sensitivity detected: `{_fmt(claim_boundary.get("walker_sensitivity_detected"))}`
- Native work decomposition available: `{_fmt(claim_boundary.get("native_work_decomposition_available"))}`
- Terrain specificity supported: `{_fmt(claim_boundary.get("terrain_specificity_supported"))}`
- Soft terrain work coupling supported: `{_fmt(claim_boundary.get("soft_terrain_work_coupling_supported"))}`
- Pooled soft terrain specificity supported: `{_fmt(claim_boundary.get("pooled_soft_terrain_specificity_supported"))}`
- Matched soft terrain specificity supported: `{_fmt(claim_boundary.get("matched_soft_terrain_specificity_supported"))}`
- Targeted event-pair status: `{_fmt(claim_boundary.get("targeted_event_pair_status"))}`

## Current Interpretation

Track 4 currently supports a narrow mechanical claim: terrain fields and walker
settings produce measurable traversal telemetry differences. The stronger
matched-cell soft terrain result suggests real data has more barrier/work
coupling than stochastic controls in most matched kernel/seed cells. This is
still not a broad semantic-terrain ontology because pooled controls also exhibit
generic terrain structure and targeted same-event evidence is still missing or
not claim-ready.

## High-Signal Numbers

- Global zone count: `{_fmt(construct.get("zone_count"), 0)}`
- Global work range: `{_fmt(construct.get("work_range"))}`
- Real/control terrain-safe gap: `{_fmt(specificity.get("real_minus_control_terrain_safe_rate"))}`
- Real/control mean-score gap: `{_fmt(specificity.get("real_minus_control_mean_score"))}`
- Real soft barrier/work correlation: `{_fmt((soft.get("real") or {}).get("corr_soft_barrier_mass_work"))}`
- Pooled real/control soft barrier-work excess: `{_fmt(soft.get("real_minus_control_barrier_work_corr"))}`
- Matched soft terrain pass rate: `{_fmt(matched.get("matched_cell_pass_rate"))}`
- Matched supporting cells: `{_fmt(matched.get("supporting_cell_count"), 0)}` / `{_fmt(matched.get("usable_matched_cell_count"), 0)}`
- Median matched excess correlation: `{_fmt(matched.get("median_excess_corr_real_minus_control"))}`
- Top contrast: `{top_contrast.get("contrast", "n/a")}` with mean work gap `{_fmt(top_contrast.get("mean_work_gap_abs"))}`
- Walker score range: `{_fmt(walker.get("score_range"))}`
- Work/path-edge correlation: `{_fmt((work.get("confound_correlations") or {}).get("work_vs_path_edge_count"))}`
- Targeted event-pair result: `{targeted.get("status", "n/a")}`

## Files

The `evidence/` folder contains JSON/CSV summaries. The Markdown note contains
the human-readable interpretation and recommended next experiment.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "packet_dir": str(out_dir),
        "source_diagnostic_dir": str(diagnostic_dir),
        "source_note": str(note_path),
        "max_files": int(max_files),
        "copied_artifacts": copied,
        "claim_boundary": claim_boundary,
        "file_count": packet_file_count(out_dir),
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["file_count"] = packet_file_count(out_dir)
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["file_count"] > int(max_files):
        raise RuntimeError(f"Track 4 diagnostic packet has {manifest['file_count']} files; max_files={max_files}")
    return manifest


def write_packet_zip(packet_dir: Path, zip_path: Optional[Path] = None) -> Path:
    packet_dir = Path(packet_dir).resolve()
    if zip_path is None:
        zip_path = packet_dir.with_suffix(".zip")
    else:
        zip_path = Path(zip_path).resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(packet_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(packet_dir))
    return zip_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--note-path", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--zip-path", type=Path, default=None)
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / f"track4_diagnostic_packet_{timestamp}")
    manifest = build_packet(
        diagnostic_dir=args.diagnostic_dir,
        out_dir=out_dir,
        note_path=args.note_path,
        max_files=args.max_files,
    )
    print(f"Built Track 4 diagnostic packet: {manifest['packet_dir']}")
    print(f"- README: {Path(manifest['packet_dir']) / 'README.md'}")
    print(f"- Manifest: {Path(manifest['packet_dir']) / 'artifact_manifest.json'}")
    print(f"- File count: {manifest['file_count']}/{args.max_files}")
    if not args.no_zip:
        zip_path = write_packet_zip(Path(manifest["packet_dir"]), args.zip_path)
        print(f"- Zip archive: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
