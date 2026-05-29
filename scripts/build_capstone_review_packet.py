#!/usr/bin/env python3
"""Build a compact review packet for external capstone/methods reviewers."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "outputs" / "thesis_validation" / "focused" / "publication_profile_with_synthetic_nmi"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "review_packets"
DEFAULT_CAPSTONE_PDF = (
    REPO_ROOT
    / "Aligned Perspectival Kernel Projections and Bias Reification - FYPAndrewClair22184813 (1).pdf"
)

EVIDENCE_FILES = (
    "paper_claim_profile.json",
    "paper_claim_profile.csv",
    "scientific_validation_summary.json",
    "claim_matrix.json",
    "variance_separation_summary.json",
    "kernel_signal_summary.json",
    "observer_relativity_summary.json",
    "observer_recenter_summary.json",
    "track4_traversal_summary.json",
    "ablation_matrix.json",
    "unsafe_claim_strategy.json",
    "track4_basis_comparison.json",
    "track4_basis_comparison.csv",
    "paper_basis_comparison.json",
    "paper_basis_comparison.csv",
    "paper_metric_signal_cartography.csv",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
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


def _claim_rows(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = profile.get("tables") if isinstance(profile, dict) else {}
    rows = tables.get("claim_profile") if isinstance(tables, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _find_claim(rows: Iterable[Dict[str, Any]], claim_id: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("claim_id") == claim_id:
            return row
    return None


def _markdown_claim_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Claim | Status | Thesis Safe | Point Estimate | Direction |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {claim} | {status} | {safe} | {point} | {direction} |".format(
                claim=row.get("claim_id", "unknown"),
                status=row.get("paper_status", "unknown"),
                safe=_fmt(row.get("thesis_safe"), digits=0),
                point=_fmt(row.get("point_estimate")),
                direction=row.get("effect_direction") or "n/a",
            )
        )
    return "\n".join(lines)


def _copy_artifacts(evidence_dir: Path, packet_dir: Path, *, copy_pdf: bool, capstone_pdf: Path) -> List[Dict[str, str]]:
    copied: List[Dict[str, str]] = []
    evidence_out = packet_dir / "evidence"
    evidence_out.mkdir(parents=True, exist_ok=True)
    for file_name in EVIDENCE_FILES:
        source = evidence_dir / file_name
        if not source.exists():
            copied.append({"name": file_name, "status": "missing", "source": str(source), "packet_path": ""})
            continue
        target = evidence_out / file_name
        shutil.copy2(source, target)
        copied.append({"name": file_name, "status": "copied", "source": str(source), "packet_path": str(target)})

    if copy_pdf and capstone_pdf.exists():
        target = packet_dir / capstone_pdf.name
        shutil.copy2(capstone_pdf, target)
        copied.append({"name": capstone_pdf.name, "status": "copied", "source": str(capstone_pdf), "packet_path": str(target)})
    elif capstone_pdf.exists():
        copied.append({"name": capstone_pdf.name, "status": "referenced", "source": str(capstone_pdf), "packet_path": ""})
    else:
        copied.append({"name": capstone_pdf.name, "status": "missing", "source": str(capstone_pdf), "packet_path": ""})
    return copied


def build_packet(evidence_dir: Path, out_dir: Path, capstone_pdf: Path, *, copy_pdf: bool) -> Dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    out_dir = out_dir.resolve()
    capstone_pdf = capstone_pdf.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_json(evidence_dir / "paper_claim_profile.json")
    scientific = _load_json(evidence_dir / "scientific_validation_summary.json")
    variance = _load_json(evidence_dir / "variance_separation_summary.json")
    kernel = _load_json(evidence_dir / "kernel_signal_summary.json")
    track4 = _load_json(evidence_dir / "track4_traversal_summary.json")
    recenter = _load_json(evidence_dir / "observer_recenter_summary.json")
    ablation = _load_json(evidence_dir / "ablation_matrix.json")
    rows = _claim_rows(profile)
    headline = profile.get("headline_findings") or {}
    synthetic = scientific.get("synthetic_recoverability") or {}
    semantic = scientific.get("semantic_signal_interpretation") or {}
    track4_agg = track4.get("aggregate") or {}
    copied = _copy_artifacts(evidence_dir, out_dir, copy_pdf=copy_pdf, capstone_pdf=capstone_pdf)

    readme = f"""# Semantic Interferometer Capstone Review Packet

Generated: {datetime.now(timezone.utc).isoformat()}

This packet is for external agents/reviewers evaluating the capstone paper against the current codebase evidence. The goal is not to rubber-stamp the paper. The goal is to separate publishable claims, secondary support, speculative mechanisms, and failure modes.

## One-Sentence Project Claim

The Semantic Interferometer is an observer-conditioned NLI geometry probe: it tests whether controlled contrastive hypotheses deform article manifolds in measurable, reproducible ways rather than assigning a single neutral bias label.

## Current Evidence Spine

- Synthetic recoverability: NMI `{_fmt(synthetic.get("mean_nmi"))}` +/- `{_fmt(synthetic.get("std_nmi"))}`, ARI `{_fmt(synthetic.get("mean_ari"))}` +/- `{_fmt(synthetic.get("std_ari"))}`.
- Procrustes control separation: `{_fmt((_find_claim(rows, "procrustes_control_separation") or {}).get("point_estimate"))}`.
- Variance separation: primary basis `{variance.get("primary_basis", "n/a")}`, mean abs log ratio `{_fmt(variance.get("mean_primary_abs_log_ratio"))}`.
- Direct payload sensitivity: mean abs log ratio `{_fmt(variance.get("mean_direct_abs_log_ratio"))}`.
- Observer relativity: mean coord delta `{_fmt(headline.get("observer_relativity_mean_coord_delta"))}`, max rotation `{_fmt(headline.get("observer_relativity_max_rotation_deg"))}` deg.
- Observer recenter ledger: status `{recenter.get("status", "n/a")}`, ok `{_fmt(recenter.get("ok_count"), 0)}/{_fmt(recenter.get("observer_count"), 0)}`, path starts `{_fmt(recenter.get("path_start_match_observer_count"), 0)}/{_fmt(recenter.get("observer_count"), 0)}`, replay paths `{_fmt(recenter.get("replay_path_observer_count"), 0)}`.
- Track 5 branch coverage: required modes present `{_fmt(ablation.get("required_modes_present"))}`.
- Kernel note: Matern/Student-t superiority supported `{_fmt((kernel.get("student_t_matern_superiority") or {}).get("supported"))}`; best simple-variance kernel `{(kernel.get("student_t_matern_superiority") or {}).get("best_kernel", "n/a")}`.

## Claim Status Table

{_markdown_claim_table(rows)}

## Known Weak Or Unsafe Areas

- Track 4 terrain validity is not yet thesis-safe: terrain-valid pass rate `{_fmt(track4_agg.get("terrain_valid_pass_rate"))}`, real/control gap pass rate `{_fmt(track4_agg.get("real_control_gap_pass_rate"))}`.
- Observer-centered visualization is a ledgered renderer contract, not scientific validation by itself; it proves Dash/MONOLITH replay provenance, not semantic truth.
- Broad all-metric control destruction remains too strong even though Procrustes and integrated variance separation are supported.
- Canonical freeze is not complete for the full paper protocol; the current packet is a focused evidence bundle.
- Matern/Student-t should be treated as a tested kernel condition, not a proven source of signal.

## Reviewer Instructions

Please read the capstone PDF as a methods/measurement paper, not as a finished classifier product. Evaluate whether the validated evidence supports the narrower claim that observer-conditioned NLI manifolds exhibit measurable deformation and recover planted structure.

Primary artifacts are in `evidence/`. If you are running in the same workspace, original source artifacts are also listed in `artifact_manifest.json`.
"""

    prompts = """# External Reviewer Prompts

Use these prompts with another agent or human reviewer.

## Scientific Legibility

1. What is the strongest defensible scientific claim in this packet?
2. Which capstone claims sound broader than the evidence supports?
3. Does the synthetic NMI/ARI result adequately establish a controlled recoverability sanity check?
4. Is the distinction between direct observer payload and comprehensive/integrated geometry clear enough?
5. What would you require before accepting the real/control separation claim?

## Methods Fidelity

1. Does the paper describe Track 1.5 as forward-pass spectral polarity rather than true backprop?
2. Does Track 5 read as Hadamard/conformal-kernel fusion with strict Riemannian as an ablation branch?
3. Are the dimensionalities and model names internally consistent?
4. Does the paper avoid treating Track 4 as validated MCMC terrain physics?

## Publication Strategy

1. Which sections should be core results, appendix, or future work?
2. What table or figure is missing for a reviewer to trust the method?
3. What is the fastest experiment that would most improve credibility?
4. Is this better framed as NLP methodology, computational social science, visualization, or measurement theory?
"""

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "packet_dir": str(out_dir),
        "capstone_pdf": str(capstone_pdf),
        "source_evidence_dir": str(evidence_dir),
        "copied_artifacts": copied,
        "headline_findings": headline,
        "synthetic_recoverability": synthetic,
        "semantic_signal_interpretation": semantic,
        "track4_aggregate": track4_agg,
        "observer_recenter_summary": recenter,
    }

    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "EXTERNAL_REVIEW_PROMPTS.md").write_text(prompts, encoding="utf-8")
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--capstone-pdf", default=str(DEFAULT_CAPSTONE_PDF))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-copy-pdf", action="store_true")
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip writing a .zip archive beside the packet directory.",
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Optional explicit path for the generated packet archive.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / f"capstone_agent_packet_{timestamp}"
    manifest = build_packet(
        Path(args.evidence_dir),
        out_dir,
        Path(args.capstone_pdf),
        copy_pdf=not args.no_copy_pdf,
    )
    print(f"Built capstone review packet: {manifest['packet_dir']}")
    print(f"- README: {Path(manifest['packet_dir']) / 'README.md'}")
    print(f"- Review prompts: {Path(manifest['packet_dir']) / 'EXTERNAL_REVIEW_PROMPTS.md'}")
    print(f"- Manifest: {Path(manifest['packet_dir']) / 'artifact_manifest.json'}")
    if not args.no_zip:
        zip_path = write_packet_zip(Path(manifest["packet_dir"]), Path(args.zip_path) if args.zip_path else None)
        print(f"- Zip archive: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
