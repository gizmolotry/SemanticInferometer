#!/usr/bin/env python3
"""Audit whether Track 2 is failing as foundation or only as a Track 4 basis.

This is an evidence-layer audit. It consumes existing thesis and Track 4
summary artifacts; it does not rerun DeBERTa or rebuild manifolds.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = (
    REPO_ROOT
    / "outputs"
    / "thesis_validation"
    / "focused"
    / "publication_profile_with_synthetic_nmi"
)
DEFAULT_TRACK4_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "track4_focused_basis_validation"
    / "seed42_kernel_slice_20260519_033359"
    / "track4_focused_basis_validation_summary.json"
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    return value_f if value_f == value_f and abs(value_f) != float("inf") else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _rate(flags: Iterable[Any]) -> float:
    vals = [bool(flag) for flag in flags]
    return float(sum(1 for value in vals if value) / len(vals)) if vals else 0.0


def _verdict(status: str, supported: bool, *, risk: str, detail: str) -> Dict[str, Any]:
    return {
        "status": status,
        "supported": bool(supported),
        "risk": risk,
        "detail": detail,
    }


def integrated_geometry_signal(evidence_dir: Path) -> Dict[str, Any]:
    variance = _load_json(Path(evidence_dir) / "variance_separation_summary.json")
    if not variance:
        return _verdict(
            "NO_DATA",
            False,
            risk="HIGH",
            detail="variance_separation_summary.json missing",
        )
    supported = bool(variance.get("primary_pass") or variance.get("pass") or variance.get("thesis_safe"))
    required_kernels = variance.get("required_kernels_evaluated") or []
    return {
        **_verdict(
            "SUPPORTED" if supported else "UNSUPPORTED",
            supported,
            risk="LOW" if supported else "HIGH",
            detail=(
                "Integrated/comprehensive geometry separates stochastic controls"
                if supported
                else "Integrated/comprehensive geometry did not pass variance separation"
            ),
        ),
        "primary_basis": variance.get("primary_basis"),
        "primary_metric": variance.get("primary_metric"),
        "mean_primary_abs_log_ratio": _safe_float(variance.get("mean_primary_abs_log_ratio")),
        "required_kernels_evaluated": required_kernels,
        "per_kernel": variance.get("per_kernel") or {},
    }


def track5_fusion_signal(evidence_dir: Path) -> Dict[str, Any]:
    ablation = _load_json(Path(evidence_dir) / "ablation_matrix.json")
    if not ablation:
        return _verdict(
            "NO_DATA",
            False,
            risk="HIGH",
            detail="ablation_matrix.json missing",
        )
    by_mode = ablation.get("by_track5_mode") if isinstance(ablation.get("by_track5_mode"), dict) else {}
    required_present = bool(ablation.get("required_modes_present"))
    mode_pass_rates = {
        mode: _safe_float(metrics.get("pass_rate"))
        for mode, metrics in by_mode.items()
        if isinstance(metrics, dict)
    }
    supported = bool(required_present and by_mode and all((rate or 0.0) > 0.0 for rate in mode_pass_rates.values()))
    return {
        **_verdict(
            "SUPPORTED" if supported else "UNSUPPORTED",
            supported,
            risk="LOW" if supported else "MEDIUM",
            detail=(
                "Track 5 hadamard/riemannian branches are both represented and pass"
                if supported
                else "Track 5 branch coverage or pass rate is incomplete"
            ),
        ),
        "required_modes_present": required_present,
        "by_track5_mode": by_mode,
        "mode_pass_rates": mode_pass_rates,
    }


def synthetic_recoverability_signal(evidence_dir: Path) -> Dict[str, Any]:
    scientific = _load_json(Path(evidence_dir) / "scientific_validation_summary.json")
    synthetic = scientific.get("synthetic_recoverability") if scientific else {}
    if not isinstance(synthetic, dict) or not synthetic:
        return _verdict(
            "NO_DATA",
            False,
            risk="MEDIUM",
            detail="synthetic_recoverability missing from scientific validation summary",
        )
    supported = bool(synthetic.get("thesis_safe") or synthetic.get("pass") or synthetic.get("safe_for_thesis_claim"))
    return {
        **_verdict(
            "SUPPORTED" if supported else "UNSUPPORTED",
            supported,
            risk="LOW" if supported else "MEDIUM",
            detail=(
                "Synthetic planted structure recovery is supported"
                if supported
                else "Synthetic planted structure recovery did not pass"
            ),
        ),
        "mean_nmi": _safe_float(synthetic.get("mean_nmi")),
        "std_nmi": _safe_float(synthetic.get("std_nmi")),
        "mean_ari": _safe_float(synthetic.get("mean_ari")),
        "std_ari": _safe_float(synthetic.get("std_ari")),
    }


def track4_basis_signal(track4_summary_path: Path) -> Dict[str, Any]:
    payload = _load_json(track4_summary_path)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if not rows:
        return _verdict(
            "NO_DATA",
            False,
            risk="MEDIUM",
            detail="Track 4 focused basis summary missing rows",
        )
    by_basis: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_basis.setdefault(str(row.get("basis")), []).append(row)
    basis_summary: Dict[str, Any] = {}
    for basis, basis_rows in sorted(by_basis.items()):
        real_rows = [row for row in basis_rows if row.get("corpus") == "real"]
        control_rows = [row for row in basis_rows if str(row.get("corpus", "")).startswith("control_")]
        basis_summary[basis] = {
            "row_count": len(basis_rows),
            "real_row_count": len(real_rows),
            "control_row_count": len(control_rows),
            "terrain_safe_rate": _rate(row.get("safe_for_thesis_claim") for row in basis_rows),
            "real_terrain_safe_rate": _rate(row.get("safe_for_thesis_claim") for row in real_rows),
            "control_terrain_safe_rate": _rate(row.get("safe_for_thesis_claim") for row in control_rows),
            "real_minus_control_terrain_safe_rate": (
                _rate(row.get("safe_for_thesis_claim") for row in real_rows)
                - _rate(row.get("safe_for_thesis_claim") for row in control_rows)
            )
            if real_rows and control_rows
            else None,
            "mean_basis_probe_score": _mean(row.get("basis_probe_score") for row in basis_rows),
            "real_mean_basis_probe_score": _mean(row.get("basis_probe_score") for row in real_rows),
            "control_mean_basis_probe_score": _mean(row.get("basis_probe_score") for row in control_rows),
        }
    track2 = basis_summary.get("track2", {})
    track2_gap = _safe_float(track2.get("real_minus_control_terrain_safe_rate"))
    track2_score_gap = None
    if track2.get("real_mean_basis_probe_score") is not None and track2.get("control_mean_basis_probe_score") is not None:
        track2_score_gap = float(track2["real_mean_basis_probe_score"] - track2["control_mean_basis_probe_score"])
    supported = bool(track2_gap is not None and track2_gap > 0.10 and (track2_score_gap is None or track2_score_gap > 0.0))
    return {
        **_verdict(
            "SUPPORTED" if supported else "UNSUPPORTED",
            supported,
            risk="MEDIUM",
            detail=(
                "Track 2 works as a Track 4 walker basis in focused real/control diagnostics"
                if supported
                else "Track 2 does not work as a Track 4 walker basis in focused real/control diagnostics"
            ),
        ),
        "basis_summary": basis_summary,
        "track2_real_minus_control_terrain_safe_rate": track2_gap,
        "track2_real_minus_control_score_gap": track2_score_gap,
    }


def direct_track2_necessity_signal(evidence_dir: Path) -> Dict[str, Any]:
    ablation = _load_json(Path(evidence_dir) / "ablation_matrix.json")
    records = ablation.get("records") if isinstance(ablation.get("records"), list) else []
    track2_removal_records = [
        row
        for row in records
        if isinstance(row, dict)
        and str(row.get("ablation") or row.get("component") or row.get("track") or "").lower() in {
            "track2",
            "track_2",
            "remove_track2",
            "track2_removed",
            "no_track2",
        }
    ]
    if not track2_removal_records:
        return {
            **_verdict(
                "UNTESTED",
                False,
                risk="OPEN",
                detail="No direct Track 2 removal/Track2-only ablation artifact was found",
            ),
            "record_count": 0,
            "records": [],
        }
    supported = any(bool(row.get("thesis_safe") or row.get("pass") or row.get("effect_pass")) for row in track2_removal_records)
    return {
        **_verdict(
            "SUPPORTED" if supported else "UNSUPPORTED",
            supported,
            risk="LOW" if supported else "HIGH",
            detail="Direct Track 2 ablation records were found",
        ),
        "record_count": len(track2_removal_records),
        "records": track2_removal_records,
    }


def audit_track2_foundation(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    track4_summary_path: Path = DEFAULT_TRACK4_SUMMARY,
) -> Dict[str, Any]:
    evidence_dir = Path(evidence_dir)
    track4_summary_path = Path(track4_summary_path)
    checks = {
        "integrated_geometry_signal": integrated_geometry_signal(evidence_dir),
        "track5_fusion_signal": track5_fusion_signal(evidence_dir),
        "synthetic_recoverability_signal": synthetic_recoverability_signal(evidence_dir),
        "track4_track2_basis_signal": track4_basis_signal(track4_summary_path),
        "direct_track2_necessity_signal": direct_track2_necessity_signal(evidence_dir),
    }
    foundational_positive = all(
        bool(checks[name].get("supported"))
        for name in (
            "integrated_geometry_signal",
            "track5_fusion_signal",
            "synthetic_recoverability_signal",
        )
    )
    walker_failure = not bool(checks["track4_track2_basis_signal"].get("supported"))
    direct_track2_untested = checks["direct_track2_necessity_signal"].get("status") == "UNTESTED"
    if foundational_positive and walker_failure:
        foundation_status = "NOT_BUSTED_TRACK4_LOCAL_FAILURE"
        thesis_interpretation = (
            "Current evidence does not show the whole system is busted. It shows Track 2 alone is "
            "not a sufficient Track 4 walker basis, while integrated geometry and Track 5 evidence remain positive."
        )
    elif foundational_positive:
        foundation_status = "FOUNDATION_SUPPORTED_BY_CURRENT_EVIDENCE"
        thesis_interpretation = "Current evidence supports the integrated Track 2/Track 5 foundation."
    else:
        foundation_status = "FOUNDATION_AT_RISK"
        thesis_interpretation = (
            "One or more core foundation checks failed or is missing; Track 2/system claims need direct ablation."
        )
    if direct_track2_untested:
        thesis_interpretation += " Direct Track 2 removal/Track2-only ablation is still untested."
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_type": "track2_foundation",
        "evidence_dir": str(evidence_dir),
        "track4_summary_path": str(track4_summary_path),
        "foundation_status": foundation_status,
        "whole_system_busted_by_current_evidence": foundation_status == "FOUNDATION_AT_RISK",
        "track4_track2_basis_failure_is_local": bool(foundational_positive and walker_failure),
        "direct_track2_necessity_tested": not direct_track2_untested,
        "checks": checks,
        "thesis_interpretation": thesis_interpretation,
        "next_required_test": (
            "Run a direct Track 2 ablation matrix: Track2-only, Track1.5-only, Track2+Track1.5, full fusion, "
            "and no-Track2 replacement across real/control/synthetic kernels."
        ),
    }


def _write_csv(path: Path, checks: Mapping[str, Mapping[str, Any]]) -> None:
    fieldnames = ["check_id", "status", "supported", "risk", "detail"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for check_id, check in checks.items():
            writer.writerow(
                {
                    "check_id": check_id,
                    "status": check.get("status"),
                    "supported": check.get("supported"),
                    "risk": check.get("risk"),
                    "detail": check.get("detail"),
                }
            )


def write_audit(report: Mapping[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "track2_foundation_audit.json"
    csv_path = output_dir / "track2_foundation_checks.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(csv_path, report.get("checks", {}))
    return {"json": str(json_path), "csv": str(csv_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--track4-summary", type=Path, default=DEFAULT_TRACK4_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "track2_foundation_audit")
    args = parser.parse_args(argv)
    report = audit_track2_foundation(
        evidence_dir=args.evidence_dir,
        track4_summary_path=args.track4_summary,
    )
    written = write_audit(report, args.output_dir)
    print(json.dumps({**written, "foundation_status": report["foundation_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
