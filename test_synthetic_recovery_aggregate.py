from pathlib import Path
import json

from analysis.verification.thesis_evidence import build_thesis_evidence
from thesis_test_support import build_canonical_fixture, write_synthetic_manifest


def test_synthetic_recovery_aggregate_reports_mean_std_and_can_fail(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    synthetic_manifest = fixture["synthetic_manifest"]
    assert synthetic_manifest is not None
    write_synthetic_manifest(
        synthetic_manifest,
        output_dir=synthetic_manifest.parent,
        kernels=["rbf", "laplacian"],
        seeds=[42, 420],
        nmi=0.0,
        ari=0.0,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    summary = payloads["scientific_validation_summary"]["synthetic_recoverability"]
    assert summary["mean_nmi"] == 0.0
    assert summary["mean_ari"] == 0.0
    assert summary["std_nmi"] == 0.0
    assert not summary["thesis_safe"]


def test_synthetic_recovery_recovers_from_summary_when_manifest_not_finalized(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    synthetic_manifest = fixture["synthetic_manifest"]
    assert synthetic_manifest is not None
    write_synthetic_manifest(
        synthetic_manifest,
        output_dir=synthetic_manifest.parent,
        kernels=["rbf", "matern"],
        seeds=[42, 420],
        nmi=0.61,
        ari=0.22,
    )
    synthetic_manifest.write_text(
        json.dumps({"timestamp": "2026-05-04T00:00:00+00:00"}, indent=2),
        encoding="utf-8",
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    summary = payloads["scientific_validation_summary"]["synthetic_recoverability"]
    selection = payloads["scientific_validation_summary"]["input_selection"]
    assert summary["mean_nmi"] == 0.61
    assert summary["mean_ari"] == 0.22
    assert summary["thesis_safe"]
    assert "experiments_20260504_020202" in selection["selected_run_ids"]


def test_synthetic_recovery_is_visible_in_paper_profile_when_safe(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    synthetic_manifest = fixture["synthetic_manifest"]
    assert synthetic_manifest is not None
    write_synthetic_manifest(
        synthetic_manifest,
        output_dir=synthetic_manifest.parent,
        kernels=["rbf", "matern", "imq"],
        seeds=[42, 420, 4200],
        nmi=0.74,
        ari=0.48,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    profile = payloads["paper_claim_profile"]
    headline = profile["headline_findings"]
    synthetic_row = next(
        row for row in profile["tables"]["claim_profile"]
        if row["claim_id"] == "synthetic_recoverability"
    )
    assert headline["synthetic_recoverability_mean_nmi"] == 0.74
    assert headline["synthetic_recoverability_mean_ari"] == 0.48
    assert headline["synthetic_recoverability_thesis_safe"] is True
    assert synthetic_row["paper_status"] == "secondary_supported"
