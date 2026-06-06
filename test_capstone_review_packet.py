import json
import zipfile
from pathlib import Path

from scripts.build_capstone_review_packet import build_packet, write_packet_zip


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_capstone_review_packet_surfaces_claims_and_copies_pdf(tmp_path: Path):
    evidence_dir = tmp_path / "evidence_source"
    packet_dir = tmp_path / "packet"
    evidence_dir.mkdir()
    capstone_pdf = tmp_path / "capstone.pdf"
    capstone_pdf.write_bytes(b"%PDF-1.4\n")

    _write_json(
        evidence_dir / "paper_claim_profile.json",
        {
            "headline_findings": {
                "observer_relativity_mean_coord_delta": 0.8,
                "observer_relativity_max_rotation_deg": 129.4,
            },
            "tables": {
                "claim_profile": [
                    {
                        "claim_id": "procrustes_control_separation",
                        "paper_status": "core_supported",
                        "thesis_safe": True,
                        "point_estimate": 1.55,
                        "effect_direction": "real_procrustes_gt_controls",
                    },
                    {
                        "claim_id": "track4_traversal_validity",
                        "paper_status": "exploratory_or_unsupported",
                        "thesis_safe": False,
                        "point_estimate": 0.0,
                        "effect_direction": "terrain_correlates_with_traversal",
                    },
                ]
            },
        },
    )
    _write_json(
        evidence_dir / "scientific_validation_summary.json",
        {
            "synthetic_recoverability": {
                "mean_nmi": 0.74,
                "std_nmi": 0.02,
                "mean_ari": 0.48,
                "std_ari": 0.04,
                "thesis_safe": True,
            },
            "semantic_signal_interpretation": {"status": "mixed_positive_not_real_control_safe"},
        },
    )
    _write_json(
        evidence_dir / "variance_separation_summary.json",
        {
            "primary_basis": "comprehensive_results",
            "mean_primary_abs_log_ratio": 0.3,
            "mean_direct_abs_log_ratio": 0.004,
        },
    )
    _write_json(
        evidence_dir / "kernel_signal_summary.json",
        {
            "student_t_matern_superiority": {
                "supported": False,
                "best_kernel": "rbf",
            }
        },
    )
    _write_json(
        evidence_dir / "track4_traversal_summary.json",
        {"aggregate": {"terrain_valid_pass_rate": 0.0, "real_control_gap_pass_rate": 0.0}},
    )
    _write_json(
        evidence_dir / "observer_recenter_summary.json",
        {
            "status": "OK",
            "observer_count": 3,
            "ok_count": 3,
            "path_start_match_observer_count": 3,
            "replay_path_observer_count": 2,
            "z_origin_policy": "xy_origin_preserve_canonical_z",
        },
    )
    _write_json(
        evidence_dir / "ablation_matrix.json",
        {"required_modes_present": True},
    )

    manifest = build_packet(evidence_dir, packet_dir, capstone_pdf, copy_pdf=True)

    readme = (packet_dir / "README.md").read_text(encoding="utf-8")
    assert "NMI `0.740`" in readme
    assert "procrustes_control_separation" in readme
    assert "Observer recenter ledger: status `OK`, ok `3/3`, path starts `3/3`, replay paths `2`" in readme
    assert "Track 4 terrain validity is not yet thesis-safe" in readme
    assert (packet_dir / "evidence" / "observer_recenter_summary.json").exists()
    assert (packet_dir / "EXTERNAL_REVIEW_PROMPTS.md").exists()
    assert (packet_dir / "artifact_manifest.json").exists()
    assert (packet_dir / "capstone.pdf").exists()
    assert manifest["synthetic_recoverability"]["mean_nmi"] == 0.74
    assert manifest["observer_recenter_summary"]["ok_count"] == 3


def test_write_packet_zip_preserves_relative_packet_layout(tmp_path: Path):
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "README.md").write_text("readme", encoding="utf-8")
    (packet_dir / "EXTERNAL_REVIEW_PROMPTS.md").write_text("prompts", encoding="utf-8")
    evidence_dir = packet_dir / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "paper_claim_profile.json").write_text("{}", encoding="utf-8")

    zip_path = write_packet_zip(packet_dir)

    assert zip_path == packet_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == [
            "EXTERNAL_REVIEW_PROMPTS.md",
            "README.md",
            "evidence/paper_claim_profile.json",
        ]
