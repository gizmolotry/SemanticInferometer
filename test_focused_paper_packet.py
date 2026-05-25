import json
from pathlib import Path

import pytest

from scripts.build_focused_paper_packet import build_packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_minimal_evidence_bundle(root: Path) -> None:
    _write_json(
        root / "scientific_validation_summary.json",
        {
            "input_selection": {"focused_filter_active": True, "selected_run_ids": ["run_a"]},
            "semantic_signal_interpretation": {"status": "mixed_positive_not_real_control_safe"},
        },
    )
    _write_json(
        root / "claim_matrix.json",
        {
            "claims": [
                {
                    "claim_id": "procrustes_control_separation",
                    "pass": True,
                    "thesis_safe": True,
                    "point_estimate": 2.0,
                    "effect_direction": "real_procrustes_gt_controls",
                },
                {
                    "claim_id": "track4_traversal_validity",
                    "pass": False,
                    "thesis_safe": False,
                    "point_estimate": 0.0,
                    "effect_direction": "terrain_unproven",
                },
                {
                    "claim_id": "track4_observer_state_action_separation",
                    "pass": True,
                    "thesis_safe": True,
                    "point_estimate": 3.6,
                    "effect_direction": "real_observer_state_action_gt_controls",
                    "claim_scope": "exploratory_track4",
                },
                {
                    "claim_id": "safe_but_failed_exploratory_claim",
                    "pass": False,
                    "thesis_safe": True,
                    "point_estimate": 0.1,
                    "effect_direction": "insufficient_effect",
                    "claim_scope": "exploratory_negative_control",
                },
                {
                    "claim_id": "",
                    "pass": True,
                    "thesis_safe": True,
                    "point_estimate": 1.0,
                    "effect_direction": "invalid_missing_claim_id",
                    "claim_scope": "exploratory_negative_control",
                },
            ]
        },
    )
    _write_json(
        root / "paper_claim_profile.json",
        {
            "publication_ready": True,
            "core_claims": [{"claim_id": "procrustes_control_separation"}],
            "blocked_core_claims": [],
        },
    )
    _write_json(root / "unsafe_claim_strategy.json", {"recommendations": []})
    _write_json(
        root / "variance_separation_summary.json",
        {
            "primary_basis": "comprehensive_results",
            "pass": True,
            "thesis_safe": True,
            "mean_primary_abs_log_ratio": 0.35,
            "effect_direction": "direction_free_variance_displacement",
        },
    )
    _write_json(root / "ablation_matrix.json", {"records": []})
    _write_json(root / "observer_relativity_summary.json", {"aggregate": {}})
    _write_json(
        root / "track4_observer_state_action_summary.json",
        {
            "source_path": "track4_summary.json",
            "selection_policy": "explicit_summary_path_then_thesis_safe_then_summary_all_then_mtime",
            "candidate_inventory": {"selected_family_key": "matched500"},
            "claim_evaluation": {
                "safe_for_thesis_claim": True,
                "point_estimate": 3.6,
            },
            "scoped_supported_findings": [{"scope_values": ["track2"]}],
        },
    )


def test_build_focused_paper_packet_caps_output_at_ten_files(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    out_dir = tmp_path / "packet"
    track4_basis = tmp_path / "track4_focused_basis_validation_summary.json"
    track4_terrain = tmp_path / "terrain" / "track4_terrain_semantics_diagnostics_summary.json"
    _write_minimal_evidence_bundle(evidence_dir)
    _write_json(track4_basis, {"summary_type": "track4_focused_basis_validation"})
    _write_json(
        track4_terrain,
        {
            "claim_boundary": {
                "soft_terrain_work_coupling_supported": True,
                "matched_soft_terrain_specificity_supported": True,
                "pooled_soft_terrain_specificity_supported": False,
                "targeted_event_pair_status": "NO_DATA",
                "targeted_event_pair_claim_ready": False,
            }
        },
    )
    _write_json(
        track4_terrain.parent / "soft_terrain_work_coupling.json",
        {
            "evidence_basis": "continuous_path_touched_density_stress_membership",
            "real_soft_work_coupling_supported": True,
            "pooled_soft_terrain_specificity_supported": False,
            "matched_soft_terrain_specificity_supported": True,
            "matched_cell_specificity": {
                "matched_cell_pass_rate": 0.778,
                "supporting_cell_count": 7,
                "usable_matched_cell_count": 9,
                "median_excess_corr_real_minus_control": 0.634,
            },
        },
    )

    summary = build_packet(
        evidence_dir=evidence_dir,
        out_dir=out_dir,
        track4_basis_summary=track4_basis,
        track4_terrain_diagnostics_summary=track4_terrain,
        max_files=10,
    )

    assert summary["file_count"] == 10
    assert len(list(out_dir.iterdir())) == 10
    digest = json.loads((out_dir / "review_digest.json").read_text(encoding="utf-8"))
    assert digest["publication_ready"] is True
    assert digest["publication_scope"] == "focused_core_claim_profile"
    assert digest["core_claims"] == ["procrustes_control_separation"]
    assert digest["supported_exploratory_claims"] == ["track4_observer_state_action_separation"]
    assert digest["unsafe_claims"] == ["track4_traversal_validity"]
    assert digest["packet_file_count"] == 10
    assert "review_digest.json" in digest["packet_files"]
    assert "track4_terrain_semantics_diagnostics_summary.json" not in digest["packet_files"]
    assert len(digest["packet_file_fingerprints"]) == 9
    assert {
        "name",
        "bytes",
        "sha256",
    }.issubset(digest["packet_file_fingerprints"][0])
    assert digest["track4_observer_state_action"]["claim_evaluation"]["point_estimate"] == 3.6
    assert digest["variance_separation"]["mean_primary_abs_log_ratio"] == 0.35
    terrain = digest["track4_terrain_semantics"]
    assert terrain["claim_boundary"]["matched_soft_terrain_specificity_supported"] is True
    assert terrain["claim_boundary"]["pooled_soft_terrain_specificity_supported"] is False
    assert terrain["soft_terrain"]["supporting_cell_count"] == 7
    assert terrain["soft_terrain"]["usable_matched_cell_count"] == 9
    assert "does not promote hard terrain ontology" in terrain["interpretation"]


def test_build_focused_paper_packet_rejects_overlarge_packets(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    out_dir = tmp_path / "packet"
    track4_basis = tmp_path / "track4_focused_basis_validation_summary.json"
    _write_minimal_evidence_bundle(evidence_dir)
    _write_json(track4_basis, {"summary_type": "track4_focused_basis_validation"})

    with pytest.raises(RuntimeError, match="above max_files=9"):
        build_packet(
            evidence_dir=evidence_dir,
            out_dir=out_dir,
            track4_basis_summary=track4_basis,
            max_files=9,
        )
    assert not out_dir.exists()


def test_build_focused_paper_packet_rejects_missing_core_files(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="missing required packet files"):
        build_packet(
            evidence_dir=evidence_dir,
            out_dir=tmp_path / "packet",
            track4_basis_summary=None,
        )


def test_build_focused_paper_packet_can_allow_missing_core_files_for_diagnostics(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    summary = build_packet(
        evidence_dir=evidence_dir,
        out_dir=tmp_path / "packet",
        track4_basis_summary=None,
        allow_missing_core_files=True,
    )

    digest = json.loads((tmp_path / "packet" / "review_digest.json").read_text(encoding="utf-8"))
    assert summary["file_count"] == 1
    assert digest["missing_core_packet_files"]
