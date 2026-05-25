import json
from pathlib import Path

from scripts.render_focused_paper_brief import render_brief


def test_render_focused_paper_brief_from_digest(tmp_path: Path):
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    digest = {
        "publication_ready": True,
        "publication_scope": "focused_core_claim_profile",
        "semantic_signal_interpretation": {
            "status": "mixed_positive_not_real_control_safe",
            "paper_profile_status": "focused_core_profile_supported",
            "paper_profile_scope_warning": "Focused only.",
        },
        "core_claims": ["procrustes_control_separation"],
        "blocked_core_claims": [],
        "supported_exploratory_claims": ["track4_observer_state_action_separation"],
        "unsafe_claims": ["track4_traversal_validity"],
        "claim_points": {
            "procrustes_control_separation": {
                "point_estimate": 2.5,
                "effect_direction": "real_procrustes_gt_controls",
            },
            "track4_traversal_validity": {
                "point_estimate": 0.2,
                "effect_direction": "terrain_unproven",
            },
            "track4_observer_state_action_separation": {
                "point_estimate": 3.6,
                "effect_direction": "real_observer_state_action_gt_controls",
            },
        },
        "track4_observer_state_action": {
            "source_path": "track4.json",
            "claim_evaluation": {
                "safe_for_thesis_claim": True,
                "point_estimate": 3.6,
                "hysteresis_gate_metric": "mean_calibrated_hysteresis_penalty",
            },
        },
        "track4_terrain_semantics": {
            "claim_boundary": {
                "soft_terrain_work_coupling_supported": True,
                "matched_soft_terrain_specificity_supported": True,
                "pooled_soft_terrain_specificity_supported": False,
                "targeted_event_pair_status": "NO_DATA",
            },
            "soft_terrain": {
                "supporting_cell_count": 7,
                "usable_matched_cell_count": 9,
            },
        },
        "variance_separation": {
            "primary_basis": "comprehensive_results",
            "thesis_safe": True,
            "mean_primary_abs_log_ratio": 0.35,
            "effect_direction": "direction_free_variance_displacement",
        },
        "packet_files": ["review_digest.json"],
    }
    (packet_dir / "review_digest.json").write_text(json.dumps(digest), encoding="utf-8")

    out_path = render_brief(packet_dir=packet_dir, out_path=tmp_path / "brief.md")

    text = out_path.read_text(encoding="utf-8")
    assert "# Focused Paper Claim Brief" in text
    assert "`focused_core_claim_profile`" in text
    assert "`procrustes_control_separation`" in text
    assert "## Supported Exploratory Claims" in text
    assert "`track4_observer_state_action_separation`" in text
    assert "`track4_traversal_validity`" in text
    assert "`mean_calibrated_hysteresis_penalty`" in text
    assert "Soft terrain work coupling supported: `True`" in text
    assert "Matched soft terrain specificity supported: `True`" in text
    assert "Pooled soft terrain specificity supported: `False`" in text
    assert "Matched soft terrain cells: `7` / `9`" in text
    assert "Targeted event-pair status: `NO_DATA`" in text


def test_render_focused_paper_brief_warns_on_blocked_core_claims(tmp_path: Path):
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    digest = {
        "publication_ready": False,
        "publication_scope": "focused_core_claim_profile",
        "semantic_signal_interpretation": {"status": "blocked"},
        "core_claims": [],
        "blocked_core_claims": ["observer_relativity"],
        "unsafe_claims": [],
        "claim_points": {
            "observer_relativity": {
                "point_estimate": 0.0,
                "effect_direction": "placeholder_or_zero_delta",
            }
        },
        "packet_files": ["review_digest.json"],
    }
    (packet_dir / "review_digest.json").write_text(json.dumps(digest), encoding="utf-8")

    out_path = render_brief(packet_dir=packet_dir, out_path=tmp_path / "brief.md")

    text = out_path.read_text(encoding="utf-8")
    assert "## Publication Warning" in text
    assert "## Blocked Core Claims" in text
    assert "`observer_relativity`" in text
