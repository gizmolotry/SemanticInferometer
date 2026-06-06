import json
from pathlib import Path

from analysis.verification.open_hypothesis_suite import (
    BLOCKED,
    PARTIAL,
    SUPPORTED,
    UNPROVEN,
    HYPOTHESIS_IDS,
    OpenHypothesisConfig,
    run_open_hypothesis_suite,
)
from scripts.run_open_hypothesis_suite import main as run_cli_main


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_recenter_summary(path: Path) -> Path:
    return _write_json(
        path,
        {
            "summary_type": "observer_recenter_robustness_suite",
            "cell_count": 9,
            "aggregate": {
                "translation_only": {
                    "cell_count": 9,
                    "pass_count": 0,
                    "mean_primary_label_gain": 0.0,
                },
                "artifact_view": {
                    "cell_count": 9,
                    "pass_count": 0,
                    "mean_primary_label_gain": None,
                },
                "raw_track2_pca": {
                    "cell_count": 9,
                    "pass_count": 6,
                    "mean_primary_label_gain": 0.2563,
                },
                "cls_mean_pca": {
                    "cell_count": 9,
                    "pass_count": 7,
                    "mean_primary_label_gain": 0.2839,
                },
                "local_track_recompute": {
                    "cell_count": 9,
                    "pass_count": 3,
                    "mean_primary_label_gain": 0.4174,
                },
                "local_track_recompute:uniform_weighted_rks": {
                    "cell_count": 9,
                    "pass_count": 7,
                    "mean_primary_label_gain": 0.2895,
                },
            },
        },
    )


def _write_claim_matrix(path: Path, *, terrain_safe: bool = False, track4_safe: bool = True) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "1.0",
            "claims": [
                {
                    "claim_id": "terrain_incremental_signal",
                    "pass": terrain_safe,
                    "thesis_safe": terrain_safe,
                    "point_estimate": 0.32 if terrain_safe else -4.99,
                    "effect_direction": (
                        "terrain_adds_within_label_traversal_signal"
                        if terrain_safe
                        else "terrain_incremental_signal_unproven"
                    ),
                    "failure_reasons": [] if terrain_safe else ["cross-terrain work-gap lift below threshold"],
                },
                {
                    "claim_id": "track4_traversal_validity",
                    "pass": False,
                    "thesis_safe": False,
                    "point_estimate": 0.81,
                    "failure_reasons": ["path-touched zone coverage insufficient"],
                },
                {
                    "claim_id": "track4_observer_state_action_only_separation",
                    "pass": True,
                    "thesis_safe": True,
                    "point_estimate": 3.56,
                },
                {
                    "claim_id": "track4_observer_state_action_separation",
                    "pass": track4_safe,
                    "thesis_safe": track4_safe,
                    "point_estimate": 3.56,
                    "failure_reasons": [] if track4_safe else ["basis_seed_robustness_failed"],
                },
            ],
        },
    )


def _write_track4_summary(path: Path, *, thesis_safe: bool = False) -> Path:
    return _write_json(
        path,
        {
            "summary_type": "track4_observer_state_ablation",
            "claim_evaluation": {
                "claim_id": "track4_observer_state_action_separation",
                "action_only_robustness_pass": True,
                "pass": thesis_safe,
                "thesis_safe": thesis_safe,
                "safe_for_thesis_claim": thesis_safe,
                "point_estimate": 3.4,
                "failure_reasons": [] if thesis_safe else ["basis_seed_robustness_failed"],
            },
        },
    )


def _write_property_theft_summary(path: Path, *, records: int = 4) -> Path:
    return _write_json(
        path,
        {
            "summary_type": "observer_slice_transport_summary",
            "status": "OK",
            "record_count": records,
            "mean_holonomy_action": 1.5,
            "mean_null_holonomy_action": 0.0,
            "mean_excess_holonomy_action": 1.5,
        },
    )


def _by_id(payload: dict) -> dict:
    return {row["hypothesis_id"]: row for row in payload["claims"]}


def test_open_hypothesis_suite_answers_all_nine_with_mock_dag(tmp_path: Path) -> None:
    recenter = _write_recenter_summary(tmp_path / "recenter" / "observer_recenter_robustness_suite.json")
    claim_matrix = _write_claim_matrix(tmp_path / "claim_matrix.json")
    track4 = _write_track4_summary(tmp_path / "track4_observer_state_action_summary.json", thesis_safe=False)
    holonomy = _write_property_theft_summary(tmp_path / "property" / "observer_slice_transport_summary.json")

    payload = run_open_hypothesis_suite(
        OpenHypothesisConfig(
            repo_root=tmp_path,
            output_dir=tmp_path / "out",
            claim_matrix_path=claim_matrix,
            recentering_summary_path=recenter,
            track4_observer_state_summary_path=track4,
            property_theft_transport_summary_path=holonomy,
        )
    )

    rows = _by_id(payload)
    assert tuple(rows) == HYPOTHESIS_IDS
    assert payload["hypothesis_count"] == 9
    assert rows["observer_recentering_robustness"]["status"] == PARTIAL
    assert rows["simple_baseline_superiority"]["status"] == PARTIAL
    assert rows["real_corpus_ideological_accuracy"]["status"] == BLOCKED
    assert rows["terrain_semantics_software"]["status"] == PARTIAL
    assert rows["track4_publishable_core"]["status"] == SUPPORTED
    assert rows["property_theft_real_scale"]["status"] == PARTIAL
    assert rows["prompt_set_invariance"]["status"] == UNPROVEN
    assert rows["track3_density_semantic_interpretation"]["status"] == UNPROVEN
    assert rows["visualization_human_readability"]["status"] == UNPROVEN

    contract = payload["orchestration_contract"]
    assert contract["executor"] == "mock_orchestrator"
    assert contract["node_count"] == 10
    assert set(contract["dependencies"]["track4_publishable_core"]) == {"discover_existing_artifacts"}
    assert (tmp_path / "out" / "open_hypothesis_matrix.json").exists()
    assert (tmp_path / "out" / "open_hypothesis_matrix.csv").exists()
    assert (tmp_path / "out" / "_hypothesis_dag" / "observer_recentering_robustness.json").exists()


def test_open_hypothesis_suite_promotes_external_gates_when_registered(tmp_path: Path) -> None:
    recenter = _write_json(
        tmp_path / "recenter.json",
        {
            "cell_count": 2,
            "aggregate": {
                "translation_only": {"cell_count": 2, "pass_count": 0},
                "artifact_view": {"cell_count": 2, "pass_count": 0},
                "raw_track2_pca": {"cell_count": 2, "pass_count": 1, "mean_primary_label_gain": 0.1},
                "local_track_recompute:uniform_weighted_rks": {
                    "cell_count": 2,
                    "pass_count": 2,
                    "mean_primary_label_gain": 0.4,
                },
            },
        },
    )
    claim_matrix = _write_claim_matrix(tmp_path / "claim_matrix.json", terrain_safe=True, track4_safe=True)
    track4 = _write_track4_summary(tmp_path / "track4.json", thesis_safe=True)
    holonomy = _write_property_theft_summary(tmp_path / "holonomy.json", records=60)
    independent = _write_json(
        tmp_path / "independent_labels.json",
        {
            "independent_labels_validated": True,
            "unit_of_analysis": "article",
            "accuracy_gate_pass": True,
        },
    )
    prompt = _write_json(tmp_path / "prompt.json", {"prompt_invariance_pass": True, "prompt_set_count": 3})
    track3 = _write_json(
        tmp_path / "track3.json",
        {"track3_density_semantic_pass": True, "heldout_density_corr": 0.51},
    )
    viz = _write_json(tmp_path / "viz.json", {"human_readability_pass": True, "participant_count": 8})

    payload = run_open_hypothesis_suite(
        OpenHypothesisConfig(
            repo_root=tmp_path,
            output_dir=tmp_path / "out",
            claim_matrix_path=claim_matrix,
            recentering_summary_path=recenter,
            track4_observer_state_summary_path=track4,
            property_theft_transport_summary_path=holonomy,
            independent_label_summary_path=independent,
            prompt_invariance_summary_path=prompt,
            track3_density_validation_path=track3,
            visualization_validation_path=viz,
        )
    )

    rows = _by_id(payload)
    assert all(row["status"] == SUPPORTED for row in rows.values())
    assert payload["status_counts"] == {SUPPORTED: 9}


def test_open_hypothesis_suite_cli_writes_expected_packet(monkeypatch, tmp_path: Path) -> None:
    recenter = _write_recenter_summary(tmp_path / "recenter.json")
    claim_matrix = _write_claim_matrix(tmp_path / "claim_matrix.json")
    holonomy = _write_property_theft_summary(tmp_path / "holonomy.json")
    out_dir = tmp_path / "packet"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_open_hypothesis_suite.py",
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--claim-matrix",
            str(claim_matrix),
            "--recentering-summary",
            str(recenter),
            "--property-theft-transport-summary",
            str(holonomy),
        ],
    )

    assert run_cli_main() == 0
    payload = json.loads((out_dir / "open_hypothesis_matrix.json").read_text(encoding="utf-8"))
    assert payload["hypothesis_count"] == 9
    assert (out_dir / "_hypothesis_dag" / "discover_existing_artifacts.json").exists()
