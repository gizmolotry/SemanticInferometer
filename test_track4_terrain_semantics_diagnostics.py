import json
from pathlib import Path

import numpy as np
import pytest

from scripts import track4_terrain_semantics_diagnostics as diag


BASE_ZONE_SUMMARY = {
    "Bridge": {
        "path_count": 4,
        "mean_work_integral": 10.0,
        "closed_loop_rate": 0.9,
    },
    "Swamp": {
        "path_count": 3,
        "mean_work_integral": 25.0,
        "closed_loop_rate": 0.5,
    },
    "Tightrope": {
        "path_count": 2,
        "mean_work_integral": 35.0,
        "closed_loop_rate": 0.2,
    },
    "Void": {
        "path_count": 5,
        "mean_work_integral": 50.0,
        "closed_loop_rate": 0.1,
    },
}


def _zone_summary(*, work_shift: float = 0.0) -> dict[str, dict[str, float]]:
    return {
        zone: {
            **metrics,
            "mean_work_integral": metrics["mean_work_integral"] + work_shift,
        }
        for zone, metrics in BASE_ZONE_SUMMARY.items()
    }


def _record(
    *,
    corpus: str = "real",
    safe_for_thesis_claim: bool = True,
    basis_probe_score: float = 0.9,
    mean_work_integral: float = 30.0,
    mean_path_edge_count: float = 10.0,
    seed: int = 42,
    work_shift: float = 0.0,
) -> dict:
    return {
        "row": {
            "corpus": corpus,
            "corpus_kind": corpus,
            "basis": "logits_flat",
            "kernel": "rbf",
            "seed": seed,
            "safe_for_thesis_claim": safe_for_thesis_claim,
            "basis_probe_score": basis_probe_score,
            "closed_loop_rate": 0.75 if safe_for_thesis_claim else 0.25,
            "mean_work_integral": mean_work_integral,
            "primary_zone_count": 4,
        },
        "run_dir": f"/synthetic/{corpus}/seed_{seed}",
        "summary": {
            "mean_work_integral": mean_work_integral,
            "mean_path_edge_count": mean_path_edge_count,
            "mean_path_node_count": mean_path_edge_count + 1.0,
            "closed_loop_rate": 0.75 if safe_for_thesis_claim else 0.25,
            "primary_zone_count": 4,
            "primary_zone_summary": _zone_summary(work_shift=work_shift),
        },
    }


def _write_method_sweep(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "ranked_conditions": [
                    {
                        "proposal_mode": "metric_softmax",
                        "k_neighbors": 4,
                        "temperature": 0.5,
                        "gamma": 3.0,
                        "proposal_quality_score": 0.80,
                    },
                    {
                        "proposal_mode": "metric_softmax",
                        "k_neighbors": 8,
                        "temperature": 1.0,
                        "gamma": 3.0,
                        "proposal_quality_score": 0.86,
                    },
                    {
                        "proposal_mode": "stress_biased",
                        "k_neighbors": 4,
                        "temperature": 0.5,
                        "gamma": 6.0,
                        "proposal_quality_score": 0.50,
                    },
                    {
                        "proposal_mode": "committor_guided",
                        "k_neighbors": 8,
                        "temperature": 1.0,
                        "gamma": 6.0,
                        "proposal_quality_score": 0.62,
                    },
                ],
                "mode_robustness": [
                    {
                        "proposal_mode": "metric_softmax",
                        "mean_score": 0.83,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_terrain_construct_validity_detects_distinct_synthetic_zones():
    records = [
        _record(corpus="real", seed=1, work_shift=0.0),
        _record(corpus="control_random", seed=2, work_shift=5.0),
    ]

    result = diag.terrain_construct_validity(records)

    assert result["status"] == "OK"
    assert result["evidence_basis"] == "primary_path_touched_zone_summary"
    assert result["zone_count"] == 4
    assert result["zones_observed"] == ["Bridge", "Swamp", "Tightrope", "Void"]
    assert result["zone_summary"]["Bridge"]["observation_count"] == 2
    assert result["work_range"] == pytest.approx(40.0)
    assert result["closed_loop_range"] == pytest.approx(0.8)
    assert result["construct_distinctness_supported"] is True


def test_walker_sensitivity_reads_compact_method_sweep_summary(tmp_path: Path):
    method_sweep_path = tmp_path / "track4_grid_method_sweep_summary.json"
    _write_method_sweep(method_sweep_path)

    result = diag.walker_sensitivity(method_sweep_path)

    assert result["status"] == "OK"
    assert result["condition_count"] == 4
    assert result["score_range"] == pytest.approx(0.36)
    assert result["walker_sensitivity_detected"] is True
    assert result["score_by_mode"]["metric_softmax"] == pytest.approx(0.83)
    assert result["score_by_k_neighbors"]["4"] == pytest.approx(0.65)
    assert result["score_by_temperature"]["1.0"] == pytest.approx(0.74)
    assert result["score_by_gamma"]["6.0"] == pytest.approx(0.56)
    assert result["mode_robustness"] == [{"proposal_mode": "metric_softmax", "mean_score": 0.83}]


def test_work_decomposition_uses_proxy_without_native_component_claim():
    records = [
        _record(corpus="real", mean_work_integral=30.0, mean_path_edge_count=10.0),
        _record(corpus="control_shuffled", mean_work_integral=20.0, mean_path_edge_count=5.0),
    ]

    result = diag.work_decomposition_summary(records)

    assert result["status"] == "OK"
    assert result["native_component_decomposition_available"] is False
    assert result["decomposition_basis"] == "proxy_from_exported_path_lengths_and_summary_metrics"
    assert "proxy" in result["warning"]
    assert result["row_count"] == 2
    assert result["mean_work_integral"] == pytest.approx(25.0)
    assert result["mean_work_per_edge_proxy"] == pytest.approx(3.5)
    assert result["component_level_claim_safe"] is False
    assert "work_vs_path_edge_count" in result["confound_correlations"]
    assert result["rows"][0]["work_per_edge_proxy"] == pytest.approx(3.0)
    assert result["rows"][1]["work_per_edge_proxy"] == pytest.approx(4.0)


def test_real_vs_control_specificity_requires_real_outperform_controls():
    records = [
        _record(corpus="real", safe_for_thesis_claim=True, basis_probe_score=0.90, seed=1),
        _record(corpus="real", safe_for_thesis_claim=True, basis_probe_score=0.86, seed=2),
        _record(corpus="control_random", safe_for_thesis_claim=False, basis_probe_score=0.60, seed=3),
        _record(corpus="control_shuffled", safe_for_thesis_claim=False, basis_probe_score=0.65, seed=4),
    ]
    validation_rows = [record["row"] for record in records]

    result = diag.real_vs_control_specificity(validation_rows)

    assert result["status"] == "OK"
    assert result["real"]["row_count"] == 2
    assert result["controls"]["row_count"] == 2
    assert result["real"]["terrain_safe_rate"] == 1.0
    assert result["controls"]["terrain_safe_rate"] == 0.0
    assert result["real_minus_control_terrain_safe_rate"] == 1.0
    assert result["real_minus_control_mean_score"] == pytest.approx(0.255)
    assert result["terrain_specificity_supported"] is True
    assert result["by_basis"]["logits_flat"]["terrain_specificity_supported"] is True
    assert result["by_kernel"]["rbf"]["terrain_specificity_supported"] is True
    assert result["by_control_family"]["stochastic_controls"]["terrain_specificity_supported"] is True
    assert result["missing_control_families"] == ["control_constant"]


def test_terrain_contrast_matrix_ranks_bridge_void_for_synthetic_records():
    result = diag.terrain_contrast_matrix([_record()])

    assert result["status"] == "OK"
    assert result["contrast_count"] == 6
    assert result["contrast_summary"]["Bridge_vs_Void"]["observation_count"] == 1
    assert result["contrast_summary"]["Bridge_vs_Void"]["mean_work_gap_abs"] == pytest.approx(40.0)
    assert result["contrast_summary"]["Bridge_vs_Void"]["mean_closed_loop_gap_abs"] == pytest.approx(0.8)
    assert result["ranked_contrasts"][0]["contrast"] == "Bridge_vs_Void"


def test_soft_terrain_membership_is_bilinear_and_direction_free():
    bridge = diag.soft_terrain_membership(1.0, 0.0)
    void = diag.soft_terrain_membership(0.0, 1.0)
    mixed = diag.soft_terrain_membership(0.25, 0.75)

    assert bridge == {
        "Bridge": 1.0,
        "Swamp": 0.0,
        "Tightrope": 0.0,
        "Void": 0.0,
    }
    assert void["Void"] == pytest.approx(1.0)
    assert sum(mixed.values()) == pytest.approx(1.0)
    assert mixed["Void"] == pytest.approx(0.75 * 0.75)
    assert mixed["Swamp"] == pytest.approx(0.25 * 0.75)


def _write_soft_run(run_dir: Path, *, barrier_paths_are_costly: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "track3_density_rho.npy", np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    np.save(run_dir / "d_spectral.npy", np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    if barrier_paths_are_costly:
        works = np.asarray([1.0, 5.0, 1.5, 6.0], dtype=np.float32)
    else:
        works = np.asarray([5.0, 1.0, 6.0, 1.5], dtype=np.float32)
    np.savez(
        run_dir / "cyclic_paths.npz",
        path_indices=np.asarray(
            [
                np.asarray([0, 2], dtype=np.int32),
                np.asarray([1, 3], dtype=np.int32),
                np.asarray([0, 2, 0], dtype=np.int32),
                np.asarray([1, 3, 1], dtype=np.int32),
            ],
            dtype=object,
        ),
        work_integral=works,
        closed_loop=np.asarray([True, False, True, False], dtype=bool),
        path_anchor_idx=np.asarray([0, 1, 2, 3], dtype=np.int32),
        path_is_hot=np.asarray([False, True, False, True], dtype=bool),
    )


def test_soft_terrain_work_coupling_reads_existing_run_artifacts(tmp_path: Path):
    real_dir = tmp_path / "real" / "track4_basis_track2"
    control_dir = tmp_path / "control_random" / "track4_basis_track2"
    _write_soft_run(real_dir, barrier_paths_are_costly=True)
    _write_soft_run(control_dir, barrier_paths_are_costly=False)
    records = [
        {
            "row": {"corpus": "real", "corpus_kind": "real", "basis": "track2", "kernel": "rbf", "seed": 42},
            "run_dir": str(real_dir),
        },
        {
            "row": {
                "corpus": "control_random",
                "corpus_kind": "control_random",
                "basis": "track2",
                "kernel": "rbf",
                "seed": 42,
            },
            "run_dir": str(control_dir),
        },
    ]

    result = diag.soft_terrain_work_coupling(
        records,
        min_path_count=4,
        min_abs_work_corr=0.10,
        min_real_minus_control_corr=0.05,
        min_matched_cells=1,
    )

    assert result["status"] == "OK"
    assert result["runs_with_soft_fields"] == 2
    assert result["real"]["path_count"] == 4
    assert result["real"]["corr_soft_barrier_mass_work"] > 0.9
    assert result["controls"]["by_family"]["control_random"]["corr_soft_barrier_mass_work"] < -0.9
    assert result["real_soft_work_coupling_supported"] is True
    assert result["pooled_soft_terrain_specificity_supported"] is True
    assert result["matched_soft_terrain_specificity_supported"] is True
    assert result["soft_terrain_specificity_supported"] is True
    assert result["matched_cell_specificity"]["matched_cell_pass_rate"] == pytest.approx(1.0)
    assert len(result["path_rows"]) == 8


def test_targeted_event_pair_results_reports_no_data_without_path():
    result = diag.targeted_event_pair_results(None)

    assert result["status"] == "NO_DATA"
    assert result["failure_reasons"] == ["targeted event-pair corpus/results not provided"]
    assert "matched same-event article groups" in result["recommended_next_step"]


def test_targeted_event_pair_results_validates_same_event_frame_work_lift(tmp_path: Path):
    path = tmp_path / "targeted_pairs.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"pair_id": "a", "event_id": "e1", "relation": "same_event_same_frame", "left_source": "s1", "right_source": "s1", "left_work": 1.00, "right_work": 1.10, "track2_distance": 0.30},
                    {"pair_id": "b", "event_id": "e1", "relation": "same_event_same_frame", "left_source": "s2", "right_source": "s2", "left_work": 1.00, "right_work": 1.15, "track2_distance": 0.31},
                    {"pair_id": "c", "event_id": "e2", "relation": "same_event_same_frame", "left_source": "s1", "right_source": "s1", "left_work": 1.00, "right_work": 1.20, "track2_distance": 0.32},
                    {"pair_id": "d", "event_id": "e1", "relation": "same_event_different_frame", "left_source": "s1", "right_source": "s2", "work_gap": 0.70, "track2_distance": 0.33},
                    {"pair_id": "e", "event_id": "e2", "relation": "same_event_different_frame", "left_source": "s1", "right_source": "s3", "work_gap": 0.65, "track2_distance": 0.34},
                    {"pair_id": "f", "event_id": "e2", "relation": "same_event_different_frame", "left_source": "s2", "right_source": "s3", "work_gap": 0.75, "track2_distance": 0.35},
                    {"pair_id": "g", "event_id": "e3", "same_event": False, "frame_relation": "different_frame", "work_gap": 0.20, "track2_distance": 0.90},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = diag.targeted_event_pair_results(path)

    assert result["status"] == "VALIDATED"
    assert result["safe_for_thesis_claim"] is True
    assert result["mixed_frame_event_count"] == 2
    assert result["same_event_pair_count"] == 6
    assert result["same_event_same_frame_pair_count"] == 3
    assert result["same_event_different_frame_pair_count"] == 3
    assert result["cross_source_different_frame_pair_count"] == 3
    assert result["mean_same_event_same_frame_work_gap"] == pytest.approx(0.15)
    assert result["mean_same_event_different_frame_work_gap"] == pytest.approx(0.70)
    assert result["different_minus_same_frame_work_gap"] == pytest.approx(0.55)
    assert result["cliffs_delta_different_gt_same"] == pytest.approx(1.0)
    assert result["event_direction_pass_rate"] == pytest.approx(1.0)


def test_targeted_event_pair_results_rejects_flat_work_lift(tmp_path: Path):
    path = tmp_path / "targeted_pairs.csv"
    path.write_text(
        "\n".join(
            [
                "pair_id,event_id,same_event,frame_relation,work_gap,track2_distance",
                "a,e1,true,same_frame,0.40,0.30",
                "b,e1,true,same_frame,0.42,0.31",
                "c,e2,true,same_frame,0.41,0.31",
                "d,e1,true,different_frame,0.43,0.32",
                "e,e2,true,different_frame,0.44,0.33",
                "f,e2,true,different_frame,0.45,0.34",
            ]
        ),
        encoding="utf-8",
    )

    result = diag.targeted_event_pair_results(path)

    assert result["status"] == "INVALID"
    assert result["safe_for_thesis_claim"] is False
    assert "different-frame work gap lift below threshold" in result["failure_reasons"]
    assert "insufficient cross-source different-frame pairs" in result["failure_reasons"]


def test_run_diagnostics_does_not_treat_uninterpreted_targeted_file_as_claim_ready(tmp_path: Path, monkeypatch):
    records = [
        _record(corpus="real", safe_for_thesis_claim=True, basis_probe_score=0.90, seed=1),
        _record(corpus="control_random", safe_for_thesis_claim=False, basis_probe_score=0.60, seed=2),
    ]
    validation_summary_path = tmp_path / "track4_focused_basis_validation_summary.json"
    validation_summary_path.write_text(
        json.dumps({"rows": [record["row"] for record in records]}),
        encoding="utf-8",
    )
    method_sweep_path = tmp_path / "track4_grid_method_sweep_summary.json"
    _write_method_sweep(method_sweep_path)
    targeted_event_path = tmp_path / "targeted_event_pairs.json"
    targeted_event_path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    monkeypatch.setattr(diag, "load_traversal_records", lambda validation_summary: records)

    summary = diag.run_diagnostics(
        validation_summary_path=validation_summary_path,
        output_dir=tmp_path / "diagnostics",
        method_sweep_path=method_sweep_path,
        targeted_event_path=targeted_event_path,
    )

    assert summary["claim_boundary"]["targeted_event_pair_status"] == "AVAILABLE_UNINTERPRETED"
    assert summary["claim_boundary"]["targeted_event_pair_available"] is False
    assert summary["claim_boundary"]["targeted_event_pair_claim_ready"] is False


def test_run_diagnostics_writes_outputs_without_traversal_pipeline(tmp_path: Path, monkeypatch):
    records = [
        _record(corpus="real", safe_for_thesis_claim=True, basis_probe_score=0.90, seed=1),
        _record(corpus="control_random", safe_for_thesis_claim=False, basis_probe_score=0.60, seed=2),
    ]
    validation_summary_path = tmp_path / "track4_focused_basis_validation_summary.json"
    validation_summary_path.write_text(
        json.dumps({"rows": [record["row"] for record in records]}),
        encoding="utf-8",
    )
    method_sweep_path = tmp_path / "track4_grid_method_sweep_summary.json"
    _write_method_sweep(method_sweep_path)
    output_dir = tmp_path / "diagnostics"

    monkeypatch.setattr(diag, "load_traversal_records", lambda validation_summary: records)

    summary = diag.run_diagnostics(
        validation_summary_path=validation_summary_path,
        output_dir=output_dir,
        method_sweep_path=method_sweep_path,
        targeted_event_path=None,
    )

    assert summary["schema_version"] == "1.0"
    assert summary["diagnostic_type"] == "track4_terrain_semantics"
    assert summary["record_count"] == 2
    assert summary["claim_boundary"] == {
        "terrain_construct_distinctness": True,
        "walker_sensitivity_detected": True,
        "native_work_decomposition_available": False,
        "terrain_specificity_supported": True,
        "soft_terrain_work_coupling_supported": False,
        "pooled_soft_terrain_specificity_supported": False,
        "matched_soft_terrain_specificity_supported": False,
        "soft_terrain_specificity_supported": False,
        "targeted_event_pair_status": "NO_DATA",
        "targeted_event_pair_available": False,
        "targeted_event_pair_claim_ready": False,
    }
    for path in summary["outputs"].values():
        assert Path(path).exists()

    saved_construct = json.loads(
        (output_dir / "terrain_construct_validity.json").read_text(encoding="utf-8")
    )
    saved_summary = json.loads(
        (output_dir / "track4_terrain_semantics_diagnostics_summary.json").read_text(encoding="utf-8")
    )

    assert saved_construct["construct_distinctness_supported"] is True
    assert saved_summary["outputs"]["terrain_contrast_matrix"].endswith("terrain_contrast_matrix.json")
    assert (output_dir / "terrain_contrast_matrix.csv").exists()
