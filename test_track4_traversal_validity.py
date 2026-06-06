from pathlib import Path
import json

import numpy as np

from analysis.verification.scientific_summaries import summarize_track4_traversal
from analysis.verification.thesis_evidence import build_thesis_evidence, _evaluate_track4_summary
from thesis_test_support import (
    build_canonical_fixture,
    write_monolith_data,
    write_view_state,
    write_walker_states,
)


def _write_cyclic_paths(
    run_dir: Path,
    *,
    anchor_indices: list[int],
    path_anchor_idx: list[int],
    path_indices: list[list[int]],
    path_is_hot: list[bool] | None = None,
    work_integral: list[float] | None = None,
    closed_loop: list[bool] | None = None,
    terrain_labels: list[str] | None = None,
    feature_basis: str | None = None,
    proposal_mode: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path_count = len(path_anchor_idx)
    payload = {
        "work_integral": np.asarray(
            work_integral if work_integral is not None else [12.0 + idx for idx in range(path_count)],
            dtype=float,
        ),
        "closed_loop": np.asarray(
            closed_loop if closed_loop is not None else [idx % 2 == 0 for idx in range(path_count)],
            dtype=bool,
        ),
        "path_anchor_idx": np.asarray(path_anchor_idx, dtype=int),
        "anchor_indices": np.asarray(anchor_indices, dtype=int),
        "path_is_hot": np.asarray(
            path_is_hot if path_is_hot is not None else [idx < 2 for idx in range(path_count)],
            dtype=bool,
        ),
        "path_indices": np.asarray(path_indices, dtype=object),
    }
    if feature_basis is not None:
        payload["path_feature_basis"] = np.asarray([feature_basis] * path_count, dtype=object)
    if proposal_mode is not None:
        payload["path_proposal_mode"] = np.asarray([proposal_mode] * path_count, dtype=object)
    if terrain_labels is not None:
        payload["anchor_selection_metadata"] = np.asarray(
            [json.dumps({"terrain_labels": terrain_labels})],
            dtype=object,
        )
    np.savez(run_dir / "cyclic_paths.npz", **payload)


def test_track4_traversal_detects_dead_paths_and_collapsed_geometry(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    real_dir = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
    )
    write_view_state(real_dir / "MONOLITH.view_state.json", survival_rate=0.0, mean_work=0.0)
    write_walker_states(real_dir / "walker_states.json", closed_count=0, total=3, work=0.0)
    write_monolith_data(real_dir / "MONOLITH_DATA.csv", collapsed=True)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_traversal_validity"
    )
    assert not claim["thesis_safe"]
    failures = payloads["scientific_validation_summary"]["failure_modes"]["records"]
    failure_types = {record["failure_type"] for record in failures}
    assert "dead_paths" in failure_types
    assert "collapsed_manifold" in failure_types


def test_track4_traversal_requires_real_to_outperform_controls(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    for corpus in ("control_constant", "control_shuffled", "control_random"):
        control_dir = (
            fixture["runs_dir"]
            / "experiments_20260504_010101"
            / "rbf"
            / "cls"
            / corpus
        )
        write_view_state(control_dir / "MONOLITH.view_state.json", survival_rate=0.72, mean_work=71.0)
        write_walker_states(control_dir / "walker_states.json", closed_count=3, total=4, work=70.0)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_traversal_validity"
    )
    assert not claim["thesis_safe"]


def test_track4_work_barrier_signal_can_pass_without_survival_gap(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    for corpus in ("control_constant", "control_shuffled", "control_random"):
        control_dir = (
            fixture["runs_dir"]
            / "experiments_20260504_010101"
            / "rbf"
            / "cls"
            / corpus
        )
        write_view_state(control_dir / "MONOLITH.view_state.json", survival_rate=0.72, mean_work=18.0)
        write_walker_states(control_dir / "walker_states.json", closed_count=3, total=4, work=18.0)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    strict_claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_traversal_validity"
    )
    work_claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_work_barrier_signal"
    )
    assert not strict_claim["thesis_safe"]
    assert work_claim["thesis_safe"]
    summary = payloads["scientific_validation_summary"]["track4_work_barrier_signal"]
    assert summary["mean_d_work_real_vs_controls"] >= 5.0


def test_track4_work_barrier_requires_real_to_beat_each_control_family(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    for corpus, work in (
        ("control_constant", 18.0),
        ("control_shuffled", 18.0),
        ("control_random", 71.0),
    ):
        control_dir = (
            fixture["runs_dir"]
            / "experiments_20260504_010101"
            / "rbf"
            / "cls"
            / corpus
        )
        write_view_state(control_dir / "MONOLITH.view_state.json", survival_rate=0.18, mean_work=work)
        write_walker_states(control_dir / "walker_states.json", closed_count=0, total=4, work=work)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    work_claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_work_barrier_signal"
    )
    assert not work_claim["thesis_safe"]
    pair = next(
        row for row in payloads["track4_traversal_summary"]["real_vs_controls"]
        if row["kernel"] == "rbf" and row["channel"] == "cls"
    )
    assert pair["d_mean_work"] >= 5.0
    assert pair["d_mean_work_min_vs_control"] < 5.0
    assert "control_random" in pair["failing_control_corpora"]


def test_track4_summary_rejects_missing_bridge_void_zone_coverage(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 1],
        path_anchor_idx=[0, 1, 0, 1, 0],
        path_indices=[[0, 1], [1, 0], [0, 1, 2], [1, 2], [0, 2]],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "INVALID"
    assert summary["safe_for_thesis_claim"] is False
    assert summary["terrain_evidence_basis"] == "path_touched"
    assert "bridge/void comparison unavailable for this run" in summary["failure_reasons"]
    assert any("anchor diagnostic: track 4 anchors cover fewer than three terrain zones" == warning for warning in summary["warnings"])


def test_track4_summary_marks_old_cyclic_npz_contract_invalid_without_crashing(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "cyclic_paths.npz", paths=np.asarray([[0, 1, 2]], dtype=object))

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "INVALID"
    assert summary["safe_for_thesis_claim"] is False
    assert "work_integral" in summary["failure_reasons"][0]
    assert "available cyclic_paths.npz keys" in summary["warnings"][0]


def test_track4_summary_exposes_touched_zone_diagnostics_separately_from_anchor_zones(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[3],
        path_anchor_idx=[3, 3],
        path_indices=[[3, 0, 2, 3], [3, 0, 3]],
        closed_loop=[False, True],
        work_integral=[40.0, 10.0],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["anchor_zone_count"] == 1
    assert summary["zone_summary"].keys() == {"Void"}
    assert summary["touched_zone_count"] == 3
    assert set(summary["touched_zone_summary"].keys()) == {"Bridge", "Tightrope", "Void"}
    assert summary["terrain_evidence_basis"] == "path_touched"
    assert set(summary["primary_zone_summary"].keys()) == {"Bridge", "Tightrope", "Void"}
    assert summary["per_anchor"][0]["touched_zones"] == ["Bridge", "Tightrope", "Void"]


def test_track4_summary_prefers_exact_anchor_article_ids_over_ordinal_fallback(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[1, 3],
        path_anchor_idx=[1, 1, 3, 3, 1, 3],
        path_indices=[[1, 0, 3], [1, 2, 3], [3, 2, 1], [3, 0, 1], [1, 3], [3, 1]],
        path_is_hot=[True, True, False, False, True, False],
        closed_loop=[True, True, False, False, True, False],
        work_integral=[10.0, 12.0, 42.0, 55.0, 11.0, 48.0],
    )

    summary = summarize_track4_traversal(run_dir)

    anchors = {row["anchor_article_idx"]: row for row in summary["per_anchor"]}
    assert set(anchors) == {1, 3}
    assert anchors[1]["zone"] == "Swamp"
    assert anchors[3]["zone"] == "Void"
    assert summary["zone_summary"]["Swamp"]["path_count"] == 3
    assert summary["zone_summary"]["Void"]["path_count"] == 3


def test_track4_summary_promotes_touched_zones_to_primary_when_anchor_coverage_is_sparse(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[3],
        path_anchor_idx=[3, 3, 3, 3, 3],
        path_indices=[
            [3, 0, 3],
            [3, 0, 3, 0],
            [3, 2, 3],
            [3],
            [3, 3],
        ],
        path_is_hot=[True, True, False, False, False],
        closed_loop=[True, True, False, False, False],
        work_integral=[10.0, 12.0, 45.0, 55.0, 60.0],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "OK"
    assert summary["safe_for_thesis_claim"] is True
    assert summary["terrain_evidence_basis"] == "path_touched"
    assert summary["anchor_zone_count"] == 1
    assert summary["primary_zone_count"] == 3
    assert set(summary["primary_zone_summary"].keys()) == {"Bridge", "Tightrope", "Void"}
    assert summary["primary_bridge_vs_void"]["closed_loop_rate_gap"] > 0.05
    assert summary["primary_bridge_vs_void"]["work_integral_gap"] > 0.05


def test_track4_summary_rejects_zero_reactive_flux_when_markov_boundaries_exist(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 3],
        path_anchor_idx=[0, 0, 3, 3, 0],
        path_indices=[[0, 2, 3], [0, 1, 3], [3, 2, 0], [3, 0], [0, 3]],
        path_is_hot=[True, True, False, False, False],
        closed_loop=[True, True, False, False, True],
        work_integral=[10.0, 12.0, 45.0, 55.0, 11.0],
    )
    with np.load(run_dir / "cyclic_paths.npz", allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}
    data["reactive_flux_values"] = np.asarray([], dtype=float)
    data["reactive_flux_edges"] = np.empty((0, 2), dtype=int)
    data["track4_markov_status"] = np.asarray(["OK"], dtype=object)
    np.savez(run_dir / "cyclic_paths.npz", **data)
    (run_dir / "track4_markov_summary.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "bridge_count": 1,
                "void_count": 1,
                "bridge_to_void_reachable": False,
                "void_to_bridge_reachable": False,
                "reactive_flux_total": 0.0,
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "INVALID"
    assert "track 4 Markov bridge/void reactive flux is zero" in summary["failure_reasons"]
    assert "track 4 metric graph has no directed Bridge-to-Void route" in summary["failure_reasons"]


def test_track4_evaluator_uses_primary_touched_summary_before_anchor_summary():
    evaluated = _evaluate_track4_summary(
        {
            "safe_for_thesis_claim": True,
            "mean_work_integral": 20.0,
            "closed_loop_rate": 0.7,
            "unique_path_shape_count": 5,
            "hot_count": 2,
            "cold_count": 3,
            "terrain_evidence_basis": "path_touched",
            "zone_summary": {"Void": {"path_count": 5}},
            "primary_zone_summary": {
                "Bridge": {"path_count": 2},
                "Tightrope": {"path_count": 1},
                "Void": {"path_count": 5},
            },
            "primary_bridge_vs_void": {"work_integral_gap": 12.0, "closed_loop_rate_gap": 0.4},
        }
    )

    assert evaluated["thesis_safe"] is True
    assert evaluated["terrain_evidence_basis"] == "path_touched"
    assert evaluated["zone_count"] == 3


def test_track4_summary_rejects_repeated_path_trace_collapse(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 1, 2, 3],
        path_anchor_idx=[0, 1, 2, 3, 0],
        path_indices=[[0, 1, 2] for _ in range(5)],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "INVALID"
    assert summary["unique_path_shape_count"] == 1
    assert summary["path_shape_entropy_norm"] == 0.0
    assert summary["path_edge_entropy_norm"] is not None
    assert "all Track 4 paths collapse to one repeated index trace" in summary["failure_reasons"]


def test_track4_summary_rejects_missing_hot_cold_split(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 1, 2, 3],
        path_anchor_idx=[0, 1, 2, 3, 0],
        path_indices=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 3]],
        path_is_hot=[True, True, True, True, True],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "INVALID"
    assert "track 4 did not preserve both hot and cold walkers" in summary["failure_reasons"]


def test_track4_summary_rejects_trivial_bridge_void_effect_size(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 1, 2, 3],
        path_anchor_idx=[0, 0, 1, 2, 3, 3],
        path_indices=[[0, 3], [0, 3, 1], [3, 0, 2], [0, 3, 2], [3, 0, 1], [3, 0]],
        work_integral=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        closed_loop=[True, True, False, False, True, True],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "INVALID"
    assert "bridge/void closed-loop gap below minimum semantic effect size" in summary["failure_reasons"]
    assert "bridge/void work-integral gap below minimum semantic effect size" in summary["failure_reasons"]


def test_track4_summary_accepts_work_gap_when_all_loops_close(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_monolith_data(run_dir / "MONOLITH_DATA.csv")
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 3],
        path_anchor_idx=[0, 0, 3, 3, 0, 3],
        path_indices=[[0, 2], [0, 1, 2], [3, 2], [3, 1, 2], [0, 2], [3, 2]],
        path_is_hot=[True, True, False, False, True, False],
        closed_loop=[True, True, True, True, True, True],
        work_integral=[10.0, 12.0, 42.0, 55.0, 11.0, 48.0],
        feature_basis="logits_flat",
        proposal_mode="metric_softmax",
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "OK"
    assert summary["safe_for_thesis_claim"] is True
    assert summary["primary_bridge_vs_void"]["closed_loop_rate_gap"] == 0.0
    assert summary["primary_bridge_vs_void"]["work_integral_gap"] > 0.05
    assert summary["path_shape_entropy_norm"] > 0.0
    assert summary["mean_path_edge_count"] > 0.0
    assert summary["feature_basis_counts"] == {"logits_flat": 6}
    assert summary["proposal_mode_counts"] == {"metric_softmax": 6}


def test_track4_summary_uses_npz_terrain_metadata_without_monolith_csv(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_cyclic_paths(
        run_dir,
        anchor_indices=[0, 2],
        path_anchor_idx=[0, 0, 2, 2, 0, 2],
        path_indices=[[0, 1], [0, 1], [2, 1], [2, 1], [0, 1], [2, 1]],
        path_is_hot=[True, True, False, False, True, False],
        closed_loop=[True, True, True, True, True, True],
        work_integral=[10.0, 12.0, 42.0, 55.0, 11.0, 48.0],
        terrain_labels=["Bridge", "Tightrope", "Void"],
    )

    summary = summarize_track4_traversal(run_dir)

    assert summary["status"] == "OK"
    assert summary["terrain_evidence_basis"] == "path_touched"
    assert set(summary["primary_zone_summary"]) == {"Bridge", "Tightrope", "Void"}
    assert summary["primary_bridge_vs_void"]["work_integral_gap"] > 0.05


def test_track4_evaluator_rejects_legacy_warning_only_bridge_void_gap():
    evaluated = _evaluate_track4_summary(
        {
            "safe_for_thesis_claim": True,
            "mean_work_integral": 12.0,
            "closed_loop_rate": 0.8,
            "zone_summary": {"Bridge": {"path_count": 5}, "Swamp": {"path_count": 5}},
            "bridge_vs_void": {},
            "warnings": ["bridge/void comparison unavailable for this run"],
        }
    )

    assert evaluated["thesis_safe"] is False
    assert "track 4 anchors cover fewer than three terrain zones" in evaluated["failure_reasons"]
    assert "bridge/void comparison unavailable for this run" in evaluated["failure_reasons"]


def test_track4_evaluator_rejects_legacy_repeated_trace_warning():
    evaluated = _evaluate_track4_summary(
        {
            "safe_for_thesis_claim": True,
            "mean_work_integral": 12.0,
            "closed_loop_rate": 0.8,
            "unique_path_shape_count": 1,
            "hot_count": 2,
            "cold_count": 3,
            "zone_summary": {
                "Bridge": {"path_count": 5},
                "Swamp": {"path_count": 5},
                "Void": {"path_count": 5},
            },
            "bridge_vs_void": {"work_integral_gap": 8.0, "closed_loop_rate_gap": 0.2},
        }
    )

    assert evaluated["thesis_safe"] is False
    assert "all Track 4 paths collapse to one repeated index trace" in evaluated["failure_reasons"]


def test_track4_evaluator_rejects_trivial_bridge_void_semantic_gap():
    evaluated = _evaluate_track4_summary(
        {
            "safe_for_thesis_claim": True,
            "mean_work_integral": 12.0,
            "closed_loop_rate": 0.8,
            "unique_path_shape_count": 4,
            "hot_count": 2,
            "cold_count": 3,
            "zone_summary": {
                "Bridge": {"path_count": 5},
                "Swamp": {"path_count": 5},
                "Void": {"path_count": 5},
            },
            "bridge_vs_void": {"work_integral_gap": 0.0, "closed_loop_rate_gap": 0.0},
        }
    )

    assert evaluated["thesis_safe"] is False
    assert "bridge/void closed-loop gap below minimum semantic effect size" in evaluated["failure_reasons"]
    assert "bridge/void work-integral gap below minimum semantic effect size" in evaluated["failure_reasons"]
