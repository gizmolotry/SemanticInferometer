from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import scripts.run_observer_recenter_meaning_probe as probe_mod

from scripts.run_observer_recenter_meaning_probe import (
    CLAIM_SCOPE,
    DIAGNOSTIC_TYPE,
    SUMMARY_TYPE,
    build_payload,
    evaluate_label_geometry_tests,
    evaluate_observer_recenter,
    evaluate_property_theft_homotopy,
    evaluate_run_dir,
    run_probe,
    select_label_column,
    synthetic_fixture_payload,
    terrain_regime_metrics,
    write_outputs,
)
from core.observer_local_recompute import (
    LOCAL_RECOMPUTE_MODE,
    LOCAL_RECOMPUTE_VARIANTS,
    compute_local_observer_recenter,
)


def test_local_observer_recompute_builds_track2_track15_track3_frame() -> None:
    cls = []
    for idx in range(6):
        label_axis = -1.0 if idx < 3 else 1.0
        observer_views = []
        for bot in range(4):
            observer_views.append(
                [
                    label_axis + (0.05 * bot),
                    (idx % 3) * 0.2,
                    (bot - 1.5) * label_axis,
                ]
            )
        cls.append(observer_views)
    payload = {
        "cls_per_bot": cls,
        "spectral_probe_magnitudes": [
            [1.0, 0.2, 0.1, 0.1],
            [0.9, 0.2, 0.1, 0.1],
            [0.8, 0.3, 0.1, 0.1],
            [0.1, 0.1, 0.2, 1.0],
            [0.1, 0.1, 0.3, 0.9],
            [0.1, 0.1, 0.2, 0.8],
        ],
    }

    recompute = compute_local_observer_recenter(payload, 0)

    assert recompute is not None
    assert recompute.diagnostics["mode"] == LOCAL_RECOMPUTE_MODE
    assert recompute.diagnostics["track15_stress_basis"]
    assert recompute.diagnostics["track3_density_basis"].startswith("rho_i=")
    assert recompute.track2_features.shape[0] == 6
    assert recompute.article_map()[0]["x"] == pytest.approx(0.0)
    assert recompute.article_map()[0]["y"] == pytest.approx(0.0)
    assert recompute.article_map()[0]["local_track3_density"] >= 0.0
    assert recompute.article_map()[0]["local_track15_stress"] >= 0.0
    assert set(recompute.zones).issubset({"Bridge", "Swamp", "Tightrope", "Void"})


def test_local_observer_recompute_variants_preserve_centering_and_contract() -> None:
    cls = []
    for idx in range(8):
        label_axis = -1.0 if idx < 4 else 1.0
        observer_views = []
        for bot in range(4):
            observer_views.append(
                [
                    label_axis + (0.03 * bot),
                    float(idx % 4) * 0.1,
                    (bot - 1.5) * label_axis,
                    float(idx) * 0.01,
                ]
            )
        cls.append(observer_views)
    payload = {
        "cls_per_bot": cls,
        "spectral_probe_magnitudes": [
            [1.0, 0.2, 0.1, 0.1],
            [0.9, 0.2, 0.1, 0.1],
            [0.8, 0.3, 0.1, 0.1],
            [0.7, 0.4, 0.1, 0.1],
            [0.1, 0.1, 0.2, 1.0],
            [0.1, 0.1, 0.3, 0.9],
            [0.1, 0.1, 0.2, 0.8],
            [0.1, 0.2, 0.2, 0.7],
        ],
    }

    for variant in LOCAL_RECOMPUTE_VARIANTS:
        recompute = compute_local_observer_recenter(payload, 0, variant=variant)

        assert recompute is not None
        assert recompute.diagnostics["mode"] == LOCAL_RECOMPUTE_MODE
        assert recompute.diagnostics["variant"] == variant
        assert recompute.to_public_dict()["variant"] == variant
        assert recompute.article_map()[0]["x"] == pytest.approx(0.0)
        assert recompute.article_map()[0]["y"] == pytest.approx(0.0)
        assert recompute.article_map()[0]["local_recompute_variant"] == variant
        assert recompute.track2_features.shape[0] == 8
        assert np.isfinite(recompute.projection_xyz).all()


def test_synthetic_fixture_requires_recenter_to_beat_translation_null() -> None:
    payload = synthetic_fixture_payload()
    recenter = payload["recenter"]

    assert payload["status"] == "PASS"
    assert recenter["focus_xy_centered"] is True
    assert recenter["nontranslation_shift_mean"] > 0.25
    assert recenter["label_gap_gain_over_translation"] > 0.50
    assert recenter["heldout_correlation_gain_over_translation"] > 0.10
    assert recenter["observer_recentered"]["label_gap"] > recenter["translation_null"]["label_gap"]
    assert recenter["observer_heldout"]["spearman_closeness_vs_similarity"] > recenter["translation_heldout"][
        "spearman_closeness_vs_similarity"
    ]


def test_translation_only_recenter_has_no_semantic_gain() -> None:
    global_articles = {
        0: {"idx": 0, "x": 2.0, "y": 1.0},
        1: {"idx": 1, "x": 3.0, "y": 1.0},
        2: {"idx": 2, "x": 2.0, "y": 3.0},
        3: {"idx": 3, "x": 4.0, "y": 1.0},
    }
    observer_articles = {
        idx: {"idx": idx, "x": row["x"] - 2.0, "y": row["y"] - 1.0}
        for idx, row in global_articles.items()
    }
    labels = {0: "a", 1: "a", 2: "b", 3: "b"}

    result = evaluate_observer_recenter(
        global_articles=global_articles,
        observer_articles=observer_articles,
        anchor_idx=0,
        labels=labels,
    )

    assert result["focus_xy_centered"] is True
    assert result["nontranslation_shift_mean"] == pytest.approx(0.0)
    assert result["label_gap_gain_over_translation"] == pytest.approx(0.0)
    assert result["scale_normalized_label_gap_gain_over_translation"] == pytest.approx(0.0)


def test_scale_normalized_label_gap_rejects_coordinate_scale_artifacts() -> None:
    global_articles = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 10.0, "y": 0.0},
        2: {"idx": 2, "x": 0.0, "y": 10.0},
        3: {"idx": 3, "x": 9.0, "y": 9.0},
        4: {"idx": 4, "x": 10.0, "y": 9.0},
        5: {"idx": 5, "x": 9.0, "y": 10.0},
    }
    observer_articles = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 0.02, "y": 0.0},
        2: {"idx": 2, "x": 0.0, "y": 0.02},
        3: {"idx": 3, "x": 1.0, "y": 1.0},
        4: {"idx": 4, "x": 1.02, "y": 1.0},
        5: {"idx": 5, "x": 1.0, "y": 1.02},
    }
    labels = {0: "a", 1: "a", 2: "a", 3: "b", 4: "b", 5: "b"}

    result = evaluate_observer_recenter(
        global_articles=global_articles,
        observer_articles=observer_articles,
        anchor_idx=0,
        labels=labels,
    )

    assert result["label_gap_gain_over_translation"] < 0.0
    assert result["scale_normalized_label_gap_gain_over_translation"] > 0.0


def test_property_theft_homotopy_detects_observer_dependent_traversability() -> None:
    result = evaluate_property_theft_homotopy()
    distances = result["distances"]

    assert result["status"] == "PASS"
    assert distances["anarchist_property_theft"] < distances["liberal_property_theft"]
    assert distances["liberal_property_contract"] < distances["anarchist_property_contract"]


def test_terrain_regime_metrics_require_behavioral_zone_distinctions() -> None:
    passing = terrain_regime_metrics(
        [
            {"zone": "Bridge", "density": 0.9, "stress": 0.1, "w_actual": 1.0},
            {"zone": "Bridge", "density": 0.8, "stress": 0.2, "w_actual": 1.2},
            {"zone": "Void", "density": 0.2, "stress": 0.9, "w_actual": 4.0},
            {"zone": "Void", "density": 0.1, "stress": 0.8, "w_actual": 4.4},
            {"zone": "Tightrope", "density": 0.3, "stress": 0.1, "w_actual": 2.2},
        ]
    )
    failing = terrain_regime_metrics(
        [
            {"zone": "Bridge", "density": 0.2, "stress": 0.9, "w_actual": 4.0},
            {"zone": "Void", "density": 0.9, "stress": 0.1, "w_actual": 1.0},
        ]
    )

    assert passing["status"] == "PASS"
    assert passing["pass_checks"]["void_stress_ge_bridge_stress"] is True
    assert failing["status"] == "FAIL"
    assert failing["pass_checks"]["bridge_density_ge_void_density"] is False


def _write_run_fixture(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.view_state.json").write_text(
        json.dumps(
            {
                "articles": [
                    {"idx": 0, "x": 0.0, "y": 0.0},
                    {"idx": 1, "x": 1.0, "y": 0.1},
                    {"idx": 2, "x": 1.1, "y": -0.1},
                    {"idx": 3, "x": 1.2, "y": 0.2},
                    {"idx": 4, "x": 1.0, "y": -0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    observer_dir = run_dir / "observer_0"
    observer_dir.mkdir()
    (observer_dir / "MONOLITH.view_state.json").write_text(
        json.dumps(
            {
                "observer_focus": {"idx": 0},
                "articles": [
                    {"idx": 0, "x": 0.0, "y": 0.0},
                    {"idx": 1, "x": 0.2, "y": 0.0},
                    {"idx": 2, "x": 0.3, "y": 0.1},
                    {"idx": 3, "x": 2.5, "y": 0.2},
                    {"idx": 4, "x": 2.7, "y": -0.2},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "MONOLITH_DATA.csv").write_text(
        "\n".join(
            [
                "index,source,zone,density,stress,w_actual",
                "0,a,Bridge,0.9,0.1,1.0",
                "1,a,Bridge,0.8,0.2,1.2",
                "2,a,Tightrope,0.3,0.1,2.0",
                "3,b,Void,0.2,0.9,4.0",
                "4,b,Void,0.1,0.8,4.4",
            ]
        ),
        encoding="utf-8",
    )


def test_select_label_column_prefers_ideological_basis_over_source_domains() -> None:
    metadata = {
        0: {"source": "Wafa News Agency", "perspective_tag": "Secular Palestinian Nationalist"},
        1: {"source": "Wafa News Agency", "perspective_tag": "Secular Palestinian Nationalist"},
        2: {"source": "Al Jazeera", "perspective_tag": "Arab Pro-Palestine"},
        3: {"source": "Al Jazeera", "perspective_tag": "Arab Pro-Palestine"},
    }

    column, counts, diagnostics = select_label_column(metadata, metadata.keys(), min_label_count=2)

    assert column == "perspective_tag"
    assert diagnostics["selected_basis"] == "ideological"
    assert diagnostics["semantic_label_basis"] is True
    assert counts == {
        "Arab Pro-Palestine": 2,
        "Secular Palestinian Nationalist": 2,
    }


def test_select_label_column_can_explicitly_use_provenance_source() -> None:
    metadata = {
        0: {"source": "Wafa News Agency", "perspective_tag": "Secular Palestinian Nationalist"},
        1: {"source": "Wafa News Agency", "perspective_tag": "Secular Palestinian Nationalist"},
        2: {"source": "Al Jazeera", "perspective_tag": "Arab Pro-Palestine"},
        3: {"source": "Al Jazeera", "perspective_tag": "Arab Pro-Palestine"},
        4: {"source": "Haaretz", "perspective_tag": "Arab Pro-Palestine"},
        5: {"source": "Haaretz", "perspective_tag": "Arab Pro-Palestine"},
    }

    column, counts, diagnostics = select_label_column(
        metadata,
        metadata.keys(),
        min_label_count=2,
        preferred_label_column="source",
    )

    assert column == "source"
    assert diagnostics["selected_basis"] == "provenance"
    assert diagnostics["provenance_label_basis"] is True
    assert counts == {"Al Jazeera": 2, "Haaretz": 2, "Wafa News Agency": 2}


def test_label_geometry_tests_report_all_five_semantic_validation_metrics() -> None:
    labels = {
        0: "frame_a",
        1: "frame_a",
        2: "frame_a",
        3: "frame_b",
        4: "frame_b",
        5: "frame_b",
        6: "frame_c",
        7: "frame_c",
        8: "frame_c",
    }
    global_articles = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 1.0, "y": 0.0},
        2: {"idx": 2, "x": 0.0, "y": 1.0},
        3: {"idx": 3, "x": 0.1, "y": 0.1},
        4: {"idx": 4, "x": 1.1, "y": 0.1},
        5: {"idx": 5, "x": 0.1, "y": 1.1},
        6: {"idx": 6, "x": 0.2, "y": 0.2},
        7: {"idx": 7, "x": 1.2, "y": 0.2},
        8: {"idx": 8, "x": 0.2, "y": 1.2},
    }
    observer_articles = {
        0: {"idx": 0, "x": 0.0, "y": 0.0},
        1: {"idx": 1, "x": 0.1, "y": 0.0},
        2: {"idx": 2, "x": 0.0, "y": 0.1},
        3: {"idx": 3, "x": 4.0, "y": 0.0},
        4: {"idx": 4, "x": 4.1, "y": 0.0},
        5: {"idx": 5, "x": 4.0, "y": 0.1},
        6: {"idx": 6, "x": -4.0, "y": 0.0},
        7: {"idx": 7, "x": -4.1, "y": 0.0},
        8: {"idx": 8, "x": -4.0, "y": 0.1},
    }

    result = evaluate_label_geometry_tests(
        global_articles=global_articles,
        observer_articles=observer_articles,
        anchor_idx=0,
        labels=labels,
    )

    assert result["status"] == "PASS"
    assert set(result["tests"]) == {
        "all_pairs_separation",
        "centroid_contraction",
        "silhouette",
        "ari_nmi",
        "permutation_within_between",
    }
    assert all(row["pass"] is True for row in result["tests"].values())
    assert result["tests"]["ari_nmi"]["observer"]["nmi"] > result["tests"]["ari_nmi"]["translation"]["nmi"]


def _write_ideology_run_fixture(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    global_articles = []
    labels = ["frame_a"] * 3 + ["frame_b"] * 3 + ["frame_c"] * 3
    global_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.1, 0.1),
        (1.1, 0.1),
        (0.1, 1.1),
        (0.2, 0.2),
        (1.2, 0.2),
        (0.2, 1.2),
    ]
    for idx, (x, y) in enumerate(global_points):
        global_articles.append({"idx": idx, "x": x, "y": y})
    (run_dir / "MONOLITH.view_state.json").write_text(
        json.dumps({"articles": global_articles}),
        encoding="utf-8",
    )
    clustered = {
        "frame_a": [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)],
        "frame_b": [(4.0, 0.0), (4.1, 0.0), (4.0, 0.1)],
        "frame_c": [(-4.0, 0.0), (-4.1, 0.0), (-4.0, 0.1)],
    }
    by_label_offset = {"frame_a": 0, "frame_b": 0, "frame_c": 0}
    base_observer_articles = []
    for idx, label in enumerate(labels):
        offset = by_label_offset[label]
        by_label_offset[label] += 1
        x, y = clustered[label][offset]
        base_observer_articles.append({"idx": idx, "x": x, "y": y})
    for anchor_idx in (0, 3, 6):
        observer_dir = run_dir / f"observer_{anchor_idx}"
        observer_dir.mkdir()
        anchor = base_observer_articles[anchor_idx]
        articles = [
            {"idx": row["idx"], "x": row["x"] - anchor["x"], "y": row["y"] - anchor["y"]}
            for row in base_observer_articles
        ]
        observer_dir.joinpath("MONOLITH.view_state.json").write_text(
            json.dumps({"observer_focus": {"idx": anchor_idx}, "articles": articles}),
            encoding="utf-8",
        )
    rows = ["index,source,perspective_tag,zone,density,stress,w_actual"]
    zones = ["Bridge", "Bridge", "Tightrope", "Void", "Void", "Swamp", "Bridge", "Void", "Tightrope"]
    densities = [0.9, 0.8, 0.3, 0.2, 0.1, 0.7, 0.85, 0.15, 0.25]
    stresses = [0.1, 0.2, 0.1, 0.9, 0.8, 0.9, 0.15, 0.85, 0.2]
    for idx, label in enumerate(labels):
        rows.append(f"{idx},source_{idx},{label},{zones[idx]},{densities[idx]},{stresses[idx]},{1.0 + idx}")
    (run_dir / "MONOLITH_DATA.csv").write_text("\n".join(rows), encoding="utf-8")


def test_evaluate_run_dir_records_ideological_validation_suite_and_anchor_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "ideology_run"
    _write_ideology_run_fixture(run_dir)

    result = evaluate_run_dir(run_dir, min_label_count=2, preferred_label_column="perspective_tag")
    suite = result["ideological_validation_suite"]

    assert result["label_column"] == "perspective_tag"
    assert result["label_basis"] == "ideological"
    assert suite["support_contracts"]["ideological_label_basis"]["pass"] is True
    assert suite["support_contracts"]["balanced_label_packet"]["pass"] is True
    assert suite["support_contracts"]["multi_anchor_coverage"]["pass"] is True
    assert suite["status"] == "PASS"
    assert all(suite["tests"][name]["pass"] is True for name in suite["tests"])


def test_evaluate_run_dir_records_requested_artifact_recenter_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "ideology_run"
    _write_ideology_run_fixture(run_dir)

    result = evaluate_run_dir(
        run_dir,
        min_label_count=2,
        preferred_label_column="perspective_tag",
        recenter_mode="artifact_view",
    )

    assert result["requested_recenter_mode"] == "artifact_view"
    assert result["actual_recenter_modes"] == ["artifact_view"]
    assert result["local_track_recompute_supported_count"] == 0
    assert all(row["actual_recenter_mode"] == "artifact_view" for row in result["observers"])


def test_evaluate_run_dir_expands_local_recompute_to_label_representative_anchors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ideology_run"
    _write_ideology_run_fixture(run_dir)
    shutil.rmtree(run_dir / "observer_3")
    shutil.rmtree(run_dir / "observer_6")

    clustered_xy = {
        0: (0.0, 0.0),
        1: (0.1, 0.0),
        2: (0.0, 0.1),
        3: (4.0, 0.0),
        4: (4.1, 0.0),
        5: (4.0, 0.1),
        6: (-4.0, 0.0),
        7: (-4.1, 0.0),
        8: (-4.0, 0.1),
    }

    class FakeLocalRecompute:
        def __init__(self, anchor_idx: int):
            self.anchor_idx = anchor_idx

        def article_map(self):
            ax, ay = clustered_xy[self.anchor_idx]
            return {
                idx: {"idx": idx, "x": x - ax, "y": y - ay}
                for idx, (x, y) in clustered_xy.items()
            }

    def fake_compute(_run_dir: Path, anchor_idx: int, **_kwargs):
        return FakeLocalRecompute(anchor_idx), {
            "status": "OK",
            "mode": LOCAL_RECOMPUTE_MODE,
            "focus_idx": int(anchor_idx),
        }

    monkeypatch.setattr(probe_mod, "load_primary_observer_payload", lambda _run_dir: ({"cls_per_bot": True}, "fake.pt", None))
    monkeypatch.setattr(probe_mod, "compute_local_observer_recenter_from_run", fake_compute)

    result = evaluate_run_dir(
        run_dir,
        min_label_count=2,
        preferred_label_column="perspective_tag",
        recenter_mode=LOCAL_RECOMPUTE_MODE,
    )

    suite = result["ideological_validation_suite"]
    assert result["observer_count"] == 3
    assert result["local_track_recompute_supported_count"] == 3
    assert result["actual_recenter_modes"] == [LOCAL_RECOMPUTE_MODE]
    assert suite["support_contracts"]["multi_anchor_coverage"]["pass"] is True
    assert sorted(suite["support_contracts"]["multi_anchor_coverage"]["covered_anchor_labels"]) == [
        "frame_a",
        "frame_b",
        "frame_c",
    ]
    assert {row["observer_candidate_source"] for row in result["observers"]} == {
        "materialized_observer_state",
        "local_track_recompute_label_anchor",
    }


def test_evaluate_run_dir_reports_translation_label_and_terrain_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir)

    result = evaluate_run_dir(run_dir, min_label_count=2)

    assert result["status"] == "OK"
    assert result["observer_count"] == 1
    assert result["label_column"] == "source"
    assert result["label_basis"] == "provenance"
    assert result["source_label_contraction"]["status"] == "OK"
    assert result["label_contraction"]["status"] == "OK"
    assert result["source_label_contraction"]["pass"] is True
    assert result["source_label_contraction"]["mean_label_gap_gain_over_translation"] > 0.5
    assert result["terrain_regime"]["status"] == "PASS"


def test_payload_outputs_follow_existing_dag_artifact_style(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run_fixture(run_dir)

    payload = build_payload(run_dirs=[run_dir], include_synthetic_fixture=True, min_label_count=2)
    artifacts = write_outputs(payload, out_dir)

    json_payload = json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
    dag_contract = json.loads(Path(artifacts["dag_contract"]).read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(Path(artifacts["csv"]).open(encoding="utf-8")))

    assert json_payload["diagnostic_type"] == DIAGNOSTIC_TYPE
    assert json_payload["summary_type"] == SUMMARY_TYPE
    assert json_payload["probe_type"] == DIAGNOSTIC_TYPE
    assert json_payload["claim_scope"] == CLAIM_SCOPE
    assert json_payload["safe_for_thesis_claim"] is True
    assert json_payload["thesis_safe"] is True
    assert json_payload["failure_reasons"] == []
    assert set(json_payload["thresholds"]) == {
        "min_nontranslation_shift",
        "min_label_gap_gain",
        "min_all_pairs_separation_gain",
        "min_centroid_contraction_gain",
        "min_silhouette_gain",
        "min_cluster_recovery_gain",
        "max_permutation_p_value",
        "min_synthetic_label_gap_gain",
        "min_heldout_correlation_gain",
    }
    assert json_payload["artifacts"]["json"].endswith("observer_recenter_meaning_probe.json")
    assert json_payload["artifacts"]["observer_manifold_bundles"].endswith("observer_manifold_bundles")
    assert len(json_payload["observer_manifold_bundle_artifacts"]) == 1
    assert json_payload["runs"][0]["observer_manifold_bundle_supported_count"] == 1
    assert json_payload["runs"][0]["edge_action_ledger"]["status"] == "OK"
    assert json_payload["runs"][0]["edge_action_ledger"]["edge_count"] == 0
    assert json_payload["runs"][0]["edge_action_ledger"]["edge_action_supported"] is False
    assert json_payload["runs"][0]["edge_action_ledger"]["fresh_edge_action_supported"] is False
    assert set(json_payload["runs"][0]["ideological_validation_suite"]["tests"]) == {
        "all_pairs_separation",
        "centroid_contraction",
        "silhouette",
        "ari_nmi",
        "permutation_within_between",
    }
    assert json_payload["claim_readiness"]["synthetic_meaning_pass"] is True
    assert json_payload["claim_readiness"]["real_label_contraction_pass"] is True
    assert json_payload["claim_readiness"]["real_source_label_contraction_pass"] is True
    assert json_payload["claim_readiness"]["edge_action_supported"] is False
    assert json_payload["claim_readiness"]["fresh_edge_action_supported"] is False
    assert json_payload["claim_boundary"]["edge_action_not_required_for_observer_recenter_claim"] is True
    assert dag_contract["dag_id"] == "observer_recenter_meaning_probe_v1"
    assert dag_contract["airflow_compatible"] is True
    assert "local_track_recompute" in {row["node_id"] for row in dag_contract["nodes"]}
    assert dag_contract["topological_order"][-1] == "summary"
    assert "local_track_recompute" in dag_contract["dependencies"]["translation_null"]
    assert csv_rows[0]["label_column"] == "source"
    assert csv_rows[0]["label_basis"] == "provenance"
    assert csv_rows[0]["actual_recenter_mode"] == "artifact_view"


def test_run_probe_is_importable_and_writes_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "probe"
    _write_run_fixture(run_dir)

    payload = run_probe(run_dirs=[run_dir], output_dir=out_dir, include_synthetic_fixture=True, min_label_count=2)

    assert payload["summary_type"] == SUMMARY_TYPE
    assert payload["artifacts"]["json"] == str(out_dir / "observer_recenter_meaning_probe.json")
    assert payload["observer_manifold_bundle_artifacts"]
    assert (out_dir / "observer_recenter_meaning_probe.json").exists()
    assert (out_dir / "observer_recenter_meaning_probe.csv").exists()
    assert (out_dir / "observer_recenter_meaning_dag_contract.json").exists()
    assert (out_dir / "observer_manifold_bundles").exists()
