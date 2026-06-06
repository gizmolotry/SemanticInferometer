from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.track4_action_graph import (
    ActionGraphConfig,
    build_action_graph_fields,
    edge_action_components,
    export_action_graph,
    least_action_path,
    path_action_components,
    run_action_graph,
)
from core.observer_slice_transport import (
    ObserverSliceTransportConfig,
    observer_slice_commutator,
    summarize_observer_slice_transport,
    write_observer_slice_transport_summary,
)


def test_least_action_routes_around_stress_barrier() -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rho = torch.ones(6, dtype=torch.float32)
    stress = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    cfg = ActionGraphConfig(k_neighbors=5, stress_weight=10.0, void_weight=0.0, curvature_weight=0.0)
    fields = build_action_graph_fields(embeddings, track3_density=rho, metric_stress=stress, k_neighbors=cfg.k_neighbors)

    direct = path_action_components(fields, [0, 1, 2], config=cfg)
    result = least_action_path(fields, 0, 2, config=cfg)

    assert result["reached"] is True
    assert 1 not in result["path_indices"]
    assert result["action"] < direct["action"]


def test_low_density_stretches_metric_cost() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    stress = torch.zeros(3, dtype=torch.float32)
    cfg = ActionGraphConfig(k_neighbors=2, stress_weight=0.0, void_weight=1.0, curvature_weight=0.0)
    dense_fields = build_action_graph_fields(
        embeddings,
        track3_density=torch.ones(3, dtype=torch.float32),
        metric_stress=stress,
        k_neighbors=cfg.k_neighbors,
    )
    void_fields = build_action_graph_fields(
        embeddings,
        track3_density=torch.tensor([1.0, 0.05, 1.0], dtype=torch.float32),
        metric_stress=stress,
        k_neighbors=cfg.k_neighbors,
    )

    dense_edge = edge_action_components(dense_fields, 0, 1, config=cfg)
    void_edge = edge_action_components(void_fields, 0, 1, config=cfg)

    assert void_edge["metric"] > dense_edge["metric"]
    assert void_edge["action"] > dense_edge["action"]


def test_curvature_penalty_makes_zigzag_more_expensive_than_smooth_path() -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    rho = torch.ones(5, dtype=torch.float32)
    stress = torch.zeros(5, dtype=torch.float32)
    cfg = ActionGraphConfig(k_neighbors=4, stress_weight=0.0, void_weight=0.0, curvature_weight=4.0)
    fields = build_action_graph_fields(embeddings, track3_density=rho, metric_stress=stress, k_neighbors=cfg.k_neighbors)

    smooth = path_action_components(fields, [0, 1, 2], config=cfg)
    zigzag = path_action_components(fields, [0, 3, 4, 2], config=cfg)

    assert smooth["curvature_penalty"] == 0.0
    assert zigzag["curvature_penalty"] > 0.0
    assert zigzag["action"] > smooth["action"]


def test_run_action_graph_exports_contract(tmp_path: Path) -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rho = torch.tensor([1.0, 0.9, 0.1, 1.0, 0.8, 0.1], dtype=torch.float32)
    stress = torch.tensor([0.0, 0.1, 1.0, 0.0, 0.6, 1.0], dtype=torch.float32)
    observer_simplex = torch.tensor(
        [
            [0.90, 0.10],
            [0.70, 0.30],
            [0.20, 0.80],
            [0.85, 0.15],
            [0.55, 0.45],
            [0.10, 0.90],
        ],
        dtype=torch.float32,
    )
    cfg = ActionGraphConfig(k_neighbors=4, max_paths=1, observer_transport_weight=1.0, hysteresis_weight=1.0)

    result = run_action_graph(
        embeddings,
        track3_density=rho,
        metric_stress=stress,
        observer_simplex=observer_simplex,
        config=cfg,
        anchors=[0],
        targets_by_anchor={0: [2]},
    )
    paths = export_action_graph(result, tmp_path)

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    npz = np.load(paths["npz"], allow_pickle=True)
    npz_files = set(npz.files)
    legacy_keys = {
        "trajectory_coordinates",
        "path_indices",
        "action_integral",
        "metric_integral",
        "density_penalty_integral",
        "shear_penalty_integral",
        "stress_penalty_integral",
        "curvature_penalty_integral",
        "reached",
        "source_idx",
        "target_idx",
        "source_zone",
        "target_zone",
        "touched_zones",
        "embedding_dim",
    }
    observer_keys = {
        "observer_transport_penalty_integral",
        "hysteresis_penalty_integral",
    }

    assert summary["track4_engine"] == "least_action_metric_graph"
    assert summary["path_count"] > 0
    assert legacy_keys <= npz_files
    assert observer_keys <= npz_files
    np.testing.assert_allclose(
        npz["observer_transport_penalty_integral"],
        [row["observer_transport_penalty"] for row in result["records"]],
    )
    np.testing.assert_allclose(
        npz["hysteresis_penalty_integral"],
        [row["hysteresis_penalty"] for row in result["records"]],
    )
    assert np.any(npz["observer_transport_penalty_integral"] > 0.0)
    assert np.any(npz["hysteresis_penalty_integral"] > 0.0)
    assert int(npz["embedding_dim"][0]) == 2


def test_directional_shear_tensor_changes_edge_action() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    rho = torch.ones(2, dtype=torch.float32)
    aligned_shear = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    opposed_shear = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    cfg = ActionGraphConfig(k_neighbors=1, stress_weight=0.0, shear_weight=2.0, void_weight=0.0, curvature_weight=0.0)

    aligned_fields = build_action_graph_fields(
        embeddings,
        track3_density=rho,
        metric_stress=aligned_shear,
        k_neighbors=cfg.k_neighbors,
    )
    opposed_fields = build_action_graph_fields(
        embeddings,
        track3_density=rho,
        metric_stress=opposed_shear,
        k_neighbors=cfg.k_neighbors,
    )

    aligned = edge_action_components(aligned_fields, 0, 1, config=cfg)
    opposed = edge_action_components(opposed_fields, 0, 1, config=cfg)

    assert aligned["shear_penalty"] == 0.0
    assert opposed["shear_penalty"] > 0.0
    assert opposed["action"] > aligned["action"]


def test_observer_simplex_transport_and_directed_hysteresis_change_action() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    rho = torch.ones(3, dtype=torch.float32)
    observer_simplex = torch.tensor(
        [
            [0.90, 0.10],
            [0.55, 0.45],
            [0.20, 0.80],
        ],
        dtype=torch.float32,
    )
    fields = build_action_graph_fields(
        embeddings,
        track3_density=rho,
        metric_stress=torch.zeros(3, dtype=torch.float32),
        observer_simplex=observer_simplex,
        k_neighbors=2,
    )
    baseline_cfg = ActionGraphConfig(
        k_neighbors=2,
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=0.0,
        hysteresis_weight=0.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )
    observer_cfg = ActionGraphConfig(
        k_neighbors=2,
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=2.0,
        hysteresis_weight=3.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )

    baseline_edge = edge_action_components(fields, 0, 1, config=baseline_cfg)
    forward_edge = edge_action_components(fields, 0, 1, config=observer_cfg)
    reverse_edge = edge_action_components(fields, 1, 0, config=observer_cfg)
    baseline_path = path_action_components(fields, [0, 1, 2], config=baseline_cfg)
    observer_path = path_action_components(fields, [0, 1, 2], config=observer_cfg)

    assert fields.observer_transport_distance[0, 1].item() > 0.0
    assert fields.observer_transport_distance[0, 1].item() == fields.observer_transport_distance[1, 0].item()
    assert fields.observer_kl_forward[0, 1].item() > 0.0
    assert fields.observer_kl_forward[1, 0].item() > 0.0
    assert not np.isclose(fields.observer_kl_forward[0, 1].item(), fields.observer_kl_forward[1, 0].item())
    assert forward_edge["observer_transport_penalty"] > 0.0
    assert forward_edge["hysteresis_penalty"] > 0.0
    assert forward_edge["action"] > baseline_edge["action"]
    assert not np.isclose(forward_edge["action"], reverse_edge["action"])
    assert observer_path["observer_transport_penalty"] > 0.0
    assert observer_path["hysteresis_penalty"] > 0.0
    assert observer_path["action"] > baseline_path["action"]


def _track4_repair_toy_inputs() -> dict[str, torch.Tensor]:
    return {
        "embeddings": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "rho": torch.ones(6, dtype=torch.float32),
        "stress": torch.zeros(6, dtype=torch.float32),
        "observer_simplex": torch.tensor(
            [
                [0.92, 0.08],
                [0.82, 0.18],
                [0.70, 0.30],
                [0.25, 0.75],
                [0.16, 0.84],
                [0.08, 0.92],
            ],
            dtype=torch.float32,
        ),
    }


def test_null_calibrated_action_exports_null_and_excess_hysteresis(tmp_path: Path) -> None:
    toy = _track4_repair_toy_inputs()
    cfg = ActionGraphConfig(
        k_neighbors=4,
        max_paths=1,
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=0.0,
        hysteresis_weight=1.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )
    shuffled_null = toy["observer_simplex"][torch.tensor([3, 4, 5, 0, 1, 2])]

    calibrated = run_action_graph(
        toy["embeddings"],
        track3_density=toy["rho"],
        metric_stress=toy["stress"],
        observer_simplex=toy["observer_simplex"],
        null_observer_simplex=shuffled_null,
        action_branch="null_calibrated_hysteresis",
        config=cfg,
        anchors=[0],
        targets_by_anchor={0: [5]},
    )
    self_null = run_action_graph(
        toy["embeddings"],
        track3_density=toy["rho"],
        metric_stress=toy["stress"],
        observer_simplex=toy["observer_simplex"],
        null_observer_simplex=toy["observer_simplex"],
        action_branch="null_calibrated_hysteresis",
        config=cfg,
        anchors=[0],
        targets_by_anchor={0: [5]},
    )

    paths = export_action_graph(calibrated, tmp_path / "calibrated")
    npz = np.load(paths["npz"], allow_pickle=True)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

    ledger_fields = {
        "hysteresis_penalty",
        "null_hysteresis_penalty",
        "excess_hysteresis_penalty",
        "positive_excess_hysteresis_penalty",
        "calibrated_hysteresis_penalty",
    }
    assert {
        "null_hysteresis_penalty_integral",
        "excess_hysteresis_penalty_integral",
        "positive_excess_hysteresis_penalty_integral",
        "calibrated_hysteresis_penalty_integral",
    } <= set(npz.files)
    assert ledger_fields <= set(calibrated["records"][0])
    assert ledger_fields <= set(summary["records"][0])
    assert float(npz["null_hysteresis_penalty_integral"][0]) > 0.0
    assert np.isfinite(float(npz["excess_hysteresis_penalty_integral"][0]))
    assert float(npz["positive_excess_hysteresis_penalty_integral"][0]) >= 0.0
    np.testing.assert_allclose(
        npz["calibrated_hysteresis_penalty_integral"],
        [row["calibrated_hysteresis_penalty"] for row in calibrated["records"]],
    )
    assert calibrated["records"][0]["excess_hysteresis_penalty"] != self_null["records"][0]["excess_hysteresis_penalty"]
    self_null_record = self_null["records"][0]
    assert self_null_record["hysteresis_penalty"] == pytest.approx(
        self_null_record["null_hysteresis_penalty"]
    )
    assert self_null_record["excess_hysteresis_penalty"] == pytest.approx(0.0)
    assert self_null_record["positive_excess_hysteresis_penalty"] == pytest.approx(0.0)
    assert self_null_record["calibrated_hysteresis_penalty"] == pytest.approx(0.0)


def test_null_calibrated_hysteresis_charges_only_positive_excess() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    rho = torch.ones(2, dtype=torch.float32)
    cfg = ActionGraphConfig(
        k_neighbors=1,
        action_mode="null_calibrated_hysteresis",
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=0.0,
        hysteresis_weight=1.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )

    null_larger = build_action_graph_fields(
        embeddings,
        track3_density=rho,
        metric_stress=torch.zeros(2, dtype=torch.float32),
        observer_simplex=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        null_observer_simplex=torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
        k_neighbors=1,
    )
    negative = edge_action_components(null_larger, 0, 1, config=cfg)

    assert negative["excess_hysteresis_penalty"] < 0.0
    assert negative["positive_excess_hysteresis_penalty"] == pytest.approx(0.0)
    assert negative["calibrated_hysteresis_penalty"] == pytest.approx(0.0)
    assert negative["action"] == pytest.approx(negative["metric"])

    observer_larger = build_action_graph_fields(
        embeddings,
        track3_density=rho,
        metric_stress=torch.zeros(2, dtype=torch.float32),
        observer_simplex=torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
        null_observer_simplex=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        k_neighbors=1,
    )
    positive = edge_action_components(observer_larger, 0, 1, config=cfg)

    assert positive["excess_hysteresis_penalty"] > 0.0
    assert positive["calibrated_hysteresis_penalty"] == pytest.approx(
        positive["positive_excess_hysteresis_penalty"]
    )
    assert positive["calibrated_hysteresis_penalty"] == pytest.approx(
        positive["hysteresis_penalty"] - positive["null_hysteresis_penalty"]
    )
    assert positive["action"] == pytest.approx(positive["metric"] + positive["calibrated_hysteresis_penalty"])


def test_baseline_hysteresis_records_null_but_charges_raw_hysteresis() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    fields = build_action_graph_fields(
        embeddings,
        track3_density=torch.ones(2, dtype=torch.float32),
        metric_stress=torch.zeros(2, dtype=torch.float32),
        observer_simplex=torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
        null_observer_simplex=torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32),
        k_neighbors=1,
    )
    base = ActionGraphConfig(
        k_neighbors=1,
        action_mode="baseline",
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=0.0,
        hysteresis_weight=1.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )
    calibrated = ActionGraphConfig(
        k_neighbors=1,
        action_mode="null_calibrated_hysteresis",
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=0.0,
        hysteresis_weight=1.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )

    baseline_edge = edge_action_components(fields, 0, 1, config=base)
    calibrated_edge = edge_action_components(fields, 0, 1, config=calibrated)

    assert baseline_edge["null_hysteresis_penalty"] > 0.0
    assert baseline_edge["calibrated_hysteresis_penalty"] == pytest.approx(baseline_edge["hysteresis_penalty"])
    assert baseline_edge["action"] == pytest.approx(baseline_edge["metric"] + baseline_edge["hysteresis_penalty"])
    assert calibrated_edge["calibrated_hysteresis_penalty"] == pytest.approx(
        calibrated_edge["positive_excess_hysteresis_penalty"]
    )


def test_richer_state_mode_preserves_reached_paths_and_records_mode_metadata() -> None:
    toy = _track4_repair_toy_inputs()
    baseline_cfg = ActionGraphConfig(
        k_neighbors=4,
        max_paths=1,
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=0.0,
        hysteresis_weight=0.0,
        void_weight=0.0,
        curvature_weight=0.0,
    )
    richer_cfg = ActionGraphConfig(
        k_neighbors=4,
        max_paths=1,
        stress_weight=0.0,
        shear_weight=0.0,
        observer_transport_weight=3.0,
        hysteresis_weight=4.0,
        void_weight=0.0,
        curvature_weight=1.5,
    )

    baseline = run_action_graph(
        toy["embeddings"],
        track3_density=toy["rho"],
        metric_stress=toy["stress"],
        observer_simplex=toy["observer_simplex"],
        action_branch="baseline_raw_action",
        config=baseline_cfg,
        anchors=[0],
        targets_by_anchor={0: [5]},
    )
    richer = run_action_graph(
        toy["embeddings"],
        track3_density=toy["rho"],
        metric_stress=toy["stress"],
        observer_simplex=toy["observer_simplex"],
        action_branch="richer_walker_state",
        richer_walker_state=True,
        config=richer_cfg,
        anchors=[0],
        targets_by_anchor={0: [5]},
    )

    assert baseline["records"][0]["reached"] is True
    assert richer["records"][0]["reached"] is True
    assert richer["records"][0]["source_idx"] == baseline["records"][0]["source_idx"]
    assert richer["records"][0]["target_idx"] == baseline["records"][0]["target_idx"]
    assert richer["summary"]["action_branch"] == "richer_walker_state"
    assert richer["summary"]["action_mode"] == "richer_state"
    assert richer["records"][0]["state_mode"] == "richer_walker_state"
    assert (
        richer["records"][0]["path_indices"] != baseline["records"][0]["path_indices"]
        or not np.isclose(richer["records"][0]["action"], baseline["records"][0]["action"])
    )


def test_virtual_transition_mode_creates_virtual_nodes_or_reports_count() -> None:
    toy = _track4_repair_toy_inputs()
    original_count = int(toy["embeddings"].shape[0])
    cfg = ActionGraphConfig(k_neighbors=4, max_paths=1, stress_weight=0.0, void_weight=0.0)

    result = run_action_graph(
        toy["embeddings"],
        track3_density=toy["rho"],
        metric_stress=toy["stress"],
        observer_simplex=toy["observer_simplex"],
        action_branch="virtual_transition_states",
        virtual_transition_states=True,
        virtual_interpolation_steps=2,
        config=cfg,
        anchors=[0],
        targets_by_anchor={0: [5]},
    )
    record = result["records"][0]
    path_types = record.get("path_node_types", [])

    assert result["summary"]["action_branch"] == "virtual_transition_states"
    assert result["summary"].get("virtual_node_count", 0) > 0 or any(idx >= original_count for idx in record["path_indices"])
    assert not path_types or "virtual" in path_types


def test_observer_slice_transport_translation_only_has_zero_holonomy() -> None:
    base = np.asarray(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [1.5, 0.0],
        ],
        dtype=np.float64,
    )
    slices = {
        "observer_a": base,
        "observer_b": base + np.asarray([4.0, -2.0], dtype=np.float64),
    }

    result = observer_slice_commutator(
        slices,
        source_idx=0,
        target_idx=1,
        source_slice="observer_a",
        target_slice="observer_b",
    )

    assert result["diagnostic_type"] == "observer_slice_transport_commutator"
    assert result["holonomy_action"] == pytest.approx(0.0)
    assert result["relative_holonomy"] == pytest.approx(0.0)
    assert result["closed_loop"][0] == {"article_idx": 0, "slice": "observer_a"}


def test_observer_slice_transport_detects_property_theft_style_chart_noncommutativity(tmp_path: Path) -> None:
    slices = {
        "anarchist": np.asarray(
            [
                [0.0, 0.0],  # property
                [0.4, 0.0],  # theft is directly traversable
                [2.5, 0.2],  # contract
            ],
            dtype=np.float64,
        ),
        "liberal": np.asarray(
            [
                [0.0, 0.0],  # property remains near the origin
                [0.4, 3.0],  # theft is separated by an observer-specific barrier
                [2.5, 0.2],  # contract remains stable across observer charts
            ],
            dtype=np.float64,
        ),
    }
    translation_null = {
        "anarchist": slices["anarchist"],
        "liberal": slices["anarchist"] + np.asarray([0.0, 3.0], dtype=np.float64),
    }
    cfg = ObserverSliceTransportConfig(simplex_weight=0.25)
    simplex = np.asarray([[0.95, 0.05], [0.10, 0.90]], dtype=np.float64)

    property_theft = observer_slice_commutator(
        slices,
        source_idx=0,
        target_idx=1,
        source_slice="anarchist",
        target_slice="liberal",
        slice_simplex=simplex,
        config=cfg,
    )
    property_contract = observer_slice_commutator(
        slices,
        source_idx=0,
        target_idx=2,
        source_slice="anarchist",
        target_slice="liberal",
        slice_simplex=simplex,
        config=cfg,
    )
    summary = summarize_observer_slice_transport(
        slices,
        article_pairs=[(0, 1), (0, 2)],
        slice_pairs=[("anarchist", "liberal")],
        slice_simplex=simplex,
        null_slices=translation_null,
        config=cfg,
    )
    written = write_observer_slice_transport_summary(summary, tmp_path)

    assert property_theft["holonomy_action"] > 0.0
    assert property_theft["relative_holonomy"] > property_contract["relative_holonomy"]
    assert summary["status"] == "OK"
    assert summary["mean_holonomy_action"] > summary["mean_null_holonomy_action"]
    assert summary["mean_excess_holonomy_action"] > 0.0
    assert Path(written["summary"]).exists()
