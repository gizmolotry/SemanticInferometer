from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.observer_manifold import (
    BUNDLE_TYPE,
    build_edge_action_ledger,
    build_observer_manifold_bundle_from_view_states,
    load_observer_manifold_bundle,
    observer_bundle_to_action_graph_inputs,
    observer_bundle_to_meaning_probe_inputs,
    read_observer_manifold_bundle,
    write_observer_manifold_bundle,
)
from core.observer_slice_transport import observer_slice_commutator


def _write_bundle_fixture(run_dir: Path, *, translation_only: bool = False, fresh_replay: bool = False) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    observer_dir = run_dir / "observer_0"
    observer_dir.mkdir()
    global_articles = [
        {"idx": 0, "x": 10.0, "y": 10.0, "z": 0.1},
        {"idx": 1, "x": 11.0, "y": 10.0, "z": 0.2},
        {"idx": 2, "x": 12.0, "y": 10.0, "z": 0.3},
    ]
    if translation_only:
        observer_articles = [
            {"idx": 0, "x": 0.0, "y": 0.0, "z": 0.1},
            {"idx": 1, "x": 1.0, "y": 0.0, "z": 0.2},
            {"idx": 2, "x": 2.0, "y": 0.0, "z": 0.3},
        ]
    else:
        observer_articles = [
            {"idx": 0, "x": 0.0, "y": 0.0, "z": 0.1},
            {"idx": 1, "x": 0.4, "y": 0.0, "z": 0.2},
            {"idx": 2, "x": 3.0, "y": 0.0, "z": 0.3},
        ]
    for row, simplex in zip(observer_articles, ([0.8, 0.2], [0.6, 0.4], [0.1, 0.9])):
        row["observer_simplex"] = simplex
        row["spectral_probe_magnitudes"] = [abs(value) for value in simplex]
    (run_dir / "MONOLITH.view_state.json").write_text(
        json.dumps({"articles": global_articles}),
        encoding="utf-8",
    )
    (observer_dir / "MONOLITH.view_state.json").write_text(
        json.dumps(
            {
                "observer_focus": {"idx": 0},
                "articles": observer_articles,
                "walker_paths": [
                    {
                        "article_idx": 0,
                        "path_indices": [0, 1, 2],
                        "focused_observer_replay": not translation_only,
                        "fresh_focused_observer_replay": bool(fresh_replay),
                    }
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
                "2,b,Void,0.1,0.9,4.0",
            ]
        ),
        encoding="utf-8",
    )
    return observer_dir


def test_load_observer_manifold_bundle_builds_nodes_and_null_calibrated_edges(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")

    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")
    summary = bundle.edge_summary()

    assert bundle.to_dict(include_nodes=False, include_edges=False)["bundle_type"] == BUNDLE_TYPE
    assert bundle.focus_xy_centered is True
    assert bundle.provenance["coordinate_source"] == "global_and_observer_view_state"
    assert bundle.provenance["focused_observer_replay"] is True
    assert bundle.provenance["fresh_focused_observer_replay"] is False
    assert len(bundle.nodes) == 3
    assert len(bundle.edges) == 2
    assert summary["edge_count"] == 2
    assert summary["mean_positive_excess_action"] > 0.0
    edge = bundle.edges[1]
    assert edge.source_idx == 1
    assert edge.target_idx == 2
    assert edge.observer_distance > edge.translation_null_distance
    assert edge.positive_excess_action > 0.0
    assert edge.source_label == "a"
    assert edge.target_label == "b"
    assert bundle.nodes[0].observer_simplex == pytest.approx((0.8, 0.2))


def test_observer_manifold_fresh_replay_provenance_uses_fresh_flag(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run", fresh_replay=True)

    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    assert bundle.provenance["focused_observer_replay"] is True
    assert bundle.provenance["fresh_focused_observer_replay"] is True


def test_observer_manifold_translation_null_has_zero_excess_distance(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run", translation_only=True)

    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    assert bundle.mean_nontranslation_shift == pytest.approx(0.0)
    assert all(edge.excess_distance == pytest.approx(0.0) for edge in bundle.edges)
    assert bundle.edge_summary()["mean_positive_excess_action"] == pytest.approx(0.0)


def test_write_observer_manifold_bundle_exports_json_and_edge_csv(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    written = write_observer_manifold_bundle(bundle, tmp_path / "out")

    payload = json.loads(Path(written["bundle_json"]).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(Path(written["edge_ledger_csv"]).open(encoding="utf-8")))
    assert payload["bundle_type"] == BUNDLE_TYPE
    assert payload["edge_count"] == 2
    assert len(payload["edge_action_ledger"]) == 2
    assert len(rows) == 2
    assert "positive_excess_action" in rows[0]


def test_read_observer_manifold_bundle_round_trips_written_bundle(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")
    written = write_observer_manifold_bundle(bundle, tmp_path / "out")

    restored = read_observer_manifold_bundle(Path(written["bundle_json"]))

    assert restored.observer_idx == bundle.observer_idx
    assert restored.focus_idx == bundle.focus_idx
    assert restored.focus_xy_centered is True
    assert len(restored.nodes) == len(bundle.nodes)
    assert len(restored.edges) == len(bundle.edges)
    assert restored.edge_summary()["mean_positive_excess_action"] == pytest.approx(
        bundle.edge_summary()["mean_positive_excess_action"]
    )


def test_observer_bundle_to_meaning_probe_inputs_exposes_translation_null_and_labels(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    inputs = observer_bundle_to_meaning_probe_inputs(bundle)

    assert inputs["anchor_idx"] == 0
    assert inputs["labels"] == {0: "a", 1: "a", 2: "b"}
    assert inputs["global_articles"][1]["x"] == pytest.approx(11.0)
    assert inputs["translation_articles"][1]["x"] == pytest.approx(1.0)
    assert inputs["observer_articles"][1]["x"] == pytest.approx(0.4)


def test_observer_bundle_to_action_graph_inputs_uses_requested_coordinate_frame(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    observer_inputs = observer_bundle_to_action_graph_inputs(bundle, coordinate_frame="observer_xyz")
    null_inputs = observer_bundle_to_action_graph_inputs(bundle, coordinate_frame="translation_null_xyz")

    assert observer_inputs["embeddings"].shape == (3, 3)
    assert observer_inputs["article_index_to_row"] == {0: 0, 1: 1, 2: 2}
    assert observer_inputs["embeddings"][1, 0] == pytest.approx(0.4)
    assert null_inputs["embeddings"][1, 0] == pytest.approx(1.0)
    assert observer_inputs["track3_density"].tolist() == pytest.approx([0.9, 0.8, 0.1])
    assert observer_inputs["metric_stress"].tolist() == pytest.approx([0.1, 0.2, 0.9])
    assert observer_inputs["observer_simplex_supported"] is True
    assert observer_inputs["observer_simplex"].shape == (3, 2)


def test_observer_bundle_coordinates_support_slice_transport_holonomy(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")
    observer_inputs = observer_bundle_to_action_graph_inputs(bundle, coordinate_frame="observer_xyz")
    null_inputs = observer_bundle_to_action_graph_inputs(bundle, coordinate_frame="translation_null_xyz")

    result = observer_slice_commutator(
        {
            "translation_null": null_inputs["embeddings"],
            "observer_recomputed": observer_inputs["embeddings"],
        },
        source_idx=0,
        target_idx=1,
        source_slice="translation_null",
        target_slice="observer_recomputed",
        density=observer_inputs["track3_density"],
        stress=observer_inputs["metric_stress"],
    )

    assert result["source_slice"] == "translation_null"
    assert result["target_slice"] == "observer_recomputed"
    assert result["semantic_first_components"]["semantic_move"] > 0.0
    assert result["observer_first_components"]["semantic_move"] > 0.0
    assert result["holonomy_action"] > 0.0


def test_observer_bundle_to_action_graph_inputs_rejects_missing_coordinate_frame(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    bundle = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    with pytest.raises(ValueError, match="coordinate_frame"):
        observer_bundle_to_action_graph_inputs(bundle, coordinate_frame="missing_xyz")


def test_build_observer_manifold_bundle_from_view_states_matches_loader(tmp_path: Path) -> None:
    observer_dir = _write_bundle_fixture(tmp_path / "run")
    global_state = json.loads((tmp_path / "run" / "MONOLITH.view_state.json").read_text(encoding="utf-8"))
    observer_state = json.loads((observer_dir / "MONOLITH.view_state.json").read_text(encoding="utf-8"))
    metadata = {
        0: {"source": "a", "zone": "Bridge", "density": 0.9, "stress": 0.1, "w_actual": 1.0},
        1: {"source": "a", "zone": "Bridge", "density": 0.8, "stress": 0.2, "w_actual": 1.2},
        2: {"source": "b", "zone": "Void", "density": 0.1, "stress": 0.9, "w_actual": 4.0},
    }

    from_views = build_observer_manifold_bundle_from_view_states(
        global_state,
        observer_state,
        metadata=metadata,
        run_dir=str(tmp_path / "run"),
        observer_dir=str(observer_dir),
        observer_idx=0,
        focus_idx=0,
        label_key="source",
    )
    loaded = load_observer_manifold_bundle(tmp_path / "run", observer_dir, label_key="source")

    assert from_views.provenance["coordinate_source"] == "in_memory_view_state"
    assert from_views.focus_xy_centered == loaded.focus_xy_centered
    assert len(from_views.edges) == len(loaded.edges)
    assert from_views.edge_summary()["mean_positive_excess_action"] == pytest.approx(
        loaded.edge_summary()["mean_positive_excess_action"]
    )


def test_build_edge_action_ledger_handles_empty_paths() -> None:
    assert build_edge_action_ledger({}, []) == []
