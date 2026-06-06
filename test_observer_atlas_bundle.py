from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.observer_atlas_bundle import (
    ATLAS_BUNDLE_TYPE,
    ObserverAtlasError,
    build_observer_atlas_bundle,
    read_observer_atlas_bundle,
    write_observer_atlas_bundle,
)
from core.observer_manifold import build_observer_manifold_bundle_from_view_states
from core.observer_slice_transport import observer_slice_commutator


ARTICLE_IDS = [10, 20, 30]


def _observer_bundle(observer_idx: int, x_scale: float = 1.0):
    global_state = {
        "articles": [
            {"idx": 10, "x": 100.0, "y": 0.0, "z": 0.0},
            {"idx": 20, "x": 101.0, "y": 0.0, "z": 0.1},
            {"idx": 30, "x": 102.0, "y": 0.0, "z": 0.2},
        ]
    }
    observer_state = {
        "observer_focus": {"idx": observer_idx},
        "articles": [
            {"idx": 10, "x": 0.0, "y": 0.0, "z": 0.0, "observer_simplex": [0.8, 0.2]},
            {"idx": 20, "x": 0.5 * x_scale, "y": 0.2, "z": 0.1, "observer_simplex": [0.5, 0.5]},
            {"idx": 30, "x": 3.0 * x_scale, "y": -0.1, "z": 0.2, "observer_simplex": [0.1, 0.9]},
        ],
        "walker_paths": [
            {
                "article_idx": observer_idx,
                "path_indices": [10, 20, 30],
                "focused_observer_replay": True,
                "fresh_focused_observer_replay": True,
            }
        ],
    }
    metadata = {
        10: {"source": "a", "zone": "Bridge", "density": 0.9, "stress": 0.1},
        20: {"source": "b", "zone": "Swamp", "density": 0.7, "stress": 0.4},
        30: {"source": "c", "zone": "Void", "density": 0.2, "stress": 0.9},
    }
    return build_observer_manifold_bundle_from_view_states(
        global_state,
        observer_state,
        metadata=metadata,
        run_dir="run",
        observer_dir=f"run/observer_{observer_idx}",
        observer_idx=observer_idx,
        focus_idx=observer_idx,
        label_key="source",
    )


def test_observer_atlas_bundle_preserves_row_and_article_identity_for_noncontiguous_ids() -> None:
    atlas = build_observer_atlas_bundle(
        [_observer_bundle(10), _observer_bundle(20, x_scale=1.4)],
        observer_manifest={"observers": [{"idx": 10, "action": "rendered"}, {"idx": 20, "action": "rendered"}]},
        max_article_pairs=2,
    )

    assert atlas["bundle_type"] == ATLAS_BUNDLE_TYPE
    assert atlas["row_article_ids"] == ARTICLE_IDS
    assert atlas["article_idx_to_row"] == {"10": 0, "20": 1, "30": 2}
    assert atlas["metrics"]["record_count"] > 0
    assert atlas["metrics"]["mean_excess_holonomy_action"] is not None
    first_route = atlas["routes"][0]
    assert first_route["source_row_index"] in {0, 1}
    assert first_route["source_article_idx"] in ARTICLE_IDS
    assert first_route["source_row_index"] != first_route["source_article_idx"]
    assert len(first_route["semantic_first"]["points"]) == 3
    assert len(first_route["observer_first"]["points"]) == 3
    assert len(first_route["closed_loop_points"]) == 5


def test_observer_atlas_bundle_joins_existing_transport_summary_with_slice_aliases() -> None:
    summary = {
        "summary_type": "observer_slice_transport_summary",
        "status": "OK",
        "records": [
            {
                "source_idx": 0,
                "target_idx": 1,
                "source_slice": "translation_null",
                "target_slice": "observer_recomputed",
                "semantic_first_action": 2.0,
                "observer_first_action": 1.0,
                "commutator_gap": 1.0,
                "holonomy_action": 1.0,
                "relative_holonomy": 1.0,
            }
        ],
        "null_records": [
            {
                "source_idx": 0,
                "target_idx": 1,
                "source_slice": "translation_null",
                "target_slice": "observer_recomputed",
                "holonomy_action": 0.0,
            }
        ],
    }

    atlas = build_observer_atlas_bundle([_observer_bundle(10)], transport_summary=summary)

    route = atlas["routes"][0]
    assert route["source_article_idx"] == 10
    assert route["target_article_idx"] == 20
    assert route["source_slice"] == "translation_null"
    assert route["target_slice"] == "observer_10"
    assert route["null_holonomy_action"] == pytest.approx(0.0)
    assert route["excess_holonomy_action"] == pytest.approx(1.0)


def test_observer_atlas_bundle_joins_null_records_with_explicit_row_identity_only() -> None:
    summary = {
        "summary_type": "observer_slice_transport_summary",
        "status": "OK",
        "records": [
            {
                "source_row_index": 0,
                "target_row_index": 1,
                "source_slice": "translation_null",
                "target_slice": "observer_recomputed",
                "semantic_first_action": 2.0,
                "observer_first_action": 1.0,
                "commutator_gap": 1.0,
                "holonomy_action": 1.0,
                "relative_holonomy": 1.0,
            }
        ],
        "null_records": [
            {
                "source_row_index": 0,
                "target_row_index": 1,
                "source_slice": "translation_null",
                "target_slice": "observer_recomputed",
                "holonomy_action": 0.25,
            }
        ],
    }

    atlas = build_observer_atlas_bundle([_observer_bundle(10)], transport_summary=summary)

    route = atlas["routes"][0]
    assert route["source_row_index"] == 0
    assert route["target_row_index"] == 1
    assert route["target_slice"] == "observer_10"
    assert route["null_holonomy_action"] == pytest.approx(0.25)
    assert route["excess_holonomy_action"] == pytest.approx(0.75)


def test_observer_atlas_bundle_prefers_explicit_article_identity_over_legacy_indices() -> None:
    summary = {
        "summary_type": "observer_slice_transport_summary",
        "status": "OK",
        "records": [
            {
                "source_idx": 1,
                "target_idx": 2,
                "source_article_idx": 10,
                "target_article_idx": 20,
                "source_slice": "global",
                "target_slice": "observer_recomputed",
                "semantic_first_action": 1.0,
                "observer_first_action": 1.0,
                "commutator_gap": 0.0,
                "holonomy_action": 0.0,
                "relative_holonomy": 0.0,
            }
        ],
    }

    atlas = build_observer_atlas_bundle([_observer_bundle(10)], transport_summary=summary)

    route = atlas["routes"][0]
    assert route["source_row_index"] == 0
    assert route["target_row_index"] == 1
    assert route["source_article_idx"] == 10
    assert route["target_article_idx"] == 20


def test_observer_atlas_bundle_labels_fallback_adjacent_pairs() -> None:
    payload = _observer_bundle(10).to_dict(include_nodes=True, include_edges=True)
    payload["edge_action_ledger"] = []

    atlas = build_observer_atlas_bundle([payload], max_article_pairs=1)

    assert atlas["provenance"]["article_pair_source"] == "fallback_adjacent_row_pairs_no_edge_action_ledger"
    assert atlas["routes"]


def test_observer_atlas_bundle_rejects_ambiguous_legacy_transport_indices() -> None:
    bundle = _observer_bundle(10)
    payload = bundle.to_dict(include_nodes=True, include_edges=True)
    for node in payload["nodes"]:
        if node["idx"] == 10:
            node["idx"] = 1
    summary = {
        "summary_type": "observer_slice_transport_summary",
        "status": "OK",
        "records": [
            {
                "source_idx": 1,
                "target_idx": 0,
                "source_slice": "global",
                "target_slice": "observer_recomputed",
                "semantic_first_action": 1.0,
                "observer_first_action": 1.0,
                "commutator_gap": 0.0,
                "holonomy_action": 0.0,
                "relative_holonomy": 0.0,
            }
        ],
    }

    with pytest.raises(ObserverAtlasError, match="ambiguous"):
        build_observer_atlas_bundle([payload], transport_summary=summary)


def test_observer_atlas_bundle_rejects_global_linked_observer_artifacts() -> None:
    with pytest.raises(ObserverAtlasError, match="focused artifact required"):
        build_observer_atlas_bundle(
            [_observer_bundle(10)],
            observer_manifest={"observers": [{"idx": 10, "action": "link"}]},
            require_focused_observer_artifacts=True,
        )


def test_observer_atlas_bundle_rejects_rendered_sidecar_fallback_view_state() -> None:
    payload = _observer_bundle(10).to_dict(include_nodes=True, include_edges=True)
    payload["provenance"]["recenter_mode"] = "sidecar_coordinate_override"
    payload["provenance"]["local_track_recompute_active"] = False

    with pytest.raises(ObserverAtlasError, match="not a focused local recompute"):
        build_observer_atlas_bundle(
            [payload],
            observer_manifest={"observers": [{"idx": 10, "action": "rendered"}]},
            require_focused_observer_artifacts=True,
        )


def test_observer_atlas_bundle_source_fingerprint_changes_when_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "observer_manifest.json"
    source.write_text("first", encoding="utf-8")
    first = build_observer_atlas_bundle([_observer_bundle(10)], source_artifact_paths=[source])
    first_hash = first["source_artifacts"]["fingerprints"]["observer_manifest.json"]["sha256"]

    source.write_text("second", encoding="utf-8")
    second = build_observer_atlas_bundle([_observer_bundle(10)], source_artifact_paths=[source])
    second_hash = second["source_artifacts"]["fingerprints"]["observer_manifest.json"]["sha256"]

    assert first_hash != second_hash


def test_observer_atlas_bundle_round_trips_json(tmp_path: Path) -> None:
    atlas = build_observer_atlas_bundle([_observer_bundle(10)], max_article_pairs=1)
    written = write_observer_atlas_bundle(atlas, tmp_path)

    payload = read_observer_atlas_bundle(written["atlas_bundle"])

    assert payload["bundle_type"] == ATLAS_BUNDLE_TYPE
    assert json.loads(Path(written["atlas_bundle"]).read_text(encoding="utf-8"))["routes"]


def test_slice_transport_emits_row_and_article_identity_separately() -> None:
    result = observer_slice_commutator(
        {
            "left": [[0.0, 0.0], [1.0, 0.0]],
            "right": [[0.0, 0.0], [2.0, 0.0]],
        },
        source_idx=0,
        target_idx=1,
        source_slice="left",
        target_slice="right",
        row_to_article_index=[10, 20],
    )

    assert result["source_row_index"] == 0
    assert result["target_row_index"] == 1
    assert result["source_article_idx"] == 10
    assert result["target_article_idx"] == 20
    assert result["closed_loop"][1]["row_index"] == 1
    assert result["closed_loop"][1]["article_idx"] == 20
