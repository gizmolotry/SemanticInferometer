import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_track4_action_graph import _select_observer_simplex
from scripts.run_track4_pipeline_basis_probe import (
    ProbeConfig,
    basis_output_dir,
    build_basis_command,
    probe_claim_boundary,
    rank_basis_rows,
    summarize_basis_run,
    write_probe_summary,
)


def _write_minimal_cyclic_paths(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    terrain_metadata = json.dumps({"terrain_labels": ["Bridge", "Tightrope", "Void"]})
    np.savez(
        run_dir / "cyclic_paths.npz",
        work_integral=np.asarray([10.0, 12.0, 50.0, 48.0], dtype=float),
        closed_loop=np.asarray([True, True, True, True], dtype=bool),
        path_anchor_idx=np.asarray([0, 0, 2, 2], dtype=int),
        anchor_indices=np.asarray([0, 2], dtype=int),
        path_is_hot=np.asarray([True, True, False, False], dtype=bool),
        path_indices=np.asarray([[0, 1], [0, 1, 2], [2, 1], [2, 1, 0]], dtype=object),
        path_feature_basis=np.asarray(["logits_flat"] * 4, dtype=object),
        path_proposal_mode=np.asarray(["metric_softmax"] * 4, dtype=object),
        anchor_selection_metadata=np.asarray([terrain_metadata], dtype=object),
    )


def test_action_graph_runner_derives_v_observer_simplex_from_bot_cls_artifacts():
    cls_per_bot = np.zeros((3, 8, 4), dtype=np.float32)
    cls_per_bot[:, :, 0] = np.arange(1, 9, dtype=np.float32)
    cls_per_bot[1, 3, :] = 20.0
    cls_per_bot[2, 7, :] = 30.0

    simplex = _select_observer_simplex({"cls_per_bot": cls_per_bot})

    assert simplex is not None
    assert simplex.shape == (3, 8)
    assert np.allclose(simplex.sum(axis=1), 1.0)
    assert simplex[1, 3] == np.max(simplex[1])
    assert simplex[2, 7] == np.max(simplex[2])


def test_action_graph_runner_prefers_spectral_probe_magnitude_simplex():
    magnitudes = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    simplex = _select_observer_simplex({"spectral_probe_magnitudes": magnitudes})

    assert simplex is not None
    assert simplex.shape == (2, 8)
    assert np.allclose(simplex.sum(axis=1), 1.0)
    assert simplex[0, 0] == np.max(simplex[0])
    assert simplex[1, 4] == np.max(simplex[1])


def test_build_basis_command_threads_track4_basis_and_runtime_knobs(tmp_path: Path):
    config = ProbeConfig(
        corpus=tmp_path / "corpus.jsonl",
        output_root=tmp_path / "out",
        seed=420,
        limit=11,
        kernel_type="imq",
        nli_cache_path=tmp_path / "cache.pt",
        adaptive_tpt_connectivity=True,
    )

    command = build_basis_command(config, "logits_flat")
    joined = " ".join(command)

    assert "--track4-basis logits_flat" in joined
    assert "--track4-adaptive-tpt-connectivity" in joined
    assert "--nli-cache-path" in command
    assert str(basis_output_dir(config.output_root, "logits_flat")) in command
    assert "--no-freeze-good-run" in command


def test_summarize_basis_run_reads_observer_runtime_and_npz_basis(tmp_path: Path):
    output_root = tmp_path / "probe"
    run_dir = basis_output_dir(output_root, "logits_flat")
    _write_minimal_cyclic_paths(run_dir)
    torch.save(
        {
            "track4_runtime_config": {
                "requested_basis": "logits_flat",
                "effective_basis": "logits_flat",
                "basis_embedding_dim": 24,
                "proposal_mode": "metric_softmax",
                "adaptive_tpt_connectivity": True,
                "effective_k_neighbors": 8,
            },
            "track4_markov_observables": {
                "status": "OK",
                "summary": {"reactive_flux_total": 0.5, "committor_mean": 0.25},
            },
        },
        run_dir / "observer_42.pt",
    )

    row = summarize_basis_run(output_root, "logits_flat", 42)

    assert row["observer_exists"] is True
    assert row["requested_basis"] == "logits_flat"
    assert row["effective_basis"] == "logits_flat"
    assert row["basis_embedding_dim"] == 24
    assert row["feature_basis_counts"] == {"logits_flat": 4}
    assert row["proposal_mode_counts"] == {"metric_softmax": 4}
    assert row["reactive_flux_total"] == 0.5


def test_run_probe_return_includes_ranked_rows_for_aggregate_consumers(tmp_path: Path):
    from scripts.run_track4_pipeline_basis_probe import run_probe

    output_root = tmp_path / "probe"
    run_dir = basis_output_dir(output_root, "logits_flat")
    _write_minimal_cyclic_paths(run_dir)
    torch.save(
        {
            "track4_runtime_config": {
                "requested_basis": "logits_flat",
                "effective_basis": "logits_flat",
                "basis_embedding_dim": 24,
                "proposal_mode": "metric_softmax",
            }
        },
        run_dir / "observer_42.pt",
    )
    config = ProbeConfig(
        corpus=tmp_path / "corpus.jsonl",
        output_root=output_root,
        seed=42,
    )

    result = run_probe(config, ["logits_flat"], skip_existing=True)

    assert result["ranked_rows"]
    assert result["ranked_rows"][0]["basis"] == "logits_flat"
    assert "basis_probe_score" in result["ranked_rows"][0]
    assert result["claim_boundary"]["instrumentation_supported"] is True


def test_write_probe_summary_emits_json_and_csv(tmp_path: Path):
    rows = [
        {
            "basis": "track2",
            "run_dir": str(tmp_path / "track2"),
            "observer_exists": True,
            "effective_basis": "track2",
            "path_count": 4,
            "traversal_status": "OK",
            "safe_for_thesis_claim": False,
        }
    ]

    out_path = write_probe_summary(tmp_path, rows, [["python", "run_experiments.py"]])

    assert out_path.exists()
    assert (tmp_path / "track4_pipeline_basis_probe_summary.csv").exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["probe_type"] == "track4_pipeline_feature_basis"
    assert payload["basis_count"] == 1
    assert payload["recommended_basis"] == "track2"
    assert payload["ranked_rows"][0]["basis_probe_score"] >= 0.0
    assert payload["claim_boundary"]["instrumentation_supported"] is True
    assert payload["claim_boundary"]["terrain_validity_supported"] is False


def test_rank_basis_rows_prefers_more_diverse_valid_traversal():
    rows = [
        {
            "basis": "flat",
            "safe_for_thesis_claim": False,
            "path_shape_entropy_norm": 0.0,
            "path_edge_entropy_norm": 0.0,
            "closed_loop_rate": 0.0,
            "primary_zone_count": 1,
            "reactive_flux_total": 0.0,
        },
        {
            "basis": "rich",
            "safe_for_thesis_claim": True,
            "path_shape_entropy_norm": 0.8,
            "path_edge_entropy_norm": 0.7,
            "closed_loop_rate": 0.9,
            "primary_zone_count": 4,
            "reactive_flux_total": 0.1,
        },
    ]

    ranked = rank_basis_rows(rows)

    assert ranked[0]["basis"] == "rich"
    assert ranked[0]["basis_probe_score"] > ranked[1]["basis_probe_score"]


def test_probe_claim_boundary_rejects_tiny_unsafe_basis_margin():
    ranked = [
        {
            "basis": "logits_flat",
            "observer_exists": True,
            "effective_basis": "logits_flat",
            "path_count": 15,
            "basis_probe_score": 0.7636,
            "safe_for_thesis_claim": False,
            "failure_reasons": ["bridge/void work-integral gap below minimum semantic effect size"],
        },
        {
            "basis": "track2",
            "observer_exists": True,
            "effective_basis": "track2",
            "path_count": 15,
            "basis_probe_score": 0.7627,
            "safe_for_thesis_claim": False,
            "failure_reasons": ["track 4 path-touched terrain coverage fewer than three terrain zones"],
        },
    ]

    boundary = probe_claim_boundary(ranked)

    assert boundary["instrumentation_supported"] is True
    assert boundary["terrain_validity_supported"] is False
    assert boundary["basis_superiority_supported"] is False
    assert boundary["basis_claim_status"] == "tentative_margin_too_small"
    assert boundary["basis_score_margin"] < boundary["minimum_basis_score_margin"]


def test_probe_claim_boundary_allows_large_safe_basis_candidate():
    ranked = [
        {
            "basis": "logits_flat",
            "observer_exists": True,
            "effective_basis": "logits_flat",
            "path_count": 60,
            "basis_probe_score": 0.91,
            "safe_for_thesis_claim": True,
        },
        {
            "basis": "track2",
            "observer_exists": True,
            "effective_basis": "track2",
            "path_count": 60,
            "basis_probe_score": 0.80,
            "safe_for_thesis_claim": True,
        },
    ]

    boundary = probe_claim_boundary(ranked)

    assert boundary["terrain_validity_supported"] is True
    assert boundary["basis_superiority_supported"] is True
    assert boundary["basis_claim_status"] == "basis_superiority_candidate"
