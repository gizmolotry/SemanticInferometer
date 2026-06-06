from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_repair_by_path_class_probe import (
    PATH_CLASSES,
    build_repair_by_path_class_probe,
)


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((18, 3, 4), dtype=np.float32)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.1, 0.0],
            [0.5, 0.0, 0.2, 0.1],
            [0.8, 0.2, 0.0, 0.2],
            [1.0, 0.1, 0.3, 0.0],
            [1.4, 0.3, 0.0, 0.1],
            [1.7, 0.0, 0.2, 0.3],
            [2.1, 0.2, 0.3, 0.0],
            [2.4, 0.1, 0.0, 0.2],
            [2.8, 0.0, 0.2, 0.1],
            [3.2, 0.2, 0.1, 0.3],
            [3.5, 0.0, 0.3, 0.0],
            [3.9, 0.1, 0.0, 0.3],
            [4.2, 0.3, 0.2, 0.0],
            [4.6, 0.1, 0.1, 0.2],
            [5.0, 0.2, 0.0, 0.1],
            [5.3, 0.4, 0.2, 0.2],
            [5.7, 0.1, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    for obs in range(3):
        cls[:, obs, :] = base + offset + np.asarray([0.0, obs * 0.25, 0.0, 0.0], dtype=np.float32)
    cls[:, 1, 2] += np.asarray(
        [0.0, 0.4, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2, 0.0, 0.4, 0.0, 0.1, 0.0, 0.3, 0.0, 0.2, 0.1, 0.3],
        dtype=np.float32,
    )
    torch.save(
        {
            "cls_per_bot": cls,
            "rks_basis_state": {
                "omega": torch.eye(4, dtype=torch.float32),
                "b": torch.zeros(4, dtype=torch.float32),
                "sigma": 1.0,
                "kernel_type": "rbf",
                "seed": 42,
                "hash": "fixture",
            },
            "spectral_probe_magnitudes": np.abs(cls[:, :, :1].mean(axis=1)),
        },
        path,
    )


def test_repair_by_path_class_probe_writes_ledger(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.02)

    payload = build_repair_by_path_class_probe(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        repair_variants=["graph_stable_filter", "edge_level_connection"],
        config_labels=["local_rks_nearest"],
        connection_pair_counts=[2],
        target_anchor_counts=[6],
        anchor_strategy="connection_plus_farthest",
        transform_mode="per_slice_orthogonal",
        repair_mode="procrustes_to_original_slice",
        device_name="cpu",
        max_pairs=16,
        eval_pair_count=5,
        eval_pair_strategy="mixed_hard",
        neighbor_count=2,
        random_seed=13,
        article_cap=None,
        pair_dim_cap=3,
        max_connection_endpoint_fraction=0.4,
        chart_count=3,
        graph_candidate_multiplier=2,
        hybrid_fallback_weight=0.5,
        availability_quantile=0.35,
        availability_action_cutoff=None,
        consensus_fraction=0.75,
        stable_cv_threshold=0.25,
        min_class_pair_count=1,
        recovery_threshold=0.75,
    )

    assert payload["diagnostic_type"] == "observer_repair_by_path_class_probe"
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["run_count"] == 4
    assert payload["summary"]["path_class_repair_summaries"]
    assert payload["summary"]["best_by_config_variant_class"]
    assert {result["path_class"] for run in payload["runs"] for result in run["path_class_results"]} == set(PATH_CLASSES)
    assert all(
        "strict_transfer_pass" in summary and "transductive_oracle_pass" in summary
        for summary in payload["summary"]["path_class_repair_summaries"]
    )
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert claim_matrix["summary_type"] == "observer_repair_by_path_class_claim_matrix"
