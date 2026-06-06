from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_two_stage_connection_probe import build_two_stage_connection_suite


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((14, 3, 4), dtype=np.float32)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.3, 0.2, 0.0, 0.1],
            [0.7, 0.0, 0.3, 0.0],
            [1.1, 0.2, 0.0, 0.2],
            [1.6, 0.1, 0.2, 0.0],
            [2.0, 0.3, 0.0, 0.3],
            [2.5, 0.0, 0.2, 0.1],
            [3.0, 0.2, 0.3, 0.0],
            [3.4, 0.1, 0.0, 0.2],
            [3.9, 0.0, 0.2, 0.3],
            [4.2, 0.2, 0.1, 0.0],
            [4.7, 0.0, 0.3, 0.2],
            [5.1, 0.1, 0.0, 0.3],
            [5.5, 0.3, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    for obs in range(3):
        cls[:, obs, :] = base + offset + np.asarray([0.0, obs * 0.25, 0.0, 0.0], dtype=np.float32)
    cls[:, 1, 2] += np.asarray(
        [0.0, 0.4, 0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0, 0.2],
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


def test_two_stage_connection_suite_uses_disjoint_eval_routes(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.02)

    payload = build_two_stage_connection_suite(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        connection_pair_counts=[2],
        target_anchor_counts=[None, 6],
        anchor_strategies=["connection_endpoints", "connection_plus_near_shell"],
        transform_mode="per_slice_orthogonal",
        repair_mode="procrustes_to_original_slice",
        device_name="cpu",
        max_pairs=12,
        eval_pair_count=4,
        neighbor_count=2,
        random_seed=11,
        article_cap=None,
        pair_dim_cap=3,
        max_connection_endpoint_fraction=0.45,
        eval_pair_strategy="mixed_hard",
        configs=[
            {
                "label": "fixture",
                "feature_source": "rks",
                "projection_dim_cap": 4,
                "normalization": "none",
                "pair_mode": "mixed",
                "null_mode": "zero",
                "weight_profile": "default_action",
                "semantic_weight": 1.0,
                "observer_switch_weight": 1.0,
                "stress_weight": 0.25,
                "density_weight": 0.15,
            }
        ],
    )

    assert payload["diagnostic_type"] == "observer_two_stage_connection_probe"
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["run_count"] == 8
    assert payload["config"]["eval_pair_strategy"] == "mixed_hard"
    assert payload["summary"]["two_stage_summaries"]
    assert all(
        summary["mean_connection_eval_endpoint_overlap"] == 0.0
        for summary in payload["summary"]["two_stage_summaries"]
    )
    assert all(
        run["split"]["eval_pair_strategy"] == "mixed_hard"
        for run in payload["runs"]
    )
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert claim_matrix["summary_type"] == "observer_two_stage_connection_claim_matrix"
