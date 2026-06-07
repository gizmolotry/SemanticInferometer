from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_landmark_anchor_strategy_probe import build_landmark_anchor_suite


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((9, 3, 4), dtype=np.float32)
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
        ],
        dtype=np.float32,
    )
    for obs in range(3):
        cls[:, obs, :] = base + offset + np.asarray([0.0, obs * 0.25, 0.0, 0.0], dtype=np.float32)
    cls[:, 1, 2] += np.asarray([0.0, 0.4, 0.0, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0], dtype=np.float32)
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


def test_landmark_anchor_strategy_suite_compares_strategies(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.02)

    payload = build_landmark_anchor_suite(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        anchor_counts=[3, 5],
        anchor_strategies=["random", "farthest_mean", "high_observer_disagreement"],
        transform_modes=["identity", "per_slice_orthogonal"],
        repair_modes=["none", "procrustes_to_original_slice"],
        device_name="cpu",
        max_pairs=5,
        neighbor_count=1,
        random_seed=7,
        article_cap=None,
        pair_dim_cap=3,
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

    assert payload["diagnostic_type"] == "observer_landmark_anchor_strategy_probe"
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["strategy_cell_count"] > 0
    assert payload["summary"]["best_by_cell"]
    strategies = {row["anchor_strategy"] for row in payload["strategy_rows"]}
    assert {"random", "farthest_mean", "high_observer_disagreement"} <= strategies
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert claim_matrix["summary_type"] == "observer_landmark_anchor_strategy_claim_matrix"
