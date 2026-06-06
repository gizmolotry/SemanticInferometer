from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_pair_dominance_ablation import build_pair_dominance_suite


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((7, 4, 5), dtype=np.float32)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.2, 0.0, 0.0],
            [1.1, 0.1, 0.0, 0.2, 0.0],
            [1.6, 0.0, 0.2, 0.0, 0.1],
            [2.0, 0.1, 0.0, 0.1, 0.0],
            [2.5, 0.0, 0.1, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    for obs in range(4):
        cls[:, obs, :] = base + offset + np.asarray([0.0, obs * 0.25, 0.0, 0.0, 0.0], dtype=np.float32)
    cls[:, 2, 1] += np.asarray([0.0, 0.7, 0.0, 0.7, 0.0, 0.7, 0.0], dtype=np.float32)
    torch.save(
        {
            "cls_per_bot": cls,
            "rks_basis_state": {
                "omega": torch.eye(5, dtype=torch.float32),
                "b": torch.zeros(5, dtype=torch.float32),
                "sigma": 1.0,
                "kernel_type": "rbf",
                "seed": 42,
                "hash": "fixture",
            },
            "spectral_probe_magnitudes": np.abs(cls[:, :, :1].mean(axis=1)),
        },
        path,
    )


def test_observer_pair_dominance_suite_writes_artifacts(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.03)

    payload = build_pair_dominance_suite(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        device_name="cpu",
        max_pairs=5,
        neighbor_count=1,
        random_seed=2,
        article_cap=None,
        pair_dim_cap=3,
        configs=[
            {
                "label": "fixture",
                "feature_source": "rks",
                "projection_dim_cap": 5,
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

    assert payload["diagnostic_type"] == "observer_pair_dominance_ablation"
    assert payload["summary"]["run_count"] == 2
    assert payload["summary"]["config_summaries"]
    run = payload["runs"][0]
    assert run["dominance"]["observer_pair_count"] == 12
    assert run["dominance"]["top_pair"]["observer_pair"]
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert claim_matrix["claims"][0]["claim_type"] == "engineering_hypothesis"
