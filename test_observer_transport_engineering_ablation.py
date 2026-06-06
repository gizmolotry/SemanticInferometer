from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_transport_engineering_ablation import (
    build_engineering_ablation,
    _config_id,
    _null_slices,
)


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.1, 0.0, 0.0],
            [0.9, 0.0, 0.2, 0.0],
            [1.4, 0.2, 0.0, 0.1],
            [2.0, 0.0, 0.3, 0.0],
            [2.5, 0.1, 0.0, 0.3],
        ],
        dtype=np.float32,
    )
    cls = np.zeros((6, 3, 4), dtype=np.float32)
    cls[:, 0, :] = base + offset
    cls[:, 1, :] = base + np.asarray([0.0, 0.7, 0.0, 0.0], dtype=np.float32) + offset
    cls[:, 2, :] = base + np.asarray([0.2, -0.5, 0.0, 0.0], dtype=np.float32) + offset
    cls[:, 1, 1] += np.asarray([0.0, 0.8, 0.0, 0.8, 0.0, 0.8], dtype=np.float32)
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


def test_engineering_ablation_writes_json_and_csv(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.05)

    artifact = build_engineering_ablation(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        feature_sources=["rks", "raw_cls"],
        projection_dim_caps=[2, 4],
        normalizations=["none", "center_per_slice"],
        pair_modes=["nearest", "random"],
        null_modes=["zero", "independent_article_shuffle"],
        max_configs=10,
        max_pairs=4,
        neighbor_count=1,
        random_seed=3,
        article_cap=None,
        pair_dim_cap=2,
        confirm_top=2,
        confirm_payload_paths=[real, control],
        confirm_max_pairs=5,
        confirm_article_cap=None,
        device_name="cpu",
        top_records=0,
    )

    assert artifact["diagnostic_type"] == "observer_transport_engineering_ablation"
    assert artifact["screen"]["row_count"] > 0
    assert artifact["confirm"]["row_count"] > 0
    assert artifact["engineering_recommendations"]
    assert Path(artifact["artifacts"]["json"]).exists()
    assert Path(artifact["artifacts"]["csv"]).exists()
    saved = json.loads(Path(artifact["artifacts"]["json"]).read_text(encoding="utf-8"))
    assert saved["config_space"]["screen_config_count"] == 10
    assert "null_mode" in saved["screen"]["axis_summary"]
    top = saved["confirm"]["config_summaries"][0]
    assert "real_minus_control_mean_excess_holonomy_action" in top


def test_null_slices_are_deterministic_and_config_ids_include_null_mode() -> None:
    slices = {
        "a": np.arange(12, dtype=np.float64).reshape(3, 4),
        "b": np.arange(12, 24, dtype=np.float64).reshape(3, 4),
    }
    first = _null_slices(slices, mode="independent_article_shuffle", random_seed=11)
    second = _null_slices(slices, mode="independent_article_shuffle", random_seed=11)
    assert first is not None
    assert second is not None
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert _config_id(
        {
            "feature_source": "rks",
            "projection_dim_cap": 4,
            "normalization": "none",
            "pair_mode": "nearest",
            "weight_profile": "semantic_only",
            "null_mode": "zero",
        }
    ).endswith("|null=zero")
