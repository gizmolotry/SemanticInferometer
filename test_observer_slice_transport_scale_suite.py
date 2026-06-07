from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.observer_slice_transport import ObserverSliceTransportConfig
from scripts.run_observer_slice_transport_scale_suite import build_suite


def _write_payload(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((6, 3, 4), dtype=np.float32)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cls[:, 0, :] = base
    cls[:, 1, :] = base + np.asarray([0.0, 0.8, 0.0, 0.0], dtype=np.float32)
    cls[:, 1, 1] += np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    cls[:, 2, :] = base + np.asarray([0.0, -0.4, 0.0, 0.0], dtype=np.float32)
    omega = torch.eye(4, dtype=torch.float32)
    torch.save(
        {
            "cls_per_bot": cls,
            "rks_basis_state": {
                "omega": omega,
                "b": torch.zeros(4, dtype=torch.float32),
                "sigma": 1.0,
                "kernel_type": "rbf",
                "seed": 7,
                "hash": "fixture",
            },
            "spectral_probe_magnitudes": np.abs(cls[:, :, :1].mean(axis=1)),
        },
        path,
    )


def test_scale_suite_builds_compact_holonomy_artifact(tmp_path: Path) -> None:
    payload_path = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    _write_payload(payload_path)
    out = tmp_path / "out"

    artifact = build_suite(
        [payload_path],
        output_dir=out,
        max_pairs=5,
        neighbor_count=2,
        projection_dim_cap=None,
        device_name="cpu",
        random_seed=1,
        top_records=3,
        config=ObserverSliceTransportConfig(),
    )

    assert artifact["diagnostic_type"] == "observer_slice_transport_scale_suite"
    assert artifact["successful_payload_count"] == 1
    assert artifact["runs"][0]["n_slices"] == 3
    assert artifact["runs"][0]["transport"]["record_count"] > 0
    assert len(artifact["runs"][0]["transport"]["top_records"]) == 3
    assert Path(artifact["artifact"]).exists()
    saved = json.loads(Path(artifact["artifact"]).read_text(encoding="utf-8"))
    assert saved["claim_boundary"]["real_corpus_claim_is_geometry_only"] is True


def test_scale_suite_reports_payload_failures_without_crashing(tmp_path: Path) -> None:
    artifact = build_suite(
        [tmp_path / "missing.pt"],
        output_dir=tmp_path / "out",
        max_pairs=2,
        neighbor_count=1,
        projection_dim_cap=4,
        device_name="cpu",
        random_seed=1,
        top_records=1,
        config=ObserverSliceTransportConfig(),
    )

    assert artifact["successful_payload_count"] == 0
    assert artifact["failed_payload_count"] == 1
    assert artifact["transport_supported"] is False
    assert artifact["failures"][0]["payload_path"].endswith("missing.pt")
