from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_path_contingency_probe import build_path_contingency_probe


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((15, 3, 4), dtype=np.float32)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, 0.1],
            [0.6, 0.0, 0.2, 0.0],
            [1.0, 0.2, 0.0, 0.2],
            [1.5, 0.1, 0.3, 0.0],
            [1.9, 0.3, 0.0, 0.1],
            [2.4, 0.0, 0.2, 0.3],
            [2.8, 0.2, 0.3, 0.0],
            [3.3, 0.1, 0.0, 0.2],
            [3.7, 0.0, 0.2, 0.1],
            [4.1, 0.2, 0.1, 0.3],
            [4.5, 0.0, 0.3, 0.0],
            [4.9, 0.1, 0.0, 0.3],
            [5.2, 0.3, 0.2, 0.0],
            [5.6, 0.1, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    for obs in range(3):
        cls[:, obs, :] = base + offset + np.asarray([0.0, obs * 0.2, 0.0, 0.0], dtype=np.float32)
    cls[:, 2, 0] *= 1.35
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


def test_path_contingency_probe_writes_ledger(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.02)

    payload = build_path_contingency_probe(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        config_labels=["local_rks_nearest"],
        device_name="cpu",
        max_pairs=12,
        eval_pair_count=4,
        eval_pair_strategy="mixed_hard",
        connection_pair_count=2,
        neighbor_count=2,
        random_seed=17,
        article_cap=10,
        pair_dim_cap=3,
        max_connection_endpoint_fraction=0.4,
        availability_quantile=0.5,
        availability_action_cutoff=None,
        consensus_fraction=0.75,
        stable_cv_threshold=0.25,
    )

    assert payload["diagnostic_type"] == "observer_path_contingency_probe"
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["run_count"] == 2
    assert payload["summary"]["path_contingency_summaries"]
    assert all("path_contingency" in run for run in payload["runs"])
    assert all("mean_observer_subset_required_rate" in row for row in payload["summary"]["path_contingency_summaries"])
    assert all("mean_excess_observer_contingent_rate" in row for row in payload["summary"]["path_contingency_summaries"])
    assert all(run["row_to_article_index"] for run in payload["runs"])
    first_run = payload["runs"][0]
    first_record = first_run["path_contingency"]["records"][0]
    assert first_record["source_article_idx"] in first_run["row_to_article_index"]
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert claim_matrix["summary_type"] == "observer_path_contingency_claim_matrix"


def test_path_contingency_probe_rejects_unknown_config_label(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    _write_payload(real)

    try:
        build_path_contingency_probe(
            payload_paths=[real],
            output_dir=tmp_path / "out",
            config_labels=["not_a_config"],
            device_name="cpu",
            max_pairs=4,
            eval_pair_count=2,
            eval_pair_strategy="ordered",
            connection_pair_count=1,
            neighbor_count=1,
            random_seed=17,
            article_cap=None,
            pair_dim_cap=3,
            max_connection_endpoint_fraction=0.4,
            availability_quantile=0.5,
            availability_action_cutoff=None,
            consensus_fraction=0.75,
            stable_cv_threshold=0.25,
        )
    except ValueError as exc:
        assert "unknown config labels" in str(exc)
    else:
        raise AssertionError("expected unknown config label to fail")
