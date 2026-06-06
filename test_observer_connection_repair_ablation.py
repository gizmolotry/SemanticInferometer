from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_observer_connection_repair_ablation import (
    REPAIR_VARIANTS,
    build_connection_repair_ablation_suite,
)


def _write_payload(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = np.zeros((16, 3, 4), dtype=np.float32)
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
        ],
        dtype=np.float32,
    )
    for obs in range(3):
        cls[:, obs, :] = base + offset + np.asarray([0.0, obs * 0.22, 0.0, 0.0], dtype=np.float32)
    cls[:, 1, 2] += np.asarray(
        [0.0, 0.4, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2, 0.0, 0.4, 0.0, 0.1, 0.0, 0.3, 0.0, 0.2],
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


def test_connection_repair_ablation_writes_five_variant_ledger(tmp_path: Path) -> None:
    real = tmp_path / "rbf" / "cls" / "real" / "observer_global.pt"
    control = tmp_path / "matern" / "cls" / "control_random" / "observer_global.pt"
    _write_payload(real)
    _write_payload(control, offset=0.01)

    payload = build_connection_repair_ablation_suite(
        payload_paths=[real, control],
        output_dir=tmp_path / "out",
        repair_variants=REPAIR_VARIANTS,
        config_labels=["local_rks_nearest"],
        connection_pair_counts=[2],
        target_anchor_counts=[6],
        anchor_strategy="connection_plus_farthest",
        transform_mode="per_slice_orthogonal",
        repair_mode="procrustes_to_original_slice",
        device_name="cpu",
        max_pairs=14,
        eval_pair_count=4,
        eval_pair_strategy="mixed_hard",
        neighbor_count=2,
        random_seed=13,
        article_cap=None,
        pair_dim_cap=3,
        max_connection_endpoint_fraction=0.4,
        chart_count=3,
        graph_candidate_multiplier=2,
        hybrid_fallback_weight=0.5,
    )

    assert payload["diagnostic_type"] == "observer_connection_repair_ablation"
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["run_count"] == 2 * len(REPAIR_VARIANTS)
    assert payload["summary"]["repair_summaries"]
    assert {row["repair_variant"] for row in payload["runs"]} == set(REPAIR_VARIANTS)
    assert all("strict_transfer_pass" in row for row in payload["summary"]["repair_summaries"])
    assert all("transductive_oracle_pass" in row for row in payload["summary"]["repair_summaries"])
    assert payload["config"]["eval_pair_strategy"] == "mixed_hard"
    assert all(run["connection_eval_endpoint_overlap"] == 0 for run in payload["runs"])
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert claim_matrix["summary_type"] == "observer_connection_repair_ablation_claim_matrix"
