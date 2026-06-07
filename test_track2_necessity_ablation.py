from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_track2_necessity_ablation import (
    DIAGNOSTIC_TYPE,
    build_artifact,
    exact_kernel,
    kernel_correlation,
    relative_frobenius_error,
    rks_approx_kernel,
    rks_features,
    write_outputs,
)


def test_rks_approx_kernel_uses_raw_feature_dot_product() -> None:
    rng = np.random.default_rng(123)
    points = rng.normal(size=(8, 3))
    exact = exact_kernel(points, "rbf")
    features = rks_features(points, kernel="rbf", dim=2048, seed=42)
    approx = rks_approx_kernel(features)

    assert np.allclose(np.diag(approx), 1.0)
    assert kernel_correlation(approx, exact) > 0.98
    assert relative_frobenius_error(approx, exact) < 0.08


def _write_cell(root: Path, cell: str) -> None:
    run_dir = root / cell
    (run_dir / "labels").mkdir(parents=True)
    labels = ["alpha", "alpha", "alpha", "beta", "beta", "beta"]
    with (run_dir / "labels" / "hidden_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", "group_topic"])
        writer.writeheader()
        for idx, label in enumerate(labels):
            writer.writerow({"article_id": idx, "group_topic": label})

    track2 = np.array(
        [
            [2.0, 0.0, 0.1],
            [2.1, 0.1, 0.0],
            [1.9, -0.1, 0.0],
            [-2.0, 0.0, -0.1],
            [-2.1, -0.1, 0.0],
            [-1.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    track15 = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.1, -0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
            [-1.1, 0.1],
        ],
        dtype=np.float32,
    )
    full = np.concatenate([track2, track15], axis=1).astype(np.float32)
    cls_per_bot = np.stack([track2 + bot * 0.01 for bot in range(4)], axis=1).astype(np.float32)

    np.save(run_dir / "dirichlet_fused.npy", track2)
    np.save(run_dir / "spectral_probe_magnitudes.npy", track15)
    np.save(run_dir / "integrated_vectors.npy", full)
    torch.save({"cls_per_bot": cls_per_bot}, run_dir / "observer_global.pt")


def test_build_artifact_and_write_outputs_from_minimal_ledger(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    _write_cell(synthetic_root, "rbf_seed7")

    artifact = build_artifact(
        synthetic_root=synthetic_root,
        kernels=["rbf"],
        seeds=[7],
        rks_dims=[32],
        rks_seeds=[3],
        label_column="group_topic",
    )
    written = write_outputs(artifact, tmp_path / "out")

    assert artifact["diagnostic_type"] == DIAGNOSTIC_TYPE
    assert artifact["foundation_audit_consumable"] is True
    assert artifact["cell_summaries"][0]["track2_shape"] == [6, 3]
    assert "q2_track2_preserves_planted_synthetic_structure" in artifact["question_results"]
    assert set(written) == {"json", "csv", "markdown", "ablation_matrix"}

    saved = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    matrix = json.loads(Path(written["ablation_matrix"]).read_text(encoding="utf-8"))
    assert saved["records"]
    assert matrix["records"][0]["ablation"] == "track2_removed"
    assert Path(written["csv"]).read_text(encoding="utf-8").startswith("record_type,")
