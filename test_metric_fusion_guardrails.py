from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import metric_fusion


def _write_metric_fusion_inputs(tmp_path: Path, embeddings: np.ndarray) -> tuple[Path, Path, Path, Path]:
    gradients = np.linspace(0.1, 1.0, embeddings.shape[0] * embeddings.shape[1], dtype=float).reshape(embeddings.shape)
    metadata = pd.DataFrame(
        {
            "bt_uid": [f"uid_{i}" for i in range(len(embeddings))],
            "title": [f"title_{i}" for i in range(len(embeddings))],
        }
    )

    embeddings_path = tmp_path / "embeddings.npy"
    gradients_path = tmp_path / "gradients.npy"
    metadata_path = tmp_path / "articles.csv"
    output_path = tmp_path / "MONOLITH_DATA.csv"

    np.save(embeddings_path, embeddings)
    np.save(gradients_path, gradients)
    metadata.to_csv(metadata_path, index=False)

    np.save(tmp_path / "walker_work_integrals.npy", np.linspace(5.0, 20.0, len(embeddings), dtype=float))
    (tmp_path / "walker_states.json").write_text(json.dumps(["HONEST"] * len(embeddings)), encoding="utf-8")
    (tmp_path / "phantom_verdicts.json").write_text(
        json.dumps(
            [
                {"verdict": "HONEST", "w_actual": float(5.0 + i), "delta": 1.0 + (i * 0.1), "d_spectral": 1.0 + i}
                for i in range(len(embeddings))
            ]
        ),
        encoding="utf-8",
    )
    return embeddings_path, gradients_path, metadata_path, output_path


def test_calculate_unified_metric_adapts_knn_for_small_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metric_fusion.ArtifactContract, "verify", lambda self: None)
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    embeddings_path, gradients_path, metadata_path, output_path = _write_metric_fusion_inputs(tmp_path, embeddings)

    result = metric_fusion.calculate_unified_metric(
        embeddings_path,
        gradients_path,
        metadata_path,
        output_path,
        knn_k=20,
    )

    assert float(np.ptp(result["density"].to_numpy(dtype=float))) > 1e-9
    assert float(np.ptp(result["z_height"].to_numpy(dtype=float))) > 1e-9
    assert set(["geometry_density_knn", "density_basis", "density_contract", "track3_density_rho"]).issubset(result.columns)
    assert result["density_basis"].iloc[0] == "geometry_knn_track3_missing"


def test_calculate_unified_metric_prefers_track3_rho_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metric_fusion.ArtifactContract, "verify", lambda self: None)
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    embeddings_path, gradients_path, metadata_path, output_path = _write_metric_fusion_inputs(tmp_path, embeddings)
    blinker_std = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [7.0, 0.0],
        ],
        dtype=float,
    )
    np.save(tmp_path / "dirichlet_fused_std.npy", blinker_std)

    result = metric_fusion.calculate_unified_metric(
        embeddings_path,
        gradients_path,
        metadata_path,
        output_path,
        knn_k=20,
    )

    density = result["density"].to_numpy(dtype=float)
    geometry_density = result["geometry_density_knn"].to_numpy(dtype=float)
    rho_raw = result["track3_density_rho"].to_numpy(dtype=float)

    assert result["density_basis"].iloc[0] == "track3_dirichlet_rho:dirichlet_fused_std"
    assert result["density_contract"].iloc[0] == "capstone_track3_dirichlet_rho_primary"
    assert float(np.ptp(rho_raw)) > 1e-9
    assert not np.allclose(density, geometry_density)
    assert int(np.argmax(density)) == 0
    assert int(np.argmin(density)) == 3


def test_calculate_unified_metric_uses_track3_rho_when_geometry_density_collapses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metric_fusion.ArtifactContract, "verify", lambda self: None)
    embeddings = np.ones((4, 3), dtype=float)
    embeddings_path, gradients_path, metadata_path, output_path = _write_metric_fusion_inputs(tmp_path, embeddings)
    blinker_std = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [2.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=float,
    )
    np.save(tmp_path / "dirichlet_fused_std.npy", blinker_std)

    result = metric_fusion.calculate_unified_metric(
        embeddings_path,
        gradients_path,
        metadata_path,
        output_path,
        knn_k=20,
    )

    assert result["density_basis"].iloc[0] == "track3_dirichlet_rho:dirichlet_fused_std"
    assert np.allclose(result["geometry_density_knn"].to_numpy(dtype=float), 0.5)
    assert float(np.ptp(result["density"].to_numpy(dtype=float))) > 1e-9


def test_calculate_unified_metric_refuses_flat_density_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metric_fusion.ArtifactContract, "verify", lambda self: None)
    embeddings = np.ones((4, 3), dtype=float)
    embeddings_path, gradients_path, metadata_path, output_path = _write_metric_fusion_inputs(tmp_path, embeddings)

    with pytest.raises(ValueError, match="Density field collapsed"):
        metric_fusion.calculate_unified_metric(
            embeddings_path,
            gradients_path,
            metadata_path,
            output_path,
            knn_k=20,
        )
