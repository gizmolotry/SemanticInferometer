import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_observer_recenter_robustness_suite import (
    LOCAL_VARIANT_BASELINES,
    SOURCE_PROXY_METRIC_BASELINE,
    aggregate_results,
    evaluate_baseline,
    run_suite,
)


def _write_small_synthetic_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    global_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.1, 0.1),
        (1.1, 0.1),
        (0.1, 1.1),
    ]
    articles = [{"idx": idx, "x": x, "y": y} for idx, (x, y) in enumerate(global_points)]
    (run_dir / "MONOLITH.view_state.json").write_text(
        json.dumps({"articles": articles}),
        encoding="utf-8",
    )
    rows = ["index,perspective_tag,source,zone,density,stress,w_actual"]
    labels = ["frame_a", "frame_a", "frame_a", "frame_b", "frame_b", "frame_b"]
    for idx, label in enumerate(labels):
        rows.append(f"{idx},{label},source_{idx},Bridge,0.8,0.2,{1.0 + idx}")
    (run_dir / "MONOLITH_DATA.csv").write_text("\n".join(rows), encoding="utf-8")
    features = np.asarray(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.1],
            [0.0, 0.1, 0.1],
            [4.0, 0.0, 0.9],
            [4.1, 0.0, 0.9],
            [4.0, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    np.save(run_dir / "features.npy", features)


def _write_observer_payload(run_dir: Path) -> None:
    torch = pytest.importorskip("torch")
    cls = []
    for idx in range(6):
        label_axis = -1.0 if idx < 3 else 1.0
        observer_views = []
        for bot in range(4):
            observer_views.append(
                [
                    label_axis + (0.04 * bot),
                    float(idx % 3) * 0.1,
                    (bot - 1.5) * label_axis,
                    float(idx) * 0.02,
                ]
            )
        cls.append(observer_views)
    payload = {
        "cls_per_bot": torch.tensor(cls, dtype=torch.float32),
        "spectral_probe_magnitudes": torch.tensor(
            [
                [1.0, 0.2, 0.1, 0.1],
                [0.9, 0.2, 0.1, 0.1],
                [0.8, 0.3, 0.1, 0.1],
                [0.1, 0.1, 0.2, 1.0],
                [0.1, 0.1, 0.3, 0.9],
                [0.1, 0.1, 0.2, 0.8],
            ],
            dtype=torch.float32,
        ),
    }
    torch.save(payload, run_dir / "observer_global.pt")


def test_raw_track2_baseline_runs_on_existing_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic" / "rbf_seed42"
    _write_small_synthetic_run(run_dir)

    result = evaluate_baseline(
        run_dir,
        "raw_track2_pca",
        min_label_count=2,
        preferred_label_column="perspective_tag",
        label_mode="auto",
    )

    assert result["baseline"] == "raw_track2_pca"
    assert result["status"] == "OK"
    assert result["observer_count"] == 2
    assert result["label_column"] == "perspective_tag"
    assert "ideological_validation_suite" in result


def test_local_variant_baseline_runs_and_preserves_variant_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic" / "rbf_seed42"
    _write_small_synthetic_run(run_dir)
    _write_observer_payload(run_dir)
    baseline = "local_track_recompute:local_tangent_pca"

    assert LOCAL_VARIANT_BASELINES[baseline] == "local_tangent_pca"

    result = evaluate_baseline(
        run_dir,
        baseline,
        min_label_count=2,
        preferred_label_column="perspective_tag",
        label_mode="auto",
    )

    assert result["baseline"] == baseline
    assert result["status"] == "OK"
    assert result["local_recompute_variant"] == "local_tangent_pca"
    assert result["observer_count"] == 2
    assert result["local_track_recompute_supported_count"] == 2
    assert {
        row.get("local_recompute_variant")
        for row in result.get("observers", [])
    } == {"local_tangent_pca"}


def test_source_proxy_metric_is_oracle_and_never_safe_recenter_claim(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic" / "rbf_seed42"
    _write_small_synthetic_run(run_dir)

    result = evaluate_baseline(
        run_dir,
        SOURCE_PROXY_METRIC_BASELINE,
        min_label_count=2,
        preferred_label_column="perspective_tag",
        label_mode="auto",
    )

    assert result["status"] == "OK"
    assert result["local_recompute_variant"] == "source_proxy_metric"
    assert result["proxy_oracle_baseline"] is True
    assert result["safe_for_recenter_claim"] is False
    assert "not_pure_local_recompute" in result["claim_boundary"]


def test_robustness_suite_writes_json_and_csv(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    _write_small_synthetic_run(synthetic_root / "rbf_seed42")
    out_dir = tmp_path / "out"

    payload = run_suite(
        synthetic_root=synthetic_root,
        output_dir=out_dir,
        kernels=["rbf"],
        seeds=[42],
        baselines=["translation_only", "raw_track2_pca"],
        min_label_count=2,
        preferred_label_column="perspective_tag",
        label_mode="auto",
    )

    assert payload["summary_type"] == "observer_recenter_robustness_suite"
    assert payload["cell_count"] == 1
    assert (out_dir / "observer_recenter_robustness_suite.json").exists()
    assert (out_dir / "observer_recenter_robustness_suite.csv").exists()
    assert "raw_track2_pca" in payload["aggregate"]


def test_aggregate_results_counts_baseline_passes() -> None:
    aggregate = aggregate_results(
        [
            {
                "baselines": [
                    {
                        "baseline": "a",
                        "safe_for_recenter_claim": True,
                        "observer_count": 2,
                        "label_contraction": {"primary_mean_label_gap_gain_over_translation": 0.2},
                        "ideological_validation_suite": {"status": "PASS", "tests": {}},
                    },
                    {
                        "baseline": "a",
                        "safe_for_recenter_claim": False,
                        "observer_count": 1,
                        "label_contraction": {"primary_mean_label_gap_gain_over_translation": 0.0},
                        "ideological_validation_suite": {"status": "FAIL", "tests": {}},
                    },
                ]
            }
        ]
    )

    assert aggregate["a"]["cell_count"] == 2
    assert aggregate["a"]["pass_count"] == 1
    assert aggregate["a"]["pass_rate"] == 0.5
