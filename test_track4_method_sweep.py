import json

import numpy as np

from scripts.track4_method_sweep import main, run_sweep


def test_track4_method_sweep_ranks_lab_proposal_modes(tmp_path):
    rng = np.random.default_rng(123)
    features = rng.normal(size=(11, 6)).astype(np.float32)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(feature_path, cls_stacked=features)

    report = run_sweep(
        feature_path=feature_path,
        output_dir=tmp_path / "sweep",
        feature_key="cls_stacked",
        modes=["metric_softmax", "stress_biased"],
        seed=5,
        max_steps=12,
        k_neighbors=4,
    )

    assert report["status"] == "OK"
    assert [row["proposal_mode"] for row in report["ranked_modes"]]
    assert {row["proposal_mode"] for row in report["mode_rows"]} == {"metric_softmax", "stress_biased"}
    assert all("route_entropy_component" in row["score"] for row in report["mode_rows"])
    for row in report["mode_rows"]:
        mode_dir = tmp_path / "sweep" / row["proposal_mode"]
        assert (mode_dir / "cyclic_paths.npz").exists()
        assert (mode_dir / "track4_method_summary.json").exists()
        with np.load(mode_dir / "cyclic_paths.npz", allow_pickle=True) as payload:
            assert "path_proposal_mode" in payload.files
            assert "path_xyz" not in payload.files
            assert set(map(str, payload["path_proposal_mode"].tolist())) == {row["proposal_mode"]}

    saved = json.loads((tmp_path / "sweep" / "track4_method_sweep_summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "OK"
    assert saved["compact_artifacts"] is True
    assert len(saved["ranked_modes"]) == 2
    assert (tmp_path / "sweep" / "track4_ranked_modes.csv").exists()
    assert "path_shape_entropy_norm" in report["mode_rows"][0]["summary"]


def test_track4_method_sweep_cli_supports_k_grid(tmp_path, monkeypatch):
    rng = np.random.default_rng(321)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(feature_path, cls_stacked=rng.normal(size=(11, 5)).astype(np.float32))
    output_dir = tmp_path / "grid"
    monkeypatch.setattr(
        "sys.argv",
        [
            "track4_method_sweep.py",
            "--features",
            str(feature_path),
            "--output-dir",
            str(output_dir),
            "--modes",
            "metric_softmax",
            "--k-grid",
            "4",
            "5",
            "--max-steps",
            "8",
        ],
    )

    assert main() == 0

    saved = json.loads((output_dir / "track4_kgrid_method_sweep_summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "OK"
    assert saved["k_grid"] == [4, 5]
    assert len(saved["ranked_k_modes"]) == 2
    assert saved["temperature_grid"] == [0.75]
    assert saved["gamma_grid"] == [5.0]
    assert (output_dir / "track4_ranked_conditions.csv").exists()
    assert (output_dir / "track4_mode_robustness.csv").exists()


def test_track4_method_sweep_cli_supports_seed_temperature_gamma_grid(tmp_path, monkeypatch):
    rng = np.random.default_rng(654)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(feature_path, cls_stacked=rng.normal(size=(11, 5)).astype(np.float32))
    output_dir = tmp_path / "robust"
    monkeypatch.setattr(
        "sys.argv",
        [
            "track4_method_sweep.py",
            "--features",
            str(feature_path),
            "--output-dir",
            str(output_dir),
            "--modes",
            "metric_softmax",
            "committor_guided",
            "--seed-grid",
            "1,2",
            "--temperature-grid",
            "0.5",
            "1.0",
            "--gamma-grid",
            "3.0",
            "--k-neighbors",
            "4",
            "--max-steps",
            "6",
            "--adaptive-tpt-connectivity",
        ],
    )

    assert main() == 0

    saved = json.loads((output_dir / "track4_grid_method_sweep_summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "OK"
    assert saved["seeds"] == [1, 2]
    assert saved["temperature_grid"] == [0.5, 1.0]
    assert saved["gamma_grid"] == [3.0]
    assert saved["adaptive_tpt_connectivity"] is True
    assert saved["compact_artifacts"] is True
    assert saved["condition_count"] == 4
    assert {row["proposal_mode"] for row in saved["mode_robustness"]} == {"metric_softmax", "committor_guided"}
    assert len(saved["ranked_conditions"]) == 8
    assert "path_shape_entropy_norm" in saved["ranked_conditions"][0]
    assert "mean_path_edge_count" in saved["ranked_conditions"][0]
    assert saved["method_recommendation"]["recommended_default_proposal_mode"] in {"metric_softmax", "committor_guided"}
    assert saved["method_recommendation"]["decision"] in {
        "keep_robust_default_and_report_peak_ablation",
        "promote_robust_mode_candidate",
    }
    assert (output_dir / "track4_ranked_conditions.csv").exists()
    assert (output_dir / "track4_mode_robustness.csv").exists()
