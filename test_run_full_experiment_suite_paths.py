import json
import sys
from datetime import datetime as real_datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import run_full_experiment_suite as suite


def test_create_experiment_directory_creates_timestamped_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixed_now = real_datetime(2026, 2, 28, 12, 34, 56)
    monkeypatch.setattr(suite, "datetime", SimpleNamespace(now=lambda: fixed_now))

    exp_dir = suite.create_experiment_directory()

    assert exp_dir == Path("outputs") / "experiments" / "runs" / "experiments_20260228_123456"
    assert exp_dir.exists()
    assert exp_dir.is_dir()


def test_create_experiment_directory_canonical_outputs_path_expected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixed_now = real_datetime(2026, 2, 28, 12, 34, 56)
    monkeypatch.setattr(suite, "datetime", SimpleNamespace(now=lambda: fixed_now))

    exp_dir = suite.create_experiment_directory()

    assert exp_dir == Path("outputs") / "experiments" / "runs" / "experiments_20260228_123456"


def test_run_post_thesis_sync_invokes_registry_builder_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "build_results_registry.py").write_text(
        "print('registry')\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(suite.subprocess, "run", fake_run)

    suite.run_post_thesis_sync(run_validation=False)

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == [suite.sys.executable, str(Path("scripts/build_results_registry.py"))]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_run_post_thesis_sync_with_validation_invokes_validator_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "build_results_registry.py").write_text(
        "print('registry')\n",
        encoding="utf-8",
    )
    (scripts_dir / "validate_thesis_artifacts.py").write_text(
        "print('validate')\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(suite.subprocess, "run", fake_run)

    suite.run_post_thesis_sync(run_validation=True)

    assert len(calls) == 2

    build_cmd, build_kwargs = calls[0]
    assert build_cmd == [suite.sys.executable, str(Path("scripts/build_results_registry.py"))]
    assert build_kwargs["capture_output"] is True
    assert build_kwargs["text"] is True
    assert build_kwargs["encoding"] == "utf-8"
    assert build_kwargs["errors"] == "replace"

    validate_cmd, validate_kwargs = calls[1]
    assert validate_cmd == [
        suite.sys.executable,
        str(Path("scripts/validate_thesis_artifacts.py")),
        "--no-registry-sync",
    ]
    assert validate_kwargs["capture_output"] is True
    assert validate_kwargs["text"] is True
    assert validate_kwargs["encoding"] == "utf-8"
    assert validate_kwargs["errors"] == "replace"


def test_run_comparison_writes_family_results_into_kernel_channel_dir(tmp_path, monkeypatch):
    exp_dir = tmp_path / "outputs" / "experiments" / "runs" / "exp"
    data_dir = exp_dir / "matern" / "cls"
    (data_dir / "real").mkdir(parents=True, exist_ok=True)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return MagicMock(returncode=0, stdout="comparison complete", stderr="")

    monkeypatch.setattr(suite.subprocess, "run", fake_run)

    ok = suite.run_comparison(exp_dir, seeds=[42, 420], kernels=["matern"], channels=["cls"])

    assert ok is True
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[:2] == [suite.sys.executable, "compare_controls.py"]
    assert "--data-dir" in cmd
    assert cmd[cmd.index("--data-dir") + 1] == str(data_dir)
    assert "--output-dir" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == str(data_dir)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_post_comparison_refresh_disables_observer_backfill(monkeypatch, tmp_path):
    leaf = tmp_path / "matern" / "cls" / "control_constant"
    leaf.mkdir(parents=True, exist_ok=True)
    calls = []

    monkeypatch.setattr(
        suite,
        "_iter_suite_leaf_dirs",
        lambda exp_dir, config: [leaf],
    )
    monkeypatch.setattr(
        suite,
        "_emit_control_metrics_json",
        lambda output_dir, metric_basis=None: output_dir / "control_metrics.json",
    )

    def fake_emit(run_dir, *, allow_observer_backfill=True):
        calls.append((Path(run_dir), allow_observer_backfill))
        return {"status": "success"}

    monkeypatch.setattr(suite, "emit_consumer_contract_bundle", fake_emit)

    result = suite._refresh_leaf_contracts_after_comparison(
        tmp_path,
        kernels=["matern"],
        channels=["cls"],
        corpora=["control_constant"],
    )

    assert result["status"] == "success"
    assert calls == [(leaf, False)]


def test_post_comparison_refresh_forwards_control_metric_basis(monkeypatch, tmp_path):
    leaf = tmp_path / "matern" / "cls" / "real"
    leaf.mkdir(parents=True, exist_ok=True)
    metric_calls = []

    monkeypatch.setattr(
        suite,
        "_iter_suite_leaf_dirs",
        lambda exp_dir, config: [leaf],
    )

    def fake_emit_control(output_dir, metric_basis=None):
        metric_calls.append((Path(output_dir), metric_basis))
        return Path(output_dir) / "control_metrics.json"

    monkeypatch.setattr(suite, "_emit_control_metrics_json", fake_emit_control)
    monkeypatch.setattr(
        suite,
        "emit_consumer_contract_bundle",
        lambda run_dir, *, allow_observer_backfill=True: {"status": "success"},
    )

    result = suite._refresh_leaf_contracts_after_comparison(
        tmp_path,
        kernels=["matern"],
        channels=["cls"],
        corpora=["real"],
        control_metric_basis="comprehensive",
    )

    assert result["status"] == "success"
    assert metric_calls == [(leaf, "comprehensive_results")]


def test_resume_restores_control_metric_basis_from_suite_config(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    suite._write_suite_config(
        exp_dir,
        {
            "limit": 12,
            "mode": "enhanced",
            "control_metric_basis": "comprehensive_results",
        },
    )
    args = SimpleNamespace(
        limit=1,
        mode="legacy",
        control_metric_basis="auto",
        seeds=[1],
        kernels=["rbf"],
        channels=["cls"],
        corpora=["real"],
        no_variance_tracking=False,
    )

    restored = suite._restore_args_from_resume(args, exp_dir)

    assert restored.limit == 12
    assert restored.mode == "enhanced"
    assert restored.control_metric_basis == "comprehensive_results"


def test_runtime_config_inference_preserves_track5_mode_from_payload_meta():
    cfg = suite._infer_runtime_config_from_payload(
        {
            "meta": {
                "track5_assembly_mode": "riemannian_strict",
                "kernel": "matern",
                "channel": "cls",
            },
            "cls_per_bot": [[[0.0] * 32]],
            "features": [[0.0] * 128],
        }
    )

    kwargs = cfg.to_initialize_kwargs()

    assert kwargs["track5_assembly_mode"] == "strict_riemannian"


def test_load_control_observers_prefers_integrated_vectors(tmp_path):
    if not getattr(suite, "TORCH_AVAILABLE", False):
        pytest.skip("torch not available")

    corpus_dir = tmp_path / "matern" / "cls" / "real"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    suite.torch.save(
        {
            "features": suite.torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            "integrated_vectors": suite.torch.tensor([[9.0, 9.0], [9.0, 9.0]]),
        },
        corpus_dir / "observer_42.pt",
    )

    observers = suite._load_control_observers(corpus_dir)

    assert 42 in observers
    assert np.allclose(observers[42]["features"], np.asarray([[9.0, 9.0], [9.0, 9.0]]))


def test_emit_validation_json_prefers_integrated_vectors(monkeypatch, tmp_path):
    run_dir = tmp_path / "matern" / "cls" / "real"
    run_dir.mkdir(parents=True, exist_ok=True)
    observer_payload = {
        "features": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "integrated_vectors": np.array([[9.0, 0.0], [0.0, 9.0]], dtype=np.float64),
        "article_metadata": [{"index": 0, "zone": "bridge"}, {"index": 1, "zone": "void"}],
    }
    captured = {}

    fake_complete_pipeline = ModuleType("core.complete_pipeline")

    def fake_compute_alignment_metrics(features, label_info):
        captured["features"] = np.asarray(features)
        captured["label_info"] = label_info
        return {
            "nmi": 0.75,
            "ari": 0.5,
            "metric_source": "captured_test_metric",
            "label_source": "test_labels",
            "label_cardinality": 2,
            "n_clusters": 2,
        }

    def fake_extract_validation_label_info(**kwargs):
        return {"labels": [0, 1], "kwargs": kwargs}

    fake_complete_pipeline._compute_alignment_metrics = fake_compute_alignment_metrics
    fake_complete_pipeline._extract_validation_label_info = fake_extract_validation_label_info

    monkeypatch.setitem(sys.modules, "core.complete_pipeline", fake_complete_pipeline)
    monkeypatch.setattr(suite, "_collect_track_metrics_from_result", lambda *args, **kwargs: {})
    monkeypatch.setattr(suite, "_load_primary_observer_payload", lambda path: observer_payload)
    monkeypatch.setattr(
        suite,
        "_load_monolith_rows",
        lambda path: [{"index": 0, "zone": "bridge"}, {"index": 1, "zone": "void"}],
    )

    validation_path = suite._emit_validation_json(run_dir)
    payload = json.loads(validation_path.read_text(encoding="utf-8"))

    assert np.array_equal(captured["features"], observer_payload["integrated_vectors"])
    assert payload["metric_source"] == "captured_test_metric"
    assert payload["nmi"] == pytest.approx(0.75)


def test_generate_waterfall_dashboards_uses_batch_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs" / "experiments" / "runs" / "exp" / "matern" / "cls" / "sythgen" / "high_quality_articles.jsonl"
    checkpoint_dir = run_dir / "checkpoints" / "batch"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    calls = []

    def fake_run_waterfall_analysis(checkpoint_dir, output_dir=None, ground_truth=None, projection_method="pca"):
        calls.append(
            {
                "checkpoint_dir": Path(checkpoint_dir),
                "output_dir": Path(output_dir),
                "ground_truth": ground_truth,
                "projection_method": projection_method,
            }
        )
        return {
            "status": "success",
            "dashboard_path": str(Path(output_dir) / "waterfall_dashboard.html"),
            "report_path": str(Path(output_dir) / "waterfall_report.txt"),
            "metrics_path": str(Path(output_dir) / "waterfall_metrics.json"),
        }

    monkeypatch.setitem(
        sys.modules,
        "analysis.waterfall_viz",
        SimpleNamespace(run_waterfall_analysis=fake_run_waterfall_analysis),
    )

    result = suite.generate_waterfall_dashboards(run_dir, ground_truth={0: "a0"})

    assert result["status"] == "success"
    assert len(calls) == 1
    assert calls[0]["checkpoint_dir"] == checkpoint_dir
    assert calls[0]["output_dir"] == run_dir / "waterfall_analysis"
    assert calls[0]["ground_truth"] == {0: "a0"}
    assert calls[0]["projection_method"] == "pca"
