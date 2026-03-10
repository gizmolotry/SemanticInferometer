import os
import shutil
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import run_full_experiment_suite as suite
from run_full_experiment_suite import materialize_baseline_bundle


@pytest.fixture
def tmp_path(request):
    """Workspace-local tmp_path override for restricted Windows temp directories."""
    root = Path.cwd() / ".pytest_local_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / request.node.name
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


@pytest.fixture
def mock_run_dir(tmp_path):
    """Create a skeleton run directory."""
    run_dir = tmp_path / "experiments_test_run"
    run_dir.mkdir()
    (run_dir / "MONOLITH_DATA.csv").write_text("dummy data", encoding="utf-8")
    return run_dir


def test_materialize_baseline_bundle_happy_path(mock_run_dir):
    """Scenario 1: Happy path - both subprocess calls succeed."""
    with (
        patch("subprocess.run") as mock_run,
        patch("run_full_experiment_suite._bundle_outputs_are_fresh") as mock_fresh,
        patch("run_full_experiment_suite.emit_consumer_contract_bundle") as mock_emit,
        patch("run_full_experiment_suite._validate_required_bundle_outputs") as mock_validate,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_fresh.return_value = False
        mock_emit.return_value = {"status": "success"}
        mock_validate.return_value = []

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "success"
        assert "run_dir" in result
        assert "monolith" in result
        assert "observer_manifest" in result
        assert "validation_json" in result
        assert result["run_dir"] == str(mock_run_dir)
        assert str(mock_run_dir) in result["monolith"]
        assert str(mock_run_dir) in result["observer_manifest"]
        assert str(mock_run_dir) in result["validation_json"]
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0].args[0]
        second_cmd = mock_run.call_args_list[1].args[0]
        assert first_cmd[:3] == [suite.sys.executable, "-m", "analysis.MONOLITH_VIZ"]
        assert second_cmd[:3] == [suite.sys.executable, "-m", "analysis.regression.precompute_observer_artifacts"]


def test_materialize_baseline_bundle_viz_fail(mock_run_dir):
    """Scenario 2: Monolith render fails - first subprocess non-zero."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "failed"
        assert result["stage"] == "monolith_render"
        assert result["returncode"] == 1
        assert result["run_dir"] == str(mock_run_dir)
        assert mock_run.call_count == 1


def test_materialize_baseline_bundle_precompute_fail(mock_run_dir):
    """Scenario 3: Observer manifest fails - first success, second non-zero."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=2),
        ]

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "failed"
        assert result["stage"] == "observer_manifest"
        assert result["returncode"] == 2
        assert result["run_dir"] == str(mock_run_dir)
        assert mock_run.call_count == 2


def test_materialize_baseline_bundle_exception(mock_run_dir):
    """Scenario 4: Subprocess exception - subprocess.run raises exception."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = RuntimeError("Subprocess crashed")

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "failed"
        assert result["stage"] == "exception"
        assert "Subprocess crashed" in result["error"]
        assert result["run_dir"] == str(mock_run_dir)

        for key, val in result.items():
            if key != "run_dir" and isinstance(val, str) and os.path.isabs(val):
                assert val.startswith(str(mock_run_dir))


def test_materialize_baseline_bundle_skipped(tmp_path):
    """Bonus: Verify skip when CSV is missing."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    result = materialize_baseline_bundle(run_dir)

    assert result["status"] == "skipped"
    assert "missing observer_*.pt" in result["reason"]


def test_hydrate_run_leaf_from_observer_writes_required_artifacts(tmp_path):
    run_dir = tmp_path / "hydrated_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.json").write_text('{"nmi": 0.5}', encoding="utf-8")

    observer_payload = {
        "features": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "walker_work_integrals": np.array([0.5, 0.7], dtype=float),
        "spectral_u_axis": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        "spectral_probe_magnitudes": np.array([[0.2, 0.8], [0.7, 0.3]], dtype=float),
        "walker_states": [{"status": "success"}, {"status": "trapped"}],
        "phantom_verdicts": [{"verdict": "HONEST"}, {"verdict": "PHANTOM"}],
        "article_metadata": [
            {"index": 0, "bt_uid": "u0", "title": "A0"},
            {"index": 1, "bt_uid": "u1", "title": "A1"},
        ],
    }
    torch.save(observer_payload, run_dir / "observer_42.pt")

    result = suite._hydrate_run_leaf_from_observer(run_dir)

    assert result["status"] == "success"
    for name in [
        "features.npy",
        "walker_work_integrals.npy",
        "spectral_u_axis.npy",
        "spectral_probe_magnitudes.npy",
        "walker_states.json",
        "phantom_verdicts.json",
        "article_metadata.csv",
        "article_metadata.json",
    ]:
        assert (run_dir / name).exists(), name

    metadata_csv = (run_dir / "article_metadata.csv").read_text(encoding="utf-8")
    assert "bt_uid" in metadata_csv
    assert "title" in metadata_csv
    walker_states = json.loads((run_dir / "walker_states.json").read_text(encoding="utf-8"))
    assert walker_states[0]["status"] == "success"


def test_materialize_baseline_bundle_attempts_csv_hydration_when_missing(mock_run_dir, monkeypatch):
    with (
        patch("subprocess.run") as mock_run,
        patch("run_full_experiment_suite._bundle_outputs_are_fresh") as mock_fresh,
        patch("run_full_experiment_suite.emit_consumer_contract_bundle") as mock_emit,
        patch("run_full_experiment_suite._validate_required_bundle_outputs") as mock_validate,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_fresh.return_value = False
        mock_emit.return_value = {"status": "success"}
        mock_validate.return_value = []

        def _fake_ensure(run_dir: Path):
            (run_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid,density,stress,zone\n0,A,u0,0.5,0.5,Bridge\n", encoding="utf-8")
            return {"status": "success"}

        monkeypatch.setattr(suite, "_ensure_monolith_csv_ready", _fake_ensure)

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "success"
        assert mock_run.call_count == 2
