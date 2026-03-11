import shutil
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import run_full_experiment_suite as suite


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


def test_create_experiment_directory_creates_timestamped_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixed_now = real_datetime(2026, 2, 28, 12, 34, 56)
    monkeypatch.setattr(suite, "datetime", SimpleNamespace(now=lambda: fixed_now))

    exp_dir = suite.create_experiment_directory()

    assert exp_dir == Path("experiments_20260228_123456")
    assert exp_dir.exists()
    assert exp_dir.is_dir()


@pytest.mark.xfail(
    strict=True,
    reason="Known location split: create_experiment_directory currently writes at repo root, not outputs/."
)
def test_create_experiment_directory_canonical_outputs_path_expected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixed_now = real_datetime(2026, 2, 28, 12, 34, 56)
    monkeypatch.setattr(suite, "datetime", SimpleNamespace(now=lambda: fixed_now))

    exp_dir = suite.create_experiment_directory()

    assert exp_dir == Path("outputs") / "experiments_20260228_123456"


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
