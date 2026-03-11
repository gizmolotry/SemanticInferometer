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
        patch("analysis.verification.contract.evaluate_consumer_contract") as mock_contract,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_fresh.return_value = False
        mock_emit.return_value = {"status": "success"}
        mock_validate.return_value = []
        mock_contract.return_value = MagicMock(contract_ok=True, missing_required_artifacts=[], schema_errors=[])

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
        assert "--mode" in second_cmd
        assert second_cmd[second_cmd.index("--mode") + 1] == "focused"
        assert "--overwrite" in second_cmd


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
        patch("analysis.verification.contract.evaluate_consumer_contract") as mock_contract,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        mock_fresh.return_value = False
        mock_emit.return_value = {"status": "success"}
        mock_validate.return_value = []
        mock_contract.return_value = MagicMock(contract_ok=True, missing_required_artifacts=[], schema_errors=[])

        def _fake_ensure(run_dir: Path):
            (run_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid,density,stress,zone\n0,A,u0,0.5,0.5,Bridge\n", encoding="utf-8")
            return {"status": "success"}

        monkeypatch.setattr(suite, "_ensure_monolith_csv_ready", _fake_ensure)

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "success"
        assert mock_run.call_count == 2


def test_materialize_baseline_bundle_rejects_fresh_but_invalid_consumer_contract(mock_run_dir):
    with (
        patch("run_full_experiment_suite._bundle_outputs_are_fresh") as mock_fresh,
        patch("analysis.verification.contract.evaluate_consumer_contract") as mock_contract,
    ):
        mock_fresh.return_value = True
        mock_contract.return_value = MagicMock(
            contract_ok=False,
            missing_required_artifacts=["verification_report.json"],
            schema_errors=["validation contains synthetic placeholder data"],
        )

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "failed"
        assert result["stage"] == "consumer_contract"
        assert "verification_report.json" in result["error"]
        assert "validation contains synthetic placeholder data" in result["error"]


def _seed_bundle_outputs(run_dir: Path, *, input_ns: int = 1_000_000_000, output_ns: int = 2_000_000_000) -> None:
    """Create a minimal bundle footprint with controlled mtimes."""
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    observer_dir = run_dir / "observer_0"
    observer_dir.mkdir(parents=True, exist_ok=True)

    files_and_payloads = {
        run_dir / "MONOLITH_DATA.csv": "index,title,bt_uid,density,stress,zone\n0,A,u0,0.5,0.7,Bridge\n",
        run_dir / "MONOLITH.html": "<html></html>\n",
        run_dir / "observer_manifest.json": json.dumps(
            {"observers": [{"observer_id": 0, "relative_path": "observer_0/MONOLITH.html"}]},
            indent=2,
        ),
        run_dir / "baseline_meta.json": json.dumps({"schema_version": "1.0"}, indent=2),
        run_dir / "baseline_state.json": json.dumps({"articles": [], "paths": []}, indent=2),
        run_dir / "validation.json": json.dumps({"nmi": 0.5}, indent=2),
        rel_dir / "state_0.json": json.dumps({"observer_id": 0}, indent=2),
        rel_dir / "delta_0.json": json.dumps({"observer_id": 0}, indent=2),
        observer_dir / "MONOLITH.html": "<html>observer</html>\n",
    }
    for path, payload in files_and_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    for path in [
        run_dir / "MONOLITH_DATA.csv",
        observer_dir / "MONOLITH.html",
    ]:
        os.utime(path, ns=(input_ns, input_ns))

    for path in [
        run_dir / "MONOLITH.html",
        run_dir / "observer_manifest.json",
        run_dir / "baseline_meta.json",
        run_dir / "baseline_state.json",
        run_dir / "validation.json",
        rel_dir / "state_0.json",
        rel_dir / "delta_0.json",
    ]:
        os.utime(path, ns=(output_ns, output_ns))


def test_bundle_outputs_are_fresh_invalidates_when_monolith_csv_changes(tmp_path):
    run_dir = tmp_path / "freshness_monolith"
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundle_outputs(run_dir)

    assert suite._bundle_outputs_are_fresh(run_dir) is True

    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    os.utime(monolith_csv, ns=(3_000_000_000, 3_000_000_000))

    assert suite._bundle_outputs_are_fresh(run_dir) is False


def test_bundle_outputs_are_fresh_invalidates_when_verification_artifact_changes(tmp_path):
    run_dir = tmp_path / "freshness_verification"
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundle_outputs(run_dir)

    verification_report = run_dir / "verification_report.json"
    verification_report.write_text(json.dumps({"global_pass": True}, indent=2), encoding="utf-8")
    os.utime(verification_report, ns=(1_500_000_000, 1_500_000_000))

    assert suite._bundle_outputs_are_fresh(run_dir) is True

    os.utime(verification_report, ns=(3_000_000_000, 3_000_000_000))

    assert suite._bundle_outputs_are_fresh(run_dir) is False


def test_bundle_outputs_are_fresh_invalidates_when_observer_input_changes(tmp_path):
    run_dir = tmp_path / "freshness_observer"
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundle_outputs(run_dir)

    assert suite._bundle_outputs_are_fresh(run_dir) is True

    observer_html = run_dir / "observer_0" / "MONOLITH.html"
    os.utime(observer_html, ns=(3_000_000_000, 3_000_000_000))

    assert suite._bundle_outputs_are_fresh(run_dir) is False


def test_emit_baseline_meta_prefers_observer_provenance_when_available(tmp_path):
    run_dir = tmp_path / "real_bundle_markers"
    run_dir.mkdir(parents=True, exist_ok=True)
    observer_payload = {
        "features": np.ones((2, 16), dtype=float),
        "meta": {
            "kernel": "matern",
            "channel": "cls",
            "seed": 42,
            "kernel_params": {"sigma": 1.25},
            "git_hash": "abc123def456",
            "timestamp": "2026-03-11T05:30:00Z",
        },
        "provenance": {
            "basis_hash": "basis-xyz",
            "crn_seed": 42,
            "alpha": 0.75,
            "weights_hash": "weights-xyz",
            "canonical_ids": ["u0", "u1"],
        },
        "bt_uid_list": ["u0", "u1"],
    }
    torch.save(observer_payload, run_dir / "observer_42.pt")

    out = suite._emit_baseline_meta(run_dir)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["dataset_hash"] == "basis-xyz"
    assert payload["code_hash_or_commit"] == "abc123def456"
    assert payload["weights_hash"] == "weights-xyz"
    assert payload["kernel_params"]["kernel"] == "matern"
    assert payload["kernel_params"]["channel"] == "cls"
    assert payload["rks_dim"] == 16
    assert payload["crn_seed"] == 42
    assert payload["alpha"] == 0.75
    assert payload["provenance_source"] == "observer_payload"
    assert payload["verification_status"] == "UNVERIFIED"


def test_emit_consumer_contract_bundle_marks_placeholder_artifacts_synthetic(tmp_path):
    run_dir = tmp_path / "synthetic_bundle_markers"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH_DATA.csv").write_text(
        "index,title,bt_uid,density,stress,zone\n0,A,u0,0.5,0.7,Bridge\n",
        encoding="utf-8",
    )
    (run_dir / "observer_manifest.json").write_text(
        json.dumps(
            {"observers": [{"observer_id": 0, "relative_path": "observer_0/MONOLITH.html"}]},
            indent=2,
        ),
        encoding="utf-8",
    )

    result = suite.emit_consumer_contract_bundle(run_dir)

    assert result["status"] == "success"

    baseline_meta = json.loads((run_dir / "baseline_meta.json").read_text(encoding="utf-8"))
    assert baseline_meta["dataset_hash"] == "suite-generated"
    assert baseline_meta["code_hash_or_commit"] == "suite-generated"
    assert baseline_meta["weights_hash"] == "suite-generated"

    baseline_state = json.loads((run_dir / "baseline_state.json").read_text(encoding="utf-8"))
    assert baseline_state["metrics"]["source"] == "MONOLITH_DATA.csv"
    assert baseline_state["paths"] == ["observer_0/MONOLITH.html"]

    validation_payload = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation_payload["source"] == "suite-default"
    assert validation_payload["synthetic_placeholder"] is True
    assert "nmi" not in validation_payload

    relativity_state = json.loads((run_dir / "relativity_cache" / "state_0.json").read_text(encoding="utf-8"))
    assert relativity_state["provenance"]["source"] == "suite-default"

    relativity_delta = json.loads((run_dir / "relativity_cache" / "delta_0.json").read_text(encoding="utf-8"))
    assert relativity_delta["null_observer_equivalence"]["max_coord_delta"] == 0.0
    assert relativity_delta["metrics_delta"]["d_rupture_rate"] == 0.0


def test_emit_relativity_defaults_prefers_real_observer_payload_metrics(tmp_path):
    run_dir = tmp_path / "real_relativity_bundle"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "index": 0,
            "bt_uid": "u0",
            "title": "A0",
            "zone": "Bridge",
            "verdict": "HONEST",
            "density": "0.20",
            "stress": "0.30",
            "z_height": "0.10",
        },
        {
            "index": 1,
            "bt_uid": "u1",
            "title": "A1",
            "zone": "Void",
            "verdict": "PHANTOM",
            "density": "0.80",
            "stress": "0.90",
            "z_height": "0.60",
        },
    ]
    observer_payload = {
        "spectral_probe_magnitudes": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        "walker_work_integrals": np.array([1.5, 5.0], dtype=float),
        "article_metadata": [
            {"index": 0, "bt_uid": "u0", "title": "A0"},
            {"index": 1, "bt_uid": "u1", "title": "A1"},
        ],
        "walker_states": [
            {"index": 0, "anomaly_flag": False},
            {"index": 1, "anomaly_flag": True},
        ],
        "phantom_verdicts": [
            {"index": 0, "verdict": "HONEST"},
            {"index": 1, "verdict": "PHANTOM"},
        ],
        "walker_paths": [
            {
                "article_idx": 0,
                "step_diagnostics": [
                    {"step_work": 1.0, "event_active": False, "step_axis_vector": [0.9, 0.1], "dominant_axis_label": "bot_0"},
                    {"step_work": 2.0, "event_active": True, "step_axis_vector": [0.8, 0.2], "dominant_axis_label": "bot_0"},
                ],
            },
            {
                "article_idx": 1,
                "step_diagnostics": [
                    {"step_work": 4.0, "event_active": True, "step_axis_vector": [0.2, 0.8], "dominant_axis_label": "bot_1"},
                    {"step_work": 6.0, "event_active": True, "step_axis_vector": [0.1, 0.9], "dominant_axis_label": "bot_1"},
                ],
            },
        ],
        "provenance": {"basis_hash": "basis-xyz"},
    }
    torch.save(observer_payload, run_dir / "observer_42.pt")

    result = suite._emit_relativity_defaults(run_dir, rows)

    assert result["mode"] == "observer_payload_relativity_v1"

    state = json.loads((run_dir / "relativity_cache" / "state_0.json").read_text(encoding="utf-8"))
    delta = json.loads((run_dir / "relativity_cache" / "delta_0.json").read_text(encoding="utf-8"))

    assert state["provenance"]["source"] == "observer_payload_relativity_v1"
    assert state["metrics"]["observer_bt_uid"] == "u0"
    assert state["metrics"]["observer_axis_label"] == "bot_0"
    assert delta["provenance"]["source"] == "observer_payload_relativity_v1"
    assert delta["null_observer_equivalence"]["max_coord_delta"] > 0.0
    assert delta["null_observer_equivalence"]["path_flip_count"] >= 0
    assert delta["axis_delta"]["rotation_deg"] > 0.0
