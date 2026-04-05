import os
import sys
import types
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
    """Scenario 2: Monolith render fails - first render subprocess non-zero."""
    with (
        patch("subprocess.run") as mock_run,
        patch("run_full_experiment_suite.emit_consumer_contract_bundle") as mock_emit,
    ):
        mock_run.return_value = MagicMock(returncode=1)
        mock_emit.return_value = {"status": "success"}

        result = materialize_baseline_bundle(mock_run_dir, strict=True)

        assert result["status"] == "failed"
        assert result["stage"] == "monolith_render"
        assert result["returncode"] == 1
        assert result["run_dir"] == str(mock_run_dir)
        assert mock_run.call_count == 1


def test_materialize_baseline_bundle_precompute_fail(mock_run_dir):
    """Scenario 3: Observer manifest fails - render succeeds, precompute non-zero."""
    with (
        patch("subprocess.run") as mock_run,
        patch("run_full_experiment_suite.emit_consumer_contract_bundle") as mock_emit,
    ):
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=2),
        ]
        mock_emit.return_value = {"status": "success"}

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


def test_load_monolith_rows_preserves_rich_metadata_columns(tmp_path):
    run_dir = tmp_path / "rich_rows"
    run_dir.mkdir(parents=True, exist_ok=True)
    monolith_csv = run_dir / "MONOLITH_DATA.csv"
    monolith_csv.write_text(
        "\n".join(
            [
                "index,bt_uid,title,source,publication,published_at,timestamp_iso_utc,snippet,density,stress,z_height,zone,verdict",
                "0,u0,Title 0,S0,P0,2023-03-04,2023-03-04T00:00:00Z,snip-0,0.5,0.7,0.2,Bridge,HONEST",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = suite._load_monolith_rows(monolith_csv)

    assert rows[0]["publication"] == "P0"
    assert rows[0]["published_at"] == "2023-03-04"
    assert rows[0]["timestamp_iso_utc"] == "2023-03-04T00:00:00Z"
    assert rows[0]["snippet"] == "snip-0"


def test_emit_baseline_state_preserves_rich_article_metadata(tmp_path):
    run_dir = tmp_path / "baseline_rich"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "observer_manifest.json").write_text(
        json.dumps({"observers": [{"observer_id": 0, "relative_path": "observer_0/MONOLITH.html"}]}, indent=2),
        encoding="utf-8",
    )

    rows = [
        {
            "index": 0,
            "bt_uid": "u0",
            "title": "Title 0",
            "source": "S0",
            "publication": "P0",
            "published_at": "2023-03-04",
            "timestamp_iso_utc": "2023-03-04T00:00:00Z",
            "snippet": "snip-0",
            "density": "0.5",
            "stress": "0.7",
            "zone": "Bridge",
            "verdict": "HONEST",
        }
    ]

    out = suite._emit_baseline_state(run_dir, rows)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["articles"][0]["publication"] == "P0"
    assert payload["articles"][0]["published_at"] == "2023-03-04"
    assert payload["articles"][0]["timestamp_iso_utc"] == "2023-03-04T00:00:00Z"
    assert payload["articles"][0]["snippet"] == "snip-0"


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
        run_dir / "verification_report.json": json.dumps(
            {
                "run_id": "freshness",
                "timestamp": "2026-04-03T00:00:00Z",
                "layers": [{"layer_id": "freshness", "layer_name": "freshness", "status": "VERIFIED", "checks": [], "fail_reasons": []}],
                "global_pass": True,
            },
            indent=2,
        ),
        run_dir / "verification_summary.csv": "layer_id,layer_name,status,crn_locked,ordering_pass,seed_stability,mi,fail_reasons\nfreshness,freshness,VERIFIED,True,True,True,0.5,\n",
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
        run_dir / "verification_report.json",
        run_dir / "verification_summary.csv",
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


def test_bundle_outputs_are_fresh_invalidates_placeholder_verification_report(tmp_path):
    run_dir = tmp_path / "freshness_placeholder_verification"
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundle_outputs(run_dir)

    placeholder_report = {
        "run_id": "placeholder",
        "timestamp": "2026-04-03T00:00:00Z",
        "layers": [
            {
                "layer_id": "placeholder",
                "layer_name": "placeholder",
                "status": "UNVERIFIED",
                "checks": [],
                "fail_reasons": ["verification layer not materialized during bundle emission"],
            }
        ],
        "global_pass": False,
    }
    (run_dir / "verification_report.json").write_text(json.dumps(placeholder_report, indent=2), encoding="utf-8")

    assert suite._bundle_outputs_are_fresh(run_dir) is False


def test_bundle_outputs_are_fresh_invalidates_when_observer_input_changes(tmp_path):
    run_dir = tmp_path / "freshness_observer"
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundle_outputs(run_dir)

    assert suite._bundle_outputs_are_fresh(run_dir) is True

    observer_html = run_dir / "observer_0" / "MONOLITH.html"
    os.utime(observer_html, ns=(3_000_000_000, 3_000_000_000))

    assert suite._bundle_outputs_are_fresh(run_dir) is False


def test_resolve_verification_layer_for_leaf_matches_legacy_leaf_directory(tmp_path):
    exp_root = tmp_path / "experiments_20260403_000000"
    leaf_dir = exp_root / "matern" / "cls" / "real"
    leaf_dir.mkdir(parents=True, exist_ok=True)
    (leaf_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid,density,stress,zone\n0,A,u0,0.5,0.7,Bridge\n", encoding="utf-8")

    layer = {
        "layer_id": "matern/cls",
        "layer_name": "cls",
        "layer_dir": exp_root / "matern" / "cls",
        "artifacts": {
            "real": {"42": {}},
            "control_random": {"42": {}},
        },
    }

    resolved = suite._resolve_verification_layer_for_leaf([layer], leaf_dir, exp_root)

    assert resolved is layer


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
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "index": 0,
            "bt_uid": "u0",
            "title": "A0",
            "publication": "P0",
            "published_at": "2023-03-04",
            "timestamp_iso_utc": "2023-03-04T00:00:00Z",
            "snippet": "snip-0",
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
            "publication": "P1",
            "published_at": "2023-03-05",
            "timestamp_iso_utc": "2023-03-05T00:00:00Z",
            "snippet": "snip-1",
            "zone": "Void",
            "verdict": "PHANTOM",
            "density": "0.80",
            "stress": "0.90",
            "z_height": "0.60",
        },
    ]
    observer_payload = {
        "features": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
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
    observer_payload_0 = dict(observer_payload)
    observer_payload_0["features"] = np.array([[0.5, 0.0], [1.5, 1.0]], dtype=float)
    observer_payload_1 = dict(observer_payload)
    observer_payload_1["features"] = np.array([[0.0, -0.5], [1.0, 0.5]], dtype=float)
    torch.save(observer_payload_0, rel_dir / "observer_0.pt")
    torch.save(observer_payload_1, rel_dir / "observer_1.pt")

    result = suite._emit_relativity_defaults(run_dir, rows)

    assert result["mode"] == "observer_payload_relativity_v1"

    state = json.loads((rel_dir / "state_0.json").read_text(encoding="utf-8"))
    delta = json.loads((rel_dir / "delta_0.json").read_text(encoding="utf-8"))

    assert state["provenance"]["source"] == "observer_payload_relativity_v1"
    assert state["metrics"]["observer_bt_uid"] == "u0"
    assert state["metrics"]["observer_axis_label"] == "bot_0"
    assert state["articles"][0]["publication"] == "P0"
    assert state["articles"][0]["published_at"] == "2023-03-04"
    assert state["articles"][0]["timestamp_iso_utc"] == "2023-03-04T00:00:00Z"
    assert state["articles"][0]["snippet"] == "snip-0"
    assert "baseline_x" in state["articles"][0]
    assert "delta_x" in state["articles"][0]
    assert delta["provenance"]["source"] == "observer_payload_relativity_v1"
    assert delta["null_observer_equivalence"]["max_coord_delta"] > 0.0
    assert delta["null_observer_equivalence"]["path_flip_count"] >= 0
    assert delta["axis_delta"]["rotation_deg"] > 0.0


def test_emit_relativity_defaults_overwrites_stale_cache_when_local_payloads_missing(tmp_path):
    run_dir = tmp_path / "stale_relativity_cache"
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"index": 0, "bt_uid": "u0", "title": "A0", "zone": "Bridge", "verdict": "HONEST", "density": "0.1", "stress": "0.2", "z_height": "0.3"},
    ]
    state_path = rel_dir / "state_0.json"
    delta_path = rel_dir / "delta_0.json"
    state_path.write_text(json.dumps({"observer_id": 0, "articles": [{"coord_delta": 9.9}]}), encoding="utf-8")
    delta_path.write_text(json.dumps({"observer_id": 0, "null_observer_equivalence": {"max_coord_delta": 9.9}}), encoding="utf-8")

    result = suite._emit_relativity_defaults(run_dir, rows)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    delta = json.loads(delta_path.read_text(encoding="utf-8"))

    assert result["mode"] == "suite-default"
    assert state["synthetic_placeholder"] is True
    assert state["message"] in {"observer-conditioned relativity payloads missing", "observer payload unavailable"}
    assert delta["synthetic_placeholder"] is True
    assert delta["null_observer_equivalence"]["max_coord_delta"] == 0.0


def test_emit_relativity_defaults_uses_observer_feature_matrix_for_nmi(tmp_path):
    run_dir = tmp_path / "observer_metric_bundle"
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "index": 0,
            "bt_uid": "u0",
            "title": "A0",
            "publication": "P0",
            "published_at": "2023-03-04",
            "timestamp_iso_utc": "2023-03-04T00:00:00Z",
            "snippet": "snip-0",
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
            "publication": "P1",
            "published_at": "2023-03-05",
            "timestamp_iso_utc": "2023-03-05T00:00:00Z",
            "snippet": "snip-1",
            "zone": "Void",
            "verdict": "PHANTOM",
            "density": "0.80",
            "stress": "0.90",
            "z_height": "0.60",
        },
        {
            "index": 2,
            "bt_uid": "u2",
            "title": "A2",
            "publication": "P2",
            "published_at": "2023-03-06",
            "timestamp_iso_utc": "2023-03-06T00:00:00Z",
            "snippet": "snip-2",
            "zone": "Plateau",
            "verdict": "HONEST",
            "density": "0.40",
            "stress": "0.55",
            "z_height": "0.20",
        },
    ]
    root_payload = {
        "features": np.zeros((3, 2), dtype=float),
        "spectral_probe_magnitudes": np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], dtype=float),
        "walker_work_integrals": np.array([1.0, 2.0, 3.0], dtype=float),
        "article_metadata": [{"index": i, "bt_uid": f"u{i}", "title": f"A{i}"} for i in range(3)],
        "walker_states": [{"index": i, "anomaly_flag": False} for i in range(3)],
        "phantom_verdicts": [{"index": i, "verdict": "HONEST"} for i in range(3)],
        "walker_paths": [{"article_idx": i, "step_diagnostics": []} for i in range(3)],
        "provenance": {"basis_hash": "basis-root"},
    }
    torch.save(root_payload, run_dir / "observer_42.pt")

    observer_feature_means = [10.0, 20.0, 30.0]
    for i, mean_val in enumerate(observer_feature_means):
        observer_payload = dict(root_payload)
        observer_payload["features"] = np.full((3, 2), mean_val, dtype=float)
        torch.save(observer_payload, rel_dir / f"observer_{i}.pt")

    def fake_extract_validation_label_info(*args, **kwargs):
        return {
            "labels": ["a", "b", "a"],
            "label_cardinality": 2,
            "label_source": "test",
        }

    def fake_compute_alignment_metrics(features_np, label_info):
        arr = np.asarray(features_np, dtype=float)
        return {
            "nmi": float(arr.mean()),
            "ari": 0.0,
            "n_clusters": 2,
            "label_cardinality": 2,
            "label_source": "test",
        }

    fake_complete_pipeline = types.ModuleType("core.complete_pipeline")
    fake_complete_pipeline._extract_validation_label_info = fake_extract_validation_label_info
    fake_complete_pipeline._compute_alignment_metrics = fake_compute_alignment_metrics
    with patch.dict(sys.modules, {"core.complete_pipeline": fake_complete_pipeline}):
        result = suite._emit_relativity_defaults(run_dir, rows)

    assert result["mode"] == "observer_payload_relativity_v1"
    state0 = json.loads((rel_dir / "state_0.json").read_text(encoding="utf-8"))
    state1 = json.loads((rel_dir / "state_1.json").read_text(encoding="utf-8"))
    state2 = json.loads((rel_dir / "state_2.json").read_text(encoding="utf-8"))

    assert state0["metrics"]["global_nmi"] == 0.0
    assert state0["metrics"]["observer_conditioned_nmi"] == 10.0
    assert state1["metrics"]["observer_conditioned_nmi"] == 20.0
    assert state2["metrics"]["observer_conditioned_nmi"] == 30.0


def test_collect_track_metrics_restores_t1_from_t1_embeddings():
    label_info = {"labels": ["a", "b"], "label_cardinality": 2, "label_source": "test"}
    result = {
        "features": np.array([[10.0, 10.0], [10.0, 10.0]], dtype=float),
        "checkpoints": {
            "T1_embeddings": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float),
            "T0_substrate": np.zeros((2, 8, 3), dtype=float),
            "T2_kernels": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=float),
        },
    }

    def fake_compute_alignment_metrics(features_np, _label_info):
        arr = np.asarray(features_np, dtype=float)
        return {"nmi": float(arr.mean())}

    metrics = suite._collect_track_metrics_from_result(result, label_info, fake_compute_alignment_metrics)

    assert metrics["SYN"]["nmi"] == pytest.approx(10.0)
    assert metrics["T1"]["nmi"] == pytest.approx(1.0)
    assert metrics["T2"]["nmi"] == pytest.approx(2.0)


def test_normalize_observer_payloads_for_verification_rewrites_provenance(tmp_path):
    run_dir = tmp_path / "verification_leaf"
    run_dir.mkdir()
    payload = {
        "features": np.ones((2, 3), dtype=float),
        "bt_uid_list": ["u0", "u1"],
        "provenance": [],
        "rks_basis_state": {"hash": "basis-abc", "kernel_type": "rbf", "sigma": 1.5, "seed": 42},
        "meta": {"kernel": "rbf", "channel": "cls", "seed": 42, "git_hash": "deadbeef"},
    }
    torch.save(payload, run_dir / "observer_global.pt")

    suite._normalize_observer_payloads_for_verification(run_dir)

    hydrated = torch.load(run_dir / "observer_global.pt", map_location="cpu", weights_only=False)
    provenance = hydrated["provenance"]

    assert isinstance(provenance, dict)
    assert provenance["basis_hash"] == "basis-abc"
    assert provenance["crn_seed"] == 42
    assert provenance["alpha"] == pytest.approx(1.0)
    assert isinstance(provenance["weights_hash"], str)
    assert len(provenance["weights_hash"]) > 0


def test_pick_primary_observer_file_prefers_observer_global(tmp_path):
    run_dir = tmp_path / "primary_observer"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "observer_42.pt").write_text("legacy", encoding="utf-8")
    (run_dir / "observer_global.pt").write_text("global", encoding="utf-8")

    picked = suite._pick_primary_observer_file(run_dir)

    assert picked == run_dir / "observer_global.pt"


def test_emit_relativity_deltas_json_aggregates_vector_fields(tmp_path):
    run_dir = tmp_path / "relativity_delta_bundle"
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "observer_id": 7,
        "articles": [
            {
                "index": 0,
                "bt_uid": "u0",
                "title": "A0",
                "baseline_x": 0.4,
                "baseline_y": 0.5,
                "baseline_z": 0.3,
                "observer_x": 0.5,
                "observer_y": 0.1,
                "observer_z": -0.2,
                "delta_x": 0.1,
                "delta_y": -0.4,
                "delta_z": -0.5,
                "coord_delta": 0.9,
            }
        ],
        "metrics": {
            "observer_bt_uid": "u7",
            "observer_title": "Obs",
            "observer_density": 0.2,
            "observer_stress": 0.3,
            "observer_z_height": 0.1,
            "observer_conditioned_nmi": 0.55,
            "global_nmi": 0.44,
        },
        "provenance": {},
    }
    delta_payload = {
        "observer_id": 7,
        "null_observer_equivalence": {"max_coord_delta": 0.9},
        "path_flip_delta": {"u0|A0": 0.9},
        "metrics_delta": {"d_nmi": 0.11},
        "axis_delta": {"rotation_deg": 12.0},
        "translation_only_comparison": {"d_path_flip_count": 1},
    }
    (rel_dir / "state_7.json").write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
    (rel_dir / "delta_7.json").write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")

    out = suite._emit_relativity_deltas_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert blob["status"] == "OK"
    assert blob["observer_count"] == 1
    vector = blob["observers"][0]["vectors"][0]
    assert vector["baseline"]["x"] == pytest.approx(0.4)
    assert vector["baseline"]["y"] == pytest.approx(0.5)
    assert vector["baseline"]["z"] == pytest.approx(0.3)
    assert vector["delta"]["x"] == pytest.approx(0.10000000000000003)
    assert vector["delta"]["y"] == pytest.approx(-0.4)
    assert vector["delta"]["z"] == pytest.approx(-0.5)


def test_emit_relativity_deltas_json_marks_placeholder_bundle_no_data(tmp_path):
    run_dir = tmp_path / "relativity_delta_placeholder_bundle"
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "observer_id": 0,
        "articles": [],
        "metrics": {},
        "synthetic_placeholder": True,
    }
    delta_payload = {
        "observer_id": 0,
        "synthetic_placeholder": True,
        "null_observer_equivalence": {"max_coord_delta": 0.0},
        "path_flip_delta": {},
        "metrics_delta": {},
        "axis_delta": {},
    }
    (rel_dir / "state_0.json").write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
    (rel_dir / "delta_0.json").write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")

    out = suite._emit_relativity_deltas_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert blob["status"] == "NO_DATA"
    assert blob["synthetic_placeholder"] is True
    assert "observer-conditioned relativity payloads were not materialized" in blob["message"]


def test_emit_consumer_contract_bundle_marks_observer_backed_bundle_non_comparable_when_validation_unavailable(tmp_path):
    if not suite.TORCH_AVAILABLE:
        pytest.skip("torch unavailable")

    run_dir = tmp_path / "observer_backed_non_comparable"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "article_metadata": [
            {"index": 0, "bt_uid": "a0", "title": "A0", "source": "same"},
            {"index": 1, "bt_uid": "a1", "title": "A1", "source": "same"},
        ],
    }
    suite.torch.save(payload, run_dir / "observer_42.pt")

    result = suite.emit_consumer_contract_bundle(run_dir)
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert result["mode"] == "non_comparable"
    assert validation["comparability_status"] == "NON_COMPARABLE"


def test_emit_consumer_contract_bundle_falls_back_to_observer_backed_bundle_without_monolith(tmp_path):
    if not suite.TORCH_AVAILABLE:
        pytest.skip("torch unavailable")

    run_dir = tmp_path / "observer_backed_bundle"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "article_metadata": [
            {"index": 0, "bt_uid": "a0", "title": "A0", "source": "src0", "perspective_tag": "p0", "zone": "z0"},
            {"index": 1, "bt_uid": "a1", "title": "A1", "source": "src1", "perspective_tag": "p1", "zone": "z1"},
        ],
    }
    suite.torch.save(payload, run_dir / "observer_42.pt")

    result = suite.emit_consumer_contract_bundle(run_dir)

    assert result["status"] == "success"
    assert result["mode"] == "observer_backed"
    assert (run_dir / "baseline_meta.json").exists()
    assert (run_dir / "baseline_state.json").exists()
    assert (run_dir / "verification_report.json").exists()
    assert (run_dir / "validation.json").exists()


def test_emit_control_metrics_json_prefers_direct_control_results_blob(tmp_path, monkeypatch):
    run_dir = tmp_path / "control_metrics_direct"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comprehensive_results.json").write_text(json.dumps({"status": "stale"}), encoding="utf-8")

    direct_blob = {
        "direct_source": str(run_dir.parent),
        "interpretation": {
            "metrics": {
                "procrustes": {"ratio": 1.9, "separates": True},
                "distance_corr": {"ratio": 1.3, "separates": True},
                "knn_overlap": {"ratio": 0.7, "separates": True},
            },
            "consensus_residual": {"real": {"consensus_pct": 72.0, "residual_pct": 28.0}},
        },
        "results": {
            "Real": {"procrustes": {"mean": 0.2}},
            "Shuffled": {"procrustes": {"mean": 0.21}},
            "Random": {"procrustes": {"mean": 0.19}},
        },
    }
    monkeypatch.setattr(suite, "_build_direct_control_results_blob", lambda _run_dir: direct_blob)

    out = suite._emit_control_metrics_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert blob["status"] == "OK"
    assert blob["message"] == "loaded from direct control observer payloads"
    assert blob["metrics"]["procrustes_ratio"] == pytest.approx(1.9)
    assert blob["metrics"]["distance_corr_ratio"] == pytest.approx(1.3)
    assert blob["metrics"]["separates_count"] == 3
    assert blob["metrics"]["consensus_pct"] == pytest.approx(72.0)
    assert blob["metrics"]["residual_pct"] == pytest.approx(28.0)
    assert "shuffled and random controls" in blob["explanation"]
    assert "constant" not in blob["explanation"]


def test_build_control_metrics_payload_explanation_matches_available_controls():
    payload = suite._build_control_metrics_payload(
        {
            "interpretation": {"metrics": {}, "consensus_residual": {}},
            "results": {"Real": {}, "Shuffled": {}, "Random": {}},
        },
        None,
    )
    assert "shuffled and random controls" in payload["explanation"]
    assert "constant" not in payload["explanation"]


def test_emit_ablation_summary_json_normalizes_legacy_metrics(tmp_path):
    run_dir = tmp_path / "ablation_bundle"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ablation_results.json").write_text(
        json.dumps(
            {
                "stage_1_nmi": 0.81,
                "stage_2_nmi": 0.52,
                "stage_3_nmi": 0.44,
                "delta_nmi": -0.37,
                "retained_percentage": 68.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out = suite._emit_ablation_summary_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert blob["status"] == "OK"
    assert blob["metrics"]["stage_1_nmi"] == pytest.approx(0.81)
    assert blob["metrics"]["stage_3_nmi"] == pytest.approx(0.44)
    assert blob["metrics"]["retained_pct"] == pytest.approx(68.0)


def test_emit_ablation_summary_json_prefers_lab_diagnostics_over_stale_no_data_summary(tmp_path):
    run_dir = tmp_path / "ablation_prefers_lab"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ablation_summary.json").write_text(
        json.dumps({"status": "NO_DATA", "message": "stale placeholder"}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "lab_diagnostics.json").write_text(
        json.dumps(
            {
                "procrustes": {
                    "mean_distance_before": 2.0,
                    "mean_distance_after": 1.0,
                },
                "structural_invariants": {"mean_survival_rate": 0.6},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out = suite._emit_ablation_summary_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert blob["status"] == "OK"
    assert blob["message"] == "translated from lab_diagnostics.json"
    assert blob["metrics"]["stage_3_nmi"] == pytest.approx(0.6)


def test_emit_ablation_summary_json_translates_lab_diagnostics(tmp_path):
    run_dir = tmp_path / "ablation_lab_bundle"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lab_diagnostics.json").write_text(
        json.dumps(
            {
                "procrustes": {
                    "mean_distance_before": 1.0,
                    "mean_distance_after": 0.25,
                    "consensus_fraction": 0.62,
                    "residual_fraction": 0.38,
                },
                "structural_invariants": {
                    "mean_survival_rate": 0.8,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out = suite._emit_ablation_summary_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert blob["status"] == "OK"
    assert blob["message"] == "translated from lab_diagnostics.json"
    assert blob["metrics"]["stage_1_nmi"] == pytest.approx(0.5)
    assert blob["metrics"]["stage_2_nmi"] == pytest.approx(0.8)
    assert blob["metrics"]["stage_3_nmi"] == pytest.approx(0.8)
    assert blob["metrics"]["delta_nmi"] == pytest.approx(0.3)
    assert blob["metrics"]["retained_pct"] == pytest.approx(80.0)


def test_emit_validation_json_falls_back_to_article_metadata_rows_without_monolith(tmp_path):
    if not suite.TORCH_AVAILABLE:
        pytest.skip("torch unavailable")

    run_dir = tmp_path / "validation_metadata_fallback"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": np.asarray([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.1, 5.2]], dtype=float),
        "article_metadata": [
            {"index": 0, "bt_uid": "a0", "title": "A0", "zone": "left", "source": "s0"},
            {"index": 1, "bt_uid": "a1", "title": "A1", "zone": "left", "source": "s1"},
            {"index": 2, "bt_uid": "a2", "title": "A2", "zone": "right", "source": "s2"},
            {"index": 3, "bt_uid": "a3", "title": "A3", "zone": "right", "source": "s3"},
        ],
    }
    suite.torch.save(payload, run_dir / "observer_42.pt")
    suite._hydrate_run_leaf_from_observer(run_dir)

    out = suite._emit_validation_json(run_dir)
    blob = json.loads(out.read_text(encoding="utf-8"))

    assert isinstance(blob["nmi"], float)
    assert 0.0 <= blob["nmi"] <= 1.0
    assert blob["trust_level"] == "MEASURED"


def test_normalize_validation_payload_promotes_existing_numeric_nmi():
    existing = {
        "status": "failed",
        "trust_level": "UNAVAILABLE",
        "nmi": 0.91,
    }

    normalized = suite._normalize_validation_payload(existing)

    assert normalized is not None
    assert normalized["nmi"] == pytest.approx(0.91)
    assert normalized["status"] == "success"
    assert normalized["trust_level"] == "MEASURED"


def test_repair_validation_payload_promotes_syn_track_nmi():
    existing = {
        "status": "failed",
        "trust_level": "UNAVAILABLE",
        "track_metrics": {
            "SYN": {"nmi": 0.83},
            "T1": {"nmi": 0.81},
        },
    }

    repaired = suite._repair_validation_payload(existing)

    assert repaired is not None
    assert repaired["nmi"] == pytest.approx(0.83)
    assert repaired["status"] == "success"
    assert repaired["trust_level"] == "MEASURED"


def test_emit_no_data_control_results_syncs_control_metrics_placeholder(tmp_path):
    run_dir = tmp_path / "control_placeholder_sync"
    run_dir.mkdir(parents=True, exist_ok=True)

    suite._emit_no_data_control_results(run_dir)

    comprehensive = json.loads((run_dir / "comprehensive_results.json").read_text(encoding="utf-8"))
    control = json.loads((run_dir / "control_metrics.json").read_text(encoding="utf-8"))

    assert comprehensive["status"] == "NO_DATA"
    assert comprehensive["reason"] == "control analysis not run for this leaf"
    assert control["status"] == "NO_DATA"
    assert control["message"] == "control analysis not run for this leaf"


def test_build_ablation_summary_payload_preserves_no_data_message():
    payload = suite._build_ablation_summary_payload({}, None)

    assert payload["status"] == "NO_DATA"
    assert payload["message"] == "ablation flow not executed for this run"
