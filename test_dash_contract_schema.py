import importlib
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest


def _install_dash_stubs() -> None:
    """Install minimal stubs so isolated_dash_prototype imports in lean test envs."""

    class _NodeFactory:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

    class _DummyDash:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.layout = None
            self.server = types.SimpleNamespace(route=lambda *a, **k: (lambda func: func))

        def callback(self, *args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

        def clientside_callback(self, *args, **kwargs):
            return None

    dash_mod = types.ModuleType("dash")
    dash_mod.Dash = _DummyDash
    dash_mod.Input = lambda *args, **kwargs: ("Input", args, kwargs)
    dash_mod.Output = lambda *args, **kwargs: ("Output", args, kwargs)
    dash_mod.State = lambda *args, **kwargs: ("State", args, kwargs)
    dash_mod.callback_context = types.SimpleNamespace(triggered=[])
    dash_mod.dcc = _NodeFactory()
    dash_mod.html = _NodeFactory()
    sys.modules["dash"] = dash_mod

    dbc_mod = types.ModuleType("dash_bootstrap_components")
    dbc_mod.themes = types.SimpleNamespace(CYBORG="CYBORG")
    dbc_mod.__getattr__ = lambda _name: (lambda *a, **k: {"args": a, "kwargs": k})  # type: ignore[attr-defined]
    sys.modules["dash_bootstrap_components"] = dbc_mod

    go_mod = types.ModuleType("plotly.graph_objects")

    class _DummyFigure:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def add_trace(self, *args, **kwargs):
            return None

        def update_layout(self, *args, **kwargs):
            return None

    go_mod.Figure = _DummyFigure
    go_mod.Heatmap = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    go_mod.Scatter3d = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod
    sys.modules["plotly"] = plotly_mod
    sys.modules["plotly.graph_objects"] = go_mod


def _load_dash_module():
    try:
        return importlib.import_module("analysis.isolated_dash_prototype")
    except ModuleNotFoundError:
        _install_dash_stubs()
        return importlib.import_module("analysis.isolated_dash_prototype")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_provenance(mod) -> dict:
    return {
        "schema_version": "1",
        "cache_version": "1",
        "dataset_hash": "abc12345",
        "code_hash_or_commit": "deadbeef",
        "weights_hash": "feedf00d",
        "kernel_params": {"kernel": "rbf"},
        "rks_dim": 2048,
        "crn_seed": 12345,
        "alpha": 1.0,
        "timestamp_utc": "2026-02-26T00:00:00Z",
        "verification_status": "VERIFIED",
    }


@pytest.fixture(scope="module")
def mod():
    return _load_dash_module()


def test_empathy_gap_uniform_matrix_is_explicitly_disabled(mod):
    contract = {
        "group_matrix": {
            "groups": ["A", "B", "C"],
            "cost_matrix": [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            "label_source": "unit-test",
        }
    }

    fig = mod._build_empathy_figure(contract, "group_topic", [])
    meta = dict(getattr(fig.layout, "meta", {}) or {})

    assert meta["panel"] == "empathy_gap"
    assert meta["status"] == "disabled"
    assert meta["label_source"] == "unit-test"
    assert meta["finite_off_diag_count"] == 6
    assert meta["off_diag_span"] == pytest.approx(0.0)
    assert "uniform off-diagonal" in meta["reason"]


def test_empathy_gap_active_matrix_has_provenance_hovertemplate(mod):
    contract = {
        "group_matrix": {
            "groups": ["A", "B", "C"],
            "cost_matrix": [
                [0.0, 1.0, 2.0],
                [1.5, 0.0, 3.0],
                [2.5, 3.5, 0.0],
            ],
            "label_source": "observer-groups",
        }
    }

    fig = mod._build_empathy_figure(contract, "group_topic", ["A", "B", "C"])
    meta = dict(getattr(fig.layout, "meta", {}) or {})

    assert meta["status"] == "active"
    assert meta["label_source"] == "observer-groups"
    assert meta["finite_off_diag_count"] == 6
    assert meta["off_diag_span"] > 0
    assert fig.data
    assert "Directed cost" in str(fig.data[0].hovertemplate)
    assert fig.data[0].customdata[0][0] == "observer-groups"


def test_consumer_contract_cache_reuses_until_artifact_fingerprint_changes(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mod._cached_consumer_contract.cache_clear()
    calls = []

    def _fake_contract(path):
        calls.append(str(path))
        return types.SimpleNamespace(call_count=len(calls))

    monkeypatch.setattr(mod, "evaluate_consumer_contract", _fake_contract)

    first = mod._consumer_contract(run_dir)
    second = mod._consumer_contract(run_dir)
    assert first is second
    assert len(calls) == 1

    (run_dir / "baseline_meta.json").write_text("{}", encoding="utf-8")
    third = mod._consumer_contract(run_dir)
    assert third is not first
    assert len(calls) == 2
    mod._cached_consumer_contract.cache_clear()


def test_collect_run_dirs_uses_targeted_bounded_scan(monkeypatch, mod, tmp_path):
    root = tmp_path / "outputs"
    run_dir = root / "experiments_demo" / "matern" / "cls" / "real"
    run_dir.mkdir(parents=True)
    (run_dir / "MONOLITH_DATA.csv").write_text("x,y,z\n0,0,0\n", encoding="utf-8")
    monkeypatch.setenv("DASH_RUN_DIR_SCAN_LIMIT", "1")

    found = mod._collect_run_dirs(root)

    assert run_dir in found


def test_focused_proof_marker_ignores_incomplete_bundle_dirs(monkeypatch, mod, tmp_path):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact_root = tmp_path / "custom_runs"
    run_dir = artifact_root / "experiments_demo" / "matern" / "cls" / "real"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH_DATA.csv").write_text("x,y,z\n0,0,0\n", encoding="utf-8")

    incomplete_bundle = tmp_path / "outputs" / "thesis_validation" / "focused" / "focused_proof_20260505_incomplete"
    incomplete_bundle.mkdir(parents=True, exist_ok=True)
    (incomplete_bundle / "focused_run_ids.txt").write_text("experiments_demo\n", encoding="utf-8")
    _write_json(
        tmp_path / "outputs" / "thesis_validation" / "focused" / "current_bundle.json",
        {
            "artifact_root": str(artifact_root),
            "evidence_dir": str(incomplete_bundle),
            "preferred_run_id": "experiments_demo",
            "preferred_run_key": "custom_runs/experiments_demo/matern/cls/real",
        },
    )

    assert mod._is_complete_focused_bundle_dir(incomplete_bundle) is False
    assert mod._load_focused_proof_marker() == {}
    roots = mod._discover_artifact_roots()
    assert artifact_root not in roots


def test_focused_proof_marker_prefers_valid_current_bundle(monkeypatch, mod, tmp_path):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact_root = tmp_path / "custom_runs"
    run_dir = artifact_root / "experiments_demo" / "matern" / "cls" / "real"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH_DATA.csv").write_text("x,y,z\n0,0,0\n", encoding="utf-8")

    fallback_root = tmp_path / "outputs" / "experiments" / "runs" / "fallback_demo" / "matern" / "cls" / "real"
    fallback_root.mkdir(parents=True, exist_ok=True)
    (fallback_root / "MONOLITH_DATA.csv").write_text("x,y,z\n1,1,1\n", encoding="utf-8")

    valid_bundle = tmp_path / "outputs" / "thesis_validation" / "focused" / "focused_proof_20260505_complete"
    valid_bundle.mkdir(parents=True, exist_ok=True)
    for name in mod._focused_bundle_required_files():
        target = valid_bundle / name
        if name == "focused_proof_status.json":
            _write_json(
                target,
                {"status": "success", "evidence_acceptance": {"safe_for_focused_defense": True}},
            )
            continue
        if target.suffix == ".json":
            _write_json(target, {"status": "OK", "name": name})
        else:
            target.write_text("ok\n", encoding="utf-8")

    marker_payload = {
        "artifact_root": str(artifact_root),
        "evidence_dir": str(valid_bundle),
        "preferred_run_id": "experiments_demo",
        "preferred_run_key": "custom_runs/experiments_demo/matern/cls/real",
    }
    _write_json(tmp_path / "outputs" / "thesis_validation" / "focused" / "current_bundle.json", marker_payload)

    loaded = mod._load_focused_proof_marker()
    assert loaded.get("artifact_root") == str(artifact_root)
    roots = mod._discover_artifact_roots()
    assert roots
    assert roots[0] == artifact_root


def test_focused_proof_marker_requires_explicit_safe_acceptance(monkeypatch, mod, tmp_path):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact_root = tmp_path / "custom_runs"
    artifact_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = tmp_path / "outputs" / "thesis_validation" / "focused" / "legacy_success"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name in mod._focused_bundle_required_files():
        target = evidence_dir / name
        if name == "focused_proof_status.json":
            _write_json(target, {"status": "success"})
        elif target.suffix == ".json":
            _write_json(target, {"status": "OK", "name": name})
        else:
            target.write_text("ok\n", encoding="utf-8")

    marker = {
        "status": "success",
        "artifact_root": str(artifact_root),
        "evidence_dir": str(evidence_dir),
    }

    assert mod._validate_focused_proof_marker(marker) == {}

    _write_json(
        evidence_dir / "focused_proof_bundle.json",
        {"evidence_acceptance": {"safe_for_focused_defense": True}},
    )
    assert mod._validate_focused_proof_marker(marker).get("artifact_root") == str(artifact_root)


def test_focused_proof_marker_ignores_failed_complete_bundle(monkeypatch, mod, tmp_path):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact_root = tmp_path / "custom_runs"
    run_dir = artifact_root / "experiments_failed" / "matern" / "cls" / "real"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH_DATA.csv").write_text("x,y,z\n0,0,0\n", encoding="utf-8")

    failed_bundle = tmp_path / "outputs" / "thesis_validation" / "focused" / "focused_proof_20260505_failed"
    failed_bundle.mkdir(parents=True, exist_ok=True)
    for name in mod._focused_bundle_required_files():
        target = failed_bundle / name
        if name == "focused_proof_status.json":
            _write_json(target, {"status": "failed"})
        elif target.suffix == ".json":
            _write_json(target, {"status": "OK", "name": name})
        else:
            target.write_text("ok\n", encoding="utf-8")

    _write_json(
        tmp_path / "outputs" / "thesis_validation" / "focused" / "current_bundle.json",
        {
            "artifact_root": str(artifact_root),
            "evidence_dir": str(failed_bundle),
            "preferred_run_id": "experiments_failed",
            "preferred_run_key": "custom_runs/experiments_failed/matern/cls/real",
        },
    )

    assert mod._is_complete_focused_bundle_dir(failed_bundle) is True
    assert mod._load_focused_proof_marker() == {}
    assert artifact_root not in mod._discover_artifact_roots()


def test_run_selection_health_penalizes_incomplete_suite_manifest(monkeypatch, mod, tmp_path):
    suite_root = tmp_path / "outputs" / "experiments" / "runs" / "experiments_active"
    run_dir = suite_root / "rbf" / "cls" / "real"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH_DATA.csv").write_text("x,y,z\n0,0,0\n", encoding="utf-8")
    _write_json(
        suite_root / "experiment_manifest.json",
        {
            "config": {"kernels": ["rbf"], "channels": ["cls"], "corpora": ["real", "control_random"]},
            "experiments": [{"status": "success", "corpus": "real"}],
        },
    )
    monkeypatch.setattr(
        mod,
        "evaluate_consumer_contract",
        lambda _run_dir: types.SimpleNamespace(
            contract_ok=True,
            verification_status="VERIFIED",
            missing_required_artifacts=[],
            schema_errors=[],
        ),
    )

    health = mod._run_selection_health(run_dir)

    assert health["suite_complete"] is False
    assert health["suite_state"]["expected_successes"] == 2
    assert health["suite_state"]["observed_successes"] == 1
    assert health["score"] == 0

    _write_json(
        suite_root / "experiment_manifest.json",
        {
            "config": {"kernels": ["rbf"], "channels": ["cls"], "corpora": ["real", "control_random"]},
            "experiments": [
                {"status": "success", "corpus": "real"},
                {"status": "success", "corpus": "control_random"},
            ],
        },
    )

    complete_health = mod._run_selection_health(run_dir)
    assert complete_health["suite_complete"] is True
    assert complete_health["score"] == 3


def test_preferred_run_ignores_focus_priority_without_safe_marker(monkeypatch, mod, tmp_path):
    focused_dir = tmp_path / "focused_leaf"
    verified_dir = tmp_path / "verified_leaf"
    focused_dir.mkdir()
    verified_dir.mkdir()
    monkeypatch.setattr(mod, "_load_focused_proof_marker", lambda: {})

    index = {
        "run_keys": ["focused", "verified"],
        "runs": {
            "focused": {
                "run_dir": focused_dir,
                "selection_suite_complete": True,
                "selection_focus_priority": 1,
                "selection_score": 1,
                "selection_root_priority": 3,
                "selection_model_priority": 4,
                "selection_corpus_priority": 4,
            },
            "verified": {
                "run_dir": verified_dir,
                "selection_suite_complete": True,
                "selection_focus_priority": 0,
                "selection_score": 3,
                "selection_root_priority": 3,
                "selection_model_priority": 1,
                "selection_corpus_priority": 4,
            },
        },
    }

    assert mod._preferred_run_key(index) == "verified"


def test_validate_provenance_valid_full_schema_no_errors(mod):
    errors = mod._validate_provenance(_valid_provenance(mod))
    assert errors == [], f"Expected no provenance errors, got: {errors}"


def test_validate_provenance_missing_keys_and_invalid_status(mod):
    payload = {"verification_status": "NOT_A_STATUS"}
    errors = mod._validate_provenance(payload)
    assert any("baseline_meta missing keys:" in e for e in errors), f"Missing-key error not found in: {errors}"
    assert any("schema_version" in e for e in errors), f"Expected schema_version in missing-key list: {errors}"
    assert any("verification_status invalid: NOT_A_STATUS" in e for e in errors), (
        f"Invalid-status error missing: {errors}"
    )


def test_validate_baseline_state_valid_schema(mod):
    blob = {"articles": [], "paths": [], "axes": {}, "metrics": {}}
    errors = mod._validate_baseline_state(blob)
    assert errors == [], f"Expected no baseline_state errors, got: {errors}"


def test_validate_baseline_state_wrong_types(mod):
    blob = {"articles": "not-list", "paths": {"bad": "type"}, "axes": {}, "metrics": {}}
    errors = mod._validate_baseline_state(blob)
    assert "baseline_state.articles must be a list" in errors, f"Expected articles type error, got: {errors}"
    assert "baseline_state.paths must be a list" in errors, f"Expected paths type error, got: {errors}"


def test_validate_observer_state_match_success(mod):
    blob = {
        "observer_id": 7,
        "articles": [],
        "paths": [],
        "axes": {},
        "metrics": {},
        "provenance": {},
    }
    errors = mod._validate_observer_state(blob, observer_id=7)
    assert errors == [], f"Expected observer_state match success, got: {errors}"


def test_validate_observer_state_mismatch_nonint_and_missing(mod):
    mismatch = {
        "observer_id": 8,
        "articles": [],
        "paths": [],
        "axes": {},
        "metrics": {},
        "provenance": {},
    }
    errors = mod._validate_observer_state(mismatch, observer_id=7)
    assert any("state observer_id mismatch: expected 7, got 8" in e for e in errors), (
        f"Expected observer_id mismatch error, got: {errors}"
    )

    nonint = {
        "observer_id": "abc",
        "articles": [],
        "paths": [],
        "axes": {},
        "metrics": {},
        "provenance": {},
    }
    errors = mod._validate_observer_state(nonint, observer_id=7)
    assert "state observer_id is not an integer" in errors, f"Expected non-int observer_id error, got: {errors}"

    missing = {"observer_id": 7}
    errors = mod._validate_observer_state(missing, observer_id=7)
    assert any("state missing 'articles'" in e for e in errors), f"Expected missing-field errors, got: {errors}"


def test_validate_observer_delta_match_mismatch_nonint_missing(mod):
    good = {
        "observer_id": 3,
        "null_observer_equivalence": {},
        "path_flip_delta": {},
        "metrics_delta": {},
        "axis_delta": {},
    }
    errors = mod._validate_observer_delta(good, observer_id=3)
    assert errors == [], f"Expected observer_delta success, got: {errors}"

    bad_id = dict(good, observer_id=4)
    errors = mod._validate_observer_delta(bad_id, observer_id=3)
    assert any("delta observer_id mismatch: expected 3, got 4" in e for e in errors), (
        f"Expected delta mismatch error, got: {errors}"
    )

    nonint = dict(good, observer_id="x")
    errors = mod._validate_observer_delta(nonint, observer_id=3)
    assert "delta observer_id is not an integer" in errors, f"Expected non-int delta observer_id error, got: {errors}"

    missing = {"observer_id": 3}
    errors = mod._validate_observer_delta(missing, observer_id=3)
    assert any("delta missing 'path_flip_delta'" in e for e in errors), f"Expected delta missing-field errors, got: {errors}"


def test_read_hidden_groups_missing_file(mod, tmp_path):
    path = tmp_path / "missing_hidden_groups.csv"
    rows, errors = mod._read_hidden_groups(path)
    assert rows == [], f"Expected no rows for missing hidden groups file, got: {rows}"
    assert errors == ["hidden_groups.csv missing"], f"Unexpected missing-file errors: {errors}"


def test_read_hidden_groups_missing_required_columns(mod, tmp_path):
    csv_path = tmp_path / "hidden_groups.csv"
    csv_path.write_text("article_id,bad_col\n1,x\n", encoding="utf-8")
    rows, errors = mod._read_hidden_groups(csv_path)
    assert rows == [], f"Expected no rows when columns are invalid, got: {rows}"
    assert any("missing required columns article_id, group_topic" in e for e in errors), (
        f"Expected required-column error, got: {errors}"
    )


def test_read_hidden_groups_malformed_parse_failure(mod, tmp_path):
    bad_path = tmp_path / "hidden_groups.csv"
    bad_path.mkdir(parents=True, exist_ok=True)  # opening as file should fail
    rows, errors = mod._read_hidden_groups(bad_path)
    assert rows == [], f"Expected no rows for malformed hidden_groups source, got: {rows}"
    assert any("failed to parse hidden_groups.csv:" in e for e in errors), (
        f"Expected parse-failure error, got: {errors}"
    )


def test_validate_group_summaries_missing_nonlist_and_missing_fields(mod, tmp_path):
    data, errors = mod._validate_group_summaries(tmp_path / "group_summaries.json")
    assert data == {}, "Expected empty data when group_summaries.json is missing"
    assert errors == ["group_summaries.json missing"], f"Unexpected missing-file error: {errors}"

    nonlist_path = tmp_path / "group_summaries_nonlist.json"
    _write_json(nonlist_path, {"groups": "bad"})
    _, errors = mod._validate_group_summaries(nonlist_path)
    assert "group_summaries.json must contain list field 'groups'" in errors, (
        f"Expected non-list groups error, got: {errors}"
    )

    bad_groups_path = tmp_path / "group_summaries_bad_groups.json"
    _write_json(bad_groups_path, {"groups": [{"group_name": "A"}, {"n_articles": "x"}]})
    _, errors = mod._validate_group_summaries(bad_groups_path)
    assert any("group_summaries entry missing numeric n_articles" in e for e in errors), (
        f"Expected missing/invalid n_articles error, got: {errors}"
    )
    assert any("group_summaries entry missing group_name" in e for e in errors), (
        f"Expected missing group_name error, got: {errors}"
    )


def test_validate_group_matrix_missing_nonsquare_and_nonnumeric(mod, tmp_path):
    data, errors = mod._validate_group_matrix(tmp_path / "group_matrix.json")
    assert data == {}, "Expected empty data when group_matrix.json is missing"
    assert errors == ["group_matrix.json missing"], f"Unexpected group_matrix missing error: {errors}"

    nonsquare_path = tmp_path / "group_matrix_nonsquare.json"
    _write_json(nonsquare_path, {"groups": ["A", "B"], "cost_matrix": [[1, 2], [3]]})
    _, errors = mod._validate_group_matrix(nonsquare_path)
    assert any("group_matrix row 1 has invalid width" in e for e in errors), (
        f"Expected non-square matrix error, got: {errors}"
    )

    nonnumeric_path = tmp_path / "group_matrix_nonnumeric.json"
    _write_json(nonnumeric_path, {"groups": ["A", "B"], "cost_matrix": [[1, "x"], [2, 3]]})
    _, errors = mod._validate_group_matrix(nonnumeric_path)
    assert any("group_matrix row 0 contains non-numeric value" in e for e in errors), (
        f"Expected non-numeric matrix value error, got: {errors}"
    )


def test_load_contract_state_invalid_schema_missing_keys(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", {"verification_status": "VERIFIED"})
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.5})
    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "g1", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["g1"], "cost_matrix": [[0.0]]})

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "global")
    assert state["status"] == "INVALID_SCHEMA", f"Expected INVALID_SCHEMA, got: {state['status']}"
    assert any("baseline_meta missing keys:" in e for e in state["errors"]), (
        f"Expected missing-keys provenance error in load_contract_state: {state['errors']}"
    )
    assert any("schema_version" in e for e in state["errors"]), (
        f"Expected specific missing key 'schema_version' in errors: {state['errors']}"
    )


def test_load_contract_state_valid_with_observer_artifacts(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_valid"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})

    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {"observer_id": 7, "articles": [], "paths": [], "axes": {}, "metrics": {}, "provenance": {}},
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {"observer_id": 7, "null_observer_equivalence": {}, "path_flip_delta": {}, "metrics_delta": {}, "axis_delta": {}},
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "article:7")
    assert state["status"] == "OK", f"Expected status OK, got: {state['status']} with errors: {state['errors']}"
    assert state["errors"] == [], f"Expected no contract errors for fully valid contract, got: {state['errors']}"


def test_load_contract_state_loads_observer_atlas_without_affecting_base_gate(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_atlas"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {"observer_id": 7, "articles": [], "paths": [], "axes": {}, "metrics": {}, "provenance": {}},
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {"observer_id": 7, "null_observer_equivalence": {}, "path_flip_delta": {}, "metrics_delta": {}, "axis_delta": {}},
    )
    _write_json(
        run_dir / "observer_atlas_bundle.json",
        {
            "schema_version": 1,
            "bundle_type": "observer_atlas_bundle",
            "slices": [
                {"slice_id": "global", "role": "global", "nodes": [{"row_index": 0, "article_idx": 7, "x": 0, "y": 0, "z": 0}]},
                {"slice_id": "observer_7", "role": "observer", "nodes": [{"row_index": 0, "article_idx": 7, "x": 1, "y": 0, "z": 0}]},
            ],
            "routes": [
                {
                    "source_row_index": 0,
                    "target_row_index": 0,
                    "source_article_idx": 7,
                    "target_article_idx": 7,
                    "source_slice": "global",
                    "target_slice": "observer_7",
                    "semantic_first": {"action": 2.0, "points": [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]},
                    "observer_first": {"action": 1.0, "points": [[0, 0, 0], [0.5, 0.5, 0], [1, 0, 0]]},
                    "closed_loop_points": [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 0, 0]],
                    "holonomy_action": 1.0,
                    "excess_holonomy_action": 1.0,
                }
            ],
            "metrics": {"record_count": 1, "mean_holonomy_action": 1.0, "mean_null_holonomy_action": 0.0, "mean_excess_holonomy_action": 1.0},
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "article:7")

    assert state["status"] == "OK"
    assert state["observer_atlas_readiness"]["status"] == "OK"
    assert state["observer_atlas_readiness"]["route_count"] == 1
    assert "observer_atlas_bundle.json" not in state["missing_optional_artifacts"]
    fig = mod._build_atlas_figure(state, "article:7")
    assert getattr(fig.layout, "meta", {})["status"] == "active"


def test_load_contract_state_marks_observer_atlas_missing_separately(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_no_atlas"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [{"layer_id": "rbf/cls", "layer_name": "cls", "status": "VERIFIED", "checks": [], "fail_reasons": []}],
            "global_pass": True,
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "global")

    assert state["status"] == "OK"
    assert state["observer_atlas_readiness"]["status"] == "MISSING"
    assert "observer_atlas_bundle.json" in state["missing_optional_artifacts"]


def test_observer_atlas_readiness_detects_sha256_source_drift(mod, tmp_path):
    source = tmp_path / "source.json"
    source.write_text("before", encoding="utf-8")
    bundle = {
        "bundle_type": "observer_atlas_bundle",
        "slices": [{"slice_id": "global"}, {"slice_id": "observer_7"}],
        "routes": [{"source_slice": "global", "target_slice": "observer_7"}],
        "source_artifacts": {
            "fingerprints": {
                "source.json": {
                    "path": str(source),
                    "size": source.stat().st_size,
                    "mtime_ns": source.stat().st_mtime_ns,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            }
        },
    }

    source.write_text("after!", encoding="utf-8")
    readiness = mod._atlas_readiness(bundle, observer_id=7)

    assert readiness["status"] == "STALE"
    assert any("sha256 changed" in reason for reason in readiness["reasons"])


def test_consumer_contract_rejects_malformed_required_json(mod, tmp_path):
    run_dir = tmp_path / "malformed_required"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75, "trust_level": "MEASURED"})
    (run_dir / "verification_report.json").write_text("{ definitely not json", encoding="utf-8")

    diagnostics = mod.evaluate_consumer_contract(run_dir)

    assert diagnostics.contract_ok is False
    assert any("verification_report.json parse error" in err for err in diagnostics.schema_errors)


def test_consumer_contract_rejects_failed_validation_status(mod, tmp_path):
    run_dir = tmp_path / "failed_validation"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [{"layer_id": "rbf/cls", "layer_name": "cls", "status": "VERIFIED", "checks": [], "fail_reasons": []}],
            "global_pass": True,
        },
    )
    _write_json(run_dir / "validation.json", {"status": "failed", "trust_level": "UNAVAILABLE", "nmi": 0.91})

    diagnostics = mod.evaluate_consumer_contract(run_dir)

    assert diagnostics.contract_ok is False
    assert any("validation status is not claim-valid" in err for err in diagnostics.schema_errors)
    assert any("validation trust_level is not claim-valid" in err for err in diagnostics.schema_errors)


def test_load_contract_state_observer_missing_artifacts_does_not_fallback_to_global(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_missing_observer"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [{"id": 1}], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "article:7")

    assert state["status"] == "NON_COMPARABLE", (
        f"Expected observer mode to be NON_COMPARABLE when observer artifacts are missing, got: {state['status']}"
    )
    assert state["observer_state"] == {}, f"Expected missing observer_state to stay empty, got: {state['observer_state']}"
    assert state["observer_delta"] == {}, f"Expected missing observer_delta to stay empty, got: {state['observer_delta']}"
    assert "relativity_cache/state_7.json" in state["missing_optional_artifacts"], state["missing_optional_artifacts"]
    assert "relativity_cache/delta_7.json" in state["missing_optional_artifacts"], state["missing_optional_artifacts"]
    assert any("observer relativity state_7.json is missing" in e for e in state["errors"]), state["errors"]
    assert any("observer relativity delta_7.json is missing" in e for e in state["errors"]), state["errors"]
    assert state["baseline_state"]["articles"] == [{"id": 1}], "Expected baseline payload to remain distinct from missing observer payload"


def test_load_contract_state_rejects_synthetic_placeholder_baseline_and_relativity(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_placeholder_contract"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "baseline_meta.json",
        {
            "schema_version": "1.0",
            "cache_version": "1.0",
            "dataset_hash": "suite-generated",
            "code_hash_or_commit": "suite-generated",
            "weights_hash": "suite-generated",
            "kernel_params": {"kernel": "unknown"},
            "rks_dim": 2048,
            "crn_seed": 0,
            "alpha": 1.0,
            "timestamp_utc": "2026-03-10T12:31:17Z",
            "verification_status": "UNVERIFIED",
        },
    )
    _write_json(
        run_dir / "baseline_state.json",
        {
            "articles": [{"index": 0, "title": "placeholder"}],
            "paths": ["observer_0/MONOLITH.html"],
            "axes": {"x": "density", "y": "stress"},
            "metrics": {"source": "MONOLITH_DATA.csv", "placeholder": True},
        },
    )
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {
            "observer_id": 7,
            "articles": [],
            "paths": [],
            "axes": {},
            "metrics": {"placeholder": True},
            "provenance": {"dataset_hash": "suite-generated"},
        },
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {
            "observer_id": 7,
            "null_observer_equivalence": {"placeholder": True},
            "path_flip_delta": {},
            "metrics_delta": {},
            "axis_delta": {},
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "article:7")

    assert state["status"] == "INVALID_SCHEMA", (
        "Synthetic placeholder baseline/relativity artifacts should be rejected instead of accepted as real observer data"
    )
    assert any("placeholder" in e.lower() or "suite-generated" in e.lower() for e in state["errors"]), state["errors"]


def test_load_contract_state_marks_synthetic_observer_relativity_non_comparable(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_placeholder_observer_only"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {
            "observer_id": 7,
            "articles": [],
            "paths": [],
            "axes": {},
            "metrics": {},
            "provenance": {"source": "suite-default"},
        },
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {
            "observer_id": 7,
            "null_observer_equivalence": {"max_coord_delta": 0.0, "path_flip_count": 0, "axis_rotation_deg": 0.0},
            "path_flip_delta": {},
            "metrics_delta": {"d_rupture_rate": 0.0, "d_mean_work": 0.0, "d_survival_pct": 0.0},
            "axis_delta": {"rotation_deg": 0.0, "d_explained_variance_axis1": 0.0},
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "article:7")

    assert state["status"] == "NON_COMPARABLE", state
    assert "relativity_cache/state_7.json (synthetic placeholder)" in state["missing_optional_artifacts"], state["missing_optional_artifacts"]
    assert "relativity_cache/delta_7.json (synthetic placeholder)" in state["missing_optional_artifacts"], state["missing_optional_artifacts"]
    assert any("synthetic placeholder data" in e for e in state["errors"]), state["errors"]


def test_load_contract_state_accepts_track_nmi_schema(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_track_nmi"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(
        run_dir / "validation.json",
        {
            "nmi": 0.75,
            "track_nmi": {"T1": 0.31, "T2": 0.47, "T1.5": 0.55, "SYN": 0.62},
            "track_metrics": {
                "T1": {"nmi": 0.31, "ari": 0.12, "n_clusters": 8, "label_source": "corpus_semantic_label"},
                "SYN": {"nmi": 0.62, "ari": 0.28, "n_clusters": 8, "label_source": "corpus_semantic_label"},
            },
        },
    )
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "global")
    assert state["status"] == "OK", f"Expected status OK, got: {state['status']} with errors: {state['errors']}"


def test_load_contract_state_rejects_invalid_track_nmi_schema(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_bad_track_nmi"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(
        run_dir / "validation.json",
        {
            "nmi": 0.75,
            "track_nmi": {"T1": 1.5},
            "track_metrics": {"T2": {"nmi": "bad"}},
        },
    )
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "global")
    assert state["status"] == "INVALID_SCHEMA", f"Expected INVALID_SCHEMA, got: {state['status']}"
    assert any("validation.track_nmi.T1 must be in [0, 1]" in e for e in state["errors"])
    assert any("validation.track_metrics.T2.nmi must be numeric" in e for e in state["errors"])


def test_load_contract_state_rejects_synthetic_placeholder_validation(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_placeholder_validation"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(
        run_dir / "validation.json",
        {
            "source": "suite-default",
            "synthetic_placeholder": True,
            "reason": "validation.json missing during bundle emission",
        },
    )
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "global")
    assert state["status"] == "INVALID_SCHEMA", state
    assert any("validation contains synthetic placeholder data" in e for e in state["errors"]), state["errors"]
    assert any("validation missing 'nmi'" in e for e in state["errors"]), state["errors"]


def test_load_contract_state_invalid_schema_missing_validation_nmi(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_bad_validation"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"ari": 0.2})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "global")
    assert state["status"] == "INVALID_SCHEMA", f"Expected INVALID_SCHEMA, got: {state['status']}"
    assert any("validation missing 'nmi'" in e for e in state["errors"]), (
        f"Expected missing validation nmi error in contract path, got: {state['errors']}"
    )


def test_load_control_state_accepts_control_metrics_json(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_control_metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "control_metrics.json",
        {
            "status": "OK",
            "message": "loaded",
            "metrics": {
                "procrustes_ratio": 1.8,
                "distance_corr_ratio": 1.2,
                "separates_count": 3,
                "consensus_pct": 74.0,
                "residual_pct": 26.0,
            },
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_control_state("rk")

    assert state["status"] == "OK"
    assert state["procrustes_ratio"] == 1.8
    assert state["distance_corr_ratio"] == 1.2
    assert state["separates_count"] == 3
    assert state["consensus_pct"] == 74.0
    assert state["residual_pct"] == 26.0


def test_load_ablation_state_accepts_normalized_ablation_summary(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_ablation_metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "ablation_summary.json",
        {
            "status": "OK",
            "metrics": {
                "stage_1_nmi": 0.91,
                "stage_2_nmi": 0.61,
                "stage_3_nmi": 0.42,
                "delta_nmi": -0.49,
                "retained_pct": 63.0,
            },
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_ablation_state("rk")

    assert state["status"] == "OK"
    assert state["stage_1_nmi"] == 0.91
    assert state["stage_2_nmi"] == 0.61
    assert state["stage_3_nmi"] == 0.42
    assert state["delta_nmi"] == -0.49
    assert state["retained_pct"] == 63.0


def test_load_ablation_state_accepts_lab_diagnostics_json(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_lab_diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "lab_diagnostics.json",
        {
            "procrustes": {
                "mean_distance_before": 1.0,
                "mean_distance_after": 0.25,
            },
            "structural_invariants": {
                "mean_survival_rate": 0.8,
            },
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_ablation_state("rk")

    assert state["status"] == "OK"
    assert state["message"] == "translated from lab_diagnostics.json"
    assert state["stage_1_nmi"] == pytest.approx(0.5)
    assert state["stage_2_nmi"] == pytest.approx(0.8)
    assert state["stage_3_nmi"] == pytest.approx(0.8)
    assert state["retained_pct"] == pytest.approx(80.0)


def test_load_contract_state_accepts_relativity_deltas_bundle(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_relativity_bundle"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    _write_json(run_dir / "control_metrics.json", {"status": "NO_DATA", "metrics": {}, "synthetic_placeholder": True})
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n1,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {"observer_id": 7, "articles": [], "paths": [], "axes": {}, "metrics": {}, "provenance": {}},
    )
    _write_json(
        run_dir / "relativity_deltas.json",
        {
            "status": "OK",
            "observer_count": 1,
            "observers": [
                {
                    "observer_id": 7,
                    "null_observer_equivalence": {"max_coord_delta": 0.3, "path_flip_count": 1, "axis_rotation_deg": 9.0},
                    "path_flip_delta": {"u0|A0": 0.3},
                    "metrics_delta": {"d_nmi": 0.2},
                    "axis_delta": {"rotation_deg": 9.0, "d_explained_variance_axis1": 0.1},
                    "translation_only_comparison": {"d_path_flip_count": 1},
                }
            ],
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    state = mod.load_contract_state("rk", "article:7")

    assert state["status"] == "OK", state
    assert state["observer_delta"]["metrics_delta"]["d_nmi"] == 0.2
    assert state["observer_delta"]["axis_delta"]["rotation_deg"] == 9.0


def test_load_contract_state_loads_observer_recenter_summary(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "contract_recenter_summary"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {}})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {"layer_id": "x", "layer_name": "x", "status": "VERIFIED", "checks": [], "fail_reasons": []}
            ],
            "global_pass": True,
        },
    )
    _write_json(run_dir / "validation.json", {"nmi": 0.5})
    _write_json(
        run_dir / "observer_recenter_summary.json",
        {
            "status": "OK",
            "observer_count": 2,
            "ok_count": 2,
            "path_start_match_observer_count": 2,
            "replay_path_observer_count": 1,
            "z_origin_policy": "xy_origin_preserve_canonical_z",
        },
    )
    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)

    state = mod.load_contract_state("rk", "global")

    assert state["status"] == "OK"
    assert state["observer_recenter_summary"]["status"] == "OK"
    assert state["observer_recenter_summary"]["ok_count"] == 2
    assert "observer_recenter_summary.json" not in state["missing_optional_artifacts"]


def test_load_control_state_legacy_explanation_matches_available_controls(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "legacy_control_explanation"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "comprehensive_results.json",
        {
            "interpretation": {
                "metrics": {
                    "procrustes": {"ratio": 1.2, "separates": True},
                    "distance_corr": {"ratio": 0.9, "separates": False},
                },
                "consensus_residual": {"real": {"consensus_pct": 55.0, "residual_pct": 45.0}},
            },
            "results": {"Real": {}, "Random": {}, "Shuffled": {}},
        },
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _run_key: run_dir)

    payload = mod.load_control_state("legacy")

    assert payload["status"] == "OK"
    assert "shuffled and random controls" in payload["explanation"]
    assert "constant" not in payload["explanation"]


def test_build_artifact_index_discovers_modern_run_layouts(monkeypatch, mod, tmp_path):
    honest_run = tmp_path / "outputs" / "honest_matern"
    honest_run.mkdir(parents=True, exist_ok=True)
    (honest_run / "MONOLITH.html").write_text("<html>honest</html>", encoding="utf-8")
    (honest_run / "MONOLITH_DATA.csv").write_text("index,title,bt_uid\n0,Honest,uid-0\n", encoding="utf-8")

    modern_run = (
        tmp_path
        / "outputs"
        / "experiments"
        / "runs"
        / "experiments_20260309_050500"
        / "matern"
        / "cls"
        / "sythgen"
        / "int"
        / "high_quality_articles.jsonl"
    )
    modern_run.mkdir(parents=True, exist_ok=True)
    (modern_run / "MONOLITH.html").write_text("<html>modern</html>", encoding="utf-8")
    (modern_run / "MONOLITH_DATA.csv").write_text("index,title,bt_uid\n7,Modern,uid-7\n", encoding="utf-8")
    _write_json(modern_run / "validation.json", {"nmi": 0.42, "ari": 0.19})

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    roots = mod._discover_artifact_roots()
    root_labels = {str(p.relative_to(tmp_path)).replace("\\", "/") for p in roots}
    assert "outputs" in root_labels
    assert "outputs/experiments/runs" in root_labels
    assert "outputs/experiments/runs/experiments_20260309_050500/matern/cls/sythgen" not in root_labels

    monkeypatch.setattr(mod, "ARTIFACT_ROOTS", roots)
    monkeypatch.setattr(mod, "PRIMARY_ARTIFACT_ROOT", roots[0] if roots else None)
    index = mod.build_artifact_index()

    modern_key = "outputs/experiments/runs/experiments_20260309_050500/matern/cls/sythgen/int/high_quality_articles.jsonl"
    assert "outputs/honest_matern" in index["run_keys"]
    assert modern_key in index["run_keys"]
    assert index["runs"][modern_key]["kernel"] == "matern"
    assert index["runs"][modern_key]["nmi"] == 0.42
    assert index["runs"][modern_key]["ari"] == 0.19


def test_build_artifact_index_prefers_monolith_run_manifest_for_variants_and_metrics(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "outputs" / "experiments" / "runs" / "exp_x" / "matern" / "cls" / "sythgen" / "high_quality_articles.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.html").write_text("<html>main</html>", encoding="utf-8")
    (run_dir / "_probe_current.html").write_text("<html>probe</html>", encoding="utf-8")
    (run_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid\n0,Main,uid-0\n", encoding="utf-8")
    _write_json(
        run_dir / "MONOLITH.run_manifest.json",
        {
            "schema_version": 1,
            "run_key": "outputs/experiments/runs/exp_x/matern/cls/sythgen/high_quality_articles.jsonl",
            "primary_artifact": "MONOLITH.html",
            "primary_metrics": {"synthesis_nmi": 0.77},
            "artifacts": [
                {"html": "MONOLITH.html", "view_state": "MONOLITH.view_state.json"},
                {"html": "_probe_current.html", "view_state": None},
            ],
        },
    )

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    roots = mod._discover_artifact_roots()
    monkeypatch.setattr(mod, "ARTIFACT_ROOTS", roots)
    monkeypatch.setattr(mod, "PRIMARY_ARTIFACT_ROOT", roots[0] if roots else None)
    index = mod.build_artifact_index()

    run_key = "outputs/experiments/runs/exp_x/matern/cls/sythgen/high_quality_articles.jsonl"
    assert index["runs"][run_key]["variants"] == ["MONOLITH.html", "_probe_current.html"]
    assert index["runs"][run_key]["primary_variant"] == "MONOLITH.html"
    assert index["runs"][run_key]["nmi"] == 0.77


def test_apply_url_state_observer_uid_overrides_query_observer(monkeypatch, mod):
    monkeypatch.setattr(mod, "_observer_value_from_uid", lambda run_key, uid: "article:7" if uid == "uid-7" else None)
    run_options = [{"label": "rk", "value": "rk"}]

    out = mod.apply_url_state(
        "?run_key=rk&observer=article:1&observer_uid=uid-7&view_mode=global&compare=1&embedded=1",
        run_options,
        "rk",
    )

    assert out == ("rk", "article:7", "observer", [])


def test_apply_url_state_preserves_explicit_observer_when_uid_missing(monkeypatch, mod):
    monkeypatch.setattr(mod, "_observer_value_from_uid", lambda run_key, uid: None)
    run_options = [{"label": "rk", "value": "rk"}]

    out = mod.apply_url_state(
        "?run_key=missing&observer=article:3&observer_uid=missing&view_mode=invalid&compare=yes",
        run_options,
        "rk",
    )

    assert out == ("rk", "article:3", "observer", ["on"])


def test_refresh_variants_hydrates_observer_from_uid(monkeypatch, mod):
    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "runs": {"rk": {"variants": ["MONOLITH.html", "ALT.html"]}},
            "observers_by_run": {
                "rk": [
                    {"label": "Global Mean", "value": "global"},
                    {"label": "Article #7", "value": "article:7"},
                ]
            },
            "article_rows_by_run": {"rk": {7: {"bt_uid": "uid-7"}}},
        },
    )

    out = mod.refresh_variants(
        "rk",
        "?observer=article:1&observer_uid=uid-7",
        "MONOLITH.html",
        "ALT.html",
        "global",
    )

    assert out[1] == "MONOLITH.html"
    assert out[3] == "ALT.html"
    assert out[5] == "article:7"


def test_refresh_variants_honors_url_variant_selection_and_prefers_monolith(monkeypatch, mod):
    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "runs": {"rk": {"variants": ["MONOLITH.html", "_probe_current.html", "ALT.html"]}},
            "observers_by_run": {"rk": [{"label": "Global Mean", "value": "global"}]},
            "article_rows_by_run": {"rk": {}},
        },
    )

    out = mod.refresh_variants(
        "rk",
        "?run_key=rk&variant_a=MONOLITH.html&variant_b=MONOLITH.html",
        "_probe_current.html",
        "_probe_current.html",
        "global",
    )

    assert out[1] == "MONOLITH.html"
    assert out[3] == "MONOLITH.html"


def test_resolve_artifact_returns_none_when_article_variant_differs(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run"
    observer_dir = run_dir / "observer_7"
    observer_dir.mkdir(parents=True, exist_ok=True)
    (observer_dir / "MONOLITH.html").write_text("<html>observer</html>", encoding="utf-8")
    (run_dir / "ALT.html").write_text("<html>alt</html>", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "runs": {
                "rk": {
                    "run_dir": run_dir,
                    "variants": ["MONOLITH.html", "ALT.html"],
                    "observer_manifest": {"variant": "MONOLITH.html"},
                    "observer_artifacts": {"article:7": observer_dir / "MONOLITH.html"},
                }
            }
        },
    )

    resolved = mod.resolve_artifact("rk", "ALT.html", "article:7")

    assert resolved is None


def test_resolve_artifact_prefers_manifest_observer_when_variant_matches(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_manifest_match"
    observer_dir = run_dir / "observer_13"
    observer_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.html").write_text("<html>global</html>", encoding="utf-8")
    (observer_dir / "MONOLITH.html").write_text("<html>observer13</html>", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "runs": {
                "rk": {
                    "run_dir": run_dir,
                    "variants": ["MONOLITH.html"],
                    "observer_manifest": {"variant": "MONOLITH.html"},
                    "observer_artifacts": {"article:13": observer_dir / "MONOLITH.html"},
                }
            }
        },
    )

    resolved = mod.resolve_artifact("rk", "MONOLITH.html", "article:13")

    assert resolved == observer_dir / "MONOLITH.html"


def test_dash_hydrates_observer_view_state_recentered_articles_and_paths(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_observer_hydration"
    observer_dir = run_dir / "observer_7"
    observer_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.html").write_text("<html>global</html>", encoding="utf-8")
    (observer_dir / "MONOLITH.html").write_text("<html>observer</html>", encoding="utf-8")
    _write_json(
        observer_dir / "MONOLITH.view_state.json",
        {
            "observer_focus": {"idx": 7, "uid": "uid-7"},
            "metrics": {"synthesis_nmi": 0.61, "walker_mean_action": 4.25},
            "articles": [
                {"idx": 6, "bt_uid": "uid-6", "x": -1.2, "y": 0.4, "z": 0.1},
                {"idx": 7, "bt_uid": "uid-7", "x": 0.0, "y": 0.0, "z": 0.3},
            ],
            "walker_paths": [
                {
                    "article_idx": 7,
                    "path_space": "rendered_synthesis",
                    "n_points": 3,
                    "start_x": 0.0,
                    "start_y": 0.0,
                    "start_z": 0.3,
                    "end_x": 1.0,
                    "end_y": 0.5,
                    "end_z": 0.2,
                    "focused_observer_replay": True,
                }
            ],
        },
    )
    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "runs": {
                "rk": {
                    "run_dir": run_dir,
                    "variants": ["MONOLITH.html"],
                    "observer_manifest": {"variant": "MONOLITH.html"},
                    "observer_artifacts": {"article:7": observer_dir / "MONOLITH.html"},
                }
            }
        },
    )

    artifact = mod.resolve_artifact("rk", "MONOLITH.html", "article:7")
    hydrated = mod._hydrate_artifact_state("rk", {"metrics": {"synthesis_nmi": 0.1}}, artifact)

    assert artifact == observer_dir / "MONOLITH.html"
    assert hydrated["metrics"]["synthesis_nmi"] == 0.61
    focus_article = next(row for row in hydrated["articles"] if row["idx"] == 7)
    assert focus_article["x"] == pytest.approx(0.0)
    assert focus_article["y"] == pytest.approx(0.0)
    focus_path = next(row for row in hydrated["walker_paths"] if row["article_idx"] == 7)
    assert focus_path["path_space"] == "rendered_synthesis"
    assert focus_path["focused_observer_replay"] is True
    assert focus_path["start_x"] == pytest.approx(0.0)
    assert focus_path["start_y"] == pytest.approx(0.0)


def test_render_dashboard_impl_uses_observer_artifact_for_observer_view(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_dashboard_observer"
    observer_dir = run_dir / "observer_7"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)
    observer_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "MONOLITH.html").write_text("<html>global artifact</html>", encoding="utf-8")
    (observer_dir / "MONOLITH.html").write_text("<html>observer artifact</html>", encoding="utf-8")
    (run_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid\n7,Observer Target,uid-7\n", encoding="utf-8")

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {"synthesis_nmi": 0.75}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [
                {
                    "layer_id": "rbf/cls",
                    "layer_name": "cls",
                    "status": "VERIFIED",
                    "checks": [{"name": "crn_locked", "pass": True}],
                    "fail_reasons": [],
                }
            ],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n7,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {"observer_id": 7, "articles": [{"idx": 7, "bt_uid": "uid-7", "density": 0.3, "stress": 0.7}], "paths": [], "axes": {}, "metrics": {"observer_conditioned_nmi": 0.66}, "provenance": {}},
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {"observer_id": 7, "null_observer_equivalence": {"max_coord_delta": 1.2}, "path_flip_delta": {}, "metrics_delta": {"d_mean_work": 0.1}, "axis_delta": {}},
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "run_keys": ["rk"],
            "runs": {
                "rk": {
                    "run_dir": run_dir,
                    "variants": ["MONOLITH.html"],
                    "kernel": "matern",
                    "seed": 42,
                    "nmi": 0.75,
                    "ari": 0.33,
                    "contract_ok": True,
                    "observer_manifest": {"variant": "MONOLITH.html"},
                    "observer_artifacts": {"article:7": observer_dir / "MONOLITH.html"},
                }
            },
            "observers": [{"label": "Global Mean", "value": "global"}, {"label": "Article #7", "value": "article:7"}],
            "observers_by_run": {"rk": [{"label": "Global Mean", "value": "global"}, {"label": "Article #7", "value": "article:7"}]},
            "article_rows_by_run": {"rk": {7: {"bt_uid": "uid-7", "zone": "Bridge", "density": 0.3, "stress": 0.7}}},
            "artifact_root_count": 1,
            "artifact_root": str(run_dir),
        },
    )
    monkeypatch.setattr(mod, "_transition_wrapper", lambda child, _style: child)
    monkeypatch.setattr(mod, "_artifact_iframe", lambda src_doc: {"iframe_srcdoc": src_doc})
    monkeypatch.setattr(
        mod,
        "_artifact_iframe_from_file",
        lambda path, run_key, variant_name, observer_value: {
            "iframe_path": str(path),
            "run_key": run_key,
            "variant": variant_name,
            "observer": observer_value,
        },
    )
    monkeypatch.setattr(mod, "_build_ablation_panel", lambda _ab: "ablation")
    monkeypatch.setattr(mod, "_build_control_panel", lambda _ctrl: "control")
    monkeypatch.setattr(mod, "_build_relativity_panel", lambda *_args, **_kwargs: "relativity")
    monkeypatch.setattr(mod, "_build_group_panel", lambda *_args, **_kwargs: "groups")
    monkeypatch.setattr(mod, "_build_empathy_figure", lambda *_args, **_kwargs: {"figure": "empathy"})
    read_paths = []
    original_safe_read_text = mod._safe_read_text

    def _counting_safe_read_text(path):
        read_paths.append(Path(path))
        return original_safe_read_text(path)

    monkeypatch.setattr(mod, "_safe_read_text", _counting_safe_read_text)

    out = mod._render_dashboard_impl(
        run_key="rk",
        observer_value="article:7",
        variant_a="MONOLITH.html",
        variant_b="MONOLITH.html",
        verification_source="auto",
        compare_enabled_values=[],
        transition_style="fade",
        poll_tick=0,
        gallery_tick=0,
        view_mode="global",
        delta_mode="absolute",
        translation_mode_values=[],
        failure_overlay_values=[],
        label_column=None,
        label_values=[],
        include_physical_path=True,
        preserve_artifact_container=False,
    )

    assert out[0]["iframe_path"] == str(observer_dir / "MONOLITH.html")
    assert out[0]["observer"] == "article:7"
    assert "observer_7" in out[1]
    assert "observer_7" in out[4]
    assert "uid=uid-7" in out[3]
    assert observer_dir / "MONOLITH.html" not in read_paths
    assert run_dir / "MONOLITH.html" not in read_paths


def test_render_dashboard_impl_hydrates_observer_view_state_over_global_conflicts(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_dashboard_observer_conflict"
    observer_dir = run_dir / "observer_7"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)
    observer_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "MONOLITH.html").write_text("<html>global artifact</html>", encoding="utf-8")
    (observer_dir / "MONOLITH.html").write_text("<html>observer artifact</html>", encoding="utf-8")
    (run_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid\n7,Global CSV,global-csv\n", encoding="utf-8")
    _write_json(
        run_dir / "MONOLITH.view_state.json",
        {
            "metrics": {"synthesis_nmi": 0.99, "walker_mean_action": 99.0, "walker_survival_rate": 0.01},
            "articles": [{"idx": 7, "bt_uid": "global-uid", "zone": "Bridge", "density": 0.1, "stress": 0.2, "x": 99.0, "y": 99.0, "z": 9.0}],
            "walker_paths": [{"article_idx": 7, "start_x": 99.0, "start_y": 99.0, "focused_observer_replay": False}],
        },
    )
    _write_json(
        observer_dir / "MONOLITH.view_state.json",
        {
            "observer_focus": {"idx": 7, "uid": "observer-uid"},
            "metrics": {"synthesis_nmi": 0.61, "walker_mean_action": 4.25, "walker_survival_rate": 0.87},
            "articles": [{"idx": 7, "bt_uid": "observer-uid", "zone": "Void", "density": 0.9, "stress": 0.8, "x": 0.0, "y": 0.0, "z": 0.3}],
            "walker_paths": [
                {
                    "article_idx": 7,
                    "path_space": "rendered_synthesis",
                    "n_points": 3,
                    "start_x": 0.0,
                    "start_y": 0.0,
                    "start_z": 0.3,
                    "end_x": 1.0,
                    "end_y": 0.5,
                    "end_z": 0.2,
                    "focused_observer_replay": True,
                }
            ],
        },
    )

    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {"synthesis_nmi": 0.75}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "rk",
            "timestamp": "2026-02-28T00:00:00Z",
            "layers": [{"layer_id": "rbf/cls", "layer_name": "cls", "status": "VERIFIED", "checks": [{"name": "crn_locked", "pass": True}], "fail_reasons": []}],
            "global_pass": True,
        },
    )
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n7,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {
            "observer_id": 7,
            "articles": [{"idx": 7, "bt_uid": "observer-uid", "density": 0.9, "stress": 0.8}],
            "paths": [],
            "axes": {},
            "metrics": {"observer_conditioned_nmi": 0.61},
            "provenance": {},
        },
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {"observer_id": 7, "null_observer_equivalence": {"max_coord_delta": 1.2}, "path_flip_delta": {}, "metrics_delta": {"d_mean_work": 0.1}, "axis_delta": {}},
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "run_keys": ["rk"],
            "runs": {
                "rk": {
                    "run_dir": run_dir,
                    "variants": ["MONOLITH.html"],
                    "kernel": "matern",
                    "seed": 42,
                    "nmi": 0.75,
                    "ari": 0.33,
                    "contract_ok": True,
                    "observer_manifest": {"variant": "MONOLITH.html"},
                    "observer_artifacts": {"article:7": observer_dir / "MONOLITH.html"},
                }
            },
            "observers": [{"label": "Global Mean", "value": "global"}, {"label": "Article #7", "value": "article:7"}],
            "observers_by_run": {"rk": [{"label": "Global Mean", "value": "global"}, {"label": "Article #7", "value": "article:7"}]},
            "article_rows_by_run": {"rk": {}},
            "artifact_root_count": 1,
            "artifact_root": str(run_dir),
        },
    )
    monkeypatch.setattr(mod, "_transition_wrapper", lambda child, _style: child)
    monkeypatch.setattr(mod, "_artifact_iframe_from_file", lambda path, run_key, variant_name, observer_value: {"iframe_path": str(path), "observer": observer_value})
    monkeypatch.setattr(mod, "_build_ablation_panel", lambda _ab: "ablation")
    monkeypatch.setattr(mod, "_build_control_panel", lambda _ctrl: "control")
    monkeypatch.setattr(mod, "_build_relativity_panel", lambda *_args, **_kwargs: "relativity")
    monkeypatch.setattr(mod, "_build_group_panel", lambda *_args, **_kwargs: "groups")
    monkeypatch.setattr(mod, "_build_empathy_figure", lambda *_args, **_kwargs: {"figure": "empathy"})
    monkeypatch.setattr(mod, "load_ablation_state", lambda _rk: {"status": "MISSING", "summary": {}})
    monkeypatch.setattr(mod, "load_control_state", lambda _rk: {"status": "MISSING", "summary": {}, "controls": {}})
    recenter_summary = {
        "status": "OK",
        "observer_count": 1,
        "ok_count": 1,
        "path_start_match_observer_count": 1,
        "replay_path_observer_count": 1,
        "z_origin_policy": "xy_origin_preserve_canonical_z",
    }
    original_load_contract_state = mod.load_contract_state

    def load_contract_state_with_recenter(run_key, observer_value):
        contract = original_load_contract_state(run_key, observer_value)
        contract["observer_recenter_summary"] = recenter_summary
        return contract

    monkeypatch.setattr(mod, "load_contract_state", load_contract_state_with_recenter)

    captured_snapshots = []

    def fake_compute_track_snapshot(_run_key, artifact_state, _contract, observer_value):
        captured_snapshots.append((observer_value, artifact_state))
        return {"T4": {"survival": artifact_state.get("metrics", {}).get("walker_survival_rate")}}

    monkeypatch.setattr(mod, "_compute_track_snapshot", fake_compute_track_snapshot)
    monkeypatch.setattr(mod, "_track_status_component", lambda snapshot: snapshot)
    monkeypatch.setattr(mod, "_track_delta_component", lambda _a, _b: "track-delta")

    out = mod._render_dashboard_impl(
        run_key="rk",
        observer_value="article:7",
        variant_a="MONOLITH.html",
        variant_b="MONOLITH.html",
        verification_source="auto",
        compare_enabled_values=[],
        transition_style="fade",
        poll_tick=0,
        gallery_tick=0,
        view_mode="global",
        delta_mode="absolute",
        translation_mode_values=[],
        failure_overlay_values=[],
        label_column=None,
        label_values=[],
        include_physical_path=False,
        preserve_artifact_container=False,
    )

    assert out[0]["iframe_path"] == str(observer_dir / "MONOLITH.html")
    assert out[0]["observer"] == "article:7"
    assert "NMI=0.61" in out[2]
    assert "T4 work=4.25" in out[2]
    assert "NMI=0.99" not in out[2]
    assert "T4 work=99.0" not in out[2]
    assert "uid=observer-uid" in out[3]
    assert "zone=Void" in out[3]
    assert "uid=global-uid" not in out[3]
    assert "observer_recenter=1/1" in out[6]
    assert "recenter_path_starts=1/1" in out[6]
    assert "observer_replays=1" in out[6]
    assert "recenter_status=ok" in out[15]
    assert "recenter_z=xy_origin_preserve_canonical_z" in out[15]
    observer_snapshot = next(state for observer, state in captured_snapshots if observer == "article:7")
    assert observer_snapshot["walker_paths"][0]["focused_observer_replay"] is True
    assert observer_snapshot["walker_paths"][0]["start_x"] == pytest.approx(0.0)
    global_snapshot = next(state for observer, state in captured_snapshots if observer == "global")
    assert global_snapshot["walker_paths"][0]["focused_observer_replay"] is False


def test_render_dashboard_impl_missing_observer_artifact_does_not_mount_global(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_dashboard_missing_observer"
    (run_dir / "labels" / "derived").mkdir(parents=True, exist_ok=True)
    (run_dir / "relativity_cache").mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.html").write_text("<html>global artifact only</html>", encoding="utf-8")
    (run_dir / "MONOLITH_DATA.csv").write_text("index,title,bt_uid\n7,Observer Target,uid-7\n", encoding="utf-8")
    _write_json(run_dir / "baseline_meta.json", _valid_provenance(mod))
    _write_json(run_dir / "baseline_state.json", {"articles": [], "paths": [], "axes": {}, "metrics": {"synthesis_nmi": 0.75}})
    _write_json(run_dir / "validation.json", {"nmi": 0.75})
    _write_json(run_dir / "verification_report.json", {"run_id": "rk", "timestamp": "2026-02-28T00:00:00Z", "layers": [], "global_pass": True})
    (run_dir / "labels" / "hidden_groups.csv").write_text("article_id,group_topic\n7,topic\n", encoding="utf-8")
    _write_json(run_dir / "labels" / "derived" / "group_summaries.json", {"groups": [{"group_name": "topic", "n_articles": 1}]})
    _write_json(run_dir / "labels" / "derived" / "group_matrix.json", {"groups": ["topic"], "cost_matrix": [[0.0]]})
    _write_json(
        run_dir / "relativity_cache" / "state_7.json",
        {"observer_id": 7, "articles": [{"idx": 7, "bt_uid": "uid-7", "density": 0.3, "stress": 0.7}], "paths": [], "axes": {}, "metrics": {"observer_conditioned_nmi": 0.66}, "provenance": {}},
    )
    _write_json(
        run_dir / "relativity_cache" / "delta_7.json",
        {"observer_id": 7, "null_observer_equivalence": {"max_coord_delta": 1.2}, "path_flip_delta": {}, "metrics_delta": {"d_mean_work": 0.1}, "axis_delta": {}},
    )

    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)
    monkeypatch.setattr(
        mod,
        "INDEX",
        {
            "run_keys": ["rk"],
            "runs": {
                "rk": {
                    "run_dir": run_dir,
                    "variants": ["MONOLITH.html"],
                    "kernel": "matern",
                    "seed": 42,
                    "nmi": 0.75,
                    "ari": 0.33,
                    "contract_ok": True,
                    "observer_manifest": {"variant": "MONOLITH.html"},
                    "observer_artifacts": {},
                }
            },
            "observers": [{"label": "Global Mean", "value": "global"}, {"label": "Article #7", "value": "article:7"}],
            "observers_by_run": {"rk": [{"label": "Global Mean", "value": "global"}, {"label": "Article #7", "value": "article:7"}]},
            "article_rows_by_run": {"rk": {7: {"bt_uid": "uid-7", "zone": "Bridge", "density": 0.3, "stress": 0.7}}},
            "artifact_root_count": 1,
            "artifact_root": str(run_dir),
        },
    )
    iframe_calls = []
    monkeypatch.setattr(mod, "_transition_wrapper", lambda child, _style: child)
    monkeypatch.setattr(mod, "_artifact_iframe_from_file", lambda *args, **kwargs: iframe_calls.append(args) or {"iframe_path": str(args[0])})
    monkeypatch.setattr(mod, "_build_ablation_panel", lambda _ab: "ablation")
    monkeypatch.setattr(mod, "_build_control_panel", lambda _ctrl: "control")
    monkeypatch.setattr(mod, "_build_relativity_panel", lambda *_args, **_kwargs: "relativity")
    monkeypatch.setattr(mod, "_build_group_panel", lambda *_args, **_kwargs: "groups")
    monkeypatch.setattr(mod, "_build_empathy_figure", lambda *_args, **_kwargs: {"figure": "empathy"})
    monkeypatch.setattr(mod, "_build_leaf_artifact_readout_bits", lambda _contract: {"run_score": [], "coverage": [], "provenance": []})
    monkeypatch.setattr(mod, "load_ablation_state", lambda _rk: {"status": "MISSING", "summary": {}})
    monkeypatch.setattr(mod, "load_control_state", lambda _rk: {"status": "MISSING", "summary": {}, "controls": {}})
    monkeypatch.setattr(mod, "_track_status_component", lambda snapshot: snapshot)
    monkeypatch.setattr(mod, "_track_delta_component", lambda _a, _b: "track-delta")

    out = mod._render_dashboard_impl(
        run_key="rk",
        observer_value="article:7",
        variant_a="MONOLITH.html",
        variant_b="MONOLITH.html",
        verification_source="auto",
        compare_enabled_values=[],
        transition_style="fade",
        poll_tick=0,
        gallery_tick=0,
        view_mode="global",
        delta_mode="absolute",
        translation_mode_values=[],
        failure_overlay_values=[],
        label_column=None,
        label_values=[],
        include_physical_path=False,
        preserve_artifact_container=False,
    )

    assert out[1] == "Artifact: NOT FOUND"
    assert iframe_calls == []


def test_compute_track_snapshot_uses_validation_and_artifact_metrics(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_snapshot"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "validation.json",
        {
            "nmi": 0.62,
            "track_metrics": {
                "T1": {"nmi": 0.88, "ari": 0.31},
                "T1.5": {"nmi": 0.77, "ari": 0.22},
                "T2": {"nmi": 0.24, "ari": 0.10},
                "T3": {"nmi": 0.24, "ari": 0.10},
                "SYN": {"nmi": 0.52, "ari": 0.30},
            },
        },
    )
    (run_dir / "walker_paths.npz").write_bytes(b"npz")
    (run_dir / "phantom_verdicts.json").write_text("[]", encoding="utf-8")
    _write_json(
        run_dir / "hott_summary.json",
        {"n_proofs": 10, "equivalence_rate": 0.7, "mean_confidence": 0.8},
    )
    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)

    artifact_state = {
        "metrics": {
            "synthesis_nmi": 0.52,
            "spectral_signal": 0.80,
            "dirichlet_bonds": 76,
            "dirichlet_cracks": 4,
            "walker_mean_action": 61.7,
            "walker_survival_rate": 1.0,
            "honest_count": 48,
            "phantom_count": 16,
            "tautology_count": 16,
            "anomaly_count": 0,
        }
    }
    contract = {
        "observer_state": {
            "metrics": {
                "observer_conditioned_nmi": 0.46,
                "observer_track_nmi": {"SYN": 0.46},
            }
        }
    }

    snapshot = mod._compute_track_snapshot("rk", artifact_state, contract, "article:7")

    assert snapshot["T1"]["status"] == "online"
    assert snapshot["T1"]["nmi"] == 0.88
    assert snapshot["T1.5"]["signal"] == 0.80
    assert snapshot["T3"]["bonds"] == 76
    assert snapshot["T4"]["action"] == 61.7
    assert snapshot["T5"]["phantom"] == 16
    assert snapshot["T6"]["n_proofs"] == 10
    assert snapshot["SYN"]["nmi"] == 0.52


def test_compute_track_snapshot_prefers_observer_local_track_metrics(monkeypatch, mod, tmp_path):
    run_dir = tmp_path / "run_local_snapshot"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "validation.json",
        {
            "track_metrics": {
                "T1.5": {"nmi": 0.12, "ari": 0.01},
                "T2": {"nmi": 0.14, "ari": 0.02},
                "T3": {"nmi": 0.16, "ari": 0.03},
            },
        },
    )
    monkeypatch.setattr(mod, "_resolve_run_dir", lambda _rk: run_dir)

    artifact_state = {
        "metrics": {
            "spectral_signal": 0.80,
            "dirichlet_bonds": 76,
            "dirichlet_cracks": 4,
        }
    }
    contract = {
        "observer_state": {
            "metrics": {
                "observer_track_metrics": {
                    "T1.5": {
                        "status": "online",
                        "source": "observer_local_recompute",
                        "nmi": 0.62,
                        "recomputed": True,
                        "coordinate_frame": "observer_xyz",
                        "input_source": "relativity_cache/obs_7",
                        "global_validation_fallback": False,
                        "signal": 0.33,
                    },
                    "T2": {
                        "status": "online",
                        "source": "observer_local_recompute",
                        "nmi": 0.64,
                        "recomputed": True,
                        "coordinate_frame": "observer_xyz",
                        "global_validation_fallback": False,
                    },
                    "T3": {
                        "status": "online",
                        "source": "observer_local_recompute",
                        "nmi": 0.66,
                        "recomputed": True,
                        "coordinate_frame": "observer_xyz",
                        "global_validation_fallback": False,
                        "bonds": 9,
                        "cracks": 2,
                    },
                }
            }
        }
    }

    snapshot = mod._compute_track_snapshot("rk", artifact_state, contract, "article:7")

    assert snapshot["T1.5"]["source"] == "observer_local_recompute"
    assert snapshot["T1.5"]["nmi"] == 0.62
    assert snapshot["T1.5"]["signal"] == 0.33
    assert snapshot["T1.5"]["recomputed"] is True
    assert snapshot["T2"]["coordinate_frame"] == "observer_xyz"
    assert snapshot["T2"]["global_validation_fallback"] is False
    assert snapshot["T3"]["status"] == "online"
    assert snapshot["T3"]["bonds"] == 9
    assert snapshot["T3"]["cracks"] == 2


def test_compute_track_delta_summary_reports_real_metric_deltas(mod):
    snapshot_a = {
        "T1": {"status": "online", "nmi": 0.88, "ari": 0.31},
        "T1.5": {"status": "online", "nmi": 0.77, "signal": 0.80},
        "T2": {"status": "online", "nmi": 0.24},
        "T3": {"status": "online", "nmi": 0.24, "bonds": 76, "cracks": 4},
        "T4": {"status": "online", "action": 61.7, "survival": 1.0},
        "T5": {"status": "online", "honest": 48, "phantom": 16, "tautology": 16, "anomaly": 0},
        "T6": {"status": "missing"},
        "SYN": {"status": "online", "nmi": 0.52},
    }
    snapshot_b = {
        "T1": {"status": "online", "nmi": 0.88, "ari": 0.31},
        "T1.5": {"status": "online", "nmi": 0.77, "signal": 0.80},
        "T2": {"status": "online", "nmi": 0.24},
        "T3": {"status": "online", "nmi": 0.24, "bonds": 76, "cracks": 4},
        "T4": {"status": "online", "action": 61.9, "survival": 0.95},
        "T5": {"status": "online", "honest": 46, "phantom": 18, "tautology": 16, "anomaly": 0},
        "T6": {"status": "missing"},
        "SYN": {"status": "online", "nmi": 0.46},
    }

    delta = mod._compute_track_delta_summary(snapshot_a, snapshot_b)

    assert delta["SYN"]["delta"] is True
    assert abs(delta["SYN"]["delta_nmi"] + 0.06) < 1e-9
    assert delta["T4"]["delta"] is True
    assert abs(delta["T4"]["delta_action"] - 0.2) < 1e-9
    assert "H d=-2" in delta["T5"]["summary"]


def test_build_leaf_artifact_readout_bits_surfaces_new_leaf_contracts(mod):
    contract = {
        "track5_summary": {
            "status": "OK",
            "track5_assembly_mode": "riemannian_strict",
            "preferred_geometry_source": "integrated_vectors",
            "uses_integrated_geometry": True,
            "safe_for_thesis_claim": True,
        },
        "track4_animation_manifest": {
            "status": "OK",
            "supports_anchor_swarm_reveal": True,
            "supports_audit_micro_animation": True,
            "supports_stepwise_path_animation": True,
            "path_count": 15,
            "anchor_count": 3,
            "closed_loop_rate": 0.7333333333,
        },
        "leaf_artifact_inventory": {
            "artifact_count_present": 14,
            "artifact_count_total": 17,
            "observer_directory_count": 30,
            "relativity_cache_observer_payloads": 30,
        },
        "observer_recenter_summary": {
            "status": "OK",
            "observer_count": 30,
            "ok_count": 30,
            "path_start_match_observer_count": 30,
            "replay_path_observer_count": 4,
            "z_origin_policy": "xy_origin_preserve_canonical_z",
            "local_track_recompute": {
                "ok_count": 29,
                "observer_count": 30,
                "coordinate_frame": "observer_xyz",
                "global_validation_fallback_count": 1,
            },
        },
    }

    bits = mod._build_leaf_artifact_readout_bits(contract)

    assert "T5 mode=riemannian_strict/integrated_vectors" in bits["run_score"]
    assert "T4 anim=swarm+audit+stepwise" in bits["run_score"]
    assert "T5 integrated=yes" in bits["coverage"]
    assert "T4 loops=15/3" in bits["coverage"]
    assert "T4 closed=73.333%" in bits["coverage"]
    assert "leaf=14/17" in bits["coverage"]
    assert "observer_cache=30/30" in bits["coverage"]
    assert "observer_recenter=30/30" in bits["coverage"]
    assert "recenter_path_starts=30/30" in bits["coverage"]
    assert "observer_replays=4" in bits["coverage"]
    assert "observer_local_tracks=29/30" in bits["coverage"]
    assert "t5_safe=yes" in bits["provenance"]
    assert "t5_geom=integrated_vectors" in bits["provenance"]
    assert "t4_stepwise=yes" in bits["provenance"]
    assert "recenter_status=ok" in bits["provenance"]
    assert "recenter_z=xy_origin_preserve_canonical_z" in bits["provenance"]
    assert "observer_track_frame=observer_xyz" in bits["provenance"]
    assert "observer_track_fallback=1" in bits["provenance"]


def test_build_leaf_artifact_readout_bits_prefers_materialized_relativity_cache_coverage(mod):
    contract = {
        "leaf_artifact_inventory": {
            "observer_directory_count": 30,
            "relativity_cache_state_files": 30,
            "relativity_cache_delta_files": 30,
            "relativity_cache_observer_payloads": 3,
        }
    }

    bits = mod._build_leaf_artifact_readout_bits(contract)

    assert "observer_cache=30/30" in bits["coverage"]
    assert "observer_payloads=3/30" in bits["coverage"]


def test_artifact_iframe_uses_stable_key_for_unchanged_content(mod):
    first = mod._artifact_iframe("<html>same</html>")
    second = mod._artifact_iframe("<html>same</html>")

    first_key = getattr(first, "key", None)
    second_key = getattr(second, "key", None)
    if first_key is None and isinstance(first, dict):
        first_key = first["kwargs"]["key"]
        second_key = second["kwargs"]["key"]
    assert first_key == second_key


def test_transition_wrapper_key_tracks_stable_artifact_key(mod):
    iframe = mod._artifact_iframe("<html>same</html>")
    wrapped_a = mod._transition_wrapper(iframe, "fade")
    wrapped_b = mod._transition_wrapper(iframe, "fade")

    key_a = getattr(wrapped_a, "key", None)
    key_b = getattr(wrapped_b, "key", None)
    if key_a is None and isinstance(wrapped_a, dict):
        key_a = wrapped_a["kwargs"]["key"]
        key_b = wrapped_b["kwargs"]["key"]
    assert key_a == key_b
    assert "artifact_" in key_a


def test_focused_marker_rejects_unsafe_defense_bundle(mod, tmp_path, monkeypatch):
    artifact_root = tmp_path / "runs"
    evidence_dir = tmp_path / "focused" / "bundle"
    artifact_root.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    for name in mod._focused_bundle_required_files():
        (evidence_dir / name).write_text("{}", encoding="utf-8")
    (evidence_dir / "focused_proof_status.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )

    marker = {
        "artifact_root": str(artifact_root),
        "evidence_dir": str(evidence_dir),
        "evidence_acceptance": {
            "mechanically_usable": True,
            "safe_for_focused_defense": False,
        },
    }

    assert mod._validate_focused_proof_marker(marker) == {}


def test_focused_marker_exposes_claim_profile_when_safe(mod, tmp_path):
    artifact_root = tmp_path / "runs"
    evidence_dir = tmp_path / "focused" / "bundle"
    artifact_root.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    for name in mod._focused_bundle_required_files():
        (evidence_dir / name).write_text("{}", encoding="utf-8")
    (evidence_dir / "focused_proof_status.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )

    marker = {
        "status": "success",
        "artifact_root": str(artifact_root),
        "evidence_dir": str(evidence_dir),
        "evidence_acceptance": {
            "mechanically_usable": True,
            "safe_for_focused_defense": True,
            "claim_profile": "procrustes_control",
        },
    }

    validated = mod._validate_focused_proof_marker(marker)

    assert validated["focused_claim_profile"] == "procrustes_control"


def test_build_artifact_index_marks_current_focused_run_with_claim_profile(monkeypatch, mod, tmp_path):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact_root = tmp_path / "custom_runs"
    run_dir = artifact_root / "experiments_demo" / "matern" / "cls" / "real"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.html").write_text("<html>focused</html>", encoding="utf-8")
    (run_dir / "MONOLITH_DATA.csv").write_text("index,x,y,z\n0,0,0,0\n", encoding="utf-8")
    evidence_dir = tmp_path / "outputs" / "thesis_validation" / "focused" / "bundle"
    evidence_dir.mkdir(parents=True)
    for name in mod._focused_bundle_required_files():
        target = evidence_dir / name
        if name == "focused_proof_status.json":
            _write_json(target, {"status": "success"})
        elif name == "focused_proof_bundle.json":
            _write_json(target, {"evidence_acceptance": {"safe_for_focused_defense": True}})
        else:
            _write_json(target, {})
    preferred_run_key = "custom_runs/experiments_demo/matern/cls/real"
    _write_json(
        tmp_path / "outputs" / "thesis_validation" / "focused" / "current_bundle.json",
        {
            "status": "success",
            "artifact_root": str(artifact_root),
            "evidence_dir": str(evidence_dir),
            "preferred_run_id": "experiments_demo",
            "preferred_run_key": preferred_run_key,
            "evidence_acceptance": {
                "safe_for_focused_defense": True,
                "claim_profile": "procrustes_control",
            },
        },
    )
    monkeypatch.setattr(mod, "ARTIFACT_ROOTS", [artifact_root])
    monkeypatch.setattr(mod, "PRIMARY_ARTIFACT_ROOT", artifact_root)

    index = mod.build_artifact_index()
    run = index["runs"][preferred_run_key]

    assert index["default_run"] == preferred_run_key
    assert run["focused_claim_profile"] == "procrustes_control"
    assert run["selection_focus_priority"] == 1
    assert run["selection_focus_source"] == "current_focused_bundle"


def test_dashboard_render_source_mounts_url_iframe_on_initial_side_panel_inputs():
    source = Path("analysis/isolated_dash_prototype.py").read_text(encoding="utf-8")

    assert "The artifact iframe is mounted by URL" in source
    assert "preserve_artifact_container = False" in source
    assert "preserve_artifact_container=preserve_artifact_container" in source
    assert "outputs = (no_update,) + outputs[1:]" not in source
    assert "_artifact_iframe_from_file" in source


def test_live_leaf_inventory_corrects_stale_presence(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MONOLITH.html").write_text("<html></html>", encoding="utf-8")
    (run_dir / "observer_recenter_summary.json").write_text(
        json.dumps({"status": "OK", "observer_count": 1, "ok_count": 1}),
        encoding="utf-8",
    )
    (run_dir / "track4_traversal_summary.json").write_text(
        json.dumps({"status": "OK", "path_count": 15, "anchor_count": 3, "closed_loop_rate": 0.7}),
        encoding="utf-8",
    )
    stale = {
        "presence": {"monolith_html": False, "track4_traversal_summary": False, "observer_recenter_summary": False},
        "artifact_count_present": 0,
        "artifact_count_total": 3,
    }

    inventory = mod._live_leaf_artifact_inventory(run_dir, stale)
    manifest = mod._track4_manifest_with_summary_fallback(run_dir, {})

    assert inventory["presence"]["monolith_html"] is True
    assert inventory["presence"]["track4_traversal_summary"] is True
    assert inventory["presence"]["observer_recenter_summary"] is True
    assert inventory["live_recomputed"] is True
    assert manifest["source"] == "track4_traversal_summary.json"
    assert manifest["supports_anchor_swarm_reveal"] is True
