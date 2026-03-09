import importlib
import json
import shutil
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

