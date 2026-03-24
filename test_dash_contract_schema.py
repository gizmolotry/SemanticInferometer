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
    assert "outputs/experiments/runs/experiments_20260309_050500/matern/cls/sythgen" in root_labels

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


def test_resolve_artifact_does_not_force_manifest_observer_when_variant_differs(monkeypatch, mod, tmp_path):
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

    assert resolved == run_dir / "ALT.html"


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
