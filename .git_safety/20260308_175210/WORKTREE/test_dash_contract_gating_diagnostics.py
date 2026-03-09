from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _install_dash_stubs() -> None:
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


try:
    from analysis import isolated_dash_prototype as dash_mod
except ModuleNotFoundError:
    _install_dash_stubs()
    from analysis import isolated_dash_prototype as dash_mod


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_valid_run_dir(root: Path) -> Path:
    run_dir = root / "run_valid"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "verification_report.json",
        {
            "run_id": "run_valid",
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
        run_dir / "baseline_meta.json",
        {
            "schema_version": "1.0",
            "cache_version": "1.0",
            "dataset_hash": "dhash",
            "code_hash_or_commit": "chash",
            "weights_hash": "whash",
            "kernel_params": {"kernel": "rbf"},
            "rks_dim": 2048,
            "crn_seed": 12345,
            "alpha": 1.0,
            "timestamp_utc": "2026-02-28T00:00:00Z",
            "verification_status": "VERIFIED",
        },
    )

    _write_json(
        run_dir / "baseline_state.json",
        {
            "articles": [],
            "paths": [],
            "axes": {},
            "metrics": {},
        },
    )
    _write_json(
        run_dir / "validation.json",
        {
            "nmi": 0.5,
        },
    )
    return run_dir


def test_fully_valid_contract_claims_enabled(tmp_path, monkeypatch):
    run_dir = _make_valid_run_dir(tmp_path)
    monkeypatch.setattr(dash_mod, "_resolve_run_dir", lambda run_key: run_dir)

    contract = dash_mod.load_contract_state("rk", "global")
    assert contract["status"] == "OK"
    assert contract["missing_required_artifacts"] == []

    gate = dash_mod._compute_gate_presentation(
        contract_status=contract["status"],
        verification_status="VERIFIED",
        global_pass=True,
        type2_dissonance=False,
    )
    assert gate["claims_enabled"] is True
    assert gate["badge_text"] == "[VERIFIED]"
    assert gate["watermark_visible"] is False


def test_required_artifacts_missing_invalid_schema(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dash_mod, "_resolve_run_dir", lambda run_key: run_dir)

    contract = dash_mod.load_contract_state("rk", "global")
    assert contract["status"] == "INVALID_SCHEMA"
    assert contract["missing_required_artifacts"]
    assert "verification_report.json" in contract["missing_required_artifacts"]
    assert contract["schema_errors"] or contract["missing_required_artifacts"]


def test_optional_artifacts_missing_do_not_hard_fail(tmp_path, monkeypatch):
    run_dir = _make_valid_run_dir(tmp_path)
    monkeypatch.setattr(dash_mod, "_resolve_run_dir", lambda run_key: run_dir)

    contract = dash_mod.load_contract_state("rk", "global")
    assert contract["status"] == "OK"
    assert contract["missing_optional_artifacts"]
    assert "labels/hidden_groups.csv" in contract["missing_optional_artifacts"]


def test_mixed_verification_status_badge_watermark_mapping():
    gate_nc = dash_mod._compute_gate_presentation("OK", "NON_COMPARABLE", True, False)
    assert gate_nc["claims_enabled"] is False
    assert gate_nc["badge_text"] == "[NON-COMPARABLE]"
    assert gate_nc["watermark_visible"] is True

    gate_ma = dash_mod._compute_gate_presentation("OK", "MISSING_ARTIFACTS", False, False)
    assert gate_ma["claims_enabled"] is False
    assert gate_ma["badge_text"] == "[MISSING ARTIFACTS]"
    assert gate_ma["watermark_visible"] is True

    gate_unv = dash_mod._compute_gate_presentation("OK", "UNVERIFIED", False, False)
    assert gate_unv["claims_enabled"] is False
    assert gate_unv["badge_text"] == "[UNVERIFIED]"
    assert gate_unv["watermark_visible"] is True

    gate_invalid = dash_mod._compute_gate_presentation("INVALID_SCHEMA", "VERIFIED", True, False)
    assert gate_invalid["claims_enabled"] is False
    assert gate_invalid["badge_text"] == "[INVALID SCHEMA]"


def test_shared_validator_path_is_invoked(tmp_path, monkeypatch):
    run_dir = _make_valid_run_dir(tmp_path)
    called = {"hit": False}

    def _fake_eval(path: Path):
        called["hit"] = True
        return SimpleNamespace(
            run_dir=path,
            verification_status="VERIFIED",
            global_pass=True,
            contract_ok=True,
            missing_required_artifacts=[],
            missing_optional_artifacts=[],
            schema_errors=[],
            warnings=[],
            paths={
                "baseline_meta.json": path / "baseline_meta.json",
                "baseline_state.json": path / "baseline_state.json",
                "verification_report.json": path / "verification_report.json",
                "verification_summary.csv": None,
                "labels/hidden_groups.csv": None,
                "labels/derived/group_summaries.json": None,
                "labels/derived/group_matrix.json": None,
            },
        )

    monkeypatch.setattr(dash_mod, "_resolve_run_dir", lambda run_key: run_dir)
    monkeypatch.setattr(dash_mod, "evaluate_consumer_contract", _fake_eval)

    contract = dash_mod.load_contract_state("rk", "global")
    assert called["hit"] is True
    assert contract["status"] == "OK"
