from pathlib import Path
import importlib
import sys
import types
import shutil

import pytest
from analysis.verification import verify_run as verify_mod


def _install_dash_stubs() -> None:
    class _ComponentNamespace:
        def __getattr__(self, name):
            def _factory(*children, **props):
                return {"component": name, "children": children, "props": props}

            return _factory

    class _Dependency:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _DashStub:
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

    dash_module = types.ModuleType("dash")
    dash_module.Dash = _DashStub
    dash_module.Input = _Dependency
    dash_module.Output = _Dependency
    dash_module.State = _Dependency
    dash_module.callback_context = types.SimpleNamespace(triggered=[])
    dash_module.dcc = _ComponentNamespace()
    dash_module.html = _ComponentNamespace()
    sys.modules.setdefault("dash", dash_module)

    dbc_module = types.ModuleType("dash_bootstrap_components")
    dbc_module.themes = types.SimpleNamespace(CYBORG="CYBORG")
    dbc_namespace = _ComponentNamespace()
    dbc_module.Container = dbc_namespace.Container
    dbc_module.Row = dbc_namespace.Row
    dbc_module.Col = dbc_namespace.Col
    dbc_module.Button = dbc_namespace.Button
    dbc_module.Progress = dbc_namespace.Progress
    sys.modules.setdefault("dash_bootstrap_components", dbc_module)


try:
    import analysis.isolated_dash_prototype as dash_mod
except ModuleNotFoundError as exc:
    if exc.name not in {"dash", "dash_bootstrap_components"}:
        raise
    _install_dash_stubs()
    dash_mod = importlib.import_module("analysis.isolated_dash_prototype")


BADGE_TEXT_IDX = 7
BADGE_STYLE_IDX = 8
T1_TEXT_IDX = 9
WATERMARK_TEXT_IDX = 22
WATERMARK_STYLE_IDX = 23


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


def _seed_index(tmp_path: Path) -> dict:
    run_dir = tmp_path / "synthetic" / "rbf_seed42"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_keys": ["rk"],
        "runs": {
            "rk": {
                "run_dir": run_dir,
                "kernel": "rbf",
                "seed": 42,
                "nmi": 0.7,
                "ari": 0.4,
                "variants": ["A.html", "B.html"],
            }
        },
        "article_rows_by_run": {"rk": {}},
        "observers_by_run": {"rk": [{"label": "Global Mean", "value": "global"}]},
        "observers": [{"label": "Global Mean", "value": "global"}],
        "artifact_root_count": 1,
        "artifact_root": str(tmp_path),
    }


def _patch_common(monkeypatch, tmp_path: Path, html_a: str, html_b: str, state: dict, contract_status: str = "OK"):
    a_path = tmp_path / "A.html"
    b_path = tmp_path / "B.html"
    a_path.write_text(html_a, encoding="utf-8")
    b_path.write_text(html_b, encoding="utf-8")

    monkeypatch.setattr(dash_mod, "INDEX", _seed_index(tmp_path))
    monkeypatch.setattr(dash_mod, "resolve_artifact", lambda run_key, variant_name, observer_value: a_path if variant_name == "A.html" else b_path)
    monkeypatch.setattr(dash_mod, "_safe_read_text", lambda p: p.read_text(encoding="utf-8"))
    monkeypatch.setattr(dash_mod, "_transition_wrapper", lambda component, transition_style: component)
    monkeypatch.setattr(dash_mod, "build_terminal_fallback", lambda *args, **kwargs: "fallback")
    monkeypatch.setattr(dash_mod, "_artifact_track_state", lambda p: {k: "ok" for k in dash_mod.TRACK_MARKERS})
    monkeypatch.setattr(dash_mod, "_track_status_component", lambda state: "track-status")
    monkeypatch.setattr(dash_mod, "_track_delta_component", lambda a, b: "track-delta")
    monkeypatch.setattr(dash_mod, "_artifact_coverage", lambda run_key, variant: (1, 1))
    monkeypatch.setattr(dash_mod, "load_ablation_state", lambda run_key: {"status": "ok"})
    monkeypatch.setattr(dash_mod, "load_control_state", lambda run_key: {"status": "ok"})
    monkeypatch.setattr(dash_mod, "_build_relativity_panel", lambda *args, **kwargs: "relativity")
    monkeypatch.setattr(dash_mod, "_build_group_panel", lambda *args, **kwargs: "group")
    monkeypatch.setattr(dash_mod, "_build_empathy_figure", lambda *args, **kwargs: {"fig": "ok"})
    monkeypatch.setattr(
        dash_mod,
        "load_contract_state",
        lambda run_key, observer_value: {
            "status": contract_status,
            "errors": [],
            "baseline_meta": {},
            "hidden_groups": [],
            "paths": {},
            "group_matrix": {},
            "group_summaries": {},
        },
    )
    monkeypatch.setattr(dash_mod, "load_verification_state", lambda run_key, verification_source="auto": state)


def _render(monkeypatch, tmp_path: Path, state: dict, contract_status: str = "OK", compare_enabled: bool = False, html_a: str = "T4 Survival: 95%", html_b: str = "T4 Survival: 90%"):
    _patch_common(monkeypatch, tmp_path, html_a=html_a, html_b=html_b, state=state, contract_status=contract_status)
    return dash_mod._render_dashboard_impl(
        run_key="rk",
        observer_value="global",
        variant_a="A.html",
        variant_b="B.html",
        verification_source="auto",
        compare_enabled_values=["on"] if compare_enabled else [],
        transition_style="none",
        poll_tick=0,
        gallery_tick=0,
        view_mode="global",
        delta_mode="full",
        translation_mode_values=[],
        failure_overlay_values=[],
        label_column=None,
        label_values=[],
    )


def test_claims_enabled_verified_badge(monkeypatch, tmp_path):
    out = _render(
        monkeypatch,
        tmp_path,
        state={"verification_status": "VERIFIED", "global_pass": True, "seed_stability": True, "crn_locked": True, "geometric_friction": 0.1, "n_broken": 0, "n_trapped": 0, "survival_pct": 99.0, "verification_source": "auto", "summary_path": "x", "report_path": "y"},
        contract_status="OK",
    )
    assert out[BADGE_TEXT_IDX] == "[VERIFIED]"
    assert out[BADGE_STYLE_IDX]["color"] == "#00FF41"
    assert "Topologic Integrity" in out[T1_TEXT_IDX]
    assert out[WATERMARK_STYLE_IDX]["display"] == "none"


@pytest.mark.parametrize(
    "state,contract_status,expected_badge,expected_color",
    [
        ({"verification_status": "UNVERIFIED", "global_pass": None}, "OK", "[UNVERIFIED]", "#FF2A00"),
        ({"verification_status": "NON_COMPARABLE", "global_pass": False}, "OK", "[NON-COMPARABLE]", dash_mod.PALETTE["amber"]),
        ({"verification_status": "MISSING_ARTIFACTS", "global_pass": False}, "OK", "[MISSING ARTIFACTS]", dash_mod.PALETTE["amber"]),
        ({"verification_status": "VERIFIED", "global_pass": False}, "OK", "[UNVERIFIED]", "#FF2A00"),
        ({"verification_status": "VERIFIED", "global_pass": True}, "INVALID_SCHEMA", "[INVALID SCHEMA]", dash_mod.PALETTE["amber"]),
    ],
)
def test_claims_disabled_badge_paths(monkeypatch, tmp_path, state, contract_status, expected_badge, expected_color):
    enriched = {
        "seed_stability": False,
        "crn_locked": False,
        "geometric_friction": 0.0,
        "n_broken": 0,
        "n_trapped": 0,
        "survival_pct": 100.0,
        "verification_source": "auto",
        "summary_path": "x",
        "report_path": "y",
    }
    enriched.update(state)
    out = _render(monkeypatch, tmp_path, state=enriched, contract_status=contract_status)
    assert out[BADGE_TEXT_IDX] == expected_badge
    assert out[BADGE_STYLE_IDX]["color"] == expected_color
    assert "Claims disabled" in out[T1_TEXT_IDX]
    assert out[WATERMARK_TEXT_IDX] == "UNVERIFIED / EXPLORATORY"
    assert out[WATERMARK_STYLE_IDX]["display"] == "flex"


def test_type2_dissonance_branch(monkeypatch, tmp_path):
    out = _render(
        monkeypatch,
        tmp_path,
        state={"verification_status": "VERIFIED", "global_pass": True, "seed_stability": True, "crn_locked": True, "geometric_friction": 0.1, "n_broken": 0, "n_trapped": 0, "survival_pct": 99.0, "verification_source": "auto", "summary_path": "x", "report_path": "y"},
        contract_status="OK",
        compare_enabled=True,
        html_a="panel ... T4 Survival: 95%",
        html_b="panel ... T4 Survival: 60%",
    )
    assert out[BADGE_TEXT_IDX] == "[TYPE 2 DISSONANCE]"
    assert out[BADGE_STYLE_IDX]["color"] == dash_mod.PALETTE["amber"]


def test_load_verification_state_missing_files_defaults(monkeypatch):
    monkeypatch.setattr(dash_mod, "_resolve_verification_pair", lambda run_key, verification_source: (None, None))
    monkeypatch.setattr(dash_mod, "_resolve_run_dir", lambda run_key: None)
    monkeypatch.setattr(dash_mod, "_run_contract_paths", lambda run_dir: {})

    state = dash_mod.load_verification_state("rk", "auto")
    assert state["verification_status"] == "UNVERIFIED"
    assert state["global_pass"] is None
    assert state["n_total"] == 0
    assert state["survival_pct"] == 100.0
    assert state["summary_path"] == "NOT FOUND"
    assert state["report_path"] == "NOT FOUND"


def test_load_verification_state_malformed_summary_rows(monkeypatch, tmp_path):
    summary = tmp_path / "verification_summary.csv"
    summary.write_text("n_broken,n_trapped,n_total\nfoo,2,bar\n", encoding="utf-8")
    report = tmp_path / "verification_report.json"
    report.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setattr(dash_mod, "_resolve_verification_pair", lambda run_key, verification_source: (summary, report))
    monkeypatch.setattr(dash_mod, "_resolve_run_dir", lambda run_key: None)
    monkeypatch.setattr(dash_mod, "_run_contract_paths", lambda run_dir: {})

    state = dash_mod.load_verification_state("rk", "auto")
    assert state["verification_status"] == "UNVERIFIED"
    assert state["global_pass"] is None
    assert state["n_broken"] == 0
    assert state["n_trapped"] == 2
    assert state["n_total"] == 1
    assert state["survival_pct"] == 0.0
    assert state["geometric_friction"] == 2.0


def test_verify_run_leaf_resolution_supports_grouped_corpus_layout(tmp_path):
    exp_dir = tmp_path / "outputs" / "experiments" / "runs" / "exp123"
    leaf = exp_dir / "matern" / "cls" / "sythgen" / "high_quality_articles.jsonl"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "MONOLITH_DATA.csv").write_text("index,title\n0,A\n", encoding="utf-8")

    all_layers = [
        {
            "layer_id": "matern/cls",
            "layer_name": "cls",
            "artifacts": {"high_quality_articles.jsonl": {"42": {"provenance": {}}}},
            "layer_dir": exp_dir / "matern" / "cls" / "sythgen",
        }
    ]

    leaves = verify_mod._leaf_output_dirs(exp_dir, all_layers)

    assert leaves == [(leaf, "matern/cls")]
