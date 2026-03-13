from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.MONOLITH_VIZ import (
    PROBE_LABELS,
    ExperimentData,
    _compress_surface_height_field,
    _recompute_focused_observer_track4,
    create_monolith_cockpit,
    load_experiment_data,
    render_data_points_3d,
    render_analysis_planes,
    render_phantom_paths_3d,
    render_terrain_surface,
)
from core.complete_pipeline import _serialize_rks_basis_state
from core.dirichlet_fusion import SharedRKSBasis


def _require_plotly() -> None:
    pytest.importorskip("plotly")


@pytest.fixture
def tmp_path(request):
    root = Path.cwd() / ".pytest_local_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / request.node.name
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _make_experiment(tmp_path: Path, spectral_probe_magnitudes: np.ndarray | None) -> ExperimentData:
    n = 4
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    monolith_csv = exp_dir / "MONOLITH_DATA.csv"
    with monolith_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bt_uid", "title", "density", "stress", "z_height", "zone", "verdict"],
        )
        writer.writeheader()
        writer.writerow(
            {"bt_uid": "a0", "title": "a0", "density": 0.8, "stress": 0.1, "z_height": 0.2, "zone": "Bridge", "verdict": "HONEST"}
        )
        writer.writerow(
            {"bt_uid": "a1", "title": "a1", "density": 0.7, "stress": 0.8, "z_height": 0.6, "zone": "Swamp", "verdict": "PHANTOM"}
        )
        writer.writerow(
            {"bt_uid": "a2", "title": "a2", "density": 0.2, "stress": 0.2, "z_height": -0.3, "zone": "Tightrope", "verdict": "TAUTOLOGY"}
        )
        writer.writerow(
            {"bt_uid": "a3", "title": "a3", "density": 0.1, "stress": 0.9, "z_height": -0.7, "zone": "Void", "verdict": "RUPTURE"}
        )

    features = np.array(
        [
            [0.1, 1.1, 0.2, 0.0],
            [1.0, 0.2, -0.1, 0.4],
            [-0.8, -0.2, 0.3, 1.0],
            [0.5, -1.2, -0.7, 0.2],
        ],
        dtype=float,
    )
    walker_paths = {
        0: np.array([[0.2, 0.3, 0.1], [0.5, 0.6, 0.2]], dtype=float),
        1: np.array([[0.6, 0.3, 0.4], [0.7, 0.5, 0.6]], dtype=float),
        2: np.array([[0.1, -0.4, -0.2], [0.2, -0.5, -0.3]], dtype=float),
        3: np.array([[-0.4, -0.1, -0.5], [-0.6, -0.2, -0.8]], dtype=float),
    }

    return ExperimentData(
        kernel="rbf",
        seed=42,
        n_articles=n,
        features=features,
        spectral_evr=np.array([0.7, 0.6, 0.5, 0.8], dtype=float),
        spectral_probe_magnitudes=spectral_probe_magnitudes,
        walker_states=["success", "trapped", "success", "broken"],
        walker_work_integrals=np.array([0.2, 0.5, 0.4, 0.8], dtype=float),
        walker_paths=walker_paths,
        phantom_verdicts=[
            {"verdict": "HONEST", "walker_state": "success", "w_actual": 0.3, "delta": 0.2},
            {"verdict": "PHANTOM", "walker_state": "trapped", "w_actual": 0.9, "delta": 1.4},
            {"verdict": "TAUTOLOGY", "walker_state": "trapped", "w_actual": 0.5, "delta": 0.4},
            {"verdict": "RUPTURE", "walker_state": "broken", "w_actual": 1.7, "delta": 2.3},
        ],
        article_metadata=[{"bt_uid": f"a{i}", "title": f"title-{i}"} for i in range(n)],
        experiment_dir=exp_dir,
    )


def _render_html_for_mode(tmp_path: Path, mode: str, spectral_probe_magnitudes: np.ndarray | None, monkeypatch) -> tuple[object, str]:
    _require_plotly()
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=spectral_probe_magnitudes)
    output_path = tmp_path / f"monolith_{mode}.html"
    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    fig = create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode=mode,
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=False,
        show_hott=False,
    )
    return fig, output_path.read_text(encoding="utf-8")


def test_rupture_inputs_collapse_to_phantom_render_contract(monkeypatch):
    _require_plotly()
    monkeypatch.setenv("MONOLITH_SHOW_SHEAR_LABELS", "1")
    positions_3d = np.array([[5.0, 6.0, 7.0], [9.0, 8.0, 7.5]], dtype=float)
    walker_paths = {
        0: np.array([[4.8, 5.9, 6.9], [4.6, 5.8, 6.7]], dtype=float),
        1: np.array([[8.8, 7.8, 7.2], [8.7, 7.7, 7.1]], dtype=float),
    }
    phantom_verdicts = [
        {"verdict": "RUPTURE", "walker_state": "broken"},
        {"verdict": "HONEST", "walker_state": "success"},
    ]

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
    )

    names = [str(getattr(t, "name", "")) for t in traces]
    assert "Phantom Path" in names, names
    assert "Ideological Shear" in names, names


def test_missing_logits_keeps_t1_nmi_unavailable_via_function():
    _require_plotly()
    n = 6
    exp = ExperimentData(
        kernel="rbf",
        seed=0,
        n_articles=n,
        features=np.random.default_rng(0).normal(size=(n, 4)),
        logits=None,
        cp_t0_logits=None,
        ground_truth_labels=np.array([0, 1, 0, 1, 0, 1], dtype=int),
    )

    _, nmi_scores = render_analysis_planes(exp)

    assert "T1" in nmi_scores
    assert np.isnan(nmi_scores["T1"])


def test_load_experiment_data_recovers_spectral_evr_from_checkpoint(monkeypatch, tmp_path):
    exp_dir = tmp_path / "matern" / "seed_42"
    checkpoint_dir = exp_dir / "checkpoints" / "batch"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    np.save(exp_dir / "features.npy", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float))
    singular_values = np.array([[3.0, 1.0], [4.0, 2.0]], dtype=float)
    probe_magnitudes = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    dipole_valid = np.array([True, False], dtype=bool)
    np.savez(
        checkpoint_dir / "T1.5_spectral_state.npz",
        singular_values=singular_values,
        probe_magnitudes=probe_magnitudes,
        dipole_valid=dipole_valid,
    )

    class _ContractStub:
        def __init__(self, *_args, **_kwargs):
            pass

        def verify(self):
            return None

    monkeypatch.setattr("analysis.MONOLITH_VIZ.ArtifactContract", _ContractStub)

    exp = load_experiment_data(exp_dir)
    expected_evr = (singular_values[:, 0] ** 2) / np.clip((singular_values ** 2).sum(axis=1), 1e-12, None)

    assert exp.spectral_evr is not None
    assert exp.spectral_probe_magnitudes is not None
    assert exp.spectral_dipole_valid is not None
    assert np.allclose(exp.spectral_evr, expected_evr)
    assert np.array_equal(exp.spectral_probe_magnitudes, probe_magnitudes)
    assert np.array_equal(exp.spectral_dipole_valid, dipole_valid)


def test_persisted_track_nmi_overrides_proxy_recompute():
    _require_plotly()
    n = 6
    exp = ExperimentData(
        kernel="rbf",
        seed=0,
        n_articles=n,
        features=np.random.default_rng(1).normal(size=(n, 4)),
        cp_t2_kernels=np.random.default_rng(2).normal(size=(n, 4)),
        cp_t15_spectral=np.random.default_rng(3).normal(size=(n, 4)),
        dirichlet_fused=np.random.default_rng(4).normal(size=(n, 4)),
        ground_truth_labels=np.array([0, 1, 0, 1, 0, 1], dtype=int),
        track_nmi={"T2": 0.42, "T1.5": 0.24, "T3": 0.18, "SYN": 0.81},
        synthesis_nmi=0.81,
    )

    _, nmi_scores = render_analysis_planes(exp)

    assert nmi_scores["T2"] == pytest.approx(0.42)
    assert nmi_scores["T1.5"] == pytest.approx(0.24)
    assert nmi_scores["T3"] == pytest.approx(0.18)
    assert nmi_scores["SYN"] == pytest.approx(0.81)


def test_mode_specific_camera_presets_are_applied(monkeypatch, tmp_path):
    fig_syn, html_syn = _render_html_for_mode(tmp_path / "syn", "synthesis", None, monkeypatch)
    fig_ana, html_ana = _render_html_for_mode(tmp_path / "ana", "analysis", None, monkeypatch)
    fig_dia, html_dia = _render_html_for_mode(tmp_path / "dia", "diagnostics", None, monkeypatch)

    eye_syn = tuple(float(fig_syn.layout.scene.camera.eye[k]) for k in ("x", "y", "z"))
    eye_ana = tuple(float(fig_ana.layout.scene.camera.eye[k]) for k in ("x", "y", "z"))
    eye_dia = tuple(float(fig_dia.layout.scene.camera.eye[k]) for k in ("x", "y", "z"))
    assert len({eye_syn, eye_ana, eye_dia}) >= 2

    for html_blob, expected_mode in ((html_syn, "synthesis"), (html_ana, "analysis"), (html_dia, "diagnostics")):
        assert f'var currentMode = "{expected_mode}";' in html_blob


def test_synthesis_shell_uses_instrument_panel_and_layer_toggles(monkeypatch, tmp_path):
    _, html_syn = _render_html_for_mode(tmp_path / "shell", "synthesis", None, monkeypatch)

    assert "Observer Compass" in html_syn
    assert ">SYNTHESIS</button>" in html_syn
    assert ">ANALYSIS</button>" not in html_syn
    assert ">DIAGNOSTICS</button>" not in html_syn
    assert "toggle-phantom-ribbons" in html_syn
    assert "toggle-honest-ribbons" in html_syn
    assert "toggle-tautology-ribbons" in html_syn
    assert "toggle-shear-flares" in html_syn
    assert "toggle-shear-labels" in html_syn
    assert "VIEW CONTRACT" not in html_syn
    assert "Show shear diagnostics" not in html_syn
    assert 'var SECONDARY_MODES_ENABLED = false;' in html_syn
    assert "â†”" not in html_syn


def test_synthesis_product_html_includes_analysis_button_but_not_diagnostics_button(monkeypatch, tmp_path):
    _require_plotly()
    exp = _make_experiment(tmp_path / "product_shell", spectral_probe_magnitudes=None)
    output_path = tmp_path / "product_shell" / "monolith_product.html"
    monkeypatch.delenv("MONOLITH_FAST_SYNTHESIS_ONLY", raising=False)
    monkeypatch.delenv("MONOLITH_INCLUDE_SECONDARY_MODES", raising=False)
    create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode="synthesis",
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=False,
        show_hott=False,
    )
    html_syn = output_path.read_text(encoding="utf-8")
    assert ">ANALYSIS</button>" in html_syn
    assert ">DIAGNOSTICS</button>" not in html_syn
    assert 'var SECONDARY_MODES_ENABLED = true;' in html_syn


def test_article_hitbox_trace_uses_large_click_target():
    _require_plotly()
    positions = np.array([[0.0, 0.0, 0.1], [1.0, 1.0, 0.2]], dtype=float)
    traces = render_data_points_3d(
        positions=positions,
        spectral_evr=np.array([0.8, 0.7], dtype=float),
        sizes=np.array([4.0, 6.0], dtype=float),
        hover_texts=["a", "b"],
        phantom_verdicts=[{"verdict": "HONEST"}, {"verdict": "PHANTOM"}],
    )
    hitbox = next(t for t in traces if getattr(t, "name", "") == "article_hitbox")
    hitbox_sizes = np.asarray(hitbox.marker.size, dtype=float)
    assert hitbox_sizes.min() >= 18.0


def test_path_invalid_points_are_filtered_not_origin_injected():
    _require_plotly()
    positions_3d = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [10.0, 20.0, 30.0],
                [np.nan, np.nan, np.nan],
                [11.0, 21.0, 31.0],
            ],
            dtype=float,
        ),
        1: np.array(
            [
                [15.0, 25.0, 35.0],
                [15.5, 25.5, 35.5],
            ],
            dtype=float,
        ),
    }
    phantom_verdicts = [
        {"verdict": "HONEST", "walker_state": "success"},
        {"verdict": "HONEST", "walker_state": "success"},
    ]

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
    )
    assert traces, "Expected at least one rendered path trace"
    coords = np.vstack(
        [
            np.column_stack(
                [
                    np.asarray(t.x, dtype=float),
                    np.asarray(t.y, dtype=float),
                    np.asarray(t.z, dtype=float),
                ]
            )
            for t in traces
        ]
    )
    assert np.isfinite(coords).all()
    assert not np.any(np.all(np.isclose(coords, 0.0), axis=1)), coords


def test_non_3d_walker_paths_are_not_rendered():
    _require_plotly()
    positions_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    walker_paths = {
        0: np.array([[0.0, 0.0], [0.5, 0.5]], dtype=float),
        1: np.array([0.0, 1.0, 2.0], dtype=float),
    }
    phantom_verdicts = [
        {"verdict": "HONEST", "walker_state": "success"},
        {"verdict": "PHANTOM", "walker_state": "success"},
    ]

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
    )

    assert traces == [], "Expected non-3D walker paths to be rejected without rendering"


def test_track4_chroma_ribbons_emit_variable_widths_and_shear_flares():
    _require_plotly()
    positions_3d = np.array([[0.0, 0.0, 0.0]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.4, 0.2, 0.1],
                [0.6, 0.4, 0.2],
            ],
            dtype=float,
        )
    }
    walker_path_diagnostics = {
        0: {
            "step_axis_idx": np.array([0, 3, 5], dtype=int),
            "step_axis_vectors": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            "step_local_friction": np.array([0.2, 0.9, 1.6], dtype=float),
            "step_work": np.array([0.1, 0.8, 1.5], dtype=float),
            "step_cumulative_work": np.array([0.1, 0.9, 2.4], dtype=float),
            "step_event_mask": np.array([False, True, False], dtype=bool),
            "step_event_severity": np.array([0.1, 0.95, 0.2], dtype=float),
        }
    }
    phantom_verdicts = [{"verdict": "PHANTOM", "walker_state": "success"}]

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
        walker_path_diagnostics=walker_path_diagnostics,
    )

    names = [str(getattr(t, "name", "")) for t in traces]
    assert "Phantom Ribbon" in names
    assert "Shear Flares" in names

    chroma_widths = [
        float(getattr(getattr(t, "line", None), "width", 0.0))
        for t in traces
        if str(getattr(t, "name", "")) == "Phantom Ribbon"
    ]
    assert chroma_widths
    assert max(chroma_widths) > min(chroma_widths), chroma_widths


def test_type2_rupture_is_canonicalized_and_rendered():
    _require_plotly()
    positions_3d = np.array([[0.0, 0.0, 0.0]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.2, 0.1],
                [0.4, 0.4, 0.2],
            ],
            dtype=float,
        )
    }

    traces = render_phantom_paths_3d(
        phantom_verdicts=[{"verdict": "TYPE_2_RUPTURE", "walker_state": "trapped"}],
        positions_3d=positions_3d,
        walker_paths=walker_paths,
    )

    names = [str(getattr(t, "name", "")) for t in traces]
    assert "Tautology Path" in names, names


def test_shear_flares_are_thresholded_and_capped():
    _require_plotly()
    positions_3d = np.array([[0.0, 0.0, 0.0]], dtype=float)
    path_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=float,
    )
    traces = render_phantom_paths_3d(
        phantom_verdicts=[{"verdict": "PHANTOM", "walker_state": "success"}],
        positions_3d=positions_3d,
        walker_paths={0: path_xyz},
        walker_path_diagnostics={
            0: {
                "step_axis_idx": np.array([0, 1, 2, 3, 4, 5], dtype=int),
                "step_cumulative_work": np.array([0.1, 0.2, 0.3, 4.5, 4.7, 9.9], dtype=float),
            }
        },
    )

    flare_traces = [t for t in traces if str(getattr(t, "name", "")) == "Shear Flares"]
    assert 1 <= len(flare_traces) <= 2
    for trace in flare_traces:
        assert float(trace.marker.size) == pytest.approx(3.0)
        assert float(trace.marker.opacity) == pytest.approx(0.8)


def test_cumulative_work_prevents_markovian_width_snapback():
    _require_plotly()
    positions_3d = np.array([[0.0, 0.0, 0.0]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            dtype=float,
        )
    }
    walker_path_diagnostics = {
        0: {
            "step_axis_vectors": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            "step_local_friction": np.array([1.2, 0.1, 0.1], dtype=float),
            "step_cumulative_work": np.array([0.3, 1.1, 2.0], dtype=float),
        }
    }
    phantom_verdicts = [{"verdict": "PHANTOM", "walker_state": "success"}]

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
        walker_path_diagnostics=walker_path_diagnostics,
    )

    chroma_widths = [
        float(getattr(getattr(t, "line", None), "width", 0.0))
        for t in traces
        if str(getattr(t, "name", "")) == "Phantom Ribbon"
    ]
    assert chroma_widths == sorted(chroma_widths), chroma_widths


def test_honest_path_falls_back_cleanly_when_chroma_diagnostics_absent():
    _require_plotly()
    positions_3d = np.array([[1.0, 2.0, 3.0]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [1.0, 2.0, 3.0],
                [1.2, 2.3, 3.2],
            ],
            dtype=float,
        )
    }
    phantom_verdicts = [{"verdict": "HONEST", "walker_state": "success"}]

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
        walker_path_diagnostics=None,
    )

    names = [str(getattr(t, "name", "")) for t in traces]
    assert "Honest Path" in names


def test_phantom_path_restores_visible_shear_label_without_step_diagnostics(monkeypatch):
    _require_plotly()
    monkeypatch.setenv("MONOLITH_SHOW_SHEAR_LABELS", "1")
    positions_3d = np.array([[0.0, 0.0, 0.1]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [0.0, 0.0, 0.1],
                [0.3, 0.2, 0.2],
                [0.5, 0.4, 0.3],
            ],
            dtype=float,
        )
    }
    phantom_verdicts = [{"verdict": "PHANTOM", "walker_state": "success"}]
    spectral_probe_magnitudes = np.array([[0.1, 0.2, 0.4, 1.8, 0.3, 0.0, 0.1, 0.2]], dtype=float)

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
        walker_path_diagnostics=None,
        spectral_probe_magnitudes=spectral_probe_magnitudes,
    )

    shear_traces = [t for t in traces if str(getattr(t, "name", "")) == "Ideological Shear"]
    assert shear_traces, "Expected visible PHANTOM shear text label on old-schema path data"
    text_values = [str(txt) for t in shear_traces for txt in getattr(t, "text", [])]
    assert any(text.startswith("[SHEAR:") for text in text_values), text_values


def test_paths_are_draped_to_surface_height_with_positive_offset():
    _require_plotly()
    positions_3d = np.array([[0.0, 0.0, -5.0]], dtype=float)
    walker_paths = {
        0: np.array(
            [
                [0.0, 0.0, -5.0],
                [0.2, 0.3, -4.0],
            ],
            dtype=float,
        )
    }
    phantom_verdicts = [{"verdict": "HONEST", "walker_state": "success"}]

    def _surface_z(xs, ys, offset=0.0, preserve_nan=False):
        xs_arr = np.asarray(xs, dtype=float)
        return np.full(xs_arr.shape, 1.0 + offset, dtype=float)

    traces = render_phantom_paths_3d(
        phantom_verdicts=phantom_verdicts,
        positions_3d=positions_3d,
        walker_paths=walker_paths,
        surface_z_func=_surface_z,
    )

    honest_traces = [t for t in traces if str(getattr(t, "name", "")) == "Honest Path"]
    assert honest_traces, "Expected draped honest path trace"
    for trace in honest_traces:
        zs = np.asarray(trace.z, dtype=float)
        assert np.allclose(zs, 1.0), zs


def test_axis_labels_are_data_driven_when_spectral_present(monkeypatch, tmp_path):
    spectral = np.zeros((4, 8), dtype=float)
    spectral[:, 3] = np.array([10.0, 9.0, 8.0, 7.0], dtype=float)
    fig, _ = _render_html_for_mode(tmp_path / "spectral", "synthesis", spectral, monkeypatch)

    titles = {
        str(fig.layout.scene.xaxis.title.text),
        str(fig.layout.scene.yaxis.title.text),
        str(fig.layout.scene.zaxis.title.text),
    }
    assert titles != {"Semantic Axis X", "Semantic Axis Y", "Semantic Axis Z"}
    assert any(any(label in title for label in PROBE_LABELS) for title in titles), titles


def test_axis_labels_fallback_when_spectral_absent(monkeypatch, tmp_path):
    fig, _ = _render_html_for_mode(tmp_path / "fallback", "synthesis", None, monkeypatch)

    assert str(fig.layout.scene.xaxis.title.text) == "Semantic Axis X"
    assert str(fig.layout.scene.yaxis.title.text) == "Semantic Axis Y"
    assert str(fig.layout.scene.zaxis.title.text) == "Semantic Axis Z"


def test_extreme_mismatched_path_scale_is_bounded_relative_to_article_manifold(monkeypatch, tmp_path):
    _require_plotly()
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    exp.walker_paths[1] = np.array(
        [
            [1_000_000.0, -1_000_000.0, 0.0],
            [1_200_000.0, -800_000.0, 0.1],
            [1_400_000.0, -1_100_000.0, 0.2],
        ],
        dtype=float,
    )
    output_path = tmp_path / "extreme_mismatch.html"
    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    fig = create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode="synthesis",
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )

    article_traces = [t for t in fig.data if str(getattr(t, "name", "")) == "Articles"]
    assert article_traces, "Expected article manifold trace"
    article_trace = article_traces[0]
    article_x = np.asarray(article_trace.x, dtype=float)
    article_y = np.asarray(article_trace.y, dtype=float)
    article_xy_span = float(max(np.ptp(article_x), np.ptp(article_y)))
    assert np.isfinite(article_xy_span) and article_xy_span > 0.0

    path_names = {"Honest Path", "Phantom Path", "Tautology Path", "Honest Ribbon", "Phantom Ribbon", "Tautology Ribbon"}
    path_traces = [
        t
        for t in fig.data
        if str(getattr(t, "mode", "")) == "lines" and str(getattr(t, "name", "")) in path_names
    ]
    assert path_traces, "Expected rendered path traces"

    max_span_ratio = 2.01
    for t in path_traces:
        tx = np.asarray(t.x, dtype=float)
        ty = np.asarray(t.y, dtype=float)
        path_xy_span = float(max(np.ptp(tx), np.ptp(ty)))
        assert np.isfinite(path_xy_span)
        assert path_xy_span <= article_xy_span * max_span_ratio


def test_terrain_footprint_expands_to_cover_rendered_paths(monkeypatch, tmp_path):
    _require_plotly()
    pytest.importorskip("scipy")
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    exp.walker_paths[1] = np.array(
        [
            [1_000_000.0, -1_000_000.0, 0.0],
            [1_200_000.0, -800_000.0, 0.1],
            [1_400_000.0, -1_100_000.0, 0.2],
        ],
        dtype=float,
    )
    output_path = tmp_path / "terrain_footprint.html"
    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    fig = create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode="synthesis",
        show_terrain=True,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )

    terrain_traces = [t for t in fig.data if str(getattr(t, "name", "")) == "Energy Terrain"]
    assert terrain_traces, "Expected terrain surface trace"
    terrain = terrain_traces[0]
    terrain_x = np.asarray(terrain.x, dtype=float)
    terrain_y = np.asarray(terrain.y, dtype=float)
    terrain_x_finite = terrain_x[np.isfinite(terrain_x)]
    terrain_y_finite = terrain_y[np.isfinite(terrain_y)]
    assert terrain_x_finite.size > 0 and terrain_y_finite.size > 0
    terrain_x_min = float(np.min(terrain_x_finite))
    terrain_x_max = float(np.max(terrain_x_finite))
    terrain_y_min = float(np.min(terrain_y_finite))
    terrain_y_max = float(np.max(terrain_y_finite))

    path_names = {"Honest Path", "Phantom Path", "Tautology Path", "Honest Ribbon", "Phantom Ribbon", "Tautology Ribbon"}
    path_traces = [
        t
        for t in fig.data
        if str(getattr(t, "mode", "")) == "lines" and str(getattr(t, "name", "")) in path_names
    ]
    assert path_traces, "Expected rendered path traces"

    eps = 1e-6
    for trace in path_traces:
        tx = np.asarray(trace.x, dtype=float)
        ty = np.asarray(trace.y, dtype=float)
        tx = tx[np.isfinite(tx)]
        ty = ty[np.isfinite(ty)]
        assert tx.size > 0 and ty.size > 0
        assert float(np.min(tx)) >= terrain_x_min - eps
        assert float(np.max(tx)) <= terrain_x_max + eps
        assert float(np.min(ty)) >= terrain_y_min - eps
        assert float(np.max(ty)) <= terrain_y_max + eps


def test_non_rendered_non_3d_paths_do_not_expand_terrain_footprint(monkeypatch, tmp_path):
    _require_plotly()
    pytest.importorskip("scipy")
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    exp.walker_paths[0] = np.array(
        [
            [1_000_000.0, -1_000_000.0],
            [1_400_000.0, -800_000.0],
        ],
        dtype=float,
    )
    output_path = tmp_path / "invalid_path_footprint.html"
    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    fig = create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode="synthesis",
        show_terrain=True,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )

    terrain_traces = [t for t in fig.data if str(getattr(t, "name", "")) == "Energy Terrain"]
    article_traces = [t for t in fig.data if str(getattr(t, "name", "")) == "Articles"]
    path_names = {"Honest Path", "Phantom Path", "Tautology Path", "Honest Ribbon", "Phantom Ribbon", "Tautology Ribbon"}
    path_traces = [
        t
        for t in fig.data
        if str(getattr(t, "mode", "")) == "lines" and str(getattr(t, "name", "")) in path_names
    ]

    assert terrain_traces, "Expected terrain surface trace"
    assert article_traces, "Expected article manifold trace"
    assert path_traces, "Expected at least one valid rendered path trace"

    terrain = terrain_traces[0]
    article_trace = article_traces[0]
    terrain_x = np.asarray(terrain.x, dtype=float)
    terrain_y = np.asarray(terrain.y, dtype=float)
    article_x = np.asarray(article_trace.x, dtype=float)
    article_y = np.asarray(article_trace.y, dtype=float)

    terrain_span = float(
        max(
            np.ptp(terrain_x[np.isfinite(terrain_x)]),
            np.ptp(terrain_y[np.isfinite(terrain_y)]),
        )
    )
    article_span = float(
        max(
            np.ptp(article_x[np.isfinite(article_x)]),
            np.ptp(article_y[np.isfinite(article_y)]),
        )
    )
    assert np.isfinite(terrain_span) and np.isfinite(article_span) and article_span > 0.0
    assert terrain_span <= article_span * 3.0, (terrain_span, article_span)

    rendered_xy = np.vstack(
        [
            np.column_stack([np.asarray(t.x, dtype=float), np.asarray(t.y, dtype=float)])
            for t in path_traces
        ]
    )
    assert np.nanmax(np.abs(rendered_xy)) < 100.0, rendered_xy


def test_focused_observer_render_uses_relativity_cache_geometry(monkeypatch, tmp_path):
    _require_plotly()
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    rel_dir = exp.experiment_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    (rel_dir / "state_1.json").write_text(
        """
{
  "observer_id": 1,
  "articles": [
    {"index": 0, "observer_x": -3.0, "observer_y": 0.2, "observer_z": 0.1, "observer_probe_similarity": 0.91, "coord_delta": 1.7},
    {"index": 1, "observer_x": 0.0, "observer_y": 0.0, "observer_z": 0.4, "observer_probe_similarity": 1.00, "coord_delta": 0.0},
    {"index": 2, "observer_x": 2.2, "observer_y": 0.8, "observer_z": -0.2, "observer_probe_similarity": 0.62, "coord_delta": 1.2},
    {"index": 3, "observer_x": 3.5, "observer_y": 1.6, "observer_z": -0.5, "observer_probe_similarity": 0.33, "coord_delta": 2.4}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    (rel_dir / "delta_1.json").write_text(
        """
{
  "observer_id": 1,
  "null_observer_equivalence": {
    "max_coord_delta": 2.4,
    "path_flip_count": 5,
    "axis_rotation_deg": 17.5
  }
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    base_output = tmp_path / "base_focus_compare.html"
    focus_output = tmp_path / "observer_focus_compare.html"
    base_fig = create_monolith_cockpit(
        exp=exp,
        output_path=base_output,
        physics_mode="synthesis",
        observer_idx=None,
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )
    focus_fig = create_monolith_cockpit(
        exp=exp,
        output_path=focus_output,
        physics_mode="synthesis",
        observer_idx=1,
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )

    base_articles = next(t for t in base_fig.data if str(getattr(t, "name", "")) == "Articles")
    focus_articles = next(t for t in focus_fig.data if str(getattr(t, "name", "")) == "Articles")
    base_x = np.asarray(base_articles.x, dtype=float)
    focus_x = np.asarray(focus_articles.x, dtype=float)
    base_y = np.asarray(base_articles.y, dtype=float)
    focus_y = np.asarray(focus_articles.y, dtype=float)

    assert not np.allclose(base_x, focus_x)
    assert not np.allclose(base_y, focus_y)

    focus_html = focus_output.read_text(encoding="utf-8")
    assert "OBSERVER FOCUS | article:1" in focus_html
    assert "coord=2.400" in focus_html
    assert "flips=5" in focus_html
    assert "axis=17.5deg" in focus_html
    assert "Observer Probe Sim" in focus_html
    assert "Observer Coord Delta" in focus_html


def test_recompute_focused_observer_track4_uses_payload_replay(tmp_path):
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    exp.antagonism = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    cls_per_bot = torch.tensor(
        [
            [[2.0, 0.0, 0.0, 0.0], [1.8, 0.1, 0.0, 0.0], [1.5, 0.2, 0.0, 0.0], [1.2, 0.3, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], [0.5, 0.9, 0.0, 0.0], [0.2, 1.2, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0], [0.0, 1.8, 0.1, 0.0], [0.0, 1.5, 0.2, 0.0], [0.0, 1.2, 0.3, 0.0], [0.0, 0.8, 0.6, 0.0], [0.0, 0.5, 0.9, 0.0], [0.0, 0.2, 1.2, 0.0], [0.0, 0.0, 1.5, 0.0]],
            [[0.0, 0.0, 2.0, 0.0], [0.1, 0.0, 1.8, 0.0], [0.2, 0.0, 1.5, 0.0], [0.3, 0.0, 1.2, 0.0], [0.6, 0.0, 0.8, 0.0], [0.9, 0.0, 0.5, 0.0], [1.2, 0.0, 0.2, 0.0], [1.5, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 2.0], [0.0, 0.1, 0.0, 1.8], [0.0, 0.2, 0.0, 1.5], [0.0, 0.3, 0.0, 1.2], [0.0, 0.6, 0.0, 0.8], [0.0, 0.9, 0.0, 0.5], [0.0, 1.2, 0.0, 0.2], [0.0, 1.5, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    basis = SharedRKSBasis(input_dim=4, output_dim=8, seed=17, kernel_type="rbf")
    basis.set_sigma(1.0)
    torch.save(
        {
            "cls_per_bot": cls_per_bot.numpy(),
            "rks_basis_state": _serialize_rks_basis_state(basis),
        },
        exp.experiment_dir / "observer_42.pt",
    )

    replay = _recompute_focused_observer_track4(exp, 1, exp.walker_paths)

    assert replay is not None
    assert replay["replay_steps"] >= 10
    assert np.asarray(replay["path_xyz"], dtype=float).ndim == 2
    assert np.asarray(replay["path_xyz"], dtype=float).shape[0] >= 2
    assert len(replay["step_diagnostics"]) > 0
    assert np.isfinite(float(replay["work_integral"]))
    assert not np.allclose(
        np.asarray(replay["path_xyz"], dtype=float)[:2, :3],
        np.asarray(exp.walker_paths[1], dtype=float),
        atol=1e-5,
    )


def test_focused_observer_render_falls_back_to_global_z_when_observer_z_collapses(monkeypatch, tmp_path):
    _require_plotly()
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    rel_dir = exp.experiment_dir / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    (rel_dir / "state_0.json").write_text(
        """
{
  "observer_id": 0,
  "articles": [
    {"index": 0, "observer_x": -2.0, "observer_y": 0.0, "observer_z": 0.0},
    {"index": 1, "observer_x": -0.5, "observer_y": 0.6, "observer_z": 0.0},
    {"index": 2, "observer_x": 1.0, "observer_y": 1.2, "observer_z": 0.0},
    {"index": 3, "observer_x": 2.5, "observer_y": 1.8, "observer_z": 0.0}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    focus_output = tmp_path / "observer_focus_z_fallback.html"
    focus_fig = create_monolith_cockpit(
        exp=exp,
        output_path=focus_output,
        physics_mode="synthesis",
        observer_idx=0,
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )

    article_trace = next(t for t in focus_fig.data if str(getattr(t, "name", "")) == "Articles")
    article_z = np.asarray(article_trace.z, dtype=float)
    assert float(np.ptp(article_z)) > 1e-3


def test_dash_embed_html_prefers_live_dash_and_keeps_local_observer_fallback_hint(monkeypatch, tmp_path):
    _require_plotly()
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    output_path = tmp_path / "dash_fallback.html"
    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode="synthesis",
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )

    html_text = output_path.read_text(encoding="utf-8")
    assert "probeDashReachable" in html_text
    assert "observer_' + String(Math.floor(articleRef.idx)) + '/MONOLITH.html" in html_text
    assert "frame.src = dashUrl.origin + '/?' + qs.toString();" in html_text
    assert "qs.set('variant_a', DASH_VARIANT_NAME);" in html_text
    assert "qs.set('variant_b', DASH_VARIANT_NAME);" in html_text


def test_render_terrain_surface_keeps_concave_center_connected():
    _require_plotly()
    pytest.importorskip("scipy")
    positions_xy = np.array(
        [
            [-2.0, -2.0],
            [-2.0, -1.0],
            [-2.0, 0.0],
            [-2.0, 1.0],
            [-1.0, -2.0],
            [0.0, -2.0],
            [1.0, -2.0],
            [2.0, -2.0],
            [2.0, -1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    energy = np.linspace(0.0, 1.0, len(positions_xy), dtype=float)
    positions_3d = np.column_stack([positions_xy, energy])

    _, Xi, Yi, Zi, _, _ = render_terrain_surface(
        positions_3d=positions_3d,
        energy_values=energy,
        grid_resolution=80,
        terrain_density=np.linspace(0.2, 0.9, len(positions_xy), dtype=float),
        terrain_stress=np.linspace(0.1, 0.8, len(positions_xy), dtype=float),
    )

    assert Xi is not None and Yi is not None and Zi is not None
    center_idx = np.unravel_index(np.nanargmin((Xi ** 2) + (Yi ** 2)), Xi.shape)
    assert np.isfinite(Zi[center_idx]), Zi[center_idx]


def test_render_terrain_surface_ignores_far_path_support_for_occupancy():
    _require_plotly()
    pytest.importorskip("scipy")
    positions_xy = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    energy = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    positions_3d = np.column_stack([positions_xy, energy])
    terrain_support_xy = np.array(
        [
            [10.0, 10.0],
            [10.5, 10.2],
            [10.2, 10.7],
        ],
        dtype=float,
    )

    _, Xi, Yi, Zi, _, _ = render_terrain_surface(
        positions_3d=positions_3d,
        energy_values=energy,
        grid_resolution=80,
        terrain_density=np.linspace(0.2, 0.8, len(positions_xy), dtype=float),
        terrain_stress=np.linspace(0.1, 0.7, len(positions_xy), dtype=float),
        terrain_support_xy=terrain_support_xy,
    )

    assert Xi is not None and Yi is not None and Zi is not None
    far_idx = np.unravel_index(np.nanargmin((Xi - 10.0) ** 2 + (Yi - 10.0) ** 2), Xi.shape)
    assert np.isnan(Zi[far_idx]), Zi[far_idx]


def test_compress_surface_height_field_tames_single_spike():
    values = np.array([0.05, 0.08, 0.12, 0.20, 0.35, 0.55, 1.10, 9.21], dtype=float)
    compressed = _compress_surface_height_field(values)
    assert compressed.shape == values.shape
    assert float(np.max(compressed)) < float(np.max(values))
    assert float(np.nanpercentile(compressed, 95.0)) < float(np.nanpercentile(values, 95.0))
    assert np.all(np.diff(compressed[np.argsort(values)]) >= -1e-9)


def test_synthesis_default_hides_spectral_axis_vectors(monkeypatch, tmp_path):
    _require_plotly()
    exp = _make_experiment(tmp_path, spectral_probe_magnitudes=None)
    exp.spectral_probe_magnitudes = np.tile(np.linspace(-1.0, 1.0, 8, dtype=float), (exp.n_articles, 1))
    output_path = tmp_path / "no_vectors.html"
    monkeypatch.setenv("MONOLITH_FAST_SYNTHESIS_ONLY", "1")
    fig = create_monolith_cockpit(
        exp=exp,
        output_path=output_path,
        physics_mode="synthesis",
        show_terrain=False,
        show_fog=False,
        show_walkers=False,
        show_phantom_paths=True,
        show_hott=False,
    )
    assert not any(str(getattr(t, "name", "")).startswith("Vector: ") for t in fig.data)
