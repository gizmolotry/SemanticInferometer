from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis.MONOLITH_VIZ import (
    PROBE_LABELS,
    ExperimentData,
    create_monolith_cockpit,
    render_analysis_planes,
    render_phantom_paths_3d,
)


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


def test_rupture_path_rendering_reachable_behavior_contract():
    _require_plotly()
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
    assert any(n in {"Walker Broken", "Walker Trapped"} for n in names), names


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
    assert "Track 4 Axis Chroma" in names
    assert "Shear Flares" in names

    chroma_widths = [
        float(getattr(getattr(t, "line", None), "width", 0.0))
        for t in traces
        if str(getattr(t, "name", "")) == "Track 4 Axis Chroma"
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
    assert "Walker Trapped" in names, names


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
        if str(getattr(t, "name", "")) == "Track 4 Axis Chroma"
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


def test_phantom_path_restores_visible_shear_label_without_step_diagnostics():
    _require_plotly()
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
        assert np.allclose(zs, 1.05), zs


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

    path_names = {"Honest Path", "Phantom Path", "Tautology Path", "Walker Broken", "Walker Trapped"}
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
