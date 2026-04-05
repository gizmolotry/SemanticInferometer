from pathlib import Path
import re


VIZ_PATH = Path("analysis/MONOLITH_VIZ.py")


def _src() -> str:
    return VIZ_PATH.read_text(encoding="utf-8")


def test_no_flattening_signatures_in_active_viz():
    s = _src()
    bad_patterns = [
        r"z\s*=\s*np\.zeros_like\(xg\)",
        r"z\s*=\s*np\.zeros\(",
        r"Surface\(z=None",
    ]
    for pat in bad_patterns:
        assert re.search(pat, s) is None, f"Flattening signature found: {pat}"


def test_hud_includes_tautology_and_t4_physics_summary():
    s = _src()
    assert "T5: {n_honest}H / {n_phantoms}P / {n_tautology}T | " in s
    assert "T4: Act {mean_action:.2f} | Surv {survival_rate:.0%}" in s


def test_phantom_path_traces_are_lines_for_primary_verdicts():
    s = _src()
    # Family-A path template should be line traces.
    path_tpl = "legend_name = f'{verdict.capitalize()} Path'"
    i = s.find(path_tpl)
    assert i != -1, "Could not find dynamic Family-A path naming template"
    window = s[i:i + 2600]
    assert "mode='lines'" in window, "Family-A paths are not rendered as lines"

    # Family-B break traces should remain explicit line traces.
    rupture_tpl = "legend_name = 'Walker Broken' if walker_state == \"broken\" else 'Walker Trapped'"
    i = s.find(rupture_tpl)
    assert i != -1, "Could not find rupture path legend template"
    window = s[i:i + 700]
    assert "mode='lines'" in window, "Rupture path traces are not rendered as lines"


def test_contour_traces_are_lines_and_legend_decluttered():
    s = _src()
    # Density isolines are rendered as marker sprites with one legend item per level.
    assert "name=f'Density isoline {level:.0%}'" in s
    assert "mode='markers'" in s
    assert "showlegend=True" in s


def test_article_z_uses_surface_interpolator_path_when_terrain_exists():
    s = _src()
    # Terrain-on branch must construct the surface interpolator.
    assert "if show_terrain and terrain_grid_x is not None and terrain_grid_y is not None and terrain_grid_z is not None:" in s
    assert "interp_terrain_z = RegularGridInterpolator(" in s

    # Surface-Z helper must route through the interpolator when available.
    assert "def get_surface_z(x_coords, y_coords, offset=0.0, preserve_nan=False):" in s
    assert "if interp_terrain_z is not None:" in s
    assert "interp_terrain_z(points_for_interp)" in s

    # Article markers should consume surface-anchored marker Z.
    assert "article_marker_z = np.asarray(energy_values_for_points, dtype=float)" in s
    assert "article_z_height=article_marker_z" in s


def test_render_contract_guards_path_span_vs_surface_span():
    s = _src()
    assert "# Render contract (fail-fast): if terrain is enabled" in s
    assert "if show_terrain:" in s
    assert "surface_span = float(max(np.ptp(sx), np.ptp(sy), np.ptp(sz)))" in s
    assert "max_path_span = 0.0" in s
    assert '(\"Path\" not in trace_name) and (trace_name not in {"Walker Broken", "Walker Trapped"})' in s
    assert "path_span = float(max(np.ptp(tx), np.ptp(ty), np.ptp(tz)))" in s
    assert "if max_path_span > (surface_span * 50.0):" in s
    assert "Path traces exceed terrain scale budget" in s


def test_terrain_z_fallback_guard_handles_invalid_arrays():
    s = _src()
    assert "terrain_values_valid = False" in s
    assert "terrain_arr = np.asarray(energy_values_for_terrain, dtype=float)" in s
    assert "if not terrain_values_valid:" in s
    assert "Falling back terrain Z to pure_z due to invalid terrain field." in s


if __name__ == "__main__":
    tests = [
        test_no_flattening_signatures_in_active_viz,
        test_hud_includes_tautology_and_t4_physics_summary,
        test_phantom_path_traces_are_lines_for_primary_verdicts,
        test_contour_traces_are_lines_and_legend_decluttered,
        test_article_z_uses_surface_interpolator_path_when_terrain_exists,
        test_render_contract_guards_path_span_vs_surface_span,
        test_terrain_z_fallback_guard_handles_invalid_arrays,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
