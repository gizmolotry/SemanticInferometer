import numpy as np
import pytest

from analysis.MONOLITH_VIZ import compute_terrain_field, render_terrain_surface


def test_compute_terrain_field_preserves_four_corner_scalar_contract():
    # blinker: low variance->high density after inversion
    blinker = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    # walker: low->high stress
    walker = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    density, stress, scalar = compute_terrain_field(blinker, walker)

    # Corner order: BRIDGE, TIGHTROPE, SWAMP, VOID
    expected_density = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
    expected_stress = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    expected_scalar = np.array([0.0, 0.25, 0.75, 1.0], dtype=float)

    assert np.allclose(density, expected_density, atol=1e-9)
    assert np.allclose(stress, expected_stress, atol=1e-9)
    assert np.allclose(scalar, expected_scalar, atol=1e-9)


def test_compute_terrain_field_can_express_all_four_zone_bands():
    # Construct values that normalize to all (density, stress) corner combinations.
    blinker = np.array([0.0, 1.0, 0.0, 1.0, 0.4, 0.6], dtype=float)
    walker = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5], dtype=float)
    _, _, scalar = compute_terrain_field(blinker, walker)
    # Expect low, tightrope band, swamp band, and high represented.
    assert float(np.nanmin(scalar)) <= 0.01
    assert np.any((scalar >= 0.20) & (scalar <= 0.30))
    assert np.any((scalar >= 0.70) & (scalar <= 0.80))
    assert float(np.nanmax(scalar)) >= 0.99


def test_render_terrain_surface_uses_full_gradient_range_with_full_manifold_input():
    # Skip if optional rendering deps are unavailable in test environment.
    try:
        import scipy  # noqa: F401
        import plotly  # noqa: F401
    except Exception:
        pytest.skip("plotly/scipy unavailable for terrain render regression test")

    xs = np.linspace(-1.0, 1.0, 7)
    ys = np.linspace(-1.0, 1.0, 7)
    xx, yy = np.meshgrid(xs, ys)
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    density = (x_flat + 1.0) / 2.0
    stress = (y_flat + 1.0) / 2.0
    energy = stress.copy()
    positions = np.column_stack([x_flat, y_flat, energy])

    terrain, *_ = render_terrain_surface(
        positions_3d=positions,
        energy_values=energy,
        grid_resolution=60,
        terrain_density=density,
        terrain_stress=stress,
        use_manifold_colormap=True,
    )

    assert terrain is not None
    surfacecolor = np.asarray(terrain.surfacecolor, dtype=float)
    assert float(np.nanmin(surfacecolor)) < 0.1
    assert float(np.nanmax(surfacecolor)) > 0.9
