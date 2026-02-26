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
    assert "T5: {n_honest}H / {n_phantoms}P / {n_ruptures}R / {n_tautology}T" in s
    assert "T4: Act {mean_action:.2f} | Surv {survival_rate:.0%}" in s


def test_phantom_path_traces_are_lines_for_primary_verdicts():
    s = _src()
    # Guard the specific path trace blocks, not unrelated marker layers.
    for verdict_name in [
        "Tautology (Spinning)",
        "Honest (Cyan River)",
        "Phantom (Magenta Knot)",
        "Walker Broken (Kinetic)",
        "Walker Trapped (Topologic)",
    ]:
        i = s.find(f"name='{verdict_name}'")
        assert i != -1, f"Could not find trace block for {verdict_name}"
        # Scan a short window before the name for mode declaration.
        window = s[max(0, i - 300):i]
        assert ("mode='lines'" in window) or ("mode='lines+markers'" in window), \
            f"{verdict_name} is not rendered as line path"


def test_contour_traces_are_lines_and_legend_decluttered():
    s = _src()
    # Density/stress isolines should be line traces and not flood legend.
    assert "name=f'Density isoline {level:.0%}'" in s
    assert "name=f'Stress isoline {level:.0%}'" in s
    assert "showlegend=False" in s
    # Keep a minimal signature guard on contour mode.
    assert "mode='lines'" in s


if __name__ == "__main__":
    tests = [
        test_no_flattening_signatures_in_active_viz,
        test_hud_includes_tautology_and_t4_physics_summary,
        test_phantom_path_traces_are_lines_for_primary_verdicts,
        test_contour_traces_are_lines_and_legend_decluttered,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
