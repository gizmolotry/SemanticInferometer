import json
from pathlib import Path

from core.observer_local_recompute import LOCAL_RECOMPUTE_DEFAULT_VARIANT
from scripts.run_observer_recenter_robustness_suite import LOCAL_VARIANT_BASELINES
from scripts.run_prompt_invariance_probe import run_probe
from scripts.run_track3_density_validation import run_validation


def test_uniform_weighted_rks_is_default_and_focus_remains_ablation() -> None:
    assert LOCAL_RECOMPUTE_DEFAULT_VARIANT == "uniform_weighted_rks"
    assert LOCAL_VARIANT_BASELINES["local_track_recompute:focus_weighted_rks"] == "focus_weighted_rks"


def test_prompt_invariance_probe_blocks_single_bank(tmp_path: Path) -> None:
    bank = tmp_path / "probe_hypotheses.json"
    bank.write_text(json.dumps(["A", "B"]), encoding="utf-8")

    payload = run_probe(prompt_banks=[("canonical", bank)], output_dir=tmp_path / "out")

    assert payload["prompt_invariance_pass"] is False
    assert "fewer_than_three_prompt_banks" in payload["failure_reasons"]
    assert (tmp_path / "out" / "prompt_invariance_summary.json").exists()


def test_track3_density_validation_passes_when_density_tracks_work_proxy(tmp_path: Path) -> None:
    cell = tmp_path / "synthetic" / "rbf_seed42"
    cell.mkdir(parents=True)
    lines = ["index,density,stress,w_actual,perspective_tag"]
    for idx in range(12):
        density = 1.0 - (idx / 20.0)
        stress = idx / 20.0
        work = float(idx)
        lines.append(f"{idx},{density},{stress},{work},label_{idx % 2}")
    (cell / "MONOLITH_DATA.csv").write_text("\n".join(lines), encoding="utf-8")

    payload = run_validation(
        synthetic_root=tmp_path / "synthetic",
        output_dir=tmp_path / "out",
        kernels=["rbf"],
        seeds=[42],
    )

    assert payload["valid_cell_count"] == 1
    assert payload["pass_count"] == 1
    assert payload["track3_density_semantic_pass"] is False
    assert "fewer_than_three_valid_cells" in payload["failure_reasons"]
    assert payload["independent_track3_validation"] is True
    assert payload["cells"][0]["independent_track3_validation"] is True
    assert payload["cells"][0]["cell_pass"] is True


def test_track3_action_calibrated_variant_is_marked_non_independent(tmp_path: Path) -> None:
    cell = tmp_path / "synthetic" / "rbf_seed42"
    cell.mkdir(parents=True)
    lines = ["index,density,stress,w_actual,perspective_tag"]
    for idx in range(12):
        lines.append(f"{idx},0.5,0.5,{float(idx)},label_{idx % 2}")
    (cell / "MONOLITH_DATA.csv").write_text("\n".join(lines), encoding="utf-8")

    payload = run_validation(
        synthetic_root=tmp_path / "synthetic",
        output_dir=tmp_path / "out",
        kernels=["rbf"],
        seeds=[42],
        variant="action_calibrated",
    )

    assert payload["variant"] == "action_calibrated"
    assert payload["independent_track3_validation"] is False
    assert payload["cells"][0]["independent_track3_validation"] is False
    assert payload["target_independence_level"] == "action_calibrated_ablation_not_independent_ground_truth"
