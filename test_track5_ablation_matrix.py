from pathlib import Path

from analysis.verification.thesis_evidence import build_thesis_evidence
from thesis_test_support import (
    build_canonical_fixture,
    write_ablation_summary,
    write_baseline_meta,
    write_control_metrics,
    write_methods,
    write_monolith_data,
    write_results,
    write_relativity_deltas,
    write_suite_manifest,
    write_verification_report,
    write_view_state,
    write_walker_states,
)


def _populate_leaf(run_dir: Path, *, kernel: str, channel: str, corpus: str, track5_mode: str) -> None:
    leaf_dir = run_dir / kernel / channel / corpus
    write_baseline_meta(
        leaf_dir / "baseline_meta.json",
        kernel=kernel,
        channel=channel,
        track5_mode=track5_mode,
    )
    write_verification_report(leaf_dir / "verification_report.json")
    write_control_metrics(leaf_dir / "control_metrics.json")
    write_view_state(leaf_dir / "MONOLITH.view_state.json")
    write_walker_states(leaf_dir / "walker_states.json")
    write_monolith_data(leaf_dir / "MONOLITH_DATA.csv")
    if corpus == "real":
        write_relativity_deltas(leaf_dir / "relativity_deltas.json")
        write_ablation_summary(leaf_dir / "ablation_summary.json", stage_2=0.5 if "hadamard" in track5_mode else 0.6)


def test_track5_ablation_matrix_groups_modes_with_fixed_template(tmp_path: Path):
    methods_path = tmp_path / "METHODS.md"
    results_path = tmp_path / "RESULTS.md"
    runs_dir = tmp_path / "outputs" / "experiments" / "runs"
    write_methods(methods_path)
    write_results(
        results_path,
        [
            "experiments_20260504_010101",
            "experiments_20260504_020202",
            "experiments_20260504_030303",
        ],
    )

    modes = {
        "experiments_20260504_010101": "hadamard_strict",
        "experiments_20260504_020202": "riemannian_strict",
        "experiments_20260504_030303": "concatenate",
    }
    for run_id, mode in modes.items():
        run_dir = runs_dir / run_id
        write_suite_manifest(
            run_dir / "experiment_manifest.json",
            run_id=run_id,
            kernels=["rbf"],
            channels=["cls"],
            corpora=["real", "control_constant", "control_shuffled", "control_random"],
            seeds=[42, 420, 4200],
            base_run_dir=run_dir,
        )
        for corpus in ["real", "control_constant", "control_shuffled", "control_random"]:
            _populate_leaf(run_dir, kernel="rbf", channel="cls", corpus=corpus, track5_mode=mode)

    payloads = build_thesis_evidence(
        runs_dir=runs_dir,
        methods_path=methods_path,
        results_path=results_path,
    )

    by_mode = payloads["ablation_matrix"]["by_track5_mode"]
    assert {"hadamard_strict", "riemannian_strict", "concatenate"} == set(by_mode.keys())
    assert all(metrics["n_runs"] == 1 for metrics in by_mode.values())


def test_track5_claim_requires_riemannian_ablation_coverage(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, ablation_modes=["hadamard_strict"])

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track5_ablation_coverage"
    )
    assert not claim["thesis_safe"]
    assert not payloads["ablation_matrix"]["required_modes_present"]


def test_track5_claim_requires_nonzero_ablation_effect(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    weak_summary = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "ablation_summary.json"
    )
    write_ablation_summary(
        weak_summary,
        stage_2=0.0,
        stage_3=0.0,
        delta=0.0,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track5_ablation_coverage"
    )
    assert not claim["thesis_safe"]
    record = next(
        row for row in payloads["ablation_matrix"]["records"]
        if row["kernel"] == "rbf" and row["channel"] == "cls" and row["run_id"] == "experiments_20260504_010101"
    )
    assert record["effect_pass"] is False
