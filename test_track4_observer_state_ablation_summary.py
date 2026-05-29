from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_track4_action_graph_runs import (
    evaluate_observer_state_claim,
    summarize_observer_state_ablation,
)


REQUIRED_KERNELS = ("rbf", "matern", "student_t")
CLAIM_THRESHOLDS = {
    "min_real_over_control_mean_action": 1.20,
    "min_real_minus_control_mean_action": 2.0,
    "min_real_over_shuffled_hysteresis": 1.20,
    "min_real_minus_disabled_observer_transport": 0.50,
}


def _write_action_summary(
    root: Path,
    *,
    corpus: str,
    kernel: str,
    observer_state_mode: str,
    ablation: str,
    mean_action: float,
    observer_transport_penalty: float,
    hysteresis_penalty: float,
    action_branch: str | None = None,
    null_hysteresis_penalty: float | None = None,
    excess_hysteresis_penalty: float | None = None,
    positive_excess_hysteresis_penalty: float | None = None,
    calibrated_hysteresis_penalty: float | None = None,
    seed: int | None = None,
    basis: str | None = None,
    run_name: str | None = None,
    manifest: dict | None = None,
) -> Path:
    seed_token = f"_seed{seed}" if seed is not None else ""
    basis_token = f"_{basis}" if basis is not None else ""
    run_dir = root / (run_name or f"{corpus}_{kernel}{seed_token}{basis_token}_{ablation}_20260521")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kernel": kernel,
        "observer_state_mode": observer_state_mode,
        "ablation": ablation,
        "basis": basis,
        "n_articles": 8,
        "path_count": 2,
        "reached_count": 2,
        "mean_action": mean_action,
        "median_action": mean_action,
        "max_action": mean_action + 1.0,
        "records": [
            {
                "action": mean_action - 0.5,
                "metric": mean_action / 2.0,
                "observer_transport_penalty": observer_transport_penalty - 0.1,
                "hysteresis_penalty": hysteresis_penalty - 0.05,
                "shear_penalty": 0.25,
                "stress_penalty": 0.5,
                "curvature_penalty": 0.125,
            },
            {
                "action": mean_action + 0.5,
                "metric": mean_action / 2.0,
                "observer_transport_penalty": observer_transport_penalty + 0.1,
                "hysteresis_penalty": hysteresis_penalty + 0.05,
                "shear_penalty": 0.75,
                "stress_penalty": 1.5,
                "curvature_penalty": 0.375,
            },
        ],
    }
    if action_branch is not None:
        payload["action_branch"] = action_branch
    extra_means = {
        "mean_null_hysteresis_penalty": null_hysteresis_penalty,
        "mean_excess_hysteresis_penalty": excess_hysteresis_penalty,
        "mean_positive_excess_hysteresis_penalty": positive_excess_hysteresis_penalty,
        "mean_calibrated_hysteresis_penalty": calibrated_hysteresis_penalty,
    }
    extra_record_keys = {
        "null_hysteresis_penalty": null_hysteresis_penalty,
        "excess_hysteresis_penalty": excess_hysteresis_penalty,
        "positive_excess_hysteresis_penalty": positive_excess_hysteresis_penalty,
        "calibrated_hysteresis_penalty": calibrated_hysteresis_penalty,
    }
    for key, value in extra_means.items():
        if value is not None:
            payload[key] = value
    for record in payload["records"]:
        for key, value in extra_record_keys.items():
            if value is not None:
                record[key] = value
    if seed is not None:
        payload["seed"] = seed
    path = run_dir / "track4_action_summary.json"
    if manifest is not None:
        (run_dir / "track4_action_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_ablation_fixture(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    for kernel in REQUIRED_KERNELS:
        paths.extend(
            [
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel=kernel,
                    observer_state_mode="enabled",
                    ablation="full",
                    mean_action=12.0,
                    observer_transport_penalty=3.0,
                    hysteresis_penalty=1.2,
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="control_random",
                    kernel=kernel,
                    observer_state_mode="enabled",
                    ablation="full",
                    mean_action=6.0,
                    observer_transport_penalty=1.0,
                    hysteresis_penalty=0.4,
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="control_shuffled",
                    kernel=kernel,
                    observer_state_mode="shuffled",
                    ablation="shuffled",
                    mean_action=5.0,
                    observer_transport_penalty=0.8,
                    hysteresis_penalty=0.3,
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel=kernel,
                    observer_state_mode="disabled",
                    ablation="observer_disabled",
                    mean_action=8.0,
                    observer_transport_penalty=0.0,
                    hysteresis_penalty=0.0,
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel=kernel,
                    observer_state_mode="zero_hysteresis",
                    ablation="zero_hysteresis",
                    mean_action=9.0,
                    observer_transport_penalty=2.5,
                    hysteresis_penalty=0.0,
                ),
            ]
        )
    return paths


def _remove_observer_state_fields(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        record.pop("observer_transport_penalty", None)
        record.pop("hysteresis_penalty", None)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_observer_state_ablation_rows_include_action_transport_and_hysteresis_means(tmp_path: Path) -> None:
    payload = summarize_observer_state_ablation(
        _complete_ablation_fixture(tmp_path),
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )

    assert payload["summary_type"] == "track4_observer_state_ablation"
    assert payload["row_count"] == 15
    assert payload["required_kernels"] == list(REQUIRED_KERNELS)
    assert payload["kernels_present"] == list(REQUIRED_KERNELS)

    row = next(
        row
        for row in payload["rows"]
        if row["corpus"] == "real" and row["kernel"] == "rbf" and row["ablation"] == "full"
    )
    assert row["observer_state_mode"] == "enabled"
    assert row["mean_action"] == 12.0
    assert row["mean_observer_transport_penalty"] == 3.0
    assert row["mean_hysteresis_penalty"] == 1.2


def test_observer_state_ablation_reports_real_vs_control_ratios_and_baseline_gaps(tmp_path: Path) -> None:
    payload = summarize_observer_state_ablation(
        _complete_ablation_fixture(tmp_path),
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )

    real_vs_control = payload["comparisons"]["real_vs_control"]
    assert real_vs_control["mean_action"]["real"] == 12.0
    assert real_vs_control["mean_action"]["control"] == 6.0
    assert real_vs_control["mean_action"]["ratio"] == 2.0
    assert real_vs_control["mean_action"]["gap"] == 6.0
    assert real_vs_control["mean_observer_transport_penalty"]["ratio"] == 3.0
    assert real_vs_control["mean_hysteresis_penalty"]["ratio"] == 3.0

    baselines = payload["baseline_comparisons"]
    assert set(baselines) == {"observer_disabled", "zero_hysteresis", "shuffled"}
    assert baselines["observer_disabled"]["mean_action"]["gap_vs_full_real"] == -4.0
    assert baselines["observer_disabled"]["mean_observer_transport_penalty"]["gap_vs_full_real"] == -3.0
    assert baselines["zero_hysteresis"]["mean_hysteresis_penalty"]["gap_vs_full_real"] == -1.2
    assert baselines["shuffled"]["mean_action"]["ratio_full_real_over_baseline"] == 12.0 / 5.0


def test_observer_state_claim_pass_requires_all_kernels_and_threshold_separation(tmp_path: Path) -> None:
    passing_summary = summarize_observer_state_ablation(
        _complete_ablation_fixture(tmp_path),
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )

    passing_claim = evaluate_observer_state_claim(passing_summary, thresholds=CLAIM_THRESHOLDS)
    assert passing_claim["pass"] is True
    assert passing_claim["required_kernels_present"] is True
    assert passing_claim["real_control_separation_pass"] is True
    assert passing_claim["failure_reasons"] == []
    assert passing_summary["safe_for_thesis_claim"] is True


def test_null_calibrated_branch_uses_calibrated_hysteresis_gate(tmp_path: Path) -> None:
    paths: list[Path] = []
    for kernel in REQUIRED_KERNELS:
        paths.extend(
            [
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel=kernel,
                    observer_state_mode="enabled",
                    ablation="full",
                    mean_action=12.0,
                    observer_transport_penalty=3.0,
                    hysteresis_penalty=1.0,
                    calibrated_hysteresis_penalty=2.0,
                    positive_excess_hysteresis_penalty=2.0,
                    action_branch="null_calibrated_hysteresis",
                    run_name=f"null_calibrated_hysteresis_real_{kernel}_seed42_track2_full_20260521",
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="control_random",
                    kernel=kernel,
                    observer_state_mode="enabled",
                    ablation="full",
                    mean_action=6.0,
                    observer_transport_penalty=1.0,
                    hysteresis_penalty=0.4,
                    calibrated_hysteresis_penalty=0.2,
                    positive_excess_hysteresis_penalty=0.2,
                    action_branch="null_calibrated_hysteresis",
                    run_name=f"null_calibrated_hysteresis_control_random_{kernel}_seed42_track2_full_20260521",
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel=kernel,
                    observer_state_mode="shuffled",
                    ablation="shuffled",
                    mean_action=9.0,
                    observer_transport_penalty=2.0,
                    hysteresis_penalty=10.0,
                    calibrated_hysteresis_penalty=0.0,
                    positive_excess_hysteresis_penalty=0.0,
                    action_branch="null_calibrated_hysteresis",
                    run_name=f"null_calibrated_hysteresis_real_{kernel}_seed42_track2_observer_shuffled_20260521",
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel=kernel,
                    observer_state_mode="disabled",
                    ablation="observer_disabled",
                    mean_action=8.0,
                    observer_transport_penalty=0.0,
                    hysteresis_penalty=0.0,
                    calibrated_hysteresis_penalty=0.0,
                    positive_excess_hysteresis_penalty=0.0,
                    action_branch="null_calibrated_hysteresis",
                    run_name=f"null_calibrated_hysteresis_real_{kernel}_seed42_track2_observer_disabled_20260521",
                ),
            ]
        )

    payload = summarize_observer_state_ablation(
        paths,
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )
    claim = evaluate_observer_state_claim(payload, thresholds=CLAIM_THRESHOLDS)

    assert payload["hysteresis_gate_metric"] == "mean_calibrated_hysteresis_penalty"
    assert claim["hysteresis_gate_metric"] == "mean_calibrated_hysteresis_penalty"
    assert payload["baseline_comparisons"]["shuffled"]["mean_hysteresis_penalty"][
        "ratio_full_real_over_baseline"
    ] == 0.1
    assert payload["baseline_comparisons"]["shuffled"]["mean_calibrated_hysteresis_penalty"][
        "baseline"
    ] == 0.0
    assert claim["shuffled_hysteresis_baseline_pass"] is True
    assert claim["pass"] is True


def test_observer_state_claim_fails_when_kernel_missing_or_real_control_gap_too_small(tmp_path: Path) -> None:
    paths = [
        path
        for path in _complete_ablation_fixture(tmp_path)
        if "_student_t_" not in str(path)
    ]
    weak_real = _write_action_summary(
        tmp_path,
        corpus="real",
        kernel="student_t",
        observer_state_mode="enabled",
        ablation="full",
        mean_action=6.5,
        observer_transport_penalty=1.1,
        hysteresis_penalty=0.45,
    )
    weak_control = _write_action_summary(
        tmp_path,
        corpus="control_random",
        kernel="student_t",
        observer_state_mode="enabled",
        ablation="full",
        mean_action=6.0,
        observer_transport_penalty=1.0,
        hysteresis_penalty=0.4,
    )

    missing_kernel_summary = summarize_observer_state_ablation(
        paths,
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )
    weak_gap_summary = summarize_observer_state_ablation(
        [weak_real, weak_control],
        required_kernels=("student_t",),
        thresholds=CLAIM_THRESHOLDS,
    )

    missing_kernel_claim = evaluate_observer_state_claim(missing_kernel_summary, thresholds=CLAIM_THRESHOLDS)
    weak_gap_claim = evaluate_observer_state_claim(weak_gap_summary, thresholds=CLAIM_THRESHOLDS)

    assert missing_kernel_claim["pass"] is False
    assert missing_kernel_claim["required_kernels_present"] is False
    assert "missing_required_kernels" in missing_kernel_claim["failure_reasons"]
    assert weak_gap_claim["pass"] is False
    assert weak_gap_claim["real_control_separation_pass"] is False
    assert "real_control_separation_below_threshold" in weak_gap_claim["failure_reasons"]


def test_observer_state_claim_fails_when_observer_or_hysteresis_contract_is_absent(tmp_path: Path) -> None:
    paths = _complete_ablation_fixture(tmp_path)
    real_full_rbf = next(path for path in paths if "real_rbf_full" in path.parent.name)
    _remove_observer_state_fields(real_full_rbf)

    missing_fields_summary = summarize_observer_state_ablation(
        paths,
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )
    missing_fields_claim = evaluate_observer_state_claim(
        missing_fields_summary,
        thresholds=CLAIM_THRESHOLDS,
    )

    broken_row = next(
        row
        for row in missing_fields_summary["rows"]
        if row["corpus"] == "real" and row["kernel"] == "rbf" and row["ablation"] == "full"
    )
    assert broken_row["mean_observer_transport_penalty"] is None
    assert broken_row["mean_hysteresis_penalty"] is None
    assert broken_row["observer_simplex_contract_supported"] is False
    assert missing_fields_summary["observer_simplex_contract_supported"] is False
    assert missing_fields_claim["pass"] is False
    assert missing_fields_claim["thesis_safe"] is False
    assert "observer_simplex_contract_missing" in missing_fields_claim["failure_reasons"]

    contract_false_summary = summarize_observer_state_ablation(
        _complete_ablation_fixture(tmp_path / "contract_false"),
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )
    contract_false_summary["observer_simplex_contract_supported"] = False
    contract_false_claim = evaluate_observer_state_claim(
        contract_false_summary,
        thresholds=CLAIM_THRESHOLDS,
    )

    assert contract_false_claim["pass"] is False
    assert contract_false_claim["observer_simplex_contract_supported"] is False
    assert "observer_simplex_contract_missing" in contract_false_claim["failure_reasons"]


def test_summary_rows_infer_seed_from_run_name_and_manifest(tmp_path: Path) -> None:
    from_run_name = _write_action_summary(
        tmp_path,
        corpus="real",
        kernel="rbf",
        observer_state_mode="enabled",
        ablation="full",
        mean_action=12.0,
        observer_transport_penalty=3.0,
        hysteresis_penalty=1.2,
        run_name="real_rbf_seed420_track2_full_20260521",
    )
    from_manifest = _write_action_summary(
        tmp_path,
        corpus="control_random",
        kernel="rbf",
        observer_state_mode="enabled",
        ablation="full",
        mean_action=6.0,
        observer_transport_penalty=1.0,
        hysteresis_penalty=0.4,
        run_name="control_random_rbf_integrated_full_20260521",
        manifest={"seed": 4200, "basis": "integrated"},
    )

    payload = summarize_observer_state_ablation(
        [from_run_name, from_manifest],
        required_kernels=("rbf",),
        thresholds=CLAIM_THRESHOLDS,
    )

    by_corpus = {row["corpus"]: row for row in payload["rows"]}
    assert by_corpus["real"]["seed"] == 420
    assert by_corpus["real"]["seed_source"] == "run_name"
    assert by_corpus["control_random"]["seed"] == 4200
    assert by_corpus["control_random"]["seed_source"] == "manifest"
    assert payload["seeds_present"] == [420, 4200]


def test_summary_rows_infer_branch_prefixed_corpus_and_action_branch(tmp_path: Path) -> None:
    path = _write_action_summary(
        tmp_path,
        corpus="real",
        kernel="rbf",
        observer_state_mode="enabled",
        ablation="full",
        mean_action=12.0,
        observer_transport_penalty=3.0,
        hysteresis_penalty=1.2,
        run_name="richer_walker_state_real_rbf_seed42_track2_full_20260521",
    )
    control = _write_action_summary(
        tmp_path,
        corpus="control_random",
        kernel="rbf",
        observer_state_mode="enabled",
        ablation="full",
        mean_action=6.0,
        observer_transport_penalty=1.0,
        hysteresis_penalty=0.4,
        run_name="richer_walker_state_control_random_rbf_seed42_track2_full_20260521",
    )

    payload = summarize_observer_state_ablation(
        [path, control],
        required_kernels=("rbf",),
        thresholds=CLAIM_THRESHOLDS,
    )

    by_corpus = {row["corpus"]: row for row in payload["rows"]}
    assert by_corpus["real"]["action_branch"] == "richer_walker_state"
    assert by_corpus["real"]["basis"] == "track2"
    assert by_corpus["control_random"]["action_branch"] == "richer_walker_state"
    assert payload["aggregates"]["by_action_branch"]["richer_walker_state"]["row_count"] == 2


def test_observer_state_claim_can_require_seed_robustness_gate(tmp_path: Path) -> None:
    paths: list[Path] = []
    for seed in (42, 420, 4200):
        for kernel in REQUIRED_KERNELS:
            paths.extend(
                [
                    _write_action_summary(
                        tmp_path,
                        corpus="real",
                        kernel=kernel,
                        observer_state_mode="enabled",
                        ablation="full",
                        mean_action=12.0,
                        observer_transport_penalty=3.0,
                        hysteresis_penalty=1.2,
                        seed=seed,
                        basis="track2",
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="control_random",
                        kernel=kernel,
                        observer_state_mode="enabled",
                        ablation="full",
                        mean_action=6.0,
                        observer_transport_penalty=1.0,
                        hysteresis_penalty=0.4,
                        seed=seed,
                        basis="track2",
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="control_shuffled",
                        kernel=kernel,
                        observer_state_mode="shuffled",
                        ablation="shuffled",
                        mean_action=5.0,
                        observer_transport_penalty=0.8,
                        hysteresis_penalty=0.3,
                        seed=seed,
                        basis="track2",
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="real",
                        kernel=kernel,
                        observer_state_mode="disabled",
                        ablation="observer_disabled",
                        mean_action=8.0,
                        observer_transport_penalty=0.0,
                        hysteresis_penalty=0.0,
                        seed=seed,
                        basis="track2",
                    ),
                ]
            )

    summary = summarize_observer_state_ablation(
        paths,
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )
    claim = evaluate_observer_state_claim(
        summary,
        thresholds={**CLAIM_THRESHOLDS, "required_seeds": [42, 420, 4200]},
    )

    assert claim["pass"] is True
    assert claim["required_seeds"] == [42, 420, 4200]
    assert claim["seeds_present"] == [42, 420, 4200]
    assert claim["required_seeds_present"] is True
    assert claim["seed_robustness_pass"] is True


def test_one_seed_summary_fails_multi_seed_robustness_gate(tmp_path: Path) -> None:
    summary = summarize_observer_state_ablation(
        _complete_ablation_fixture(tmp_path),
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )
    claim = evaluate_observer_state_claim(
        summary,
        thresholds={**CLAIM_THRESHOLDS, "required_seeds": [42, 420, 4200]},
    )

    assert claim["pass"] is False
    assert claim["required_seeds_present"] is False
    assert claim["missing_required_seeds"] == [420, 4200]
    assert "missing_required_seeds" in claim["failure_reasons"]
    assert summary["safe_for_multi_seed_robustness_claim"] is False


def test_observer_state_summary_reports_per_basis_and_per_kernel_aggregates(tmp_path: Path) -> None:
    paths: list[Path] = []
    for basis, real_mean in (("track2", 12.0), ("integrated", 4.0)):
        for kernel in REQUIRED_KERNELS:
            paths.extend(
                [
                    _write_action_summary(
                        tmp_path,
                        corpus="real",
                        kernel=kernel,
                        observer_state_mode="enabled",
                        ablation="full",
                        mean_action=real_mean,
                        observer_transport_penalty=3.0,
                        hysteresis_penalty=1.2,
                        seed=42,
                        basis=basis,
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="control_random",
                        kernel=kernel,
                        observer_state_mode="enabled",
                        ablation="full",
                        mean_action=6.0,
                        observer_transport_penalty=1.0,
                        hysteresis_penalty=0.4,
                        seed=42,
                        basis=basis,
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="control_shuffled",
                        kernel=kernel,
                        observer_state_mode="shuffled",
                        ablation="shuffled",
                        mean_action=5.0,
                        observer_transport_penalty=0.8,
                        hysteresis_penalty=0.3,
                        seed=42,
                        basis=basis,
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="real",
                        kernel=kernel,
                        observer_state_mode="disabled",
                        ablation="observer_disabled",
                        mean_action=8.0,
                        observer_transport_penalty=0.0,
                        hysteresis_penalty=0.0,
                        seed=42,
                        basis=basis,
                    ),
                ]
            )

    payload = summarize_observer_state_ablation(
        paths,
        required_kernels=REQUIRED_KERNELS,
        thresholds=CLAIM_THRESHOLDS,
    )

    aggregates = payload["aggregates"]
    assert set(aggregates) >= {"by_basis", "by_kernel"}
    assert aggregates["by_basis"]["track2"]["status"] == "pass"
    assert aggregates["by_basis"]["integrated"]["status"] == "fail"
    assert "real_control_separation_below_threshold" in aggregates["by_basis"]["integrated"]["failure_reasons"]
    assert set(aggregates["by_kernel"]) == set(REQUIRED_KERNELS)
    assert all(
        "status" in kernel_payload and "pass" in kernel_payload
        for kernel_payload in aggregates["by_kernel"].values()
    )


def test_observer_state_claim_fails_when_basis_seed_cell_is_hidden_by_pooling(tmp_path: Path) -> None:
    paths: list[Path] = []
    for basis in ("track2", "integrated"):
        for seed in (42, 420, 4200):
            real_mean = 6.5 if basis == "integrated" and seed == 420 else 12.0
            paths.extend(
                [
                    _write_action_summary(
                        tmp_path,
                        corpus="real",
                        kernel="rbf",
                        observer_state_mode="enabled",
                        ablation="full",
                        mean_action=real_mean,
                        observer_transport_penalty=3.0,
                        hysteresis_penalty=1.2,
                        seed=seed,
                        basis=basis,
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="control_random",
                        kernel="rbf",
                        observer_state_mode="enabled",
                        ablation="full",
                        mean_action=6.0,
                        observer_transport_penalty=1.0,
                        hysteresis_penalty=0.4,
                        seed=seed,
                        basis=basis,
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="control_shuffled",
                        kernel="rbf",
                        observer_state_mode="shuffled",
                        ablation="shuffled",
                        mean_action=5.0,
                        observer_transport_penalty=0.8,
                        hysteresis_penalty=0.3,
                        seed=seed,
                        basis=basis,
                    ),
                    _write_action_summary(
                        tmp_path,
                        corpus="real",
                        kernel="rbf",
                        observer_state_mode="disabled",
                        ablation="observer_disabled",
                        mean_action=8.0,
                        observer_transport_penalty=0.0,
                        hysteresis_penalty=0.0,
                        seed=seed,
                        basis=basis,
                    ),
                ]
            )

    payload = summarize_observer_state_ablation(
        paths,
        required_kernels=("rbf",),
        required_seeds=(42, 420, 4200),
        required_bases=("track2", "integrated"),
        thresholds=CLAIM_THRESHOLDS,
    )
    claim = evaluate_observer_state_claim(payload, thresholds=CLAIM_THRESHOLDS)

    assert payload["aggregates"]["by_seed"]["420"]["status"] == "pass"
    assert payload["aggregates"]["by_basis_seed"]["integrated|seed420"]["status"] == "fail"
    assert payload["basis_seed_robustness_pass"] is False
    assert payload["kernel_seed_basis_robustness_pass"] is False
    assert payload["action_only_robustness_pass"] is False
    assert payload["action_basis_seed_robustness_pass"] is False
    assert claim["pass"] is False
    assert claim["action_only_robustness_pass"] is False
    assert claim["hysteresis_mechanism_pass"] is True
    assert "basis_seed_robustness_failed" in claim["failure_reasons"]


def test_observer_state_summary_separates_action_robustness_from_hysteresis_gate(tmp_path: Path) -> None:
    paths: list[Path] = []
    for seed in (42, 420, 4200):
        paths.extend(
            [
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel="rbf",
                    observer_state_mode="enabled",
                    ablation="full",
                    mean_action=18.0,
                    observer_transport_penalty=3.0,
                    hysteresis_penalty=0.2,
                    seed=seed,
                    basis="track2",
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="control_random",
                    kernel="rbf",
                    observer_state_mode="enabled",
                    ablation="full",
                    mean_action=6.0,
                    observer_transport_penalty=1.0,
                    hysteresis_penalty=0.1,
                    seed=seed,
                    basis="track2",
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="control_shuffled",
                    kernel="rbf",
                    observer_state_mode="shuffled",
                    ablation="shuffled",
                    mean_action=7.0,
                    observer_transport_penalty=0.8,
                    hysteresis_penalty=2.0,
                    seed=seed,
                    basis="track2",
                ),
                _write_action_summary(
                    tmp_path,
                    corpus="real",
                    kernel="rbf",
                    observer_state_mode="disabled",
                    ablation="observer_disabled",
                    mean_action=8.0,
                    observer_transport_penalty=0.0,
                    hysteresis_penalty=0.0,
                    seed=seed,
                    basis="track2",
                ),
            ]
        )

    payload = summarize_observer_state_ablation(
        paths,
        required_kernels=("rbf",),
        required_seeds=(42, 420, 4200),
        required_bases=("track2",),
        thresholds=CLAIM_THRESHOLDS,
    )
    claim = evaluate_observer_state_claim(payload, thresholds=CLAIM_THRESHOLDS)

    assert payload["action_only_robustness_pass"] is True
    assert payload["action_kernel_seed_basis_robustness_pass"] is True
    assert payload["kernel_seed_basis_robustness_pass"] is False
    assert claim["action_only_robustness_pass"] is True
    assert claim["hysteresis_mechanism_pass"] is False
    assert "kernel_seed_basis_robustness_failed" in claim["failure_reasons"]
