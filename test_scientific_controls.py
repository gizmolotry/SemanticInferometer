import json
from pathlib import Path

from analysis.verification.thesis_evidence import (
    CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN,
    build_thesis_evidence,
    _variance_separation_stats,
)
from thesis_test_support import build_canonical_fixture, write_control_metrics


def test_variance_separation_abs_log_is_direction_free():
    compression = _variance_separation_stats(0.70, 1.00)
    expansion = _variance_separation_stats(1.30, 1.00)
    parity = _variance_separation_stats(1.00, 1.00)

    assert compression["passes_separation_threshold"]
    assert compression["direction"] == "real_lt_control"
    assert expansion["passes_separation_threshold"]
    assert expansion["direction"] == "real_gt_control"
    assert not parity["passes_separation_threshold"]
    assert parity["direction"] == "near_parity"
    assert CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN > 0


def test_control_destruction_claim_fails_when_real_does_not_separate(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    weak_real = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    write_control_metrics(weak_real, procrustes_ratio=0.98, separates_count=1, distance_corr_ratio=1.01)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "control_destruction"
    )
    assert not claim["thesis_safe"]
    procrustes_claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "procrustes_control_separation"
    )
    assert not procrustes_claim["thesis_safe"]
    summary = payloads["scientific_validation_summary"]["control_destruction"]
    assert summary["pass_rate"] < 1.0


def test_control_destruction_claim_fails_when_distance_correlation_is_too_high(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    weak_real = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    write_control_metrics(weak_real, procrustes_ratio=1.4, separates_count=3, distance_corr_ratio=1.02)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "control_destruction"
    )
    assert not claim["thesis_safe"]
    procrustes_claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "procrustes_control_separation"
    )
    assert procrustes_claim["thesis_safe"]


def test_procrustes_summary_preserves_per_control_rows(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    summary = payloads["scientific_validation_summary"]["procrustes_control_separation"]
    assert summary["mean_min_per_control_ratio"] >= 1.05
    assert summary["per_control_rows"]
    controls = {row["control"] for row in summary["per_control_rows"]}
    assert {"Constant", "Shuffled", "Random"}.issubset(controls)


def test_procrustes_claim_fails_when_any_per_control_ratio_is_weak(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    real_metrics = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    blob = json.loads(real_metrics.read_text(encoding="utf-8"))
    blob["metrics"]["procrustes_ratio"] = 1.6
    blob["metrics"]["procrustes_min_control_ratio"] = 0.92
    blob["procrustes_real_vs_controls"]["min_ratio_real_over_control"] = 0.92
    blob["procrustes_real_vs_controls"]["rows"][1]["ratio_real_over_control"] = 0.92
    real_metrics.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    procrustes_claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "procrustes_control_separation"
    )
    assert not procrustes_claim["thesis_safe"]
    summary = payloads["scientific_validation_summary"]["procrustes_control_separation"]
    record = next(row for row in summary["records"] if row["kernel"] == "rbf" and row["channel"] == "cls")
    assert record["procrustes_ratio"] == 1.6
    assert record["procrustes_min_control_ratio"] == 0.92


def test_stochastic_control_variance_claim_passes_against_noise_controls(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "stochastic_control_variance_compression"
    )
    assert claim["thesis_safe"]
    summary = payloads["scientific_validation_summary"]["stochastic_control_variance_compression"]
    assert summary["mean_real_over_stochastic_variance_ratio"] < 0.95


def test_stochastic_control_variance_claim_fails_when_real_is_noisier(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    weak_real = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    write_control_metrics(
        weak_real,
        simple_variance_real=1.2,
        simple_variance_shuffled=1.0,
        simple_variance_random=1.05,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "stochastic_control_variance_compression"
    )
    assert not claim["thesis_safe"]


def test_variance_separation_claim_can_pass_when_compression_claim_fails(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, kernels=["rbf", "matern", "imq"], channels=["cls"])
    for metrics_path in fixture["runs_dir"].glob("experiments_*/rbf/cls/real/control_metrics.json"):
        write_control_metrics(
            metrics_path,
            simple_variance_real=1.30,
            simple_variance_shuffled=1.0,
            simple_variance_random=1.0,
        )
    for metrics_path in fixture["runs_dir"].glob("experiments_*/matern/cls/real/control_metrics.json"):
        write_control_metrics(
            metrics_path,
            simple_variance_real=1.30,
            simple_variance_shuffled=1.0,
            simple_variance_random=1.0,
        )
    for metrics_path in fixture["runs_dir"].glob("experiments_*/imq/cls/real/control_metrics.json"):
        write_control_metrics(
            metrics_path,
            simple_variance_real=1.30,
            simple_variance_shuffled=1.0,
            simple_variance_random=1.0,
        )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {claim["claim_id"]: claim for claim in payloads["claim_matrix"]["claims"]}
    assert not claims["stochastic_control_variance_compression"]["thesis_safe"]
    assert claims["stochastic_control_variance_separation"]["thesis_safe"]
    assert payloads["variance_separation_summary"]["primary_pass"]


def test_variance_cartography_reports_direct_and_comprehensive_bases_separately(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, kernels=["rbf"], channels=["cls"])
    for metrics_path in fixture["runs_dir"].glob("experiments_*/rbf/cls/real/control_metrics.json"):
        write_control_metrics(
            metrics_path,
            simple_variance_real=1.0,
            simple_variance_shuffled=1.0,
            simple_variance_random=1.0,
        )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    rows = payloads["metric_signal_cartography"]["records"]
    bases = {row["basis"] for row in rows}
    assert {"direct_observer_payload", "comprehensive_results"}.issubset(bases)
    summary = payloads["variance_separation_summary"]
    assert summary["primary_pass"]
    assert not summary["direct_sensitivity_pass"]
    assert (summary["mean_direct_abs_log_ratio"] or 0.0) < CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN
    assert (summary["mean_primary_abs_log_ratio"] or 0.0) >= CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN


def test_variance_separation_summary_reports_required_kernels_and_controls(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, kernels=["rbf", "matern", "imq"], channels=["cls"])

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    summary = payloads["variance_separation_summary"]
    assert {"rbf", "matern", "imq"}.issubset(set(summary["required_kernels_evaluated"]))
    assert all(summary["per_kernel"][kernel]["passes"] for kernel in ("rbf", "matern", "imq"))

    control_families = {
        row["control_family"]
        for row in payloads["metric_signal_cartography"]["records"]
        if row["metric"] == "simple_variance"
    }
    assert {"Constant", "Shuffled", "Random", "stochastic_controls"}.issubset(control_families)


def test_control_metric_provenance_surfaces_direct_vs_comprehensive_disagreement(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    real_metrics = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    blob = json.loads(real_metrics.read_text(encoding="utf-8"))
    blob["metric_basis"] = "direct_observer_payload"
    blob["metrics"]["simple_variance_stochastic_ratio"] = 1.01
    blob["alternate_sources"] = {
        "comprehensive_results": {
            "metric_basis": "comprehensive_results",
            "metrics": {"simple_variance_stochastic_ratio": 0.86},
        }
    }
    real_metrics.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    provenance = payloads["scientific_validation_summary"]["control_metric_provenance"]
    assert provenance["basis_mismatch_count"] == 1
    record = next(row for row in provenance["records"] if row["kernel"] == "rbf" and row["channel"] == "cls")
    assert record["primary_simple_variance_stochastic_ratio"] == 1.01
    assert record["alternate_comprehensive_simple_variance_stochastic_ratio"] == 0.86


def test_control_metric_provenance_surfaces_comprehensive_primary_vs_direct_alternate(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    real_metrics = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    blob = json.loads(real_metrics.read_text(encoding="utf-8"))
    blob["metric_basis"] = "comprehensive_results"
    blob["requested_metric_basis"] = "comprehensive_results"
    blob["metrics"]["simple_variance_stochastic_ratio"] = 0.86
    blob["alternate_sources"] = {
        "direct_observer_payload": {
            "metric_basis": "direct_observer_payload",
            "metrics": {"simple_variance_stochastic_ratio": 1.01},
        }
    }
    real_metrics.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    provenance = payloads["scientific_validation_summary"]["control_metric_provenance"]
    assert provenance["basis_mismatch_count"] == 1
    record = next(row for row in provenance["records"] if row["kernel"] == "rbf" and row["channel"] == "cls")
    assert record["primary_simple_variance_stochastic_ratio"] == 0.86
    assert record["alternate_direct_simple_variance_stochastic_ratio"] == 1.01


def test_control_metric_basis_reads_requested_snapshot_over_default_payload(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    real_dir = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
    )
    default_metrics = real_dir / "control_metrics.json"
    comprehensive_metrics = real_dir / "control_metrics.comprehensive_results.json"

    write_control_metrics(
        default_metrics,
        simple_variance_real=1.01,
        simple_variance_shuffled=1.0,
        simple_variance_random=1.0,
    )
    write_control_metrics(
        comprehensive_metrics,
        simple_variance_real=0.86,
        simple_variance_shuffled=1.0,
        simple_variance_random=1.0,
    )
    comprehensive_blob = json.loads(comprehensive_metrics.read_text(encoding="utf-8"))
    comprehensive_blob["metric_basis"] = "comprehensive_results"
    comprehensive_blob["alternate_sources"] = {
        "direct_observer_payload": {
            "metric_basis": "direct_observer_payload",
            "metrics": {"simple_variance_stochastic_ratio": 1.01},
        }
    }
    comprehensive_metrics.write_text(json.dumps(comprehensive_blob, indent=2), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
        control_metric_basis="comprehensive",
    )

    summary = payloads["scientific_validation_summary"]
    assert summary["selected_control_metric_basis"] == "comprehensive_results"
    provenance = summary["control_metric_provenance"]
    record = next(row for row in provenance["records"] if row["kernel"] == "rbf" and row["channel"] == "cls")
    assert record["primary_simple_variance_stochastic_ratio"] == 0.86
    assert record["evidence_requested_metric_basis"] == "comprehensive_results"
    assert record["evidence_control_metric_basis_path"].endswith(
        "control_metrics.comprehensive_results.json"
    )


def test_missing_claim_artifact_counts_as_negative_evidence(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    missing_metrics = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    missing_metrics.unlink()

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {
        claim["claim_id"]: claim
        for claim in payloads["claim_matrix"]["claims"]
    }
    for claim_id in (
        "control_destruction",
        "procrustes_control_separation",
        "stochastic_control_variance_compression",
    ):
        assert not claims[claim_id]["thesis_safe"]
        assert claims[claim_id]["missing_evidence"]

    ledger = payloads["scientific_validation_summary"]["evidence_ledger"]
    assert ledger["missing_by_claim"]["control_destruction"] == 1
    assert any(
        row["failure_type"] == "missing_control_metrics"
        for row in payloads["scientific_validation_summary"]["failure_modes"]["records"]
    )
