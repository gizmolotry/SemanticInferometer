import json
from pathlib import Path

from analysis.verification.thesis_evidence import build_thesis_evidence, write_thesis_evidence
from thesis_test_support import build_canonical_fixture


def test_thesis_claim_matrix_emits_thesis_safe_claims(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim_matrix = payloads["claim_matrix"]
    claim_ids = {claim["claim_id"] for claim in claim_matrix["claims"]}
    assert {
        "control_destruction",
        "procrustes_control_separation",
        "stochastic_control_variance_compression",
        "stochastic_control_variance_separation",
        "synthetic_recoverability",
        "observer_relativity",
        "track4_traversal_validity",
        "track4_work_barrier_signal",
        "track5_ablation_coverage",
        "verification_provenance",
        "procrustes_verification_provenance",
        "canonical_freeze",
    }.issubset(claim_ids)
    assert all(bool(claim["thesis_safe"]) for claim in claim_matrix["claims"])
    interpretation = payloads["scientific_validation_summary"]["semantic_signal_interpretation"]
    assert interpretation["status"] == "real_control_supported"
    assert interpretation["semantic_signal_detected"] is True
    assert interpretation["publishable_real_control_signal"] is True


def test_write_thesis_evidence_writes_expected_summary_files(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    out_dir = tmp_path / "outputs" / "thesis_validation"

    written = write_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
        out_dir=out_dir,
    )

    assert set(written.keys()) == {
        "scientific_validation_summary",
        "claim_matrix",
        "ablation_matrix",
        "observer_relativity_summary",
        "track4_traversal_summary",
        "metric_signal_cartography",
        "variance_separation_summary",
        "unsafe_claim_strategy",
    }
    assert all(path.exists() for path in written.values())


def test_thesis_evidence_can_focus_on_explicit_run_allowlist(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    allowlist = tmp_path / "focused_run_ids.txt"
    allowlist.write_text("experiments_20260504_010101\nexperiments_20260504_020202\n", encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
        run_id_allowlist_path=allowlist,
    )

    selection = payloads["scientific_validation_summary"]["input_selection"]
    assert selection["focused_filter_active"] is True
    assert selection["selected_run_ids"] == ["experiments_20260504_010101", "experiments_20260504_020202"]
    assert selection["missing_requested_run_ids"] == []


def test_semantic_signal_interpretation_keeps_mixed_positive_distinct_from_no_signal(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    weak_control = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "control_metrics.json"
    )
    from thesis_test_support import write_control_metrics

    write_control_metrics(
        weak_control,
        procrustes_ratio=0.98,
        separates_count=1,
        distance_corr_ratio=1.02,
        simple_variance_real=1.1,
        simple_variance_shuffled=1.0,
        simple_variance_random=1.0,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    interpretation = payloads["scientific_validation_summary"]["semantic_signal_interpretation"]
    assert interpretation["status"] == "mixed_positive_not_real_control_safe"
    assert interpretation["semantic_signal_detected"] is True
    assert interpretation["publishable_real_control_signal"] is False
    assert "control_destruction" in interpretation["real_control_blocker_claim_ids"]


def test_unverified_leaf_blocks_publishable_signal_claim(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    from thesis_test_support import write_verification_report

    weak_verification = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "verification_report.json"
    )
    write_verification_report(weak_verification, global_pass=False)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {
        claim["claim_id"]: claim
        for claim in payloads["claim_matrix"]["claims"]
    }
    assert not claims["verification_provenance"]["thesis_safe"]
    interpretation = payloads["scientific_validation_summary"]["semantic_signal_interpretation"]
    assert interpretation["publishable_real_control_signal"] is False
    assert "verification_provenance" in interpretation["real_control_blocker_claim_ids"]
    strategy = payloads["unsafe_claim_strategy"]
    recommendation = next(row for row in strategy["recommendations"] if row["claim_id"] == "verification_provenance")
    assert recommendation["action"] == "fix"
    assert strategy["publishable_real_control_signal"] is False


def test_procrustes_profile_provenance_ignores_legacy_control_ordering_failure(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    path = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "verification_report.json"
    )
    path.write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "timestamp": "2026-05-04T00:00:00+00:00",
                "layers": [
                    {
                        "layer_id": "rbf/cls",
                        "layer_name": "cls",
                        "status": "UNVERIFIED",
                        "checks": [
                            {"name": "crn_locked", "pass": True},
                            {"name": "seed_stability", "pass": True, "value": 0.12},
                            {"name": "control_ordering", "pass": False},
                            {"name": "alpha_sweep_sanity", "pass": None},
                            {"name": "mi_score", "pass": True, "value": 0.9},
                        ],
                        "fail_reasons": ["Ordering failed under broad verifier"],
                    }
                ],
                "global_pass": False,
                "verification_status": "UNVERIFIED",
                "dataset_hash": "dataset_fixture_hash",
                "code_hash_or_commit": "code_fixture_hash",
                "weights_hash": "weights_fixture_hash",
                "kernel_params": {"kernel": "rbf", "channel": "cls"},
                "crn_seed": 12345,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {
        claim["claim_id"]: claim
        for claim in payloads["claim_matrix"]["claims"]
    }
    assert not claims["verification_provenance"]["thesis_safe"]
    assert claims["procrustes_verification_provenance"]["thesis_safe"]
    summary = payloads["scientific_validation_summary"]["procrustes_verification_provenance"]
    assert summary["unverified_count"] == 0


def test_failed_verification_report_is_not_misclassified_as_missing_evidence(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    path = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "verification_report.json"
    )
    path.write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "timestamp": "2026-05-04T00:00:00+00:00",
                "layers": [
                    {
                        "layer_id": "rbf/cls",
                        "layer_name": "cls",
                        "status": "UNVERIFIED",
                        "checks": [
                            {"name": "crn_locked", "pass": True},
                            {"name": "seed_stability", "pass": True, "value": 0.12},
                            {"name": "control_ordering", "pass": False},
                        ],
                        "fail_reasons": ["Ordering failed under broad verifier"],
                    }
                ],
                "global_pass": False,
                "verification_status": "UNVERIFIED",
                "dataset_hash": "dataset_fixture_hash",
                "code_hash_or_commit": "code_fixture_hash",
                "weights_hash": "weights_fixture_hash",
                "kernel_params": {"kernel": "rbf", "channel": "cls"},
                "crn_seed": 12345,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {
        claim["claim_id"]: claim
        for claim in payloads["claim_matrix"]["claims"]
    }
    assert not claims["verification_provenance"]["thesis_safe"]
    assert claims["verification_provenance"]["missing_evidence"] == []

    summary = payloads["scientific_validation_summary"]["verification_provenance"]
    record = next(row for row in summary["records"] if row["kernel"] == "rbf" and row["channel"] == "cls")
    assert "control_ordering" in record["failed_check_names"]
    assert "Ordering failed under broad verifier" in record["fail_reasons"]
