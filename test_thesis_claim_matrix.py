import json
import os
from pathlib import Path

from analysis.verification.thesis_evidence import build_thesis_evidence, write_thesis_evidence
from thesis_test_support import (
    build_canonical_fixture,
    write_csv,
    write_hidden_groups,
    write_track4_observer_state_action_summary,
)


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
        "terrain_incremental_signal",
        "observer_relativity",
        "track4_traversal_validity",
        "track4_work_barrier_signal",
        "track4_observer_state_action_only_separation",
        "track4_observer_state_action_separation",
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
    assert interpretation["paper_profile_ready"] is True
    assert interpretation["paper_profile_status"] == "focused_core_profile_supported"
    assert "full thesis/canonical claims may still be unsafe" in interpretation["paper_profile_scope_warning"]


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
        "terrain_incremental_signal_summary",
        "track4_observer_state_action_summary",
        "metric_signal_cartography",
        "variance_separation_summary",
        "kernel_signal_summary",
        "paper_claim_profile",
        "unsafe_claim_strategy",
        "paper_claim_profile_csv",
        "paper_metric_signal_cartography_csv",
        "paper_basis_comparison_csv",
    }
    assert all(path.exists() for path in written.values())
    assert written["paper_claim_profile_csv"].read_text(encoding="utf-8").startswith("claim_id,")
    assert "comprehensive_results" in written["paper_basis_comparison_csv"].read_text(encoding="utf-8")


def test_paper_claim_profile_only_promotes_supported_core_claims(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    from thesis_test_support import write_relativity_deltas

    broken_relativity = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "relativity_deltas.json"
    )
    write_relativity_deltas(
        broken_relativity,
        mean_coord_delta=0.0,
        max_coord_delta=0.0,
        rotation_deg=0.0,
        path_flip_count=0,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    profile = payloads["paper_claim_profile"]
    assert profile["publication_ready"] is False
    core_claim_ids = {row["claim_id"] for row in profile["core_claims"]}
    blocked_claim_ids = {row["claim_id"] for row in profile["blocked_core_claims"]}
    assert "observer_relativity" not in core_claim_ids
    assert "observer_relativity" in blocked_claim_ids
    assert "track4_traversal_validity" not in core_claim_ids
    assert "stochastic_control_variance_compression" not in core_claim_ids


def test_soft_terrain_packet_diagnostic_is_not_promoted_to_claim_matrix(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim_ids = {claim["claim_id"] for claim in payloads["claim_matrix"]["claims"]}
    core_claim_ids = {row["claim_id"] for row in payloads["paper_claim_profile"]["core_claims"]}
    blocked_claim_ids = {
        row["claim_id"] for row in payloads["paper_claim_profile"]["blocked_core_claims"]
    }
    interpretation = payloads["scientific_validation_summary"]["semantic_signal_interpretation"]

    assert "track4_soft_terrain_matched_specificity" not in claim_ids
    assert "track4_soft_terrain_matched_specificity" not in core_claim_ids
    assert "track4_soft_terrain_matched_specificity" not in blocked_claim_ids
    assert interpretation["publishable_real_control_signal"] is True


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
    assert interpretation["paper_profile_ready"] is False
    assert "verification_provenance" in interpretation["paper_blocked_core_claim_ids"]
    assert "verification_provenance" in interpretation["real_control_blocker_claim_ids"]
    strategy = payloads["unsafe_claim_strategy"]
    recommendation = next(row for row in strategy["recommendations"] if row["claim_id"] == "verification_provenance")
    assert recommendation["action"] == "fix"
    assert strategy["publishable_real_control_signal"] is False


def test_unsafe_strategy_does_not_describe_failed_terrain_signal_as_supported(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, kernels=["rbf"], channels=["cls"], seeds=[42])
    synthetic_manifest = fixture["synthetic_manifest"]
    assert synthetic_manifest is not None
    run_dir = synthetic_manifest.parent / "synthetic" / "rbf_seed42"
    rows = [
        {"index": 0, "zone": "Bridge", "w_actual": 10.0, "density": 0.9, "stress": 0.1, "x": 0.0, "y": 0.0},
        {"index": 1, "zone": "Bridge", "w_actual": 11.0, "density": 0.8, "stress": 0.1, "x": 1.0, "y": 0.0},
        {"index": 2, "zone": "Void", "w_actual": 12.0, "density": 0.1, "stress": 0.9, "x": 2.0, "y": 0.0},
        {"index": 3, "zone": "Void", "w_actual": 13.0, "density": 0.1, "stress": 0.8, "x": 3.0, "y": 0.0},
    ]
    write_csv(
        run_dir / "MONOLITH_DATA.csv",
        rows,
        fieldnames=["index", "zone", "w_actual", "density", "stress", "x", "y"],
    )
    write_hidden_groups(
        run_dir / "labels" / "hidden_groups.csv",
        {0: "Bridge", 1: "Bridge", 2: "Void", 3: "Void"},
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {claim["claim_id"]: claim for claim in payloads["claim_matrix"]["claims"]}
    assert claims["terrain_incremental_signal"]["thesis_safe"] is False
    recommendation = next(
        row
        for row in payloads["unsafe_claim_strategy"]["recommendations"]
        if row["claim_id"] == "terrain_incremental_signal"
    )
    assert recommendation["action"] == "hold_out"
    assert "supports the claim" not in recommendation["reason"]


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


def test_track4_observer_state_claim_is_present_but_unsafe_when_summary_artifact_missing(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    observer_graph_dir = fixture["root"] / "outputs" / "track4_action_graph"
    for path in observer_graph_dir.glob("**/track4_observer_state_ablation_summary.json"):
        path.unlink()

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {
        claim["claim_id"]: claim
        for claim in payloads["claim_matrix"]["claims"]
    }
    claim = claims["track4_observer_state_action_separation"]
    assert claim["pass"] is False
    assert claim["thesis_safe"] is False
    assert claim["missing_evidence"] == [
        {
            "claim_id": "track4_observer_state_action_separation",
            "artifact_family": "track4_observer_state_ablation_summary.json",
            "artifact_path": None,
            "present": False,
            "detail": "missing",
        }
    ]

    summary = payloads["track4_observer_state_action_summary"]
    assert summary["status"] == "NO_DATA"
    assert summary["claim_evaluation"]["pass"] is False


def test_track4_observer_state_summary_all_is_preferred_over_per_basis_summaries(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    graph_root = fixture["root"] / "outputs" / "track4_action_graph"
    per_basis = graph_root / "matern_basis" / "track4_observer_state_ablation_summary.json"
    summary_all = graph_root / "summary_all" / "track4_observer_state_ablation_summary.json"

    write_track4_observer_state_action_summary(per_basis)
    per_basis_payload = json.loads(per_basis.read_text(encoding="utf-8"))
    per_basis_payload["claim_evaluation"]["point_estimate"] = 999.0
    per_basis_payload["rows"][0]["mean_action"] = 999.0
    per_basis.write_text(json.dumps(per_basis_payload), encoding="utf-8")

    write_track4_observer_state_action_summary(summary_all)
    summary_all_payload = json.loads(summary_all.read_text(encoding="utf-8"))
    summary_all_payload["claim_evaluation"]["point_estimate"] = 2.5
    summary_all_payload["rows"][0]["mean_action"] = 14.0
    summary_all.write_text(json.dumps(summary_all_payload), encoding="utf-8")
    older = per_basis.stat().st_mtime - 10
    os.utime(summary_all, (older, older))
    newer = per_basis.stat().st_mtime + 10
    os.utime(per_basis, (newer, newer))

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    summary = payloads["track4_observer_state_action_summary"]
    claim = next(
        claim
        for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_observer_state_action_separation"
    )
    assert "summary_all" in Path(summary["source_path"]).parts
    assert summary["claim_evaluation"]["point_estimate"] == 2.5
    assert claim["point_estimate"] == 2.5
    assert summary["rows"][0]["mean_action"] == 14.0


def test_track4_observer_state_prefers_safe_scoped_basis_when_summary_all_fails(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    graph_root = fixture["root"] / "outputs" / "track4_action_graph"
    for path in graph_root.glob("**/track4_observer_state_ablation_summary.json"):
        path.unlink()
    summary_track2 = graph_root / "summary_track2" / "track4_observer_state_ablation_summary.json"
    summary_all = graph_root / "summary_all" / "track4_observer_state_ablation_summary.json"

    write_track4_observer_state_action_summary(summary_track2)
    track2_payload = json.loads(summary_track2.read_text(encoding="utf-8"))
    track2_payload["required_seeds"] = [42, 420, 4200]
    track2_payload["required_bases"] = ["track2"]
    track2_payload["claim_evaluation"]["required_seeds"] = [42, 420, 4200]
    track2_payload["claim_evaluation"]["required_bases"] = ["track2"]
    track2_payload["claim_evaluation"]["point_estimate"] = 5.25
    summary_track2.write_text(json.dumps(track2_payload), encoding="utf-8")

    write_track4_observer_state_action_summary(summary_all)
    all_payload = json.loads(summary_all.read_text(encoding="utf-8"))
    all_payload["required_seeds"] = [42, 420, 4200]
    all_payload["required_bases"] = ["track2", "integrated"]
    all_payload["safe_for_thesis_claim"] = False
    all_payload["pass"] = False
    all_payload["thesis_safe"] = False
    all_payload["claim_evaluation"]["safe_for_thesis_claim"] = False
    all_payload["claim_evaluation"]["pass"] = False
    all_payload["claim_evaluation"]["thesis_safe"] = False
    all_payload["claim_evaluation"]["failure_reasons"] = ["basis_seed_robustness_failed"]
    all_payload["claim_evaluation"]["point_estimate"] = 3.4
    summary_all.write_text(json.dumps(all_payload), encoding="utf-8")
    stale_summary = graph_root / "older_seed42" / "summary_track2" / "track4_observer_state_ablation_summary.json"
    write_track4_observer_state_action_summary(stale_summary)
    stale_payload = json.loads(stale_summary.read_text(encoding="utf-8"))
    stale_payload["claim_evaluation"]["point_estimate"] = 999.0
    stale_summary.write_text(json.dumps(stale_payload), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    summary = payloads["track4_observer_state_action_summary"]
    claim = next(
        claim
        for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "track4_observer_state_action_separation"
    )
    assert "summary_track2" in Path(summary["source_path"]).parts
    assert "older_seed42" not in Path(summary["source_path"]).parts
    assert claim["pass"] is True
    assert claim["thesis_safe"] is True
    scoped = claim["scoped_supported_findings"]
    assert scoped
    assert scoped[0]["scope_values"] == ["track2"]
    assert scoped[0]["point_estimate"] == 5.25
    assert len(scoped) >= 1
    assert "summary_track2" in Path(scoped[0]["source_path"]).parts


def test_track4_observer_state_explicit_summary_override_is_respected(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    graph_root = fixture["root"] / "outputs" / "track4_action_graph"
    for path in graph_root.glob("**/track4_observer_state_ablation_summary.json"):
        path.unlink()
    automatic = graph_root / "summary_all" / "track4_observer_state_ablation_summary.json"
    explicit = graph_root / "manual_review" / "track4_observer_state_ablation_summary.json"

    write_track4_observer_state_action_summary(automatic)
    automatic_payload = json.loads(automatic.read_text(encoding="utf-8"))
    automatic_payload["claim_evaluation"]["point_estimate"] = 2.0
    automatic.write_text(json.dumps(automatic_payload), encoding="utf-8")

    write_track4_observer_state_action_summary(explicit)
    explicit_payload = json.loads(explicit.read_text(encoding="utf-8"))
    explicit_payload["claim_evaluation"]["point_estimate"] = 7.0
    explicit.write_text(json.dumps(explicit_payload), encoding="utf-8")
    older = automatic.stat().st_mtime - 10
    os.utime(explicit, (older, older))

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
        track4_observer_state_summary_path=explicit,
    )

    summary = payloads["track4_observer_state_action_summary"]
    assert Path(summary["source_path"]) == explicit.resolve()
    assert summary["selection_policy"] == "explicit_summary_path_then_thesis_safe_then_summary_all_then_mtime"
    assert summary["claim_evaluation"]["point_estimate"] == 7.0


def test_track4_observer_state_candidate_inventory_records_unreadable_summaries(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    graph_root = fixture["root"] / "outputs" / "track4_action_graph"
    for path in graph_root.glob("**/track4_observer_state_ablation_summary.json"):
        path.unlink()
    valid = graph_root / "summary_all" / "track4_observer_state_ablation_summary.json"
    invalid = graph_root / "corrupt_live_root" / "track4_observer_state_ablation_summary.json"
    write_track4_observer_state_action_summary(valid)
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("{not-json", encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    inventory = payloads["track4_observer_state_action_summary"]["candidate_inventory"]
    assert inventory["candidate_count"] >= 1
    assert inventory["unreadable_candidate_count"] == 1
    assert inventory["unreadable_candidates"][0]["path"] == str(invalid)
    assert inventory["unreadable_candidates"][0]["error_type"] == "JSONDecodeError"


def test_track4_observer_state_action_only_claim_can_pass_when_hysteresis_mechanism_fails(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    summary_path = (
        fixture["root"]
        / "outputs"
        / "track4_action_graph"
        / "fixture_observer_state"
        / "track4_observer_state_ablation_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["claim_evaluation"]["pass"] = False
    payload["claim_evaluation"]["thesis_safe"] = False
    payload["claim_evaluation"]["safe_for_thesis_claim"] = False
    payload["claim_evaluation"]["failure_reasons"] = ["shuffled_hysteresis_baseline_not_separated"]
    payload["claim_evaluation"]["shuffled_hysteresis_baseline_pass"] = False
    payload["claim_evaluation"]["hysteresis_mechanism_pass"] = False
    payload["claim_evaluation"]["action_only_robustness_pass"] = True
    payload["claim_evaluation"]["action_kernel_robustness_pass"] = True
    payload["claim_evaluation"]["action_seed_robustness_pass"] = True
    payload["claim_evaluation"]["action_basis_robustness_pass"] = True
    payload["claim_evaluation"]["action_basis_seed_robustness_pass"] = True
    payload["claim_evaluation"]["action_kernel_basis_robustness_pass"] = True
    payload["claim_evaluation"]["action_kernel_seed_basis_robustness_pass"] = True
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
        track4_observer_state_summary_path=summary_path,
    )
    claims = {claim["claim_id"]: claim for claim in payloads["claim_matrix"]["claims"]}

    assert claims["track4_observer_state_action_only_separation"]["thesis_safe"] is True
    assert claims["track4_observer_state_action_separation"]["thesis_safe"] is False
    assert claims["track4_observer_state_action_separation"]["failure_reasons"] == [
        "shuffled_hysteresis_baseline_not_separated"
    ]
