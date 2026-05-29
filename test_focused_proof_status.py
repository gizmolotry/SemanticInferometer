from datetime import datetime, timezone
from pathlib import Path

from scripts import focused_proof_status as status


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(status.json.dumps(payload), encoding="utf-8")


def test_status_report_summarizes_running_bundle_and_run_progress(tmp_path):
    repo = tmp_path
    focused_root = repo / "outputs" / "thesis_validation" / "focused"
    runs_root = repo / "outputs" / "experiments" / "runs"
    bundle = focused_root / "focused_proof_20260507_010203"
    run_dir = runs_root / "experiments_20260507_010204" / "matern" / "cls" / "real"

    _write_json(
        bundle / "focused_proof_status.json",
        {
            "bundle_name": bundle.name,
            "status": "running",
            "stage": "riemannian_main",
            "generated_at": "2026-05-07T01:05:00+00:00",
            "bundle_dir": str(bundle),
            "runs": [
                {
                    "run_id": "experiments_20260507_010204",
                    "returncode": 0,
                    "track5_mode": "hadamard_strict",
                    "command": [
                        "python",
                        "run_full_experiment_suite.py",
                        "--kernels",
                        "matern",
                        "rbf",
                        "--channels",
                        "cls",
                        "--corpora",
                        "real",
                        "control_constant",
                    ],
                }
            ],
            "evidence_acceptance": {"safe_for_focused_defense": True},
            "orchestration_contract": {
                "dag_id": "focused_proof.fixture",
                "nodes": [
                    {"node_id": "hadamard_main", "status": "completed"},
                    {"node_id": "riemannian_main", "status": "running"},
                    {"node_id": "evidence_refresh", "status": "pending"},
                ],
                "dependencies": {
                    "hadamard_main": [],
                    "riemannian_main": ["hadamard_main"],
                    "evidence_refresh": ["hadamard_main", "riemannian_main"],
                },
            },
        },
    )
    _write_json(run_dir / "checkpoints" / "batch" / "manifest.json", {"finalized": "now"})
    _write_json(run_dir / "verification_report.json", {"global_pass": True})

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
        now=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
    )

    assert report["active"]["running_status_count"] == 1
    running = report["focused_statuses"][0]
    assert running["stage"] == "riemannian_main"
    assert running["orchestration_contract"]["nodes"][1]["node_id"] == "riemannian_main"
    progress = running["run_progress"][0]["progress"]
    assert progress["completed_leaf_count"] == 1
    assert progress["expected_leaf_count"] == 4
    assert progress["percent"] == 25.0
    assert running["acceptance"]["safe_for_focused_defense"] is True


def test_status_report_reads_current_bundle_acceptance(tmp_path):
    repo = tmp_path
    focused_root = repo / "outputs" / "thesis_validation" / "focused"
    runs_root = repo / "outputs" / "experiments" / "runs"
    _write_json(
        focused_root / "current_bundle.json",
        {
            "status": "success",
            "bundle_name": "focused_proof_good",
            "preferred_run_id": "experiments_good",
            "run_ids": ["experiments_good"],
            "evidence_acceptance": {
                "safe_for_focused_defense": False,
                "unsafe_focused_claim_ids": ["observer_relativity"],
            },
        },
    )

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
    )

    current = report["current_bundle"]
    assert current["bundle_name"] == "focused_proof_good"
    assert current["preferred_run_id"] == "experiments_good"
    assert current["evidence_acceptance"]["unsafe_focused_claim_ids"] == ["observer_relativity"]


def test_human_output_is_json_free_summary(tmp_path):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    _write_json(
        focused_root / "current_bundle.json",
        {
            "status": "success",
            "bundle_name": "focused_proof_good",
            "preferred_run_id": "experiments_good",
            "evidence_acceptance": {"safe_for_focused_defense": True, "claim_profile": "procrustes_control"},
        },
    )
    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
    )

    human = status.format_human(report)

    assert "Focused Proof Status" in human
    assert "Current Bundle:" in human
    assert "focused defense safe: yes" in human
    assert "claim profile: procrustes_control" in human


def test_human_output_includes_orchestration_node_summary(tmp_path):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    _write_json(
        focused_root / "focused_proof_contract" / "focused_proof_status.json",
        {
            "bundle_name": "focused_proof_contract",
            "status": "running",
            "stage": "evidence_refresh",
            "generated_at": "2026-05-07T01:05:00+00:00",
            "bundle_dir": str(focused_root / "focused_proof_contract"),
            "orchestration_contract": {
                "nodes": [
                    {"node_id": "hadamard_main", "status": "completed"},
                    {"node_id": "evidence_refresh", "status": "running"},
                ]
            },
        },
    )

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
        now=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
    )

    human = status.format_human(report)

    assert "dag: hadamard_main=completed, evidence_refresh=running" in human


def test_human_output_includes_recent_run_artifact_summary(tmp_path):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    run_dir = runs_root / "experiments_recent" / "rbf" / "cls" / "real"
    _write_json(run_dir / "checkpoints" / "batch" / "manifest.json", {"ok": True})
    _write_json(run_dir / "verification_report.json", {"global_pass": True})
    _write_json(run_dir / "control_metrics.json", {"status": "OK"})
    _write_json(run_dir / "observer_0" / "MONOLITH.view_state.json", {"ok": True})
    _write_json(run_dir / "observer_recenter_summary.json", {"status": "OK"})

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
        now=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
    )
    human = status.format_human(report)

    assert "Recent Run Artifacts:" in human
    counts = report["recent_runs"][0]["artifact_counts"]
    assert counts["observer_recenter_summaries"] == 1
    assert "experiments_recent: leaves=1 reports=1 controls=1 observers=1 recenter=1" in human


def test_status_report_can_include_track4_replay_progress(tmp_path):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    replay_root = repo / "track4_replay"
    _write_json(
        replay_root / "observer_state_matrix_inventory.json",
        {
            "action_branches": ["baseline_raw_action"],
            "bases": ["track2"],
            "leaf_count": 1,
            "leaves": [{"corpus": "real", "kernel": "rbf", "seed": 42, "path": "real.pt"}],
        },
    )
    _write_json(
        replay_root
        / "baseline_raw_action_real_rbf_seed42_track2_full_20260521"
        / "track4_action_summary.json",
        {"ok": True},
    )

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
        track4_replay_root=replay_root,
    )
    human = status.format_human(report)

    assert report["track4_replay"]["present"] is True
    assert report["track4_replay"]["completed_count"] == 1
    assert report["track4_replay"]["expected_count"] == 4
    assert "Track 4 Replay:" in human
    assert "cells: 1/4" in human


def test_process_tokens_cover_leaf_and_comparison_processes():
    assert "run_experiments.py" in status.PROOF_PROCESS_TOKENS
    assert "compare_controls.py" in status.PROOF_PROCESS_TOKENS
    assert "run_track4_observer_state_matrix.py" in status.PROOF_PROCESS_TOKENS
    assert "run_track4_action_graph.py" in status.PROOF_PROCESS_TOKENS


def test_status_report_separates_stale_running_bundles(tmp_path):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    _write_json(
        focused_root / "focused_proof_active" / "focused_proof_status.json",
        {
            "bundle_name": "focused_proof_active",
            "status": "running",
            "stage": "riemannian_main",
            "generated_at": "2026-05-07T01:09:00+00:00",
            "bundle_dir": str(focused_root / "focused_proof_active"),
        },
    )
    _write_json(
        focused_root / "focused_proof_stale" / "focused_proof_status.json",
        {
            "bundle_name": "focused_proof_stale",
            "status": "running",
            "stage": "hadamard_main",
            "generated_at": "2026-05-06T23:00:00+00:00",
            "bundle_dir": str(focused_root / "focused_proof_stale"),
        },
    )

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=False,
        now=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
    )
    human = status.format_human(report)

    assert report["active"]["running_status_count"] == 2
    assert report["active"]["active_running_status_count"] == 1
    assert report["active"]["stale_running_status_count"] == 1
    assert report["active"]["running_bundles"][0]["bundle_name"] == "focused_proof_active"
    assert report["active"]["stale_running_bundles"][0]["bundle_name"] == "focused_proof_stale"
    assert "Stale running status files: 1" in human


def test_status_report_classifies_running_bundle_with_matching_process_as_active(tmp_path, monkeypatch):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    bundle = focused_root / "focused_proof_active_process"
    _write_json(
        bundle / "focused_proof_status.json",
        {
            "bundle_name": bundle.name,
            "status": "running",
            "stage": "riemannian_main",
            "generated_at": "2026-05-07T01:09:00+00:00",
            "bundle_dir": str(bundle),
        },
    )
    monkeypatch.setattr(
        status,
        "_scan_processes",
        lambda: {
            "available": True,
            "count": 1,
            "processes": [
                {
                    "pid": 123,
                    "name": "python",
                    "command_line": f"python scripts/run_focused_proof_bundle.py --bundle-dir {bundle}",
                }
            ],
        },
    )

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=True,
        now=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
    )

    running = report["focused_statuses"][0]
    assert running["running_classification"] == "active"
    assert running["activity"]["matching_active_process"] is True
    assert report["active"]["active_running_status_count"] == 1
    assert report["active"]["orphan_running_status_count"] == 0


def test_status_report_classifies_running_bundle_without_activity_as_orphan(tmp_path, monkeypatch):
    repo = tmp_path
    focused_root = repo / "focused"
    runs_root = repo / "runs"
    bundle = focused_root / "focused_proof_orphan"
    _write_json(
        bundle / "focused_proof_status.json",
        {
            "bundle_name": bundle.name,
            "status": "running",
            "stage": "hadamard_main",
            "generated_at": "2026-05-07T01:09:00+00:00",
            "bundle_dir": str(bundle),
        },
    )
    monkeypatch.setattr(status, "_scan_processes", lambda: {"available": True, "count": 0, "processes": []})

    report = status.build_status_report(
        repo_root=repo,
        runs_root=runs_root,
        focused_root=focused_root,
        include_processes=True,
        now=datetime(2026, 5, 7, 1, 10, tzinfo=timezone.utc),
    )
    human = status.format_human(report)

    running = report["focused_statuses"][0]
    assert running["running_classification"] == "orphan"
    assert running["orphan_hint"] is True
    assert running["running_classification_reasons"] == ["no_matching_active_process_or_recent_run_activity"]
    assert report["active"]["active_running_status_count"] == 0
    assert report["active"]["orphan_running_status_count"] == 1
    assert report["active"]["orphan_running_bundles"][0]["bundle_name"] == bundle.name
    assert "Orphan running status files: 1" in human
    assert "classification=orphan" in human
