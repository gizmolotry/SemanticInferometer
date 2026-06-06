import scripts.run_focused_proof_bundle as focused
import scripts.build_thesis_evidence as thesis_evidence_cli
from scripts.run_focused_proof_bundle import _failed_run_summaries, _successful_run_ids


def test_script_keeps_repo_root_importable():
    assert str(focused.REPO_ROOT) in focused.sys.path


def test_successful_run_ids_filters_failed_or_missing_runs():
    results = [
        {"returncode": 0, "run_id": "experiments_ok"},
        {"returncode": 1, "run_id": "experiments_failed"},
        {"returncode": 0, "run_id": ""},
        {"returncode": 0},
    ]

    assert _successful_run_ids(results) == ["experiments_ok"]


def test_failed_run_summaries_keep_small_diagnostic_payload():
    results = [
        {"returncode": 0, "run_id": "experiments_ok", "track5_mode": "hadamard_strict", "limit": 60},
        {"returncode": 1, "run_id": "experiments_bad", "track5_mode": "riemannian_strict", "limit": 40},
    ]

    assert _failed_run_summaries(results) == [
        {
            "run_id": "experiments_bad",
            "track5_mode": "riemannian_strict",
            "limit": 40,
            "returncode": 1,
        }
    ]


def test_run_suite_forwards_control_metric_basis(monkeypatch):
    monkeypatch.setattr(focused, "_list_run_ids", lambda: set())
    monkeypatch.setattr(focused, "_free_space_gb", lambda path: 999.0)

    captured = {}

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return Proc()

    monkeypatch.setattr(focused, "_run_command", fake_run_command)

    result = focused._run_suite(
        limit=5,
        kernels=["matern"],
        channels=["cls"],
        corpora=["real"],
        seeds=[42, 420, 4200],
        track5_mode="hadamard_strict",
        control_metric_basis="comprehensive",
        observer_bundle_scope="none",
        observer_indices=[0, 9, 13],
        observer_index_policy="anchors",
        observer_index_count=2,
        no_cache=True,
    )

    cmd = captured["cmd"]
    idx = cmd.index("--control-metric-basis")
    assert cmd[idx + 1] == "comprehensive"
    scope_idx = cmd.index("--observer-bundle-scope")
    assert cmd[scope_idx + 1] == "none"
    observer_idx = cmd.index("--observer-indices")
    assert cmd[observer_idx + 1:observer_idx + 4] == ["0", "9", "13"]
    policy_idx = cmd.index("--observer-index-policy")
    assert cmd[policy_idx + 1] == "anchors"
    count_idx = cmd.index("--observer-index-count")
    assert cmd[count_idx + 1] == "2"
    assert "--isolate-contract-bundle" in cmd
    assert "--no-cache" in cmd
    assert result["observer_bundle_scope"] == "none"
    assert result["observer_indices"] == [0, 9, 13]
    assert result["observer_index_policy"] == "anchors"
    assert result["observer_index_count"] == 2
    assert result["no_cache"] is True
    assert result["disk_preflight"]["pass"] is True
    assert result["returncode"] == 0
    contract = result["orchestration_contract"]
    assert contract["executor"] == "external_subprocess"
    assert contract["airflow_compatible"] is True
    assert contract["nodes"][0]["node_id"] == "hadamard_main"
    assert contract["nodes"][0]["status"] == "completed"
    assert contract["nodes"][0]["run_id"] is None
    assert contract["metadata"]["airflow_sidecar"] == "analysis.airflow_ablation_orchestrator.run_ablation_matrix"


def test_run_suite_blocks_when_disk_preflight_fails(monkeypatch):
    monkeypatch.setattr(focused, "_free_space_gb", lambda path: 1.25)

    def fail_if_called(cmd, *, cwd):
        raise AssertionError("suite command should not run when disk preflight fails")

    monkeypatch.setattr(focused, "_run_command", fail_if_called)

    result = focused._run_suite(
        limit=5,
        kernels=["matern"],
        channels=["cls"],
        corpora=["real"],
        seeds=[42],
        track5_mode="hadamard_strict",
        min_free_gb=2.0,
    )

    assert result["returncode"] == 70
    assert result["run_id"] is None
    assert result["disk_preflight"] == {"pass": False, "free_gb": 1.25, "min_free_gb": 2.0}
    assert "disk preflight blocked" in result["stderr_tail"]
    assert result["orchestration_contract"]["nodes"][0]["status"] == "blocked"


def test_reused_suite_result_marks_stage_completed_with_reuse_metadata():
    result = focused._reused_suite_result(
        run_id=" experiments_reused_h ",
        track5_mode="hadamard_strict",
        limit=120,
    )

    assert result["command"] == []
    assert result["returncode"] == 0
    assert result["run_id"] == "experiments_reused_h"
    assert result["synthetic"] is False
    assert result["track5_mode"] == "hadamard_strict"
    assert result["limit"] == 120
    assert result["observer_bundle_scope"] == "reused"
    assert result["observer_indices"] is None
    assert result["observer_index_policy"] == "reused"
    assert result["observer_index_count"] == 0
    assert result["no_cache"] is None
    assert result["disk_preflight"] is None
    assert result["timeout_seconds"] is None
    assert result["reused"] is True

    contract = result["orchestration_contract"]
    assert contract["executor"] == "external_subprocess"
    assert contract["manifest_paths"] == [
        str(focused.RUNS_ROOT / "experiments_reused_h" / "experiment_manifest.json")
    ]
    assert contract["dependencies"] == {"hadamard_main": []}
    node = contract["nodes"][0]
    assert node["node_id"] == "hadamard_main"
    assert node["status"] == "completed"
    assert node["run_id"] == "experiments_reused_h"
    assert node["command"] == []
    assert node["limit"] == 120
    assert contract["metadata"]["airflow_sidecar"] == "analysis.airflow_ablation_orchestrator.run_ablation_matrix"


def test_run_command_returns_timeout_completed_process(monkeypatch):
    def fake_run(*args, **kwargs):
        raise focused.subprocess.TimeoutExpired(
            cmd=["python", "slow.py"],
            timeout=2,
            output="partial out",
            stderr="partial err",
        )

    monkeypatch.setattr(focused.subprocess, "run", fake_run)

    proc = focused._run_command(["python", "slow.py"], cwd=focused.REPO_ROOT, timeout_seconds=2)

    assert proc.returncode == 124
    assert proc.stdout == "partial out"
    assert "timed out after 2s" in proc.stderr


def test_focused_orchestration_contract_exposes_stage_dependencies():
    contract = focused._focused_orchestration_contract(
        bundle_name="focused_probe",
        stage="evidence_refresh",
        suite_results=[
            {"track5_mode": "hadamard_strict", "synthetic": False, "run_id": "experiments_h", "returncode": 0},
            {"track5_mode": "riemannian_strict", "synthetic": False, "run_id": "experiments_r", "returncode": 1},
            {"track5_mode": "hadamard_strict", "synthetic": True, "run_id": "experiments_s", "returncode": 0},
        ],
        evidence=None,
        run_ids=["experiments_h", "experiments_s"],
    )

    nodes = {node["node_id"]: node for node in contract["nodes"]}
    assert nodes["hadamard_main"]["status"] == "completed"
    assert nodes["riemannian_main"]["status"] == "failed"
    assert nodes["synthetic_bundle"]["status"] == "completed"
    assert nodes["evidence_refresh"]["status"] == "running"
    assert contract["dependencies"]["riemannian_main"] == ["hadamard_main"]
    assert contract["dependencies"]["synthetic_bundle"] == ["riemannian_main"]
    assert contract["dependencies"]["evidence_refresh"] == [
        "hadamard_main",
        "riemannian_main",
        "synthetic_bundle",
    ]
    assert contract["metadata"]["airflow_sidecars"] == [
        "analysis.airflow_ablation_orchestrator.run_ablation_matrix",
        "core.master_ablation.AblationDag",
    ]


def test_focused_orchestration_contract_reuse_mode_is_evidence_only():
    contract = focused._focused_orchestration_contract(
        bundle_name="focused_reuse",
        stage="evidence_refresh_reuse",
        suite_results=[],
        evidence={"returncode": 0, "command": ["python", "scripts/build_thesis_evidence.py"]},
        run_ids=["experiments_h", "experiments_r"],
        reuse=True,
    )

    node_ids = [node["node_id"] for node in contract["nodes"]]
    assert node_ids == ["evidence_refresh", "marker_publish"]
    assert contract["dependencies"]["evidence_refresh"] == []
    assert contract["dependencies"]["marker_publish"] == ["evidence_refresh"]
    assert contract["manifest_paths"] == [
        str(focused.RUNS_ROOT / "experiments_h" / "experiment_manifest.json"),
        str(focused.RUNS_ROOT / "experiments_r" / "experiment_manifest.json"),
    ]


def test_build_evidence_bundle_forwards_control_metric_basis(monkeypatch, tmp_path):
    captured = {}

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return Proc()

    monkeypatch.setattr(focused, "_run_command", fake_run_command)

    result = focused._build_evidence_bundle(
        bundle_dir=tmp_path / "bundle",
        run_id_allowlist=tmp_path / "run_ids.txt",
        control_metric_basis="comprehensive",
    )

    cmd = captured["cmd"]
    idx = cmd.index("--control-metric-basis")
    assert cmd[idx + 1] == "comprehensive"
    assert result["returncode"] == 0


def test_inactive_marker_does_not_replace_current_bundle(monkeypatch, tmp_path):
    thesis_root = tmp_path / "focused"
    runs_root = tmp_path / "runs"
    current_marker = thesis_root / "current_bundle.json"
    current_marker.parent.mkdir(parents=True, exist_ok=True)
    current_marker.write_text('{"bundle_name": "previous_good"}', encoding="utf-8")
    bundle_dir = thesis_root / "new_failed"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(focused, "THESIS_ROOT", thesis_root)
    monkeypatch.setattr(focused, "RUNS_ROOT", runs_root)

    marker = focused._write_marker(
        bundle_dir=bundle_dir,
        bundle_name="new_failed",
        run_ids=["experiments_bad"],
        preferred_run_id="experiments_bad",
        hadamard_run_id="experiments_bad",
        riemannian_run_id=None,
        synthetic_run_id=None,
        activate_current=False,
    )

    assert marker == bundle_dir / "inactive_bundle_marker.json"
    assert '"previous_good"' in current_marker.read_text(encoding="utf-8")
    assert '"status": "inactive"' in marker.read_text(encoding="utf-8")


def test_focused_evidence_acceptance_reports_unsafe_claims(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "claim_matrix.json").write_text(
        focused.json.dumps(
            {
                "claims": [
                    {"claim_id": "control_destruction", "thesis_safe": True},
                    {"claim_id": "observer_relativity", "thesis_safe": True},
                    {"claim_id": "track4_traversal_validity", "thesis_safe": False},
                    {"claim_id": "track5_ablation_coverage", "thesis_safe": True},
                    {"claim_id": "synthetic_recoverability", "thesis_safe": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "scientific_validation_summary.json").write_text(
        focused.json.dumps({"failure_modes": {"count": 2}}),
        encoding="utf-8",
    )

    acceptance = focused._focused_evidence_acceptance(bundle)

    assert acceptance["mechanically_usable"] is True
    assert acceptance["safe_for_thesis_claims"] is False
    assert acceptance["safe_for_focused_defense"] is False
    assert acceptance["unsafe_claim_ids"] == ["track4_traversal_validity", "synthetic_recoverability"]
    assert acceptance["unsafe_focused_claim_ids"] == ["track4_traversal_validity"]
    assert acceptance["failure_mode_count"] == 2


def test_focused_evidence_acceptance_supports_procrustes_claim_profile(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "claim_matrix.json").write_text(
        focused.json.dumps(
            {
                "claims": [
                    {"claim_id": "control_destruction", "thesis_safe": False},
                    {"claim_id": "procrustes_control_separation", "thesis_safe": True},
                    {"claim_id": "observer_relativity", "thesis_safe": True},
                    {"claim_id": "track4_traversal_validity", "thesis_safe": False},
                    {"claim_id": "track5_ablation_coverage", "thesis_safe": True},
                    {"claim_id": "verification_provenance", "thesis_safe": False},
                    {"claim_id": "procrustes_verification_provenance", "thesis_safe": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "scientific_validation_summary.json").write_text(
        focused.json.dumps({"failure_modes": {"count": 2}}),
        encoding="utf-8",
    )

    full = focused._focused_evidence_acceptance(bundle)
    procrustes = focused._focused_evidence_acceptance(bundle, claim_profile="procrustes_control")

    assert full["safe_for_focused_defense"] is False
    assert full["claim_profile"] == "full_system"
    assert procrustes["safe_for_focused_defense"] is True
    assert procrustes["claim_profile"] == "procrustes_control"
    assert "control_destruction" not in procrustes["focused_required_claim_ids"]
    assert "track4_traversal_validity" not in procrustes["focused_required_claim_ids"]
    assert procrustes["focused_required_claim_ids"] == [
        "observer_relativity",
        "procrustes_control_separation",
        "procrustes_verification_provenance",
        "track5_ablation_coverage",
    ]


def test_focused_evidence_acceptance_rejects_missing_allowlisted_runs(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "claim_matrix.json").write_text(
        focused.json.dumps(
            {
                "claims": [
                    {"claim_id": "control_destruction", "thesis_safe": True},
                    {"claim_id": "observer_relativity", "thesis_safe": True},
                    {"claim_id": "track4_traversal_validity", "thesis_safe": True},
                    {"claim_id": "track5_ablation_coverage", "thesis_safe": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "scientific_validation_summary.json").write_text(
        focused.json.dumps(
            {
                "input_selection": {
                    "focused_filter_active": True,
                    "selected_manifest_count": 0,
                    "missing_requested_run_ids": ["experiments_missing"],
                    "missing_requested_manifest_paths": [],
                }
            }
        ),
        encoding="utf-8",
    )

    acceptance = focused._focused_evidence_acceptance(bundle)

    assert acceptance["mechanically_usable"] is False
    assert acceptance["safe_for_focused_defense"] is False
    assert acceptance["focused_selection_valid"] is False
    assert acceptance["missing_requested_run_ids"] == ["experiments_missing"]


def test_thesis_evidence_cli_rejects_empty_focused_selection():
    ok, reason = thesis_evidence_cli._focused_selection_is_valid(
        {
            "input_selection": {
                "focused_filter_active": True,
                "selected_manifest_count": 0,
                "missing_requested_run_ids": ["experiments_missing"],
            }
        }
    )

    assert ok is False
    assert "zero manifests" in reason


def test_reuse_run_ids_builds_bundle_without_running_suites(monkeypatch, tmp_path):
    thesis_root = tmp_path / "focused"
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    monkeypatch.setattr(focused, "THESIS_ROOT", thesis_root)
    monkeypatch.setattr(focused, "RUNS_ROOT", runs_root)

    def fake_build(*, bundle_dir, run_id_allowlist, control_metric_basis="auto"):
        (bundle_dir / "claim_matrix.json").write_text(
            focused.json.dumps({"claims": [{"claim_id": "control_destruction", "thesis_safe": True}]}),
            encoding="utf-8",
        )
        return {"returncode": 0, "command": ["fake"], "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(focused, "_build_evidence_bundle", fake_build)
    monkeypatch.setattr(
        focused,
        "_focused_evidence_acceptance",
        lambda bundle_dir, **kwargs: {
            "mechanically_usable": True,
            "safe_for_thesis_claims": True,
            "safe_for_focused_defense": True,
        },
    )
    monkeypatch.setattr(
        focused.sys,
        "argv",
        [
            "run_focused_proof_bundle.py",
            "--reuse-run-ids",
            "experiments_h",
            "experiments_r",
            "--preferred-run-id",
            "experiments_h",
            "--hadamard-run-id",
            "experiments_h",
            "--riemannian-run-id",
            "experiments_r",
        ],
    )

    assert focused.main() == 0
    marker = focused.json.loads((thesis_root / "current_bundle.json").read_text(encoding="utf-8"))
    assert marker["preferred_run_id"] == "experiments_h"
    assert marker["run_families"]["riemannian_main"] == "experiments_r"


def test_reuse_hadamard_run_id_continues_remaining_stages(monkeypatch, tmp_path):
    thesis_root = tmp_path / "focused"
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    monkeypatch.setattr(focused, "THESIS_ROOT", thesis_root)
    monkeypatch.setattr(focused, "RUNS_ROOT", runs_root)

    calls = []

    def fake_run_suite(**kwargs):
        calls.append(kwargs)
        if kwargs["track5_mode"] == "hadamard_strict" and not kwargs.get("synthetic", False):
            raise AssertionError("Hadamard stage should be represented as reused, not rerun")
        run_id = "experiments_generated_s" if kwargs.get("synthetic", False) else "experiments_generated_r"
        return {
            "command": ["fake-suite"],
            "returncode": 0,
            "run_id": run_id,
            "synthetic": bool(kwargs.get("synthetic", False)),
            "track5_mode": kwargs["track5_mode"],
            "limit": kwargs["limit"],
            "observer_bundle_scope": kwargs["observer_bundle_scope"],
            "observer_indices": kwargs["observer_indices"],
            "observer_index_policy": kwargs["observer_index_policy"],
            "observer_index_count": kwargs["observer_index_count"],
            "no_cache": kwargs["no_cache"],
            "disk_preflight": {"pass": True, "free_gb": 999.0, "min_free_gb": 0.0},
            "timeout_seconds": kwargs["timeout_seconds"],
            "stdout_tail": "",
            "stderr_tail": "",
            "orchestration_contract": focused._suite_orchestration_contract(
                track5_mode=kwargs["track5_mode"],
                synthetic=bool(kwargs.get("synthetic", False)),
                run_id=run_id,
                status="completed",
                command=["fake-suite"],
                limit=kwargs["limit"],
            ),
        }

    def fake_build(*, bundle_dir, run_id_allowlist, control_metric_basis="auto"):
        assert run_id_allowlist.read_text(encoding="utf-8").splitlines() == [
            "experiments_reused_h",
            "experiments_generated_r",
            "experiments_generated_s",
        ]
        (bundle_dir / "claim_matrix.json").write_text(
            focused.json.dumps({"claims": [{"claim_id": "control_destruction", "thesis_safe": True}]}),
            encoding="utf-8",
        )
        return {"returncode": 0, "command": ["fake-evidence"], "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(focused, "_run_suite", fake_run_suite)
    monkeypatch.setattr(focused, "_build_evidence_bundle", fake_build)
    monkeypatch.setattr(
        focused,
        "_focused_evidence_acceptance",
        lambda bundle_dir, **kwargs: {
            "mechanically_usable": True,
            "safe_for_thesis_claims": True,
            "safe_for_focused_defense": True,
        },
    )
    monkeypatch.setattr(
        focused.sys,
        "argv",
        [
            "run_focused_proof_bundle.py",
            "--reuse-hadamard-run-id",
            "experiments_reused_h",
            "--main-limit",
            "5",
            "--fallback-limit",
            "3",
            "--synthetic-limit",
            "4",
            "--min-free-gb",
            "0",
        ],
    )

    assert focused.main() == 0
    assert [call["track5_mode"] for call in calls] == ["riemannian_strict", "hadamard_strict"]
    assert calls[0].get("synthetic", False) is False
    assert calls[1].get("synthetic", False) is True

    summary = focused.json.loads(
        next(thesis_root.glob("focused_proof_*/focused_proof_bundle.json")).read_text(encoding="utf-8")
    )
    runs = summary["runs"]
    assert [run["run_id"] for run in runs] == [
        "experiments_reused_h",
        "experiments_generated_r",
        "experiments_generated_s",
    ]
    assert runs[0]["reused"] is True
    assert runs[0]["orchestration_contract"]["nodes"][0]["status"] == "completed"
    assert "reused" not in runs[1] or runs[1]["reused"] is not True
    assert "reused" not in runs[2] or runs[2]["reused"] is not True

    nodes = {node["node_id"]: node for node in summary["orchestration_contract"]["nodes"]}
    assert nodes["hadamard_main"]["status"] == "completed"
    assert nodes["hadamard_main"]["run_ids"] == ["experiments_reused_h"]
    assert nodes["riemannian_main"]["status"] == "completed"
    assert nodes["riemannian_main"]["run_ids"] == ["experiments_generated_r"]
    assert nodes["synthetic_bundle"]["status"] == "completed"
    assert nodes["synthetic_bundle"]["run_ids"] == ["experiments_generated_s"]

    marker = focused.json.loads((thesis_root / "current_bundle.json").read_text(encoding="utf-8"))
    assert marker["run_ids"] == [
        "experiments_reused_h",
        "experiments_generated_r",
        "experiments_generated_s",
    ]
    assert marker["preferred_run_id"] == "experiments_reused_h"
    assert marker["run_families"] == {
        "hadamard_main": "experiments_reused_h",
        "riemannian_main": "experiments_generated_r",
        "synthetic": "experiments_generated_s",
    }


def test_reuse_run_ids_does_not_activate_unsafe_focused_defense(monkeypatch, tmp_path):
    thesis_root = tmp_path / "focused"
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    monkeypatch.setattr(focused, "THESIS_ROOT", thesis_root)
    monkeypatch.setattr(focused, "RUNS_ROOT", runs_root)

    def fake_build(*, bundle_dir, run_id_allowlist, control_metric_basis="auto"):
        (bundle_dir / "claim_matrix.json").write_text(
            focused.json.dumps(
                {
                    "claims": [
                        {"claim_id": "control_destruction", "thesis_safe": False},
                        {"claim_id": "observer_relativity", "thesis_safe": True},
                        {"claim_id": "track4_traversal_validity", "thesis_safe": True},
                        {"claim_id": "track5_ablation_coverage", "thesis_safe": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {"returncode": 0, "command": ["fake"], "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(focused, "_build_evidence_bundle", fake_build)
    monkeypatch.setattr(
        focused,
        "_focused_evidence_acceptance",
        lambda bundle_dir, **kwargs: {
            "mechanically_usable": True,
            "safe_for_thesis_claims": False,
            "safe_for_focused_defense": False,
            "unsafe_focused_claim_ids": ["control_destruction"],
        },
    )
    monkeypatch.setattr(
        focused.sys,
        "argv",
        [
            "run_focused_proof_bundle.py",
            "--reuse-run-ids",
            "experiments_h",
            "experiments_r",
            "--preferred-run-id",
            "experiments_h",
        ],
    )

    assert focused.main() == 1
    assert not (thesis_root / "current_bundle.json").exists()
    marker = focused.json.loads(
        next(thesis_root.glob("focused_proof_*/inactive_bundle_marker.json")).read_text(encoding="utf-8")
    )
    assert marker["status"] == "inactive"
