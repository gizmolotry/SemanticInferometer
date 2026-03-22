import csv
import json
import shutil
from pathlib import Path

import run_full_experiment_suite as suite


def _write_monolith_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "bt_uid",
                "title",
                "density",
                "stress",
                "zone",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "index": 0,
                "bt_uid": "a0",
                "title": "t0",
                "density": "0.1",
                "stress": "0.2",
                "zone": "Bridge",
            }
        )
        writer.writerow(
            {
                "index": 1,
                "bt_uid": "a1",
                "title": "t1",
                "density": "0.7",
                "stress": "0.9",
                "zone": "Swamp",
            }
        )


def _write_verification_report(path: Path, global_pass: bool = True) -> None:
    payload = {
        "run_id": "r1",
        "timestamp": "2026-02-28T00:00:00Z",
        "layers": [
            {
                "layer_id": "rbf/cls",
                "layer_name": "cls",
                "status": "VERIFIED" if global_pass else "NON_COMPARABLE",
                "checks": [{"name": "crn_locked", "pass": True}],
                "fail_reasons": [],
            }
        ],
        "global_pass": global_pass,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_verification_summary(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_broken", "n_trapped", "n_total"])
        writer.writeheader()
        writer.writerow({"n_broken": 0, "n_trapped": 0, "n_total": 2})


def _workspace_tmp(test_name: str) -> Path:
    root = Path.cwd() / ".pytest_local_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / test_name
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def test_emit_consumer_contract_bundle_happy_path():
    tmp = _workspace_tmp("test_emit_consumer_contract_bundle_happy_path")
    exp_dir = tmp / "outputs" / "experiments" / "runs" / "experiments_20260228_000000"
    run_dir = exp_dir / "synthetic" / "rbf_seed42"
    _write_monolith_csv(run_dir / "MONOLITH_DATA.csv")
    _write_verification_report(exp_dir / "verification_report.json", global_pass=True)
    _write_verification_summary(exp_dir / "verification_summary.csv")

    res = suite.emit_consumer_contract_bundle(run_dir)
    assert res["status"] == "success"

    required = [
        run_dir / "baseline_meta.json",
        run_dir / "baseline_state.json",
        run_dir / "validation.json",
        run_dir / "ablation_summary.json",
        run_dir / "control_metrics.json",
        run_dir / "relativity_deltas.json",
        run_dir / "relativity_cache" / "state_0.json",
        run_dir / "relativity_cache" / "delta_0.json",
        run_dir / "labels" / "hidden_groups.csv",
        run_dir / "labels" / "derived" / "group_summaries.json",
        run_dir / "labels" / "derived" / "group_matrix.json",
    ]
    for p in required:
        assert p.exists(), f"missing expected artifact: {p}"


def test_emit_consumer_contract_bundle_missing_csv_fails():
    tmp = _workspace_tmp("test_emit_consumer_contract_bundle_missing_csv_fails")
    run_dir = tmp / "run_missing_csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    res = suite.emit_consumer_contract_bundle(run_dir)
    assert res["status"] == "failed"
    assert "missing MONOLITH_DATA.csv" in res["error"]


def test_emit_consumer_contract_bundle_deterministic_observer_file_count():
    tmp = _workspace_tmp("test_emit_consumer_contract_bundle_deterministic_observer_file_count")
    run_dir = tmp / "run_count"
    _write_monolith_csv(run_dir / "MONOLITH_DATA.csv")
    _write_verification_report(run_dir / "verification_report.json", global_pass=False)

    res = suite.emit_consumer_contract_bundle(run_dir)
    assert res["status"] == "success"
    assert (run_dir / "validation.json").exists()
    rel = res["relativity"]
    assert rel["state_files"] == 2
    assert rel["delta_files"] == 2
