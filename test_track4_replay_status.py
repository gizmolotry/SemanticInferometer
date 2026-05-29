import json
from pathlib import Path

from scripts.track4_replay_status import main, summarize_replay_root


def _write_summary(root: Path, name: str) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "track4_action_summary.json").write_text("{}", encoding="utf-8")


def test_track4_replay_status_reports_missing_and_group_counts(tmp_path: Path):
    root = tmp_path / "replay"
    _write_summary(root, "baseline_raw_action_real_rbf_seed42_track2_full_20260521")
    _write_summary(root, "baseline_raw_action_control_random_rbf_seed42_track2_full_20260521")
    _write_summary(root, "baseline_raw_action_real_rbf_seed42_track2_observer_disabled_20260521")
    _write_summary(root, "unrelated_probe")

    payload = summarize_replay_root(
        root,
        branches=("baseline_raw_action",),
        bases=("track2",),
        corpora=("real", "control_random"),
        kernels=("rbf",),
        seeds=(42,),
    )

    assert payload["expected_count"] == 5
    assert payload["completed_count"] == 3
    assert payload["missing_count"] == 2
    assert payload["extra_count"] == 1
    assert payload["complete"] is False
    assert payload["expected_source"] == "arguments"
    assert payload["run_suffix"] == "20260521"
    assert payload["last_completed_summary"] is not None
    assert payload["last_completed_at_unix"] is not None
    assert payload["minutes_since_last_completion"] is not None
    assert payload["by_group"]["baseline_raw_action|track2|full"] == 2
    assert payload["by_group"]["baseline_raw_action|track2|observer_disabled"] == 1
    assert "baseline_raw_action_real_rbf_seed42_track2_observer_shuffled_20260521" in payload["missing_sample"]


def test_track4_replay_status_can_write_json(tmp_path: Path):
    root = tmp_path / "replay"
    out = tmp_path / "status.json"

    payload = summarize_replay_root(
        root,
        branches=("baseline_raw_action",),
        bases=("track2",),
        corpora=("real",),
        kernels=("rbf",),
        seeds=(42,),
    )
    out.write_text(json.dumps(payload), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["summary_type"] == "track4_replay_status"
    assert loaded["expected_count"] == 4


def test_track4_replay_status_prefers_runner_inventory(tmp_path: Path):
    root = tmp_path / "replay"
    inventory = {
        "action_branches": ["baseline_raw_action"],
        "bases": ["track2"],
        "branch_preset": "repair3",
        "dry_run": False,
        "skip_existing": True,
        "leaf_count": 2,
        "leaves": [
            {"corpus": "real", "kernel": "rbf", "seed": 42, "path": "real.pt"},
            {"corpus": "control_random", "kernel": "rbf", "seed": 42, "path": "random.pt"},
        ],
        "action_branch_specs": {"baseline_raw_action": {"force_basis": None}},
    }
    root.mkdir(parents=True)
    (root / "observer_state_matrix_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    _write_summary(root, "baseline_raw_action_real_rbf_seed42_track2_full_20260521")

    payload = summarize_replay_root(
        root,
        branches=("baseline_raw_action",),
        bases=("track2",),
        corpora=("real", "control_random", "control_shuffled"),
        kernels=("rbf", "matern"),
        seeds=(42, 420),
    )

    assert payload["expected_source"] == "inventory"
    assert payload["inventory_leaf_count"] == 2
    assert payload["inventory_skip_existing"] is True
    assert payload["inventory_warning"] is None
    assert payload["expected_count"] == 5
    assert payload["completed_count"] == 1
    assert "control_shuffled" not in "\n".join(payload["missing_sample"])


def test_track4_replay_status_can_ignore_inventory(tmp_path: Path):
    root = tmp_path / "replay"
    root.mkdir(parents=True)
    (root / "observer_state_matrix_inventory.json").write_text(
        json.dumps(
            {
                "action_branches": ["baseline_raw_action"],
                "bases": ["track2"],
                "leaves": [{"corpus": "real", "kernel": "rbf", "seed": 42}],
            }
        ),
        encoding="utf-8",
    )

    payload = summarize_replay_root(
        root,
        branches=("baseline_raw_action",),
        bases=("track2",),
        corpora=("real", "control_random"),
        kernels=("rbf",),
        seeds=(42,),
        prefer_inventory=False,
    )

    assert payload["expected_source"] == "arguments"
    assert payload["expected_count"] == 5


def test_track4_replay_status_require_complete_returns_nonzero_for_missing_cells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "replay"
    root.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "track4_replay_status.py",
            str(root),
            "--branch",
            "baseline_raw_action",
            "--basis",
            "track2",
            "--corpus",
            "real",
            "--kernel",
            "rbf",
            "--seed",
            "42",
            "--no-inventory",
            "--require-complete",
        ],
    )

    assert main() == 1


def test_track4_replay_status_respects_inventory_force_basis_and_custom_branch(tmp_path: Path):
    root = tmp_path / "replay"
    inventory = {
        "action_branches": ["track2_default"],
        "bases": ["track2", "integrated"],
        "leaves": [{"corpus": "real", "kernel": "rbf", "seed": 42, "path": "real.pt"}],
        "action_branch_specs": {"track2_default": {"force_basis": "track2"}},
    }
    root.mkdir(parents=True)
    (root / "observer_state_matrix_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    _write_summary(root, "track2_default_real_rbf_seed42_track2_full_20260599")

    payload = summarize_replay_root(root)

    assert payload["expected_source"] == "inventory"
    assert payload["run_suffix"] == "20260599"
    assert payload["expected_count"] == 4
    assert payload["completed_count"] == 1
    assert payload["by_group"]["track2_default|track2|full"] == 1
    assert not any("_integrated_" in item for item in payload["missing_sample"])


def test_track4_replay_status_warns_on_dry_run_inventory_with_outputs(tmp_path: Path):
    root = tmp_path / "replay"
    inventory = {
        "action_branches": ["baseline_raw_action"],
        "bases": ["track2"],
        "dry_run": True,
        "leaves": [{"corpus": "real", "kernel": "rbf", "seed": 42, "path": "real.pt"}],
    }
    root.mkdir(parents=True)
    (root / "observer_state_matrix_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    _write_summary(root, "baseline_raw_action_real_rbf_seed42_track2_full_20260521")

    payload = summarize_replay_root(root)

    assert payload["inventory_warning"] == "inventory_records_dry_run_but_completed_summaries_exist"
