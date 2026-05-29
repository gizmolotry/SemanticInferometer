import json
from pathlib import Path

import numpy as np

from scripts import backfill_observer_contract as backfill


def _write_recenter(path: Path, *, status: str = "OK", ok: bool = True) -> Path:
    payload = {
        "status": status,
        "observer_count": 1,
        "ok_count": 1 if ok else 0,
        "focus_xy_centered_count": 1 if ok else 0,
        "path_start_match_observer_count": 1 if ok else 0,
        "replay_path_observer_count": 1 if ok else 0,
        "z_origin_policy": "xy_origin_preserve_canonical_z",
        "observers": [
            {
                "observer_idx": 7,
                "focus_idx": 7,
                "artifact_exists": ok,
                "view_state_exists": ok,
                "focus_xy_centered": ok,
                "path_starts_match_articles": ok,
                "relativity_state_exists": ok,
                "relativity_delta_exists": ok,
                "relativity_state_has_walker_paths": ok,
                "path_mismatches": [] if ok else [{"article_idx": 7}],
                "ok": ok,
            }
        ],
    }
    out = path / "observer_recenter_summary.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _write_replay_contract(leaf: Path, observer_idx: int = 0) -> None:
    observer_dir = leaf / f"observer_{observer_idx}"
    observer_dir.mkdir(parents=True, exist_ok=True)
    (leaf / "MONOLITH.view_state.json").write_text("{}", encoding="utf-8")
    (observer_dir / "MONOLITH.view_state.json").write_text(
        json.dumps(
            {
                "walker_paths": [
                    {
                        "article_idx": observer_idx,
                        "start_x": 0.0,
                        "start_y": 0.0,
                        "focused_observer_replay": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (observer_dir / "MONOLITH.html").write_text("<html></html>", encoding="utf-8")
    rel_dir = leaf / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    (rel_dir / f"state_{observer_idx}.json").write_text(
        json.dumps({"walker_paths": [{"article_idx": observer_idx}]}),
        encoding="utf-8",
    )


def _write_legacy_cyclic_paths(leaf: Path, observer_idx: int = 7) -> None:
    np.savez(
        leaf / "cyclic_paths.npz",
        path_indices=np.asarray([[observer_idx, 4, observer_idx]], dtype=object),
        path_anchor_idx=np.asarray([observer_idx], dtype=np.int32),
        path_is_hot=np.asarray([False], dtype=bool),
        work_integral=np.asarray([1.25], dtype=np.float32),
        closed_loop=np.asarray([True], dtype=bool),
    )


def _write_legacy_observer_leaf(leaf: Path, observer_idx: int = 7) -> None:
    observer_dir = leaf / f"observer_{observer_idx}"
    observer_dir.mkdir(parents=True, exist_ok=True)
    (leaf / "MONOLITH.view_state.json").write_text(
        json.dumps({"articles": [{"idx": observer_idx, "x": 2.0, "y": -1.0, "z": 0.2}]}),
        encoding="utf-8",
    )
    (observer_dir / "MONOLITH.html").write_text("<html>observer</html>", encoding="utf-8")
    (observer_dir / "MONOLITH.view_state.json").write_text(
        json.dumps(
            {
                "observer_focus": {"idx": observer_idx},
                "articles": [
                    {"idx": observer_idx, "x": 0.0, "y": 0.0, "z": 0.2},
                    {"idx": 4, "x": 1.0, "y": 0.5, "z": 0.4},
                ],
            }
        ),
        encoding="utf-8",
    )
    rel_dir = leaf / "relativity_cache"
    rel_dir.mkdir(parents=True, exist_ok=True)
    (rel_dir / f"state_{observer_idx}.json").write_text(json.dumps({"observer_id": observer_idx}), encoding="utf-8")
    (rel_dir / f"delta_{observer_idx}.json").write_text(json.dumps({"observer_id": observer_idx}), encoding="utf-8")
    _write_legacy_cyclic_paths(leaf, observer_idx=observer_idx)


def test_plan_only_classifies_recenter_backfill_actions_without_mutating(monkeypatch, tmp_path: Path):
    root = tmp_path / "runs"
    valid_leaf = root / "experiments_fixture" / "rbf" / "cls" / "real"
    emit_leaf = root / "experiments_fixture" / "matern" / "cls" / "real"
    backfill_leaf = root / "experiments_fixture" / "imq" / "cls" / "real"
    bundle_leaf = root / "experiments_fixture" / "rbf" / "cls" / "control_random"
    out_json = tmp_path / "plan.json"

    for leaf in (valid_leaf, emit_leaf):
        _write_replay_contract(leaf)
    _write_recenter(valid_leaf)

    backfill_leaf.mkdir(parents=True)
    (backfill_leaf / "MONOLITH_DATA.csv").write_text("index,title\n0,A\n", encoding="utf-8")
    (backfill_leaf / "observer_global.pt").write_text("payload", encoding="utf-8")

    bundle_leaf.mkdir(parents=True)
    (bundle_leaf / "MONOLITH_DATA.csv").write_text("index,title\n0,A\n", encoding="utf-8")

    monkeypatch.setattr(
        backfill.suite,
        "_emit_observer_recenter_summary_json",
        lambda _run_dir: (_ for _ in ()).throw(AssertionError("plan-only must not emit summaries")),
    )
    monkeypatch.setattr(
        backfill.suite,
        "emit_consumer_contract_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("plan-only must not backfill")),
    )

    rc = backfill.main([str(root), "--recursive", "--dry-run", "--output-json", str(out_json)])

    assert rc == 0
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["summary_type"] == "observer_contract_backfill_plan"
    assert plan["status"] == "success"
    assert plan["dry_run"] is True
    assert plan["mode"] == "consumer_contract_bundle_plan"
    assert plan["leaf_count"] == 4
    assert plan["action_counts"] == {
        "emit_recenter_summary": 1,
        "needs_full_contract_bundle": 1,
        "needs_observer_backfill": 1,
        "skip_already_valid": 1,
    }
    actions = {row["action"]: row for row in plan["leaves"]}
    assert actions["skip_already_valid"]["classification"] == "valid_existing_recenter"
    assert actions["skip_already_valid"]["skip_reason"] == "already_valid"
    assert actions["emit_recenter_summary"]["classification"] == "ready_for_recenter_emit"
    assert actions["needs_observer_backfill"]["classification"] == "observer_payloads_need_materialization"
    assert actions["needs_full_contract_bundle"]["classification"] == "leaf_needs_full_contract_bundle"
    assert any(row["action"] == "skip_already_valid" and row["would_mutate"] is False for row in plan["leaves"])
    assert any(row["action"] == "emit_recenter_summary" and row["would_mutate"] is True for row in plan["leaves"])
    assert any(row["action"] == "emit_recenter_summary" and row["would_emit"] is True for row in plan["leaves"])
    assert any(row["action"] == "needs_observer_backfill" for row in plan["leaves"])
    assert any(row["action"] == "needs_full_contract_bundle" for row in plan["leaves"])


def test_plan_only_reports_no_leaves_nonzero(tmp_path: Path):
    out_json = tmp_path / "empty_plan.json"

    rc = backfill.main([str(tmp_path / "missing"), "--recursive", "--plan-only", "--output-json", str(out_json)])

    assert rc == 1
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["summary_type"] == "observer_contract_backfill_plan"
    assert plan["status"] == "failed"
    assert plan["leaf_count"] == 0


def test_plan_json_is_written_before_recenter_backfill(monkeypatch, tmp_path: Path):
    leaf = tmp_path / "experiments_fixture" / "rbf" / "cls" / "real"
    _write_replay_contract(leaf, observer_idx=7)
    plan_json = tmp_path / "preflight_plan.json"

    monkeypatch.setattr(backfill.suite, "_emit_observer_recenter_summary_json", lambda run_dir: _write_recenter(run_dir))

    rc = backfill.main([str(leaf), "--recenter-only", "--plan-json", str(plan_json)])

    assert rc == 0
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    assert plan["summary_type"] == "observer_contract_backfill_plan"
    assert plan["leaf_count"] == 1
    assert plan["mode"] == "recenter_only_plan"
    assert plan["leaves"][0]["action"] == "emit_recenter_summary"
    assert (leaf / "observer_recenter_summary.json").exists()


def test_plan_only_classifies_observer_dirs_without_complete_contract(monkeypatch, tmp_path: Path):
    leaf = tmp_path / "partial_leaf"
    (leaf / "observer_0").mkdir(parents=True)
    out_json = tmp_path / "partial_plan.json"

    monkeypatch.setattr(
        backfill.suite,
        "_emit_observer_recenter_summary_json",
        lambda _run_dir: (_ for _ in ()).throw(AssertionError("dry-run must not emit summaries")),
    )

    rc = backfill.main([str(leaf), "--dry-run", "--recenter-only", "--output-json", str(out_json)])

    assert rc == 0
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan["mode"] == "recenter_only_plan"
    assert plan["leaf_count"] == 1
    row = plan["leaves"][0]
    assert row["action"] == "emit_recenter_summary_expect_invalid"
    assert row["classification"] == "observer_dirs_without_complete_view_state_contract"
    assert row["has_monolith_csv"] is False


def test_plan_only_classifies_view_states_without_path_replay_contract(monkeypatch, tmp_path: Path):
    leaf = tmp_path / "legacy_centered_leaf"
    (leaf / "observer_0").mkdir(parents=True)
    (leaf / "MONOLITH.view_state.json").write_text("{}", encoding="utf-8")
    (leaf / "observer_0" / "MONOLITH.view_state.json").write_text(
        json.dumps({"articles": [{"idx": 0, "x": 0.0, "y": 0.0}]}),
        encoding="utf-8",
    )
    out_json = tmp_path / "legacy_plan.json"

    monkeypatch.setattr(
        backfill.suite,
        "_emit_observer_recenter_summary_json",
        lambda _run_dir: (_ for _ in ()).throw(AssertionError("dry-run must not emit summaries")),
    )

    rc = backfill.main([str(leaf), "--dry-run", "--recenter-only", "--output-json", str(out_json)])

    assert rc == 0
    plan = json.loads(out_json.read_text(encoding="utf-8"))
    row = plan["leaves"][0]
    assert row["action"] == "emit_recenter_summary_expect_invalid"
    assert row["classification"] == "observer_view_states_missing_path_replay_contract"
    assert row["observer_view_state_count"] == 1
    assert row["observer_walker_path_count"] == 0
    assert row["observer_replay_path_count"] == 0
    assert row["relativity_state_walker_path_count"] == 0


def test_plan_only_classifies_legacy_cyclic_paths_as_hydratable(monkeypatch, tmp_path: Path):
    leaf = tmp_path / "legacy_hydratable_leaf"
    _write_legacy_observer_leaf(leaf, observer_idx=7)
    out_json = tmp_path / "legacy_hydratable_plan.json"

    monkeypatch.setattr(
        backfill.suite,
        "_emit_observer_recenter_summary_json",
        lambda _run_dir: (_ for _ in ()).throw(AssertionError("dry-run must not emit summaries")),
    )

    rc = backfill.main([str(leaf), "--dry-run", "--recenter-only", "--output-json", str(out_json)])

    assert rc == 0
    row = json.loads(out_json.read_text(encoding="utf-8"))["leaves"][0]
    assert row["action"] == "hydrate_legacy_path_ledger"
    assert row["classification"] == "legacy_cyclic_paths_available_for_projection"
    assert row["legacy_cyclic_path_available"] is True


def test_hydrate_legacy_path_ledger_projects_cyclic_indices_with_provenance(tmp_path: Path):
    leaf = tmp_path / "legacy_hydration"
    _write_legacy_observer_leaf(leaf, observer_idx=7)

    result = backfill.hydrate_legacy_path_ledger(leaf)

    assert result["status"] == "success"
    assert result["fresh_focused_observer_replay"] is False
    assert result["hydrated_observer_count"] == 1
    observer_state = json.loads((leaf / "observer_7" / "MONOLITH.view_state.json").read_text(encoding="utf-8"))
    path = observer_state["walker_paths"][0]
    assert path["path_space"] == "rendered_synthesis_legacy_projected"
    assert path["path_row_source"] == "legacy_observer_view_state_article_coords"
    assert path["path_geometry_role"] == "legacy_projected_article_polyline"
    assert path["legacy_projected_path"] is True
    assert path["legacy_path_ledger_hydrated"] is True
    assert path["fresh_focused_observer_replay"] is False
    assert path["focused_observer_replay"] is False
    assert path["path_indices"] == [7, 4, 7]
    assert path["start_x"] == 0.0
    assert path["start_y"] == 0.0
    assert path["end_x"] == 0.0
    assert path["end_y"] == 0.0
    state = json.loads((leaf / "relativity_cache" / "state_7.json").read_text(encoding="utf-8"))
    assert state["walker_paths"][0]["legacy_projected_path"] is True
    assert state["path_ledger_provenance"]["source"] == "observer_view_state_legacy_path_ledger_hydration_v1"
    assert state["path_ledger_provenance"]["legacy_path_ledger_hydrated"] is True
    assert state["path_ledger_provenance"]["fresh_focused_observer_replay"] is False
    assert state["path_ledger_provenance"]["synthetic_placeholder"] is False


def test_recenter_only_can_hydrate_legacy_path_ledger_before_summary(tmp_path: Path):
    leaf = tmp_path / "legacy_hydration_summary"
    _write_legacy_observer_leaf(leaf, observer_idx=7)
    result_json = tmp_path / "result.json"
    review_json = tmp_path / "review.json"

    rc = backfill.main(
        [
            str(leaf),
            "--recenter-only",
            "--hydrate-legacy-path-ledger",
            "--output-json",
            str(result_json),
            "--observer-recenter-review-json",
            str(review_json),
            "--summary-only",
        ]
    )

    assert rc == 0
    result = json.loads(result_json.read_text(encoding="utf-8"))
    leaf_result = result["leaves"][0]
    assert leaf_result["legacy_path_hydration"]["hydrated_observer_count"] == 1
    assert leaf_result["recenter_status"] == "OK"
    assert leaf_result["recenter_path_start_match_observer_count"] == 1
    assert leaf_result["recenter_replay_path_observer_count"] == 0
    review = json.loads(review_json.read_text(encoding="utf-8"))
    assert review["status"] == "OK"
    assert review["leaves"][0]["replay_path_observer_count"] == 0
    assert review["legacy_hydrated_leaf_count"] == 1
    assert review["fresh_replay_complete_leaf_count"] == 0
    assert review["leaves"][0]["legacy_path_ledger_hydrated"] is True
    assert review["leaves"][0]["fresh_focused_observer_replay_complete"] is False
    assert review["leaves"][0]["legacy_path_hydration"]["hydrated_observer_count"] == 1


def test_recenter_only_recursive_backfill_writes_compact_review(monkeypatch, tmp_path: Path):
    root = tmp_path / "runs"
    leaf = root / "experiments_fixture" / "rbf" / "cls" / "real"
    (leaf / "observer_7").mkdir(parents=True)
    (leaf / "MONOLITH.view_state.json").write_text("{}", encoding="utf-8")
    result_json = tmp_path / "result.json"
    review_json = tmp_path / "review.json"

    monkeypatch.setattr(backfill.suite, "_emit_observer_recenter_summary_json", lambda run_dir: _write_recenter(run_dir))

    rc = backfill.main(
        [
            str(root),
            "--recursive",
            "--recenter-only",
            "--summary-only",
            "--require-recenter-ok",
            "--output-json",
            str(result_json),
            "--observer-recenter-review-json",
            str(review_json),
        ]
    )

    assert rc == 0
    result = json.loads(result_json.read_text(encoding="utf-8"))
    assert result["leaf_count"] == 1
    assert result["all_recenter_valid"] is True
    assert result["recenter_status_counts"] == {"OK": 1}

    review = json.loads(review_json.read_text(encoding="utf-8"))
    assert review["summary_type"] == "observer_recenter_review"
    assert review["status"] == "OK"
    assert review["leaf_count"] == 1
    assert review["observer_count"] == 1
    assert review["ok_count"] == 1
    assert review["focus_xy_centered_count"] == 1
    assert review["path_start_match_observer_count"] == 1
    assert review["replay_path_observer_count"] == 1
    assert review["leaves"][0]["path_start_match_observer_count"] == 1
    assert "observer_rows" not in review["leaves"][0]


def test_recenter_only_isolate_copy_root_leaves_source_untouched(monkeypatch, tmp_path: Path):
    source_root = tmp_path / "runs"
    source_leaf = source_root / "experiments_fixture" / "rbf" / "cls" / "real"
    _write_replay_contract(source_leaf, observer_idx=7)
    copy_root = tmp_path / "isolated_copy"
    result_json = tmp_path / "isolated_result.json"
    review_json = tmp_path / "isolated_review.json"

    monkeypatch.setattr(backfill.suite, "_emit_observer_recenter_summary_json", lambda run_dir: _write_recenter(run_dir))

    rc = backfill.main(
        [
            str(source_root),
            "--recursive",
            "--recenter-only",
            "--isolate-copy-root",
            str(copy_root),
            "--output-json",
            str(result_json),
            "--observer-recenter-review-json",
            str(review_json),
            "--summary-only",
        ]
    )

    assert rc == 0
    result = json.loads(result_json.read_text(encoding="utf-8"))
    copied = result["isolation"]["copies"][0]
    copied_leaf = Path(copied["target"])
    assert copied["source"] == str(source_leaf)
    assert copied_leaf.exists()
    assert (copied_leaf / "observer_recenter_summary.json").exists()
    assert not (source_leaf / "observer_recenter_summary.json").exists()

    review = json.loads(review_json.read_text(encoding="utf-8"))
    assert review["status"] == "OK"
    assert review["leaves"][0]["run_dir"] == str(copied_leaf)
    assert review["leaves"][0]["legacy_path_ledger_hydrated"] is False


def test_isolate_copy_root_preserves_context_for_same_named_leaf_roots(monkeypatch, tmp_path: Path):
    source_root = tmp_path / "runs" / "experiments_fixture"
    rbf_leaf = source_root / "rbf" / "cls" / "real"
    imq_leaf = source_root / "imq" / "cls" / "real"
    _write_replay_contract(rbf_leaf, observer_idx=7)
    _write_replay_contract(imq_leaf, observer_idx=7)
    copy_root = tmp_path / "isolated_same_name"
    result_json = tmp_path / "isolated_same_name_result.json"

    monkeypatch.setattr(backfill.suite, "_emit_observer_recenter_summary_json", lambda run_dir: _write_recenter(run_dir))

    rc = backfill.main(
        [
            str(rbf_leaf),
            str(imq_leaf),
            "--recenter-only",
            "--isolate-copy-root",
            str(copy_root),
            "--output-json",
            str(result_json),
        ]
    )

    assert rc == 0
    result = json.loads(result_json.read_text(encoding="utf-8"))
    targets = [Path(row["target"]) for row in result["isolation"]["copies"]]
    assert len(targets) == 2
    assert len(set(targets)) == 2
    assert copy_root / "rbf" / "cls" / "real" in targets
    assert copy_root / "imq" / "cls" / "real" in targets
    assert all((target / "observer_recenter_summary.json").exists() for target in targets)
    assert not (rbf_leaf / "observer_recenter_summary.json").exists()
    assert not (imq_leaf / "observer_recenter_summary.json").exists()


def test_recenter_only_isolated_hydration_leaves_source_untouched(tmp_path: Path):
    source_root = tmp_path / "runs"
    source_leaf = source_root / "experiments_fixture" / "rbf" / "cls" / "real"
    _write_legacy_observer_leaf(source_leaf, observer_idx=7)
    copy_root = tmp_path / "isolated_hydration"
    result_json = tmp_path / "isolated_hydration_result.json"

    rc = backfill.main(
        [
            str(source_root),
            "--recursive",
            "--recenter-only",
            "--hydrate-legacy-path-ledger",
            "--isolate-copy-root",
            str(copy_root),
            "--output-json",
            str(result_json),
        ]
    )

    assert rc == 0
    result = json.loads(result_json.read_text(encoding="utf-8"))
    copied_leaf = Path(result["isolation"]["copies"][0]["target"])
    assert (copied_leaf / "observer_recenter_summary.json").exists()
    assert json.loads((copied_leaf / "observer_recenter_summary.json").read_text(encoding="utf-8"))["status"] == "OK"
    assert (copied_leaf / "observer_7" / "MONOLITH.view_state.json").read_text(encoding="utf-8") != (
        source_leaf / "observer_7" / "MONOLITH.view_state.json"
    ).read_text(encoding="utf-8")
    assert not (source_leaf / "observer_recenter_summary.json").exists()


def test_recenter_only_backfill_gates_invalid_recenter_review(monkeypatch, tmp_path: Path):
    leaf = tmp_path / "experiments_fixture" / "matern" / "cls" / "real"
    (leaf / "observer_7").mkdir(parents=True)
    (leaf / "MONOLITH.view_state.json").write_text("{}", encoding="utf-8")
    review_json = tmp_path / "invalid_review.json"

    monkeypatch.setattr(
        backfill.suite,
        "_emit_observer_recenter_summary_json",
        lambda run_dir: _write_recenter(run_dir, status="INVALID", ok=False),
    )

    rc = backfill.main(
        [
            str(leaf),
            "--recenter-only",
            "--require-valid-recenter",
            "--observer-recenter-review-json",
            str(review_json),
        ]
    )

    assert rc == 1
    review = json.loads(review_json.read_text(encoding="utf-8"))
    assert review["status"] == "INVALID"
    assert review["invalid_leaf_count"] == 1
    invalid = review["leaves"][0]["missing_or_invalid_observers"][0]
    assert invalid["observer_idx"] == 7
    assert "focus_not_centered" in invalid["reasons"]
    assert "path_start_mismatch" in invalid["reasons"]
