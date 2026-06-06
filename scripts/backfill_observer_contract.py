#!/usr/bin/env python3
"""Materialize observer-relativity contract artifacts for an existing leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_full_experiment_suite as suite


np = suite.np


LEAF_MARKERS = (
    "MONOLITH_DATA.csv",
    "MONOLITH.view_state.json",
    "observer_global.pt",
    "MONOLITH.html",
)


def _count_paths(paths: Iterable[Path]) -> int:
    return sum(1 for path in paths if path.exists())


def _is_internal_artifact_dir(path: Path) -> bool:
    name = path.name
    return name == "relativity_cache" or name == "checkpoints" or name.startswith("observer_")


def _looks_like_leaf_dir(path: Path) -> bool:
    if not path.is_dir() or _is_internal_artifact_dir(path):
        return False
    if any((path / marker).exists() for marker in LEAF_MARKERS):
        return True
    return any(child.is_dir() and child.name.startswith("observer_") for child in path.iterdir())


def iter_leaf_dirs(root: Path, *, recursive: bool) -> List[Path]:
    root = Path(root)
    if _looks_like_leaf_dir(root):
        return [root]
    if not recursive or not root.exists():
        return []
    leaves: List[Path] = []
    for candidate in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: str(p)):
        if _looks_like_leaf_dir(candidate):
            leaves.append(candidate)
    # A malformed tree can expose both a parent and child as leaves. Keep the
    # shallowest one so the same artifact family is not backfilled twice.
    deduped: List[Path] = []
    resolved_seen: List[Path] = []
    for leaf in leaves:
        try:
            resolved = leaf.resolve()
        except Exception:
            resolved = leaf
        if any(resolved == prior or prior in resolved.parents for prior in resolved_seen):
            continue
        resolved_seen.append(resolved)
        deduped.append(leaf)
    return deduped


def _is_empty_relative_path(path: Path) -> bool:
    return str(path) in {"", "."}


def _isolated_leaf_relative_path(leaf: Path, root: Path, all_leaves: Sequence[Path]) -> Path:
    """Choose a collision-resistant relative path for an isolated leaf copy."""
    leaf_resolved = leaf.resolve()
    try:
        rel = leaf_resolved.relative_to(root.resolve())
        if not _is_empty_relative_path(rel):
            return rel
    except Exception:
        pass

    try:
        parent_paths = [str(path.resolve().parent) for path in all_leaves]
        common_parent = Path(os.path.commonpath(parent_paths))
        rel = leaf_resolved.relative_to(common_parent)
        if not _is_empty_relative_path(rel):
            return rel
    except Exception:
        pass

    digest = hashlib.sha1(str(leaf_resolved).encode("utf-8")).hexdigest()[:12]
    return Path(f"{leaf.name}_{digest}")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _recenter_fields(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "observer_recenter_summary.json"
    payload = _load_json(summary_path)
    return {
        "observer_recenter_summary": str(summary_path),
        "recenter_status": payload.get("status"),
        "recenter_observer_count": payload.get("observer_count"),
        "recenter_ok_count": payload.get("ok_count"),
        "recenter_focus_xy_centered_count": payload.get("focus_xy_centered_count"),
        "recenter_path_start_match_observer_count": payload.get("path_start_match_observer_count"),
        "recenter_replay_path_observer_count": payload.get("replay_path_observer_count"),
    }


def _recenter_summary_is_valid(payload: Dict[str, Any]) -> bool:
    try:
        observer_count = int(payload.get("observer_count") or 0)
        ok_count = int(payload.get("ok_count") or 0)
    except Exception:
        return False
    return bool(payload.get("status") == "OK" and observer_count > 0 and ok_count == observer_count)


def _observer_payload_count(run_dir: Path) -> int:
    count = 1 if (run_dir / "observer_global.pt").exists() else 0
    count += sum(1 for path in run_dir.glob("observer_*.pt") if path.is_file())
    rel_dir = run_dir / "relativity_cache"
    if rel_dir.exists():
        count += sum(1 for path in rel_dir.glob("observer_*.pt") if path.is_file())
    return count


def _observer_view_state_path_counts(observer_dirs: Sequence[Path]) -> Dict[str, int]:
    walker_path_count = 0
    replay_path_count = 0
    readable_view_state_count = 0
    for observer_dir in observer_dirs:
        payload = _load_json(observer_dir / "MONOLITH.view_state.json")
        if not payload:
            continue
        readable_view_state_count += 1
        paths = payload.get("walker_paths") if isinstance(payload.get("walker_paths"), list) else []
        walker_path_count += len(paths)
        replay_path_count += sum(1 for row in paths if isinstance(row, dict) and bool(row.get("focused_observer_replay")))
    return {
        "readable_observer_view_state_count": readable_view_state_count,
        "observer_walker_path_count": walker_path_count,
        "observer_replay_path_count": replay_path_count,
    }


def _relativity_state_walker_path_count(rel_dir: Path) -> int:
    if not rel_dir.exists():
        return 0
    count = 0
    for state_path in rel_dir.glob("state_*.json"):
        payload = _load_json(state_path)
        paths = payload.get("walker_paths") if isinstance(payload.get("walker_paths"), list) else []
        if paths:
            count += 1
    return count


def _legacy_cyclic_path_available(run_dir: Path) -> bool:
    cyclic_path = run_dir / "cyclic_paths.npz"
    if not cyclic_path.exists():
        return False
    try:
        with np.load(cyclic_path, allow_pickle=True) as npz:
            return bool("path_indices" in npz.files and "path_anchor_idx" in npz.files)
    except Exception:
        return False


def _plan_leaf(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    observer_dirs = sorted(path for path in run_dir.glob("observer_*") if path.is_dir())
    observer_view_state_count = _count_paths(path / "MONOLITH.view_state.json" for path in observer_dirs)
    observer_html_count = _count_paths(path / "MONOLITH.html" for path in observer_dirs)
    rel_dir = run_dir / "relativity_cache"
    state_count = len(list(rel_dir.glob("state_*.json"))) if rel_dir.exists() else 0
    delta_count = len(list(rel_dir.glob("delta_*.json"))) if rel_dir.exists() else 0
    path_counts = _observer_view_state_path_counts(observer_dirs)
    state_walker_path_count = _relativity_state_walker_path_count(rel_dir)
    recenter_path = run_dir / "observer_recenter_summary.json"
    recenter_payload = _load_json(recenter_path)
    summary_valid = _recenter_summary_is_valid(recenter_payload)
    has_global_view_state = (run_dir / "MONOLITH.view_state.json").exists()
    has_monolith_csv = (run_dir / "MONOLITH_DATA.csv").exists()
    observer_payload_count = _observer_payload_count(run_dir)
    legacy_cyclic_available = _legacy_cyclic_path_available(run_dir)

    if summary_valid:
        action = "skip_already_valid"
        classification = "valid_existing_recenter"
        mutates = False
        reasons = ["observer_recenter_summary_already_ok"]
    elif (
        observer_view_state_count > 0
        and has_global_view_state
        and int(path_counts["observer_walker_path_count"]) > 0
        and int(path_counts["observer_replay_path_count"]) > 0
        and state_walker_path_count > 0
    ):
        action = "emit_recenter_summary"
        classification = "ready_for_recenter_emit"
        mutates = True
        reasons = ["observer_view_states_and_path_replay_available"]
    elif observer_view_state_count > 0 and has_global_view_state and legacy_cyclic_available:
        action = "hydrate_legacy_path_ledger"
        classification = "legacy_cyclic_paths_available_for_projection"
        mutates = True
        reasons = ["observer_view_states_missing_replay_but_cyclic_path_indices_available"]
    elif observer_view_state_count > 0 and has_global_view_state:
        action = "emit_recenter_summary_expect_invalid"
        classification = "observer_view_states_missing_path_replay_contract"
        mutates = True
        reasons = ["observer_view_states_available_but_path_replay_contract_missing"]
    elif observer_dirs:
        action = "emit_recenter_summary_expect_invalid"
        classification = "observer_dirs_without_complete_view_state_contract"
        mutates = True
        reasons = ["observer_dirs_present_but_view_state_or_global_state_missing"]
    elif has_monolith_csv and observer_payload_count > 0:
        action = "needs_observer_backfill"
        classification = "observer_payloads_need_materialization"
        mutates = True
        reasons = ["leaf_has_monolith_and_observer_payloads_but_no_observer_view_states"]
    elif has_monolith_csv:
        action = "needs_full_contract_bundle"
        classification = "leaf_needs_full_contract_bundle"
        mutates = True
        reasons = ["leaf_has_monolith_csv_but_no_observer_payloads"]
    else:
        action = "not_ready"
        classification = "missing_leaf_contract_inputs"
        mutates = False
        reasons = ["missing_leaf_contract_inputs"]

    return {
        "run_dir": str(run_dir),
        "action": action,
        "planned_action": action,
        "classification": classification,
        "would_mutate": mutates,
        "would_emit": bool(action != "skip_already_valid" and mutates),
        "skip_reason": "already_valid" if action == "skip_already_valid" else None,
        "reasons": reasons,
        "has_monolith_csv": has_monolith_csv,
        "has_global_view_state": has_global_view_state,
        "observer_directory_count": len(observer_dirs),
        "observer_view_state_count": observer_view_state_count,
        "observer_html_count": observer_html_count,
        **path_counts,
        "observer_payload_count": observer_payload_count,
        "legacy_cyclic_path_available": legacy_cyclic_available,
        "relativity_state_count": state_count,
        "relativity_delta_count": delta_count,
        "relativity_state_walker_path_count": state_walker_path_count,
        "observer_recenter_summary_exists": recenter_path.exists(),
        "observer_recenter_summary_path": str(recenter_path),
        "observer_recenter_status": recenter_payload.get("status"),
        "observer_count": recenter_payload.get("observer_count"),
        "ok_count": recenter_payload.get("ok_count"),
    }


def _article_map(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    articles = payload.get("articles") if isinstance(payload.get("articles"), list) else []
    mapped: Dict[int, Dict[str, Any]] = {}
    for row in articles:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("idx", row.get("index")))
        except Exception:
            continue
        mapped[idx] = row
    return mapped


def _coord(row: Dict[str, Any], key: str) -> Optional[float]:
    try:
        value = float(row.get(key))
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _legacy_path_rows_for_observer(
    run_dir: Path,
    observer_idx: int,
    observer_articles: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cyclic_path = run_dir / "cyclic_paths.npz"
    if not cyclic_path.exists() or not observer_articles:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with np.load(cyclic_path, allow_pickle=True) as npz:
            if "path_indices" not in npz.files or "path_anchor_idx" not in npz.files:
                return []
            path_indices = npz["path_indices"]
            path_anchor_idx = np.asarray(npz["path_anchor_idx"], dtype=int).reshape(-1)
            path_is_hot = (
                np.asarray(npz["path_is_hot"], dtype=bool).reshape(-1)
                if "path_is_hot" in npz.files
                else np.zeros(path_anchor_idx.shape, dtype=bool)
            )
            work_integral = (
                np.asarray(npz["work_integral"], dtype=float).reshape(-1)
                if "work_integral" in npz.files
                else np.full(path_anchor_idx.shape, np.nan, dtype=float)
            )
            closed_loop = (
                np.asarray(npz["closed_loop"], dtype=bool).reshape(-1)
                if "closed_loop" in npz.files
                else np.zeros(path_anchor_idx.shape, dtype=bool)
            )
    except Exception:
        return []

    preferred_indices = [i for i, anchor in enumerate(path_anchor_idx.tolist()) if int(anchor) == int(observer_idx)]
    if not preferred_indices:
        preferred_indices = [
            i
            for i in range(len(path_anchor_idx))
            if int(observer_idx) in {int(value) for value in np.asarray(path_indices[i]).reshape(-1).tolist()}
        ]
    for path_ordinal in preferred_indices:
        try:
            sequence = [int(value) for value in np.asarray(path_indices[path_ordinal]).reshape(-1).tolist()]
        except Exception:
            continue
        coords = []
        for article_idx in sequence:
            owner = observer_articles.get(article_idx)
            if not isinstance(owner, dict):
                continue
            x = _coord(owner, "x")
            y = _coord(owner, "y")
            z = _coord(owner, "z")
            if x is None or y is None:
                continue
            coords.append((article_idx, x, y, z))
        if not coords:
            continue
        start = coords[0]
        end = coords[-1]
        row = {
            "article_idx": int(start[0]),
            "path_space": "rendered_synthesis_legacy_projected",
            "path_contract_version": "legacy_projected_v1",
            "path_row_source": "legacy_observer_view_state_article_coords",
            "path_geometry_role": "legacy_projected_article_polyline",
            "legacy_projected_path": True,
            "legacy_path_ledger_hydrated": True,
            "fresh_focused_observer_replay": False,
            "focused_observer_replay": False,
            "source": "cyclic_paths.npz:path_indices",
            "path_ordinal": int(path_ordinal),
            "path_anchor_idx": int(path_anchor_idx[path_ordinal]),
            "path_is_hot": bool(path_is_hot[path_ordinal]) if path_ordinal < len(path_is_hot) else None,
            "work_integral": (
                float(work_integral[path_ordinal])
                if path_ordinal < len(work_integral) and np.isfinite(work_integral[path_ordinal])
                else None
            ),
            "closed_loop": bool(closed_loop[path_ordinal]) if path_ordinal < len(closed_loop) else None,
            "n_points": int(len(coords)),
            "path_indices": [int(item[0]) for item in coords],
            "start_x": float(start[1]),
            "start_y": float(start[2]),
            "start_z": float(start[3]) if start[3] is not None else None,
            "end_x": float(end[1]),
            "end_y": float(end[2]),
            "end_z": float(end[3]) if end[3] is not None else None,
        }
        rows.append(row)
    return rows


def hydrate_legacy_path_ledger(run_dir: Path, *, overwrite: bool = False) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    observer_dirs = sorted(path for path in run_dir.glob("observer_*") if path.is_dir())
    rel_dir = run_dir / "relativity_cache"
    rows: List[Dict[str, Any]] = []
    for observer_dir in observer_dirs:
        try:
            observer_idx = int(observer_dir.name.split("_", 1)[1])
        except Exception:
            continue
        view_state_path = observer_dir / "MONOLITH.view_state.json"
        view_state = _load_json(view_state_path)
        if not view_state:
            rows.append({"observer_idx": observer_idx, "status": "skipped", "reason": "view_state_missing_or_unreadable"})
            continue
        existing_paths = view_state.get("walker_paths") if isinstance(view_state.get("walker_paths"), list) else []
        if existing_paths and not overwrite:
            rows.append(
                {
                    "observer_idx": observer_idx,
                    "status": "skipped",
                    "reason": "walker_paths_already_present",
                    "path_count": len(existing_paths),
                }
            )
            continue
        path_rows = _legacy_path_rows_for_observer(run_dir, observer_idx, _article_map(view_state))
        if not path_rows:
            rows.append({"observer_idx": observer_idx, "status": "skipped", "reason": "no_projectable_cyclic_paths"})
            continue
        view_state["walker_paths"] = path_rows
        provenance = view_state.get("path_ledger_provenance") if isinstance(view_state.get("path_ledger_provenance"), dict) else {}
        provenance.update(
            {
                "mode": "observer_view_state_legacy_path_ledger_hydration_v1",
                "source": "observer_view_state_legacy_path_ledger_hydration_v1",
                "path_row_source": "legacy_observer_view_state_article_coords",
                "cyclic_path_source": "cyclic_paths.npz:path_indices",
                "fresh_focused_observer_replay": False,
                "legacy_path_ledger_hydrated": True,
                "synthetic_placeholder": False,
                "warning": "Projected from legacy cyclic path article indices into observer-rendered coordinates.",
            }
        )
        view_state["path_ledger_provenance"] = provenance
        view_state_path.write_text(json.dumps(view_state, indent=2), encoding="utf-8")

        state_path = rel_dir / f"state_{observer_idx}.json"
        state_payload = _load_json(state_path)
        if state_payload:
            state_payload["walker_paths"] = path_rows
            state_payload["path_ledger_provenance"] = provenance
            state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "observer_idx": observer_idx,
                "status": "hydrated",
                "path_count": len(path_rows),
                "view_state": str(view_state_path),
                "relativity_state": str(state_path) if state_path.exists() else None,
            }
        )
    hydrated_count = sum(1 for row in rows if row.get("status") == "hydrated")
    return {
        "schema_version": "1.0",
        "summary_type": "legacy_observer_path_ledger_hydration",
        "status": "success" if hydrated_count else "skipped",
        "run_dir": str(run_dir),
        "hydrated_observer_count": hydrated_count,
        "observer_count": len(rows),
        "fresh_focused_observer_replay": False,
        "rows": rows,
    }


def build_backfill_plan(roots: Iterable[Path], *, recursive: bool, mode: str = "recenter_only_plan") -> Dict[str, Any]:
    root_paths = [Path(root) for root in roots]
    leaf_dirs: List[Path] = []
    for root in root_paths:
        for leaf in iter_leaf_dirs(root, recursive=recursive):
            if leaf not in leaf_dirs:
                leaf_dirs.append(leaf)
    leaves = [_plan_leaf(leaf) for leaf in leaf_dirs]
    action_counts: Dict[str, int] = {}
    for leaf in leaves:
        action = str(leaf.get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    mutating_count = sum(1 for leaf in leaves if bool(leaf.get("would_mutate")))
    return {
        "schema_version": "1.0",
        "summary_type": "observer_contract_backfill_plan",
        "status": "success" if leaves else "failed",
        "dry_run": True,
        "mode": mode,
        "recursive": bool(recursive),
        "roots": [str(root) for root in root_paths],
        "leaf_count": len(leaves),
        "mutating_action_count": int(mutating_count),
        "non_mutating_action_count": int(len(leaves) - mutating_count),
        "action_counts": dict(sorted(action_counts.items())),
        "leaves": leaves,
    }


def _observer_failure_reasons(observer: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not bool(observer.get("artifact_exists")):
        reasons.append("artifact_missing")
    if not bool(observer.get("view_state_exists")):
        reasons.append("view_state_missing")
    if not bool(observer.get("focus_xy_centered")):
        reasons.append("focus_not_centered")
    if not bool(observer.get("path_starts_match_articles")):
        reasons.append("path_start_mismatch")
    if not bool(observer.get("relativity_state_exists")):
        reasons.append("relativity_state_missing")
    if not bool(observer.get("relativity_delta_exists")):
        reasons.append("relativity_delta_missing")
    if observer.get("relativity_state_exists") and not bool(observer.get("relativity_state_has_walker_paths")):
        reasons.append("relativity_state_missing_walker_paths")
    return reasons


def _recenter_review_leaf(result: Dict[str, Any], *, summary_only: bool) -> Dict[str, Any]:
    run_dir = Path(str(result.get("run_dir") or "."))
    summary_path = Path(str(result.get("observer_recenter_summary") or run_dir / "observer_recenter_summary.json"))
    recenter = _load_json(summary_path)
    observers = recenter.get("observers") if isinstance(recenter.get("observers"), list) else []
    invalid_observers: List[Dict[str, Any]] = []
    for observer in observers:
        if not isinstance(observer, dict) or bool(observer.get("ok")):
            continue
        row = {
            "observer_idx": observer.get("observer_idx"),
            "focus_idx": observer.get("focus_idx"),
            "reasons": _observer_failure_reasons(observer),
        }
        if observer.get("path_mismatches"):
            row["path_mismatches"] = observer.get("path_mismatches")
        invalid_observers.append(row)
    legacy_path_hydration = result.get("legacy_path_hydration") if isinstance(result.get("legacy_path_hydration"), dict) else None
    legacy_hydrated = bool(
        legacy_path_hydration and int(legacy_path_hydration.get("hydrated_observer_count") or 0) > 0
    )
    replay_count = recenter.get("replay_path_observer_count", result.get("recenter_replay_path_observer_count"))
    try:
        fresh_replay_complete = bool(int(replay_count or 0) >= int(recenter.get("observer_count", result.get("recenter_observer_count")) or 0) > 0)
    except Exception:
        fresh_replay_complete = False

    leaf = {
        "run_dir": str(run_dir),
        "contract_status": result.get("status"),
        "mode": result.get("mode"),
        "observer_backfill": result.get("observer_backfill"),
        "observer_recenter_summary_path": str(summary_path),
        "observer_recenter_status": recenter.get("status") or result.get("recenter_status"),
        "observer_count": recenter.get("observer_count", result.get("recenter_observer_count")),
        "ok_count": recenter.get("ok_count", result.get("recenter_ok_count")),
        "focus_xy_centered_count": recenter.get(
            "focus_xy_centered_count",
            result.get("recenter_focus_xy_centered_count"),
        ),
        "path_start_match_observer_count": recenter.get(
            "path_start_match_observer_count",
            result.get("recenter_path_start_match_observer_count"),
        ),
        "replay_path_observer_count": recenter.get(
            "replay_path_observer_count",
            result.get("recenter_replay_path_observer_count"),
        ),
        "fresh_focused_observer_replay_complete": fresh_replay_complete,
        "legacy_path_ledger_hydrated": legacy_hydrated,
        "legacy_path_hydration": legacy_path_hydration,
        "z_origin_policy": recenter.get("z_origin_policy"),
        "artifact_inventory_path": result.get("leaf_artifact_inventory") or str(run_dir / "leaf_artifact_inventory.json"),
        "missing_or_invalid_observers": invalid_observers,
    }
    if not summary_only:
        leaf["observer_rows"] = observers
    return leaf


def build_recenter_review(result: Dict[str, Any], *, summary_only: bool = False) -> Dict[str, Any]:
    leaves = result.get("leaves") if isinstance(result.get("leaves"), list) else None
    leaf_reviews = [
        _recenter_review_leaf(leaf, summary_only=summary_only)
        for leaf in (leaves if leaves is not None else [result])
        if isinstance(leaf, dict)
    ]
    status_counts: Dict[str, int] = {}
    invalid_leaf_count = 0
    observer_count = 0
    ok_count = 0
    focus_xy_centered_count = 0
    path_start_match_observer_count = 0
    replay_path_observer_count = 0
    legacy_hydrated_leaf_count = 0
    fresh_replay_complete_leaf_count = 0
    for leaf in leaf_reviews:
        recenter_status = str(leaf.get("observer_recenter_status") or "MISSING")
        status_counts[recenter_status] = status_counts.get(recenter_status, 0) + 1
        if recenter_status != "OK" or leaf.get("observer_count") != leaf.get("ok_count"):
            invalid_leaf_count += 1
        if bool(leaf.get("legacy_path_ledger_hydrated")):
            legacy_hydrated_leaf_count += 1
        if bool(leaf.get("fresh_focused_observer_replay_complete")):
            fresh_replay_complete_leaf_count += 1
        try:
            observer_count += int(leaf.get("observer_count") or 0)
            ok_count += int(leaf.get("ok_count") or 0)
            focus_xy_centered_count += int(leaf.get("focus_xy_centered_count") or 0)
            path_start_match_observer_count += int(leaf.get("path_start_match_observer_count") or 0)
            replay_path_observer_count += int(leaf.get("replay_path_observer_count") or 0)
        except Exception:
            pass
    return {
        "schema_version": "1.0",
        "summary_type": "observer_recenter_review",
        "source_summary_type": result.get("summary_type"),
        "status": "OK" if leaf_reviews and invalid_leaf_count == 0 else "INVALID",
        "contract_status": result.get("status"),
        "mode": result.get("mode"),
        "roots": result.get("roots") or [leaf.get("run_dir") for leaf in leaf_reviews],
        "leaf_count": len(leaf_reviews),
        "valid_leaf_count": len(leaf_reviews) - invalid_leaf_count,
        "invalid_leaf_count": invalid_leaf_count,
        "observer_count": observer_count,
        "ok_count": ok_count,
        "focus_xy_centered_count": focus_xy_centered_count,
        "path_start_match_observer_count": path_start_match_observer_count,
        "replay_path_observer_count": replay_path_observer_count,
        "legacy_hydrated_leaf_count": legacy_hydrated_leaf_count,
        "fresh_replay_complete_leaf_count": fresh_replay_complete_leaf_count,
        "recenter_status_counts": dict(sorted(status_counts.items())),
        "leaves": leaf_reviews,
    }


def _materialize_leaf(
    run_dir: Path,
    *,
    recenter_only: bool,
    allow_observer_backfill: bool,
    observer_indices: Optional[Sequence[int]],
    observer_index_policy: str,
    observer_index_count: int,
    hydrate_legacy_paths: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    try:
        legacy_path_hydration = None
        if hydrate_legacy_paths:
            legacy_path_hydration = hydrate_legacy_path_ledger(run_dir)
        if recenter_only:
            summary_path = suite._emit_observer_recenter_summary_json(run_dir)
            result: Dict[str, Any] = {
                "status": "success",
                "mode": "recenter_only",
                "run_dir": str(run_dir),
                "observer_recenter_summary": str(summary_path),
            }
        else:
            result = suite.emit_consumer_contract_bundle(
                run_dir,
                allow_observer_backfill=allow_observer_backfill,
                observer_indices=observer_indices,
                observer_index_policy=observer_index_policy,
                observer_index_count=observer_index_count,
            )
            result = dict(result)
            result["mode"] = "consumer_contract_bundle"
            result["run_dir"] = str(run_dir)
        if legacy_path_hydration is not None:
            result["legacy_path_hydration"] = legacy_path_hydration
        result.update(_recenter_fields(run_dir))
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "mode": "recenter_only" if recenter_only else "consumer_contract_bundle",
            "run_dir": str(run_dir),
            "error": str(exc),
            **_recenter_fields(run_dir),
        }


def materialize_many(
    roots: Iterable[Path],
    *,
    recursive: bool,
    recenter_only: bool,
    allow_observer_backfill: bool,
    observer_indices: Optional[Sequence[int]],
    observer_index_policy: str,
    observer_index_count: int,
    isolate_copy_root: Optional[Path] = None,
    hydrate_legacy_paths: bool = False,
) -> Dict[str, Any]:
    root_paths = [Path(root) for root in roots]
    leaf_dirs: List[Path] = []
    root_by_leaf: Dict[Path, Path] = {}
    for root in root_paths:
        for leaf in iter_leaf_dirs(root, recursive=recursive):
            if leaf not in leaf_dirs:
                leaf_dirs.append(leaf)
                root_by_leaf[leaf] = root

    if not leaf_dirs:
        return {
            "schema_version": "1.0",
            "summary_type": "observer_contract_backfill",
            "status": "failed",
            "mode": "recenter_only" if recenter_only else "consumer_contract_bundle",
            "recursive": bool(recursive),
            "roots": [str(root) for root in root_paths],
            "leaf_count": 0,
            "success_count": 0,
            "failure_count": 1,
            "error": "no_leaf_dirs_found",
            "leaves": [],
        }

    copy_rows: List[Dict[str, Any]] = []
    materialize_dirs = leaf_dirs
    if isolate_copy_root is not None:
        copy_root = Path(isolate_copy_root)
        copy_root.mkdir(parents=True, exist_ok=True)
        materialize_dirs = []
        for leaf in leaf_dirs:
            root = root_by_leaf.get(leaf, leaf)
            rel = _isolated_leaf_relative_path(leaf, root, leaf_dirs)
            target = copy_root / rel
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(leaf, target)
            materialize_dirs.append(target)
            copy_rows.append({"source": str(leaf), "target": str(target)})

    leaves = [
        _materialize_leaf(
            leaf,
            recenter_only=recenter_only,
            allow_observer_backfill=allow_observer_backfill,
            observer_indices=observer_indices,
            observer_index_policy=observer_index_policy,
            observer_index_count=observer_index_count,
            hydrate_legacy_paths=hydrate_legacy_paths,
        )
        for leaf in materialize_dirs
    ]
    success_count = sum(1 for leaf in leaves if leaf.get("status") == "success")
    failure_count = len(leaves) - success_count
    recenter_counts: Dict[str, int] = {}
    for leaf in leaves:
        status = str(leaf.get("recenter_status") or "MISSING")
        recenter_counts[status] = recenter_counts.get(status, 0) + 1
    ok_recenter = int(recenter_counts.get("OK", 0))
    aggregate_status = "success" if failure_count == 0 else ("failed" if success_count == 0 else "partial")
    return {
        "schema_version": "1.0",
        "summary_type": "observer_contract_backfill",
        "status": aggregate_status,
        "mode": "recenter_only" if recenter_only else "consumer_contract_bundle",
        "recursive": bool(recursive),
        "isolation": {
            "enabled": isolate_copy_root is not None,
            "copy_root": str(isolate_copy_root) if isolate_copy_root is not None else None,
            "copied_leaf_count": len(copy_rows),
            "copies": copy_rows,
        },
        "roots": [str(root) for root in root_paths],
        "leaf_count": len(leaves),
        "success_count": success_count,
        "failure_count": failure_count,
        "recenter_status_counts": dict(sorted(recenter_counts.items())),
        "recenter_valid_leaf_count": ok_recenter,
        "all_recenter_valid": bool(ok_recenter == len(leaves)),
        "leaves": leaves,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+", help="Leaf directory, or root directory when --recursive is used.")
    parser.add_argument(
        "--no-observer-backfill",
        action="store_true",
        help="Refresh leaf contracts without materializing missing observer universes.",
    )
    parser.add_argument(
        "--observer-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional article indices to materialize/refresh; omitted means all observers.",
    )
    parser.add_argument(
        "--observer-index-policy",
        choices=["all", "anchors"],
        default="all",
        help="Observer selection policy when --observer-indices is omitted.",
    )
    parser.add_argument(
        "--observer-index-count",
        type=int,
        default=3,
        help="Number of observer indices selected by the anchors policy.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan each input directory for run leaves before backfilling.",
    )
    parser.add_argument(
        "--recenter-only",
        action="store_true",
        help="Only emit observer_recenter_summary.json; do not refresh the full consumer bundle.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only inspect leaves and emit a non-mutating observer recenter backfill plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --plan-only; inspect leaves without mutating artifacts.",
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="Optional path to write a non-mutating observer recenter backfill plan before any requested backfill.",
    )
    parser.add_argument(
        "--isolate-copy-root",
        default=None,
        help="Copy discovered leaves into this directory and backfill the copies instead of mutating source leaves.",
    )
    parser.add_argument(
        "--hydrate-legacy-path-ledger",
        action="store_true",
        help="Before recenter emission, project legacy cyclic path indices into observer view-state path ledgers.",
    )
    parser.add_argument(
        "--require-valid-recenter",
        action="store_true",
        help="Return nonzero when any emitted observer_recenter_summary.json is not OK.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the backfill result JSON.")
    parser.add_argument(
        "--observer-recenter-review-json",
        default=None,
        help="Optional path to write a compact observer recenter review JSON.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Keep observer recenter review compact by omitting full observer rows.",
    )
    parser.add_argument(
        "--require-recenter-ok",
        action="store_true",
        help="Alias for --require-valid-recenter.",
    )
    args = parser.parse_args(argv)

    require_valid_recenter = bool(args.require_valid_recenter or args.require_recenter_ok)

    plan_requested = bool(args.plan_only or args.dry_run)

    if plan_requested or args.plan_json:
        plan = build_backfill_plan(
            [Path(item) for item in args.run_dir],
            recursive=bool(args.recursive),
            mode="recenter_only_plan" if args.recenter_only else "consumer_contract_bundle_plan",
        )
        if args.plan_json:
            plan_path = Path(args.plan_json)
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
        if plan_requested:
            result = plan
            text = json.dumps(result, indent=2, default=str)
            print(text)
            if args.output_json:
                out = Path(args.output_json)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text + "\n", encoding="utf-8")
            return 0 if result.get("status") == "success" else 1

    if (
        len(args.run_dir) == 1
        and not args.recursive
        and not args.recenter_only
        and not require_valid_recenter
        and not args.isolate_copy_root
    ):
        run_dir = Path(args.run_dir[0])
        result = suite.emit_consumer_contract_bundle(
            run_dir,
            allow_observer_backfill=not bool(args.no_observer_backfill),
            observer_indices=args.observer_indices,
            observer_index_policy=args.observer_index_policy,
            observer_index_count=args.observer_index_count,
        )
        result = dict(result)
        result.setdefault("run_dir", str(run_dir))
        result.setdefault("mode", "consumer_contract_bundle")
        result.update(_recenter_fields(run_dir))
    else:
        result = materialize_many(
            [Path(item) for item in args.run_dir],
            recursive=bool(args.recursive),
            recenter_only=bool(args.recenter_only),
            allow_observer_backfill=not bool(args.no_observer_backfill),
            observer_indices=args.observer_indices,
            observer_index_policy=args.observer_index_policy,
            observer_index_count=args.observer_index_count,
            isolate_copy_root=Path(args.isolate_copy_root) if args.isolate_copy_root else None,
            hydrate_legacy_paths=bool(args.hydrate_legacy_path_ledger),
        )
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.observer_recenter_review_json:
        review = build_recenter_review(result, summary_only=bool(args.summary_only))
        review_path = Path(args.observer_recenter_review_json)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(review, indent=2, default=str) + "\n", encoding="utf-8")
    if require_valid_recenter and not bool(
        result.get("all_recenter_valid", result.get("recenter_status") == "OK")
    ):
        return 1
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
