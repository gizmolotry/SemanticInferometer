#!/usr/bin/env python3
"""Inspect focused proof bundle status without mutating output artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "outputs" / "experiments" / "runs"
DEFAULT_FOCUSED_ROOT = REPO_ROOT / "outputs" / "thesis_validation" / "focused"
PROOF_PROCESS_TOKENS = (
    "run_focused_proof_bundle.py",
    "run_full_experiment_suite.py",
    "run_experiments.py",
    "compare_controls.py",
    "run_track4_focused_basis_validation.py",
    "run_track4_observer_state_matrix.py",
    "run_track4_action_graph.py",
)
STALE_RUNNING_SECONDS = 3600


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def _as_datetime(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: object, *, now: Optional[datetime] = None) -> Optional[float]:
    parsed = _as_datetime(value)
    if parsed is None:
        return None
    return max(0.0, ((_utc_now() if now is None else now) - parsed).total_seconds())


def _path_or_none(raw: object) -> Optional[Path]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _iter_bundle_dirs(focused_root: Path) -> Iterable[Path]:
    if not focused_root.exists():
        return []
    return (path for path in focused_root.iterdir() if path.is_dir())


def _status_files(focused_root: Path) -> List[Path]:
    paths = [path / "focused_proof_status.json" for path in _iter_bundle_dirs(focused_root)]
    return [path for path in paths if path.exists()]


def _bundle_sort_key(path: Path) -> Tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except OSError:
        return (0.0, path.name)


def _latest_status_files(focused_root: Path, *, limit: int) -> List[Path]:
    return sorted(_status_files(focused_root), key=_bundle_sort_key, reverse=True)[: max(0, limit)]


def _load_current_marker(focused_root: Path) -> Dict[str, object]:
    marker_path = focused_root / "current_bundle.json"
    if not marker_path.exists():
        return {"path": str(marker_path), "present": False, "error": None, "payload": None}
    payload, error = _read_json(marker_path)
    return {
        "path": str(marker_path),
        "present": error is None,
        "error": error,
        "payload": payload if isinstance(payload, dict) else None,
    }


def _load_status(path: Path, *, repo_root: Path, now: Optional[datetime] = None) -> Dict[str, object]:
    payload, error = _read_json(path)
    if not isinstance(payload, dict):
        payload = {}
    bundle_dir = _path_or_none(payload.get("bundle_dir")) or path.parent
    status = str(payload.get("status") or "unknown")
    generated_at = payload.get("generated_at")
    age = _age_seconds(generated_at, now=now)
    return {
        "bundle_name": str(payload.get("bundle_name") or path.parent.name),
        "status": status,
        "stage": payload.get("stage"),
        "generated_at": generated_at,
        "age_seconds": age,
        "stale_hint": bool(status == "running" and age is not None and age > STALE_RUNNING_SECONDS),
        "bundle_dir": str(bundle_dir),
        "bundle_dir_display": _safe_relative(bundle_dir, repo_root),
        "status_path": str(path),
        "summary_path": payload.get("summary_path"),
        "runs": payload.get("runs") if isinstance(payload.get("runs"), list) else [],
        "evidence": payload.get("evidence"),
        "evidence_acceptance": payload.get("evidence_acceptance") if isinstance(payload.get("evidence_acceptance"), dict) else None,
        "orchestration_contract": (
            payload.get("orchestration_contract")
            if isinstance(payload.get("orchestration_contract"), dict)
            else None
        ),
        "error": error,
    }


def _infer_expected_from_command(command: object) -> Dict[str, object]:
    if not isinstance(command, list):
        return {"kernels": [], "channels": [], "corpora": [], "expected_leaf_count": None}
    tokens = [str(item) for item in command]

    def values_after(flag: str) -> List[str]:
        if flag not in tokens:
            return []
        values: List[str] = []
        for token in tokens[tokens.index(flag) + 1 :]:
            if token.startswith("--"):
                break
            values.append(token)
        return values

    kernels = values_after("--kernels")
    channels = values_after("--channels")
    corpora = values_after("--corpora")
    expected = len(kernels) * len(channels) * len(corpora) if kernels and channels and corpora else None
    return {
        "kernels": kernels,
        "channels": channels,
        "corpora": corpora,
        "expected_leaf_count": expected,
    }


def _count_run_artifacts(run_dir: Path) -> Dict[str, int]:
    counts = {
        "leaf_dirs": 0,
        "checkpoint_manifests": 0,
        "verification_reports": 0,
        "control_metrics": 0,
        "observer_view_states": 0,
        "observer_recenter_summaries": 0,
        "observer_cache_manifests": 0,
    }
    if not run_dir.exists():
        return counts
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name == "manifest.json" and path.parent.name == "batch":
            counts["checkpoint_manifests"] += 1
            if len(path.parts) >= 4:
                counts["leaf_dirs"] += 1
        elif name == "verification_report.json":
            counts["verification_reports"] += 1
        elif name == "control_metrics.json":
            counts["control_metrics"] += 1
        elif name == "MONOLITH.view_state.json":
            counts["observer_view_states"] += 1
        elif name == "observer_recenter_summary.json":
            counts["observer_recenter_summaries"] += 1
        elif name == "manifest.json" and "relativity_cache" in path.parts:
            counts["observer_cache_manifests"] += 1
    return counts


def _summarize_run(run_id: str, *, runs_root: Path, command: object = None) -> Dict[str, object]:
    run_dir = runs_root / run_id
    expected = _infer_expected_from_command(command)
    counts = _count_run_artifacts(run_dir)
    expected_leaf_count = expected.get("expected_leaf_count")
    completed_leaf_count = int(counts["checkpoint_manifests"])
    progress_pct = None
    if isinstance(expected_leaf_count, int) and expected_leaf_count > 0:
        progress_pct = min(100.0, round((completed_leaf_count / expected_leaf_count) * 100.0, 1))
    try:
        modified_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        modified_at = None
    return {
        "run_id": run_id,
        "path": str(run_dir),
        "exists": run_dir.exists(),
        "modified_at": modified_at,
        "expected": expected,
        "artifact_counts": counts,
        "progress": {
            "completed_leaf_count": completed_leaf_count,
            "expected_leaf_count": expected_leaf_count,
            "percent": progress_pct,
        },
    }


def _summarize_status_runs(status: Dict[str, object], *, runs_root: Path) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for item in status.get("runs") or []:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            summaries.append(
                {
                    "run_id": None,
                    "exists": False,
                    "track5_mode": item.get("track5_mode"),
                    "returncode": item.get("returncode"),
                    "progress": {"completed_leaf_count": 0, "expected_leaf_count": None, "percent": None},
                    "artifact_counts": _count_run_artifacts(Path("__missing__")),
                    "command_expected": _infer_expected_from_command(item.get("command")),
                }
            )
            continue
        summary = _summarize_run(run_id.strip(), runs_root=runs_root, command=item.get("command"))
        summary["track5_mode"] = item.get("track5_mode")
        summary["returncode"] = item.get("returncode")
        summary["disk_preflight"] = item.get("disk_preflight")
        summaries.append(summary)
    return summaries


def _status_run_ids(status: Dict[str, object]) -> List[str]:
    run_ids: List[str] = []
    for item in status.get("runs") or []:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            run_ids.append(run_id.strip())
    return run_ids


def _has_matching_process(status: Dict[str, object], process_scan: Dict[str, object]) -> Optional[bool]:
    if not process_scan.get("available"):
        return None
    probes = [
        str(status.get("bundle_name") or ""),
        str(status.get("bundle_dir") or ""),
        *_status_run_ids(status),
    ]
    probes = [probe for probe in probes if probe.strip()]
    for process in process_scan.get("processes") or []:
        if not isinstance(process, dict):
            continue
        command_line = str(process.get("command_line") or "")
        if any(probe in command_line for probe in probes):
            return True
    return False


def _has_recent_run_activity(
    status: Dict[str, object],
    *,
    now: datetime,
    stale_threshold_seconds: int,
) -> Optional[bool]:
    run_progress = status.get("run_progress") if isinstance(status.get("run_progress"), list) else []
    if not run_progress:
        return None
    for run in run_progress:
        if not isinstance(run, dict) or not run.get("exists"):
            continue
        modified_age = _age_seconds(run.get("modified_at"), now=now)
        if modified_age is not None and modified_age <= stale_threshold_seconds:
            return True
    return False


def _classify_running_status(
    status: Dict[str, object],
    *,
    process_scan: Dict[str, object],
    now: datetime,
    stale_threshold_seconds: int,
) -> None:
    if status.get("status") != "running":
        status["running_classification"] = None
        status["running_classification_reasons"] = []
        return

    age = status.get("age_seconds")
    stale_by_age = isinstance(age, (int, float)) and age > stale_threshold_seconds
    matching_process = _has_matching_process(status, process_scan)
    recent_run_activity = _has_recent_run_activity(
        status,
        now=now,
        stale_threshold_seconds=stale_threshold_seconds,
    )
    orphaned = matching_process is False and recent_run_activity is not True

    reasons: List[str] = []
    if stale_by_age:
        reasons.append("status_file_older_than_stale_threshold")
    if orphaned:
        reasons.append("no_matching_active_process_or_recent_run_activity")

    if stale_by_age and orphaned:
        classification = "stale_orphan"
    elif stale_by_age:
        classification = "stale"
    elif orphaned:
        classification = "orphan"
    else:
        classification = "active"

    status["stale_hint"] = bool(stale_by_age)
    status["orphan_hint"] = bool(orphaned)
    status["running_classification"] = classification
    status["running_classification_reasons"] = reasons
    status["activity"] = {
        "matching_active_process": matching_process,
        "recent_run_activity": recent_run_activity,
        "stale_threshold_seconds": stale_threshold_seconds,
    }


def _latest_run_summaries(runs_root: Path, *, limit: int) -> List[Dict[str, object]]:
    if not runs_root.exists():
        return []
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("experiments_")]
    recent = sorted(run_dirs, key=_bundle_sort_key, reverse=True)[: max(0, limit)]
    return [_summarize_run(path.name, runs_root=runs_root) for path in recent]


def _extract_acceptance(bundle_dir: Path, status: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if status and isinstance(status.get("evidence_acceptance"), dict):
        return dict(status["evidence_acceptance"])  # type: ignore[index]
    summary_path = bundle_dir / "focused_proof_bundle.json"
    if summary_path.exists():
        payload, _ = _read_json(summary_path)
        if isinstance(payload, dict) and isinstance(payload.get("evidence_acceptance"), dict):
            return dict(payload["evidence_acceptance"])  # type: ignore[index]
    marker_path = bundle_dir / "inactive_bundle_marker.json"
    if marker_path.exists():
        payload, _ = _read_json(marker_path)
        if isinstance(payload, dict) and isinstance(payload.get("evidence_acceptance"), dict):
            return dict(payload["evidence_acceptance"])  # type: ignore[index]
    claim_path = bundle_dir / "claim_matrix.json"
    if not claim_path.exists():
        return {"mechanically_usable": False, "safe_for_focused_defense": False, "reason": "claim_matrix_missing"}
    payload, error = _read_json(claim_path)
    if error or not isinstance(payload, dict):
        return {"mechanically_usable": False, "safe_for_focused_defense": False, "reason": "claim_matrix_unreadable", "error": error}
    claims = [claim for claim in payload.get("claims", []) if isinstance(claim, dict)]
    unsafe = [str(claim.get("claim_id") or "unknown") for claim in claims if not bool(claim.get("thesis_safe"))]
    return {
        "mechanically_usable": bool(claims),
        "claim_count": len(claims),
        "unsafe_claim_ids": unsafe,
        "safe_for_thesis_claims": bool(claims) and not unsafe,
        "safe_for_focused_defense": bool(claims) and not unsafe,
    }


def _disk_hint(path: Path) -> Dict[str, object]:
    try:
        usage = shutil.disk_usage(path)
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    free_gb = usage.free / float(1024**3)
    total_gb = usage.total / float(1024**3)
    return {
        "available": True,
        "path": str(path),
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "used_pct": round(((usage.total - usage.free) / usage.total) * 100.0, 1) if usage.total else None,
        "low_free_hint": free_gb < 8.0,
    }


def _scan_processes() -> Dict[str, object]:
    if os.name == "nt":
        token_pattern = "|".join(PROOF_PROCESS_TOKENS)
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                f"Where-Object {{ $_.ProcessId -ne $PID -and $_.Name -match 'python' -and $_.CommandLine -match '{token_pattern}' }} | "
                "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
            ),
        ]
    else:
        cmd = ["ps", "-eo", "pid=,comm=,args="]
    try:
        proc = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=8)
    except Exception as exc:
        return {"available": False, "error": str(exc), "processes": []}
    if proc.returncode != 0:
        return {"available": False, "error": (proc.stderr or proc.stdout).strip(), "processes": []}
    processes: List[Dict[str, object]] = []
    if os.name == "nt":
        raw = proc.stdout.strip()
        if raw:
            try:
                payload = json.loads(raw)
                rows = payload if isinstance(payload, list) else [payload]
                for row in rows:
                    if isinstance(row, dict):
                        processes.append(
                            {
                                "pid": row.get("ProcessId"),
                                "name": row.get("Name"),
                                "command_line": row.get("CommandLine"),
                            }
                        )
            except json.JSONDecodeError as exc:
                return {"available": False, "error": f"process JSON parse failed: {exc}", "processes": []}
    else:
        for line in proc.stdout.splitlines():
            if not any(token in line for token in PROOF_PROCESS_TOKENS):
                continue
            parts = line.strip().split(None, 2)
            if parts:
                processes.append({"pid": parts[0], "name": parts[1] if len(parts) > 1 else None, "command_line": parts[2] if len(parts) > 2 else line.strip()})
    return {"available": True, "processes": processes, "count": len(processes)}


def _summarize_track4_replay(root: Optional[Path]) -> Dict[str, object]:
    if root is None:
        return {"present": False, "reason": "not_requested"}
    root = Path(root)
    if not root.exists():
        return {"present": False, "path": str(root), "reason": "missing_root"}
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from scripts.track4_replay_status import summarize_replay_root

        payload = summarize_replay_root(root)
    except Exception as exc:
        return {
            "present": False,
            "path": str(root),
            "reason": "status_failed",
            "error": str(exc),
        }
    return {
        "present": True,
        "path": str(root),
        "complete": bool(payload.get("complete")),
        "completed_count": payload.get("completed_count"),
        "expected_count": payload.get("expected_count"),
        "missing_count": payload.get("missing_count"),
        "completion_ratio": payload.get("completion_ratio"),
        "last_completed_summary": payload.get("last_completed_summary"),
        "minutes_since_last_completion": payload.get("minutes_since_last_completion"),
        "inventory_warning": payload.get("inventory_warning"),
        "by_group": payload.get("by_group"),
    }


def build_status_report(
    *,
    repo_root: Path = REPO_ROOT,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    focused_root: Path = DEFAULT_FOCUSED_ROOT,
    limit: int = 5,
    include_processes: bool = True,
    now: Optional[datetime] = None,
    stale_threshold_seconds: int = STALE_RUNNING_SECONDS,
    track4_replay_root: Optional[Path] = None,
) -> Dict[str, object]:
    now_dt = now or _utc_now()
    statuses = [_load_status(path, repo_root=repo_root, now=now_dt) for path in _latest_status_files(focused_root, limit=limit)]
    for status in statuses:
        status["run_progress"] = _summarize_status_runs(status, runs_root=runs_root)
        status["acceptance"] = _extract_acceptance(Path(str(status["bundle_dir"])), status)
    process_scan = _scan_processes() if include_processes else {"available": False, "skipped": True, "processes": []}
    for status in statuses:
        _classify_running_status(
            status,
            process_scan=process_scan,
            now=now_dt,
            stale_threshold_seconds=stale_threshold_seconds,
        )
    running = [status for status in statuses if status.get("status") == "running"]
    active_running = [status for status in running if status.get("running_classification") == "active"]
    stale_running = [status for status in running if status.get("running_classification") in {"stale", "stale_orphan"}]
    orphan_running = [status for status in running if status.get("running_classification") in {"orphan", "stale_orphan"}]
    current = _load_current_marker(focused_root)
    current_payload = current.get("payload") if isinstance(current.get("payload"), dict) else {}
    current_acceptance = current_payload.get("evidence_acceptance") if isinstance(current_payload, dict) else None
    report = {
        "schema_version": "1.0",
        "generated_at": now_dt.isoformat(),
        "repo_root": str(repo_root),
        "runs_root": str(runs_root),
        "focused_root": str(focused_root),
        "active": {
            "stale_threshold_seconds": stale_threshold_seconds,
            "running_status_count": len(running),
            "active_running_status_count": len(active_running),
            "stale_running_status_count": len(stale_running),
            "orphan_running_status_count": len(orphan_running),
            "running_bundles": [
                {
                    "bundle_name": status.get("bundle_name"),
                    "stage": status.get("stage"),
                    "age_seconds": status.get("age_seconds"),
                    "stale_hint": status.get("stale_hint"),
                    "orphan_hint": status.get("orphan_hint"),
                    "running_classification": status.get("running_classification"),
                    "bundle_dir": status.get("bundle_dir"),
                }
                for status in active_running
            ],
            "stale_running_bundles": [
                {
                    "bundle_name": status.get("bundle_name"),
                    "stage": status.get("stage"),
                    "age_seconds": status.get("age_seconds"),
                    "orphan_hint": status.get("orphan_hint"),
                    "running_classification": status.get("running_classification"),
                    "reasons": status.get("running_classification_reasons"),
                    "bundle_dir": status.get("bundle_dir"),
                }
                for status in stale_running
            ],
            "orphan_running_bundles": [
                {
                    "bundle_name": status.get("bundle_name"),
                    "stage": status.get("stage"),
                    "age_seconds": status.get("age_seconds"),
                    "stale_hint": status.get("stale_hint"),
                    "running_classification": status.get("running_classification"),
                    "reasons": status.get("running_classification_reasons"),
                    "bundle_dir": status.get("bundle_dir"),
                }
                for status in orphan_running
            ],
            "process_scan": process_scan,
        },
        "current_bundle": {
            "path": current.get("path"),
            "present": current.get("present"),
            "error": current.get("error"),
            "bundle_name": current_payload.get("bundle_name") if isinstance(current_payload, dict) else None,
            "status": current_payload.get("status") if isinstance(current_payload, dict) else None,
            "run_ids": current_payload.get("run_ids") if isinstance(current_payload, dict) else [],
            "preferred_run_id": current_payload.get("preferred_run_id") if isinstance(current_payload, dict) else None,
            "evidence_dir": current_payload.get("evidence_dir") if isinstance(current_payload, dict) else None,
            "evidence_acceptance": current_acceptance,
        },
        "focused_statuses": statuses,
        "recent_runs": _latest_run_summaries(runs_root, limit=limit),
        "track4_replay": _summarize_track4_replay(track4_replay_root),
        "memory_hints": {
            "repo_disk": _disk_hint(repo_root),
            "runs_disk": _disk_hint(runs_root if runs_root.exists() else repo_root),
            "note": "Disk pressure is the available stdlib signal; process RSS is intentionally not required.",
        },
    }
    return report


def _format_bool(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def format_human(report: Dict[str, object]) -> str:
    lines: List[str] = []
    active = report.get("active") if isinstance(report.get("active"), dict) else {}
    process_scan = active.get("process_scan") if isinstance(active, dict) and isinstance(active.get("process_scan"), dict) else {}
    lines.append("Focused Proof Status")
    lines.append(f"Generated: {report.get('generated_at')}")
    active_count = active.get("active_running_status_count", active.get("running_status_count", 0)) if isinstance(active, dict) else 0
    stale_count = active.get("stale_running_status_count", 0) if isinstance(active, dict) else 0
    orphan_count = active.get("orphan_running_status_count", 0) if isinstance(active, dict) else 0
    lines.append(f"Active status files: {active_count}")
    if stale_count:
        lines.append(f"Stale running status files: {stale_count}")
    if orphan_count:
        lines.append(f"Orphan running status files: {orphan_count}")
    if process_scan.get("available"):
        lines.append(f"Active proof processes: {process_scan.get('count', len(process_scan.get('processes', [])))}")
    elif process_scan.get("skipped"):
        lines.append("Active proof processes: skipped")
    else:
        lines.append(f"Active proof processes: unavailable ({process_scan.get('error', 'unknown error')})")

    running = active.get("running_bundles") if isinstance(active, dict) else []
    if running:
        lines.append("")
        lines.append("Running Bundles:")
        for bundle in running:
            if not isinstance(bundle, dict):
                continue
            lines.append(f"- {bundle.get('bundle_name')} stage={bundle.get('stage')} age={bundle.get('age_seconds')}s")

    stale_running = active.get("stale_running_bundles") if isinstance(active, dict) else []
    if stale_running:
        lines.append("")
        lines.append("Stale Running Status Files:")
        for bundle in stale_running:
            if not isinstance(bundle, dict):
                continue
            reasons = bundle.get("reasons") if isinstance(bundle.get("reasons"), list) else []
            reason_text = f" reasons={','.join(str(reason) for reason in reasons)}" if reasons else ""
            lines.append(
                f"- {bundle.get('bundle_name')} stage={bundle.get('stage')} "
                f"classification={bundle.get('running_classification')} age={bundle.get('age_seconds')}s{reason_text}"
            )

    orphan_running = active.get("orphan_running_bundles") if isinstance(active, dict) else []
    if orphan_running:
        lines.append("")
        lines.append("Orphan Running Status Files:")
        for bundle in orphan_running:
            if not isinstance(bundle, dict):
                continue
            reasons = bundle.get("reasons") if isinstance(bundle.get("reasons"), list) else []
            reason_text = f" reasons={','.join(str(reason) for reason in reasons)}" if reasons else ""
            lines.append(
                f"- {bundle.get('bundle_name')} stage={bundle.get('stage')} "
                f"classification={bundle.get('running_classification')} age={bundle.get('age_seconds')}s{reason_text}"
            )

    current = report.get("current_bundle") if isinstance(report.get("current_bundle"), dict) else {}
    acceptance = current.get("evidence_acceptance") if isinstance(current.get("evidence_acceptance"), dict) else {}
    lines.append("")
    lines.append("Current Bundle:")
    lines.append(f"- name: {current.get('bundle_name')}")
    lines.append(f"- status: {current.get('status')}")
    lines.append(f"- preferred run: {current.get('preferred_run_id')}")
    lines.append(f"- focused defense safe: {_format_bool(acceptance.get('safe_for_focused_defense'))}")
    if acceptance.get("claim_profile"):
        lines.append(f"- claim profile: {acceptance.get('claim_profile')}")
    unsafe = acceptance.get("unsafe_focused_claim_ids") or acceptance.get("unsafe_claim_ids") or []
    if unsafe:
        lines.append(f"- unsafe claims: {', '.join(str(item) for item in unsafe)}")

    statuses = report.get("focused_statuses") if isinstance(report.get("focused_statuses"), list) else []
    if statuses:
        lines.append("")
        lines.append("Recent Bundle Status:")
        for status in statuses:
            if not isinstance(status, dict):
                continue
            classification = status.get("running_classification")
            classification_text = f" classification={classification}" if classification else ""
            lines.append(
                f"- {status.get('bundle_name')}: {status.get('status')} stage={status.get('stage')}"
                f"{classification_text} path={status.get('bundle_dir_display')}"
            )
            contract = status.get("orchestration_contract") if isinstance(status.get("orchestration_contract"), dict) else {}
            contract_nodes = contract.get("nodes") if isinstance(contract.get("nodes"), list) else []
            if contract_nodes:
                brief = ", ".join(
                    f"{node.get('node_id')}={node.get('status')}"
                    for node in contract_nodes
                    if isinstance(node, dict)
                )
                if brief:
                    lines.append(f"  dag: {brief}")
            for run in status.get("run_progress") or []:
                if not isinstance(run, dict):
                    continue
                progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
                percent = progress.get("percent")
                expected = progress.get("expected_leaf_count")
                completed = progress.get("completed_leaf_count")
                label = f"{completed}/{expected}" if expected is not None else str(completed)
                percent_text = f" ({percent}%)" if percent is not None else ""
                lines.append(f"  run {run.get('run_id')}: leaves={label}{percent_text} reports={run.get('artifact_counts', {}).get('verification_reports') if isinstance(run.get('artifact_counts'), dict) else '?'}")

    recent_runs = report.get("recent_runs") if isinstance(report.get("recent_runs"), list) else []
    if recent_runs:
        lines.append("")
        lines.append("Recent Run Artifacts:")
        for run in recent_runs[:3]:
            if not isinstance(run, dict):
                continue
            progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
            counts = run.get("artifact_counts") if isinstance(run.get("artifact_counts"), dict) else {}
            lines.append(
                f"- {run.get('run_id')}: leaves={progress.get('completed_leaf_count')} "
                f"reports={counts.get('verification_reports')} controls={counts.get('control_metrics')} "
                f"observers={counts.get('observer_view_states')} recenter={counts.get('observer_recenter_summaries')}"
            )

    track4 = report.get("track4_replay") if isinstance(report.get("track4_replay"), dict) else {}
    if track4.get("present"):
        completed = track4.get("completed_count")
        expected = track4.get("expected_count")
        ratio = track4.get("completion_ratio")
        ratio_text = f" ({round(float(ratio) * 100.0, 1)}%)" if isinstance(ratio, (int, float)) else ""
        lines.append("")
        lines.append("Track 4 Replay:")
        lines.append(f"- complete: {_format_bool(track4.get('complete'))}")
        lines.append(f"- cells: {completed}/{expected}{ratio_text}")
        lines.append(f"- missing: {track4.get('missing_count')}")
        lines.append(f"- last completed: {track4.get('last_completed_summary')}")
        if track4.get("inventory_warning"):
            lines.append(f"- warning: {track4.get('inventory_warning')}")

    memory = report.get("memory_hints") if isinstance(report.get("memory_hints"), dict) else {}
    repo_disk = memory.get("repo_disk") if isinstance(memory.get("repo_disk"), dict) else {}
    lines.append("")
    lines.append("Memory/Disk Hints:")
    if repo_disk.get("available"):
        low = " LOW" if repo_disk.get("low_free_hint") else ""
        lines.append(f"- repo disk free: {repo_disk.get('free_gb')}GB / {repo_disk.get('total_gb')}GB ({repo_disk.get('used_pct')}% used){low}")
    else:
        lines.append(f"- repo disk: unavailable ({repo_disk.get('error')})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runs-root", type=Path, default=None, help="Defaults to <repo-root>/outputs/experiments/runs.")
    parser.add_argument("--focused-root", type=Path, default=None, help="Defaults to <repo-root>/outputs/thesis_validation/focused.")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent bundles and runs to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable summary.")
    parser.add_argument("--no-process-scan", action="store_true", help="Skip best-effort proof process inspection.")
    parser.add_argument("--track4-replay-root", type=Path, default=None, help="Optional Track 4 replay root to include.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    runs_root = (args.runs_root or repo_root / "outputs" / "experiments" / "runs").resolve()
    focused_root = (args.focused_root or repo_root / "outputs" / "thesis_validation" / "focused").resolve()
    report = build_status_report(
        repo_root=repo_root,
        runs_root=runs_root,
        focused_root=focused_root,
        limit=args.limit,
        include_processes=not args.no_process_scan,
        track4_replay_root=args.track4_replay_root,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_human(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
