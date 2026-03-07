#!/usr/bin/env python3
"""
MONOLITH Artifact Command Center

Purpose:
- Browse pre-rendered MONOLITH HTML artifacts by run/observer.
- Compare two artifact variants side-by-side.
- Sync verification telemetry from harness outputs.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from urllib.parse import parse_qs
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback_context, dcc, html
import plotly.graph_objects as go
try:
    from analysis.verification.contract import (
        LayerStatus,
        REQUIRED_CONSUMER_ARTIFACTS,
        OPTIONAL_CONSUMER_ARTIFACTS,
        REQUIRED_PROVENANCE_KEYS,
        evaluate_consumer_contract,
        resolve_run_directory,
    )
except Exception:
    from verification.contract import (
        LayerStatus,
        REQUIRED_CONSUMER_ARTIFACTS,
        OPTIONAL_CONSUMER_ARTIFACTS,
        REQUIRED_PROVENANCE_KEYS,
        evaluate_consumer_contract,
        resolve_run_directory,
    )


PALETTE = {
    "void": "#050505",
    "panel": "#0b0b14",
    "grid": "#1a1a2e",
    "cyan": "#00F0FF",
    "green": "#00FF41",
    "red": "#FF2A00",
    "amber": "#FFB347",
    "text": "#E0E0E0",
    "dim": "#8A8A8A",
}

ROOT = REPO_ROOT


def _is_run_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if list(path.glob("MONOLITH*.html")):
        return True
    return (path / "MONOLITH_DATA.csv").exists()


def _discover_artifact_roots() -> List[Path]:
    explicit = ROOT / "experiments_20260221_175416" / "synthetic"
    patterns = (
        "experiments_*/synthetic",
        "experiments/experiments_*/synthetic",
        "outputs/experiments/runs/experiments_*/synthetic",
        "outputs/experiments/runs/*/synthetic",
        "outputs/experiments/*/synthetic",
    )
    seen = set()
    candidate_roots: List[Path] = []
    for pattern in patterns:
        for synthetic in ROOT.glob(pattern):
            key = str(synthetic.resolve())
            if key not in seen and synthetic.exists():
                seen.add(key)
                candidate_roots.append(synthetic)
    if explicit.exists():
        key = str(explicit.resolve())
        if key not in seen:
            candidate_roots.append(explicit)
    viable = [p for p in candidate_roots if any(_is_run_directory(d) for d in p.iterdir() if d.is_dir())]
    viable.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return viable


ARTIFACT_ROOTS = _discover_artifact_roots()
PRIMARY_ARTIFACT_ROOT = ARTIFACT_ROOTS[0] if ARTIFACT_ROOTS else None
TRACK_MARKERS = {
    "T1": ["track 1", "logit", "confidence halo"],
    "T1.5": ["track 1.5", "spectral", "rupture"],
    "T2": ["track 2", "terrain", "hologram"],
    "T3": ["track 3", "fog", "dirichlet"],
    "T4": ["track 4", "walker", "surv"],
    "T5": ["track 5", "phantom", "tautology", "honest"],
    "T6": ["track 6", "hott", "proof"],
}
VERIFICATION_STATUSES = {s.value for s in LayerStatus}


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_json(path: Path, default):
    try:
        return json.loads(_safe_read_text(path))
    except Exception:
        return default


def _is_number(val) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def _run_contract_paths(run_dir: Optional[Path]) -> Dict[str, Optional[Path]]:
    if not run_dir or not run_dir.exists():
        return {
            "baseline_meta": None,
            "baseline_state": None,
            "verification_report": None,
            "verification_summary": None,
            "hidden_groups": None,
            "group_summaries": None,
            "group_matrix": None,
        }

    labels_dir = run_dir / "labels"
    derived_dir = labels_dir / "derived"
    candidates = {
        "baseline_meta": [run_dir / "baseline_meta.json"],
        "baseline_state": [run_dir / "baseline_state.json"],
        "verification_report": [
            run_dir / "verification_report.json",
            run_dir / "verification" / "verification_report.json",
        ],
        "verification_summary": [
            run_dir / "verification_summary.csv",
            run_dir / "verification" / "verification_summary.csv",
        ],
        "hidden_groups": [labels_dir / "hidden_groups.csv", run_dir / "hidden_groups.csv"],
        "group_summaries": [derived_dir / "group_summaries.json", run_dir / "group_summaries.json"],
        "group_matrix": [derived_dir / "group_matrix.json", run_dir / "group_matrix.json"],
    }

    out: Dict[str, Optional[Path]] = {}
    for key, paths in candidates.items():
        found = None
        for p in paths:
            if p and p.exists():
                found = p
                break
        out[key] = found
    return out


def _validate_provenance(meta: dict) -> List[str]:
    if not isinstance(meta, dict):
        return ["baseline_meta is not a JSON object"]
    missing = [k for k in sorted(REQUIRED_PROVENANCE_KEYS) if k not in meta]
    errors: List[str] = []
    if missing:
        errors.append(f"baseline_meta missing keys: {', '.join(missing)}")
    status = str(meta.get("verification_status", "")).upper()
    if status and status not in VERIFICATION_STATUSES:
        errors.append(f"baseline_meta.verification_status invalid: {status}")
    return errors


def _validate_baseline_state(blob: dict) -> List[str]:
    if not isinstance(blob, dict):
        return ["baseline_state is not a JSON object"]
    errors: List[str] = []
    for field in ("articles", "paths", "axes", "metrics"):
        if field not in blob:
            errors.append(f"baseline_state missing '{field}'")
    if "articles" in blob and not isinstance(blob.get("articles"), list):
        errors.append("baseline_state.articles must be a list")
    if "paths" in blob and not isinstance(blob.get("paths"), list):
        errors.append("baseline_state.paths must be a list")
    return errors


def _validate_observer_state(blob: dict, observer_id: int) -> List[str]:
    if not isinstance(blob, dict):
        return ["observer state is not a JSON object"]
    errors: List[str] = []
    for field in ("observer_id", "articles", "paths", "axes", "metrics", "provenance"):
        if field not in blob:
            errors.append(f"state missing '{field}'")
    if "observer_id" in blob:
        try:
            obs = int(blob.get("observer_id"))
            if obs != observer_id:
                errors.append(f"state observer_id mismatch: expected {observer_id}, got {obs}")
        except Exception:
            errors.append("state observer_id is not an integer")
    return errors


def _validate_observer_delta(blob: dict, observer_id: int) -> List[str]:
    if not isinstance(blob, dict):
        return ["observer delta is not a JSON object"]
    errors: List[str] = []
    for field in (
        "observer_id",
        "null_observer_equivalence",
        "path_flip_delta",
        "metrics_delta",
        "axis_delta",
    ):
        if field not in blob:
            errors.append(f"delta missing '{field}'")
    if "observer_id" in blob:
        try:
            obs = int(blob.get("observer_id"))
            if obs != observer_id:
                errors.append(f"delta observer_id mismatch: expected {observer_id}, got {obs}")
        except Exception:
            errors.append("delta observer_id is not an integer")
    return errors


def _read_hidden_groups(path: Optional[Path]) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    errors: List[str] = []
    if not path or not path.exists():
        return rows, ["hidden_groups.csv missing"]
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            required = {"article_id", "group_topic"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                errors.append("hidden_groups.csv missing required columns article_id, group_topic")
                return rows, errors
            for row in reader:
                rows.append(dict(row))
    except Exception as exc:
        errors.append(f"failed to parse hidden_groups.csv: {exc}")
    return rows, errors


def _validate_group_summaries(path: Optional[Path]) -> Tuple[dict, List[str]]:
    data = _safe_json(path, {}) if path and path.exists() else {}
    errors: List[str] = []
    if not data:
        return {}, ["group_summaries.json missing"]
    groups = data.get("groups")
    if not isinstance(groups, list):
        errors.append("group_summaries.json must contain list field 'groups'")
        return data, errors
    for g in groups:
        if not isinstance(g, dict):
            errors.append("group_summaries.groups entries must be objects")
            continue
        if "group_name" not in g:
            errors.append("group_summaries entry missing group_name")
        if "n_articles" not in g or not _is_number(g.get("n_articles")):
            errors.append("group_summaries entry missing numeric n_articles")
    return data, errors


def _validate_group_matrix(path: Optional[Path]) -> Tuple[dict, List[str]]:
    data = _safe_json(path, {}) if path and path.exists() else {}
    errors: List[str] = []
    if not data:
        return {}, ["group_matrix.json missing"]
    groups = data.get("groups")
    matrix = data.get("cost_matrix")
    if not isinstance(groups, list) or not groups:
        errors.append("group_matrix.groups must be a non-empty list")
        return data, errors
    if not isinstance(matrix, list):
        errors.append("group_matrix.cost_matrix must be a list")
        return data, errors
    n = len(groups)
    if len(matrix) != n:
        errors.append(f"group_matrix rows mismatch: expected {n}, got {len(matrix)}")
        return data, errors
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            errors.append(f"group_matrix row {i} has invalid width")
            continue
        for val in row:
            if not _is_number(val):
                errors.append(f"group_matrix row {i} contains non-numeric value")
                break
    return data, errors


def _observer_id_from_value(observer_value: str) -> Optional[int]:
    if not observer_value or not str(observer_value).startswith("article:"):
        return None
    try:
        return int(str(observer_value).split(":", 1)[1])
    except Exception:
        return None


def load_contract_state(run_key: Optional[str], observer_value: str) -> dict:
    run_dir = _resolve_run_dir(run_key)
    # Smart Discovery: Try to resolve deep nesting (rbf/cls/real) if path doesn't exist
    if run_dir and not run_dir.exists():
        # Heuristic: try to infer kernel/channel from run_key if it looks like experiments_*/real
        # For Dash, we'll try a common default if it's missing.
        run_dir = resolve_run_directory(run_dir.parent, "rbf", "cls", run_dir.name)

    if not run_dir or not run_dir.exists():
        return {
            "status": "INVALID_SCHEMA",
            "errors": [f"run_dir missing for run_key={run_key}"],
            "missing_required_artifacts": list(REQUIRED_CONSUMER_ARTIFACTS),
            "missing_optional_artifacts": list(OPTIONAL_CONSUMER_ARTIFACTS),
            "schema_errors": [f"run_dir missing for run_key={run_key}"],
            "paths": {},
            "baseline_meta": {},
            "baseline_state": {},
            "observer_state": {},
            "observer_delta": {},
            "hidden_groups": [],
            "group_summaries": {},
            "group_matrix": {},
        }

    diag = evaluate_consumer_contract(run_dir)
    paths = {k: (str(v) if v else "NOT FOUND") for k, v in diag.paths.items()}
    errors: List[str] = list(diag.schema_errors)
    missing_required = list(diag.missing_required_artifacts)
    missing_optional = list(diag.missing_optional_artifacts)

    contract_path = run_dir / "EPISTEMIC_CONTRACT.json"
    if contract_path.exists():
        contract_blob = _safe_json(contract_path, {})
        baseline_meta = contract_blob.get("provenance", {}) if isinstance(contract_blob, dict) else {}
        baseline_state = _safe_json(diag.paths["baseline_state.json"], {}) if diag.paths.get("baseline_state.json") else {}
    else:
        baseline_meta = _safe_json(diag.paths["baseline_meta.json"], {}) if diag.paths.get("baseline_meta.json") else {}
        baseline_state = _safe_json(diag.paths["baseline_state.json"], {}) if diag.paths.get("baseline_state.json") else {}

    observer_id = _observer_id_from_value(observer_value)
    state_blob = {}
    delta_blob = {}
    if observer_id is not None:
        rel_dir = run_dir / "relativity_cache"
        state_path = rel_dir / f"state_{observer_id}.json"
        delta_path = rel_dir / f"delta_{observer_id}.json"
        if state_path.exists():
            state_blob = _safe_json(state_path, {})
        else:
            missing_optional.append(f"relativity_cache/state_{observer_id}.json")
        if delta_path.exists():
            delta_blob = _safe_json(delta_path, {})
        else:
            missing_optional.append(f"relativity_cache/delta_{observer_id}.json")

    hidden_path = diag.paths.get("labels/hidden_groups.csv")
    hidden_rows, hidden_errors = _read_hidden_groups(hidden_path)
    if hidden_path is None:
        hidden_errors = []

    gs_path = diag.paths.get("labels/derived/group_summaries.json")
    gm_path = diag.paths.get("labels/derived/group_matrix.json")
    group_summaries, gs_errors = _validate_group_summaries(gs_path)
    group_matrix, gm_errors = _validate_group_matrix(gm_path)
    if gs_path is None:
        gs_errors = []
    if gm_path is None:
        gm_errors = []

    # Optional artifacts can degrade panels but should not hard-fail gating.
    optional_schema_errors = hidden_errors + gs_errors + gm_errors
    status = "OK" if (diag.contract_ok and len(missing_required) == 0) else "INVALID_SCHEMA"
    return {
        "status": status,
        "errors": errors + optional_schema_errors,
        "missing_required_artifacts": sorted(set(missing_required)),
        "missing_optional_artifacts": sorted(set(missing_optional)),
        "schema_errors": errors,
        "paths": paths,
        "baseline_meta": baseline_meta,
        "baseline_state": baseline_state,
        "observer_state": state_blob,
        "observer_delta": delta_blob,
        "hidden_groups": hidden_rows,
        "group_summaries": group_summaries,
        "group_matrix": group_matrix,
    }


def _find_latest_file(filename: str) -> Optional[Path]:
    hits = list(ROOT.rglob(filename))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _find_latest_file_scoped(filename: str, run_key: Optional[str]) -> Optional[Path]:
    if run_key:
        run_dir = None
        if "INDEX" in globals():
            run = INDEX.get("runs", {}).get(str(run_key), {})
            run_dir = run.get("run_dir")
        candidate_roots = [run_dir, run_dir / "verification"] if run_dir else []
    else:
        candidate_roots = [ROOT / "analysis", ROOT / "verification", ROOT]
    hits: List[Path] = []
    for root in candidate_roots:
        if root and root.exists():
            hits.extend(root.rglob(filename))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _find_latest_pair_under(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not root.exists():
        return None, None
    reports = list(root.rglob("verification_report.json"))
    summaries = list(root.rglob("verification_summary.csv"))
    if not reports and not summaries:
        return None, None
    reports_by_parent = {str(p.parent): p for p in reports}
    summaries_by_parent = {str(p.parent): p for p in summaries}
    common_parents = set(reports_by_parent.keys()).intersection(set(summaries_by_parent.keys()))
    if common_parents:
        scored = []
        for parent in common_parents:
            rp = reports_by_parent[parent]
            sp = summaries_by_parent[parent]
            mtime = max(rp.stat().st_mtime, sp.stat().st_mtime)
            scored.append((mtime, rp, sp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored[0][2]
    report = sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)[0] if reports else None
    summary = sorted(summaries, key=lambda p: p.stat().st_mtime, reverse=True)[0] if summaries else None
    return report, summary


def _resolve_verification_pair(run_key: Optional[str], verification_source: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    if verification_source and verification_source != "auto":
        report, summary = _find_latest_pair_under(Path(verification_source))
        return summary, report

    run_dir = None
    if run_key and "INDEX" in globals():
        run = INDEX.get("runs", {}).get(str(run_key), {})
        run_dir = run.get("run_dir")

    # When a run is selected, keep verification resolution local to that run family.
    if run_dir and run_dir.exists():
        exact_summary = run_dir / "verification_summary.csv"
        exact_report = run_dir / "verification_report.json"
        if exact_summary.exists() and exact_report.exists():
            return exact_summary, exact_report

        search_roots: List[Path] = [run_dir, run_dir / "verification"]
        seen = set()
        for root in search_roots:
            if not root or not root.exists():
                continue
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            report, summary = _find_latest_pair_under(root)
            if report and summary:
                return summary, report
        return None, None

    summary = _find_latest_file_scoped("verification_summary.csv", run_key) or _find_latest_file("verification_summary.csv")
    report = _find_latest_file_scoped("verification_report.json", run_key) or _find_latest_file("verification_report.json")
    return summary, report


def _find_latest_under(root: Path, filename: str) -> Optional[Path]:
    if not root.exists():
        return None
    hits = list(root.rglob(filename))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def discover_verification_sources(run_key: Optional[str]) -> List[dict]:
    options = [{"label": "Auto (nearest latest)", "value": "auto"}]
    seen = {"auto"}
    roots: List[Path] = []
    if run_key and "INDEX" in globals():
        run = INDEX.get("runs", {}).get(str(run_key), {})
        run_dir = run.get("run_dir")
        if run_dir:
            roots.extend([run_dir, run_dir / "verification"])
    else:
        roots.extend([ROOT / "analysis", ROOT / "verification", ROOT])
    for root in roots:
        if not root.exists():
            continue
        for name in ("verification_report.json", "verification_summary.csv"):
            for hit in root.rglob(name):
                parent = str(hit.parent)
                if parent not in seen:
                    seen.add(parent)
                    options.append({"label": parent, "value": parent})
    return options[:50]


def _collect_run_dirs(artifact_root: Optional[Path]) -> List[Path]:
    if not artifact_root or not artifact_root.exists():
        return []
    run_dirs: List[Path] = []
    for d in artifact_root.iterdir():
        if not d.is_dir():
            continue
        if _is_run_directory(d):
            run_dirs.append(d)
            continue
        for child in d.iterdir():
            if child.is_dir() and _is_run_directory(child):
                run_dirs.append(child)
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs


def _run_display_key(run_dir: Path) -> str:
    try:
        rel = run_dir.relative_to(ROOT)
        return str(rel).replace("\\", "/")
    except Exception:
        experiment_key = run_dir.parent.parent.name if run_dir.parent and run_dir.parent.parent else "unknown"
        return f"{experiment_key}/{run_dir.name}"


def _collect_variant_names(run_dir: Path) -> List[str]:
    html_files = [p for p in run_dir.glob("*.html") if p.is_file()]
    if not html_files:
        return ["MONOLITH.html"]
    html_files.sort(
        key=lambda p: (
            0 if p.name.upper().startswith("MONOLITH") else 1,
            -int(p.stat().st_mtime_ns),
            p.name.lower(),
        )
    )
    return [p.name for p in html_files]


def _fmt_pass(v: Optional[bool]) -> str:
    if v is True:
        return "PASS"
    if v is False:
        return "FAIL"
    return "UNKNOWN"


def _collect_json_bools(node, out: Dict[str, bool]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            lk = str(k).lower()
            if isinstance(v, bool):
                out[lk] = v
            _collect_json_bools(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_json_bools(item, out)


def _find_bool_key(blob: dict, keys: List[str]) -> Optional[bool]:
    flat: Dict[str, bool] = {}
    _collect_json_bools(blob, flat)
    for key in keys:
        if key in flat:
            return flat[key]
    return None


def load_verification_state(run_key: Optional[str], verification_source: Optional[str] = "auto") -> dict:
    summary_csv, report_json = _resolve_verification_pair(run_key, verification_source)
    run_dir = _resolve_run_dir(run_key)
    contract_diag = evaluate_consumer_contract(run_dir) if run_dir and run_dir.exists() else None
    if not report_json and contract_diag and contract_diag.paths.get("verification_report.json"):
        report_json = contract_diag.paths.get("verification_report.json")
    if not summary_csv and contract_diag and contract_diag.paths.get("verification_summary.csv"):
        summary_csv = contract_diag.paths.get("verification_summary.csv")

    global_pass: Optional[bool] = None
    seed_stability: Optional[bool] = None
    crn_locked: Optional[bool] = None
    verification_status = contract_diag.verification_status if contract_diag else LayerStatus.UNVERIFIED.value
    broken = 0
    trapped = 0
    total = 0

    if report_json and report_json.exists():
        payload = _safe_json(report_json, {})
        global_pass = _find_bool_key(payload, ["global_pass"])
        seed_stability = _find_bool_key(payload, ["seed_stability", "seed_stability_pass", "seed_stable"])
        crn_locked = _find_bool_key(payload, ["crn_locked", "crn_lock_pass", "crn_pass"])
        if isinstance(payload, dict):
            for key in ("verification_status", "status", "comparability_status"):
                raw = payload.get(key)
                if raw is not None:
                    candidate = str(raw).upper().strip()
                    if candidate in VERIFICATION_STATUSES:
                        verification_status = candidate
                        break

    if not report_json and contract_diag and contract_diag.missing_required_artifacts:
        verification_status = LayerStatus.MISSING_ARTIFACTS.value
    elif verification_status == LayerStatus.UNVERIFIED.value and global_pass is True:
        verification_status = LayerStatus.VERIFIED.value

    if summary_csv and summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    broken += int(float(row.get("n_broken", 0)))
                except Exception:
                    pass
                try:
                    trapped += int(float(row.get("n_trapped", 0)))
                except Exception:
                    pass
                try:
                    total += int(float(row.get("n_total", 0)))
                except Exception:
                    total += 1

    survival_pct = 100.0
    friction = 0.0
    if total > 0:
        failures = broken + trapped
        survival_pct = max(0.0, 100.0 * (1.0 - (failures / float(total))))
        friction = failures / float(total)

    return {
        "verification_status": verification_status,
        "global_pass": global_pass,
        "seed_stability": seed_stability,
        "crn_locked": crn_locked,
        "n_broken": broken,
        "n_trapped": trapped,
        "n_total": total,
        "survival_pct": survival_pct,
        "geometric_friction": friction,
        "summary_path": str(summary_csv) if summary_csv else "NOT FOUND",
        "report_path": str(report_json) if report_json else "NOT FOUND",
        "verification_source": verification_source if verification_source else "auto",
    }


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _resolve_run_dir(run_key: Optional[str]) -> Optional[Path]:
    if not run_key:
        return None
    run = INDEX.get("runs", {}).get(str(run_key), {})
    run_dir = run.get("run_dir")
    return run_dir if isinstance(run_dir, Path) else None


def _candidate_cls_dirs(run_key: Optional[str]) -> List[Path]:
    out: List[Path] = []
    run = INDEX.get("runs", {}).get(str(run_key), {}) if run_key else {}
    run_dir = run.get("run_dir")
    if not isinstance(run_dir, Path) or not run_dir.exists():
        return out
    kernel = str(run.get("kernel", "")).strip()
    exp_dir = run_dir.parent.parent if run_dir.parent and run_dir.parent.parent else None
    if exp_dir and exp_dir.exists():
        if kernel and kernel.lower() != "unknown":
            out.append(exp_dir / kernel / "cls")
        out.extend(exp_dir.glob("*/cls"))
    parent_cls = run_dir.parent / "cls"
    if parent_cls.exists():
        out.append(parent_cls)
    seen = set()
    uniq = []
    for p in out:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq


def _control_results_candidates(run_key: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    run_dir = _resolve_run_dir(run_key)
    if run_dir and run_dir.exists():
        candidates.extend(
            [
                run_dir / "comprehensive_results.json",
                run_dir.parent / "comprehensive_results.json",
            ]
        )
        for cls_dir in _candidate_cls_dirs(run_key):
            if cls_dir.exists():
                candidates.extend(sorted(cls_dir.glob("comprehensive_analysis*/comprehensive_results.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    fallback = ROOT / "outputs" / "comprehensive_analysis" / "comprehensive_results.json"
    if fallback.exists():
        candidates.append(fallback)
    seen = set()
    out: List[Path] = []
    for p in candidates:
        sp = str(p)
        if sp not in seen and p.exists():
            seen.add(sp)
            out.append(p)
    return out


def load_control_state(run_key: Optional[str]) -> dict:
    for path in _control_results_candidates(run_key):
        data = _safe_json(path, {})
        interp = data.get("interpretation", {}) if isinstance(data, dict) else {}
        if not isinstance(interp, dict):
            continue
        if interp.get("error"):
            return {
                "status": "UNAVAILABLE",
                "source": str(path),
                "message": str(interp.get("error")),
                "procrustes_ratio": None,
                "distance_corr_ratio": None,
                "separates_count": 0,
                "consensus_pct": None,
                "residual_pct": None,
            }
        metrics = interp.get("metrics", {}) if isinstance(interp.get("metrics"), dict) else {}
        procrustes_ratio = _safe_float((metrics.get("procrustes", {}) or {}).get("ratio"), default=float("nan"))
        distance_corr_ratio = _safe_float((metrics.get("distance_corr", {}) or {}).get("ratio"), default=float("nan"))
        separates_count = sum(1 for m in metrics.values() if isinstance(m, dict) and bool(m.get("separates")))
        cres = ((interp.get("consensus_residual", {}) or {}).get("real", {}) or {})
        consensus_pct = _safe_float(cres.get("consensus_pct"), default=float("nan"))
        residual_pct = _safe_float(cres.get("residual_pct"), default=float("nan"))
        return {
            "status": "OK",
            "source": str(path),
            "message": "loaded",
            "procrustes_ratio": None if str(procrustes_ratio) == "nan" else procrustes_ratio,
            "distance_corr_ratio": None if str(distance_corr_ratio) == "nan" else distance_corr_ratio,
            "separates_count": int(separates_count),
            "consensus_pct": None if str(consensus_pct) == "nan" else consensus_pct,
            "residual_pct": None if str(residual_pct) == "nan" else residual_pct,
        }
    return {
        "status": "MISSING",
        "source": "NOT FOUND",
        "message": "no control analysis results",
        "procrustes_ratio": None,
        "distance_corr_ratio": None,
        "separates_count": 0,
        "consensus_pct": None,
        "residual_pct": None,
    }


def _ablation_candidates(run_key: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    run_dir = _resolve_run_dir(run_key)
    if run_dir and run_dir.exists():
        candidates.extend(
            [
                run_dir / "ablation_summary.json",
                run_dir / "ablation_results.json",
                run_dir / "critical_ablation_summary.csv",
            ]
        )
        candidates.extend(sorted(run_dir.glob("*ablation*summary*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(run_dir.glob("*ablation*summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
        for cls_dir in _candidate_cls_dirs(run_key):
            if cls_dir.exists():
                candidates.extend(sorted(cls_dir.glob("**/*ablation*summary*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
                candidates.extend(sorted(cls_dir.glob("**/*ablation*summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
    seen = set()
    out: List[Path] = []
    for p in candidates:
        sp = str(p)
        if sp not in seen and p.exists():
            seen.add(sp)
            out.append(p)
    return out


def load_ablation_state(run_key: Optional[str]) -> dict:
    for path in _ablation_candidates(run_key):
        if path.suffix.lower() == ".json":
            blob = _safe_json(path, {})
            if not isinstance(blob, dict):
                continue
            s1 = blob.get("stage_1_nmi")
            s2 = blob.get("stage_2_nmi")
            s3 = blob.get("stage_3_nmi")
            delta = blob.get("delta_nmi")
            retained = blob.get("retained_percentage")
            if any(v is not None for v in (s1, s2, s3, delta, retained)):
                return {
                    "status": "OK",
                    "source": str(path),
                    "stage_1_nmi": s1,
                    "stage_2_nmi": s2,
                    "stage_3_nmi": s3,
                    "delta_nmi": delta,
                    "retained_pct": retained,
                    "legacy_mean_variance": None,
                }
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    first = next(reader, None)
                if not first:
                    continue
                if "stage_1_nmi" in first or "stage_3_nmi" in first:
                    return {
                        "status": "OK",
                        "source": str(path),
                        "stage_1_nmi": first.get("stage_1_nmi"),
                        "stage_2_nmi": first.get("stage_2_nmi"),
                        "stage_3_nmi": first.get("stage_3_nmi"),
                        "delta_nmi": first.get("delta_nmi"),
                        "retained_pct": first.get("retained_percentage"),
                        "legacy_mean_variance": None,
                    }
                if "mean_variance" in first:
                    return {
                        "status": "LEGACY",
                        "source": str(path),
                        "stage_1_nmi": None,
                        "stage_2_nmi": None,
                        "stage_3_nmi": None,
                        "delta_nmi": None,
                        "retained_pct": None,
                        "legacy_mean_variance": first.get("mean_variance"),
                    }
            except Exception:
                continue
    return {
        "status": "MISSING",
        "source": "NOT FOUND",
        "stage_1_nmi": None,
        "stage_2_nmi": None,
        "stage_3_nmi": None,
        "delta_nmi": None,
        "retained_pct": None,
        "legacy_mean_variance": None,
    }


@lru_cache(maxsize=64)
def _cached_artifact_text(path_str: str, mtime_ns: int, size: int) -> str:
    return _safe_read_text(Path(path_str)).lower()


def _artifact_track_state(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {k: "unknown" for k in TRACK_MARKERS}
    try:
        stat = path.stat()
        text = _cached_artifact_text(str(path), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return {k: "unknown" for k in TRACK_MARKERS}
    out: Dict[str, str] = {}
    for track, markers in TRACK_MARKERS.items():
        out[track] = "online" if any(m in text for m in markers) else "missing"
    return out


def _artifact_coverage(run_key: str, variant_name: str) -> Tuple[int, int]:
    run = INDEX["runs"].get(run_key, {})
    manifest = run.get("observer_manifest") or {}
    manifest_variant = str(manifest.get("variant", "")).strip()
    if manifest and (not manifest_variant or manifest_variant == str(variant_name or "").strip()):
        cov = manifest.get("coverage", {})
        try:
            return int(cov.get("found", 0)), int(cov.get("total", 0))
        except Exception:
            pass

    observer_values = [
        opt.get("value")
        for opt in INDEX["observers_by_run"].get(run_key, [])
        if isinstance(opt, dict) and str(opt.get("value", "")).startswith("article:")
    ]
    total = len(observer_values)
    if total == 0:
        return 0, 0
    found = 0
    for ov in observer_values:
        p = resolve_artifact(run_key, variant_name, ov)
        if p and p.exists():
            found += 1
    return found, total


def _track_status_component(track_state: Dict[str, str]):
    chips = []
    for track in ["T1", "T1.5", "T2", "T3", "T4", "T5", "T6"]:
        st = track_state.get(track, "unknown")
        color = PALETTE["green"] if st == "online" else (PALETTE["red"] if st == "missing" else PALETTE["amber"])
        chips.append(
            html.Span(
                f"{track}:{st.upper()}",
                style={
                    "display": "inline-block",
                    "marginRight": "6px",
                    "marginBottom": "6px",
                    "padding": "2px 6px",
                    "borderRadius": "4px",
                    "border": f"1px solid {color}",
                    "color": color,
                    "fontSize": "0.74rem",
                    "letterSpacing": "0.02em",
                },
            )
        )
    return html.Div(chips)


def _track_delta_component(track_a: Dict[str, str], track_b: Dict[str, str]):
    chips = []
    for track in ["T1", "T1.5", "T2", "T3", "T4", "T5", "T6"]:
        a = track_a.get(track, "unknown")
        b = track_b.get(track, "unknown")
        same = a == b
        if same:
            color = PALETTE["green"] if a == "online" else PALETTE["amber"]
            text = f"{track}:A={a.upper()} B={b.upper()}"
        else:
            color = PALETTE["red"]
            text = f"{track}:A={a.upper()} B={b.upper()} DELTA"
        chips.append(
            html.Span(
                text,
                style={
                    "display": "inline-block",
                    "marginRight": "6px",
                    "marginBottom": "6px",
                    "padding": "2px 6px",
                    "borderRadius": "4px",
                    "border": f"1px solid {color}",
                    "color": color,
                    "fontSize": "0.72rem",
                    "letterSpacing": "0.02em",
                },
            )
        )
    return html.Div(chips)


def _build_run_observers_and_rows(run: dict) -> Tuple[List[dict], Dict[int, dict]]:
    observers = [{"label": "Global Mean", "value": "global"}]
    article_rows: Dict[int, dict] = {}
    csv_path: Path = run["monolith_data_path"]
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row.get("index", -1))
                except Exception:
                    continue
                article_rows[idx] = row
                title = (row.get("title") or "").strip()
                title_short = (title[:42] + "...") if len(title) > 45 else title
                observers.append({"label": f"Article #{idx} | {title_short}", "value": f"article:{idx}"})

    manifest = run.get("observer_manifest") or {}
    manifest_observers = manifest.get("observers", [])
    if isinstance(manifest_observers, list):
        seen = {o.get("value") for o in observers if isinstance(o, dict)}
        for item in manifest_observers:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value", "")).strip()
            if not value.startswith("article:") or value in seen:
                continue
            idx_text = value.split(":", 1)[1]
            label = f"Article #{idx_text}"
            observers.append({"label": label, "value": value})
            seen.add(value)

    return observers, article_rows


def build_artifact_index() -> dict:
    all_run_dirs: List[Path] = []
    metrics_by_root: Dict[str, Dict[str, dict]] = {}
    for root in ARTIFACT_ROOTS:
        run_dirs = _collect_run_dirs(root)
        all_run_dirs.extend(run_dirs)
        summary_path = root / "synthetic_summary.json"
        summary = _safe_json(summary_path, {}) if summary_path.exists() else {}
        by_key: Dict[str, dict] = {}
        for item in summary.get("results", []):
            key = str(item.get("run_key", "")).strip()
            if key:
                by_key[key] = item
        metrics_by_root[str(root)] = by_key

    all_run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    runs: Dict[str, dict] = {}
    run_keys: List[str] = []
    for run_dir in all_run_dirs:
        run_key = _run_display_key(run_dir)
        root_key = str(run_dir.parent)
        item = metrics_by_root.get(root_key, {}).get(run_dir.name, {})
        variants = _collect_variant_names(run_dir)
        run = {
            "run_key": run_key,
            "kernel": str(item.get("kernel", "unknown")),
            "seed": str(item.get("seed", "unknown")),
            "nmi": item.get("nmi"),
            "ari": item.get("ari"),
            "run_dir": run_dir,
            "artifact_root": run_dir.parent,
            "variants": variants,
            "article_meta_path": run_dir / "article_metadata.json",
            "monolith_data_path": run_dir / "MONOLITH_DATA.csv",
            "observer_manifest_path": run_dir / "observer_manifest.json",
            "observer_manifest": {},
            "observer_artifacts": {},
        }
        manifest_path = run["observer_manifest_path"]
        if manifest_path.exists():
            manifest = _safe_json(manifest_path, {})
            if isinstance(manifest, dict):
                run["observer_manifest"] = manifest
                for obs in manifest.get("observers", []):
                    if not isinstance(obs, dict):
                        continue
                    value = str(obs.get("value", "")).strip()
                    rel = str(obs.get("relative_path", "")).strip()
                    if not value or not rel:
                        continue
                    run["observer_artifacts"][value] = run_dir / rel.replace("/", "\\")

        runs[run_key] = run
        run_keys.append(run_key)

    default_run = run_keys[0] if run_keys else ""

    observers_by_run: Dict[str, List[dict]] = {}
    article_rows_by_run: Dict[str, Dict[int, dict]] = {}
    for rk in run_keys:
        obs, rows = _build_run_observers_and_rows(runs[rk])
        observers_by_run[rk] = obs
        article_rows_by_run[rk] = rows

    default_observers = observers_by_run.get(default_run, [{"label": "Global Mean", "value": "global"}])
    root_text = str(PRIMARY_ARTIFACT_ROOT) if PRIMARY_ARTIFACT_ROOT else ""
    root_count = len(ARTIFACT_ROOTS)
    return {
        "artifact_root": root_text,
        "artifact_roots": [str(p) for p in ARTIFACT_ROOTS],
        "artifact_root_count": root_count,
        "runs": runs,
        "run_keys": run_keys,
        "default_run": default_run,
        "observers": default_observers,
        "observers_by_run": observers_by_run,
        "article_rows_by_run": article_rows_by_run,
    }


INDEX = build_artifact_index()


def _run_options_from_index(index: dict) -> List[dict]:
    return [
        {
            "label": f"{rk} | kernel={index['runs'][rk]['kernel']} seed={index['runs'][rk]['seed']}",
            "value": rk,
        }
        for rk in index.get("run_keys", [])
        if rk in index.get("runs", {})
    ]


def resolve_artifact(run_key: str, variant_name: str, observer_value: str) -> Optional[Path]:
    run = INDEX["runs"].get(run_key)
    if not run:
        return None
    run_dir: Path = run["run_dir"]
    selected_variant = str(variant_name or "").strip()
    if not selected_variant:
        selected_variant = run.get("variants", ["MONOLITH.html"])[0]
    # Observer-specific naming support (future-compatible) should take priority
    # for article viewpoints, then fallback to chosen global variant.
    if observer_value.startswith("article:"):
        manifest_hit = run.get("observer_artifacts", {}).get(observer_value)
        if manifest_hit and manifest_hit.exists():
            return manifest_hit

        idx = observer_value.split(":", 1)[1]
        stem = Path(selected_variant).stem
        observer_candidates = [
            run_dir / f"observer_{idx}" / selected_variant,
            run_dir / f"article_{idx}" / selected_variant,
            run_dir / f"observer_{idx}" / "MONOLITH.html",
            run_dir / f"article_{idx}" / "MONOLITH.html",
            run_dir / f"{stem}_article_{idx}.html",
            run_dir / f"{stem}_observer_{idx}.html",
            run_dir / f"MONOLITH_article_{idx}.html",
            run_dir / f"MONOLITH_observer_{idx}.html",
        ]
        for p in observer_candidates:
            if p.exists():
                return p
    variant_path = run_dir / selected_variant
    if variant_path.exists():
        return variant_path
    for fallback_name in run.get("variants", []):
        p = run_dir / str(fallback_name)
        if p.exists():
            return p
    return None


def build_terminal_fallback(observer_value: str, run_key: str, variant_name: str, tick: int):
    observer_label = "GLOBAL" if observer_value == "global" else observer_value.replace(":", " ").upper()
    dots = "." * ((int(tick or 0) % 4) + 1)
    return html.Pre(
        f"""
==================================================================
 PRE-COMPUTATION REQUIRED
------------------------------------------------------------------
 Run                  : {run_key}
 Variant              : {variant_name}
 Requested Perspective: {observer_label}
 Artifact Status      : NOT FOUND
 Verification Sync    : WAITING{dots}
 Action               : Generate matching MONOLITH artifact
==================================================================
> booting artifact resolver{dots} _
""".strip("\n"),
        style={
            "margin": "0",
            "padding": "24px",
            "height": "100%",
            "backgroundColor": "#020202",
            "color": "#00FF41",
            "fontFamily": "Consolas, 'Courier New', monospace",
            "fontSize": "0.95rem",
            "lineHeight": "1.4",
            "whiteSpace": "pre",
            "animation": "fadeInFast 0.28s ease-out",
        },
    )


def build_empty_index_fallback() -> html.Pre:
    root_text = str(PRIMARY_ARTIFACT_ROOT) if PRIMARY_ARTIFACT_ROOT else "NOT FOUND"
    return html.Pre(
        f"""
==================================================================
 NO RUNS DISCOVERED
------------------------------------------------------------------
 Artifact Root : {root_text}
 Expected Data : synthetic/<run_key>/MONOLITH*.html
 Next Step     : run generator and refresh this dashboard
==================================================================
> waiting for MONOLITH artifacts _
""".strip("\n"),
        style={
            "margin": "0",
            "padding": "24px",
            "height": "100%",
            "backgroundColor": "#020202",
            "color": "#FFB347",
            "fontFamily": "Consolas, 'Courier New', monospace",
            "fontSize": "0.95rem",
            "lineHeight": "1.4",
            "whiteSpace": "pre",
            "animation": "fadeInFast 0.28s ease-out",
        },
    )


def _transition_wrapper(content, transition_style: str):
    transition_css = "fadeInFast 0.28s ease-out"
    overlay_style = {}
    if transition_style == "scan":
        overlay_style = {
            "backgroundImage": "repeating-linear-gradient(180deg, rgba(0,240,255,0.04) 0px, rgba(0,240,255,0.04) 1px, transparent 1px, transparent 3px)",
            "backgroundSize": "100% 12px",
            "animation": "scanSweep 0.45s linear 1",
        }
    elif transition_style == "glitch":
        transition_css = "jitterIn 0.22s steps(2,end)"
    return html.Div(
        [html.Div(style={"position": "absolute", "inset": 0, "pointerEvents": "none", **overlay_style}), content],
        style={"width": "100%", "height": "100%", "position": "relative", "animation": transition_css},
    )


app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="MONOLITH Artifact Command Center")

RUN_OPTIONS = _run_options_from_index(INDEX)
DEFAULT_RUN = INDEX["default_run"]
DEFAULT_VARIANTS = INDEX["runs"].get(DEFAULT_RUN, {}).get("variants", ["MONOLITH.html"])

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": PALETTE["void"], "minHeight": "100vh", "padding": "0"},
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(style={"display": "none"}),
        dcc.Store(id="gallery-dir", data={"dir": 1}),
        dcc.Store(id="hotkey-signal", data={"event": "none", "seq": 0}),
        dcc.Interval(id="hotkey-poll", interval=250, n_intervals=0, disabled=True),
        dcc.Interval(id="verification-poll", interval=10000, n_intervals=0, disabled=True),
        dcc.Interval(id="gallery-interval", interval=2200, n_intervals=0, disabled=True),
        dbc.Row(
            className="g-0",
            style={"minHeight": "100vh"},
            children=[
                dbc.Col(
                    xs=12,
                    md=3,
                    lg=3,
                    style={
                        "background": "linear-gradient(180deg, #050505 0%, #0b0b14 100%)",
                        "borderRight": f"1px solid {PALETTE['grid']}",
                        "padding": "16px",
                        "maxHeight": "100vh",
                        "overflowY": "auto",
                    },
                    children=[
                        html.H4("MONOLITH COMMAND CENTER", style={"color": PALETTE["cyan"], "letterSpacing": "0.05em", "marginBottom": "12px"}),
                        html.Div(
                            id="artifact-root-info",
                            children=f"Artifact Roots: {INDEX.get('artifact_root_count', 0)} | Primary: {INDEX.get('artifact_root') or 'NOT FOUND'}",
                            style={"color": PALETTE["dim"], "fontSize": "0.74rem", "wordBreak": "break-all", "marginBottom": "8px"},
                        ),
                        dbc.Button("Reindex", id="reindex-btn", color="info", size="sm", style={"width": "100%", "marginBottom": "6px"}),
                        html.Div(id="reindex-status", style={"color": PALETTE["dim"], "fontSize": "0.76rem", "marginBottom": "8px"}),
                        html.Label("Run", style={"color": PALETTE["text"], "fontWeight": "600"}),
                        dcc.Dropdown(id="run-dropdown", options=RUN_OPTIONS, value=DEFAULT_RUN, clearable=False, disabled=(len(RUN_OPTIONS) == 0), style={"color": "#111", "marginBottom": "8px"}),
                        html.Label("Observer", style={"color": PALETTE["text"], "fontWeight": "600"}),
                        dcc.Dropdown(id="observer-dropdown", options=INDEX["observers"], value="global", clearable=False, style={"color": "#111", "marginBottom": "8px"}),
                        html.Label("Verification Source", style={"color": PALETTE["text"], "fontWeight": "600"}),
                        dcc.Dropdown(id="verification-source", options=discover_verification_sources(DEFAULT_RUN), value="auto", clearable=False, style={"color": "#111", "marginBottom": "8px"}),
                        html.Div(id="provenance-line", style={"color": PALETTE["dim"], "fontSize": "0.76rem", "marginBottom": "8px", "wordBreak": "break-all"}),
                        html.Label("View Mode", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}),
                        dcc.RadioItems(
                            id="view-mode",
                            options=[{"label": "Global", "value": "global"}, {"label": "Observer", "value": "observer"}],
                            value="observer",
                            labelStyle={"display": "inline-block", "marginRight": "10px", "color": PALETTE["text"], "fontSize": "0.8rem"},
                            style={"marginBottom": "6px"},
                        ),
                        html.Label("Delta Mode", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}),
                        dcc.RadioItems(
                            id="delta-mode",
                            options=[{"label": "Baseline", "value": "baseline"}, {"label": "Observer", "value": "observer"}, {"label": "Delta", "value": "delta"}],
                            value="delta",
                            labelStyle={"display": "inline-block", "marginRight": "10px", "color": PALETTE["text"], "fontSize": "0.8rem"},
                            style={"marginBottom": "6px"},
                        ),
                        dcc.Checklist(
                            id="translation-mode",
                            options=[{"label": "Translation Only", "value": "translation_only"}],
                            value=[],
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"color": PALETTE["text"], "fontSize": "0.8rem"},
                            style={"marginBottom": "6px"},
                        ),
                        dcc.Checklist(
                            id="failure-overlays",
                            options=[{"label": "Show Failure Overlays", "value": "on"}],
                            value=[],
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"color": PALETTE["text"], "fontSize": "0.8rem"},
                            style={"marginBottom": "8px"},
                        ),
                        html.Label("Hidden Label Column", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}),
                        dcc.Dropdown(id="label-column-dropdown", options=[], value=None, clearable=True, style={"color": "#111", "marginBottom": "6px"}),
                        html.Label("Hidden Label Group", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}),
                        dcc.Dropdown(id="label-value-dropdown", options=[], value=[], multi=True, clearable=True, style={"color": "#111", "marginBottom": "8px"}),
                        html.Div(id="hidden-label-badge", style={"padding": "6px 8px", "borderRadius": "6px", "fontWeight": "700", "letterSpacing": "0.03em", "marginBottom": "8px", "textAlign": "center"}),
                        html.Div(id="hidden-label-detail", style={"color": PALETTE["dim"], "fontSize": "0.76rem", "marginBottom": "8px", "whiteSpace": "pre-wrap"}),
                        dcc.Checklist(
                            id="compare-enabled",
                            options=[{"label": "Enable A/B Compare", "value": "on"}],
                            value=[],
                            inputStyle={"marginRight": "8px"},
                            labelStyle={"color": PALETTE["text"], "fontSize": "0.83rem"},
                            style={"marginBottom": "6px"},
                        ),
                        dbc.Row(
                            className="g-1",
                            children=[
                                dbc.Col([html.Label("Variant A", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}), dcc.Dropdown(id="variant-a-dropdown", options=[{"label": v, "value": v} for v in DEFAULT_VARIANTS], value=DEFAULT_VARIANTS[0] if DEFAULT_VARIANTS else "MONOLITH.html", clearable=False, style={"color": "#111"})], width=6),
                                dbc.Col([html.Label("Variant B", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}), dcc.Dropdown(id="variant-b-dropdown", options=[{"label": v, "value": v} for v in DEFAULT_VARIANTS], value=(DEFAULT_VARIANTS[1] if len(DEFAULT_VARIANTS) > 1 else DEFAULT_VARIANTS[0]) if DEFAULT_VARIANTS else "MONOLITH.html", clearable=False, style={"color": "#111"})], width=6),
                            ],
                            style={"marginBottom": "8px"},
                        ),
                        dbc.Row(
                            className="g-1",
                            style={"marginBottom": "8px"},
                            children=[
                                dbc.Col(dbc.Button("Prev", id="prev-observer-btn", color="secondary", size="sm", style={"width": "100%"}), width=4),
                                dbc.Col(dbc.Button("Next", id="next-observer-btn", color="secondary", size="sm", style={"width": "100%"}), width=4),
                                dbc.Col(dcc.Checklist(id="autoplay-enabled", options=[{"label": "Auto", "value": "on"}], value=[], inputStyle={"marginRight": "6px"}, labelStyle={"color": PALETTE["text"], "fontSize": "0.82rem"}), width=4),
                            ],
                        ),
                        dcc.RadioItems(id="autoplay-mode", options=[{"label": "Loop", "value": "loop"}, {"label": "Ping-Pong", "value": "pingpong"}], value="loop", labelStyle={"display": "inline-block", "marginRight": "10px"}, style={"color": PALETTE["text"], "fontSize": "0.82rem", "marginBottom": "6px"}),
                        dcc.Dropdown(id="transition-style", options=[{"label": "Fade", "value": "fade"}, {"label": "Scanline", "value": "scan"}, {"label": "Glitch", "value": "glitch"}], value="fade", clearable=False, style={"color": "#111", "marginBottom": "8px"}),
                        dcc.Slider(id="autoplay-seconds", min=0.6, max=8.0, step=0.2, value=2.2, marks={0.6: "0.6", 2.0: "2.0", 4.0: "4.0", 8.0: "8.0"}, tooltip={"placement": "bottom", "always_visible": False}),
                        html.Div(id="animation-status", style={"color": PALETTE["dim"], "fontSize": "0.8rem", "marginTop": "6px", "marginBottom": "6px"}),
                        dbc.Progress(id="gallery-progress", value=0, label="0%", striped=True, animated=True, style={"height": "10px", "marginBottom": "8px"}),
                        html.Div(id="artifact-path", style={"color": PALETTE["dim"], "fontSize": "0.8rem", "wordBreak": "break-all", "marginBottom": "8px"}),
                        html.Div(id="run-score", style={"color": PALETTE["cyan"], "fontSize": "0.84rem", "marginBottom": "8px"}),
                        html.Div(id="article-metrics", style={"color": PALETTE["amber"], "fontSize": "0.84rem", "marginBottom": "8px"}),
                        html.Div("Track Presence", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="track-readout", style={"marginBottom": "8px"}),
                        html.Div("Track A/B Delta", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="track-compare-readout", style={"marginBottom": "8px"}),
                        html.Div(id="coverage-readout", style={"color": PALETTE["dim"], "fontSize": "0.8rem", "marginBottom": "8px"}),
                        dcc.Input(id="hotkey-input", type="text", placeholder="Focus here for hotkeys: j=prev, k=next, p=play/pause, c=compare", debounce=False, style={"width": "100%", "fontSize": "0.78rem", "padding": "4px 6px", "marginBottom": "6px", "backgroundColor": "#101018", "color": PALETTE["cyan"], "border": f"1px solid {PALETTE['grid']}"}, value=""),
                        html.Div(id="hotkey-status", style={"color": PALETTE["dim"], "fontSize": "0.78rem", "marginBottom": "8px"}),
                        html.Div(id="verification-badge", style={"padding": "8px 10px", "borderRadius": "6px", "fontWeight": "700", "letterSpacing": "0.05em", "marginBottom": "12px", "textAlign": "center"}),
                        html.Div(id="telemetry-1", style={"color": PALETTE["green"], "fontWeight": "700", "fontSize": "0.92rem"}),
                        html.Div(id="telemetry-2", style={"color": PALETTE["amber"], "fontWeight": "700", "fontSize": "0.92rem", "marginTop": "4px"}),
                        html.Div(id="telemetry-3", style={"color": PALETTE["cyan"], "fontWeight": "700", "fontSize": "0.92rem", "marginTop": "4px"}),
                        html.Div(id="telemetry-detail", style={"color": PALETTE["dim"], "fontSize": "0.8rem", "marginTop": "8px"}),
                        html.Hr(style={"borderColor": PALETTE["grid"], "margin": "10px 0"}),
                        html.Div("Ablation Metrics", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="ablation-metrics", style={"color": PALETTE["amber"], "fontSize": "0.8rem", "whiteSpace": "pre-wrap", "marginBottom": "8px"}),
                        html.Div("Control Metrics", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="control-metrics", style={"color": PALETTE["cyan"], "fontSize": "0.8rem", "whiteSpace": "pre-wrap", "marginBottom": "6px"}),
                        html.Hr(style={"borderColor": PALETTE["grid"], "margin": "10px 0"}),
                        html.Div("Relativity Deltas", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="relativity-panel", style={"padding": "8px", "maxHeight": "24vh", "overflowY": "auto", "border": f"1px solid {PALETTE['grid']}", "borderRadius": "6px", "marginBottom": "8px"}),
                        html.Div("Group Path Patterns", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="group-panel", style={"padding": "8px", "maxHeight": "22vh", "overflowY": "auto", "border": f"1px solid {PALETTE['grid']}", "borderRadius": "6px", "marginBottom": "8px"}),
                        html.Div("Empathy Gap", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        dcc.Graph(id="empathy-heatmap", style={"height": "28vh", "marginBottom": "8px"}),
                    ],
                ),
                dbc.Col(
                    xs=12,
                    md=9,
                    lg=9,
                    style={"minHeight": "100vh", "padding": "0", "backgroundColor": "#020208"},
                    children=[
                        html.Div(
                            style={"height": "90vh", "width": "100%"},
                            children=[
                                dcc.Loading(
                                    id="artifact-loading",
                                    type="default",
                                    color=PALETTE["cyan"],
                                    children=[html.Div(id="artifact-container", style={"height": "90vh", "width": "100%"})],
                                ),
                                html.Div(id="watermark-overlay", style={"display": "none"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("run-dropdown", "options"),
    Output("run-dropdown", "value"),
    Output("run-dropdown", "disabled"),
    Output("artifact-root-info", "children"),
    Output("reindex-status", "children"),
    Input("reindex-btn", "n_clicks"),
    State("run-dropdown", "value"),
)
def reindex_runs(n_clicks: Optional[int], current_run: Optional[str]):
    global ARTIFACT_ROOTS, PRIMARY_ARTIFACT_ROOT, INDEX, RUN_OPTIONS, DEFAULT_RUN, DEFAULT_VARIANTS
    ARTIFACT_ROOTS = _discover_artifact_roots()
    PRIMARY_ARTIFACT_ROOT = ARTIFACT_ROOTS[0] if ARTIFACT_ROOTS else None
    INDEX = build_artifact_index()
    RUN_OPTIONS = _run_options_from_index(INDEX)
    DEFAULT_RUN = INDEX.get("default_run", "")
    DEFAULT_VARIANTS = INDEX.get("runs", {}).get(DEFAULT_RUN, {}).get("variants", ["MONOLITH.html"])

    option_values = [opt["value"] for opt in RUN_OPTIONS]
    if current_run in option_values:
        run_value = current_run
    else:
        run_value = DEFAULT_RUN if DEFAULT_RUN in option_values else (option_values[0] if option_values else "")
    disabled = len(RUN_OPTIONS) == 0
    root_info = f"Artifact Roots: {INDEX.get('artifact_root_count', 0)} | Primary: {INDEX.get('artifact_root') or 'NOT FOUND'}"
    status = f"Reindex complete | runs={len(INDEX.get('run_keys', []))} | clicks={int(n_clicks or 0)}"
    return RUN_OPTIONS, run_value, disabled, root_info, status


@app.callback(
    Output("variant-a-dropdown", "options"),
    Output("variant-a-dropdown", "value"),
    Output("variant-b-dropdown", "options"),
    Output("variant-b-dropdown", "value"),
    Output("observer-dropdown", "options"),
    Output("observer-dropdown", "value"),
    Input("run-dropdown", "value"),
    State("variant-a-dropdown", "value"),
    State("variant-b-dropdown", "value"),
    State("observer-dropdown", "value"),
)
def refresh_variants(run_key: str, current_a: str, current_b: str, current_observer: str):
    run = INDEX["runs"].get(run_key, {})
    variants = run.get("variants", ["MONOLITH.html"])
    if not variants:
        variants = ["MONOLITH.html"]
    opts = [{"label": v, "value": v} for v in variants]
    a = current_a if current_a in variants else variants[0]
    b = current_b if current_b in variants else (variants[1] if len(variants) > 1 else variants[0])
    obs_opts = INDEX["observers_by_run"].get(run_key, [{"label": "Global Mean", "value": "global"}])
    obs_values = [o["value"] for o in obs_opts]
    observer = current_observer if current_observer in obs_values else "global"
    return opts, a, opts, b, obs_opts, observer


@app.callback(
    Output("run-dropdown", "value", allow_duplicate=True),
    Output("observer-dropdown", "value", allow_duplicate=True),
    Output("view-mode", "value", allow_duplicate=True),
    Output("compare-enabled", "value", allow_duplicate=True),
    Input("url", "search"),
    State("run-dropdown", "options"),
    State("run-dropdown", "value"),
    prevent_initial_call="initial_duplicate",
)
def apply_url_state(search: Optional[str], run_options, current_run):
    if not search:
        return current_run, "global", "observer", []
    try:
        qs = parse_qs((search or "").lstrip("?"))
    except Exception:
        return current_run, "global", "observer", []

    run_values = [o.get("value") for o in (run_options or []) if isinstance(o, dict)]
    requested_run = str((qs.get("run_key") or [current_run])[0] or current_run)
    run_value = requested_run if requested_run in run_values else current_run

    observer = str((qs.get("observer") or ["global"])[0] or "global")
    view_mode = str((qs.get("view_mode") or ["observer"])[0] or "observer").lower()
    if view_mode not in {"global", "observer"}:
        view_mode = "observer"
    compare_raw = str((qs.get("compare") or ["0"])[0]).strip().lower()
    compare_values = ["on"] if compare_raw in {"1", "true", "on", "yes"} else []
    return run_value, observer, view_mode, compare_values


@app.callback(
    Output("verification-source", "options"),
    Output("verification-source", "value"),
    Input("run-dropdown", "value"),
    State("verification-source", "value"),
)
def refresh_verification_sources(run_key: str, current_value: str):
    options = discover_verification_sources(run_key)
    values = [o["value"] for o in options]
    value = current_value if current_value in values else "auto"
    return options, value


@app.callback(
    Output("label-column-dropdown", "options"),
    Output("label-column-dropdown", "value"),
    Output("label-value-dropdown", "options"),
    Output("label-value-dropdown", "value"),
    Input("run-dropdown", "value"),
    State("label-column-dropdown", "value"),
    State("label-value-dropdown", "value"),
)
def refresh_hidden_label_filters(run_key: str, current_col: Optional[str], current_values: Optional[List[str]]):
    contract = load_contract_state(run_key, "global")
    rows = contract.get("hidden_groups", []) or []
    if not rows:
        return [], None, [], []

    columns = sorted({k for r in rows if isinstance(r, dict) for k in r.keys() if k.startswith("group_")})
    col_opts = [{"label": c, "value": c} for c in columns]
    chosen_col = current_col if current_col in columns else (columns[0] if columns else None)

    value_opts = []
    if chosen_col:
        values = sorted({str(r.get(chosen_col, "")).strip() for r in rows if str(r.get(chosen_col, "")).strip()})
        value_opts = [{"label": v, "value": v} for v in values]
    current_values = current_values or []
    valid_values = [v for v in current_values if any(opt["value"] == v for opt in value_opts)]
    return col_opts, chosen_col, value_opts, valid_values


app.clientside_callback(
    """
    function(tick, current) {
        if (!window.__monolith_hotkeys_bound) {
            window.__monolith_hotkeys_bound = true;
            window.__monolith_hotkey_seq = 0;
            document.addEventListener('keydown', function(ev) {
                var k = (ev.key || '').toLowerCase();
                if (['j','k','p','c'].indexOf(k) >= 0) {
                    window.__monolith_hotkey_seq += 1;
                    window.__monolith_hotkey_event = {event: k, seq: window.__monolith_hotkey_seq, ts: Date.now()};
                }
            });
        }
        return window.__monolith_hotkey_event || current || {event: 'none', seq: 0};
    }
    """,
    Output("hotkey-signal", "data"),
    Input("hotkey-poll", "n_intervals"),
    State("hotkey-signal", "data"),
)


@app.callback(
    Output("gallery-interval", "interval"),
    Output("gallery-interval", "disabled"),
    Output("animation-status", "children"),
    Input("autoplay-enabled", "value"),
    Input("autoplay-seconds", "value"),
    Input("autoplay-mode", "value"),
    Input("transition-style", "value"),
)
def configure_gallery_autoplay(enabled_values: List[str], seconds: float, autoplay_mode: str, transition_style: str):
    seconds_safe = max(0.6, min(8.0, float(seconds) if seconds else 2.2))
    enabled = enabled_values is not None and "on" in enabled_values
    state = "RUNNING" if enabled else "PAUSED"
    return int(seconds_safe * 1000), (not enabled), f"Gallery: {state} | {seconds_safe:.1f}s | {autoplay_mode} | {transition_style}"


@app.callback(
    Output("observer-dropdown", "value", allow_duplicate=True),
    Output("autoplay-enabled", "value", allow_duplicate=True),
    Output("compare-enabled", "value", allow_duplicate=True),
    Output("hotkey-status", "children"),
    Input("hotkey-signal", "data"),
    State("observer-dropdown", "value"),
    State("observer-dropdown", "options"),
    State("autoplay-enabled", "value"),
    State("compare-enabled", "value"),
    prevent_initial_call=True,
)
def handle_hotkeys(hotkey_signal, observer_value, observer_options, autoplay_values, compare_values):
    event = str((hotkey_signal or {}).get("event", "none")).lower()
    if event not in {"j", "k", "p", "c"}:
        return observer_value, autoplay_values, compare_values, "Hotkeys idle"
    seq = [opt.get("value") for opt in (observer_options or []) if isinstance(opt, dict) and "value" in opt]
    if not seq:
        seq = ["global"]
    cur = observer_value if observer_value in seq else seq[0]
    idx = seq.index(cur)
    new_observer = cur
    autoplay = list(autoplay_values or [])
    compare = list(compare_values or [])
    note = "Hotkeys ready"
    if event == "j":
        new_observer = seq[(idx - 1) % len(seq)]
        note = "Hotkey j: previous observer"
    elif event == "k":
        new_observer = seq[(idx + 1) % len(seq)]
        note = "Hotkey k: next observer"
    elif event == "p":
        autoplay = [] if "on" in autoplay else ["on"]
        note = "Hotkey p: autoplay toggle"
    elif event == "c":
        compare = [] if "on" in compare else ["on"]
        note = "Hotkey c: compare toggle"
    return new_observer, autoplay, compare, note


@app.callback(
    Output("observer-dropdown", "value", allow_duplicate=True),
    Output("gallery-dir", "data"),
    Input("gallery-interval", "n_intervals"),
    Input("prev-observer-btn", "n_clicks"),
    Input("next-observer-btn", "n_clicks"),
    Input("autoplay-mode", "value"),
    State("observer-dropdown", "value"),
    State("gallery-dir", "data"),
    State("observer-dropdown", "options"),
    prevent_initial_call=True,
)
def step_observer(
    _auto_tick: int,
    _prev: Optional[int],
    _next: Optional[int],
    autoplay_mode: str,
    current: str,
    dir_data,
    observer_options,
):
    run_trigger = None
    if callback_context.triggered:
        run_trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
    sequence = [opt.get("value") for opt in (observer_options or []) if isinstance(opt, dict) and "value" in opt]
    if not sequence:
        sequence = [opt["value"] for opt in INDEX["observers"]]
    if not sequence:
        return current, {"dir": 1}
    cur = current if current in sequence else sequence[0]
    idx = sequence.index(cur)
    direction = int((dir_data or {"dir": 1}).get("dir", 1))
    trigger = run_trigger
    if trigger == "prev-observer-btn":
        return sequence[(idx - 1) % len(sequence)], {"dir": -1}
    if trigger in ("next-observer-btn", "gallery-interval"):
        if autoplay_mode == "pingpong" and len(sequence) > 1:
            if idx == len(sequence) - 1:
                direction = -1
            elif idx == 0:
                direction = 1
            return sequence[(idx + direction) % len(sequence)], {"dir": direction}
        return sequence[(idx + 1) % len(sequence)], {"dir": 1}
    return cur, {"dir": direction}


@app.callback(
    Output("gallery-progress", "value"),
    Output("gallery-progress", "label"),
    Input("observer-dropdown", "value"),
    State("observer-dropdown", "options"),
)
def update_gallery_progress(observer_value: str, observer_options):
    seq = [opt.get("value") for opt in (observer_options or []) if isinstance(opt, dict) and "value" in opt]
    if not seq:
        seq = [opt["value"] for opt in INDEX["observers"]]
    if not seq:
        return 0, "0%"
    idx = seq.index(observer_value) if observer_value in seq else 0
    pct = int(round(100.0 * idx / max(1, len(seq) - 1)))
    return pct, f"{pct}%"


@app.callback(
    Output("artifact-container", "children"),
    Output("artifact-path", "children"),
    Output("run-score", "children"),
    Output("article-metrics", "children"),
    Output("track-readout", "children"),
    Output("track-compare-readout", "children"),
    Output("coverage-readout", "children"),
    Output("verification-badge", "children"),
    Output("verification-badge", "style"),
    Output("telemetry-1", "children"),
    Output("telemetry-2", "children"),
    Output("telemetry-3", "children"),
    Output("telemetry-detail", "children"),
    Output("ablation-metrics", "children"),
    Output("control-metrics", "children"),
    Output("provenance-line", "children"),
    Output("hidden-label-badge", "children"),
    Output("hidden-label-badge", "style"),
    Output("hidden-label-detail", "children"),
    Output("relativity-panel", "children"),
    Output("group-panel", "children"),
    Output("empathy-heatmap", "figure"),
    Output("watermark-overlay", "children"),
    Output("watermark-overlay", "style"),
    Input("observer-dropdown", "value"),
    State("run-dropdown", "value"),
    State("variant-a-dropdown", "value"),
    State("variant-b-dropdown", "value"),
    State("verification-source", "value"),
    State("compare-enabled", "value"),
    State("transition-style", "value"),
    State("view-mode", "value"),
    State("delta-mode", "value"),
    State("translation-mode", "value"),
    State("failure-overlays", "value"),
    State("label-column-dropdown", "value"),
    State("label-value-dropdown", "value"),
)
def render_dashboard(
    observer_value: str,
    run_key: str,
    variant_a: str,
    variant_b: str,
    verification_source: str,
    compare_enabled_values: List[str],
    transition_style: str,
    view_mode: str,
    delta_mode: str,
    translation_mode_values: List[str],
    failure_overlay_values: List[str],
    label_column: Optional[str],
    label_values: List[str],
):
    return _render_dashboard_impl(
        run_key=run_key,
        observer_value=observer_value,
        variant_a=variant_a,
        variant_b=variant_b,
        verification_source=verification_source,
        compare_enabled_values=compare_enabled_values,
        transition_style=transition_style,
        poll_tick=0,
        gallery_tick=0,
        view_mode=view_mode,
        delta_mode=delta_mode,
        translation_mode_values=translation_mode_values,
        failure_overlay_values=failure_overlay_values,
        label_column=label_column,
        label_values=label_values,
    )


def _extract_survival_rate(html_text: str) -> Optional[float]:
    """Extract T4 Survival % from the Epistemic Panel in the HTML."""
    try:
        match = re.search(r"T4 Survival:.*?(\d+)%", html_text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
    except Exception:
        pass
    return None


def _claims_enabled(contract_status: str, verification_status: str, global_pass: Optional[bool]) -> bool:
    return (
        str(contract_status).upper() == "OK"
        and str(verification_status).upper() == LayerStatus.VERIFIED.value
        and (global_pass is True)
    )


def _compute_gate_presentation(
    contract_status: str,
    verification_status: str,
    global_pass: Optional[bool],
    type2_dissonance: bool,
) -> Dict[str, Any]:
    claims_enabled = _claims_enabled(contract_status, verification_status, global_pass)
    if type2_dissonance:
        badge_text = "[TYPE 2 DISSONANCE]"
    elif claims_enabled:
        badge_text = "[VERIFIED]"
    elif str(contract_status).upper() != "OK":
        badge_text = "[INVALID SCHEMA]"
    elif str(verification_status).upper() == LayerStatus.MISSING_ARTIFACTS.value:
        badge_text = "[MISSING ARTIFACTS]"
    elif str(verification_status).upper() == LayerStatus.NON_COMPARABLE.value:
        badge_text = "[NON-COMPARABLE]"
    else:
        badge_text = "[UNVERIFIED]"
    watermark_visible = not claims_enabled
    return {
        "claims_enabled": claims_enabled,
        "badge_text": badge_text,
        "watermark_visible": watermark_visible,
    }


def _build_empathy_figure(contract: dict, label_col: Optional[str], label_values: Optional[List[str]]):
    matrix_blob = contract.get("group_matrix", {}) or {}
    groups = matrix_blob.get("groups", []) if isinstance(matrix_blob, dict) else []
    matrix = matrix_blob.get("cost_matrix", []) if isinstance(matrix_blob, dict) else []

    fig = go.Figure()
    if not groups or not matrix:
        fig.update_layout(
            template="plotly_dark",
            margin={"l": 30, "r": 10, "t": 30, "b": 30},
            title="Empathy Gap Matrix (unavailable)",
            annotations=[{"text": "group_matrix not available", "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}],
        )
        return fig

    filtered_idx = list(range(len(groups)))
    if label_values:
        selected = set(label_values)
        filtered_idx = [i for i, g in enumerate(groups) if g in selected]
        if not filtered_idx:
            filtered_idx = list(range(len(groups)))

    fg = [groups[i] for i in filtered_idx]
    fm = [[float(matrix[i][j]) for j in filtered_idx] for i in filtered_idx]
    fig.add_trace(
        go.Heatmap(
            z=fm,
            x=fg,
            y=fg,
            colorscale="Viridis",
            colorbar={"title": "Cost"},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        title="Empathy Gap Matrix (Directed Cost)",
        xaxis_title=label_col or "Group",
        yaxis_title="Observer Group",
    )
    return fig


def _build_group_panel(contract: dict, label_col: Optional[str], label_values: Optional[List[str]]):
    rows = contract.get("hidden_groups", []) or []
    summaries = (contract.get("group_summaries", {}) or {}).get("groups", [])
    if not rows:
        return html.Div("Hidden labels unavailable for this run.", style={"color": PALETTE["amber"]})

    if not label_col:
        label_col = "group_topic"

    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(label_col, "UNLABELED"))
        if label_values and key not in set(label_values):
            continue
        counts[key] = counts.get(key, 0) + 1

    lines = [f"{label_col} counts:"]
    for k in sorted(counts):
        lines.append(f"  - {k}: {counts[k]}")

    summary_lines = []
    if isinstance(summaries, list):
        summary_lines.append("Group summaries:")
        for g in summaries[:8]:
            if not isinstance(g, dict):
                continue
            name = g.get("group_name", "unknown")
            n = g.get("n_articles", "n/a")
            summary_lines.append(f"  - {name}: n={n}")

    return html.Pre("\n".join(lines + [""] + summary_lines), style={"margin": "0", "color": PALETTE["text"], "fontSize": "0.82rem"})


def _build_relativity_panel(contract: dict, observer_value: str, delta_mode: str, translation_mode_values: List[str]):
    if observer_value == "global":
        return html.Div("Global baseline selected. Choose an article observer to view Type-2 deltas.", style={"color": PALETTE["dim"]})

    delta = contract.get("observer_delta", {}) or {}
    if not delta:
        return html.Div("Observer delta artifact missing.", style={"color": PALETTE["amber"]})

    null_eq = delta.get("null_observer_equivalence", {}) or {}
    metrics_delta = delta.get("metrics_delta", {}) or {}
    axis_delta = delta.get("axis_delta", {}) or {}
    flips = delta.get("path_flip_delta", {}) or {}

    translation_only = translation_mode_values is not None and "translation_only" in translation_mode_values
    tcomp = delta.get("translation_only_comparison", {}) if translation_only else {}

    lines = [
        f"Observer: {observer_value} | mode={delta_mode}",
        f"Null Eq -> max_coord_delta={null_eq.get('max_coord_delta', 'n/a')} | path_flip_count={null_eq.get('path_flip_count', 'n/a')} | axis_rotation_deg={null_eq.get('axis_rotation_deg', 'n/a')}",
        f"Metrics Delta -> d_rupture_rate={metrics_delta.get('d_rupture_rate', 'n/a')} | d_mean_work={metrics_delta.get('d_mean_work', 'n/a')} | d_survival_pct={metrics_delta.get('d_survival_pct', 'n/a')}",
        f"Axis Delta -> rotation_deg={axis_delta.get('rotation_deg', 'n/a')} | d_explained_variance_axis1={axis_delta.get('d_explained_variance_axis1', 'n/a')}",
        "Path Flip Delta (top):",
    ]
    flip_items = sorted(flips.items(), key=lambda kv: -float(kv[1]) if _is_number(kv[1]) else 0.0)
    for k, v in flip_items[:8]:
        lines.append(f"  - {k}: {v}")
    if translation_only and tcomp:
        lines.append("Translation-only comparison:")
        lines.append(f"  - d_path_flip_count={tcomp.get('d_path_flip_count', 'n/a')} | d_mean_work={tcomp.get('d_mean_work', 'n/a')}")

    return html.Pre("\n".join(lines), style={"margin": "0", "color": PALETTE["text"], "fontSize": "0.82rem"})


def _render_dashboard_impl(
    run_key: str,
    observer_value: str,
    variant_a: str,
    variant_b: str,
    verification_source: str,
    compare_enabled_values: List[str],
    transition_style: str,
    poll_tick: int,
    gallery_tick: int,
    view_mode: str,
    delta_mode: str,
    translation_mode_values: List[str],
    failure_overlay_values: List[str],
    label_column: Optional[str],
    label_values: List[str],
):
    if not INDEX["run_keys"] or not run_key:
        empty = build_empty_index_fallback()
        badge_style = {
            "padding": "8px 10px",
            "borderRadius": "6px",
            "fontWeight": "700",
            "letterSpacing": "0.05em",
            "marginBottom": "10px",
            "textAlign": "center",
            "color": PALETTE["amber"],
            "border": f"1px solid {PALETTE['amber']}",
            "backgroundColor": "rgba(255,179,71,0.12)",
        }
        detail = f"artifact_roots={INDEX.get('artifact_root_count', 0)} | primary={INDEX.get('artifact_root') or 'NOT FOUND'}"
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", title="Empathy Gap Matrix (unavailable)")
        return (
            empty,
            "Artifact: NOT FOUND",
            "Run Score | n/a",
            "Article Metrics: n/a",
            _track_status_component({k: "unknown" for k in TRACK_MARKERS}),
            _track_delta_component({k: "unknown" for k in TRACK_MARKERS}, {k: "unknown" for k in TRACK_MARKERS}),
            "Observer Artifact Coverage: n/a",
            "[NO RUNS]",
            badge_style,
            "System 1: Topologic Integrity | n/a",
            "System 2: Geometric Friction = n/a",
            "System 2: Survival % = n/a",
            detail,
            "Ablation: n/a",
            "Control: n/a",
            "provenance: n/a",
            "[HIDDEN LABELS MISSING]",
            {"padding": "6px 8px", "borderRadius": "6px", "textAlign": "center", "color": PALETTE["amber"], "border": f"1px solid {PALETTE['amber']}", "backgroundColor": "rgba(255,179,71,0.12)"},
            "labels/hidden_groups.csv not found",
            html.Div("Relativity data unavailable", style={"color": PALETTE["amber"]}),
            html.Div("Group data unavailable", style={"color": PALETTE["amber"]}),
            empty_fig,
            "",
            {"display": "none"},
        )

    run = INDEX["runs"].get(run_key, {})
    compare_enabled = compare_enabled_values is not None and "on" in compare_enabled_values
    effective_observer = "global" if view_mode == "global" else observer_value
    contract = load_contract_state(run_key, effective_observer)
    contract_status = contract.get("status", "INVALID_SCHEMA")
    contract_errors = contract.get("errors", [])
    schema_errors = contract.get("schema_errors", [])
    missing_required = contract.get("missing_required_artifacts", [])
    missing_optional = contract.get("missing_optional_artifacts", [])
    baseline_meta = contract.get("baseline_meta", {}) or {}

    # VALIDATION TYPE 2: Perspective Sensitivity (Global vs Subjective)
    # If in observer mode, Variant A shows Global perspective for comparison.
    if view_mode == "observer" and observer_value.startswith("article:"):
        p_a = resolve_artifact(run_key, variant_a, "global") if run_key else None
        p_b = resolve_artifact(run_key, variant_b, observer_value) if run_key else None
    else:
        p_a = resolve_artifact(run_key, variant_a, effective_observer) if run_key else None
        p_b = resolve_artifact(run_key, variant_b, effective_observer) if run_key else None

    text_a = _safe_read_text(p_a) if (p_a and p_a.exists()) else ""
    text_b = _safe_read_text(p_b) if (p_b and p_b.exists()) else ""

    if text_a:
        c_a = _transition_wrapper(html.Iframe(srcDoc=text_a, style={"width": "100%", "height": "100%", "border": "0"}), transition_style)
    else:
        c_a = build_terminal_fallback(effective_observer, run_key, variant_a, gallery_tick)

    if compare_enabled:
        if text_b:
            c_b = _transition_wrapper(html.Iframe(srcDoc=text_b, style={"width": "100%", "height": "100%", "border": "0"}), transition_style)
        else:
            c_b = build_terminal_fallback(effective_observer, run_key, variant_b, gallery_tick)
        container = dbc.Row(
            className="g-0",
            style={"height": "90vh"},
            children=[
                dbc.Col([html.Div("Variant A", style={"color": PALETTE["cyan"], "padding": "4px 8px"}), html.Div(c_a, style={"height": "calc(90vh - 28px)"})], width=6),
                dbc.Col([html.Div("Variant B", style={"color": PALETTE["cyan"], "padding": "4px 8px"}), html.Div(c_b, style={"height": "calc(90vh - 28px)"})], width=6),
            ],
        )
        path_text = f"A: {p_a if p_a else 'NOT FOUND'} | B: {p_b if p_b else 'NOT FOUND'}"
    else:
        single_view_path = p_a
        single_view = c_a
        if view_mode == "observer" and observer_value.startswith("article:") and p_b and p_b.exists():
            single_view_path = p_b
            single_view = _transition_wrapper(
                html.Iframe(srcDoc=text_b, style={"width": "100%", "height": "100%", "border": "0"}),
                transition_style,
            )
        container = single_view
        path_text = f"Artifact: {single_view_path if single_view_path else 'NOT FOUND'}"

    run_score = f"Run Score | kernel={run.get('kernel', 'unknown')} seed={run.get('seed', 'unknown')} NMI={run.get('nmi', 'n/a')} ARI={run.get('ari', 'n/a')}"

    article_metric_text = "Article Metrics: n/a"
    if effective_observer.startswith("article:"):
        try:
            idx = int(effective_observer.split(":", 1)[1])
            row = INDEX["article_rows_by_run"].get(run_key, {}).get(idx, {})
            if row:
                article_metric_text = (
                    f"Article #{idx} | uid={row.get('bt_uid', 'n/a')} | zone={row.get('zone', 'n/a')} "
                    f"| density={row.get('density', 'n/a')} | stress={row.get('stress', 'n/a')}"
                )
        except Exception:
            pass

    track_state_a = _artifact_track_state(p_a)
    track_state_b = _artifact_track_state(p_b)
    main_path = p_a if (p_a and p_a.exists()) else p_b
    track_readout = _track_status_component(track_state_a if main_path == p_a else track_state_b)
    track_compare_readout = _track_delta_component(track_state_a, track_state_b)
    found, total = _artifact_coverage(run_key, variant_a) if run_key else (0, 0)
    coverage_text = f"Observer Artifact Coverage (Variant A): {found}/{total}" if total > 0 else "Observer Artifact Coverage: n/a"

    state = load_verification_state(run_key, verification_source)
    verification_status = str(state.get("verification_status", "UNVERIFIED")).upper()
    global_pass = state.get("global_pass")

    # VALIDATION TYPE 2: Perspective Sensitivity (Divergence Check)
    type2_dissonance = False
    if compare_enabled and text_a and text_b:
        surv_a = _extract_survival_rate(text_a)
        surv_b = _extract_survival_rate(text_b)
        if surv_a is not None and surv_b is not None:
            if abs(surv_a - surv_b) > 0.20:
                type2_dissonance = True

    gate = _compute_gate_presentation(contract_status, verification_status, global_pass, type2_dissonance)
    claims_enabled = gate["claims_enabled"]
    badge_text = gate["badge_text"]
    base_badge = {"padding": "8px 10px", "borderRadius": "6px", "fontWeight": "700", "letterSpacing": "0.05em", "marginBottom": "10px", "textAlign": "center"}
    if badge_text == "[VERIFIED]":
        badge_style = dict(base_badge, color="#00FF41", border="1px solid #00FF41", backgroundColor="rgba(0,255,65,0.12)", boxShadow="0 0 14px rgba(0,255,65,0.45)", animation="neonPulse 1.4s ease-in-out infinite")
    elif badge_text in {"[TYPE 2 DISSONANCE]", "[INVALID SCHEMA]", "[MISSING ARTIFACTS]", "[NON-COMPARABLE]"}:
        badge_style = dict(base_badge, color=PALETTE["amber"], border=f"1px solid {PALETTE['amber']}", backgroundColor="rgba(255,179,71,0.12)")
    else:
        badge_style = dict(base_badge, color="#FF2A00", border="1px solid #FF2A00", backgroundColor="rgba(255,42,0,0.12)", boxShadow="0 0 12px rgba(255,42,0,0.4)", animation="glitchFlash 0.9s steps(2,end) infinite")

    if claims_enabled:
        t1 = f"System 1: Topologic Integrity | seed_stability={_fmt_pass(state.get('seed_stability'))} | crn_locked={_fmt_pass(state.get('crn_locked'))} [cite: 2026-02-04]"
        t2 = f"System 2: Geometric Friction = {state.get('geometric_friction', 0.0):.3f} (broken={state.get('n_broken', 0)}, trapped={state.get('n_trapped', 0)}) [cite: 2026-02-04]"
        t3 = f"System 2: Survival % = {state.get('survival_pct', 100.0):.2f}% [cite: 2026-02-04]"
    else:
        t1 = "System 1: Claims disabled (verification/provenance gate not satisfied)"
        t2 = "System 2: Claims disabled (exploratory mode)"
        t3 = "System 2: Claims disabled (exploratory mode)"

    detail = f"source={state.get('verification_source')} | verification_summary.csv: {state.get('summary_path')} | verification_report.json: {state.get('report_path')}"
    if missing_required:
        detail += " | missing_required_artifacts=" + ",".join(missing_required)
    if missing_optional:
        detail += " | missing_optional_artifacts=" + ",".join(missing_optional[:5])
    if schema_errors:
        detail += " | schema_errors=" + "; ".join(schema_errors[:3])
    elif contract_errors:
        detail += " | diagnostics=" + "; ".join(contract_errors[:3])
    ab = load_ablation_state(run_key)
    ctrl = load_control_state(run_key)
    ablation_text = (
        f"status={ab.get('status')} | stage1={ab.get('stage_1_nmi', 'n/a')} | stage2={ab.get('stage_2_nmi', 'n/a')} | "
        f"stage3={ab.get('stage_3_nmi', 'n/a')} | delta={ab.get('delta_nmi', 'n/a')} | retained={ab.get('retained_pct', 'n/a')}\n"
        f"source={ab.get('source', 'NOT FOUND')}"
    )
    if ab.get("legacy_mean_variance") is not None:
        ablation_text = (
            f"status=LEGACY | mean_variance={ab.get('legacy_mean_variance')} (no stage NMI fields)\n"
            f"source={ab.get('source', 'NOT FOUND')}"
        )
    control_text = (
        f"status={ctrl.get('status')} | procrustes_ratio={ctrl.get('procrustes_ratio', 'n/a')} | "
        f"distance_corr_ratio={ctrl.get('distance_corr_ratio', 'n/a')} | separates={ctrl.get('separates_count', 0)} | "
        f"consensus={ctrl.get('consensus_pct', 'n/a')}% | residual={ctrl.get('residual_pct', 'n/a')}%\n"
        f"source={ctrl.get('source', 'NOT FOUND')}"
    )

    provenance_line = (
        " | ".join(
            [
                f"weights={baseline_meta.get('weights_hash', 'n/a')}",
                f"dataset={baseline_meta.get('dataset_hash', 'n/a')}",
                f"kernel={baseline_meta.get('kernel_params', 'n/a')}",
                f"rks_dim={baseline_meta.get('rks_dim', 'n/a')}",
                f"seed={baseline_meta.get('crn_seed', 'n/a')}",
                f"alpha={baseline_meta.get('alpha', 'n/a')}",
            ]
        )
    )

    hidden_rows = contract.get("hidden_groups", []) or []
    if hidden_rows:
        hidden_badge_text = "[HIDDEN LABELS READY]"
        hidden_badge_style = {"padding": "6px 8px", "borderRadius": "6px", "textAlign": "center", "color": PALETTE["green"], "border": f"1px solid {PALETTE['green']}", "backgroundColor": "rgba(0,255,65,0.10)"}
        hidden_detail = f"rows={len(hidden_rows)} | group_summaries={contract.get('paths', {}).get('group_summaries', 'NOT FOUND')} | group_matrix={contract.get('paths', {}).get('group_matrix', 'NOT FOUND')}"
    else:
        hidden_badge_text = "[HIDDEN LABELS MISSING]"
        hidden_badge_style = {"padding": "6px 8px", "borderRadius": "6px", "textAlign": "center", "color": PALETTE["amber"], "border": f"1px solid {PALETTE['amber']}", "backgroundColor": "rgba(255,179,71,0.12)"}
        hidden_detail = "Expected labels/hidden_groups.csv and labels/derived/group_*.json"
        if missing_optional:
            hidden_detail += f" | missing_optional_artifacts={','.join(missing_optional)}"

    relativity_panel = _build_relativity_panel(contract, effective_observer, delta_mode, translation_mode_values or [])
    group_panel = _build_group_panel(contract, label_column, label_values or [])
    empathy_fig = _build_empathy_figure(contract, label_column, label_values or [])

    watermark_visible = False
    watermark_text = ""
    watermark_style = {"display": "none"}

    return (
        container,
        path_text,
        run_score,
        article_metric_text,
        track_readout,
        track_compare_readout,
        coverage_text,
        badge_text,
        badge_style,
        t1,
        t2,
        t3,
        detail,
        ablation_text,
        control_text,
        provenance_line,
        hidden_badge_text,
        hidden_badge_style,
        hidden_detail,
        relativity_panel,
        group_panel,
        empathy_fig,
        watermark_text,
        watermark_style,
    )


if __name__ == "__main__":
    run_fn = getattr(app, "run", None)
    if callable(run_fn):
        run_fn(debug=True, host="127.0.0.1", port=8050)
    else:
        app.run_server(debug=True, host="127.0.0.1", port=8050)
