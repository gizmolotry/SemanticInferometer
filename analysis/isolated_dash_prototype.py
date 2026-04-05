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
import hashlib
import json
import math
import os
import re
import sys
from urllib.parse import parse_qs
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import plotly.graph_objects as go
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
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
    if path.name.startswith("observer_"):
        return False
    if list(path.glob("MONOLITH*.html")):
        return True
    return (path / "MONOLITH_DATA.csv").exists()


def _root_has_run_directory(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        if _is_run_directory(path):
            return True
        for child in path.rglob("*"):
            if child.is_dir() and _is_run_directory(child):
                return True
    except Exception:
        return False
    return False


def _discover_artifact_roots() -> List[Path]:
    explicit_roots = [
        ROOT / "experiments_20260221_175416" / "synthetic",
        ROOT / "outputs" / "experiments" / "runs",
        ROOT / "outputs",
    ]
    patterns = (
        "experiments_*/synthetic",
        "experiments/experiments_*/synthetic",
        "outputs/experiments/runs/experiments_*/synthetic",
        "outputs/experiments/runs/*/synthetic",
        "outputs/experiments/runs/experiments_*/*/*/*",
        "outputs/experiments/runs/*/*/*/*",
        "outputs/experiments/*/synthetic",
    )
    seen = set()
    candidate_roots: List[Path] = []
    for pattern in patterns:
        for synthetic in ROOT.glob(pattern):
            key = str(synthetic.resolve())
            if key not in seen and synthetic.exists() and synthetic.is_dir():
                seen.add(key)
                candidate_roots.append(synthetic)
    for explicit in explicit_roots:
        if not explicit.exists():
            continue
        key = str(explicit.resolve())
        if key not in seen:
            seen.add(key)
            candidate_roots.append(explicit)
    viable = [p for p in candidate_roots if _root_has_run_directory(p)]
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
TRACK_ORDER = ["T1", "T1.5", "T2", "T3", "T4", "T5", "T6"]
VERIFICATION_STATUSES = {s.value for s in LayerStatus}


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_artifact_view_state(path: Optional[Path]) -> dict:
    if not path:
        return {}
    candidate = path.with_suffix(".view_state.json")
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _mean_npy(path: Optional[Path]) -> Optional[float]:
    if not path or not path.exists():
        return None
    try:
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return None
    return float(np.nanmean(finite))


def _walker_survival_rate_from_states(path: Optional[Path]) -> Optional[float]:
    if not path or not path.exists():
        return None
    payload = _safe_json(path, [])
    if not isinstance(payload, list):
        return None
    total = 0
    survived = 0
    fatal_states = {
        "FAILED",
        "RUPTURE",
        "BROKEN",
        "TRAPPED",
        "ANOMALY",
        "TYPE 1 RUPTURE",
        "TYPE 2 RUPTURE",
    }
    for item in payload:
        if not isinstance(item, dict):
            continue
        total += 1
        status = str(item.get("status", "")).strip().upper()
        raw_state = str(item.get("raw_state", item.get("label", ""))).strip().upper()
        if status in fatal_states or raw_state in fatal_states:
            continue
        survived += 1
    if total <= 0:
        return None
    return float(survived / total)


def _verdict_counts_from_payload(path: Optional[Path]) -> Dict[str, int]:
    counts = {"honest_count": 0, "phantom_count": 0, "tautology_count": 0, "anomaly_count": 0}
    if not path or not path.exists():
        return counts
    payload = _safe_json(path, [])
    if not isinstance(payload, list):
        return counts
    for item in payload:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "")).strip().upper()
        if verdict == "HONEST":
            counts["honest_count"] += 1
        elif verdict == "PHANTOM":
            counts["phantom_count"] += 1
        elif verdict == "TAUTOLOGY":
            counts["tautology_count"] += 1
        elif verdict in {"ANOMALY", "RUPTURE"} or bool(item.get("anomaly_flag")):
            counts["anomaly_count"] += 1
    return counts


def _has_canonical_t3_payload(run_dir: Optional[Path]) -> bool:
    if not run_dir or not run_dir.exists():
        return False
    if (run_dir / "dirichlet_fused_std.npy").exists():
        return True
    checkpoint_t3 = run_dir / "checkpoints" / "batch" / "T3_topology.npz"
    if checkpoint_t3.exists():
        try:
            with np.load(checkpoint_t3) as payload:
                return "dirichlet_fused_std" in payload.files
        except Exception:
            return False
    return False


def _aggregate_hott_proofs(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    payload = _safe_json(path, [])
    if not isinstance(payload, list):
        return {}
    equivalence = 0
    non_equivalence = 0
    uncertain = 0
    total_conf = 0.0
    counted_conf = 0
    total = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        total += 1
        status = str(item.get("status", "")).strip().lower()
        if status in {"equivalence", "consensus", "elastic", "honest", "tautology"}:
            equivalence += 1
        elif status in {"non_equivalence", "phantom", "anomaly", "rupture"}:
            non_equivalence += 1
        else:
            uncertain += 1
        conf = item.get("confidence")
        if _is_number(conf):
            total_conf += float(conf)
            counted_conf += 1
    if total <= 0:
        return {}
    return {
        "n_proofs": total,
        "equivalence_rate": float(equivalence / total),
        "non_equivalence_rate": float(non_equivalence / total),
        "uncertain_rate": float(uncertain / total),
        "mean_confidence": float(total_conf / counted_conf) if counted_conf > 0 else None,
        "source": "hott_proofs.json",
    }


def _hydrate_artifact_state(run_key: Optional[str], artifact_state: Optional[dict], artifact_path: Optional[Path]) -> dict:
    run_dir = _resolve_run_dir(run_key)
    hydrated = dict(artifact_state) if isinstance(artifact_state, dict) else {}
    metrics = dict(hydrated.get("metrics", {})) if isinstance(hydrated.get("metrics"), dict) else {}
    view_state = _load_artifact_view_state(artifact_path)
    view_metrics = view_state.get("metrics", {}) if isinstance(view_state.get("metrics"), dict) else {}
    if view_metrics:
        metrics.update(view_metrics)
    view_articles = view_state.get("articles")
    if isinstance(view_articles, list) and view_articles:
        hydrated["articles"] = view_articles

    if run_dir:
        if not _is_number(metrics.get("walker_mean_action")):
            mean_action = _mean_npy(run_dir / "walker_work_integrals.npy")
            if mean_action is not None:
                metrics["walker_mean_action"] = mean_action
        if not _is_number(metrics.get("walker_survival_rate")):
            survival = _walker_survival_rate_from_states(run_dir / "walker_states.json")
            if survival is not None:
                metrics["walker_survival_rate"] = survival
        if not any(_is_number(metrics.get(k)) for k in ("honest_count", "phantom_count", "tautology_count", "anomaly_count")):
            verdict_counts = _verdict_counts_from_payload(run_dir / "phantom_verdicts.json")
            metrics.update(verdict_counts)

    hydrated["metrics"] = metrics
    return hydrated


def _load_run_manifest(run_dir: Path) -> dict:
    candidate = run_dir / "MONOLITH.run_manifest.json"
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_json(path: Path, default):
    try:
        return json.loads(_safe_read_text(path))
    except Exception:
        return default


@lru_cache(maxsize=128)
def _cached_payload(path_str: str, mtime_ns: int, size: int) -> dict:
    if not TORCH_AVAILABLE:
        return {}
    try:
        payload = torch.load(path_str, map_location="cpu", weights_only=False)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_payload(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    try:
        stat = path.stat()
    except Exception:
        return {}
    return _cached_payload(str(path), int(stat.st_mtime_ns), int(stat.st_size))


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


def _observer_value_from_uid(run_key: Optional[str], observer_uid: Optional[str]) -> Optional[str]:
    uid = str(observer_uid or "").strip()
    if not uid:
        return None
    rows = INDEX.get("article_rows_by_run", {}).get(str(run_key), {})
    for idx, row in rows.items():
        try:
            row_uid = str((row or {}).get("bt_uid", "")).strip()
        except Exception:
            row_uid = ""
        if row_uid and row_uid == uid:
            return f"article:{int(idx)}"
    return None


def _is_synthetic_placeholder_blob(blob: Any) -> bool:
    if not isinstance(blob, dict) or not blob:
        return False
    if bool(blob.get("synthetic_placeholder", False)):
        return True
    provenance = blob.get("provenance", {})
    if isinstance(provenance, dict):
        source = str(provenance.get("source", "")).strip().lower()
        if source in {"suite-default", "suite-generated", "suite-generated-placeholder"}:
            return True
        if bool(provenance.get("synthetic_placeholder", False)):
            return True
    for key in ("dataset_hash", "code_hash_or_commit", "weights_hash", "provenance_source"):
        val = str(blob.get(key, "")).strip().lower()
        if val in {"suite-generated", "suite-generated-placeholder"}:
            return True
    # Legacy suite-default observer deltas were emitted as an unannotated zero-delta skeleton.
    # Treat that exact shape as synthetic so observer-mode claims do not get a false green light.
    if {
        "observer_id",
        "null_observer_equivalence",
        "path_flip_delta",
        "metrics_delta",
        "axis_delta",
    }.issubset(blob.keys()):
        n = blob.get("null_observer_equivalence", {})
        m = blob.get("metrics_delta", {})
        a = blob.get("axis_delta", {})
        if (
            isinstance(n, dict)
            and isinstance(m, dict)
            and isinstance(a, dict)
            and blob.get("path_flip_delta", {}) in ({}, None)
            and float(n.get("max_coord_delta", 1.0)) == 0.0
            and int(n.get("path_flip_count", 1)) == 0
            and float(n.get("axis_rotation_deg", 1.0)) == 0.0
            and float(m.get("d_rupture_rate", 1.0)) == 0.0
            and float(m.get("d_mean_work", 1.0)) == 0.0
            and float(m.get("d_survival_pct", 1.0)) == 0.0
            and float(a.get("rotation_deg", 1.0)) == 0.0
            and float(a.get("d_explained_variance_axis1", 1.0)) == 0.0
        ):
            return True
    return False


def _artifact_iframe(src_doc: str) -> html.Iframe:
    import time
    # Force a unique key on every render by combining the content hash with a timestamp.
    # This busts the browser's internal iframe cache.
    content_hash = hashlib.sha1(src_doc.encode("utf-8", errors="ignore")).hexdigest()[:12]
    iframe_key = f"{content_hash}_{int(time.time() * 1000)}"
    return html.Iframe(
        key=iframe_key,
        srcDoc=src_doc,
        sandbox="allow-scripts",
        referrerPolicy="no-referrer",
        style={"width": "100%", "height": "100%", "border": "0"},
    )


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

    baseline_meta = _safe_json(diag.paths["baseline_meta.json"], {}) if diag.paths.get("baseline_meta.json") else {}
    baseline_state = _safe_json(diag.paths["baseline_state.json"], {}) if diag.paths.get("baseline_state.json") else {}

    if _is_synthetic_placeholder_blob(baseline_meta):
        missing_required.append("baseline_meta.json (synthetic placeholder)")
        errors.append("baseline_meta is synthetic placeholder data")
        baseline_meta = {}

    observer_id = _observer_id_from_value(observer_value)
    state_blob = {}
    delta_blob = {}
    observer_non_comparable_reasons: List[str] = []
    if observer_id is not None:
        rel_dir = run_dir / "relativity_cache"
        state_path = rel_dir / f"state_{observer_id}.json"
        delta_path = rel_dir / f"delta_{observer_id}.json"
        if state_path.exists():
            state_blob = _safe_json(state_path, {})
            if _is_synthetic_placeholder_blob(state_blob):
                missing_optional.append(f"relativity_cache/state_{observer_id}.json (synthetic placeholder)")
                observer_non_comparable_reasons.append(
                    f"observer relativity state_{observer_id}.json is synthetic placeholder data"
                )
                state_blob = {}
        else:
            missing_optional.append(f"relativity_cache/state_{observer_id}.json")
            observer_non_comparable_reasons.append(
                f"observer relativity state_{observer_id}.json is missing"
            )
        bundled_delta = _load_relativity_delta_bundle(run_dir, observer_id)
        if isinstance(bundled_delta, dict) and bundled_delta:
            delta_blob = {
                "observer_id": bundled_delta.get("observer_id", observer_id),
                "null_observer_equivalence": bundled_delta.get("null_observer_equivalence", {}),
                "path_flip_delta": bundled_delta.get("path_flip_delta", {}),
                "metrics_delta": bundled_delta.get("metrics_delta", {}),
                "axis_delta": bundled_delta.get("axis_delta", {}),
                "translation_only_comparison": bundled_delta.get("translation_only_comparison", {}),
                "provenance": {
                    "source": str(run_dir / "relativity_deltas.json"),
                    "synthetic_placeholder": bool(bundled_delta.get("synthetic_placeholder", False)),
                },
            }
            if _is_synthetic_placeholder_blob(delta_blob):
                missing_optional.append(f"relativity_deltas.json observer {observer_id} (synthetic placeholder)")
                observer_non_comparable_reasons.append(
                    f"observer relativity deltas for observer {observer_id} are synthetic placeholder data"
                )
                delta_blob = {}
        elif delta_path.exists():
            delta_blob = _safe_json(delta_path, {})
            if _is_synthetic_placeholder_blob(delta_blob):
                missing_optional.append(f"relativity_cache/delta_{observer_id}.json (synthetic placeholder)")
                observer_non_comparable_reasons.append(
                    f"observer relativity delta_{observer_id}.json is synthetic placeholder data"
                )
                delta_blob = {}
        else:
            missing_optional.append(f"relativity_cache/delta_{observer_id}.json")
            observer_non_comparable_reasons.append(
                f"observer relativity delta_{observer_id}.json is missing"
            )

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
    if status == "OK" and observer_non_comparable_reasons:
        status = LayerStatus.NON_COMPARABLE.value
    return {
        "status": status,
        "errors": errors + observer_non_comparable_reasons + optional_schema_errors,
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
    seen: set[str] = set()
    try:
        candidates = [artifact_root]
        candidates.extend(p for p in artifact_root.rglob("*") if p.is_dir())
        for d in candidates:
            if not _is_run_directory(d):
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            run_dirs.append(d)
    except Exception:
        return []
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs


def _run_selection_health(run_dir: Path) -> Dict[str, Any]:
    try:
        diag = evaluate_consumer_contract(run_dir)
    except Exception:
        return {
            "score": 0,
            "contract_ok": False,
            "verification_status": LayerStatus.UNVERIFIED.value,
            "schema_errors": ["contract evaluation failed"],
        }

    verification_status = str(diag.verification_status or LayerStatus.UNVERIFIED.value).upper()
    if diag.contract_ok:
        score = 3
    elif verification_status == LayerStatus.NON_COMPARABLE.value:
        # Keep explicit NON_COMPARABLE controls selectable, but do not prefer
        # them over fully valid leaves when auto-selecting a run.
        score = 2
    elif len(diag.missing_required_artifacts) == 0:
        score = 1
    else:
        score = 0
    corpus_name = str(run_dir.name).lower()
    if corpus_name == "real":
        corpus_priority = 4
    elif corpus_name in {"control_shuffled", "control_random"}:
        corpus_priority = 3
    elif corpus_name == "synthetic":
        corpus_priority = 2
    elif corpus_name == "control_constant":
        corpus_priority = 1
    else:
        corpus_priority = 0
    return {
        "score": score,
        "corpus_priority": corpus_priority,
        "contract_ok": bool(diag.contract_ok),
        "verification_status": verification_status,
        "schema_errors": list(diag.schema_errors),
    }


def _preferred_run_key(index: dict) -> str:
    run_keys = list(index.get("run_keys", []))
    runs = index.get("runs", {})
    if not run_keys:
        return ""

    ranked = sorted(
        run_keys,
        key=lambda rk: (
            int((runs.get(rk, {}) or {}).get("selection_score", 0)),
            int((runs.get(rk, {}) or {}).get("selection_root_priority", 0)),
            int((runs.get(rk, {}) or {}).get("selection_model_priority", 0)),
            int((runs.get(rk, {}) or {}).get("selection_corpus_priority", 0)),
            int((runs.get(rk, {}) or {}).get("run_dir").stat().st_mtime_ns if isinstance((runs.get(rk, {}) or {}).get("run_dir"), Path) and (runs.get(rk, {}) or {}).get("run_dir").exists() else 0),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else run_keys[0]


def _run_source_priority(run_dir: Path) -> int:
    run_key = _run_display_key(run_dir).lower()
    if run_key.startswith("outputs/experiments/runs/"):
        return 3
    if run_key.startswith("outputs/experiments/"):
        return 2
    if "ablation" in run_key:
        return 1
    return 0


def _run_model_priority(run_dir: Path) -> int:
    run_key = _run_display_key(run_dir).lower()
    if "/matern/cls/real" in run_key:
        return 4
    if "/rbf/cls/real" in run_key:
        return 3
    if "/cls/real" in run_key:
        return 2
    if run_key.endswith("/real"):
        return 1
    return 0


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


def _preferred_variant(variants: List[str]) -> str:
    if not variants:
        return "MONOLITH.html"
    for name in variants:
        if str(name).upper() == "MONOLITH.HTML":
            return name
    return variants[0]


def _infer_run_metrics(run_dir: Path, summary_item: dict) -> dict:
    item = dict(summary_item or {})
    validation = _safe_json(run_dir / "validation.json", {}) if (run_dir / "validation.json").exists() else {}

    if item.get("nmi") is None and _is_number(validation.get("nmi")):
        item["nmi"] = float(validation.get("nmi"))
    if item.get("ari") is None and _is_number(validation.get("ari")):
        item["ari"] = float(validation.get("ari"))

    kernel = str(item.get("kernel", "unknown") or "unknown")
    if kernel == "unknown":
        try:
            rel_parts = run_dir.relative_to(ROOT).parts
        except Exception:
            rel_parts = ()
        if len(rel_parts) >= 5 and rel_parts[:3] == ("outputs", "experiments", "runs"):
            kernel = str(rel_parts[4] or "unknown")
        elif "_" in run_dir.name:
            kernel = str(run_dir.name.rsplit("_", 1)[-1] or "unknown")
        item["kernel"] = kernel

    seed = item.get("seed", "unknown")
    item["seed"] = str(seed if seed not in (None, "") else "unknown")
    return item


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

    survival_pct = None
    friction = None
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
        family_dir = run_dir.parent
        candidates.extend(
            [
                family_dir / "control_random" / "control_metrics.json",
                family_dir / "control_shuffled" / "control_metrics.json",
                family_dir / "control_constant" / "control_metrics.json",
                run_dir / "control_metrics.json",
                run_dir / "comprehensive_results.json",
                family_dir / "control_metrics.json",
                family_dir / "comprehensive_results.json",
            ]
        )
        for cls_dir in _candidate_cls_dirs(run_key):
            if cls_dir.exists():
                candidates.extend(sorted(cls_dir.glob("comprehensive_analysis*/control_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True))
                candidates.extend(sorted(cls_dir.glob("comprehensive_analysis*/comprehensive_results.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    for fallback in (
        ROOT / "outputs" / "comprehensive_analysis" / "control_metrics.json",
        ROOT / "outputs" / "comprehensive_analysis" / "comprehensive_results.json",
    ):
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


def _empty_panel_state(status: str, source: Path, message: str, fields: List[str]) -> dict:
    payload = {
        "status": status,
        "source": str(source),
        "message": message,
        "explanation": "",
    }
    for field in fields:
        payload[field] = None
    return payload


def _describe_control_explanation(controls: Any) -> str:
    if not isinstance(controls, dict):
        return "Type 1 controls compare the real leaf against matched control siblings."
    present_set = set()
    for raw_name in controls.keys():
        name = str(raw_name).strip().lower()
        if name in {"", "real"}:
            continue
        if name == "constant":
            present_set.add("constant")
        elif name == "shuffled":
            present_set.add("shuffled")
        elif name == "random":
            present_set.add("random")
        else:
            present_set.add(name)
    ordered = ["constant", "shuffled", "random"]
    present = [name for name in ordered if name in present_set]
    present.extend(sorted(name for name in present_set if name not in ordered))
    if not present:
        sibling_phrase = "matched control siblings"
    elif len(present) == 1:
        sibling_phrase = f"the matched {present[0]} control"
    elif len(present) == 2:
        sibling_phrase = f"matched {present[0]} and {present[1]} controls"
    else:
        sibling_phrase = "matched " + ", ".join(present[:-1]) + f", and {present[-1]} controls"
    return f"Type 1 controls compare the real leaf against {sibling_phrase}."


def load_control_state(run_key: Optional[str]) -> dict:
    for path in _control_results_candidates(run_key):
        data = _safe_json(path, {})
        if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
            metrics = data.get("metrics", {})
            status = str(data.get("status", "")).strip().upper()
            if status in {"NO_DATA", "UNAVAILABLE"} or bool(data.get("synthetic_placeholder", False)):
                payload = _empty_panel_state(
                    "NO_DATA" if status == "NO_DATA" or bool(data.get("synthetic_placeholder", False)) else "UNAVAILABLE",
                    path,
                    str(data.get("message", "control analysis unavailable")),
                    ["procrustes_ratio", "distance_corr_ratio", "consensus_pct", "residual_pct"],
                )
                payload["separates_count"] = 0
                payload["summary"] = data.get("summary", {})
                payload["controls"] = data.get("controls", {})
                payload["explanation"] = str(data.get("explanation", ""))
                return payload
            return {
                "status": "OK",
                "source": str(path),
                "message": str(data.get("message", "loaded")),
                "explanation": str(data.get("explanation", "")),
                "procrustes_ratio": metrics.get("procrustes_ratio"),
                "distance_corr_ratio": metrics.get("distance_corr_ratio"),
                "separates_count": int(metrics.get("separates_count", 0) or 0),
                "consensus_pct": metrics.get("consensus_pct"),
                "residual_pct": metrics.get("residual_pct"),
                "summary": data.get("summary", {}),
                "controls": data.get("controls", {}),
            }
        interp = data.get("interpretation", {}) if isinstance(data, dict) else {}
        if not isinstance(interp, dict):
            continue
        if interp.get("error"):
            payload = _empty_panel_state(
                "UNAVAILABLE",
                path,
                str(interp.get("error")),
                ["procrustes_ratio", "distance_corr_ratio", "consensus_pct", "residual_pct"],
            )
            payload["separates_count"] = 0
            payload["summary"] = {}
            payload["controls"] = {}
            return payload
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
            "explanation": _describe_control_explanation(data.get("results", {})),
            "procrustes_ratio": None if str(procrustes_ratio) == "nan" else procrustes_ratio,
            "distance_corr_ratio": None if str(distance_corr_ratio) == "nan" else distance_corr_ratio,
            "separates_count": int(separates_count),
            "consensus_pct": None if str(consensus_pct) == "nan" else consensus_pct,
            "residual_pct": None if str(residual_pct) == "nan" else residual_pct,
            "summary": {},
            "controls": data.get("results", {}) if isinstance(data, dict) else {},
        }
    return {
        "status": "MISSING",
        "source": "NOT FOUND",
        "message": "no control analysis results",
        "explanation": "Type 1 controls are missing for this leaf.",
        "procrustes_ratio": None,
        "distance_corr_ratio": None,
        "separates_count": 0,
        "consensus_pct": None,
        "residual_pct": None,
        "summary": {},
        "controls": {},
    }


def _load_relativity_delta_bundle(run_dir: Path, observer_id: int) -> dict:
    bundle_path = run_dir / "relativity_deltas.json"
    if not bundle_path.exists():
        return {}
    bundle = _safe_json(bundle_path, {})
    if not isinstance(bundle, dict):
        return {}
    observers = bundle.get("observers", [])
    if not isinstance(observers, list):
        return {}
    for observer in observers:
        if not isinstance(observer, dict):
            continue
        try:
            if int(observer.get("observer_id")) != int(observer_id):
                continue
        except Exception:
            continue
        return observer
    return {}


def _ablation_candidates(run_key: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    run_dir = _resolve_run_dir(run_key)
    if run_dir and run_dir.exists():
        candidates.extend(
            [
                run_dir / "ablation_summary.json",
                run_dir / "ablation_results.json",
                run_dir / "lab_diagnostics.json",
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
            if "procrustes" in blob and "structural_invariants" in blob:
                procrustes = blob.get("procrustes", {}) if isinstance(blob.get("procrustes"), dict) else {}
                invariants = blob.get("structural_invariants", {}) if isinstance(blob.get("structural_invariants"), dict) else {}
                mean_before = _safe_float(procrustes.get("mean_distance_before"), default=float("nan"))
                mean_after = _safe_float(procrustes.get("mean_distance_after"), default=float("nan"))
                survival_rate = _safe_float(invariants.get("mean_survival_rate"), default=float("nan"))
                stage_1 = None if str(mean_before) == "nan" else float(1.0 / (1.0 + max(mean_before, 0.0)))
                stage_2 = None if str(mean_after) == "nan" else float(1.0 / (1.0 + max(mean_after, 0.0)))
                stage_3 = None if str(survival_rate) == "nan" else survival_rate
                return {
                    "status": "OK",
                    "source": str(path),
                    "message": "translated from lab_diagnostics.json",
                    "explanation": "Ablation laboratory output translated from observer alignment distances and invariant survival rates. Alignment scores are distance-derived proxies, not literal NMI.",
                    "stage_1_alignment_score": stage_1,
                    "stage_2_alignment_score": stage_2,
                    "stage_3_survival_rate": stage_3,
                    "delta_alignment_score": (stage_2 - stage_1) if stage_1 is not None and stage_2 is not None else None,
                    "stage_1_nmi": stage_1,
                    "stage_2_nmi": stage_2,
                    "stage_3_nmi": stage_3,
                    "delta_nmi": (stage_2 - stage_1) if stage_1 is not None and stage_2 is not None else None,
                    "retained_pct": None if stage_3 is None else float(stage_3 * 100.0),
                    "legacy_mean_variance": None,
                    "summary": {
                        "stage_map": {
                            "stage_1_alignment_score": "alignment before repair / null-side similarity proxy",
                            "stage_2_alignment_score": "alignment after repair / mature-side similarity proxy",
                            "stage_3_survival_rate": "structural invariant survival rate",
                        }
                    },
                }
            status_marker = str(blob.get("status", "")).strip().upper()
            if status_marker in {"NO_DATA", "UNAVAILABLE"} or bool(blob.get("synthetic_placeholder", False)):
                payload = _empty_panel_state(
                    "NO_DATA" if status_marker == "NO_DATA" or bool(blob.get("synthetic_placeholder", False)) else "UNAVAILABLE",
                    path,
                    str(blob.get("reason") or blob.get("message") or "ablation analysis unavailable"),
                    ["stage_1_nmi", "stage_2_nmi", "stage_3_nmi", "delta_nmi", "retained_pct", "legacy_mean_variance"],
                )
                payload["summary"] = blob.get("summary", {})
                payload["explanation"] = str(blob.get("explanation", ""))
                payload["stage_1_alignment_score"] = blob.get("stage_1_alignment_score")
                payload["stage_2_alignment_score"] = blob.get("stage_2_alignment_score")
                payload["stage_3_survival_rate"] = blob.get("stage_3_survival_rate")
                payload["delta_alignment_score"] = blob.get("delta_alignment_score")
                return payload
            metrics = blob.get("metrics", {})
            metrics = metrics if isinstance(metrics, dict) else {}
            a1 = blob.get("stage_1_alignment_score", metrics.get("stage_1_alignment_score"))
            a2 = blob.get("stage_2_alignment_score", metrics.get("stage_2_alignment_score"))
            a3 = blob.get("stage_3_survival_rate", metrics.get("stage_3_survival_rate"))
            adelta = blob.get("delta_alignment_score", metrics.get("delta_alignment_score"))
            s1 = blob.get("stage_1_nmi", metrics.get("stage_1_nmi"))
            s2 = blob.get("stage_2_nmi", metrics.get("stage_2_nmi"))
            s3 = blob.get("stage_3_nmi", metrics.get("stage_3_nmi"))
            delta = blob.get("delta_nmi", metrics.get("delta_nmi"))
            if a1 is None:
                a1 = s1
            if a2 is None:
                a2 = s2
            if a3 is None:
                a3 = s3
            if adelta is None:
                adelta = delta
            retained = blob.get("retained_percentage", metrics.get("retained_pct"))
            legacy_mean_variance = blob.get("legacy_mean_variance", metrics.get("legacy_mean_variance"))
            if any(v is not None for v in (s1, s2, s3, delta, retained)):
                return {
                    "status": "OK",
                    "source": str(path),
                    "message": str(blob.get("message", "loaded")),
                    "explanation": str(blob.get("explanation", "")),
                    "stage_1_alignment_score": a1,
                    "stage_2_alignment_score": a2,
                    "stage_3_survival_rate": a3,
                    "delta_alignment_score": adelta,
                    "stage_1_nmi": s1,
                    "stage_2_nmi": s2,
                    "stage_3_nmi": s3,
                    "delta_nmi": delta,
                    "retained_pct": retained,
                    "legacy_mean_variance": legacy_mean_variance,
                    "summary": blob.get("summary", {}),
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
                        "message": "loaded from csv summary",
                        "explanation": "Legacy CSV ablation summary loaded.",
                        "stage_1_alignment_score": first.get("stage_1_alignment_score", first.get("stage_1_nmi")),
                        "stage_2_alignment_score": first.get("stage_2_alignment_score", first.get("stage_2_nmi")),
                        "stage_3_survival_rate": first.get("stage_3_survival_rate", first.get("stage_3_nmi")),
                        "delta_alignment_score": first.get("delta_alignment_score", first.get("delta_nmi")),
                        "stage_1_nmi": first.get("stage_1_nmi"),
                        "stage_2_nmi": first.get("stage_2_nmi"),
                        "stage_3_nmi": first.get("stage_3_nmi"),
                        "delta_nmi": first.get("delta_nmi"),
                        "retained_pct": first.get("retained_percentage"),
                        "legacy_mean_variance": None,
                        "summary": {},
                    }
                if "mean_variance" in first:
                    return {
                        "status": "LEGACY",
                        "source": str(path),
                        "message": "loaded legacy variance-only ablation output",
                        "explanation": "Legacy ablation output predates the stage-based schema.",
                        "stage_1_nmi": None,
                        "stage_2_nmi": None,
                        "stage_3_nmi": None,
                        "delta_nmi": None,
                        "retained_pct": None,
                        "legacy_mean_variance": first.get("mean_variance"),
                        "summary": {},
                    }
            except Exception:
                continue
    return {
        "status": "MISSING",
        "source": "NOT FOUND",
        "message": "no ablation analysis results",
        "explanation": "No ablation laboratory outputs were found for this leaf.",
        "stage_1_alignment_score": None,
        "stage_2_alignment_score": None,
        "stage_3_survival_rate": None,
        "delta_alignment_score": None,
        "stage_1_nmi": None,
        "stage_2_nmi": None,
        "stage_3_nmi": None,
        "delta_nmi": None,
        "retained_pct": None,
        "legacy_mean_variance": None,
        "summary": {},
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


def _safe_json_list(path: Optional[Path], default: Optional[List[Any]] = None) -> List[Any]:
    default = default or []
    if not path or not path.exists():
        return list(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return payload if isinstance(payload, list) else list(default)
    except Exception:
        return list(default)


def _fmt_metric(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return str(value)
    if _is_number(value):
        num = float(value)
        if math.isfinite(num):
            return f"{num:.{digits}f}"
    return "n/a"


def _panel_status_style(status: str) -> Dict[str, str]:
    status_upper = str(status or "").upper()
    if status_upper == "OK":
        accent = PALETTE["green"]
        bg = "rgba(0,255,65,0.08)"
    elif status_upper in {"NO_DATA", "NON_COMPARABLE"}:
        accent = PALETTE["amber"]
        bg = "rgba(255,179,71,0.10)"
    elif status_upper in {"UNAVAILABLE", "MISSING", "INVALID_SCHEMA"}:
        accent = PALETTE["red"]
        bg = "rgba(255,42,0,0.10)"
    else:
        accent = PALETTE["cyan"]
        bg = "rgba(0,240,255,0.08)"
    return {"accent": accent, "background": bg}


def _status_pill(status: str, label: Optional[str] = None):
    style = _panel_status_style(status)
    return html.Span(
        label or str(status).upper(),
        style={
            "display": "inline-block",
            "padding": "3px 8px",
            "borderRadius": "999px",
            "fontSize": "0.72rem",
            "fontWeight": "700",
            "letterSpacing": "0.05em",
            "color": style["accent"],
            "border": f"1px solid {style['accent']}",
            "backgroundColor": style["background"],
        },
    )


def _metric_tile(label: str, value: Any, hint: Optional[str] = None):
    return html.Div(
        [
            html.Div(label, style={"color": PALETTE["dim"], "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.06em"}),
            html.Div(_fmt_metric(value) if _is_number(value) else (str(value) if value not in (None, "") else "n/a"), style={"color": PALETTE["text"], "fontSize": "1rem", "fontWeight": "700"}),
            html.Div(hint or "", style={"color": PALETTE["dim"], "fontSize": "0.7rem", "lineHeight": "1.2"}) if hint else None,
        ],
        style={
            "padding": "8px 10px",
            "borderRadius": "10px",
            "border": f"1px solid {PALETTE['grid']}",
            "background": "linear-gradient(180deg, rgba(15,15,24,0.96) 0%, rgba(9,9,14,0.96) 100%)",
            "minHeight": "68px",
        },
    )


def _panel_shell(title: str, subtitle: str, status: str, children: List[Any]):
    style = _panel_status_style(status)
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(title, style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem"}),
                            html.Div(subtitle, style={"color": PALETTE["dim"], "fontSize": "0.72rem", "lineHeight": "1.3", "marginTop": "2px"}),
                        ],
                        style={"flex": "1"},
                    ),
                    _status_pill(status),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "flex-start", "marginBottom": "8px"},
            ),
            html.Div(children, style={"display": "grid", "gap": "8px"}),
        ],
        style={
            "padding": "10px",
            "borderRadius": "12px",
            "border": f"1px solid {style['accent']}",
            "background": f"linear-gradient(180deg, {style['background']} 0%, rgba(8,8,12,0.96) 100%)",
            "boxShadow": f"0 0 0 1px {style['background']}",
        },
    )


def _load_validation_payload(run_key: Optional[str]) -> dict:
    run_dir = _resolve_run_dir(run_key)
    if not run_dir:
        return {}
    return _safe_json(run_dir / "validation.json", {})


def _load_hott_summary(run_key: Optional[str]) -> dict:
    run_dir = _resolve_run_dir(run_key)
    if not run_dir:
        return {}
    summary_path = run_dir / "hott_summary.json"
    if summary_path.exists():
        return _safe_json(summary_path, {})
    return _aggregate_hott_proofs(run_dir / "hott_proofs.json")


def _compute_track_snapshot(
    run_key: Optional[str],
    artifact_state: Optional[dict],
    contract: Optional[dict],
    observer_value: str = "global",
) -> Dict[str, Dict[str, Any]]:
    run_dir = _resolve_run_dir(run_key)
    validation = _load_validation_payload(run_key)
    track_metrics = validation.get("track_metrics", {}) if isinstance(validation.get("track_metrics"), dict) else {}
    artifact_metrics = artifact_state.get("metrics", {}) if isinstance(artifact_state, dict) and isinstance(artifact_state.get("metrics"), dict) else {}
    observer_metrics = {}
    if isinstance(contract, dict):
        observer_state = contract.get("observer_state", {}) or {}
        if isinstance(observer_state, dict) and isinstance(observer_state.get("metrics"), dict):
            observer_metrics = observer_state.get("metrics", {}) or {}
    observer_track_nmi = observer_metrics.get("observer_track_nmi", {}) if isinstance(observer_metrics.get("observer_track_nmi"), dict) else {}
    snapshot: Dict[str, Dict[str, Any]] = {}

    def _base_track(track_key: str) -> Dict[str, Any]:
        metrics = track_metrics.get(track_key, {}) if isinstance(track_metrics.get(track_key), dict) else {}
        item = {
            "status": "online" if metrics else "missing",
            "source": f"validation.track_metrics.{track_key}" if metrics else "missing",
            "nmi": float(metrics.get("nmi")) if _is_number(metrics.get("nmi")) else None,
            "ari": float(metrics.get("ari")) if _is_number(metrics.get("ari")) else None,
        }
        obs_nmi = observer_track_nmi.get(track_key)
        if observer_value.startswith("article:") and _is_number(obs_nmi):
            item["nmi"] = float(obs_nmi)
            item["source"] = f"observer_state.metrics.observer_track_nmi.{track_key}"
        return item

    snapshot["T1"] = _base_track("T1")
    snapshot["T1.5"] = _base_track("T1.5")
    if _is_number(artifact_metrics.get("spectral_signal")):
        snapshot["T1.5"]["signal"] = float(artifact_metrics.get("spectral_signal"))
    snapshot["T2"] = _base_track("T2")
    t3_canonical = _has_canonical_t3_payload(run_dir)
    snapshot["T3"] = _base_track("T3")
    if not t3_canonical:
        snapshot["T3"]["status"] = "missing"
        snapshot["T3"]["source"] = "missing"
        snapshot["T3"]["nmi"] = None
        snapshot["T3"]["ari"] = None
    else:
        if _is_number(artifact_metrics.get("dirichlet_bonds")):
            snapshot["T3"]["bonds"] = int(float(artifact_metrics.get("dirichlet_bonds")))
        if _is_number(artifact_metrics.get("dirichlet_cracks")):
            snapshot["T3"]["cracks"] = int(float(artifact_metrics.get("dirichlet_cracks")))

    t4_exists = bool(run_dir and ((run_dir / "walker_paths.npz").exists() or (run_dir / "walker_states.json").exists() or (run_dir / "walker_work_integrals.npy").exists()))
    t4_has_metrics = _is_number(artifact_metrics.get("walker_mean_action")) or _is_number(artifact_metrics.get("walker_survival_rate"))
    snapshot["T4"] = {
        "status": "online" if t4_has_metrics else ("partial" if t4_exists else "missing"),
        "source": "artifact.metrics.walker_*" if (t4_has_metrics or t4_exists) else "missing",
        "action": float(artifact_metrics.get("walker_mean_action")) if _is_number(artifact_metrics.get("walker_mean_action")) else None,
        "survival": float(artifact_metrics.get("walker_survival_rate")) if _is_number(artifact_metrics.get("walker_survival_rate")) else None,
    }

    t5_exists = bool(run_dir and (run_dir / "phantom_verdicts.json").exists())
    t5_has_metrics = any(
        _is_number(artifact_metrics.get(k))
        for k in ("honest_count", "phantom_count", "tautology_count", "anomaly_count")
    )
    snapshot["T5"] = {
        "status": "online" if t5_has_metrics else ("partial" if t5_exists else "missing"),
        "source": "artifact.metrics.*_count" if (t5_has_metrics or t5_exists) else "missing",
        "honest": int(float(artifact_metrics.get("honest_count"))) if _is_number(artifact_metrics.get("honest_count")) else None,
        "phantom": int(float(artifact_metrics.get("phantom_count"))) if _is_number(artifact_metrics.get("phantom_count")) else None,
        "tautology": int(float(artifact_metrics.get("tautology_count"))) if _is_number(artifact_metrics.get("tautology_count")) else None,
        "anomaly": int(float(artifact_metrics.get("anomaly_count"))) if _is_number(artifact_metrics.get("anomaly_count")) else None,
    }

    hott_summary = _load_hott_summary(run_key)
    t6_exists = bool(run_dir and ((run_dir / "hott_summary.json").exists() or (run_dir / "hott_proofs.json").exists()))
    t6_has_metrics = any(_is_number(hott_summary.get(k)) for k in ("n_proofs", "equivalence_rate", "mean_confidence"))
    snapshot["T6"] = {
        "status": "online" if t6_has_metrics else ("partial" if t6_exists else "missing"),
        "source": str(hott_summary.get("source", "hott_summary.json")) if (t6_has_metrics or t6_exists) else "missing",
        "n_proofs": int(float(hott_summary.get("n_proofs"))) if _is_number(hott_summary.get("n_proofs")) else None,
        "equivalence_rate": float(hott_summary.get("equivalence_rate")) if _is_number(hott_summary.get("equivalence_rate")) else None,
        "mean_confidence": float(hott_summary.get("mean_confidence")) if _is_number(hott_summary.get("mean_confidence")) else None,
    }

    syn_metrics = track_metrics.get("SYN", {}) if isinstance(track_metrics.get("SYN"), dict) else {}
    syn_nmi = artifact_metrics.get("synthesis_nmi")
    if not _is_number(syn_nmi):
        syn_nmi = observer_metrics.get("observer_conditioned_nmi") if observer_value.startswith("article:") else syn_metrics.get("nmi")
    snapshot["SYN"] = {
        "status": "online" if (_is_number(syn_nmi) or syn_metrics) else "missing",
        "source": "artifact.metrics.synthesis_nmi" if _is_number(artifact_metrics.get("synthesis_nmi")) else ("observer_state.metrics.observer_conditioned_nmi" if observer_value.startswith("article:") and _is_number(observer_metrics.get("observer_conditioned_nmi")) else "validation.track_metrics.SYN"),
        "nmi": float(syn_nmi) if _is_number(syn_nmi) else None,
        "ari": float(syn_metrics.get("ari")) if _is_number(syn_metrics.get("ari")) else (float(observer_metrics.get("observer_conditioned_ari")) if observer_value.startswith("article:") and _is_number(observer_metrics.get("observer_conditioned_ari")) else None),
    }
    return snapshot


def _compute_track_delta_summary(
    snapshot_a: Dict[str, Dict[str, Any]],
    snapshot_b: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for track in TRACK_ORDER + ["SYN"]:
        a = snapshot_a.get(track, {})
        b = snapshot_b.get(track, {})
        status_a = a.get("status", "missing")
        status_b = b.get("status", "missing")
        entry: Dict[str, Any] = {
            "status_a": status_a,
            "status_b": status_b,
            "same_status": status_a == status_b,
            "delta": False,
            "summary": "",
        }
        if track in {"T1", "T1.5", "T2", "T3", "SYN"}:
            a_nmi = a.get("nmi")
            b_nmi = b.get("nmi")
            if _is_number(a_nmi) and _is_number(b_nmi):
                delta_nmi = float(b_nmi) - float(a_nmi)
                entry["delta_nmi"] = delta_nmi
                entry["delta"] = abs(delta_nmi) > 1e-9
                entry["summary"] = f"NMI A={_fmt_metric(a_nmi)} B={_fmt_metric(b_nmi)} d={delta_nmi:+.3f}"
            elif status_a != status_b:
                entry["delta"] = True
                entry["summary"] = f"status A={status_a.upper()} B={status_b.upper()}"
        elif track == "T4":
            a_action = a.get("action")
            b_action = b.get("action")
            a_surv = a.get("survival")
            b_surv = b.get("survival")
            parts: List[str] = []
            if _is_number(a_action) and _is_number(b_action):
                delta_action = float(b_action) - float(a_action)
                entry["delta_action"] = delta_action
                entry["delta"] = entry["delta"] or abs(delta_action) > 1e-9
                parts.append(f"work A={_fmt_metric(a_action)} B={_fmt_metric(b_action)} d={delta_action:+.3f}")
            if _is_number(a_surv) and _is_number(b_surv):
                delta_surv = float(b_surv) - float(a_surv)
                entry["delta_survival"] = delta_surv
                entry["delta"] = entry["delta"] or abs(delta_surv) > 1e-9
                parts.append(f"closed-loop A={_fmt_metric(a_surv)} B={_fmt_metric(b_surv)} d={delta_surv:+.3f}")
            entry["summary"] = " | ".join(parts) if parts else f"status A={status_a.upper()} B={status_b.upper()}"
        elif track == "T5":
            keys = ["honest", "phantom", "tautology", "anomaly"]
            parts = []
            for key in keys:
                av = a.get(key)
                bv = b.get(key)
                if _is_number(av) and _is_number(bv):
                    delta_v = int(float(bv) - float(av))
                    if delta_v != 0:
                        entry["delta"] = True
                    parts.append(f"{key[0].upper()} d={delta_v:+d}")
            entry["summary"] = " | ".join(parts) if parts else f"status A={status_a.upper()} B={status_b.upper()}"
        elif track == "T6":
            a_eq = a.get("equivalence_rate")
            b_eq = b.get("equivalence_rate")
            if _is_number(a_eq) and _is_number(b_eq):
                delta_eq = float(b_eq) - float(a_eq)
                entry["delta_equivalence"] = delta_eq
                entry["delta"] = abs(delta_eq) > 1e-9
                entry["summary"] = f"equiv A={_fmt_metric(a_eq)} B={_fmt_metric(b_eq)} d={delta_eq:+.3f}"
            else:
                entry["summary"] = f"status A={status_a.upper()} B={status_b.upper()}"
        summary[track] = entry
    return summary


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
    detail_lines: List[str] = []
    for track in TRACK_ORDER:
        payload = track_state.get(track, {}) if isinstance(track_state.get(track), dict) else {"status": track_state.get(track, "unknown")}
        st = payload.get("status", "unknown")
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
        if track in {"T1", "T1.5", "T2", "T3"}:
            line = f"{track}: NMI={_fmt_metric(payload.get('nmi'))}"
            if _is_number(payload.get("ari")):
                line += f" | ARI={_fmt_metric(payload.get('ari'))}"
            if track == "T1.5" and _is_number(payload.get("signal")):
                line += f" | Signal={_fmt_metric(payload.get('signal'))}"
            if track == "T3" and (_is_number(payload.get("bonds")) or _is_number(payload.get("cracks"))):
                line += f" | bonds/cracks={payload.get('bonds', 'n/a')}/{payload.get('cracks', 'n/a')}"
        elif track == "T4":
            line = f"T4: work={_fmt_metric(payload.get('action'))} | closed-loop={_fmt_metric(payload.get('survival'))}"
        elif track == "T5":
            line = f"T5 terminal labels: H={payload.get('honest', 'n/a')} P={payload.get('phantom', 'n/a')} T={payload.get('tautology', 'n/a')} A={payload.get('anomaly', 'n/a')}"
        else:
            line = f"T6: proofs={payload.get('n_proofs', 'n/a')} | equiv={_fmt_metric(payload.get('equivalence_rate'))} | conf={_fmt_metric(payload.get('mean_confidence'))}"
        detail_lines.append(line)
    return html.Div([
        html.Div(chips, style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}),
        html.Pre("\n".join(detail_lines), style={"margin": "6px 0 0 0", "color": PALETTE["dim"], "fontSize": "0.74rem", "whiteSpace": "pre-wrap"})
    ])


def _track_delta_component(track_a: Dict[str, str], track_b: Dict[str, str]):
    chips = []
    detail_lines: List[str] = []
    summary = _compute_track_delta_summary(track_a, track_b)
    for track in TRACK_ORDER:
        payload = summary.get(track, {})
        a = payload.get("status_a", "unknown")
        b = payload.get("status_b", "unknown")
        same = bool(payload.get("same_status", False)) and not bool(payload.get("delta", False))
        if same:
            color = PALETTE["green"] if a == "online" else PALETTE["amber"]
            text = f"{track}:A={a.upper()} B={b.upper()}"
        else:
            color = PALETTE["red"] if payload.get("delta") or a != b else PALETTE["amber"]
            text = f"{track}:A={a.upper()} B={b.upper()}" + (" DELTA" if (payload.get("delta") or a != b) else "")
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
        detail_lines.append(f"{track}: {payload.get('summary', 'n/a')}")
    syn_payload = summary.get("SYN", {})
    if syn_payload:
        detail_lines.insert(0, f"SYN: {syn_payload.get('summary', 'n/a')}")
    return html.Div([
        html.Div(chips, style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}),
        html.Pre("\n".join(detail_lines), style={"margin": "6px 0 0 0", "color": PALETTE["dim"], "fontSize": "0.74rem", "whiteSpace": "pre-wrap"})
    ])


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
        if run_key in runs:
            continue
        root_key = str(run_dir.parent)
        item = _infer_run_metrics(run_dir, metrics_by_root.get(root_key, {}).get(run_dir.name, {}))
        run_manifest = _load_run_manifest(run_dir)
        manifest_artifacts = run_manifest.get("artifacts", []) if isinstance(run_manifest, dict) else []
        variants = [str(a.get("html", "")).strip() for a in manifest_artifacts if isinstance(a, dict) and str(a.get("html", "")).strip()]
        if not variants:
            variants = _collect_variant_names(run_dir)
        primary_metrics = run_manifest.get("primary_metrics", {}) if isinstance(run_manifest, dict) else {}
        if item.get("nmi") is None and _is_number(primary_metrics.get("synthesis_nmi")):
            item["nmi"] = float(primary_metrics.get("synthesis_nmi"))
        selection_health = _run_selection_health(run_dir)
        run = {
            "run_key": run_key,
            "kernel": str(item.get("kernel", "unknown")),
            "seed": str(item.get("seed", "unknown")),
            "nmi": item.get("nmi"),
            "ari": item.get("ari"),
            "run_dir": run_dir,
            "artifact_root": run_dir.parent,
            "variants": variants,
            "primary_variant": _preferred_variant(variants),
            "article_meta_path": run_dir / "article_metadata.json",
            "monolith_data_path": run_dir / "MONOLITH_DATA.csv",
            "observer_manifest_path": run_dir / "observer_manifest.json",
            "observer_manifest": {},
            "observer_artifacts": {},
            "run_manifest": run_manifest if isinstance(run_manifest, dict) else {},
            "selection_score": int(selection_health.get("score", 0)),
            "selection_root_priority": _run_source_priority(run_dir),
            "selection_model_priority": _run_model_priority(run_dir),
            "selection_corpus_priority": int(selection_health.get("corpus_priority", 0)),
            "contract_ok": bool(selection_health.get("contract_ok", False)),
            "selection_verification_status": str(selection_health.get("verification_status", LayerStatus.UNVERIFIED.value)),
            "selection_schema_errors": list(selection_health.get("schema_errors", [])),
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

    default_run = _preferred_run_key({"run_keys": run_keys, "runs": runs}) if run_keys else ""

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
    
    found_path = None
    # Observer-specific naming support (future-compatible) should take priority
    # for article viewpoints, then fallback to chosen global variant.
    if observer_value.startswith("article:"):
        manifest = run.get("observer_manifest") or {}
        manifest_variant = str(manifest.get("variant", "")).strip()
        manifest_hit = run.get("observer_artifacts", {}).get(observer_value)
        if (
            manifest_hit
            and manifest_hit.exists()
            and (not manifest_variant or manifest_variant == selected_variant)
        ):
            found_path = manifest_hit

        if not found_path:
            idx = observer_value.split(":", 1)[1]
            stem = Path(selected_variant).stem
            observer_candidates = [
                run_dir / f"observer_{idx}" / selected_variant,
                run_dir / f"article_{idx}" / selected_variant,
                run_dir / f"{stem}_article_{idx}.html",
                run_dir / f"{stem}_observer_{idx}.html",
            ]
            monolith_named = str(selected_variant).strip().upper() == "MONOLITH.HTML"
            if monolith_named:
                observer_candidates.extend(
                    [
                        run_dir / f"observer_{idx}" / "MONOLITH.html",
                        run_dir / f"article_{idx}" / "MONOLITH.html",
                        run_dir / f"MONOLITH_article_{idx}.html",
                        run_dir / f"MONOLITH_observer_{idx}.html",
                    ]
                )
            for p in observer_candidates:
                if p.exists():
                    found_path = p
                    break
    
    if not found_path:
        variant_path = run_dir / selected_variant
        if variant_path.exists():
            found_path = variant_path
        else:
            for fallback_name in run.get("variants", []):
                p = run_dir / str(fallback_name)
                if p.exists():
                    found_path = p
                    break
    
    if found_path:
        print(f"[DASH DEBUG] Serving artifact: {found_path}")
    return found_path


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


app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="Artifact Viewer")

RUN_OPTIONS = _run_options_from_index(INDEX)
DEFAULT_RUN = INDEX["default_run"]
DEFAULT_VARIANTS = INDEX["runs"].get(DEFAULT_RUN, {}).get("variants", ["MONOLITH.html"])
DEFAULT_PRIMARY_VARIANT = _preferred_variant(DEFAULT_VARIANTS)

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": PALETTE["void"], "minHeight": "100vh", "padding": "0"},
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(style={"display": "none"}),
        dcc.Store(id="gallery-dir", data={"dir": 1}),
        dcc.Store(id="hotkey-signal", data={"event": "none", "seq": 0}),
        dcc.Store(id="index-revision", data=0),
        dcc.Interval(id="hotkey-poll", interval=250, n_intervals=0, disabled=True),
        dcc.Interval(id="verification-poll", interval=10000, n_intervals=0, disabled=True),
        dcc.Interval(id="gallery-interval", interval=2200, n_intervals=0, disabled=True),
        dbc.Row(
            className="g-0",
            style={"minHeight": "100vh"},
            children=[
                dbc.Col(
                    xs=12,
                    md=2,
                    lg=2,
                    style={
                        "background": "linear-gradient(180deg, #050505 0%, #0b0b14 100%)",
                        "borderRight": f"1px solid {PALETTE['grid']}",
                        "padding": "10px 12px",
                        "maxHeight": "100vh",
                        "overflowY": "auto",
                    },
                    children=[
                        html.Div(
                            id="artifact-root-info",
                            children=f"Artifact Roots: {INDEX.get('artifact_root_count', 0)} | Primary: {INDEX.get('artifact_root') or 'NOT FOUND'}",
                            style={"color": PALETTE["dim"], "fontSize": "0.74rem", "wordBreak": "break-all", "marginBottom": "4px"},
                        ),
                        html.Div(
                            id="physical-path-readout",
                            children="PHYSICAL PATH: INITIALIZING...",
                            style={"color": PALETTE["cyan"], "fontSize": "0.68rem", "wordBreak": "break-all", "marginBottom": "8px", "fontWeight": "700"},
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
                        html.Div(
                            style={"display": "none"},
                            children=[
                                dcc.RadioItems(
                                    id="view-mode",
                                    options=[{"label": "Global", "value": "global"}, {"label": "Observer", "value": "observer"}],
                                    value="global",
                                ),
                                dcc.RadioItems(
                                    id="delta-mode",
                                    options=[{"label": "Baseline", "value": "baseline"}, {"label": "Observer", "value": "observer"}, {"label": "Delta", "value": "delta"}],
                                    value="baseline",
                                ),
                                dcc.Checklist(
                                    id="translation-mode",
                                    options=[{"label": "Translation Only", "value": "translation_only"}],
                                    value=[],
                                ),
                                dcc.Checklist(
                                    id="failure-overlays",
                                    options=[{"label": "Show Failure Overlays", "value": "on"}],
                                    value=[],
                                ),
                            ],
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
                                dbc.Col([html.Label("Variant B", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.82rem"}), dcc.Dropdown(id="variant-b-dropdown", options=[{"label": v, "value": v} for v in DEFAULT_VARIANTS], value=DEFAULT_PRIMARY_VARIANT if DEFAULT_VARIANTS else "MONOLITH.html", clearable=False, style={"color": "#111"})], width=6),
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
                        html.Div(id="ablation-metrics", style={"marginBottom": "8px"}),
                        html.Div(id="control-metrics", style={"marginBottom": "6px"}),
                        html.Hr(style={"borderColor": PALETTE["grid"], "margin": "10px 0"}),
                        html.Div(id="relativity-panel", style={"padding": "8px", "maxHeight": "24vh", "overflowY": "auto", "border": f"1px solid {PALETTE['grid']}", "borderRadius": "6px", "marginBottom": "8px"}),
                        html.Div("Group Path Patterns", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        html.Div(id="group-panel", style={"padding": "8px", "maxHeight": "22vh", "overflowY": "auto", "border": f"1px solid {PALETTE['grid']}", "borderRadius": "6px", "marginBottom": "8px"}),
                        html.Div("Empathy Gap", style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.84rem", "marginBottom": "4px"}),
                        dcc.Graph(id="empathy-heatmap", style={"height": "28vh", "marginBottom": "8px"}),
                    ],
                ),
                dbc.Col(
                    xs=12,
                    md=10,
                    lg=10,
                    style={"minHeight": "100vh", "padding": "0", "backgroundColor": "#020208"},
                    children=[
                        html.Div(
                            style={"height": "100vh", "width": "100%"},
                            children=[
                                dcc.Loading(
                                    id="artifact-loading",
                                    type="default",
                                    color=PALETTE["cyan"],
                                    children=[html.Div(id="artifact-container", style={"height": "100vh", "width": "100%"})],
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
    Output("index-revision", "data"),
    Input("reindex-btn", "n_clicks"),
    State("run-dropdown", "value"),
)
def reindex_runs(n_clicks: Optional[int], current_run: Optional[str]):
    global ARTIFACT_ROOTS, PRIMARY_ARTIFACT_ROOT, INDEX, RUN_OPTIONS, DEFAULT_RUN, DEFAULT_VARIANTS, DEFAULT_PRIMARY_VARIANT
    ARTIFACT_ROOTS = _discover_artifact_roots()
    PRIMARY_ARTIFACT_ROOT = ARTIFACT_ROOTS[0] if ARTIFACT_ROOTS else None
    INDEX = build_artifact_index()
    RUN_OPTIONS = _run_options_from_index(INDEX)
    DEFAULT_RUN = INDEX.get("default_run", "")
    DEFAULT_VARIANTS = INDEX.get("runs", {}).get(DEFAULT_RUN, {}).get("variants", ["MONOLITH.html"])
    DEFAULT_PRIMARY_VARIANT = _preferred_variant(DEFAULT_VARIANTS)

    option_values = [opt["value"] for opt in RUN_OPTIONS]
    current_run_blob = INDEX.get("runs", {}).get(current_run or "", {})
    current_run_healthy = bool(current_run_blob.get("contract_ok")) or str(current_run_blob.get("selection_verification_status", "")).upper() == LayerStatus.NON_COMPARABLE.value
    if current_run in option_values and current_run_healthy:
        run_value = current_run
    else:
        run_value = DEFAULT_RUN if DEFAULT_RUN in option_values else (option_values[0] if option_values else "")
    disabled = len(RUN_OPTIONS) == 0
    root_info = f"Artifact Roots: {INDEX.get('artifact_root_count', 0)} | Primary: {INDEX.get('artifact_root') or 'NOT FOUND'}"
    status = f"Reindex complete | runs={len(INDEX.get('run_keys', []))} | clicks={int(n_clicks or 0)}"
    return RUN_OPTIONS, run_value, disabled, root_info, status, int(n_clicks or 0)


@app.callback(
    Output("variant-a-dropdown", "options"),
    Output("variant-a-dropdown", "value"),
    Output("variant-b-dropdown", "options"),
    Output("variant-b-dropdown", "value"),
    Output("observer-dropdown", "options"),
    Output("observer-dropdown", "value"),
    Input("run-dropdown", "value"),
    Input("url", "search"),
    Input("index-revision", "data"),
    State("variant-a-dropdown", "value"),
    State("variant-b-dropdown", "value"),
    State("observer-dropdown", "value"),
)
def refresh_variants(run_key: str, search: Optional[str], _index_revision: int, current_a: str, current_b: str, current_observer: str):
    try:
        qs = parse_qs((search or "").lstrip("?"))
    except Exception:
        qs = {}
    embedded_raw = str((qs.get("embedded") or ["0"])[0]).strip().lower()
    embedded = embedded_raw in {"1", "true", "on", "yes"}
    requested_run = str((qs.get("run_key") or [run_key])[0] or run_key)
    effective_run_key = requested_run if requested_run in INDEX.get("runs", {}) else run_key
    run = INDEX["runs"].get(effective_run_key, {})
    variants = run.get("variants", ["MONOLITH.html"])
    if not variants:
        variants = ["MONOLITH.html"]
    opts = [{"label": v, "value": v} for v in variants]
    preferred_variant = _preferred_variant(variants)
    requested_variant_a = str((qs.get("variant_a") or [""])[0] or "").strip()
    requested_variant_b = str((qs.get("variant_b") or [""])[0] or "").strip()
    a = current_a if current_a in variants else preferred_variant
    b = current_b if current_b in variants else preferred_variant
    if requested_variant_a in variants:
        a = requested_variant_a
    if requested_variant_b in variants:
        b = requested_variant_b
    obs_opts = INDEX["observers_by_run"].get(effective_run_key, [{"label": "Global Mean", "value": "global"}])
    obs_values = [o["value"] for o in obs_opts]
    observer = current_observer if current_observer in obs_values else "global"
    if not embedded:
        requested_observer = str((qs.get("observer") or [""])[0] or "").strip()
        requested_uid = str((qs.get("observer_uid") or [""])[0] or "").strip()
        observer_from_uid = _observer_value_from_uid(effective_run_key, requested_uid)
        if observer_from_uid and observer_from_uid in obs_values:
            observer = observer_from_uid
        elif requested_observer in obs_values:
            observer = requested_observer
    else:
        observer = "global"
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
        return no_update, no_update, no_update, no_update
    try:
        qs = parse_qs((search or "").lstrip("?"))
    except Exception:
        return no_update, no_update, no_update, no_update

    run_values = [o.get("value") for o in (run_options or []) if isinstance(o, dict)]
    requested_run = str((qs.get("run_key") or [current_run])[0] or current_run)
    run_value = requested_run if requested_run in run_values else current_run
    embedded_raw = str((qs.get("embedded") or ["0"])[0]).strip().lower()
    embedded = embedded_raw in {"1", "true", "on", "yes"}

    observer = "global"
    if not embedded:
        observer = str((qs.get("observer") or ["global"])[0] or "global")
        observer_uid = str((qs.get("observer_uid") or [""])[0] or "").strip()
        observer_from_uid = _observer_value_from_uid(run_value, observer_uid)
        if observer_from_uid:
            observer = observer_from_uid
    view_mode = str((qs.get("view_mode") or ["global"])[0] or "global").lower()
    if view_mode not in {"global", "observer"}:
        view_mode = "global"
    compare_raw = str((qs.get("compare") or ["0"])[0]).strip().lower()
    compare_values = ["on"] if compare_raw in {"1", "true", "on", "yes"} else []
    if embedded:
        observer = "global"
        view_mode = "global"
        compare_values = []
    return run_value, observer, view_mode, compare_values


@app.callback(
    Output("view-mode", "value", allow_duplicate=True),
    Input("observer-dropdown", "value"),
    State("view-mode", "value"),
    prevent_initial_call=True,
)
def sync_view_mode_with_observer(observer_value: Optional[str], current_view_mode: Optional[str]):
    observer_text = str(observer_value or "global").strip()
    next_view_mode = "global" if observer_text == "global" else "observer"
    if str(current_view_mode or "").strip().lower() == next_view_mode:
        return no_update
    return next_view_mode


@app.callback(
    Output("verification-source", "options"),
    Output("verification-source", "value"),
    Input("run-dropdown", "value"),
    Input("index-revision", "data"),
    State("verification-source", "value"),
)
def refresh_verification_sources(run_key: str, _index_revision: int, current_value: str):
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
    Input("index-revision", "data"),
    State("label-column-dropdown", "value"),
    State("label-value-dropdown", "value"),
)
def refresh_hidden_label_filters(run_key: str, _index_revision: int, current_col: Optional[str], current_values: Optional[List[str]]):
    contract = load_contract_state(run_key, "global")
    rows = contract.get("hidden_groups", []) or []
    if not rows:
        return [], None, [], []

    columns = sorted({k for r in rows if isinstance(r, dict) for k in r.keys() if k.startswith("group_")})
    col_opts = [{"label": c, "value": c} for c in columns]
    if current_col in columns:
        chosen_col = current_col
    elif "group_topic" in columns:
        chosen_col = "group_topic"
    else:
        chosen_col = columns[0] if columns else None

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
    Output("physical-path-readout", "children"),
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
    Input("view-mode", "value"),
    Input("run-dropdown", "value"),
    Input("variant-a-dropdown", "value"),
    Input("variant-b-dropdown", "value"),
    Input("verification-source", "value"),
    Input("compare-enabled", "value"),
    Input("transition-style", "value"),
    Input("delta-mode", "value"),
    Input("translation-mode", "value"),
    Input("failure-overlays", "value"),
    Input("label-column-dropdown", "value"),
    Input("label-value-dropdown", "value"),
    Input("index-revision", "data"),
)
def render_dashboard(
    observer_value: str,
    view_mode: str,
    run_key: str,
    variant_a: str,
    variant_b: str,
    verification_source: str,
    compare_enabled_values: List[str],
    transition_style: str,
    delta_mode: str,
    translation_mode_values: List[str],
    failure_overlay_values: List[str],
    label_column: Optional[str],
    label_values: List[str],
    _index_revision: int,
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
    elif str(contract_status).upper() == LayerStatus.NON_COMPARABLE.value:
        badge_text = "[NON-COMPARABLE]"
    elif str(contract_status).upper() != "OK":
        badge_text = "[INVALID SCHEMA]"
    elif str(verification_status).upper() == LayerStatus.MISSING_ARTIFACTS.value:
        badge_text = "[MISSING ARTIFACTS]"
    elif str(verification_status).upper() == LayerStatus.NON_COMPARABLE.value:
        badge_text = "[NON-COMPARABLE]"
    else:
        badge_text = "[UNVERIFIED]"
    watermark_visible = badge_text in {"[INVALID SCHEMA]", "[MISSING ARTIFACTS]", "[NON-COMPARABLE]"}
    return {
        "claims_enabled": claims_enabled,
        "badge_text": badge_text,
        "watermark_visible": watermark_visible,
    }


def _build_empathy_figure(contract: dict, label_col: Optional[str], label_values: Optional[List[str]]):
    matrix_blob = contract.get("group_matrix", {}) or {}
    groups = matrix_blob.get("groups", []) if isinstance(matrix_blob, dict) else []
    matrix = matrix_blob.get("cost_matrix", []) if isinstance(matrix_blob, dict) else []
    label_source = str(matrix_blob.get("label_source", "unknown")) if isinstance(matrix_blob, dict) else "unknown"

    fig = go.Figure()
    def _disabled_figure(title: str, reason: str):
        fig.update_layout(
            template="plotly_dark",
            margin={"l": 30, "r": 10, "t": 30, "b": 30},
            title=title,
            annotations=[{"text": reason, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}],
        )
        return fig

    if not groups or not matrix:
        return _disabled_figure("Empathy Gap (disabled)", "group_matrix not available")

    filtered_idx = list(range(len(groups)))
    if label_values:
        selected = set(label_values)
        filtered_idx = [i for i, g in enumerate(groups) if g in selected]
        if not filtered_idx:
            filtered_idx = list(range(len(groups)))

    fg = [groups[i] for i in filtered_idx]
    fm = [[float(matrix[i][j]) for j in filtered_idx] for i in filtered_idx]
    arr = np.asarray(fm, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return _disabled_figure("Empathy Gap (disabled)", "group_matrix shape is invalid")
    off_diag_mask = ~np.eye(arr.shape[0], dtype=bool)
    off_diag = arr[off_diag_mask]
    finite_off_diag = off_diag[np.isfinite(off_diag)]
    if finite_off_diag.size <= 0:
        return _disabled_figure("Empathy Gap (disabled)", "group_matrix has no finite off-diagonal values")
    if np.nanmax(finite_off_diag) - np.nanmin(finite_off_diag) <= 1e-9:
        return _disabled_figure(
            "Empathy Gap (disabled)",
            f"uniform off-diagonal costs from {label_source}; panel suppressed because it carries no structure",
        )

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


def _build_control_panel(ctrl: dict):
    status = str(ctrl.get("status", "MISSING")).upper()
    controls = ctrl.get("controls", {}) if isinstance(ctrl.get("controls"), dict) else {}
    summary = ctrl.get("summary", {}) if isinstance(ctrl.get("summary"), dict) else {}
    subtitle = "Type 1 checks whether the real leaf separates cleanly from matched control leaves."
    detail_bits = [
        html.Div(ctrl.get("explanation") or summary.get("interpretation") or ctrl.get("message") or "", style={"color": PALETTE["dim"], "fontSize": "0.76rem", "lineHeight": "1.35"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "8px"},
            children=[
                _metric_tile("Procrustes Ratio", ctrl.get("procrustes_ratio"), "Real vs control shape residual"),
                _metric_tile("Distance-Corr Ratio", ctrl.get("distance_corr_ratio"), "Real vs control geometry coupling"),
                _metric_tile("Separations", ctrl.get("separates_count"), "How many control tests cleanly separate"),
                _metric_tile("Consensus / Residual", f"{_fmt_metric(ctrl.get('consensus_pct'))}% / {_fmt_metric(ctrl.get('residual_pct'))}%", "Shared signal vs leftover variance"),
            ],
        ),
        html.Div(f"Source: {ctrl.get('source', 'NOT FOUND')}", style={"color": PALETTE["dim"], "fontSize": "0.72rem", "wordBreak": "break-all"}),
    ]
    if controls:
        detail_bits.insert(
            1,
            html.Div(
                "Available controls: " + ", ".join(sorted(map(str, controls.keys()))),
                style={"color": PALETTE["cyan"], "fontSize": "0.74rem"},
            ),
        )
    if status in {"NO_DATA", "UNAVAILABLE", "MISSING"} and ctrl.get("message"):
        detail_bits.insert(
            1,
            html.Div(f"Why missing: {ctrl.get('message')}", style={"color": PALETTE["amber"], "fontSize": "0.74rem", "lineHeight": "1.3"}),
        )
    return _panel_shell("Type 1 Controls", subtitle, status, detail_bits)


def _build_ablation_panel(ab: dict):
    status = str(ab.get("status", "MISSING")).upper()
    summary = ab.get("summary", {}) if isinstance(ab.get("summary"), dict) else {}
    stage_map = summary.get("stage_map", {}) if isinstance(summary.get("stage_map"), dict) else {}
    stage_1 = ab.get("stage_1_alignment_score", ab.get("stage_1_nmi"))
    stage_2 = ab.get("stage_2_alignment_score", ab.get("stage_2_nmi"))
    stage_3 = ab.get("stage_3_survival_rate", ab.get("stage_3_nmi"))
    delta_alignment = ab.get("delta_alignment_score", ab.get("delta_nmi"))
    subtitle = "Ablations tell us which pipeline choices preserve structure and which ones flatten or destabilize it."
    if status in {"NO_DATA", "UNAVAILABLE", "MISSING"}:
        body: List[Any] = [
            html.Div(ab.get("message") or "Ablation flow was not run for this leaf.", style={"color": PALETTE["amber"], "fontSize": "0.76rem", "fontWeight": "600"}),
            html.Div(ab.get("explanation") or summary.get("interpretation") or "", style={"color": PALETTE["dim"], "fontSize": "0.76rem", "lineHeight": "1.35"}),
            html.Div(f"Source: {ab.get('source', 'NOT FOUND')}", style={"color": PALETTE["dim"], "fontSize": "0.72rem", "wordBreak": "break-all"}),
        ]
    else:
        body = [
            html.Div(ab.get("explanation") or summary.get("interpretation") or ab.get("message") or "", style={"color": PALETTE["dim"], "fontSize": "0.76rem", "lineHeight": "1.35"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "8px"},
                children=[
                    _metric_tile("Stage 1 Align", stage_1, stage_map.get("stage_1_alignment_score", stage_map.get("stage_1_nmi"))),
                    _metric_tile("Stage 2 Align", stage_2, stage_map.get("stage_2_alignment_score", stage_map.get("stage_2_nmi"))),
                    _metric_tile("Stage 3 Survival", stage_3, stage_map.get("stage_3_survival_rate", stage_map.get("stage_3_nmi"))),
                    _metric_tile("Delta Align / Retained", f"{_fmt_metric(delta_alignment)} / {_fmt_metric(ab.get('retained_pct'))}%", "Alignment lift vs retained invariants"),
                ],
            ),
            html.Div(f"Source: {ab.get('source', 'NOT FOUND')}", style={"color": PALETTE["dim"], "fontSize": "0.72rem", "wordBreak": "break-all"}),
        ]
    if ab.get("legacy_mean_variance") is not None:
        body.insert(1, html.Div(f"Legacy variance metric: {ab.get('legacy_mean_variance')}", style={"color": PALETTE["cyan"], "fontSize": "0.74rem"}))
    return _panel_shell("Ablation Laboratory", subtitle, status, body)


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

    count_tiles = [
        html.Div(
            [
                html.Div(str(k), style={"color": PALETTE["text"], "fontWeight": "700", "fontSize": "0.75rem"}),
                html.Div(f"{counts[k]} articles", style={"color": PALETTE["dim"], "fontSize": "0.72rem"}),
            ],
            style={"padding": "8px", "borderRadius": "10px", "border": f"1px solid {PALETTE['grid']}", "backgroundColor": "rgba(10,10,18,0.8)"},
        )
        for k in sorted(counts)
    ]
    summary_rows: List[Any] = []
    if isinstance(summaries, list):
        for g in summaries[:8]:
            if not isinstance(g, dict):
                continue
            name = g.get("group_name", "unknown")
            n = g.get("n_articles", "n/a")
            markers = g.get("top_markers") or []
            summary_rows.append(
                html.Div(
                    [
                        html.Div(f"{name} ({n})", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.74rem"}),
                        html.Div(", ".join(map(str, markers[:4])) if markers else "no markers", style={"color": PALETTE["dim"], "fontSize": "0.7rem"}),
                    ],
                    style={"padding": "6px 0", "borderBottom": f"1px solid {PALETTE['grid']}"},
                )
            )
    return _panel_shell(
        "Hidden Label Topology",
        "Metadata-derived groupings help compare narrative camps without letting terrain zones define the labels.",
        "OK",
        [
            html.Div(f"Grouping column: {label_col}", style={"color": PALETTE["cyan"], "fontSize": "0.74rem"}),
            html.Div(count_tiles, style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "8px"}),
            html.Div(summary_rows or [html.Div("No group summaries available.", style={"color": PALETTE["dim"], "fontSize": "0.74rem"})]),
        ],
    )


def _build_relativity_panel(contract: dict, observer_value: str, delta_mode: str, translation_mode_values: List[str]):
    if observer_value == "global":
        return _panel_shell(
            "Type 2 Relativity",
            "Type 2 measures vector displacement between the global mean manifold and an observer-conditioned manifold.",
            "BASELINE",
            [
                html.Div("Global baseline selected.", style={"color": PALETTE["text"], "fontSize": "0.76rem", "fontWeight": "600"}),
                html.Div("Choose an article observer to reveal elasticity and polarization deltas relative to this baseline.", style={"color": PALETTE["dim"], "fontSize": "0.76rem"}),
            ],
        )

    delta = contract.get("observer_delta", {}) or {}
    observer_state = contract.get("observer_state", {}) or {}
    if not delta:
        return _panel_shell(
            "Type 2 Relativity",
            "Type 2 measures vector displacement between the global mean manifold and an observer-conditioned manifold.",
            "MISSING",
            [html.Div("Observer delta artifact missing for this observer.", style={"color": PALETTE["amber"], "fontSize": "0.76rem"})],
        )

    null_eq = delta.get("null_observer_equivalence", {}) or {}
    metrics_delta = delta.get("metrics_delta", {}) or {}
    axis_delta = delta.get("axis_delta", {}) or {}
    flips = delta.get("path_flip_delta", {}) or {}

    translation_only = translation_mode_values is not None and "translation_only" in translation_mode_values
    tcomp = delta.get("translation_only_comparison", {}) if translation_only else {}

    state_metrics = observer_state.get("metrics", {}) if isinstance(observer_state.get("metrics"), dict) else {}
    flip_items = sorted(flips.items(), key=lambda kv: -float(kv[1]) if _is_number(kv[1]) else 0.0)
    flip_rows = [
        html.Div(
            f"{k}: {_fmt_metric(v)}",
            style={"color": PALETTE["text"], "fontSize": "0.72rem", "padding": "3px 0", "borderBottom": f"1px solid {PALETTE['grid']}"},
        )
        for k, v in flip_items[:8]
    ] or [html.Div("No path flips recorded.", style={"color": PALETTE["dim"], "fontSize": "0.72rem"})]
    body: List[Any] = [
        html.Div(
            [
                html.Div(f"Observer: {observer_value}", style={"color": PALETTE["cyan"], "fontSize": "0.74rem"}),
            ]
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "8px"},
            children=[
                _metric_tile("Max Elasticity", null_eq.get("max_coord_delta"), "Largest point displacement"),
                _metric_tile("Path Flips", null_eq.get("path_flip_count"), "Topology disagreements vs global"),
                _metric_tile("Axis Rotation", null_eq.get("axis_rotation_deg"), "Frame rotation in degrees"),
                _metric_tile("Delta NMI", metrics_delta.get("d_nmi"), "Observer-conditioned minus global"),
                _metric_tile("Mean Work Δ", metrics_delta.get("d_mean_work"), "Energetic shift under this observer"),
                _metric_tile("Survival Δ", metrics_delta.get("d_survival_pct"), "Track 4 survival difference"),
            ],
        ),
        html.Div(
            [
                html.Div("Observer-conditioned similarity", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.74rem"}),
                html.Div(
                    f"global={_fmt_metric(state_metrics.get('global_nmi'))} | observer={_fmt_metric(state_metrics.get('observer_conditioned_nmi'))} | axis_var_delta={_fmt_metric(axis_delta.get('d_explained_variance_axis1'))}",
                    style={"color": PALETTE["dim"], "fontSize": "0.72rem"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div("Largest path flips", style={"color": PALETTE["text"], "fontWeight": "600", "fontSize": "0.74rem", "marginBottom": "4px"}),
                html.Div(flip_rows),
            ]
        ),
    ]
    if translation_only and tcomp:
        body.append(
            html.Div(
                f"Translation-only comparison -> path flips Δ={_fmt_metric(tcomp.get('d_path_flip_count'))} | mean work Δ={_fmt_metric(tcomp.get('d_mean_work'))}",
                style={"color": PALETTE["amber"], "fontSize": "0.74rem"},
            )
        )
    return _panel_shell(
        "Type 2 Relativity",
        "Type 2 measures the observer-specific displacement field relative to the global mean manifold.",
        "OK",
        body,
    )


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
            _build_ablation_panel({"status": "MISSING", "source": "NOT FOUND", "message": "no ablation analysis results", "summary": {}}),
            _build_control_panel({"status": "MISSING", "source": "NOT FOUND", "message": "no control analysis results", "summary": {}, "controls": {}}),
            "provenance: n/a",
            "[HIDDEN LABELS MISSING]",
            {"padding": "6px 8px", "borderRadius": "6px", "textAlign": "center", "color": PALETTE["amber"], "border": f"1px solid {PALETTE['amber']}", "backgroundColor": "rgba(255,179,71,0.12)"},
            "labels/hidden_groups.csv not found",
            _panel_shell("Type 2 Relativity", "Type 2 measures observer displacement fields relative to the global mean frame.", "MISSING", [html.Div("Relativity data unavailable.", style={"color": PALETTE["amber"], "fontSize": "0.76rem"})]),
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
    contract_global = load_contract_state(run_key, "global") if (run_key and view_mode == "observer" and observer_value.startswith("article:")) else contract

    if view_mode == "observer" and observer_value.startswith("article:"):
        p_a = resolve_artifact(run_key, variant_a, "global") if run_key else None
        p_b = resolve_artifact(run_key, variant_b, observer_value) if run_key else None
    else:
        p_a = resolve_artifact(run_key, variant_a, effective_observer) if run_key else None
        p_b = resolve_artifact(run_key, variant_b, effective_observer) if run_key else None

    text_a = _safe_read_text(p_a) if (p_a and p_a.exists()) else ""
    text_b = _safe_read_text(p_b) if (p_b and p_b.exists()) else ""

    if text_a:
        c_a = _transition_wrapper(_artifact_iframe(text_a), transition_style)
    else:
        c_a = build_terminal_fallback(effective_observer, run_key, variant_a, gallery_tick)

    if compare_enabled:
        if text_b:
            c_b = _transition_wrapper(_artifact_iframe(text_b), transition_style)
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
        if view_mode == "observer" and observer_value.startswith("article:"):
            if p_b and p_b.exists() and text_b:
                single_view_path = p_b
                single_view = _transition_wrapper(_artifact_iframe(text_b), transition_style)
            else:
                single_view_path = None
                single_view = build_terminal_fallback(observer_value, run_key, variant_b, gallery_tick)
        container = single_view
        path_text = f"Artifact: {single_view_path if single_view_path else 'NOT FOUND'}"

    graph_observer_a = "global" if (view_mode == "observer" and observer_value.startswith("article:")) else effective_observer
    graph_observer_b = effective_observer
    raw_state_a = contract_global.get("baseline_state", {}) if graph_observer_a == "global" else contract_global.get("observer_state", {})
    raw_state_b = contract.get("baseline_state", {}) if graph_observer_b == "global" else contract.get("observer_state", {})
    artifact_state_a = _hydrate_artifact_state(run_key, raw_state_a, p_a)
    artifact_state_b = _hydrate_artifact_state(run_key, raw_state_b, p_b)
    artifact_state = artifact_state_b if (view_mode == "observer" and observer_value.startswith("article:")) else artifact_state_a
    artifact_metrics = artifact_state.get("metrics", {}) if isinstance(artifact_state, dict) else {}

    run_score = f"Run Score | kernel={run.get('kernel', 'unknown')} seed={run.get('seed', 'unknown')} NMI={run.get('nmi', 'n/a')} ARI={run.get('ari', 'n/a')}"
    if artifact_metrics:
        run_score = (
            f"Run Score | kernel={run.get('kernel', 'unknown')} seed={run.get('seed', 'unknown')} "
            f"NMI={artifact_metrics.get('synthesis_nmi', run.get('nmi', 'n/a'))} "
            f"| Signal={artifact_metrics.get('spectral_signal', 'n/a')} "
            f"| T3 bonds/cracks={artifact_metrics.get('dirichlet_bonds', 'n/a')}/{artifact_metrics.get('dirichlet_cracks', 'n/a')} "
            f"| T4 work={artifact_metrics.get('walker_mean_action', 'n/a')} closed_loop={artifact_metrics.get('walker_survival_rate', 'n/a')} "
            f"| T5 terminal_labels H/P/T/A={artifact_metrics.get('honest_count', 'n/a')}/{artifact_metrics.get('phantom_count', 'n/a')}/{artifact_metrics.get('tautology_count', 'n/a')}/{artifact_metrics.get('anomaly_count', 'n/a')}"
        )
    if effective_observer.startswith("article:"):
        observer_state_metrics = contract.get("observer_state", {}).get("metrics", {}) if isinstance(contract.get("observer_state", {}), dict) else {}
        obs_nmi = observer_state_metrics.get("observer_conditioned_nmi")
        obs_ari = observer_state_metrics.get("observer_conditioned_ari")
        if _is_number(obs_nmi) or _is_number(obs_ari):
            run_score += f" | Observer NMI={obs_nmi if _is_number(obs_nmi) else 'n/a'} ARI={obs_ari if _is_number(obs_ari) else 'n/a'}"

    article_metric_text = "Article Metrics: n/a"
    if effective_observer.startswith("article:"):
        try:
            idx = int(effective_observer.split(":", 1)[1])
            row = INDEX["article_rows_by_run"].get(run_key, {}).get(idx, {})
            if not row and artifact_state:
                for item in (artifact_state.get("articles", []) or []):
                    if isinstance(item, dict) and int(item.get("idx", -1)) == idx:
                        row = item
                        break
            if row:
                article_metric_text = (
                    f"Article #{idx} | uid={row.get('bt_uid', 'n/a')} | zone={row.get('zone', 'n/a')} "
                    f"| density={row.get('density', 'n/a')} | stress={row.get('stress', 'n/a')}"
                )
        except Exception:
            pass

    snapshot_a = _compute_track_snapshot(run_key, artifact_state_a, contract_global, "global" if view_mode == "observer" and observer_value.startswith("article:") else effective_observer)
    snapshot_b = _compute_track_snapshot(run_key, artifact_state_b, contract, effective_observer)
    primary_snapshot = snapshot_b if (view_mode == "observer" and observer_value.startswith("article:")) else snapshot_a
    track_readout = _track_status_component(primary_snapshot)
    track_compare_readout = _track_delta_component(snapshot_a, snapshot_b)
    found_a, total_a = _artifact_coverage(run_key, variant_a) if run_key else (0, 0)
    found_b, total_b = _artifact_coverage(run_key, variant_b) if run_key else (0, 0)
    if compare_enabled:
        coverage_text = (
            f"Observer Artifact Coverage | A={found_a}/{total_a if total_a > 0 else 'n/a'}"
            f" | B={found_b}/{total_b if total_b > 0 else 'n/a'}"
        )
    else:
        coverage_text = f"Observer Artifact Coverage (Variant A): {found_a}/{total_a}" if total_a > 0 else "Observer Artifact Coverage: n/a"

    state = load_verification_state(run_key, verification_source)
    verification_status = str(state.get("verification_status", "UNVERIFIED")).upper()
    global_pass = state.get("global_pass")
    sanitized_missing_optional = []
    for item in (missing_optional or []):
        name = str(item).strip()
        if name == "verification_summary.csv" and str(state.get("summary_path", "NOT FOUND")) != "NOT FOUND":
            continue
        if name == "verification_report.json" and str(state.get("report_path", "NOT FOUND")) != "NOT FOUND":
            continue
        sanitized_missing_optional.append(name)

    # VALIDATION TYPE 2: Perspective Sensitivity (Divergence Check)
    type2_dissonance = False
    if compare_enabled:
        surv_a = snapshot_a.get("T4", {}).get("survival")
        surv_b = snapshot_b.get("T4", {}).get("survival")
        if _is_number(surv_a) and _is_number(surv_b) and abs(float(surv_a) - float(surv_b)) > 0.20:
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
        t1 = f"System 1: Verification gate passed | seed_stability={_fmt_pass(state.get('seed_stability'))} | crn_locked={_fmt_pass(state.get('crn_locked'))}"
        t2 = f"System 2: Geometric Friction = {_fmt_metric(state.get('geometric_friction'))} (broken={state.get('n_broken', 0)}, trapped={state.get('n_trapped', 0)})"
        t3 = f"System 2: Survival % = {_fmt_metric(state.get('survival_pct'))}%"
    elif badge_text == "[UNVERIFIED]":
        t1 = "System 1: Verification bundle is pending; provenance is present but the formal gate has not been finalized for this leaf."
        t2 = "System 2: Geometric Friction = " + (
            f"{_fmt_metric(state.get('geometric_friction'))} (telemetry present, interpret as provisional)"
            if _is_number(state.get("geometric_friction"))
            else "unavailable (no parsed verification telemetry)"
        )
        t3 = "System 2: Survival % = " + (
            f"{_fmt_metric(state.get('survival_pct'))}% (telemetry present, interpret as provisional)"
            if _is_number(state.get("survival_pct"))
            else "unavailable (no parsed verification telemetry)"
        )
    else:
        t1 = "System 1: Claims disabled (verification/provenance gate not satisfied)"
        t2 = "System 2: Claims disabled (artifact contract incomplete or non-comparable)"
        t3 = "System 2: Claims disabled (artifact contract incomplete or non-comparable)"

    detail = f"source={state.get('verification_source')} | verification_summary.csv: {state.get('summary_path')} | verification_report.json: {state.get('report_path')}"
    if missing_required:
        detail += " | missing_required_artifacts=" + ",".join(missing_required)
    if sanitized_missing_optional:
        detail += " | missing_optional_artifacts=" + ",".join(sanitized_missing_optional[:5])
    if schema_errors:
        detail += " | schema_errors=" + "; ".join(schema_errors[:3])
    elif contract_errors:
        detail += " | diagnostics=" + "; ".join(contract_errors[:3])
    ab = load_ablation_state(run_key)
    ctrl = load_control_state(run_key)
    ablation_text = _build_ablation_panel(ab)
    control_text = _build_control_panel(ctrl)

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
        hidden_detail = (
            f"rows={len(hidden_rows)} | "
            f"group_summaries={contract.get('paths', {}).get('labels/derived/group_summaries.json', 'NOT FOUND')} | "
            f"group_matrix={contract.get('paths', {}).get('labels/derived/group_matrix.json', 'NOT FOUND')}"
        )
    else:
        hidden_badge_text = "[HIDDEN LABELS MISSING]"
        hidden_badge_style = {"padding": "6px 8px", "borderRadius": "6px", "textAlign": "center", "color": PALETTE["amber"], "border": f"1px solid {PALETTE['amber']}", "backgroundColor": "rgba(255,179,71,0.12)"}
        hidden_detail = "Expected labels/hidden_groups.csv and labels/derived/group_*.json"
        if sanitized_missing_optional:
            hidden_detail += f" | missing_optional_artifacts={','.join(sanitized_missing_optional)}"

    relativity_panel = _build_relativity_panel(contract, effective_observer, delta_mode, translation_mode_values or [])
    group_panel = _build_group_panel(contract, label_column, label_values or [])
    empathy_fig = _build_empathy_figure(contract, label_column, label_values or [])

    watermark_visible = bool(gate.get("watermark_visible"))
    if watermark_visible:
        watermark_text = badge_text.strip("[]")
        watermark_style = {
            "display": "flex",
            "position": "fixed",
            "inset": "0",
            "alignItems": "center",
            "justifyContent": "center",
            "pointerEvents": "none",
            "zIndex": 2000,
            "fontSize": "2.8rem",
            "fontWeight": "800",
            "letterSpacing": "0.18em",
            "textTransform": "uppercase",
            "color": "rgba(255, 42, 0, 0.16)",
            "textShadow": "0 0 24px rgba(255, 42, 0, 0.20)",
        }
    else:
        watermark_text = ""
        watermark_style = {"display": "none"}

    physical_path = str(single_view_path.resolve()) if (not compare_enabled and single_view_path) else (f"A: {p_a.resolve() if p_a else 'n/a'} | B: {p_b.resolve() if p_b else 'n/a'}")

    return (
        container,
        path_text,
        run_score,
        article_metric_text,
        f"PHYSICAL PATH: {physical_path}",
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
    debug_enabled = str(os.environ.get("BT_DASH_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    run_kwargs = {"debug": debug_enabled, "host": "0.0.0.0", "port": 8050}
    if not debug_enabled:
        run_kwargs["use_reloader"] = False
    run_fn = getattr(app, "run", None)
    try:
        if callable(run_fn):
            run_fn(**run_kwargs)
        else:
            app.run_server(**run_kwargs)
    except TypeError:
        run_kwargs.pop("use_reloader", None)
        if callable(run_fn):
            run_fn(**run_kwargs)
        else:
            app.run_server(**run_kwargs)
