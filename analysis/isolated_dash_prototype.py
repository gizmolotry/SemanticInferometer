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
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback_context, dcc, html


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

ROOT = Path(__file__).resolve().parents[1]


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


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_json(path: Path, default):
    try:
        return json.loads(_safe_read_text(path))
    except Exception:
        return default


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
        run_parent = run_dir.parent if run_dir else None
        candidate_roots = [run_dir, run_parent, ROOT / "analysis", ROOT / "verification", ROOT]
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
        return _find_latest_pair_under(Path(verification_source))

    run_dir = None
    if run_key and "INDEX" in globals():
        run = INDEX.get("runs", {}).get(str(run_key), {})
        run_dir = run.get("run_dir")

    search_roots: List[Path] = []
    if run_dir and run_dir.exists():
        search_roots.extend(
            [
                run_dir,
                run_dir / "verification",
                run_dir.parent,
                run_dir.parent / "verification",
                run_dir.parent.parent / "verification" if run_dir.parent and run_dir.parent.parent else None,
            ]
        )
    search_roots.extend([ROOT / "verification", ROOT / "analysis"])
    seen = set()
    normalized_roots = []
    for r in search_roots:
        if r and r.exists():
            sr = str(r)
            if sr not in seen:
                seen.add(sr)
                normalized_roots.append(r)

    for root in normalized_roots:
        report, summary = _find_latest_pair_under(root)
        if report and summary:
            return summary, report

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
            roots.extend([run_dir, run_dir.parent])
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
    global_pass: Optional[bool] = None
    seed_stability: Optional[bool] = None
    crn_locked: Optional[bool] = None
    broken = 0
    trapped = 0
    total = 0

    if report_json and report_json.exists():
        payload = _safe_json(report_json, {})
        global_pass = _find_bool_key(payload, ["global_pass"])
        seed_stability = _find_bool_key(payload, ["seed_stability", "seed_stability_pass", "seed_stable"])
        crn_locked = _find_bool_key(payload, ["crn_locked", "crn_lock_pass", "crn_pass"])

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
        }
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
        html.Div(style={"display": "none"}),
        dcc.Store(id="gallery-dir", data={"dir": 1}),
        dcc.Store(id="hotkey-signal", data={"event": "none", "seq": 0}),
        dcc.Interval(id="hotkey-poll", interval=250, n_intervals=0),
        dcc.Interval(id="verification-poll", interval=10000, n_intervals=0),
        dcc.Interval(id="gallery-interval", interval=2200, n_intervals=0, disabled=True),
        dbc.Row(
            className="g-0",
            style={"minHeight": "100vh"},
            children=[
                dbc.Col(
                    xs=12,
                    md=4,
                    lg=3,
                    style={"background": "linear-gradient(180deg, #050505 0%, #0b0b14 100%)", "borderRight": f"1px solid {PALETTE['grid']}", "padding": "16px"},
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
                    ],
                ),
                dbc.Col(xs=12, md=8, lg=9, style={"minHeight": "100vh", "padding": "0"}, children=[dcc.Loading(id="artifact-loading", type="default", color=PALETTE["cyan"], children=[html.Div(id="artifact-container", style={"height": "100vh", "width": "100%"})])]),
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
    Output("observer-dropdown", "value"),
    Output("gallery-dir", "data"),
    Input("gallery-interval", "n_intervals"),
    Input("prev-observer-btn", "n_clicks"),
    Input("next-observer-btn", "n_clicks"),
    Input("autoplay-mode", "value"),
    State("observer-dropdown", "value"),
    State("gallery-dir", "data"),
    State("observer-dropdown", "options"),
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
    Input("run-dropdown", "value"),
    Input("observer-dropdown", "value"),
    Input("variant-a-dropdown", "value"),
    Input("variant-b-dropdown", "value"),
    Input("verification-source", "value"),
    Input("compare-enabled", "value"),
    Input("transition-style", "value"),
    Input("verification-poll", "n_intervals"),
    Input("gallery-interval", "n_intervals"),
)
def render_dashboard(
    run_key: str,
    observer_value: str,
    variant_a: str,
    variant_b: str,
    verification_source: str,
    compare_enabled_values: List[str],
    transition_style: str,
    poll_tick: int,
    gallery_tick: int,
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
        )

    run = INDEX["runs"].get(run_key, {})
    compare_enabled = compare_enabled_values is not None and "on" in compare_enabled_values

    p_a = resolve_artifact(run_key, variant_a, observer_value) if run_key else None
    p_b = resolve_artifact(run_key, variant_b, observer_value) if run_key else None

    if p_a and p_a.exists():
        c_a = _transition_wrapper(html.Iframe(srcDoc=_safe_read_text(p_a), style={"width": "100%", "height": "100%", "border": "0"}), transition_style)
    else:
        c_a = build_terminal_fallback(observer_value, run_key, variant_a, gallery_tick)

    if compare_enabled:
        if p_b and p_b.exists():
            c_b = _transition_wrapper(html.Iframe(srcDoc=_safe_read_text(p_b), style={"width": "100%", "height": "100%", "border": "0"}), transition_style)
        else:
            c_b = build_terminal_fallback(observer_value, run_key, variant_b, gallery_tick)
        container = dbc.Row(
            className="g-0",
            style={"height": "100vh"},
            children=[
                dbc.Col([html.Div("Variant A", style={"color": PALETTE["cyan"], "padding": "4px 8px"}), html.Div(c_a, style={"height": "calc(100vh - 28px)"})], width=6),
                dbc.Col([html.Div("Variant B", style={"color": PALETTE["cyan"], "padding": "4px 8px"}), html.Div(c_b, style={"height": "calc(100vh - 28px)"})], width=6),
            ],
        )
        path_text = f"A: {p_a if p_a else 'NOT FOUND'} | B: {p_b if p_b else 'NOT FOUND'}"
    else:
        container = c_a
        path_text = f"Artifact: {p_a if p_a else 'NOT FOUND'}"

    run_score = f"Run Score | kernel={run.get('kernel', 'unknown')} seed={run.get('seed', 'unknown')} NMI={run.get('nmi', 'n/a')} ARI={run.get('ari', 'n/a')}"

    article_metric_text = "Article Metrics: n/a"
    if observer_value.startswith("article:"):
        try:
            idx = int(observer_value.split(":", 1)[1])
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
    global_pass = state.get("global_pass")
    base_badge = {"padding": "8px 10px", "borderRadius": "6px", "fontWeight": "700", "letterSpacing": "0.05em", "marginBottom": "10px", "textAlign": "center"}
    if global_pass is True:
        badge_text = "[VERIFIED]"
        badge_style = dict(base_badge, color="#00FF41", border="1px solid #00FF41", backgroundColor="rgba(0,255,65,0.12)", boxShadow="0 0 14px rgba(0,255,65,0.45)", animation="neonPulse 1.4s ease-in-out infinite")
    else:
        badge_text = "[SIGNAL UNSTABLE]"
        badge_style = dict(base_badge, color="#FF2A00", border="1px solid #FF2A00", backgroundColor="rgba(255,42,0,0.12)", boxShadow="0 0 12px rgba(255,42,0,0.4)", animation="glitchFlash 0.9s steps(2,end) infinite")

    t1 = f"System 1: Topologic Integrity | seed_stability={_fmt_pass(state.get('seed_stability'))} | crn_locked={_fmt_pass(state.get('crn_locked'))} [cite: 2026-02-04]"
    t2 = f"System 2: Geometric Friction = {state.get('geometric_friction', 0.0):.3f} (broken={state.get('n_broken', 0)}, trapped={state.get('n_trapped', 0)}) [cite: 2026-02-04]"
    t3 = f"System 2: Survival % = {state.get('survival_pct', 100.0):.2f}% [cite: 2026-02-04]"
    detail = f"source={state.get('verification_source')} | verification_summary.csv: {state.get('summary_path')} | verification_report.json: {state.get('report_path')}"
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
    )


if __name__ == "__main__":
    run_fn = getattr(app, "run", None)
    if callable(run_fn):
        run_fn(debug=True, host="127.0.0.1", port=8050)
    else:
        app.run_server(debug=True, host="127.0.0.1", port=8050)
