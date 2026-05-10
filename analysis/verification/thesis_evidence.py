from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from analysis.verification.scientific_summaries import (
    summarize_observer_relativity,
    summarize_track4_traversal,
)

EVIDENCE_SCHEMA_VERSION = "1.0"
SUMMARY_FILES = (
    "scientific_validation_summary.json",
    "claim_matrix.json",
    "ablation_matrix.json",
    "observer_relativity_summary.json",
    "track4_traversal_summary.json",
    "metric_signal_cartography.json",
    "variance_separation_summary.json",
    "unsafe_claim_strategy.json",
)
DEFAULT_CANONICAL_KERNELS = ["rbf", "laplacian", "rq", "imq"]
DEFAULT_CANONICAL_SEEDS = [42, 420, 4200]
DEFAULT_CANONICAL_CHANNELS = ["logits", "cls"]
DEFAULT_CANONICAL_CORPORA = [
    "real",
    "control_constant",
    "control_shuffled",
    "control_random",
]
DEFAULT_CANONICAL_LIMIT = 500

CONTROL_PROCRUSTES_RATIO_MIN = 1.05
CONTROL_DISTANCE_CORR_RATIO_MAX = 0.95
CONTROL_SEPARATION_COUNT_MIN = 2
CONTROL_STOCHASTIC_VARIANCE_RATIO_MAX = 0.95
CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN = 0.20
CONTROL_SIGNAL_CARTOGRAPHY_METRICS = (
    "simple_variance",
    "procrustes",
    "distance_corr",
    "knn_overlap",
    "cluster_stability",
)
CONTROL_SIGNAL_CARTOGRAPHY_BASES = ("direct_observer_payload", "comprehensive_results")
CONTROL_SIGNAL_CARTOGRAPHY_CONTROLS = ("Constant", "Shuffled", "Random")
SYNTHETIC_MEAN_NMI_MIN = 0.25
SYNTHETIC_MEAN_ARI_MIN = 0.10
SYNTHETIC_PER_KERNEL_NMI_MIN = SYNTHETIC_MEAN_NMI_MIN
SYNTHETIC_PER_SEED_NMI_MIN = SYNTHETIC_MEAN_NMI_MIN
RELATIVITY_MEAN_DELTA_MIN = 0.01
RELATIVITY_ROTATION_MIN_DEG = 1.0
TRACK4_SURVIVAL_MIN = 0.05
TRACK4_WORK_MIN = 1e-6
TRACK4_ZONE_MEAN_GAP_MIN = 0.05
TRACK4_CONTROL_SURVIVAL_GAP_MIN = 0.05
TRACK4_CONTROL_WORK_GAP_MIN = 5.0
REQUIRED_TRACK5_ABLATION_MODES = ("hadamard_strict", "riemannian_strict")
TRACK5_STAGE2_ALIGNMENT_MIN = 0.01
TRACK5_STAGE3_SURVIVAL_MIN = 0.01
TRACK5_DELTA_ALIGNMENT_MIN = 0.01
PROCRUSTES_PROVENANCE_REQUIRED_CHECKS = ("crn_locked", "seed_stability")
PROCRUSTES_PROVENANCE_IGNORED_FAILURES = ("control_ordering",)
PROCRUSTES_PROVENANCE_OPTIONAL_CHECKS = ("alpha_sweep_sanity",)
XY_COLLAPSE_EPS = 1e-6


@dataclass(frozen=True)
class CanonicalProtocol:
    runner: str
    mode: str
    kernels: Tuple[str, ...]
    seeds: Tuple[int, ...]
    channels: Tuple[str, ...]
    corpora: Tuple[str, ...]
    limit: int
    main_command: str
    synthetic_command: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runner": self.runner,
            "mode": self.mode,
            "kernels": list(self.kernels),
            "seeds": list(self.seeds),
            "channels": list(self.channels),
            "corpora": list(self.corpora),
            "limit": self.limit,
            "main_command": self.main_command,
            "synthetic_command": self.synthetic_command,
        }


def _load_line_list(path: Path) -> List[str]:
    items: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _mean(values: Sequence[float]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not filtered:
        return None
    return mean(filtered)


def _std(values: Sequence[float]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(filtered) <= 1:
        return 0.0 if filtered else None
    return pstdev(filtered)


def _normalize_token_list(raw: str, *, numeric: bool = False) -> List[Any]:
    ticked = re.findall(r"`([^`]+)`", raw)
    if ticked:
        tokens: List[str] = []
        for token in ticked:
            parts = [part for part in re.split(r"[\s,]+", token.strip()) if part]
            tokens.extend(parts or [token.strip()])
    else:
        tokens = [tok for tok in re.split(r"[\s,]+", raw.strip()) if tok]
    if numeric:
        out: List[int] = []
        for token in tokens:
            value = _safe_int(token)
            if value is not None:
                out.append(value)
        return out
    return [token.strip() for token in tokens if token.strip()]


def load_canonical_protocol(methods_path: Path) -> CanonicalProtocol:
    text = methods_path.read_text(encoding="utf-8")
    bullets: Dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"-\s+([A-Za-z ]+):\s*(.+)", line.strip())
        if match:
            bullets[match.group(1).strip().lower()] = match.group(2).strip()

    commands = re.findall(r"```powershell\s*(.*?)```", text, flags=re.S)
    main_command = commands[0].strip() if commands else ""
    synthetic_command = commands[1].strip() if len(commands) > 1 else ""

    runner = bullets.get("runner", "`run_full_experiment_suite.py`").strip("`")
    mode = bullets.get("mode", "enhanced").strip("`")
    kernels = _normalize_token_list(bullets.get("kernels", " ".join(DEFAULT_CANONICAL_KERNELS)))
    seeds = _normalize_token_list(
        bullets.get("seeds", " ".join(str(s) for s in DEFAULT_CANONICAL_SEEDS)),
        numeric=True,
    )
    channels = _normalize_token_list(
        bullets.get("channels", " ".join(DEFAULT_CANONICAL_CHANNELS))
    )
    corpora = _normalize_token_list(
        bullets.get("corpora", " ".join(DEFAULT_CANONICAL_CORPORA))
    )
    limit_match = re.search(r"Article limit per corpus:\s*`?(\d+)`?", text)
    limit = int(limit_match.group(1)) if limit_match else DEFAULT_CANONICAL_LIMIT

    return CanonicalProtocol(
        runner=runner,
        mode=mode,
        kernels=tuple(kernels or DEFAULT_CANONICAL_KERNELS),
        seeds=tuple(seeds or DEFAULT_CANONICAL_SEEDS),
        channels=tuple(channels or DEFAULT_CANONICAL_CHANNELS),
        corpora=tuple(corpora or DEFAULT_CANONICAL_CORPORA),
        limit=limit,
        main_command=main_command,
        synthetic_command=synthetic_command,
    )


def _resolve_repo_root(runs_dir: Path) -> Path:
    return runs_dir.resolve().parent.parent.parent


def _resolve_path(raw: Optional[str], *, repo_root: Path, base_dir: Path) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [
        repo_root / path,
        base_dir / path,
        base_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_manifest_path(raw: str, *, repo_root: Path, runs_dir: Path) -> Path:
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
        if not candidate.exists():
            candidate = (runs_dir / raw.strip()).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_dir():
        candidate = candidate / "experiment_manifest.json"
    return candidate


def _normalize_manifest_paths(
    raw_paths: Optional[Sequence[Path]],
    *,
    repo_root: Path,
    runs_dir: Path,
) -> List[Path]:
    normalized: List[Path] = []
    seen: Set[Path] = set()
    for raw_path in raw_paths or []:
        candidate = _normalize_manifest_path(str(raw_path), repo_root=repo_root, runs_dir=runs_dir)
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _manifest_run_id(manifest_path: Path) -> str:
    return manifest_path.parent.name


def _normalize_run_id_filter(run_ids: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if not run_ids:
        return None
    normalized = {str(run_id).strip() for run_id in run_ids if str(run_id).strip()}
    return normalized or None


def _extract_track5_mode(
    manifest: Dict[str, Any],
    baseline_meta: Optional[Dict[str, Any]],
    experiment_blob: Optional[Dict[str, Any]] = None,
) -> str:
    for source in (
        baseline_meta or {},
        experiment_blob or {},
        manifest.get("config") or {},
        manifest.get("synthetic_result") or {},
    ):
        mode = source.get("track5_assembly_mode") or source.get("track5_mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    return "hadamard_strict"


def _load_baseline_meta(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "baseline_meta.json"
    return _load_json(path) if path.exists() else {}


def _load_view_state(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "MONOLITH.view_state.json"
    return _load_json(path) if path.exists() else {}


def _normalize_control_metric_basis(raw: Optional[str]) -> str:
    value = str(raw or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "direct": "direct_observer_payload",
        "direct_observer": "direct_observer_payload",
        "direct_observer_payload": "direct_observer_payload",
        "comprehensive": "comprehensive_results",
        "comprehensive_results": "comprehensive_results",
    }
    if value not in aliases:
        raise ValueError("control metric basis must be one of: auto, direct, comprehensive")
    return aliases[value]


def _load_control_metrics(run_dir: Path, *, metric_basis: str = "auto") -> Dict[str, Any]:
    basis = _normalize_control_metric_basis(metric_basis)
    selected_path: Optional[Path] = None
    if basis != "auto":
        basis_path = run_dir / f"control_metrics.{basis}.json"
        if basis_path.exists():
            selected_path = basis_path
    path = run_dir / "control_metrics.json"
    if selected_path is None and path.exists():
        selected_path = path
    if selected_path is None:
        return {}
    blob = dict(_load_json(selected_path))
    blob.setdefault("requested_metric_basis", basis)
    blob["_evidence_requested_metric_basis"] = basis
    blob["_evidence_control_metric_basis_path"] = str(selected_path)
    return blob


def _metric_basis_from_blob(blob: Dict[str, Any], *, default: str = "unknown") -> str:
    return str(blob.get("metric_basis") or blob.get("requested_metric_basis") or default)


def _load_control_metric_snapshot(run_dir: Path, basis: str) -> Dict[str, Any]:
    """Load one named control-metric basis without silently mixing bases.

    Historical runs wrote only ``control_metrics.json``. We treat that default
    payload as direct observer evidence unless it explicitly declares another
    basis; comprehensive/integrated evidence must either be explicit or declared.
    """
    normalized = _normalize_control_metric_basis(basis)
    if normalized == "auto":
        return _load_control_metrics(run_dir)

    explicit_path = run_dir / f"control_metrics.{normalized}.json"
    if explicit_path.exists():
        blob = dict(_load_json(explicit_path))
        blob.setdefault("metric_basis", normalized)
        blob["_evidence_requested_metric_basis"] = normalized
        blob["_evidence_control_metric_basis_path"] = str(explicit_path)
        return blob

    default_path = run_dir / "control_metrics.json"
    if not default_path.exists():
        return {}
    default_blob = dict(_load_json(default_path))
    default_basis = _metric_basis_from_blob(
        default_blob,
        default="direct_observer_payload",
    )
    if default_basis == normalized:
        default_blob.setdefault("metric_basis", default_basis)
        default_blob["_evidence_requested_metric_basis"] = normalized
        default_blob["_evidence_control_metric_basis_path"] = str(default_path)
        return default_blob

    alternate = (default_blob.get("alternate_sources") or {}).get(normalized)
    if isinstance(alternate, dict):
        blob = dict(alternate)
        blob.setdefault("metric_basis", normalized)
        blob["_evidence_requested_metric_basis"] = normalized
        blob["_evidence_control_metric_basis_path"] = f"{default_path}#alternate_sources.{normalized}"
        return blob

    return {}


def _load_ablation_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "ablation_summary.json"
    return _load_json(path) if path.exists() else {}


def _load_relativity(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "relativity_deltas.json"
    return _load_json(path) if path.exists() else {}


def _load_verification(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "verification_report.json"
    return _load_json(path) if path.exists() else {}


def _load_monolith_rows(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / "MONOLITH_DATA.csv"
    return _load_csv_rows(path) if path.exists() else []


def _load_walker_states(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "walker_states.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _flatten_controls_blob(blob: Dict[str, Any]) -> Dict[str, Any]:
    metrics = blob.get("metrics") or {}
    real = (blob.get("controls") or {}).get("Real") or {}
    controls = blob.get("controls") or {}
    alternate_sources = blob.get("alternate_sources") or {}
    alternate_comprehensive = (alternate_sources.get("comprehensive_results") or {}).get("metrics") or {}
    alternate_direct = (alternate_sources.get("direct_observer_payload") or {}).get("metrics") or {}
    procrustes_block = blob.get("procrustes_real_vs_controls") or {}
    procrustes_rows = [
        row for row in (procrustes_block.get("rows") if isinstance(procrustes_block, dict) else []) or []
        if isinstance(row, dict)
    ]
    procrustes_per_control_ratios = [
        value
        for value in (
            _safe_float(row.get("ratio_real_over_control"))
            for row in procrustes_rows
        )
        if value is not None
    ]
    procrustes_min_control_ratio = _safe_float(
        metrics.get("procrustes_min_control_ratio"),
        _safe_float(
            procrustes_block.get("min_ratio_real_over_control")
            if isinstance(procrustes_block, dict)
            else None
        ),
    )
    if procrustes_min_control_ratio is None and procrustes_per_control_ratios:
        procrustes_min_control_ratio = min(procrustes_per_control_ratios)
    stochastic_variance_values = [
        _safe_float(((controls.get(name) or {}).get("simple_variance") or {}).get("mean"))
        for name in ("Shuffled", "Random")
    ]
    stochastic_variance_mean = _mean([
        value for value in stochastic_variance_values if value is not None
    ])
    real_simple_variance = _safe_float(((real.get("simple_variance") or {}).get("mean")))
    stochastic_distance_corr_values = [
        _safe_float(((controls.get(name) or {}).get("distance_corr") or {}).get("mean"))
        for name in ("Shuffled", "Random")
    ]
    stochastic_distance_corr_mean = _mean([
        value for value in stochastic_distance_corr_values if value is not None
    ])
    real_distance_corr = _safe_float(((real.get("distance_corr") or {}).get("mean")))
    distance_corr_stochastic_ratio = None
    if (
        real_distance_corr is not None
        and stochastic_distance_corr_mean is not None
        and abs(float(stochastic_distance_corr_mean)) > 1e-12
    ):
        distance_corr_stochastic_ratio = float(real_distance_corr / float(stochastic_distance_corr_mean))
    metric_variance_ratio = _safe_float(metrics.get("simple_variance_stochastic_ratio"))
    if metric_variance_ratio is None:
        metric_variance_ratio = _safe_float(metrics.get("simple_variance_ratio"))
    if metric_variance_ratio is None and real_simple_variance is not None and stochastic_variance_mean:
        metric_variance_ratio = (
            float(real_simple_variance / stochastic_variance_mean)
            if abs(stochastic_variance_mean) > 1e-12 else None
        )
    return {
        "status": str(blob.get("status", "MISSING")).upper(),
        "synthetic_placeholder": bool(blob.get("synthetic_placeholder", False)),
        "metric_basis": str(blob.get("metric_basis") or "unknown"),
        "requested_metric_basis": str(
            blob.get("requested_metric_basis")
            or blob.get("_evidence_requested_metric_basis")
            or "unknown"
        ),
        "evidence_requested_metric_basis": str(blob.get("_evidence_requested_metric_basis") or "unknown"),
        "evidence_control_metric_basis_path": str(blob.get("_evidence_control_metric_basis_path") or ""),
        "primary_metric_source": str(blob.get("primary_metric_source") or blob.get("source") or "unknown"),
        "procrustes_ratio": _safe_float(metrics.get("procrustes_ratio")),
        "procrustes_min_control_ratio": procrustes_min_control_ratio,
        "procrustes_per_control": [
            {
                "control": str(row.get("control") or "unknown"),
                "real_mean": _safe_float(row.get("real_mean")),
                "control_mean": _safe_float(row.get("control_mean")),
                "ratio_real_over_control": _safe_float(row.get("ratio_real_over_control")),
            }
            for row in procrustes_rows
        ],
        "distance_corr_ratio": _safe_float(metrics.get("distance_corr_ratio")),
        "distance_corr_real": real_distance_corr,
        "distance_corr_stochastic_control_mean": stochastic_distance_corr_mean,
        "distance_corr_stochastic_ratio": distance_corr_stochastic_ratio,
        "simple_variance_ratio": metric_variance_ratio,
        "simple_variance_stochastic_ratio": metric_variance_ratio,
        "alternate_comprehensive_simple_variance_stochastic_ratio": _safe_float(
            alternate_comprehensive.get("simple_variance_stochastic_ratio")
        ),
        "alternate_direct_simple_variance_stochastic_ratio": _safe_float(
            alternate_direct.get("simple_variance_stochastic_ratio")
        ),
        "simple_variance_real": _safe_float(metrics.get("simple_variance_real"), real_simple_variance),
        "simple_variance_control_avg": _safe_float(metrics.get("simple_variance_control_avg")),
        "simple_variance_stochastic_control_mean": _safe_float(
            metrics.get("simple_variance_stochastic_control_mean"),
            stochastic_variance_mean,
        ),
        "separates_count": _safe_int(metrics.get("separates_count")),
        "consensus_pct": _safe_float(metrics.get("consensus_pct")),
        "residual_pct": _safe_float(metrics.get("residual_pct")),
        "real_procrustes_std": _safe_float(((real.get("procrustes") or {}).get("std"))),
        "real_distance_corr_std": _safe_float(((real.get("distance_corr") or {}).get("std"))),
        "seeds": list(real.get("seeds") or []),
    }


def _metric_value(section: Dict[str, Any], metric: str) -> Optional[float]:
    value = section.get(metric)
    if isinstance(value, dict):
        for key in ("mean", "value", "score", "ratio"):
            numeric = _safe_float(value.get(key))
            if numeric is not None:
                return numeric
        return None
    return _safe_float(value)


def _variance_separation_stats(
    real_value: Optional[float],
    control_value: Optional[float],
    *,
    threshold: float = CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN,
) -> Dict[str, Any]:
    ratio = None
    log_ratio = None
    abs_log_ratio = None
    if (
        real_value is not None
        and control_value is not None
        and real_value > 0.0
        and control_value > 0.0
    ):
        ratio = float(real_value / control_value)
        log_ratio = float(math.log(ratio))
        abs_log_ratio = abs(log_ratio)

    if abs_log_ratio is None:
        direction = "undefined"
        passes = False
    elif abs_log_ratio < 0.05:
        direction = "near_parity"
        passes = False
    elif ratio is not None and ratio > 1.0:
        direction = "real_gt_control"
        passes = abs_log_ratio >= threshold
    else:
        direction = "real_lt_control"
        passes = abs_log_ratio >= threshold

    return {
        "ratio_real_over_control": ratio,
        "log_ratio": log_ratio,
        "abs_log_ratio": abs_log_ratio,
        "direction": direction,
        "passes_separation_threshold": bool(passes),
    }


def _cartography_row(
    *,
    leaf_common: Dict[str, Any],
    basis: str,
    basis_path: str,
    metric: str,
    control_family: str,
    real_value: Optional[float],
    control_value: Optional[float],
) -> Dict[str, Any]:
    return {
        **leaf_common,
        "basis": basis,
        "metric_basis": basis,
        "metric_basis_path": basis_path,
        "metric": metric,
        "control_family": control_family,
        "real_value": real_value,
        "control_value": control_value,
        **_variance_separation_stats(real_value, control_value),
    }


def _metric_signal_cartography_rows(
    run_dir: Path,
    leaf_common: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for basis in CONTROL_SIGNAL_CARTOGRAPHY_BASES:
        blob = _load_control_metric_snapshot(run_dir, basis)
        if not blob or str(blob.get("status", "MISSING")).upper() != "OK":
            continue

        controls = blob.get("controls") or {}
        real = controls.get("Real") or {}
        metrics = blob.get("metrics") or {}
        basis_path = str(blob.get("_evidence_control_metric_basis_path") or "")

        for metric in CONTROL_SIGNAL_CARTOGRAPHY_METRICS:
            real_value = _metric_value(real, metric)
            if metric == "simple_variance" and real_value is None:
                real_value = _safe_float(metrics.get("simple_variance_real"))

            for control_name in CONTROL_SIGNAL_CARTOGRAPHY_CONTROLS:
                control_section = controls.get(control_name) or {}
                control_value = _metric_value(control_section, metric)
                if metric == "simple_variance" and control_value is None and control_name != "Constant":
                    control_value = _safe_float(metrics.get("simple_variance_stochastic_control_mean"))
                if real_value is None and control_value is None:
                    continue
                rows.append(
                    _cartography_row(
                        leaf_common=leaf_common,
                        basis=basis,
                        basis_path=basis_path,
                        metric=metric,
                        control_family=control_name,
                        real_value=real_value,
                        control_value=control_value,
                    )
                )

            stochastic_values = [
                _metric_value(controls.get(name) or {}, metric)
                for name in ("Shuffled", "Random")
            ]
            stochastic_values = [value for value in stochastic_values if value is not None]
            stochastic_control_value = _mean(stochastic_values)
            if metric == "simple_variance":
                if real_value is None:
                    real_value = _safe_float(metrics.get("simple_variance_real"))
                if stochastic_control_value is None:
                    stochastic_control_value = _safe_float(metrics.get("simple_variance_stochastic_control_mean"))
                    if stochastic_control_value is None:
                        ratio = _safe_float(metrics.get("simple_variance_stochastic_ratio"))
                        if ratio is not None and real_value is not None and ratio > 0:
                            stochastic_control_value = float(real_value / ratio)
            if real_value is not None or stochastic_control_value is not None:
                rows.append(
                    _cartography_row(
                        leaf_common=leaf_common,
                        basis=basis,
                        basis_path=basis_path,
                        metric=metric,
                        control_family="stochastic_controls",
                        real_value=real_value,
                        control_value=stochastic_control_value,
                    )
                )

        procrustes_block = blob.get("procrustes_real_vs_controls") or {}
        procrustes_rows = procrustes_block.get("rows") if isinstance(procrustes_block, dict) else []
        for row in procrustes_rows or []:
            if not isinstance(row, dict):
                continue
            control_name = str(row.get("control") or "unknown")
            real_value = _safe_float(row.get("real_mean"))
            control_value = _safe_float(row.get("control_mean"))
            rows.append(
                _cartography_row(
                    leaf_common=leaf_common,
                    basis=basis,
                    basis_path=basis_path,
                    metric="procrustes",
                    control_family=control_name,
                    real_value=real_value,
                    control_value=control_value,
                )
            )

    return rows


def _summarize_variance_separation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    simple_rows = [row for row in rows if row.get("metric") == "simple_variance"]
    primary_rows = [
        row for row in simple_rows
        if row.get("basis") == "comprehensive_results"
        and row.get("control_family") == "stochastic_controls"
    ]
    direct_rows = [
        row for row in simple_rows
        if row.get("basis") == "direct_observer_payload"
        and row.get("control_family") == "stochastic_controls"
    ]
    primary_kernel_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in primary_rows:
        primary_kernel_rows.setdefault(str(row.get("kernel") or "unknown"), []).append(row)
    observed_required = [
        kernel for kernel in ("rbf", "matern", "imq")
        if kernel in primary_kernel_rows
    ]
    required_kernels = observed_required or sorted(primary_kernel_rows)
    per_kernel = {
        kernel: {
            "n_rows": len(kernel_rows),
            "mean_abs_log_ratio": _mean([row.get("abs_log_ratio") for row in kernel_rows]),
            "directions": sorted({str(row.get("direction")) for row in kernel_rows}),
            "pass_rate": (
                sum(1 for row in kernel_rows if row.get("passes_separation_threshold"))
                / float(len(kernel_rows))
                if kernel_rows else 0.0
            ),
            "all_rows_pass": bool(kernel_rows) and all(
                row.get("passes_separation_threshold") for row in kernel_rows
            ),
            "passes": (
                bool(kernel_rows)
                and (
                    _mean([row.get("abs_log_ratio") for row in kernel_rows]) or 0.0
                ) >= CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN
                and any(row.get("passes_separation_threshold") for row in kernel_rows)
            ),
        }
        for kernel, kernel_rows in sorted(primary_kernel_rows.items())
    }
    primary_pass = (
        bool(primary_rows)
        and bool(required_kernels)
        and all(
            per_kernel.get(kernel, {}).get("passes") is True
            for kernel in required_kernels
        )
    )
    direct_pass = bool(direct_rows) and all(row.get("passes_separation_threshold") for row in direct_rows)

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "primary_basis": "comprehensive_results",
        "sensitivity_basis": "direct_observer_payload",
        "primary_metric": "simple_variance",
        "primary_control_family": "stochastic_controls",
        "threshold_abs_log_ratio_min": CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN,
        "required_kernel_policy": "rbf/matern/imq when present in selected evidence",
        "required_kernels_evaluated": required_kernels,
        "per_kernel": per_kernel,
        "records": primary_rows,
        "sensitivity_records": direct_rows,
        "mean_primary_abs_log_ratio": _mean([row.get("abs_log_ratio") for row in primary_rows]),
        "mean_direct_abs_log_ratio": _mean([row.get("abs_log_ratio") for row in direct_rows]),
        "primary_pass": primary_pass,
        "direct_sensitivity_pass": direct_pass,
        "pass": primary_pass,
        "thesis_safe": primary_pass,
        "effect_direction": "direction_free_variance_displacement",
        "interpretation": (
            "Variance separation is evaluated as absolute log displacement. Direct observer payload "
            "failure is sensitivity evidence, not a failure of the integrated comprehensive-basis claim."
        ),
    }


def _evaluate_stochastic_variance_controls(blob: Dict[str, Any]) -> Dict[str, Any]:
    flat = _flatten_controls_blob(blob)
    ratio = flat.get("simple_variance_ratio")
    thesis_safe = (
        flat["status"] == "OK"
        and not flat["synthetic_placeholder"]
        and ratio is not None
        and ratio <= CONTROL_STOCHASTIC_VARIANCE_RATIO_MAX
    )
    return {
        **flat,
        "effect_direction": (
            "real_lower_variance_than_shuffled_random"
            if thesis_safe else "weak_or_absent"
        ),
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
    }


def _evaluate_controls(blob: Dict[str, Any]) -> Dict[str, Any]:
    flat = _flatten_controls_blob(blob)
    per_control_ratio = flat.get("procrustes_min_control_ratio")
    procrustes_ok = (
        per_control_ratio >= CONTROL_PROCRUSTES_RATIO_MIN
        if per_control_ratio is not None
        else (
            flat["procrustes_ratio"] is not None
            and flat["procrustes_ratio"] >= CONTROL_PROCRUSTES_RATIO_MIN
        )
    )
    ratios_ok = (
        procrustes_ok
        and (flat["separates_count"] or 0) >= CONTROL_SEPARATION_COUNT_MIN
        and flat["distance_corr_ratio"] is not None
        and flat["distance_corr_ratio"] <= CONTROL_DISTANCE_CORR_RATIO_MAX
    )
    thesis_safe = (
        flat["status"] == "OK"
        and not flat["synthetic_placeholder"]
        and ratios_ok
    )
    return {
        **flat,
        "effect_direction": "real_separates_from_controls" if thesis_safe else "weak_or_absent",
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
    }


def _evaluate_ablation(blob: Dict[str, Any], track5_mode: str) -> Dict[str, Any]:
    metrics = blob.get("metrics") or {}
    placeholder = bool(blob.get("synthetic_placeholder", False))
    status = str(blob.get("status", "MISSING")).upper()
    stage_1 = _safe_float(metrics.get("stage_1_alignment_score") or blob.get("stage_1_alignment_score"))
    stage_2 = _safe_float(metrics.get("stage_2_alignment_score") or blob.get("stage_2_alignment_score"))
    stage_3 = _safe_float(metrics.get("stage_3_survival_rate") or blob.get("stage_3_survival_rate"))
    delta = _safe_float(metrics.get("delta_alignment_score") or blob.get("delta_alignment_score"))
    complete = all(value is not None for value in (stage_1, stage_2, stage_3, delta))
    effect_pass = (
        complete
        and (stage_2 or 0.0) >= TRACK5_STAGE2_ALIGNMENT_MIN
        and (stage_3 or 0.0) >= TRACK5_STAGE3_SURVIVAL_MIN
        and (delta or 0.0) >= TRACK5_DELTA_ALIGNMENT_MIN
    )
    thesis_safe = status == "OK" and not placeholder and effect_pass
    return {
        "status": status,
        "synthetic_placeholder": placeholder,
        "track5_mode": track5_mode,
        "stage_1_alignment_score": stage_1,
        "stage_2_alignment_score": stage_2,
        "stage_3_survival_rate": stage_3,
        "delta_alignment_score": delta,
        "effect_pass": bool(effect_pass),
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
        "reason": blob.get("reason") or blob.get("message"),
    }


def _evaluate_relativity(blob: Dict[str, Any]) -> Dict[str, Any]:
    summary = blob.get("summary") or {}
    observers = blob.get("observers") or []
    rotations: List[float] = []
    path_flips: List[int] = []
    equivalence_true = 0
    for observer in observers:
        null_eq = observer.get("null_observer_equivalence") or {}
        rotation = _safe_float((observer.get("axis_delta") or {}).get("rotation_deg"))
        if rotation is None:
            rotation = _safe_float(null_eq.get("axis_rotation_deg"))
        if rotation is not None:
            rotations.append(rotation)
        flips = _safe_int(null_eq.get("path_flip_count"))
        if flips is None:
            flips = _safe_int((observer.get("translation_only_comparison") or {}).get("d_path_flip_count"))
        if flips is not None:
            path_flips.append(flips)
        if bool(null_eq.get("equivalent", False)):
            equivalence_true += 1

    mean_delta = _safe_float(summary.get("mean_coord_delta"))
    max_delta = _safe_float(summary.get("max_coord_delta"))
    mean_rotation = _mean(rotations)
    max_rotation = max(rotations) if rotations else None
    max_path_flip = max(path_flips) if path_flips else 0
    placeholder = bool(blob.get("synthetic_placeholder", False))
    thesis_safe = (
        str(blob.get("status", "MISSING")).upper() == "OK"
        and not placeholder
        and (mean_delta or 0.0) >= RELATIVITY_MEAN_DELTA_MIN
        and (max_rotation or 0.0) >= RELATIVITY_ROTATION_MIN_DEG
        and max_path_flip > 0
        and equivalence_true < len(observers)
    )
    return {
        "status": str(blob.get("status", "MISSING")).upper(),
        "synthetic_placeholder": placeholder,
        "observer_count": _safe_int(blob.get("observer_count"), default=len(observers)),
        "mean_coord_delta": mean_delta,
        "max_coord_delta": max_delta,
        "mean_rotation_deg": mean_rotation,
        "max_rotation_deg": max_rotation,
        "max_path_flip_count": max_path_flip,
        "null_equivalence_count": equivalence_true,
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
    }


def _evaluate_relativity_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    observer_count = _safe_int(summary.get("observer_count"), default=0) or 0
    null_equivalent_count = _safe_int(summary.get("null_equivalent_count"), default=0) or 0
    max_rotation = _safe_float(summary.get("max_abs_rotation_deg"))
    mean_rotation = _safe_float(summary.get("mean_abs_rotation_deg"))
    mean_delta = _safe_float(summary.get("mean_max_coord_delta"))
    max_delta = _safe_float(summary.get("max_coord_delta"))
    path_flip = _safe_float(summary.get("mean_path_flip_count"))
    thesis_safe = bool(summary.get("safe_for_thesis_claim", False))
    return {
        "status": str(summary.get("status", "MISSING")).upper(),
        "synthetic_placeholder": False,
        "observer_count": observer_count,
        "mean_coord_delta": mean_delta,
        "max_coord_delta": max_delta,
        "mean_rotation_deg": mean_rotation,
        "max_rotation_deg": max_rotation,
        "max_path_flip_count": path_flip,
        "null_equivalence_count": null_equivalent_count,
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
        "failure_reasons": list(summary.get("failure_reasons") or []),
    }


def _zone_stats(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        zone = str(row.get("zone", "")).strip()
        if not zone:
            continue
        grouped.setdefault(zone, {"density": [], "stress": [], "w_actual": []})
        density = _safe_float(row.get("density"))
        stress = _safe_float(row.get("stress"))
        work = _safe_float(row.get("w_actual"))
        if density is not None:
            grouped[zone]["density"].append(density)
        if stress is not None:
            grouped[zone]["stress"].append(stress)
        if work is not None:
            grouped[zone]["w_actual"].append(work)
    out: Dict[str, Dict[str, float]] = {}
    for zone, metrics in grouped.items():
        out[zone] = {
            "count": float(len(metrics["density"]) or len(metrics["stress"]) or len(metrics["w_actual"])),
            "mean_density": _mean(metrics["density"]) or 0.0,
            "mean_stress": _mean(metrics["stress"]) or 0.0,
            "mean_work": _mean(metrics["w_actual"]) or 0.0,
        }
    return out


def _evaluate_track4(view_state: Dict[str, Any], rows: Sequence[Dict[str, str]], walker_states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = view_state.get("metrics") or {}
    mean_work = _safe_float(metrics.get("walker_mean_action"))
    survival = _safe_float(metrics.get("walker_survival_rate"))
    if mean_work is None and walker_states:
        mean_work = _mean([
            _safe_float(row.get("work_integral")) or 0.0
            for row in walker_states
        ])
    if survival is None and walker_states:
        closed = [1.0 for row in walker_states if bool(row.get("closed_loop"))]
        survival = sum(closed) / float(len(walker_states)) if walker_states else None

    zone_stats = _zone_stats(rows)
    zone_names = sorted(zone_stats.keys())
    density_means = [zone_stats[name]["mean_density"] for name in zone_names]
    stress_means = [zone_stats[name]["mean_stress"] for name in zone_names]
    density_span = (max(density_means) - min(density_means)) if len(density_means) >= 2 else 0.0
    stress_span = (max(stress_means) - min(stress_means)) if len(stress_means) >= 2 else 0.0

    thesis_safe = (
        mean_work is not None
        and mean_work >= TRACK4_WORK_MIN
        and survival is not None
        and survival >= TRACK4_SURVIVAL_MIN
        and len(zone_names) >= 3
        and density_span >= TRACK4_ZONE_MEAN_GAP_MIN
        and stress_span >= TRACK4_ZONE_MEAN_GAP_MIN
    )
    return {
        "mean_work": mean_work,
        "survival_rate": survival,
        "zone_count": len(zone_names),
        "zones": zone_stats,
        "density_span": density_span,
        "stress_span": stress_span,
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
    }


def _evaluate_track4_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    terrain_evidence_basis = str(summary.get("terrain_evidence_basis") or "anchor")
    zones = summary.get("primary_zone_summary") or summary.get("zone_summary") or {}
    if not isinstance(zones, dict):
        zones = {}
    bridge_vs_void = summary.get("primary_bridge_vs_void") or summary.get("bridge_vs_void") or {}
    if not isinstance(bridge_vs_void, dict):
        bridge_vs_void = {}
    density_span = None
    stress_span = None
    work_gap = _safe_float(bridge_vs_void.get("work_integral_gap"))
    closed_gap = _safe_float(bridge_vs_void.get("closed_loop_rate_gap"))
    if work_gap is not None:
        stress_span = work_gap
    if closed_gap is not None:
        density_span = abs(closed_gap)
    failure_reasons = list(summary.get("failure_reasons") or [])
    warnings = list(summary.get("warnings") or [])
    zone_count = len(zones)
    hard_failures: List[str] = []
    if zone_count < 3:
        if terrain_evidence_basis == "path_touched":
            hard_failures.append("track 4 path-touched terrain coverage fewer than three terrain zones")
        else:
            hard_failures.append("track 4 anchors cover fewer than three terrain zones")
    if work_gap is None and closed_gap is None:
        hard_failures.append("bridge/void comparison unavailable for this run")
    if closed_gap is not None and closed_gap <= TRACK4_ZONE_MEAN_GAP_MIN:
        hard_failures.append("bridge/void closed-loop gap below minimum semantic effect size")
    if work_gap is not None and work_gap <= TRACK4_ZONE_MEAN_GAP_MIN:
        hard_failures.append("bridge/void work-integral gap below minimum semantic effect size")
    unique_path_shapes = _safe_float(summary.get("unique_path_shape_count"))
    if unique_path_shapes is not None and unique_path_shapes <= 1.0:
        hard_failures.append("all Track 4 paths collapse to one repeated index trace")
    hot_count = _safe_float(summary.get("hot_count"))
    cold_count = _safe_float(summary.get("cold_count"))
    if (hot_count is not None and hot_count <= 0.0) or (cold_count is not None and cold_count <= 0.0):
        hard_failures.append("track 4 did not preserve both hot and cold walkers")
    warning_text = " | ".join(str(item).lower() for item in warnings)
    if "bridge/void comparison unavailable" in warning_text:
        hard_failures.append("bridge/void comparison unavailable for this run")
    if "all track 4 paths collapse" in warning_text:
        hard_failures.append("all Track 4 paths collapse to one repeated index trace")
    merged_failures = list(dict.fromkeys([*failure_reasons, *hard_failures]))
    thesis_safe = bool(summary.get("safe_for_thesis_claim", False)) and not hard_failures
    return {
        "mean_work": _safe_float(summary.get("mean_work_integral")),
        "survival_rate": _safe_float(summary.get("closed_loop_rate")),
        "zone_count": zone_count,
        "zones": zones,
        "terrain_evidence_basis": terrain_evidence_basis,
        "density_span": density_span if density_span is not None else 0.0,
        "stress_span": stress_span if stress_span is not None else 0.0,
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
        "failure_reasons": merged_failures,
        "warnings": warnings,
    }


def _evaluate_verification(blob: Dict[str, Any]) -> Dict[str, Any]:
    layers = blob.get("layers") or []
    failed_layers = [layer for layer in layers if str(layer.get("status", "")).upper() != "VERIFIED"]
    failed_check_names: List[str] = []
    fail_reasons: List[str] = []
    seed_checks: List[float] = []
    for layer in layers:
        for reason in layer.get("fail_reasons") or []:
            text = str(reason).strip()
            if text:
                fail_reasons.append(text)
        for check in layer.get("checks") or []:
            name = str(check.get("name", "")).strip()
            if name and check.get("pass") is False:
                failed_check_names.append(name)
            if str(check.get("name", "")).strip() == "seed_stability":
                value = _safe_float(check.get("value"))
                if value is not None:
                    seed_checks.append(value)
    return {
        "status": str(blob.get("verification_status") or ("VERIFIED" if blob.get("global_pass") else "UNVERIFIED")).upper(),
        "global_pass": bool(blob.get("global_pass", False)),
        "failed_layers": len(failed_layers),
        "seed_stability_cv": _mean(seed_checks),
        "failed_check_names": sorted(set(failed_check_names)),
        "fail_reasons": list(dict.fromkeys(fail_reasons)),
    }


def _evaluate_procrustes_verification(blob: Dict[str, Any]) -> Dict[str, Any]:
    """Profile-specific provenance check for the Procrustes-control claim.

    The broad verifier still encodes older full-system directional gates such as
    control ordering. For the Procrustes-control profile, those verdict gates are
    not part of the claim; CRN lock, seed stability, and provenance metadata are.
    """
    layers = blob.get("layers") or []
    required_checks = set(PROCRUSTES_PROVENANCE_REQUIRED_CHECKS)
    ignored_failures = set(PROCRUSTES_PROVENANCE_IGNORED_FAILURES)
    optional_checks = set(PROCRUSTES_PROVENANCE_OPTIONAL_CHECKS)
    found_required: Set[str] = set()
    required_failures: List[str] = []
    unexpected_failures: List[str] = []
    ignored_failure_names: List[str] = []
    seed_checks: List[float] = []
    missing_metadata = [
        field for field in ("dataset_hash", "code_hash_or_commit", "weights_hash", "kernel_params", "crn_seed")
        if blob.get(field) in (None, "", {})
    ]
    for layer in layers:
        for check in layer.get("checks") or []:
            name = str(check.get("name") or "").strip()
            if not name:
                continue
            passed = check.get("pass")
            if name == "seed_stability":
                value = _safe_float(check.get("value"))
                if value is not None:
                    seed_checks.append(value)
            if name in required_checks:
                found_required.add(name)
                if passed is not True:
                    required_failures.append(name)
            elif passed is False:
                if name in ignored_failures:
                    ignored_failure_names.append(name)
                else:
                    unexpected_failures.append(name)
            elif passed is None and name not in optional_checks:
                unexpected_failures.append(name)
    missing_checks = sorted(required_checks - found_required)
    safe = (
        bool(blob)
        and bool(layers)
        and not missing_metadata
        and not missing_checks
        and not required_failures
        and not unexpected_failures
    )
    return {
        "status": "PROFILE_VERIFIED" if safe else "PROFILE_UNVERIFIED",
        "global_pass": bool(blob.get("global_pass", False)),
        "pass": safe,
        "thesis_safe": safe,
        "failed_layers": sum(1 for layer in layers if str(layer.get("status", "")).upper() != "VERIFIED"),
        "seed_stability_cv": _mean(seed_checks),
        "missing_metadata": missing_metadata,
        "missing_required_checks": missing_checks,
        "required_check_failures": sorted(set(required_failures)),
        "unexpected_check_failures": sorted(set(unexpected_failures)),
        "ignored_check_failures": sorted(set(ignored_failure_names)),
        "ignored_failure_policy": sorted(ignored_failures),
    }


def _build_claim_strategy(
    claims_by_id: Dict[str, Dict[str, Any]],
    scientific_validation_summary: Dict[str, Any],
    canonical_freeze: Dict[str, Any],
    narrative_audit: Dict[str, Any],
) -> Dict[str, Any]:
    recommendations: List[Dict[str, Any]] = []
    for claim_id, claim in claims_by_id.items():
        thesis_safe = bool(claim.get("thesis_safe"))
        action = "keep" if thesis_safe else "review_manually"
        reason = "Current focused evidence supports the claim."
        next_step = "Keep this claim in the focused defense bundle."
        kernel_note = None

        if claim_id == "control_destruction" and not thesis_safe:
            action = "narrow"
            reason = (
                "Procrustes separation is consistently positive, but reviewer-facing "
                "distance-correlation support does not generalize across kernels."
            )
            next_step = "Cite `procrustes_control_separation` as the supported control claim."
            kernel_note = "RBF carries the strongest support; Matern and IMQ fail the distance-correlation gate."
        elif claim_id == "stochastic_control_variance_compression" and not thesis_safe:
            action = "retire"
            reason = "The real/stochastic variance ratio trends the wrong way in the focused bundle."
            next_step = "Remove this as a positive thesis claim unless future reruns reverse the effect direction."
        elif claim_id == "track4_traversal_validity" and not thesis_safe:
            action = "fix"
            reason = (
                "Track 4 still fails its terrain-semantic validity gate because anchor-zone coverage and "
                "bridge/void survival semantics are not robust enough across kernels."
            )
            next_step = "Patch Track 4 terrain semantics first, then rerun the focused real/control bundle."
            kernel_note = "Matern is closest to validity; RBF and IMQ still collapse terrain coverage under the current summary."
        elif claim_id == "track4_work_barrier_signal" and not thesis_safe:
            action = "narrow"
            reason = (
                "Mean work gaps are positive overall, but the strict minimum-gap requirement does not hold "
                "against every control family for every kernel."
            )
            next_step = "Present this as exploratory or kernel-local until the per-control minimum-gap gate is met everywhere."
            kernel_note = "RBF carries the clearest work-gap signal; Matern and IMQ remain below the strict minimum threshold."
        elif claim_id == "verification_provenance" and not thesis_safe:
            action = "fix"
            reason = "Verification artifacts remain present but one or more layer checks still fail."
            next_step = "Trace the failing verifier checks and refresh the leaf reports before citing the broad provenance claim."
        elif claim_id == "canonical_freeze" and not thesis_safe:
            action = "rerun"
            reason = (
                "The focused bundle is below the frozen protocol limit or is not fully registered in the thesis narrative."
            )
            next_step = "Run the canonical 120-article protocol and register the cited run IDs in RESULTS.md."
            kernel_note = (
                f"Protocol pass={canonical_freeze.get('pass')} | narrative pass={narrative_audit.get('pass')}"
            )

        recommendations.append(
            {
                "claim_id": claim_id,
                "thesis_safe": thesis_safe,
                "action": action,
                "reason": reason,
                "next_step": next_step,
                "kernel_note": kernel_note,
                "supporting_runs": list(claim.get("supporting_runs") or []),
            }
        )

    unsafe_claim_ids = [claim_id for claim_id, claim in claims_by_id.items() if not bool(claim.get("thesis_safe"))]
    safe_claim_ids = [claim_id for claim_id, claim in claims_by_id.items() if bool(claim.get("thesis_safe"))]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unsafe_claim_ids": unsafe_claim_ids,
        "safe_claim_ids": safe_claim_ids,
        "publishable_real_control_signal": bool(
            scientific_validation_summary.get("semantic_signal_interpretation", {}).get("publishable_real_control_signal", False)
        ),
        "recommendations": recommendations,
    }


def _ledger_record(
    leaf: Dict[str, Any],
    *,
    claim_id: str,
    artifact_family: str,
    present: bool,
    path: Optional[Path] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": leaf["run_id"],
        "kernel": leaf["kernel"],
        "channel": leaf["channel"],
        "corpus": leaf["corpus"],
        "run_dir": str(leaf["run_dir"]),
        "track5_mode": leaf.get("track5_mode"),
        "claim_id": claim_id,
        "artifact_family": artifact_family,
        "artifact_path": str(path) if path is not None else None,
        "present": bool(present),
        "detail": detail or ("present" if present else "missing"),
    }


def _missing_ledger_entries(
    ledger: Sequence[Dict[str, Any]],
    claim_id: str,
    *,
    corpus: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [
        row for row in ledger
        if row.get("claim_id") == claim_id
        and (corpus is None or row.get("corpus") == corpus)
        and not bool(row.get("present"))
    ]


def _xy_collapse(rows: Sequence[Dict[str, str]]) -> bool:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = _safe_float(row.get("x"))
        y = _safe_float(row.get("y"))
        if x is not None:
            xs.append(x)
        if y is not None:
            ys.append(y)
    if len(xs) < 2 or len(ys) < 2:
        return False
    return (max(xs) - min(xs) <= XY_COLLAPSE_EPS) or (max(ys) - min(ys) <= XY_COLLAPSE_EPS)


def _suite_leaf_records(
    manifest_path: Path,
    manifest: Dict[str, Any],
    *,
    repo_root: Path,
) -> Iterable[Dict[str, Any]]:
    for experiment in manifest.get("experiments") or []:
        if str(experiment.get("status", "")).lower() != "success":
            continue
        corpus = str(experiment.get("corpus", "")).strip()
        output_dir = _resolve_path(
            experiment.get("output_dir"),
            repo_root=repo_root,
            base_dir=manifest_path.parent,
        )
        if not output_dir:
            continue
        baseline_meta = _load_baseline_meta(output_dir)
        kernel_params = baseline_meta.get("kernel_params") or {}
        kernel = str(experiment.get("kernel") or kernel_params.get("kernel") or output_dir.parent.parent.name)
        channel = str(experiment.get("channel") or kernel_params.get("channel") or output_dir.parent.name)
        yield {
            "run_id": _manifest_run_id(manifest_path),
            "manifest_path": str(manifest_path),
            "run_dir": output_dir,
            "kernel": kernel,
            "channel": channel,
            "corpus": corpus,
            "seeds": list(experiment.get("seeds") or []),
            "track5_mode": _extract_track5_mode(manifest, baseline_meta, experiment),
        }


def _synthetic_records(
    manifest_path: Path,
    manifest: Dict[str, Any],
    *,
    repo_root: Path,
) -> Iterable[Dict[str, Any]]:
    synthetic = manifest.get("synthetic_result") or {}
    summary = synthetic.get("summary") or {}
    results = synthetic.get("results") or []
    output_dir = _resolve_path(synthetic.get("output_dir") or manifest.get("output_dir"), repo_root=repo_root, base_dir=manifest_path.parent)
    summary_path = (output_dir / "synthetic_summary.json") if output_dir else None
    summary_blob = _load_json(summary_path) if summary_path and summary_path.exists() else {"summary": summary, "results": results}
    status = str(synthetic.get("status", "")).lower()
    if not status and summary_path and summary_path.exists():
        status = "success"
    yield {
        "run_id": _manifest_run_id(manifest_path),
        "manifest_path": str(manifest_path),
        "summary": summary_blob.get("summary") or summary,
        "results": summary_blob.get("results") or results,
        "config": summary_blob.get("config") or {},
        "status": status,
    }


def _synthetic_summary_dir_for_manifest(manifest_path: Path) -> Optional[Path]:
    summary_dir = manifest_path.parent / "synthetic"
    if (summary_dir / "synthetic_summary.json").exists():
        return summary_dir
    return None


def _is_synthetic_manifest_or_recoverable(manifest_path: Path, manifest: Dict[str, Any]) -> bool:
    if str(manifest.get("experiment_type", "")).strip().lower() == "synthetic":
        return True
    return _synthetic_summary_dir_for_manifest(manifest_path) is not None


def _coerce_synthetic_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    if str(manifest.get("experiment_type", "")).strip().lower() == "synthetic":
        return manifest
    summary_dir = _synthetic_summary_dir_for_manifest(manifest_path)
    if summary_dir is None:
        return manifest
    recovered = dict(manifest)
    recovered["experiment_type"] = "synthetic"
    recovered["synthetic_result"] = {
        **(manifest.get("synthetic_result") or {}),
        "status": (manifest.get("synthetic_result") or {}).get("status") or "success",
        "output_dir": str(summary_dir),
        "recovered_from_summary": True,
    }
    return recovered


def _aggregate_synthetic(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    nmi_values: List[float] = []
    ari_values: List[float] = []
    kernels: Dict[str, List[float]] = {}
    seeds: Dict[str, List[float]] = {}
    failing_runs: List[str] = []
    for record in records:
        summary = record.get("summary") or {}
        for item in record.get("results") or []:
            nmi = _safe_float(item.get("nmi"))
            ari = _safe_float(item.get("ari"))
            kernel = str(item.get("kernel", "unknown"))
            seed = str(item.get("seed", "unknown"))
            if nmi is not None:
                nmi_values.append(nmi)
                kernels.setdefault(kernel, []).append(nmi)
                seeds.setdefault(seed, []).append(nmi)
            if ari is not None:
                ari_values.append(ari)
            if str(item.get("status", "")).lower() != "success":
                failing_runs.append(f"{record['run_id']}:{item.get('run_key', 'unknown')}")
        if not (record.get("results") or []):
            for value in summary.get("nmi_scores") or []:
                nmi = _safe_float(value)
                if nmi is not None:
                    nmi_values.append(nmi)
            for value in summary.get("ari_scores") or []:
                ari = _safe_float(value)
                if ari is not None:
                    ari_values.append(ari)
    mean_nmi = _mean(nmi_values)
    mean_ari = _mean(ari_values)
    std_nmi = _std(nmi_values)
    std_ari = _std(ari_values)
    per_kernel_mean_nmi = {kernel: _mean(values) for kernel, values in kernels.items()}
    per_seed_mean_nmi = {seed: _mean(values) for seed, values in seeds.items()}
    thesis_safe = (
        bool(nmi_values)
        and (mean_nmi or 0.0) >= SYNTHETIC_MEAN_NMI_MIN
        and (mean_ari or 0.0) >= SYNTHETIC_MEAN_ARI_MIN
        and bool(per_kernel_mean_nmi)
        and all((value or 0.0) >= SYNTHETIC_PER_KERNEL_NMI_MIN for value in per_kernel_mean_nmi.values())
        and bool(per_seed_mean_nmi)
        and all((value or 0.0) >= SYNTHETIC_PER_SEED_NMI_MIN for value in per_seed_mean_nmi.values())
        and not failing_runs
    )
    return {
        "n_runs": len(records),
        "n_results": len(nmi_values),
        "mean_nmi": mean_nmi,
        "std_nmi": std_nmi,
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "per_kernel_mean_nmi": per_kernel_mean_nmi,
        "per_seed_mean_nmi": per_seed_mean_nmi,
        "failing_runs": failing_runs,
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
        "effect_direction": "recovers_planted_structure" if thesis_safe else "weak_or_failed_recovery",
    }


def _suite_failure_modes(manifest_path: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for experiment in manifest.get("experiments") or []:
        status = str(experiment.get("status", "")).lower()
        if status and status != "success":
            failures.append(
                {
                    "run_id": _manifest_run_id(manifest_path),
                    "kernel": experiment.get("kernel"),
                    "channel": experiment.get("channel"),
                    "corpus": experiment.get("corpus"),
                    "status": status,
                    "error": experiment.get("error") or experiment.get("stderr") or experiment.get("stdout"),
                }
            )
    return failures


def _narrative_audit(results_path: Path, manifest_ids: Sequence[str]) -> Dict[str, Any]:
    text = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
    registry_present = "## Canonical Run Registry" in text
    cited_ids = set(re.findall(r"experiments_\d{8}_\d{6}(?:_synthetic\d+)?", text))
    selected_ids = set(manifest_ids)
    missing = sorted(cited_ids - selected_ids)
    return {
        "results_md_present": results_path.exists(),
        "canonical_registry_present": registry_present,
        "cited_run_count": len(cited_ids),
        "cited_missing_count": len(missing),
        "missing_run_ids": missing,
        "pass": results_path.exists() and registry_present and not missing,
        "thesis_safe": results_path.exists() and registry_present and not missing,
    }


def _focused_narrative_audit(results_path: Path, manifest_ids: Sequence[str]) -> Dict[str, Any]:
    text = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
    registry_present = "## Canonical Run Registry" in text
    cited_ids = set(re.findall(r"experiments_\d{8}_\d{6}(?:_synthetic\d+)?", text))
    selected_ids = set(manifest_ids)
    missing_selected = sorted(selected_ids - cited_ids)
    ignored_historical = sorted(cited_ids - selected_ids)
    return {
        "results_md_present": results_path.exists(),
        "canonical_registry_present": registry_present,
        "cited_run_count": len(cited_ids),
        "cited_missing_count": len(missing_selected),
        "missing_run_ids": missing_selected,
        "ignored_cited_run_count": len(ignored_historical),
        "ignored_cited_run_ids": ignored_historical,
        "focused_filter_active": True,
        "pass": results_path.exists() and registry_present and not missing_selected,
        "thesis_safe": results_path.exists() and registry_present and not missing_selected,
    }


def _evaluate_canonical_freeze(
    protocol: CanonicalProtocol,
    suite_manifests: Sequence[Tuple[Path, Dict[str, Any]]],
    synthetic_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    actual_kernels: set[str] = set()
    actual_channels: set[str] = set()
    actual_corpora: set[str] = set()
    actual_seeds: set[int] = set()
    actual_synthetic_kernels: set[str] = set()
    actual_synthetic_seeds: set[int] = set()
    actual_cells: Set[Tuple[str, str, str, int]] = set()
    actual_synthetic_cells: Set[Tuple[str, int]] = set()
    observed_suite_limits: Dict[str, Optional[int]] = {}
    under_protocol_limit: List[Dict[str, Any]] = []
    missing_limit_run_ids: List[str] = []
    successful_runs = 0
    failed_runs = 0
    for manifest_path, manifest in suite_manifests:
        run_id = _manifest_run_id(manifest_path)
        manifest_limit = _safe_int((manifest.get("config") or {}).get("limit"))
        observed_suite_limits[run_id] = manifest_limit
        if manifest_limit is None:
            missing_limit_run_ids.append(run_id)
        elif manifest_limit < int(protocol.limit):
            under_protocol_limit.append(
                {
                    "run_id": run_id,
                    "manifest": str(manifest_path),
                    "observed_limit": manifest_limit,
                    "required_limit": int(protocol.limit),
                }
            )
        for experiment in manifest.get("experiments") or []:
            status = str(experiment.get("status", "")).lower()
            if status == "success":
                successful_runs += 1
                if experiment.get("kernel"):
                    actual_kernels.add(str(experiment["kernel"]))
                if experiment.get("channel"):
                    actual_channels.add(str(experiment["channel"]))
                if experiment.get("corpus"):
                    actual_corpora.add(str(experiment["corpus"]))
                kernel = str(experiment.get("kernel") or "")
                channel = str(experiment.get("channel") or "")
                corpus = str(experiment.get("corpus") or "")
                for seed in experiment.get("seeds") or []:
                    seed_value = _safe_int(seed)
                    if seed_value is not None:
                        actual_seeds.add(seed_value)
                        if kernel and channel and corpus:
                            actual_cells.add((kernel, channel, corpus, seed_value))
            elif status:
                failed_runs += 1

    synthetic_success = 0
    for record in synthetic_records:
        if record.get("status") == "success":
            synthetic_success += 1
        for item in record.get("results") or []:
            kernel = item.get("kernel")
            if kernel:
                actual_synthetic_kernels.add(str(kernel))
            seed_value = _safe_int(item.get("seed"))
            if seed_value is not None:
                actual_synthetic_seeds.add(seed_value)
                if kernel:
                    actual_synthetic_cells.add((str(kernel), seed_value))
    missing = {
        "kernels": sorted(set(protocol.kernels) - actual_kernels),
        "channels": sorted(set(protocol.channels) - actual_channels),
        "corpora": sorted(set(protocol.corpora) - actual_corpora),
        "seeds": sorted(set(protocol.seeds) - actual_seeds),
    }
    missing_synthetic = {
        "kernels": sorted(set(protocol.kernels) - actual_synthetic_kernels),
        "seeds": sorted(set(protocol.seeds) - actual_synthetic_seeds),
    }
    expected_cells = {
        (kernel, channel, corpus, seed)
        for kernel in protocol.kernels
        for channel in protocol.channels
        for corpus in protocol.corpora
        for seed in protocol.seeds
    }
    observed_expected_cells = actual_cells & expected_cells
    extra_cells = sorted(actual_cells - expected_cells)
    missing_cells = sorted(expected_cells - actual_cells)
    expected_synthetic_cells = {
        (kernel, seed)
        for kernel in protocol.kernels
        for seed in protocol.seeds
    }
    observed_expected_synthetic_cells = actual_synthetic_cells & expected_synthetic_cells
    extra_synthetic_cells = sorted(actual_synthetic_cells - expected_synthetic_cells)
    missing_synthetic_cells = sorted(expected_synthetic_cells - actual_synthetic_cells)
    thesis_safe = (
        successful_runs > 0
        and synthetic_success > 0
        and failed_runs == 0
        and all(not values for values in missing.values())
        and all(not values for values in missing_synthetic.values())
        and not missing_cells
        and not missing_synthetic_cells
        and not under_protocol_limit
        and not missing_limit_run_ids
    )
    return {
        "successful_suite_experiments": successful_runs,
        "failed_suite_experiments": failed_runs,
        "successful_synthetic_runs": synthetic_success,
        "missing_protocol_coverage": missing,
        "missing_synthetic_protocol_coverage": missing_synthetic,
        "protocol_article_limit": int(protocol.limit),
        "observed_suite_limits": observed_suite_limits,
        "under_protocol_limit_count": len(under_protocol_limit),
        "under_protocol_limit_runs": under_protocol_limit,
        "missing_limit_run_ids": missing_limit_run_ids,
        "expected_protocol_cell_count": len(expected_cells),
        "observed_protocol_cell_count": len(observed_expected_cells),
        "extra_non_protocol_cell_count": len(extra_cells),
        "extra_non_protocol_cells_sample": [
            {
                "kernel": kernel,
                "channel": channel,
                "corpus": corpus,
                "seed": seed,
            }
            for kernel, channel, corpus, seed in extra_cells[:25]
        ],
        "missing_protocol_cells_count": len(missing_cells),
        "missing_protocol_cells_sample": [
            {
                "kernel": kernel,
                "channel": channel,
                "corpus": corpus,
                "seed": seed,
            }
            for kernel, channel, corpus, seed in missing_cells[:25]
        ],
        "expected_synthetic_cell_count": len(expected_synthetic_cells),
        "observed_synthetic_cell_count": len(observed_expected_synthetic_cells),
        "extra_non_protocol_synthetic_cell_count": len(extra_synthetic_cells),
        "extra_non_protocol_synthetic_cells_sample": [
            {"kernel": kernel, "seed": seed}
            for kernel, seed in extra_synthetic_cells[:25]
        ],
        "missing_synthetic_cells_count": len(missing_synthetic_cells),
        "missing_synthetic_cells_sample": [
            {"kernel": kernel, "seed": seed}
            for kernel, seed in missing_synthetic_cells[:25]
        ],
        "pass": thesis_safe,
        "thesis_safe": thesis_safe,
    }


def _select_manifests(
    runs_dir: Path,
    *,
    repo_root: Path,
    run_id_allowlist_path: Optional[Path],
    suite_manifest_paths: Optional[Sequence[Path]],
    synthetic_manifest_paths: Optional[Sequence[Path]],
) -> Tuple[List[Path], Dict[str, Any]]:
    manifests = sorted(runs_dir.rglob("experiment_manifest.json"))
    all_by_id: Dict[str, List[Path]] = {}
    for manifest_path in manifests:
        all_by_id.setdefault(_manifest_run_id(manifest_path), []).append(manifest_path.resolve())

    normalized_suite = _normalize_manifest_paths(suite_manifest_paths, repo_root=repo_root, runs_dir=runs_dir)
    normalized_synthetic = _normalize_manifest_paths(synthetic_manifest_paths, repo_root=repo_root, runs_dir=runs_dir)

    allowed_run_ids: List[str] = []
    selected: Set[Path] = set()
    missing_requested_paths: List[str] = []
    missing_requested_run_ids: List[str] = []
    focused_filter_active = bool(run_id_allowlist_path or normalized_suite or normalized_synthetic)

    if run_id_allowlist_path:
        seen_run_ids: Set[str] = set()
        for run_id in _load_line_list(run_id_allowlist_path):
            if run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            allowed_run_ids.append(run_id)
            matched = all_by_id.get(run_id)
            if matched:
                selected.update(matched)
            else:
                missing_requested_run_ids.append(run_id)

    if normalized_suite or normalized_synthetic:
        known_manifests = {manifest.resolve() for manifest in manifests}
        for manifest_path in [*normalized_suite, *normalized_synthetic]:
            resolved = manifest_path.resolve()
            if resolved in known_manifests:
                selected.add(resolved)
            else:
                missing_requested_paths.append(str(resolved))

    selected_manifests = sorted(selected) if focused_filter_active else [manifest.resolve() for manifest in manifests]
    selection_summary = {
        "focused_filter_active": focused_filter_active,
        "total_available_manifests": len(manifests),
        "selected_manifest_count": len(selected_manifests),
        "selected_run_ids": sorted({_manifest_run_id(path) for path in selected_manifests}),
        "requested_run_ids": allowed_run_ids,
        "requested_suite_manifests": [str(path) for path in normalized_suite],
        "requested_synthetic_manifests": [str(path) for path in normalized_synthetic],
        "missing_requested_run_ids": missing_requested_run_ids,
        "missing_requested_manifest_paths": missing_requested_paths,
    }
    return selected_manifests, selection_summary


def build_thesis_evidence(
    *,
    runs_dir: Path,
    methods_path: Path,
    results_path: Path,
    run_id_allowlist_path: Optional[Path] = None,
    suite_manifest_paths: Optional[Sequence[Path]] = None,
    synthetic_manifest_paths: Optional[Sequence[Path]] = None,
    control_metric_basis: str = "auto",
) -> Dict[str, Dict[str, Any]]:
    runs_dir = runs_dir.resolve()
    repo_root = _resolve_repo_root(runs_dir)
    protocol = load_canonical_protocol(methods_path)
    selected_control_metric_basis = _normalize_control_metric_basis(control_metric_basis)

    manifests, selection_summary = _select_manifests(
        runs_dir,
        repo_root=repo_root,
        run_id_allowlist_path=run_id_allowlist_path.resolve() if run_id_allowlist_path else None,
        suite_manifest_paths=suite_manifest_paths,
        synthetic_manifest_paths=synthetic_manifest_paths,
    )
    manifest_ids = [_manifest_run_id(path) for path in manifests]
    suite_manifests: List[Tuple[Path, Dict[str, Any]]] = []
    synthetic_manifests: List[Tuple[Path, Dict[str, Any]]] = []
    failure_modes: List[Dict[str, Any]] = []
    suite_leafs: List[Dict[str, Any]] = []
    synthetic_records: List[Dict[str, Any]] = []

    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        if _is_synthetic_manifest_or_recoverable(manifest_path, manifest):
            synthetic_manifest = _coerce_synthetic_manifest(manifest_path, manifest)
            synthetic_manifests.append((manifest_path, synthetic_manifest))
            synthetic_records.extend(_synthetic_records(manifest_path, synthetic_manifest, repo_root=repo_root))
            continue
        suite_manifests.append((manifest_path, manifest))
        failure_modes.extend(_suite_failure_modes(manifest_path, manifest))
        suite_leafs.extend(_suite_leaf_records(manifest_path, manifest, repo_root=repo_root))

    control_records: List[Dict[str, Any]] = []
    stochastic_variance_records: List[Dict[str, Any]] = []
    metric_signal_cartography_records: List[Dict[str, Any]] = []
    relativity_records: List[Dict[str, Any]] = []
    ablation_records: List[Dict[str, Any]] = []
    track4_records: List[Dict[str, Any]] = []
    verification_records: List[Dict[str, Any]] = []
    procrustes_verification_records: List[Dict[str, Any]] = []
    evidence_ledger: List[Dict[str, Any]] = []

    grouped_controls: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for leaf in suite_leafs:
        run_dir = Path(leaf["run_dir"])
        baseline_meta = _load_baseline_meta(run_dir)
        track5_mode = _extract_track5_mode({}, baseline_meta, leaf)
        leaf_key = (leaf["run_id"], leaf["kernel"], leaf["channel"])
        leaf_common = {
            "run_id": leaf["run_id"],
            "kernel": leaf["kernel"],
            "channel": leaf["channel"],
            "corpus": leaf["corpus"],
            "run_dir": str(run_dir),
            "track5_mode": track5_mode,
        }

        verification_path = run_dir / "verification_report.json"
        verification_blob = _load_verification(run_dir)
        verification = _evaluate_verification(verification_blob) if verification_blob else {
            "status": "MISSING",
            "global_pass": False,
            "failed_layers": None,
            "seed_stability_cv": None,
        }
        verification_safe = (
            bool(verification_blob)
            and verification.get("status") == "VERIFIED"
            and bool(verification.get("global_pass"))
            and int(verification.get("failed_layers") or 0) == 0
        )
        verification_records.append({
            **leaf_common,
            **verification,
            "pass": verification_safe,
            "thesis_safe": verification_safe,
            "verification_report_path": str(verification_path),
        })
        evidence_ledger.append(
            _ledger_record(
                leaf_common,
                claim_id="verification_provenance",
                artifact_family="verification_report.json",
                present=verification_path.exists(),
                path=verification_path,
                detail=str(verification.get("status") or "MISSING"),
            )
        )
        if not verification_safe:
            failure_modes.append({
                **leaf_common,
                "failure_type": "unverified_leaf",
                "detail": f"{verification_path} status={verification.get('status')}",
            })

        procrustes_verification = _evaluate_procrustes_verification(verification_blob) if verification_blob else {
            "status": "MISSING",
            "global_pass": False,
            "pass": False,
            "thesis_safe": False,
            "failed_layers": None,
            "seed_stability_cv": None,
            "missing_metadata": ["verification_report.json"],
            "missing_required_checks": list(PROCRUSTES_PROVENANCE_REQUIRED_CHECKS),
            "required_check_failures": [],
            "unexpected_check_failures": [],
            "ignored_check_failures": [],
            "ignored_failure_policy": list(PROCRUSTES_PROVENANCE_IGNORED_FAILURES),
        }
        procrustes_verification_safe = bool(procrustes_verification.get("thesis_safe"))
        procrustes_verification_records.append({
            **leaf_common,
            **procrustes_verification,
            "verification_report_path": str(verification_path),
        })
        evidence_ledger.append(
            _ledger_record(
                leaf_common,
                claim_id="procrustes_verification_provenance",
                artifact_family="verification_report.json",
                present=verification_path.exists(),
                path=verification_path,
                detail=str(procrustes_verification.get("status") or "MISSING"),
            )
        )
        if not procrustes_verification_safe:
            failure_modes.append({
                **leaf_common,
                "failure_type": "procrustes_profile_unverified_leaf",
                "detail": (
                    f"{verification_path} status={procrustes_verification.get('status')} "
                    f"missing_checks={procrustes_verification.get('missing_required_checks')} "
                    f"unexpected_failures={procrustes_verification.get('unexpected_check_failures')}"
                ),
            })

        control_blob = _load_control_metrics(run_dir, metric_basis=selected_control_metric_basis)
        if leaf["corpus"] == "real":
            control_path = Path(control_blob.get("_evidence_control_metric_basis_path") or run_dir / "control_metrics.json")
            cartography_rows = _metric_signal_cartography_rows(run_dir, leaf_common)
            metric_signal_cartography_records.extend(cartography_rows)
            for claim_id in (
                "control_destruction",
                "procrustes_control_separation",
                "stochastic_control_variance_compression",
            ):
                evidence_ledger.append(
                    _ledger_record(
                        leaf_common,
                        claim_id=claim_id,
                        artifact_family="control_metrics.json",
                        present=bool(control_blob),
                        path=control_path,
                    )
                )
            evidence_ledger.append(
                _ledger_record(
                    leaf_common,
                    claim_id="stochastic_control_variance_separation",
                    artifact_family="control_metrics*.json",
                    present=bool(cartography_rows),
                    path=control_path,
                )
            )
            if not control_blob:
                failure_modes.append({**leaf_common, "failure_type": "missing_control_metrics", "detail": str(control_path)})
        if control_blob:
            record = {**leaf_common, **_evaluate_controls(control_blob)}
            control_records.append(record)
            stochastic_variance_records.append({
                **leaf_common,
                **_evaluate_stochastic_variance_controls(control_blob),
            })
            if leaf["corpus"] == "real":
                grouped_controls[leaf_key] = record

        if leaf["corpus"] == "real":
            relativity_present = False
            relativity_path = run_dir / "relativity_deltas.json"
            relativity_blob = _load_relativity(run_dir)
            if relativity_blob:
                relativity_present = True
                relativity_records.append({**leaf_common, **_evaluate_relativity(relativity_blob)})
            else:
                relativity_summary = summarize_observer_relativity(run_dir)
                if relativity_summary.get("status") != "NO_DATA":
                    relativity_present = True
                    relativity_records.append({**leaf_common, **_evaluate_relativity_summary(relativity_summary)})
            evidence_ledger.append(
                _ledger_record(
                    leaf_common,
                    claim_id="observer_relativity",
                    artifact_family="relativity_deltas.json or relativity_cache",
                    present=relativity_present,
                    path=relativity_path,
                )
            )
            if not relativity_present:
                failure_modes.append({**leaf_common, "failure_type": "missing_observer_relativity", "detail": str(relativity_path)})

        ablation_blob = _load_ablation_summary(run_dir)
        if ablation_blob and leaf["corpus"] == "real":
            ablation_records.append({**leaf_common, **_evaluate_ablation(ablation_blob, track5_mode)})
        if leaf["corpus"] == "real":
            ablation_path = run_dir / "ablation_summary.json"
            evidence_ledger.append(
                _ledger_record(
                    leaf_common,
                    claim_id="track5_ablation_coverage",
                    artifact_family="ablation_summary.json",
                    present=bool(ablation_blob),
                    path=ablation_path,
                )
            )
            if not ablation_blob:
                failure_modes.append({**leaf_common, "failure_type": "missing_track5_ablation_summary", "detail": str(ablation_path)})

        view_state = _load_view_state(run_dir)
        rows = _load_monolith_rows(run_dir)
        walker_states = _load_walker_states(run_dir)
        track4_summary = summarize_track4_traversal(run_dir)
        track4_present = False
        if (
            track4_summary.get("status") not in {"NO_DATA", "LEGACY_FALLBACK"}
            or view_state or rows or walker_states
        ):
            track4_present = True
            if track4_summary.get("status") not in {"NO_DATA", "LEGACY_FALLBACK"}:
                track4_eval = _evaluate_track4_summary(track4_summary)
            else:
                track4_eval = _evaluate_track4(view_state, rows, walker_states)
            track4_records.append({
                **leaf_common,
                **track4_eval,
                "verification_status": verification.get("status"),
                "xy_collapsed": _xy_collapse(rows),
            })
            if track4_eval["survival_rate"] is not None and track4_eval["survival_rate"] <= 0.0:
                failure_modes.append({**leaf_common, "failure_type": "dead_paths", "detail": "walker_survival_rate <= 0"})
            if track4_eval["zone_count"] < 3:
                failure_modes.append({**leaf_common, "failure_type": "insufficient_zone_support", "detail": f"zone_count={track4_eval['zone_count']}"})
            if _xy_collapse(rows):
                failure_modes.append({**leaf_common, "failure_type": "collapsed_manifold", "detail": "x/y spread below collapse threshold"})
        for claim_id in ("track4_traversal_validity", "track4_work_barrier_signal"):
            evidence_ledger.append(
                _ledger_record(
                    leaf_common,
                    claim_id=claim_id,
                    artifact_family="track4_traversal_summary.json or walker/view artifacts",
                    present=track4_present,
                    path=run_dir / "track4_traversal_summary.json",
                )
            )
        if not track4_present:
            failure_modes.append({**leaf_common, "failure_type": "missing_track4_traversal", "detail": str(run_dir)})

        if leaf["corpus"] == "real" and (run_dir / "relativity_deltas.json").exists():
            rel_eval = relativity_records[-1] if relativity_records else None
            if rel_eval:
                if (rel_eval.get("max_rotation_deg") or 0.0) < RELATIVITY_ROTATION_MIN_DEG:
                    failure_modes.append({**leaf_common, "failure_type": "zero_rotation_observer", "detail": f"max_rotation_deg={rel_eval.get('max_rotation_deg')}"})
                if (rel_eval.get("max_path_flip_count") or 0) <= 0:
                    failure_modes.append({**leaf_common, "failure_type": "zero_path_flip_observer", "detail": f"max_path_flip_count={rel_eval.get('max_path_flip_count')}"})

        if leaf["corpus"] == "real" and (run_dir / "ablation_summary.json").exists():
            abl_eval = ablation_records[-1] if ablation_records else None
            if abl_eval and not abl_eval["thesis_safe"]:
                failure_modes.append({**leaf_common, "failure_type": "placeholder_ablation", "detail": abl_eval.get("reason")})

    synthetic_summary = _aggregate_synthetic(synthetic_records)
    canonical_freeze = _evaluate_canonical_freeze(protocol, suite_manifests, synthetic_records)
    if selection_summary["focused_filter_active"]:
        narrative_audit = _focused_narrative_audit(results_path, manifest_ids)
    else:
        narrative_audit = _narrative_audit(results_path, manifest_ids)

    control_safe = [row for row in control_records if row["corpus"] == "real"]
    procrustes_control_safe = [
        row for row in control_safe
        if row.get("status") == "OK"
        and not row.get("synthetic_placeholder")
        and row.get("procrustes_ratio") is not None
    ]
    procrustes_control_pass = [
        row for row in procrustes_control_safe
        if (
            row.get("procrustes_min_control_ratio")
            if row.get("procrustes_min_control_ratio") is not None
            else row.get("procrustes_ratio")
        ) is not None
        and (
            row.get("procrustes_min_control_ratio")
            if row.get("procrustes_min_control_ratio") is not None
            else row.get("procrustes_ratio")
        ) >= CONTROL_PROCRUSTES_RATIO_MIN
    ]
    stochastic_variance_safe = [
        row for row in stochastic_variance_records if row["corpus"] == "real"
    ]
    control_metric_provenance_records = [
        {
            "run_id": row["run_id"],
            "kernel": row["kernel"],
            "channel": row["channel"],
            "run_dir": row["run_dir"],
            "metric_basis": row.get("metric_basis"),
            "requested_metric_basis": row.get("requested_metric_basis"),
            "evidence_requested_metric_basis": row.get("evidence_requested_metric_basis"),
            "evidence_control_metric_basis_path": row.get("evidence_control_metric_basis_path"),
            "primary_metric_source": row.get("primary_metric_source"),
            "primary_simple_variance_stochastic_ratio": row.get("simple_variance_stochastic_ratio"),
            "alternate_comprehensive_simple_variance_stochastic_ratio": row.get(
                "alternate_comprehensive_simple_variance_stochastic_ratio"
            ),
            "alternate_direct_simple_variance_stochastic_ratio": row.get(
                "alternate_direct_simple_variance_stochastic_ratio"
            ),
        }
        for row in stochastic_variance_safe
    ]
    control_metric_provenance = {
        "records": control_metric_provenance_records,
        "basis_mismatch_count": sum(
            1
            for row in control_metric_provenance_records
            if (
                row.get("alternate_comprehensive_simple_variance_stochastic_ratio") is not None
                or row.get("alternate_direct_simple_variance_stochastic_ratio") is not None
            )
            and row.get("primary_simple_variance_stochastic_ratio") is not None
            and (
                (row["primary_simple_variance_stochastic_ratio"] <= CONTROL_STOCHASTIC_VARIANCE_RATIO_MAX)
                != (
                    (
                        row.get("alternate_comprehensive_simple_variance_stochastic_ratio")
                        if row.get("alternate_comprehensive_simple_variance_stochastic_ratio") is not None
                        else row.get("alternate_direct_simple_variance_stochastic_ratio")
                    )
                    <= CONTROL_STOCHASTIC_VARIANCE_RATIO_MAX
                )
            )
        ),
    }
    metric_signal_cartography = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": metric_signal_cartography_records,
        "metrics": list(CONTROL_SIGNAL_CARTOGRAPHY_METRICS),
        "bases": list(CONTROL_SIGNAL_CARTOGRAPHY_BASES),
        "control_families": list(CONTROL_SIGNAL_CARTOGRAPHY_CONTROLS) + ["stochastic_controls"],
        "threshold_abs_log_ratio_min": CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN,
        "interpretation": (
            "Rows are direction-free metric displacement probes. The direction field records whether real "
            "is greater than, lower than, or near parity with each matched control."
        ),
    }
    variance_separation_summary = _summarize_variance_separation(
        metric_signal_cartography_records
    )
    seed_dispersion = _mean([row.get("real_procrustes_std") or 0.0 for row in control_safe])
    kernel_pass_map: Dict[str, int] = {}
    kernel_total_map: Dict[str, int] = {}
    for row in control_safe:
        kernel = str(row["kernel"])
        kernel_total_map[kernel] = kernel_total_map.get(kernel, 0) + 1
        if row["thesis_safe"]:
            kernel_pass_map[kernel] = kernel_pass_map.get(kernel, 0) + 1
    kernel_robustness = {
        "per_kernel_pass_rate": {
            kernel: (kernel_pass_map.get(kernel, 0) / float(total))
            for kernel, total in kernel_total_map.items()
        },
        "pass": bool(kernel_total_map) and all(kernel_pass_map.get(kernel, 0) > 0 for kernel in kernel_total_map),
        "thesis_safe": bool(kernel_total_map) and all(kernel_pass_map.get(kernel, 0) > 0 for kernel in kernel_total_map),
    }

    observer_relativity_summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": relativity_records,
        "aggregate": {
            "n_runs": len(relativity_records),
            "mean_coord_delta": _mean([row.get("mean_coord_delta") or 0.0 for row in relativity_records]),
            "max_rotation_deg": max([row.get("max_rotation_deg") or 0.0 for row in relativity_records], default=None),
            "pass_rate": (
                sum(1 for row in relativity_records if row["thesis_safe"]) / float(len(relativity_records))
                if relativity_records else 0.0
            ),
        },
    }

    controls_by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in track4_records:
        key = (row["run_id"], row["kernel"], row["channel"])
        controls_by_group.setdefault(key, []).append(row)
    traversal_pairs: List[Dict[str, Any]] = []
    for key, rows in controls_by_group.items():
        real_row = next((row for row in rows if row["corpus"] == "real"), None)
        if not real_row:
            continue
        control_rows = [row for row in rows if row["corpus"] != "real"]
        real_survival = real_row.get("survival_rate")
        real_work = real_row.get("mean_work")
        control_survival_values = [
            (row["corpus"], row.get("survival_rate"))
            for row in control_rows
            if row.get("survival_rate") is not None
        ]
        control_work_values = [
            (row["corpus"], row.get("mean_work"))
            for row in control_rows
            if row.get("mean_work") is not None
        ]
        control_survival = _mean([value for _, value in control_survival_values])
        control_work = _mean([value for _, value in control_work_values])
        min_survival_gap = (
            min((real_survival - value) for _, value in control_survival_values)
            if real_survival is not None and control_survival_values else None
        )
        min_work_gap = (
            min((real_work - value) for _, value in control_work_values)
            if real_work is not None and control_work_values else None
        )
        traversal_pairs.append({
            "run_id": key[0],
            "kernel": key[1],
            "channel": key[2],
            "real_survival_rate": real_survival,
            "control_survival_rate_mean": control_survival,
            "real_mean_work": real_work,
            "control_mean_work": control_work,
            "d_survival_rate": (
                (real_survival - control_survival)
                if real_survival is not None and control_survival is not None else None
            ),
            "d_mean_work": (
                (real_work - control_work)
                if real_work is not None and control_work is not None else None
            ),
            "d_survival_rate_min_vs_control": min_survival_gap,
            "d_mean_work_min_vs_control": min_work_gap,
            "failing_control_corpora": sorted({
                corpus
                for corpus, value in control_survival_values
                if real_survival is not None
                and (real_survival - value) < TRACK4_CONTROL_SURVIVAL_GAP_MIN
            } | {
                corpus
                for corpus, value in control_work_values
                if real_work is not None
                and (real_work - value) < TRACK4_CONTROL_WORK_GAP_MIN
            }),
        })
    traversal_pair_safe = [
        pair for pair in traversal_pairs
        if pair["d_survival_rate"] is not None and pair["d_mean_work"] is not None
    ]
    traversal_work_pair_safe = [
        pair for pair in traversal_pairs
        if pair["d_mean_work"] is not None and pair["d_mean_work_min_vs_control"] is not None
    ]
    track4_real_work_safe = [
        row for row in track4_records
        if row["corpus"] == "real"
        and row.get("mean_work") is not None
        and (row.get("mean_work") or 0.0) >= TRACK4_WORK_MIN
    ]

    track4_traversal_summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": track4_records,
        "real_vs_controls": traversal_pairs,
        "aggregate": {
            "n_runs": len(track4_records),
            "mean_survival_rate": _mean([row.get("survival_rate") or 0.0 for row in track4_records]),
            "mean_work": _mean([row.get("mean_work") or 0.0 for row in track4_records]),
            "terrain_valid_pass_rate": (
                sum(1 for row in track4_records if row["thesis_safe"]) / float(len(track4_records))
                if track4_records else 0.0
            ),
            "mean_d_survival_rate_real_vs_controls": _mean([pair.get("d_survival_rate") or 0.0 for pair in traversal_pair_safe]),
            "mean_d_work_real_vs_controls": _mean([pair.get("d_mean_work") or 0.0 for pair in traversal_pair_safe]),
            "mean_min_d_survival_rate_real_vs_each_control": _mean([
                pair.get("d_survival_rate_min_vs_control") for pair in traversal_pair_safe
            ]),
            "mean_min_d_work_real_vs_each_control": _mean([
                pair.get("d_mean_work_min_vs_control") for pair in traversal_work_pair_safe
            ]),
            "real_control_gap_pass_rate": (
                sum(
                    1
                    for pair in traversal_pair_safe
                    if (pair.get("d_survival_rate_min_vs_control") or 0.0) >= TRACK4_CONTROL_SURVIVAL_GAP_MIN
                    and (pair.get("d_mean_work_min_vs_control") or 0.0) >= TRACK4_CONTROL_WORK_GAP_MIN
                ) / float(len(traversal_pair_safe))
                if traversal_pair_safe else 0.0
            ),
        },
    }

    ablation_modes: Dict[str, List[Dict[str, Any]]] = {}
    for row in ablation_records:
        ablation_modes.setdefault(str(row["track5_mode"]), []).append(row)
    ablation_required_modes_present = all(
        mode in ablation_modes and any(bool(row["thesis_safe"]) for row in ablation_modes[mode])
        for mode in REQUIRED_TRACK5_ABLATION_MODES
    )
    ablation_matrix = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": ablation_records,
        "required_modes": list(REQUIRED_TRACK5_ABLATION_MODES),
        "required_modes_present": ablation_required_modes_present,
        "by_track5_mode": {
            mode: {
                "n_runs": len(rows),
                "pass_rate": sum(1 for row in rows if row["thesis_safe"]) / float(len(rows)),
                "mean_stage_2_alignment_score": _mean([
                    row.get("stage_2_alignment_score") or 0.0 for row in rows
                ]),
                "mean_stage_3_survival_rate": _mean([
                    row.get("stage_3_survival_rate") or 0.0 for row in rows
                ]),
            }
            for mode, rows in ablation_modes.items()
        },
    }
    missing_evidence_by_claim = {
        claim_id: _missing_ledger_entries(evidence_ledger, claim_id)
        for claim_id in (
            "control_destruction",
            "procrustes_control_separation",
            "stochastic_control_variance_compression",
            "stochastic_control_variance_separation",
            "observer_relativity",
            "track4_traversal_validity",
            "track4_work_barrier_signal",
            "track5_ablation_coverage",
            "verification_provenance",
            "procrustes_verification_provenance",
        )
    }

    claim_matrix_claims = [
        {
            "claim_id": "control_destruction",
            "description": "Real corpus separates from constant/shuffled/random controls on reviewer-facing metrics.",
            "artifact_family": "control_metrics.json",
            "point_estimate": _mean([row.get("procrustes_ratio") or 0.0 for row in control_safe]),
            "dispersion_across_seeds": seed_dispersion,
            "dispersion_across_kernels": _std(list(kernel_robustness["per_kernel_pass_rate"].values())),
            "effect_direction": "real_gt_controls",
            "pass": (
                bool(control_safe)
                and not missing_evidence_by_claim["control_destruction"]
                and all(row["thesis_safe"] for row in control_safe)
            ),
            "thesis_safe": (
                bool(control_safe)
                and not missing_evidence_by_claim["control_destruction"]
                and all(row["thesis_safe"] for row in control_safe)
            ),
            "supporting_runs": [row["run_dir"] for row in control_safe],
            "missing_evidence": missing_evidence_by_claim["control_destruction"],
        },
        {
            "claim_id": "procrustes_control_separation",
            "description": "Real corpus separates from matched controls on Procrustes observer-geometry residuals alone.",
            "artifact_family": "control_metrics.json",
            "point_estimate": _mean([
                row.get("procrustes_ratio")
                for row in procrustes_control_safe
            ]),
            "dispersion_across_seeds": seed_dispersion,
            "dispersion_across_kernels": _std([
                row.get("procrustes_ratio")
                for row in procrustes_control_safe
            ]),
            "effect_direction": "real_procrustes_gt_controls",
            "pass": (
                bool(procrustes_control_safe)
                and not missing_evidence_by_claim["procrustes_control_separation"]
                and len(procrustes_control_pass) == len(procrustes_control_safe)
            ),
            "thesis_safe": (
                bool(procrustes_control_safe)
                and not missing_evidence_by_claim["procrustes_control_separation"]
                and len(procrustes_control_pass) == len(procrustes_control_safe)
            ),
            "supporting_runs": [row["run_dir"] for row in procrustes_control_safe],
            "missing_evidence": missing_evidence_by_claim["procrustes_control_separation"],
        },
        {
            "claim_id": "stochastic_control_variance_compression",
            "description": (
                "Retired directional hypothesis: real observer geometry has lower simple variance than "
                "shuffled/random controls. The direction-free replacement claim is "
                "stochastic_control_variance_separation."
            ),
            "artifact_family": "control_metrics.json",
            "point_estimate": _mean([
                row.get("simple_variance_ratio")
                for row in stochastic_variance_safe
            ]),
            "dispersion_across_seeds": _std([
                row.get("simple_variance_ratio")
                for row in stochastic_variance_safe
            ]),
            "dispersion_across_kernels": _std([
                row.get("simple_variance_ratio")
                for row in stochastic_variance_safe
            ]),
            "effect_direction": "real_lt_shuffled_random_controls",
            "pass": (
                bool(stochastic_variance_safe)
                and not missing_evidence_by_claim["stochastic_control_variance_compression"]
                and all(row["thesis_safe"] for row in stochastic_variance_safe)
            ),
            "thesis_safe": (
                bool(stochastic_variance_safe)
                and not missing_evidence_by_claim["stochastic_control_variance_compression"]
                and all(row["thesis_safe"] for row in stochastic_variance_safe)
            ),
            "supporting_runs": [row["run_dir"] for row in stochastic_variance_safe],
            "missing_evidence": missing_evidence_by_claim["stochastic_control_variance_compression"],
            "status": "retired_directional_hypothesis",
            "replacement_claim_id": "stochastic_control_variance_separation",
        },
        {
            "claim_id": "stochastic_control_variance_separation",
            "description": (
                "Integrated Track 5 geometry shows direction-free simple-variance displacement from "
                "matched stochastic controls, with direct observer payload reported separately as sensitivity evidence."
            ),
            "artifact_family": "metric_signal_cartography.json + variance_separation_summary.json",
            "point_estimate": variance_separation_summary.get("mean_primary_abs_log_ratio"),
            "dispersion_across_seeds": None,
            "dispersion_across_kernels": _std([
                row.get("abs_log_ratio")
                for row in variance_separation_summary.get("records", [])
            ]),
            "effect_direction": variance_separation_summary.get("effect_direction"),
            "pass": (
                bool(variance_separation_summary.get("pass"))
                and not missing_evidence_by_claim["stochastic_control_variance_separation"]
            ),
            "thesis_safe": (
                bool(variance_separation_summary.get("thesis_safe"))
                and not missing_evidence_by_claim["stochastic_control_variance_separation"]
            ),
            "supporting_runs": [
                row["run_dir"]
                for row in variance_separation_summary.get("records", [])
                if row.get("passes_separation_threshold")
            ],
            "missing_evidence": missing_evidence_by_claim["stochastic_control_variance_separation"],
            "primary_basis": variance_separation_summary.get("primary_basis"),
            "sensitivity_basis": variance_separation_summary.get("sensitivity_basis"),
        },
        {
            "claim_id": "synthetic_recoverability",
            "description": "Synthetic planted structure is recoverable with aggregate mean/std evidence.",
            "artifact_family": "synthetic_summary.json",
            "point_estimate": synthetic_summary.get("mean_nmi"),
            "dispersion_across_seeds": _std(list((synthetic_summary.get("per_seed_mean_nmi") or {}).values())),
            "dispersion_across_kernels": _std(list((synthetic_summary.get("per_kernel_mean_nmi") or {}).values())),
            "effect_direction": synthetic_summary.get("effect_direction"),
            "pass": synthetic_summary["pass"],
            "thesis_safe": synthetic_summary["thesis_safe"],
            "supporting_runs": [record["run_id"] for record in synthetic_records],
        },
        {
            "claim_id": "observer_relativity",
            "description": "Observer-conditioned manifolds induce non-placeholder displacement, rotation, and path flips.",
            "artifact_family": "relativity_deltas.json",
            "point_estimate": observer_relativity_summary["aggregate"]["mean_coord_delta"],
            "dispersion_across_seeds": None,
            "dispersion_across_kernels": None,
            "effect_direction": "observer_conditioning_nontrivial",
            "pass": (
                bool(relativity_records)
                and not missing_evidence_by_claim["observer_relativity"]
                and all(row["thesis_safe"] for row in relativity_records)
            ),
            "thesis_safe": (
                bool(relativity_records)
                and not missing_evidence_by_claim["observer_relativity"]
                and all(row["thesis_safe"] for row in relativity_records)
            ),
            "supporting_runs": [row["run_dir"] for row in relativity_records],
            "missing_evidence": missing_evidence_by_claim["observer_relativity"],
        },
        {
            "claim_id": "track4_traversal_validity",
            "description": "Track 4 traversal exhibits non-degenerate work/survival and terrain-semantic structure.",
            "artifact_family": "MONOLITH.view_state.json + walker_states.json + MONOLITH_DATA.csv",
            "point_estimate": track4_traversal_summary["aggregate"]["mean_survival_rate"],
            "dispersion_across_seeds": None,
            "dispersion_across_kernels": None,
            "effect_direction": "terrain_correlates_with_traversal",
            "pass": (
                bool(track4_records)
                and not missing_evidence_by_claim["track4_traversal_validity"]
                and all(row["thesis_safe"] for row in track4_records if row["corpus"] == "real")
                and bool(traversal_pair_safe)
                and all(
                    (pair.get("d_survival_rate_min_vs_control") or 0.0) >= TRACK4_CONTROL_SURVIVAL_GAP_MIN
                    and (pair.get("d_mean_work_min_vs_control") or 0.0) >= TRACK4_CONTROL_WORK_GAP_MIN
                    for pair in traversal_pair_safe
                )
            ),
            "thesis_safe": (
                bool(track4_records)
                and not missing_evidence_by_claim["track4_traversal_validity"]
                and all(row["thesis_safe"] for row in track4_records if row["corpus"] == "real")
                and bool(traversal_pair_safe)
                and all(
                    (pair.get("d_survival_rate_min_vs_control") or 0.0) >= TRACK4_CONTROL_SURVIVAL_GAP_MIN
                    and (pair.get("d_mean_work_min_vs_control") or 0.0) >= TRACK4_CONTROL_WORK_GAP_MIN
                    for pair in traversal_pair_safe
                )
            ),
            "supporting_runs": [row["run_dir"] for row in track4_records if row["corpus"] == "real"],
            "missing_evidence": missing_evidence_by_claim["track4_traversal_validity"],
        },
        {
            "claim_id": "track4_work_barrier_signal",
            "description": "Track 4 work integrals separate real traversal from controls even when survival-rate gaps do not.",
            "artifact_family": "walker_states.json + track4_traversal_summary.json",
            "point_estimate": _mean([
                pair.get("d_mean_work")
                for pair in traversal_work_pair_safe
            ]),
            "dispersion_across_seeds": None,
            "dispersion_across_kernels": _std([
                pair.get("d_mean_work")
                for pair in traversal_work_pair_safe
            ]),
            "effect_direction": "real_work_gt_controls",
            "pass": (
                bool(track4_real_work_safe)
                and not missing_evidence_by_claim["track4_work_barrier_signal"]
                and bool(traversal_work_pair_safe)
                and all(
                    (pair.get("d_mean_work_min_vs_control") or 0.0) >= TRACK4_CONTROL_WORK_GAP_MIN
                    for pair in traversal_work_pair_safe
                )
            ),
            "thesis_safe": (
                bool(track4_real_work_safe)
                and not missing_evidence_by_claim["track4_work_barrier_signal"]
                and bool(traversal_work_pair_safe)
                and all(
                    (pair.get("d_mean_work_min_vs_control") or 0.0) >= TRACK4_CONTROL_WORK_GAP_MIN
                    for pair in traversal_work_pair_safe
                )
            ),
            "supporting_runs": [row["run_dir"] for row in track4_real_work_safe],
            "missing_evidence": missing_evidence_by_claim["track4_work_barrier_signal"],
        },
        {
            "claim_id": "track5_ablation_coverage",
            "description": "Track 5 ablation outputs exist and are non-placeholder for cited branch comparisons.",
            "artifact_family": "ablation_summary.json",
            "point_estimate": _mean([row.get("stage_2_alignment_score") or 0.0 for row in ablation_records]),
            "dispersion_across_seeds": None,
            "dispersion_across_kernels": None,
            "effect_direction": "branch_comparison_backed",
            "pass": (
                bool(ablation_records)
                and not missing_evidence_by_claim["track5_ablation_coverage"]
                and all(row["thesis_safe"] for row in ablation_records)
                and ablation_required_modes_present
            ),
            "thesis_safe": (
                bool(ablation_records)
                and not missing_evidence_by_claim["track5_ablation_coverage"]
                and all(row["thesis_safe"] for row in ablation_records)
                and ablation_required_modes_present
            ),
            "supporting_runs": [row["run_dir"] for row in ablation_records],
            "missing_evidence": missing_evidence_by_claim["track5_ablation_coverage"],
        },
        {
            "claim_id": "procrustes_verification_provenance",
            "description": (
                "Selected Procrustes-control leaves have intact provenance, CRN-lock, and seed-stability checks; "
                "legacy full-system ordering verdicts are reported but not used for this narrower claim profile."
            ),
            "artifact_family": "verification_report.json",
            "point_estimate": (
                sum(1 for row in procrustes_verification_records if row["thesis_safe"]) / float(len(procrustes_verification_records))
                if procrustes_verification_records else 0.0
            ),
            "dispersion_across_seeds": _std([
                row.get("seed_stability_cv")
                for row in procrustes_verification_records
                if row.get("seed_stability_cv") is not None
            ]),
            "dispersion_across_kernels": None,
            "effect_direction": "profile_provenance_checks_pass",
            "pass": (
                bool(procrustes_verification_records)
                and not missing_evidence_by_claim["procrustes_verification_provenance"]
                and all(row["thesis_safe"] for row in procrustes_verification_records)
            ),
            "thesis_safe": (
                bool(procrustes_verification_records)
                and not missing_evidence_by_claim["procrustes_verification_provenance"]
                and all(row["thesis_safe"] for row in procrustes_verification_records)
            ),
            "supporting_runs": [row["run_dir"] for row in procrustes_verification_records if row["thesis_safe"]],
            "missing_evidence": missing_evidence_by_claim["procrustes_verification_provenance"],
            "ignored_failure_policy": list(PROCRUSTES_PROVENANCE_IGNORED_FAILURES),
        },
        {
            "claim_id": "verification_provenance",
            "description": "All selected claim-bearing leaves have verified provenance reports and no failed verification layers.",
            "artifact_family": "verification_report.json",
            "point_estimate": (
                sum(1 for row in verification_records if row["thesis_safe"]) / float(len(verification_records))
                if verification_records else 0.0
            ),
            "dispersion_across_seeds": _std([
                row.get("seed_stability_cv")
                for row in verification_records
                if row.get("seed_stability_cv") is not None
            ]),
            "dispersion_across_kernels": None,
            "effect_direction": "verified_artifacts_only",
            "pass": bool(verification_records) and all(row["thesis_safe"] for row in verification_records),
            "thesis_safe": bool(verification_records) and all(row["thesis_safe"] for row in verification_records),
            "supporting_runs": [row["run_dir"] for row in verification_records if row["thesis_safe"]],
            "missing_evidence": missing_evidence_by_claim["verification_provenance"],
        },
        {
            "claim_id": "canonical_freeze",
            "description": "Canonical methods protocol is frozen and fully covered by current evidence runs.",
            "artifact_family": "METHODS.md + experiment_manifest.json + RESULTS.md",
            "point_estimate": canonical_freeze["successful_suite_experiments"],
            "dispersion_across_seeds": None,
            "dispersion_across_kernels": None,
            "effect_direction": "protocol_coverage_complete",
            "pass": canonical_freeze["pass"] and narrative_audit["pass"],
            "thesis_safe": canonical_freeze["thesis_safe"] and narrative_audit["thesis_safe"],
            "supporting_runs": manifest_ids,
        },
    ]
    claims_by_id = {str(claim["claim_id"]): claim for claim in claim_matrix_claims}
    positive_signal_claim_ids = [
        claim_id for claim_id in (
            "procrustes_control_separation",
            "stochastic_control_variance_separation",
            "procrustes_verification_provenance",
            "synthetic_recoverability",
            "observer_relativity",
            "track4_work_barrier_signal",
            "track5_ablation_coverage",
        )
        if bool((claims_by_id.get(claim_id) or {}).get("thesis_safe"))
    ]
    real_control_blocker_ids = [
        claim_id for claim_id in (
            "control_destruction",
            "stochastic_control_variance_separation",
            "track4_traversal_validity",
            "verification_provenance",
            "procrustes_verification_provenance",
            "canonical_freeze",
        )
        if not bool((claims_by_id.get(claim_id) or {}).get("thesis_safe"))
    ]
    publishable_real_control_signal = not real_control_blocker_ids
    semantic_signal_detected = bool(positive_signal_claim_ids) or publishable_real_control_signal
    if publishable_real_control_signal:
        semantic_signal_status = "real_control_supported"
    elif positive_signal_claim_ids:
        semantic_signal_status = "mixed_positive_not_real_control_safe"
    else:
        semantic_signal_status = "no_current_signal_support"
    semantic_signal_interpretation = {
        "status": semantic_signal_status,
        "semantic_signal_detected": semantic_signal_detected,
        "publishable_real_control_signal": publishable_real_control_signal,
        "positive_evidence_claim_ids": positive_signal_claim_ids,
        "real_control_blocker_claim_ids": real_control_blocker_ids,
        "interpretation": (
            "Current evidence does not imply absence of semantic signal; positive synthetic, "
            "observer, or Track 5 evidence is present, but real-vs-control claims remain blocked."
            if semantic_signal_status == "mixed_positive_not_real_control_safe"
            else "Current evidence clears the real-vs-control publication gate."
            if semantic_signal_status == "real_control_supported"
            else "Current evidence does not yet support a semantic-signal claim."
        ),
    }

    scientific_validation_summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs_dir),
        "methods_path": str(methods_path),
        "results_path": str(results_path),
        "input_selection": selection_summary,
        "selected_control_metric_basis": selected_control_metric_basis,
        "canonical_protocol": protocol.to_dict(),
        "control_destruction": {
            "records": control_safe,
            "pass_rate": (
                sum(1 for row in control_safe if row["thesis_safe"]) / float(len(control_safe))
                if control_safe else 0.0
            ),
            "seed_dispersion": seed_dispersion,
            "kernel_pass_rate": kernel_robustness["per_kernel_pass_rate"],
        },
        "procrustes_control_separation": {
            "records": procrustes_control_safe,
            "pass_rate": (
                len(procrustes_control_pass) / float(len(procrustes_control_safe))
                if procrustes_control_safe else 0.0
            ),
            "threshold_procrustes_ratio_min": CONTROL_PROCRUSTES_RATIO_MIN,
            "mean_procrustes_ratio": _mean([
                row.get("procrustes_ratio")
                for row in procrustes_control_safe
            ]),
            "mean_min_per_control_ratio": _mean([
                row.get("procrustes_min_control_ratio")
                for row in procrustes_control_safe
            ]),
            "per_control_rows": [
                {
                    "run_id": row["run_id"],
                    "kernel": row["kernel"],
                    "channel": row["channel"],
                    **per_control,
                }
                for row in procrustes_control_safe
                for per_control in (row.get("procrustes_per_control") or [])
            ],
        },
        "control_metric_provenance": control_metric_provenance,
        "metric_signal_cartography": {
            "records": metric_signal_cartography_records,
            "n_records": len(metric_signal_cartography_records),
            "threshold_abs_log_ratio_min": CONTROL_VARIANCE_SEPARATION_ABS_LOG_MIN,
        },
        "stochastic_control_variance_compression": {
            "status": "retired_directional_hypothesis",
            "replacement_claim_id": "stochastic_control_variance_separation",
            "records": stochastic_variance_safe,
            "pass_rate": (
                sum(1 for row in stochastic_variance_safe if row["thesis_safe"]) / float(len(stochastic_variance_safe))
                if stochastic_variance_safe else 0.0
            ),
            "mean_real_over_stochastic_variance_ratio": _mean([
                row.get("simple_variance_ratio")
                for row in stochastic_variance_safe
            ]),
            "threshold_real_over_stochastic_variance_ratio_max": CONTROL_STOCHASTIC_VARIANCE_RATIO_MAX,
        },
        "stochastic_control_variance_separation": variance_separation_summary,
        "synthetic_recoverability": synthetic_summary,
        "seed_robustness": {
            "mean_real_procrustes_std": seed_dispersion,
            "pass": seed_dispersion is not None and seed_dispersion <= 0.35,
            "thesis_safe": seed_dispersion is not None and seed_dispersion <= 0.35,
        },
        "kernel_robustness": kernel_robustness,
        "semantic_signal_interpretation": semantic_signal_interpretation,
        "observer_relativity": observer_relativity_summary["aggregate"],
        "track4_traversal": track4_traversal_summary["aggregate"],
        "track4_work_barrier_signal": {
            "records": traversal_work_pair_safe,
            "pass_rate": (
                sum(
                    1
                    for pair in traversal_work_pair_safe
                    if (pair.get("d_mean_work") or 0.0) >= TRACK4_CONTROL_WORK_GAP_MIN
                ) / float(len(traversal_work_pair_safe))
                if traversal_work_pair_safe else 0.0
            ),
            "mean_d_work_real_vs_controls": _mean([
                pair.get("d_mean_work")
                for pair in traversal_work_pair_safe
            ]),
            "threshold_d_work_min": TRACK4_CONTROL_WORK_GAP_MIN,
        },
        "track5_mode_comparison": ablation_matrix["by_track5_mode"],
        "verification_provenance": {
            "records": verification_records,
            "pass_rate": (
                sum(1 for row in verification_records if row["thesis_safe"]) / float(len(verification_records))
                if verification_records else 0.0
            ),
            "unverified_count": sum(1 for row in verification_records if not row["thesis_safe"]),
        },
        "procrustes_verification_provenance": {
            "records": procrustes_verification_records,
            "pass_rate": (
                sum(1 for row in procrustes_verification_records if row["thesis_safe"]) / float(len(procrustes_verification_records))
                if procrustes_verification_records else 0.0
            ),
            "unverified_count": sum(1 for row in procrustes_verification_records if not row["thesis_safe"]),
            "ignored_failure_policy": list(PROCRUSTES_PROVENANCE_IGNORED_FAILURES),
        },
        "canonical_freeze": canonical_freeze,
        "narrative_audit": narrative_audit,
        "evidence_ledger": {
            "records": evidence_ledger,
            "missing_count": sum(1 for row in evidence_ledger if not bool(row.get("present"))),
            "missing_by_claim": {
                claim_id: len(rows)
                for claim_id, rows in missing_evidence_by_claim.items()
            },
        },
        "failure_modes": {
            "count": len(failure_modes),
            "records": failure_modes,
        },
    }

    claim_matrix = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims": claim_matrix_claims,
    }
    unsafe_claim_strategy = _build_claim_strategy(
        claims_by_id=claims_by_id,
        scientific_validation_summary=scientific_validation_summary,
        canonical_freeze=canonical_freeze,
        narrative_audit=narrative_audit,
    )

    return {
        "scientific_validation_summary": scientific_validation_summary,
        "claim_matrix": claim_matrix,
        "ablation_matrix": ablation_matrix,
        "observer_relativity_summary": observer_relativity_summary,
        "track4_traversal_summary": track4_traversal_summary,
        "metric_signal_cartography": metric_signal_cartography,
        "variance_separation_summary": variance_separation_summary,
        "unsafe_claim_strategy": unsafe_claim_strategy,
    }


def write_thesis_evidence(
    *,
    runs_dir: Path,
    methods_path: Path,
    results_path: Path,
    out_dir: Path,
    run_id_allowlist_path: Optional[Path] = None,
    suite_manifest_paths: Optional[Sequence[Path]] = None,
    synthetic_manifest_paths: Optional[Sequence[Path]] = None,
    control_metric_basis: str = "auto",
) -> Dict[str, Path]:
    payloads = build_thesis_evidence(
        runs_dir=runs_dir,
        methods_path=methods_path,
        results_path=results_path,
        run_id_allowlist_path=run_id_allowlist_path,
        suite_manifest_paths=suite_manifest_paths,
        synthetic_manifest_paths=synthetic_manifest_paths,
        control_metric_basis=control_metric_basis,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, payload in payloads.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written[name] = path
    return written
