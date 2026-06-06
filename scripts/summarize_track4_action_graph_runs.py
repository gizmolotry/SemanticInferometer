#!/usr/bin/env python3
"""Summarize least-action Track 4 replay runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    return value_f if math.isfinite(value_f) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [v for v in (_safe_float(value) for value in values) if v is not None]
    return float(np.mean(finite)) if finite else None


def _payload_or_record_mean(
    payload: Dict[str, Any],
    records: Sequence[Any],
    payload_key: str,
    record_key: str,
) -> float | None:
    payload_value = _safe_float(payload.get(payload_key))
    if payload_value is not None:
        return payload_value
    return _mean(
        row.get(record_key)
        for row in records
        if isinstance(row, dict)
    )


def _std(values: Iterable[Any]) -> float | None:
    finite = [v for v in (_safe_float(value) for value in values) if v is not None]
    if not finite:
        return None
    return float(np.std(finite))


def _clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    rounded = round(value_f, 12)
    return float(rounded) if math.isfinite(rounded) else None


def _infer_corpus(name: str) -> str:
    if name.startswith("control_random"):
        return "control_random"
    if name.startswith("control_shuffled"):
        return "control_shuffled"
    if name.startswith("control_constant"):
        return "control_constant"
    if name.startswith("real"):
        return "real"
    if "_control_random_" in name:
        return "control_random"
    if "_control_shuffled_" in name:
        return "control_shuffled"
    if "_control_constant_" in name:
        return "control_constant"
    if "_real_" in name:
        return "real"
    return "unknown"


def _infer_kernel(name: str, payload: Dict[str, Any]) -> str:
    raw = payload.get("kernel")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    tokens = [token for token in name.split("_") if token]
    for token in tokens:
        if token in {"rbf", "matern", "imq", "rq", "laplacian", "student", "student_t"}:
            return "student_t" if token == "student" else token
    return "unknown"


def _coerce_seed(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _infer_seed(name: str, payload: Dict[str, Any], manifest: Dict[str, Any]) -> tuple[int | None, str | None]:
    payload_seed = _coerce_seed(payload.get("seed"))
    if payload_seed is not None:
        return payload_seed, "payload"
    manifest_seed = _coerce_seed(manifest.get("seed"))
    if manifest_seed is not None:
        return manifest_seed, "manifest"
    match = re.search(r"(?:^|_)seed_?(\d+)(?:_|$)", name.lower())
    if match:
        return int(match.group(1)), "run_name"
    return None, None


def _clean_basis(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value_s = value.strip()
    return value_s or None


def _infer_basis(name: str, payload: Dict[str, Any], manifest: Dict[str, Any]) -> tuple[str | None, str | None]:
    payload_basis = _clean_basis(payload.get("basis"))
    if payload_basis is not None:
        return payload_basis, "payload"
    manifest_basis = _clean_basis(manifest.get("basis"))
    if manifest_basis is not None:
        return manifest_basis, "manifest"
    lowered = name.lower()
    if re.search(r"(?:^|_)track2(?:_|$)", lowered) or "track4_basis_track2" in lowered:
        return "track2", "run_name"
    if re.search(r"(?:^|_)integrated(?:_|$)", lowered) or "track4_basis_integrated" in lowered:
        return "integrated", "run_name"
    if "logits_flat" in lowered or "track4_basis_logits_flat" in lowered:
        return "logits_flat", "run_name"
    return None, None


def _infer_action_branch(name: str, payload: Dict[str, Any]) -> str:
    raw = payload.get("action_branch")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    lowered = name.lower()
    known = (
        "baseline_raw_action",
        "track2_default",
        "null_calibrated_hysteresis",
        "richer_walker_state",
        "separated_action_channels",
        "virtual_transition_states",
        "path_ensemble_tpt",
        "per_basis_gates",
    )
    for branch in known:
        if branch in lowered:
            return branch
    return "baseline_raw_action"


def _infer_ablation(name: str, payload: Dict[str, Any]) -> str:
    raw = payload.get("ablation")
    if isinstance(raw, str) and raw.strip():
        raw_norm = raw.strip()
        if raw_norm == "observer_shuffled":
            return "shuffled"
        return raw_norm
    lowered = name.lower()
    if "observer_disabled" in lowered or "disabled" in lowered:
        return "observer_disabled"
    if "zero_hysteresis" in lowered:
        return "zero_hysteresis"
    if "shuffled" in lowered and "control_shuffled" not in lowered:
        return "shuffled"
    if "observer_shuffled" in lowered:
        return "shuffled"
    return "full"


def _infer_observer_state_mode(name: str, payload: Dict[str, Any], ablation: str) -> str:
    for key in ("observer_state_mode", "observer_simplex_mode"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            mode = raw.strip().lower()
            return "enabled" if mode == "artifact" else mode
    if ablation == "observer_disabled":
        return "disabled"
    if ablation == "zero_hysteresis":
        return "zero_hysteresis"
    if ablation == "shuffled":
        return "shuffled"
    return "enabled"


def _load_manifest_metadata(summary_path: Path) -> Dict[str, Any]:
    manifest_path = summary_path.parent / "track4_action_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = _load_manifest_metadata(path)
    seed, seed_source = _infer_seed(path.parent.name, payload, manifest)
    basis, basis_source = _infer_basis(path.parent.name, payload, manifest)
    if manifest:
        payload = {**manifest, **payload}
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    observer_transport_mean = _payload_or_record_mean(
        payload,
        records,
        "mean_observer_transport_penalty",
        "observer_transport_penalty",
    )
    hysteresis_mean = _payload_or_record_mean(
        payload,
        records,
        "mean_hysteresis_penalty",
        "hysteresis_penalty",
    )
    null_hysteresis_mean = _payload_or_record_mean(
        payload,
        records,
        "mean_null_hysteresis_penalty",
        "null_hysteresis_penalty",
    )
    excess_hysteresis_mean = _payload_or_record_mean(
        payload,
        records,
        "mean_excess_hysteresis_penalty",
        "excess_hysteresis_penalty",
    )
    positive_excess_hysteresis_mean = _payload_or_record_mean(
        payload,
        records,
        "mean_positive_excess_hysteresis_penalty",
        "positive_excess_hysteresis_penalty",
    )
    calibrated_hysteresis_mean = _payload_or_record_mean(
        payload,
        records,
        "mean_calibrated_hysteresis_penalty",
        "calibrated_hysteresis_penalty",
    )
    run_name = path.parent.name
    ablation = _infer_ablation(run_name, payload)
    observer_state_mode = _infer_observer_state_mode(run_name, payload, ablation)
    return {
        "run_name": run_name,
        "summary_path": str(path),
        "corpus": _infer_corpus(run_name),
        "kernel": _infer_kernel(run_name, payload),
        "seed": seed,
        "seed_source": seed_source,
        "basis": basis,
        "basis_source": basis_source,
        "action_branch": _infer_action_branch(run_name, payload),
        "ablation": ablation,
        "observer_state_mode": observer_state_mode,
        "n_articles": payload.get("n_articles"),
        "path_count": payload.get("path_count"),
        "reached_count": payload.get("reached_count"),
        "mean_action": _safe_float(payload.get("mean_action")),
        "median_action": _safe_float(payload.get("median_action")),
        "max_action": _safe_float(payload.get("max_action")),
        "mean_metric": _mean(row.get("metric") for row in records if isinstance(row, dict)),
        "mean_shear_penalty": _mean(row.get("shear_penalty") for row in records if isinstance(row, dict)),
        "mean_observer_transport_penalty": observer_transport_mean,
        "mean_hysteresis_penalty": hysteresis_mean,
        "mean_null_hysteresis_penalty": null_hysteresis_mean,
        "mean_excess_hysteresis_penalty": excess_hysteresis_mean,
        "mean_positive_excess_hysteresis_penalty": positive_excess_hysteresis_mean,
        "mean_calibrated_hysteresis_penalty": calibrated_hysteresis_mean,
        "mean_stress_penalty": _mean(row.get("stress_penalty") for row in records if isinstance(row, dict)),
        "mean_curvature_penalty": _mean(row.get("curvature_penalty") for row in records if isinstance(row, dict)),
        "observer_simplex_contract_supported": observer_transport_mean is not None and hysteresis_mean is not None,
        "terrain_zone_counts": payload.get("terrain_zone_counts", {}),
        "anchor_zones": payload.get("anchor_zones", []),
    }


DEFAULT_OBSERVER_STATE_THRESHOLDS = {
    "min_real_over_control_mean_action": 1.20,
    "min_real_minus_control_mean_action": 2.0,
    "min_real_over_shuffled_hysteresis": 1.20,
    "min_real_minus_disabled_observer_transport": 0.50,
}


CALIBRATED_HYSTERESIS_BRANCHES = {"null_calibrated_hysteresis"}


OBSERVER_STATE_METRICS = (
    "mean_action",
    "mean_observer_transport_penalty",
    "mean_hysteresis_penalty",
    "mean_null_hysteresis_penalty",
    "mean_excess_hysteresis_penalty",
    "mean_positive_excess_hysteresis_penalty",
    "mean_calibrated_hysteresis_penalty",
)


def _ordered_present(values: Iterable[str], required: Sequence[str] | None = None) -> List[str]:
    present = {
        str(value)
        for value in values
        if value is not None and str(value) and str(value) not in {"unknown", "None"}
    }
    if required:
        ordered = [str(value) for value in required if str(value) in present]
        ordered.extend(sorted(present - set(ordered)))
        return ordered
    return sorted(present)


def _ordered_present_int(values: Iterable[Any], required: Sequence[int] | None = None) -> List[int]:
    present = {seed for seed in (_coerce_seed(value) for value in values) if seed is not None}
    if required:
        required_int = [int(value) for value in required]
        ordered = [value for value in required_int if value in present]
        ordered.extend(sorted(present - set(ordered)))
        return ordered
    return sorted(present)


def _missing_required(required: Sequence[Any], present: Sequence[Any]) -> List[Any]:
    present_set = set(present)
    return [value for value in required if value not in present_set]


def _metric_comparison(real_value: float | None, control_value: float | None) -> Dict[str, float | None]:
    gap = (
        float(real_value - control_value)
        if real_value is not None and control_value is not None
        else None
    )
    ratio = (
        float(real_value / control_value)
        if real_value is not None and control_value not in (None, 0.0)
        else None
    )
    return {
        "real": _clean_float(real_value),
        "control": _clean_float(control_value),
        "gap": _clean_float(gap),
        "ratio": _clean_float(ratio),
    }


def _baseline_comparison(full_real_value: float | None, baseline_value: float | None) -> Dict[str, float | None]:
    gap = (
        float(baseline_value - full_real_value)
        if full_real_value is not None and baseline_value is not None
        else None
    )
    ratio = (
        float(full_real_value / baseline_value)
        if full_real_value is not None and baseline_value not in (None, 0.0)
        else None
    )
    return {
        "full_real": _clean_float(full_real_value),
        "baseline": _clean_float(baseline_value),
        "gap_vs_full_real": _clean_float(gap),
        "ratio_full_real_over_baseline": _clean_float(ratio),
    }


def _select_hysteresis_gate_metric(rows: Sequence[Dict[str, Any]]) -> str:
    full_branches = {
        str(row.get("action_branch"))
        for row in rows
        if row.get("ablation") == "full" and row.get("action_branch")
    }
    if full_branches and full_branches.issubset(CALIBRATED_HYSTERESIS_BRANCHES):
        return "mean_calibrated_hysteresis_penalty"
    return "mean_hysteresis_penalty"


def _hysteresis_gate_pass(
    comparison: Dict[str, Any],
    *,
    min_ratio: float,
) -> bool:
    ratio = _safe_float(comparison.get("ratio_full_real_over_baseline"))
    if ratio is not None:
        return ratio >= float(min_ratio)
    full_real = _safe_float(comparison.get("full_real"))
    baseline = _safe_float(comparison.get("baseline"))
    # Null-calibrated positive-excess hysteresis often destroys to exact zero.
    # A positive full value over a zero destroyed baseline is a clean separation,
    # even though the ratio is mathematically infinite and not JSON-friendly.
    if baseline == 0.0 and full_real is not None:
        return full_real > 0.0
    return False


def _append_missing_metric(failures: List[str], comparison: Dict[str, Any], reason: str) -> None:
    if comparison.get("ratio") is None or comparison.get("gap") is None:
        failures.append(reason)


def _evaluate_observer_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    limits = {**DEFAULT_OBSERVER_STATE_THRESHOLDS, **(thresholds or {})}
    full_rows = [row for row in rows if row.get("ablation") == "full"]
    full_real_rows = [row for row in full_rows if row.get("corpus") == "real"]
    full_control_rows = [row for row in full_rows if str(row.get("corpus", "")).startswith("control_")]
    metrics = OBSERVER_STATE_METRICS
    hysteresis_gate_metric = _select_hysteresis_gate_metric(rows)
    real_vs_control = {
        metric: _metric_comparison(
            _mean(row.get(metric) for row in full_real_rows),
            _mean(row.get(metric) for row in full_control_rows),
        )
        for metric in metrics
    }
    baseline_comparisons: Dict[str, Dict[str, Any]] = {}
    for baseline_name in ("observer_disabled", "zero_hysteresis", "shuffled"):
        baseline_rows = [row for row in rows if row.get("ablation") == baseline_name]
        baseline_comparisons[baseline_name] = {
            metric: _baseline_comparison(
                _mean(row.get(metric) for row in full_real_rows),
                _mean(row.get(metric) for row in baseline_rows),
            )
            for metric in metrics
        }

    action_cmp = real_vs_control["mean_action"]
    action_ratio = _safe_float(action_cmp.get("ratio"))
    action_gap = _safe_float(action_cmp.get("gap"))
    real_control_pass = (
        action_ratio is not None
        and action_ratio >= float(limits["min_real_over_control_mean_action"])
        and action_gap is not None
        and action_gap >= float(limits["min_real_minus_control_mean_action"])
    )
    shuffled_hysteresis = baseline_comparisons["shuffled"].get(hysteresis_gate_metric, {})
    shuffled_hysteresis_ratio = _safe_float(shuffled_hysteresis.get("ratio_full_real_over_baseline"))
    shuffled_hysteresis_pass = _hysteresis_gate_pass(
        shuffled_hysteresis,
        min_ratio=float(limits["min_real_over_shuffled_hysteresis"]),
    )
    disabled_transport = baseline_comparisons["observer_disabled"]["mean_observer_transport_penalty"]
    disabled_transport_gap = _safe_float(disabled_transport.get("gap_vs_full_real"))
    disabled_transport_pass = (
        disabled_transport_gap is not None
        and abs(disabled_transport_gap) >= float(limits["min_real_minus_disabled_observer_transport"])
        and disabled_transport_gap < 0.0
    )

    failures: List[str] = []
    _append_missing_metric(failures, action_cmp, "missing_real_or_control_action_baseline")
    if action_cmp.get("ratio") is not None and action_cmp.get("gap") is not None and not real_control_pass:
        failures.append("real_control_separation_below_threshold")
    if (
        shuffled_hysteresis_ratio is None
        and _safe_float(shuffled_hysteresis.get("full_real")) is None
        and _safe_float(shuffled_hysteresis.get("baseline")) is None
    ):
        failures.append("missing_shuffled_hysteresis_baseline")
    elif not shuffled_hysteresis_pass:
        failures.append("shuffled_hysteresis_baseline_not_separated")
    if disabled_transport_gap is None:
        failures.append("missing_observer_disabled_transport_baseline")
    elif not disabled_transport_pass:
        failures.append("observer_disabled_transport_baseline_not_separated")
    observer_rows = [row for row in full_rows if bool(row.get("observer_simplex_contract_supported"))]
    observer_contract = bool(full_rows) and len(observer_rows) == len(full_rows)
    if not observer_contract:
        failures.append("observer_simplex_contract_missing")

    return {
        "row_count": len(rows),
        "pass": not failures,
        "status": "pass" if not failures else "fail",
        "failure_reasons": failures,
        "real_control_separation_pass": real_control_pass,
        "shuffled_hysteresis_baseline_pass": shuffled_hysteresis_pass,
        "observer_disabled_transport_baseline_pass": disabled_transport_pass,
        "observer_simplex_contract_supported": observer_contract,
        "hysteresis_gate_metric": hysteresis_gate_metric,
        "comparisons": {"real_vs_control": real_vs_control},
        "baseline_comparisons": baseline_comparisons,
    }


def _aggregate_by(
    rows: Sequence[Dict[str, Any]],
    key: str,
    *,
    thresholds: Dict[str, float] | None = None,
    key_normalizer: Callable[[Any], Any] | None = None,
) -> Dict[str, Any]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or value == "unknown":
            continue
        value = key_normalizer(value) if key_normalizer else str(value)
        grouped.setdefault(value, []).append(row)
    return {
        str(value): _evaluate_observer_rows(group_rows, thresholds=thresholds)
        for value, group_rows in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def _aggregate_by_keys(
    rows: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    *,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        parts: List[str] = []
        missing = False
        for key in keys:
            value = row.get(key)
            if value is None or value == "unknown":
                missing = True
                break
            if key == "seed":
                seed = _coerce_seed(value)
                if seed is None:
                    missing = True
                    break
                parts.append(f"seed{seed}")
            else:
                parts.append(str(value))
        if missing:
            continue
        grouped.setdefault("|".join(parts), []).append(row)
    return {
        str(value): _evaluate_observer_rows(group_rows, thresholds=thresholds)
        for value, group_rows in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def _aggregate_action_ratios(aggregate: Dict[str, Any]) -> Dict[str, float | None]:
    ratios: Dict[str, float | None] = {}
    for key, payload in aggregate.items():
        if not isinstance(payload, dict):
            ratios[str(key)] = None
            continue
        comparison = (
            payload.get("comparisons", {})
            .get("real_vs_control", {})
            .get("mean_action", {})
        )
        ratios[str(key)] = _clean_float(_safe_float(comparison.get("ratio")))
    return ratios


def _aggregate_action_pass(aggregate: Dict[str, Any]) -> bool:
    return bool(aggregate) and all(
        bool(payload.get("real_control_separation_pass"))
        and bool(payload.get("observer_simplex_contract_supported"))
        for payload in aggregate.values()
        if isinstance(payload, dict)
    )


def evaluate_observer_state_claim(
    summary: Dict[str, Any],
    *,
    thresholds: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate the exploratory observer-state Track 4 action-separation claim."""

    threshold_payload = thresholds or {}
    limits = {**DEFAULT_OBSERVER_STATE_THRESHOLDS, **threshold_payload}
    comparisons = summary.get("comparisons", {}) if isinstance(summary.get("comparisons"), dict) else {}
    real_vs_control = (
        comparisons.get("real_vs_control", {})
        if isinstance(comparisons.get("real_vs_control"), dict)
        else {}
    )
    action_cmp = (
        real_vs_control.get("mean_action", {})
        if isinstance(real_vs_control.get("mean_action"), dict)
        else {}
    )
    baselines = (
        summary.get("baseline_comparisons", {})
        if isinstance(summary.get("baseline_comparisons"), dict)
        else {}
    )
    hysteresis_gate_metric = str(
        summary.get("hysteresis_gate_metric")
        or _select_hysteresis_gate_metric(
            summary.get("rows", []) if isinstance(summary.get("rows"), list) else []
        )
    )
    shuffled_hysteresis = (
        baselines.get("shuffled", {}).get(hysteresis_gate_metric, {})
        if isinstance(baselines.get("shuffled"), dict)
        else {}
    )
    disabled_transport = (
        baselines.get("observer_disabled", {}).get("mean_observer_transport_penalty", {})
        if isinstance(baselines.get("observer_disabled"), dict)
        else {}
    )

    required_kernels_present = bool(summary.get("required_kernels_present"))
    required_seeds = [
        int(value)
        for value in threshold_payload.get("required_seeds", summary.get("required_seeds", [])) or []
    ]
    required_bases = [
        str(value)
        for value in threshold_payload.get("required_bases", summary.get("required_bases", [])) or []
    ]
    seeds_present = [
        int(value)
        for value in summary.get("seeds_present", [])
        if _coerce_seed(value) is not None
    ]
    # Backward compatibility: legacy single-seed summaries often omitted seed
    # metadata, but represented the default seed. Keep multi-seed gates additive.
    if required_seeds and not seeds_present:
        seeds_present = [required_seeds[0]]
    bases_present = [str(value) for value in summary.get("bases_present", []) if str(value)]
    missing_required_seeds = _missing_required(required_seeds, seeds_present)
    missing_required_bases = _missing_required(required_bases, bases_present)
    required_seeds_present = not missing_required_seeds
    required_bases_present = not missing_required_bases
    observer_contract = bool(summary.get("observer_simplex_contract_supported"))
    action_ratio = _safe_float(action_cmp.get("ratio"))
    action_gap = _safe_float(action_cmp.get("gap"))
    real_control_pass = (
        action_ratio is not None
        and action_ratio >= float(limits["min_real_over_control_mean_action"])
        and action_gap is not None
        and action_gap >= float(limits["min_real_minus_control_mean_action"])
    )
    shuffled_hysteresis_ratio = _safe_float(shuffled_hysteresis.get("ratio_full_real_over_baseline"))
    shuffled_hysteresis_pass = _hysteresis_gate_pass(
        shuffled_hysteresis,
        min_ratio=float(limits["min_real_over_shuffled_hysteresis"]),
    )
    disabled_transport_gap = _safe_float(disabled_transport.get("gap_vs_full_real"))
    disabled_transport_pass = (
        disabled_transport_gap is not None
        and abs(disabled_transport_gap) >= float(limits["min_real_minus_disabled_observer_transport"])
        and disabled_transport_gap < 0.0
    )

    failures: List[str] = []
    if not required_kernels_present:
        failures.append("missing_required_kernels")
    if not observer_contract:
        failures.append("observer_simplex_contract_missing")
    if not real_control_pass:
        failures.append("real_control_separation_below_threshold")
    if not shuffled_hysteresis_pass:
        failures.append("shuffled_hysteresis_baseline_not_separated")
    if not disabled_transport_pass:
        failures.append("observer_disabled_transport_baseline_not_separated")
    if required_seeds and not required_seeds_present:
        failures.append("missing_required_seeds")
    if required_bases and not required_bases_present:
        failures.append("missing_required_bases")
    kernel_robustness_pass = summary.get("kernel_robustness_pass")
    seed_robustness_pass = summary.get("seed_robustness_pass")
    basis_robustness_pass = summary.get("basis_robustness_pass")
    basis_seed_robustness_pass = summary.get("basis_seed_robustness_pass")
    kernel_basis_robustness_pass = summary.get("kernel_basis_robustness_pass")
    kernel_seed_basis_robustness_pass = summary.get("kernel_seed_basis_robustness_pass")
    action_only_robustness_pass = summary.get("action_only_robustness_pass")
    action_kernel_robustness_pass = summary.get("action_kernel_robustness_pass")
    action_seed_robustness_pass = summary.get("action_seed_robustness_pass")
    action_basis_robustness_pass = summary.get("action_basis_robustness_pass")
    action_basis_seed_robustness_pass = summary.get("action_basis_seed_robustness_pass")
    action_kernel_basis_robustness_pass = summary.get("action_kernel_basis_robustness_pass")
    action_kernel_seed_basis_robustness_pass = summary.get("action_kernel_seed_basis_robustness_pass")
    if action_only_robustness_pass is None:
        action_only_robustness_pass = (
            real_control_pass
            and (kernel_robustness_pass is not False)
            and (seed_robustness_pass is not False)
            and (basis_robustness_pass is not False)
        )
    if kernel_robustness_pass is False:
        failures.append("kernel_robustness_failed")
    if required_seeds and seed_robustness_pass is False:
        failures.append("seed_robustness_failed")
    if required_bases and basis_robustness_pass is False:
        failures.append("basis_robustness_failed")
    if required_seeds and required_bases and basis_seed_robustness_pass is False:
        failures.append("basis_seed_robustness_failed")
    if required_bases and kernel_basis_robustness_pass is False:
        failures.append("kernel_basis_robustness_failed")
    if required_seeds and required_bases and kernel_seed_basis_robustness_pass is False:
        failures.append("kernel_seed_basis_robustness_failed")
    seed_robustness_return = bool(seed_robustness_pass) if required_seeds else seed_robustness_pass
    basis_robustness_return = bool(basis_robustness_pass) if required_bases else basis_robustness_pass

    return {
        "claim_id": "track4_observer_state_action_separation",
        "claim_scope": "exploratory_track4",
        "pass": not failures,
        "thesis_safe": not failures,
        "safe_for_thesis_claim": not failures,
        "failure_reasons": failures,
        "required_kernels_present": required_kernels_present,
        "required_seeds": required_seeds,
        "seeds_present": seeds_present,
        "missing_required_seeds": missing_required_seeds,
        "required_seeds_present": required_seeds_present,
        "required_bases": required_bases,
        "bases_present": bases_present,
        "missing_required_bases": missing_required_bases,
        "required_bases_present": required_bases_present,
        "observer_simplex_contract_supported": observer_contract,
        "real_control_separation_pass": real_control_pass,
        "shuffled_hysteresis_baseline_pass": shuffled_hysteresis_pass,
        "hysteresis_gate_metric": hysteresis_gate_metric,
        "observer_disabled_transport_baseline_pass": disabled_transport_pass,
        "kernel_robustness_pass": kernel_robustness_pass,
        "seed_robustness_pass": seed_robustness_return,
        "basis_robustness_pass": basis_robustness_return,
        "basis_seed_robustness_pass": basis_seed_robustness_pass,
        "kernel_basis_robustness_pass": kernel_basis_robustness_pass,
        "kernel_seed_basis_robustness_pass": kernel_seed_basis_robustness_pass,
        "action_only_robustness_pass": bool(action_only_robustness_pass),
        "action_kernel_robustness_pass": action_kernel_robustness_pass,
        "action_seed_robustness_pass": action_seed_robustness_pass,
        "action_basis_robustness_pass": action_basis_robustness_pass,
        "action_basis_seed_robustness_pass": action_basis_seed_robustness_pass,
        "action_kernel_basis_robustness_pass": action_kernel_basis_robustness_pass,
        "action_kernel_seed_basis_robustness_pass": action_kernel_seed_basis_robustness_pass,
        "hysteresis_mechanism_pass": bool(shuffled_hysteresis_pass),
        "action_ratio_by_kernel": summary.get("action_ratio_by_kernel"),
        "action_ratio_by_seed": summary.get("action_ratio_by_seed"),
        "action_ratio_by_basis": summary.get("action_ratio_by_basis"),
        "dispersion_across_kernels": summary.get("dispersion_across_kernels"),
        "dispersion_across_seeds": summary.get("dispersion_across_seeds"),
        "dispersion_across_bases": summary.get("dispersion_across_bases"),
        "thresholds": limits,
        "point_estimate": action_ratio,
        "effect_direction": "real_observer_state_action_gt_controls",
    }


def summarize_observer_state_ablation(
    paths: Iterable[Path],
    *,
    required_kernels: Sequence[str] = ("rbf", "matern", "imq"),
    required_seeds: Sequence[int] | None = None,
    required_bases: Sequence[str] | None = None,
    thresholds: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Summarize observer-state Track 4 action replay and ablation baselines."""

    rows = [_load_summary(Path(path)) for path in paths]
    full_rows = [row for row in rows if row.get("ablation") == "full"]
    full_real_rows = [row for row in full_rows if row.get("corpus") == "real"]
    full_control_rows = [row for row in full_rows if str(row.get("corpus", "")).startswith("control_")]
    required_seed_list = [int(seed) for seed in required_seeds or ()]
    required_basis_list = [str(basis) for basis in required_bases or ()]
    kernels_present = _ordered_present((row.get("kernel", "unknown") for row in full_rows), required_kernels)
    seeds_present = _ordered_present_int((row.get("seed") for row in full_rows), required_seed_list)
    bases_present = _ordered_present((row.get("basis", "unknown") for row in full_rows), required_basis_list)
    action_branches_present = _ordered_present(row.get("action_branch", "unknown") for row in full_rows)
    missing_required_kernels = _missing_required([str(kernel) for kernel in required_kernels], kernels_present)
    missing_required_seeds = _missing_required(required_seed_list, seeds_present)
    missing_required_bases = _missing_required(required_basis_list, bases_present)

    metrics = OBSERVER_STATE_METRICS
    hysteresis_gate_metric = _select_hysteresis_gate_metric(rows)
    real_vs_control = {
        metric: _metric_comparison(
            _mean(row.get(metric) for row in full_real_rows),
            _mean(row.get(metric) for row in full_control_rows),
        )
        for metric in metrics
    }
    baseline_comparisons: Dict[str, Dict[str, Any]] = {}
    for baseline_name in ("observer_disabled", "zero_hysteresis", "shuffled"):
        baseline_rows = [row for row in rows if row.get("ablation") == baseline_name]
        baseline_comparisons[baseline_name] = {
            metric: _baseline_comparison(
                _mean(row.get(metric) for row in full_real_rows),
                _mean(row.get(metric) for row in baseline_rows),
            )
            for metric in metrics
        }

    observer_rows = [row for row in full_rows if bool(row.get("observer_simplex_contract_supported"))]
    aggregates = {
        "by_kernel": _aggregate_by(rows, "kernel", thresholds=thresholds),
        "by_seed": _aggregate_by(rows, "seed", thresholds=thresholds, key_normalizer=lambda value: int(value)),
        "by_basis": _aggregate_by(rows, "basis", thresholds=thresholds),
        "by_action_branch": _aggregate_by(rows, "action_branch", thresholds=thresholds),
        "by_action_branch_basis": _aggregate_by_keys(rows, ("action_branch", "basis"), thresholds=thresholds),
        "by_basis_seed": _aggregate_by_keys(rows, ("basis", "seed"), thresholds=thresholds),
        "by_kernel_basis": _aggregate_by_keys(rows, ("kernel", "basis"), thresholds=thresholds),
        "by_kernel_seed_basis": _aggregate_by_keys(rows, ("kernel", "seed", "basis"), thresholds=thresholds),
    }
    kernel_robustness_pass = bool(aggregates["by_kernel"]) and all(
        bool(payload.get("pass")) for payload in aggregates["by_kernel"].values()
    )
    seed_robustness_pass = bool(aggregates["by_seed"]) and all(
        bool(payload.get("pass")) for payload in aggregates["by_seed"].values()
    )
    basis_robustness_pass = bool(aggregates["by_basis"]) and all(
        bool(payload.get("pass")) for payload in aggregates["by_basis"].values()
    )
    basis_seed_robustness_pass = bool(aggregates["by_basis_seed"]) and all(
        bool(payload.get("pass")) for payload in aggregates["by_basis_seed"].values()
    )
    kernel_basis_robustness_pass = bool(aggregates["by_kernel_basis"]) and all(
        bool(payload.get("pass")) for payload in aggregates["by_kernel_basis"].values()
    )
    kernel_seed_basis_robustness_pass = bool(aggregates["by_kernel_seed_basis"]) and all(
        bool(payload.get("pass")) for payload in aggregates["by_kernel_seed_basis"].values()
    )
    action_kernel_robustness_pass = _aggregate_action_pass(aggregates["by_kernel"])
    action_seed_robustness_pass = _aggregate_action_pass(aggregates["by_seed"])
    action_basis_robustness_pass = _aggregate_action_pass(aggregates["by_basis"])
    action_basis_seed_robustness_pass = _aggregate_action_pass(aggregates["by_basis_seed"])
    action_kernel_basis_robustness_pass = _aggregate_action_pass(aggregates["by_kernel_basis"])
    action_kernel_seed_basis_robustness_pass = _aggregate_action_pass(aggregates["by_kernel_seed_basis"])
    action_only_robustness_pass = (
        action_kernel_robustness_pass
        and action_seed_robustness_pass
        and action_basis_robustness_pass
        and action_basis_seed_robustness_pass
        and action_kernel_basis_robustness_pass
        and action_kernel_seed_basis_robustness_pass
    )
    action_ratio_by_kernel = _aggregate_action_ratios(aggregates["by_kernel"])
    action_ratio_by_seed = _aggregate_action_ratios(aggregates["by_seed"])
    action_ratio_by_basis = _aggregate_action_ratios(aggregates["by_basis"])
    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "summary_type": "track4_observer_state_ablation",
        "claim_scope": "exploratory_track4",
        "safe_for_thesis_claim": False,
        "safe_for_multi_seed_robustness_claim": False,
        "safe_for_multi_basis_robustness_claim": False,
        "row_count": len(rows),
        "rows": rows,
        "required_kernels": list(required_kernels),
        "kernels_present": kernels_present,
        "missing_required_kernels": missing_required_kernels,
        "required_kernels_present": not missing_required_kernels,
        "required_seeds": required_seed_list,
        "seeds_present": seeds_present,
        "missing_required_seeds": missing_required_seeds,
        "required_seeds_present": not missing_required_seeds,
        "required_bases": required_basis_list,
        "bases_present": bases_present,
        "action_branches_present": action_branches_present,
        "hysteresis_gate_metric": hysteresis_gate_metric,
        "missing_required_bases": missing_required_bases,
        "required_bases_present": not missing_required_bases,
        "observer_simplex_row_count": len(observer_rows),
        "observer_simplex_contract_supported": bool(full_rows) and len(observer_rows) == len(full_rows),
        "comparisons": {
            "real_vs_control": real_vs_control,
        },
        "baseline_comparisons": baseline_comparisons,
        "aggregates": aggregates,
        "kernel_robustness_pass": kernel_robustness_pass,
        "seed_robustness_pass": seed_robustness_pass,
        "basis_robustness_pass": basis_robustness_pass,
        "basis_seed_robustness_pass": basis_seed_robustness_pass,
        "kernel_basis_robustness_pass": kernel_basis_robustness_pass,
        "kernel_seed_basis_robustness_pass": kernel_seed_basis_robustness_pass,
        "action_only_robustness_pass": action_only_robustness_pass,
        "action_kernel_robustness_pass": action_kernel_robustness_pass,
        "action_seed_robustness_pass": action_seed_robustness_pass,
        "action_basis_robustness_pass": action_basis_robustness_pass,
        "action_basis_seed_robustness_pass": action_basis_seed_robustness_pass,
        "action_kernel_basis_robustness_pass": action_kernel_basis_robustness_pass,
        "action_kernel_seed_basis_robustness_pass": action_kernel_seed_basis_robustness_pass,
        "action_ratio_by_kernel": action_ratio_by_kernel,
        "action_ratio_by_seed": action_ratio_by_seed,
        "action_ratio_by_basis": action_ratio_by_basis,
        "dispersion_across_kernels": _clean_float(_std(action_ratio_by_kernel.values())),
        "dispersion_across_seeds": _clean_float(_std(action_ratio_by_seed.values())),
        "dispersion_across_bases": _clean_float(_std(action_ratio_by_basis.values())),
        "interpretation": (
            "Observer-state Track 4 action separation tests whether traversing real discourse "
            "requires more article-coordinate action, V-observer transport, and directed hysteresis "
            "than matched controls or destroyed observer-state baselines."
        ),
    }
    claim = evaluate_observer_state_claim(summary, thresholds=thresholds)
    summary["claim_evaluation"] = claim
    summary["pass"] = bool(claim["pass"])
    summary["thesis_safe"] = bool(claim["thesis_safe"])
    summary["safe_for_thesis_claim"] = bool(claim["safe_for_thesis_claim"])
    summary["safe_for_multi_seed_robustness_claim"] = (
        bool(claim["safe_for_thesis_claim"]) and len(seeds_present) > 1 and bool(seed_robustness_pass)
    )
    summary["safe_for_multi_basis_robustness_claim"] = (
        bool(claim["safe_for_thesis_claim"]) and len(bases_present) > 1 and bool(basis_robustness_pass)
    )
    return summary


def summarize(paths: Iterable[Path]) -> Dict[str, Any]:
    rows = [_load_summary(path) for path in paths]
    real_rows = [row for row in rows if row.get("corpus") == "real"]
    control_rows = [row for row in rows if str(row.get("corpus", "")).startswith("control_")]
    real_mean_action = _mean(row.get("mean_action") for row in real_rows)
    control_mean_action = _mean(row.get("mean_action") for row in control_rows)
    action_gap = (
        float(real_mean_action - control_mean_action)
        if real_mean_action is not None and control_mean_action is not None
        else None
    )
    action_ratio = (
        float(real_mean_action / control_mean_action)
        if real_mean_action is not None and control_mean_action not in (None, 0.0)
        else None
    )
    observer_rows = [row for row in rows if bool(row.get("observer_simplex_contract_supported"))]
    real_hysteresis_mean = _mean(row.get("mean_hysteresis_penalty") for row in real_rows)
    control_hysteresis_mean = _mean(row.get("mean_hysteresis_penalty") for row in control_rows)
    real_observer_transport_mean = _mean(row.get("mean_observer_transport_penalty") for row in real_rows)
    control_observer_transport_mean = _mean(row.get("mean_observer_transport_penalty") for row in control_rows)
    return {
        "schema_version": "1.0",
        "summary_type": "track4_action_graph_probe",
        "claim_scope": "engineering_probe_not_thesis_claim",
        "safe_for_thesis_claim": False,
        "row_count": len(rows),
        "rows": rows,
        "real_mean_action": real_mean_action,
        "control_mean_action": control_mean_action,
        "real_minus_control_mean_action": action_gap,
        "real_over_control_mean_action": action_ratio,
        "action_separation_candidate": bool(action_gap is not None and action_gap > 0.0),
        "observer_simplex_row_count": len(observer_rows),
        "observer_simplex_contract_supported": len(observer_rows) == len(rows) and bool(rows),
        "real_mean_observer_transport_penalty": real_observer_transport_mean,
        "control_mean_observer_transport_penalty": control_observer_transport_mean,
        "real_mean_hysteresis_penalty": real_hysteresis_mean,
        "control_mean_hysteresis_penalty": control_hysteresis_mean,
        "observer_simplex_interpretation": (
            "Observer penalties treat the eight V-observers as the Track 4 node-simplex state. "
            "Transport is symmetric simplex displacement; hysteresis is directed KL(next || current)."
        ),
        "interpretation": (
            "Least-action replay compares deterministic action cost on existing artifacts. "
            "Positive separation is an engineering lead for Track 4, not standalone validation."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "outputs" / "track4_action_graph",
        help="Directory containing action graph run folders.",
    )
    parser.add_argument(
        "--pattern",
        default="*/track4_action_summary.json",
        help="Glob under --input-root. The default is suffix-agnostic so new replay batches do not masquerade as stale dated runs.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        action="append",
        default=[],
        help="Explicit track4_action_summary.json path. Repeat to bypass glob discovery.",
    )
    parser.add_argument(
        "--summary-mode",
        choices=("probe", "observer_state_ablation"),
        default="probe",
        help="Write the original engineering probe summary or the observer-state ablation claim summary.",
    )
    parser.add_argument(
        "--required-kernel",
        action="append",
        default=[],
        help="Required kernel for observer-state ablation summaries. Repeatable.",
    )
    parser.add_argument(
        "--required-seed",
        type=int,
        action="append",
        default=[],
        help="Required seed for observer-state ablation robustness summaries. Repeatable.",
    )
    parser.add_argument(
        "--required-basis",
        action="append",
        default=[],
        help="Required basis for observer-state ablation robustness summaries. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in args.summary_path] if args.summary_path else sorted(Path(args.input_root).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no action summaries matched {args.input_root / args.pattern}")
    if args.summary_mode == "observer_state_ablation":
        payload = summarize_observer_state_ablation(
            paths,
            required_kernels=tuple(args.required_kernel or ("rbf", "matern", "imq")),
            required_seeds=tuple(args.required_seed),
            required_bases=tuple(args.required_basis),
        )
        filename = "track4_observer_state_ablation_summary.json"
    else:
        payload = summarize(paths)
        filename = "track4_action_graph_probe_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / filename
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=OK")
    print(f"summary={out}")
    print(f"rows={payload['row_count']}")
    if args.summary_mode == "observer_state_ablation":
        claim = payload.get("claim_evaluation", {})
        print(f"safe_for_thesis_claim={payload.get('safe_for_thesis_claim')}")
        print(f"observer_state_claim_pass={claim.get('pass')}")
    else:
        print(f"real_over_control_mean_action={payload['real_over_control_mean_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
