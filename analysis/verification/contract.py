from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
CONTRACT_VERSION = "1.0"
REQUIRED_CONSUMER_ARTIFACTS = ("baseline_meta.json", "baseline_state.json", "verification_report.json")
OPTIONAL_CONSUMER_ARTIFACTS = (
    "verification_summary.csv",
    "labels/hidden_groups.csv",
    "labels/derived/group_summaries.json",
    "labels/derived/group_matrix.json",
)
REQUIRED_PROVENANCE_KEYS = (
    "schema_version",
    "cache_version",
    "dataset_hash",
    "code_hash_or_commit",
    "weights_hash",
    "kernel_params",
    "rks_dim",
    "crn_seed",
    "alpha",
    "timestamp_utc",
    "verification_status",
)


class LayerStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NON_COMPARABLE = "NON_COMPARABLE"
    MISSING_ARTIFACTS = "MISSING_ARTIFACTS"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class VerificationBundle:
    report_path: Path
    summary_path: Optional[Path]
    report: Dict[str, Any]
    summary_rows: List[Dict[str, Any]]
    validation: ValidationResult


@dataclass
class ConsumerDiagnostics:
    run_dir: Path
    verification_status: str
    global_pass: bool
    contract_ok: bool
    missing_required_artifacts: List[str] = field(default_factory=list)
    missing_optional_artifacts: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    paths: Dict[str, Optional[Path]] = field(default_factory=dict)


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _safe_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return loaded


def _safe_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _safe_json_or_empty(path: Path) -> Dict[str, Any]:
    try:
        return _safe_json(path)
    except Exception:
        return {}


def validate_layer(layer: Dict[str, Any], idx: int = 0) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    prefix = f"layers[{idx}]"

    required = {"layer_id", "layer_name", "status", "checks", "fail_reasons"}
    missing = sorted(required - set(layer.keys()))
    if missing:
        errors.append(f"{prefix} missing fields: {', '.join(missing)}")
        return ValidationResult(False, errors, warnings)

    status = str(layer.get("status", ""))
    if status not in {s.value for s in LayerStatus}:
        errors.append(f"{prefix}.status invalid: {status}")

    checks = layer.get("checks")
    if not _is_list(checks):
        errors.append(f"{prefix}.checks must be a list")
    else:
        seen_check_names = set()
        for cidx, check in enumerate(checks):
            if not _is_dict(check):
                errors.append(f"{prefix}.checks[{cidx}] must be an object")
                continue
            name = str(check.get("name", "")).strip()
            if not name:
                errors.append(f"{prefix}.checks[{cidx}] missing name")
                continue
            if name in seen_check_names:
                warnings.append(f"{prefix}.checks has duplicate name '{name}'")
            seen_check_names.add(name)
            if "pass" not in check:
                warnings.append(f"{prefix}.checks[{cidx}] has no 'pass' field")

    fail_reasons = layer.get("fail_reasons")
    if not _is_list(fail_reasons):
        errors.append(f"{prefix}.fail_reasons must be a list")

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_report(report: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    required_top = {"run_id", "timestamp", "layers", "global_pass"}
    missing_top = sorted(required_top - set(report.keys()))
    if missing_top:
        errors.append(f"report missing fields: {', '.join(missing_top)}")
        return ValidationResult(False, errors, warnings)

    if "schema_version" in report and str(report["schema_version"]) != SCHEMA_VERSION:
        warnings.append(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, found {report['schema_version']}"
        )
    if "contract_version" in report and str(report["contract_version"]) != CONTRACT_VERSION:
        warnings.append(
            f"contract_version mismatch: expected {CONTRACT_VERSION}, found {report['contract_version']}"
        )

    layers = report.get("layers")
    if not _is_list(layers):
        errors.append("report.layers must be a list")
    else:
        for i, layer in enumerate(layers):
            if not _is_dict(layer):
                errors.append(f"layers[{i}] must be an object")
                continue
            vr = validate_layer(layer, i)
            errors.extend(vr.errors)
            warnings.extend(vr.warnings)

        statuses = [str(layer.get("status")) for layer in layers if _is_dict(layer)]
        inferred_global = all(status == LayerStatus.VERIFIED.value for status in statuses) and bool(statuses)
        declared_global = _coerce_bool(report.get("global_pass"))
        if declared_global is None:
            errors.append("global_pass must be a boolean")
        elif declared_global != inferred_global:
            warnings.append(
                f"global_pass mismatch with layer statuses (declared={declared_global}, inferred={inferred_global})"
            )

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_baseline_meta(meta: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    required = set(REQUIRED_PROVENANCE_KEYS)
    missing = sorted(required - set(meta.keys()))
    if missing:
        errors.append(f"baseline_meta missing keys: {', '.join(missing)}")
    status = str(meta.get("verification_status", "")).upper()
    if status and status not in {s.value for s in LayerStatus}:
        errors.append(f"baseline_meta verification_status invalid: {status}")
    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_baseline_state(state: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    for key in ("articles", "paths", "axes", "metrics"):
        if key not in state:
            errors.append(f"baseline_state missing '{key}'")
    if "articles" in state and not isinstance(state.get("articles"), list):
        errors.append("baseline_state.articles must be a list")
    if "paths" in state and not isinstance(state.get("paths"), list):
        errors.append("baseline_state.paths must be a list")
    return ValidationResult(len(errors) == 0, errors, warnings)


def _artifact_map(run_dir: Path) -> Dict[str, Optional[Path]]:
    run_dir = Path(run_dir)
    return {
        "baseline_meta.json": run_dir / "baseline_meta.json",
        "baseline_state.json": run_dir / "baseline_state.json",
        "verification_report.json": run_dir / "verification_report.json",
        "verification_summary.csv": run_dir / "verification_summary.csv",
        "labels/hidden_groups.csv": run_dir / "labels" / "hidden_groups.csv",
        "labels/derived/group_summaries.json": run_dir / "labels" / "derived" / "group_summaries.json",
        "labels/derived/group_matrix.json": run_dir / "labels" / "derived" / "group_matrix.json",
    }


def evaluate_consumer_contract(run_dir: Path) -> ConsumerDiagnostics:
    run_dir = Path(run_dir)
    paths = _artifact_map(run_dir)
    missing_required = [k for k in REQUIRED_CONSUMER_ARTIFACTS if not paths[k].exists()]
    missing_optional = [k for k in OPTIONAL_CONSUMER_ARTIFACTS if not paths[k].exists()]

    schema_errors: List[str] = []
    warnings: List[str] = []
    verification_status = LayerStatus.UNVERIFIED.value
    global_pass = False

    report = _safe_json_or_empty(paths["verification_report.json"])
    if report:
        v_report = validate_report(report)
        schema_errors.extend(v_report.errors)
        warnings.extend(v_report.warnings)
        try:
            global_pass = bool(report.get("global_pass") is True)
        except Exception:
            global_pass = False
        layers = report.get("layers", [])
        if isinstance(layers, list) and layers:
            statuses = [str(layer.get("status", "")) for layer in layers if isinstance(layer, dict)]
            if statuses and all(s == LayerStatus.VERIFIED.value for s in statuses):
                verification_status = LayerStatus.VERIFIED.value
            elif any(s == LayerStatus.MISSING_ARTIFACTS.value for s in statuses):
                verification_status = LayerStatus.MISSING_ARTIFACTS.value
            elif any(s == LayerStatus.NON_COMPARABLE.value for s in statuses):
                verification_status = LayerStatus.NON_COMPARABLE.value
            else:
                verification_status = LayerStatus.UNVERIFIED.value

    meta = _safe_json_or_empty(paths["baseline_meta.json"])
    if meta:
        vm = validate_baseline_meta(meta)
        schema_errors.extend(vm.errors)
        warnings.extend(vm.warnings)
        meta_status = str(meta.get("verification_status", "")).upper().strip()
        if meta_status in {s.value for s in LayerStatus} and verification_status == LayerStatus.UNVERIFIED.value:
            verification_status = meta_status

    state = _safe_json_or_empty(paths["baseline_state.json"])
    if state:
        vs = validate_baseline_state(state)
        schema_errors.extend(vs.errors)
        warnings.extend(vs.warnings)

    contract_ok = (len(missing_required) == 0) and (len(schema_errors) == 0)
    return ConsumerDiagnostics(
        run_dir=run_dir,
        verification_status=verification_status,
        global_pass=global_pass,
        contract_ok=contract_ok,
        missing_required_artifacts=missing_required,
        missing_optional_artifacts=missing_optional,
        schema_errors=schema_errors,
        warnings=warnings,
        paths=paths,
    )


def resolve_report_path(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    report_path = run_dir / "verification_report.json"
    if report_path.exists():
        return report_path
    raise FileNotFoundError(f"verification_report.json not found in {run_dir}")


def resolve_summary_path(run_dir: Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    summary_path = run_dir / "verification_summary.csv"
    return summary_path if summary_path.exists() else None


def load_bundle(run_dir: Path, strict: bool = True) -> VerificationBundle:
    report_path = resolve_report_path(run_dir)
    summary_path = resolve_summary_path(run_dir)
    report = _safe_json(report_path)
    summary_rows = _safe_csv(summary_path) if summary_path else []
    validation = validate_report(report)
    if strict and not validation.ok:
        joined = "; ".join(validation.errors)
        raise ValueError(f"Invalid verification report contract: {joined}")
    return VerificationBundle(
        report_path=report_path,
        summary_path=summary_path,
        report=report,
        summary_rows=summary_rows,
        validation=validation,
    )


def is_verified(report: Dict[str, Any], strict_contract: bool = True) -> bool:
    if strict_contract:
        vr = validate_report(report)
        if not vr.ok:
            return False
    global_pass = _coerce_bool(report.get("global_pass"))
    return bool(global_pass is True)


def summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    layers = report.get("layers", [])
    layer_rows = []
    for layer in layers:
        if not _is_dict(layer):
            continue
        checks = layer.get("checks", [])
        check_map = {}
        if _is_list(checks):
            for check in checks:
                if _is_dict(check):
                    name = str(check.get("name", "")).strip()
                    if name:
                        check_map[name] = check.get("pass")
        layer_rows.append(
            {
                "layer_id": layer.get("layer_id"),
                "status": layer.get("status"),
                "crn_locked": check_map.get("crn_locked"),
                "control_ordering": check_map.get("control_ordering"),
                "seed_stability": check_map.get("seed_stability"),
                "fail_reasons": layer.get("fail_reasons", []),
            }
        )
    return {
        "run_id": report.get("run_id"),
        "timestamp": report.get("timestamp"),
        "global_pass": report.get("global_pass"),
        "layers": layer_rows,
    }
