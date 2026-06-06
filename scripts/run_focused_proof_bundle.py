#!/usr/bin/env python3
"""Run a deadline-optimized focused proof bundle and refresh thesis evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ablation_dag import build_orchestration_contract

RUNS_ROOT = REPO_ROOT / "outputs" / "experiments" / "runs"
THESIS_ROOT = REPO_ROOT / "outputs" / "thesis_validation" / "focused"

DEFAULT_CORPORA = ["real", "control_constant", "control_shuffled", "control_random"]
DEFAULT_KERNELS = ["rbf", "matern", "imq"]
DEFAULT_SEEDS = [42, 420, 4200]
DEFAULT_CHANNELS = ["cls"]
DEFAULT_MIN_FREE_GB = 8.0
CLAIM_PROFILES: Dict[str, Sequence[str]] = {
    "full_system": (
        "control_destruction",
        "observer_relativity",
        "track4_traversal_validity",
        "track5_ablation_coverage",
        "verification_provenance",
    ),
    "procrustes_control": (
        "procrustes_control_separation",
        "observer_relativity",
        "track5_ablation_coverage",
        "procrustes_verification_provenance",
    ),
}


def _suite_stage_id(track5_mode: str, *, synthetic: bool = False) -> str:
    if synthetic:
        return "synthetic_bundle"
    if track5_mode == "riemannian_strict":
        return "riemannian_main"
    return "hadamard_main"


def _returncode_status(returncode: object) -> str:
    try:
        code = int(returncode)
    except Exception:
        return "unknown"
    if code == 0:
        return "completed"
    if code == 70:
        return "blocked"
    return "failed"


def _suite_orchestration_contract(
    *,
    track5_mode: str,
    synthetic: bool,
    run_id: Optional[str],
    status: str,
    command: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    node_id = _suite_stage_id(track5_mode, synthetic=synthetic)
    manifest_paths = []
    if run_id:
        manifest_paths.append(RUNS_ROOT / str(run_id) / "experiment_manifest.json")
    return build_orchestration_contract(
        dag_id=f"focused_proof_stage.{node_id}",
        executor="external_subprocess",
        nodes=[
            {
                "node_id": node_id,
                "status": status,
                "track5_mode": track5_mode,
                "synthetic": bool(synthetic),
                "run_id": run_id,
                "limit": limit,
                "command": list(command or []),
            }
        ],
        dependencies={node_id: []},
        manifest_paths=manifest_paths,
        metadata={
            "runner": "run_full_experiment_suite.py",
            "airflow_sidecar": "analysis.airflow_ablation_orchestrator.run_ablation_matrix",
        },
    )


def _reused_suite_result(
    *,
    run_id: str,
    track5_mode: str,
    synthetic: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    run_id = str(run_id).strip()
    return {
        "command": [],
        "returncode": 0,
        "run_id": run_id,
        "synthetic": bool(synthetic),
        "track5_mode": track5_mode,
        "limit": limit,
        "observer_bundle_scope": "reused",
        "observer_indices": None,
        "observer_index_policy": "reused",
        "observer_index_count": 0,
        "no_cache": None,
        "disk_preflight": None,
        "timeout_seconds": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "reused": True,
        "orchestration_contract": _suite_orchestration_contract(
            track5_mode=track5_mode,
            synthetic=synthetic,
            run_id=run_id,
            status="completed",
            command=[],
            limit=limit,
        ),
    }


def _focused_orchestration_contract(
    *,
    bundle_name: str,
    stage: str,
    suite_results: Optional[Sequence[Dict[str, object]]] = None,
    evidence: Optional[Dict[str, object]] = None,
    run_ids: Optional[Sequence[str]] = None,
    reuse: bool = False,
    skip_riemannian: bool = False,
    skip_synthetic: bool = False,
) -> Dict[str, object]:
    suite_results = list(suite_results or [])
    run_ids = [str(run_id) for run_id in (run_ids or []) if str(run_id).strip()]
    if not run_ids:
        run_ids = [
            str(result.get("run_id")).strip()
            for result in suite_results
            if str(result.get("run_id") or "").strip()
        ]

    enabled_suite_nodes: List[str] = []
    if not reuse:
        enabled_suite_nodes.append("hadamard_main")
        if not skip_riemannian:
            enabled_suite_nodes.append("riemannian_main")
        if not skip_synthetic:
            enabled_suite_nodes.append("synthetic_bundle")

    status_by_node: Dict[str, str] = {}
    attempts_by_node: Dict[str, int] = {}
    run_ids_by_node: Dict[str, List[str]] = {}
    for result in suite_results:
        node_id = _suite_stage_id(
            str(result.get("track5_mode") or "hadamard_strict"),
            synthetic=bool(result.get("synthetic", False)),
        )
        attempts_by_node[node_id] = attempts_by_node.get(node_id, 0) + 1
        run_id = str(result.get("run_id") or "").strip()
        if run_id:
            run_ids_by_node.setdefault(node_id, []).append(run_id)
        status_by_node[node_id] = _returncode_status(result.get("returncode"))

    running_stage_map = {
        "hadamard_main": "hadamard_main",
        "hadamard_fallback": "hadamard_main",
        "riemannian_main": "riemannian_main",
        "riemannian_fallback": "riemannian_main",
        "synthetic_bundle": "synthetic_bundle",
        "evidence_refresh": "evidence_refresh",
        "evidence_refresh_reuse": "evidence_refresh",
    }
    active_node = running_stage_map.get(stage)

    nodes: List[Dict[str, object]] = []
    for node_id in enabled_suite_nodes:
        node_status = status_by_node.get(node_id, "pending")
        if active_node == node_id and node_status == "pending":
            node_status = "running"
        nodes.append(
            {
                "node_id": node_id,
                "status": node_status,
                "attempts": attempts_by_node.get(node_id, 0),
                "run_ids": run_ids_by_node.get(node_id, []),
            }
        )

    evidence_status = "pending"
    if evidence is not None:
        evidence_status = _returncode_status(evidence.get("returncode"))
    elif active_node == "evidence_refresh":
        evidence_status = "running"
    nodes.append(
        {
            "node_id": "evidence_refresh",
            "status": evidence_status,
            "command": list((evidence or {}).get("command") or []),
        }
    )
    nodes.append(
        {
            "node_id": "marker_publish",
            "status": "completed" if stage == "complete" else "pending",
        }
    )

    dependencies: Dict[str, Sequence[str]] = {}
    if "hadamard_main" in enabled_suite_nodes:
        dependencies["hadamard_main"] = []
    if "riemannian_main" in enabled_suite_nodes:
        dependencies["riemannian_main"] = ["hadamard_main"] if "hadamard_main" in enabled_suite_nodes else []
    if "synthetic_bundle" in enabled_suite_nodes:
        parent = "riemannian_main" if "riemannian_main" in enabled_suite_nodes else "hadamard_main"
        dependencies["synthetic_bundle"] = [parent] if parent in enabled_suite_nodes else []
    dependencies["evidence_refresh"] = list(enabled_suite_nodes)
    dependencies["marker_publish"] = ["evidence_refresh"]

    manifest_paths = [RUNS_ROOT / run_id / "experiment_manifest.json" for run_id in run_ids]
    return build_orchestration_contract(
        dag_id=f"focused_proof.{bundle_name}",
        executor="external_subprocess",
        nodes=nodes,
        dependencies=dependencies,
        manifest_paths=manifest_paths,
        metadata={
            "bundle_name": bundle_name,
            "stage": stage,
            "reused_run_ids": bool(reuse),
            "airflow_sidecars": [
                "analysis.airflow_ablation_orchestrator.run_ablation_matrix",
                "core.master_ablation.AblationDag",
            ],
        },
    )


def _write_status(path: Path, payload: Dict[str, object]) -> None:
    if "orchestration_contract" not in payload and payload.get("bundle_name"):
        payload = dict(payload)
        payload["orchestration_contract"] = _focused_orchestration_contract(
            bundle_name=str(payload.get("bundle_name")),
            stage=str(payload.get("stage") or "unknown"),
            suite_results=payload.get("runs") if isinstance(payload.get("runs"), list) else [],
            evidence=payload.get("evidence") if isinstance(payload.get("evidence"), dict) else None,
            run_ids=(
                payload.get("accepted_run_ids")
                if isinstance(payload.get("accepted_run_ids"), list)
                else payload.get("reused_run_ids")
                if isinstance(payload.get("reused_run_ids"), list)
                else []
            ),
            reuse=bool(payload.get("reused")) or bool(payload.get("reused_run_ids")),
            skip_riemannian=bool(payload.get("skip_riemannian", False)),
            skip_synthetic=bool(payload.get("skip_synthetic", False)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _list_run_ids() -> set[str]:
    if not RUNS_ROOT.exists():
        return set()
    return {
        path.name
        for path in RUNS_ROOT.glob("experiments_*")
        if path.is_dir()
    }


def _run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    print(f"\n[FOCUSED] Running: {' '.join(cmd)}")
    try:
        return subprocess.run(
            list(cmd),
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            list(cmd),
            returncode=124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(
                (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            ) + f"\nfocused proof stage timed out after {timeout_seconds:.0f}s",
        )


def _free_space_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / float(1024 ** 3)


def _disk_blocked_payload(
    *,
    min_free_gb: float,
    stage: str,
    synthetic: bool = False,
    limit: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    if float(min_free_gb or 0.0) <= 0:
        return None
    free_gb = _free_space_gb(REPO_ROOT)
    if free_gb >= float(min_free_gb):
        return None
    payload = {
        "command": [],
        "returncode": 70,
        "run_id": None,
        "synthetic": bool(synthetic),
        "track5_mode": stage,
        "limit": limit,
        "observer_bundle_scope": None,
        "no_cache": None,
        "stdout_tail": "",
        "stderr_tail": (
            f"focused proof disk preflight blocked {stage}: "
            f"{free_gb:.2f}GB free < {float(min_free_gb):.2f}GB required"
        ),
        "disk_preflight": {
            "pass": False,
            "free_gb": free_gb,
            "min_free_gb": float(min_free_gb),
        },
    }
    payload["orchestration_contract"] = _suite_orchestration_contract(
        track5_mode=stage,
        synthetic=synthetic,
        run_id=None,
        status="blocked",
        command=[],
        limit=limit,
    )
    return payload


def _parse_observer_indices_arg(raw: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not raw:
        return None
    values: List[int] = []
    for chunk in raw:
        for token in str(chunk).replace(",", " ").split():
            if token.strip():
                values.append(int(token))
    return values


def _run_suite(
    *,
    limit: int,
    kernels: Sequence[str],
    channels: Sequence[str],
    corpora: Sequence[str],
    seeds: Sequence[int],
    track5_mode: str,
    synthetic: bool = False,
    synthetic_n_articles: int = 15,
    synthetic_clusters: int = 4,
    verify: bool = False,
    post_sync_results: bool = False,
    control_metric_basis: str = "auto",
    observer_bundle_scope: str = "real",
    observer_indices: Optional[Sequence[int]] = None,
    observer_index_policy: str = "all",
    observer_index_count: int = 3,
    no_cache: bool = False,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, object]:
    blocked = _disk_blocked_payload(
        min_free_gb=min_free_gb,
        stage=track5_mode,
        synthetic=synthetic,
        limit=limit,
    )
    if blocked is not None:
        return blocked

    before = _list_run_ids()
    cmd = [
        sys.executable,
        "run_full_experiment_suite.py",
        "--mode",
        "enhanced",
        "--limit",
        str(limit),
        "--kernels",
        *[str(kernel) for kernel in kernels],
        "--channels",
        *[str(channel) for channel in channels],
        "--seeds",
        *[str(seed) for seed in seeds],
        "--track5-assembly-mode",
        track5_mode,
        "--observer-bundle-scope",
        observer_bundle_scope,
        "--control-metric-basis",
        control_metric_basis,
        "--no-probe",
    ]
    if observer_indices is not None:
        cmd.extend(["--observer-indices", *[str(idx) for idx in observer_indices]])
    cmd.extend(["--observer-index-policy", str(observer_index_policy)])
    cmd.extend(["--observer-index-count", str(observer_index_count)])
    cmd.append("--isolate-contract-bundle")
    if no_cache:
        cmd.append("--no-cache")
    if verify:
        cmd.append("--verify")
    if not post_sync_results:
        cmd.append("--no-post-sync-results")
    if synthetic:
        cmd.extend(
            [
                "--synthetic",
                "--synthetic-n-articles",
                str(synthetic_n_articles),
                "--synthetic-clusters",
                str(synthetic_clusters),
            ]
        )
    else:
        cmd.extend(["--corpora", *[str(corpus) for corpus in corpora]])

    if timeout_seconds is None:
        proc = _run_command(cmd, cwd=REPO_ROOT)
    else:
        proc = _run_command(cmd, cwd=REPO_ROOT, timeout_seconds=timeout_seconds)
    after = _list_run_ids()
    created = sorted(after - before)
    run_id = created[-1] if created else None
    stage_status = _returncode_status(proc.returncode)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "run_id": run_id,
        "synthetic": synthetic,
        "track5_mode": track5_mode,
        "limit": limit,
        "observer_bundle_scope": observer_bundle_scope,
        "observer_indices": list(observer_indices) if observer_indices is not None else None,
        "observer_index_policy": observer_index_policy,
        "observer_index_count": int(observer_index_count),
        "no_cache": bool(no_cache),
        "disk_preflight": {
            "pass": True,
            "free_gb": _free_space_gb(REPO_ROOT),
            "min_free_gb": float(min_free_gb or 0.0),
        },
        "timeout_seconds": timeout_seconds,
        "stdout_tail": proc.stdout[-8000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-8000:] if proc.stderr else "",
        "orchestration_contract": _suite_orchestration_contract(
            track5_mode=track5_mode,
            synthetic=synthetic,
            run_id=run_id,
            status=stage_status,
            command=cmd,
            limit=limit,
        ),
    }


def _write_run_id_allowlist(path: Path, run_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(run_id) for run_id in run_ids if str(run_id).strip()) + "\n", encoding="utf-8")


def _successful_run_ids(results: Sequence[Dict[str, object]]) -> List[str]:
    return [
        str(result["run_id"])
        for result in results
        if int(result.get("returncode", 1)) == 0
        and isinstance(result.get("run_id"), str)
        and str(result.get("run_id", "")).strip()
    ]


def _failed_run_summaries(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    failed: List[Dict[str, object]] = []
    for result in results:
        if int(result.get("returncode", 1)) == 0:
            continue
        failed.append(
            {
                "run_id": result.get("run_id"),
                "track5_mode": result.get("track5_mode"),
                "limit": result.get("limit"),
                "returncode": result.get("returncode"),
            }
        )
    return failed


def _build_evidence_bundle(
    *,
    bundle_dir: Path,
    run_id_allowlist: Path,
    control_metric_basis: str = "auto",
) -> Dict[str, object]:
    cmd = [
        sys.executable,
        "scripts/build_thesis_evidence.py",
        "--runs-dir",
        str(RUNS_ROOT),
        "--methods-path",
        "METHODS.md",
        "--results-path",
        "RESULTS.md",
        "--output-dir",
        str(bundle_dir),
        "--run-id-allowlist",
        str(run_id_allowlist),
        "--control-metric-basis",
        control_metric_basis,
    ]
    proc = _run_command(cmd, cwd=REPO_ROOT)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-8000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-8000:] if proc.stderr else "",
    }


def _sync_results_registry() -> Dict[str, object]:
    cmd = [sys.executable, "scripts/build_results_registry.py"]
    proc = _run_command(cmd, cwd=REPO_ROOT)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }


def _focused_evidence_acceptance(bundle_dir: Path, *, claim_profile: str = "full_system") -> Dict[str, object]:
    claim_path = bundle_dir / "claim_matrix.json"
    summary_path = bundle_dir / "scientific_validation_summary.json"
    if not claim_path.exists():
        return {
            "mechanically_usable": False,
            "safe_for_thesis_claims": False,
            "safe_for_focused_defense": False,
            "unsafe_claim_ids": ["claim_matrix_missing"],
            "failure_mode_count": None,
        }
    try:
        claim_payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "mechanically_usable": False,
            "safe_for_thesis_claims": False,
            "safe_for_focused_defense": False,
            "unsafe_claim_ids": ["claim_matrix_unreadable"],
            "error": str(exc),
            "failure_mode_count": None,
        }

    claims = claim_payload.get("claims") if isinstance(claim_payload, dict) else []
    claims = [claim for claim in claims if isinstance(claim, dict)]
    unsafe = [
        str(claim.get("claim_id") or "unknown")
        for claim in claims
        if not bool(claim.get("thesis_safe", False))
    ]
    if claim_profile not in CLAIM_PROFILES:
        claim_profile = "full_system"
    focused_required = set(CLAIM_PROFILES[claim_profile])
    by_id = {str(claim.get("claim_id") or ""): claim for claim in claims}
    missing_focused = sorted(focused_required - set(by_id))
    unsafe_focused = [
        claim_id
        for claim_id in sorted(focused_required)
        if claim_id in by_id and not bool(by_id[claim_id].get("thesis_safe", False))
    ]
    failure_mode_count = None
    focused_selection_valid = True
    selected_manifest_count = None
    missing_requested_run_ids: List[str] = []
    missing_requested_manifest_paths: List[str] = []
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            failure_mode_count = int(((summary_payload.get("failure_modes") or {}).get("count") or 0))
            selection = summary_payload.get("input_selection") or {}
            if bool(selection.get("focused_filter_active", False)):
                selected_manifest_count = int(selection.get("selected_manifest_count") or 0)
                missing_requested_run_ids = [
                    str(item)
                    for item in (selection.get("missing_requested_run_ids") or [])
                    if str(item).strip()
                ]
                missing_requested_manifest_paths = [
                    str(item)
                    for item in (selection.get("missing_requested_manifest_paths") or [])
                    if str(item).strip()
                ]
                focused_selection_valid = (
                    selected_manifest_count > 0
                    and not missing_requested_run_ids
                    and not missing_requested_manifest_paths
                )
        except Exception:
            failure_mode_count = None
            focused_selection_valid = False
            selected_manifest_count = None
            missing_requested_run_ids = ["scientific_validation_summary_unreadable"]
    return {
        "mechanically_usable": bool(claims) and focused_selection_valid,
        "claim_count": len(claims),
        "thesis_safe_claim_count": len(claims) - len(unsafe),
        "unsafe_claim_ids": unsafe,
        "claim_profile": claim_profile,
        "focused_required_claim_ids": sorted(focused_required),
        "missing_focused_claim_ids": missing_focused,
        "unsafe_focused_claim_ids": unsafe_focused,
        "failure_mode_count": failure_mode_count,
        "focused_selection_valid": focused_selection_valid,
        "selected_manifest_count": selected_manifest_count,
        "missing_requested_run_ids": missing_requested_run_ids,
        "missing_requested_manifest_paths": missing_requested_manifest_paths,
        "safe_for_thesis_claims": bool(claims) and not unsafe,
        "safe_for_focused_defense": (
            bool(claims)
            and focused_selection_valid
            and not missing_focused
            and not unsafe_focused
        ),
    }


def _preferred_run_key(preferred_run_id: Optional[str], *, kernel: str = "matern", channel: str = "cls", corpus: str = "real") -> Optional[str]:
    if not preferred_run_id:
        return None
    run_dir = RUNS_ROOT / preferred_run_id / kernel / channel / corpus
    if run_dir.exists():
        return str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/")
    for fallback in (RUNS_ROOT / preferred_run_id).rglob("MONOLITH_DATA.csv"):
        return str(fallback.parent.relative_to(REPO_ROOT)).replace("\\", "/")
    return None


def _write_marker(
    *,
    bundle_dir: Path,
    bundle_name: str,
    run_ids: Sequence[str],
    preferred_run_id: Optional[str],
    hadamard_run_id: Optional[str],
    riemannian_run_id: Optional[str],
    synthetic_run_id: Optional[str],
    activate_current: bool = True,
    evidence_acceptance: Optional[Dict[str, object]] = None,
) -> Path:
    marker_path = THESIS_ROOT / "current_bundle.json" if activate_current else bundle_dir / "inactive_bundle_marker.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": "success" if activate_current else "inactive",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_name": bundle_name,
        "artifact_root": str(RUNS_ROOT),
        "evidence_dir": str(bundle_dir),
        "run_ids": list(run_ids),
        "preferred_run_id": preferred_run_id,
        "preferred_run_key": _preferred_run_key(preferred_run_id),
        "run_families": {
            "hadamard_main": hadamard_run_id,
            "riemannian_main": riemannian_run_id,
            "synthetic": synthetic_run_id,
        },
        "evidence_acceptance": dict(evidence_acceptance or {}),
    }
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-limit", type=int, default=120)
    parser.add_argument("--fallback-limit", type=int, default=60)
    parser.add_argument("--synthetic-limit", type=int, default=60)
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--corpora", nargs="+", default=list(DEFAULT_CORPORA))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--skip-riemannian", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--post-sync-results", action="store_true")
    parser.add_argument(
        "--control-metric-basis",
        choices=["auto", "direct", "comprehensive"],
        default="auto",
        help="Forwarded to run_full_experiment_suite.py for control_metrics.json provenance.",
    )
    parser.add_argument(
        "--observer-bundle-scope",
        choices=["all", "real", "none"],
        default="real",
        help="Forwarded to run_full_experiment_suite.py; use real for thesis observer relativity, none for smoke runs.",
    )
    parser.add_argument(
        "--observer-indices",
        nargs="+",
        default=None,
        help="Optional observer/article indices to materialize; accepts space- or comma-separated integers.",
    )
    parser.add_argument(
        "--observer-index-policy",
        choices=["all", "anchors"],
        default="all",
        help="When explicit indices are omitted, materialize all observers or only Track 4 anchor observers.",
    )
    parser.add_argument(
        "--observer-index-count",
        type=int,
        default=3,
        help="Maximum automatic observers when --observer-index-policy anchors is active.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Forward --no-cache to suite runs to reduce disk pressure during focused proof bundles.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            "Block the next suite stage if free space under the repo drive is below this threshold "
            f"(default: {DEFAULT_MIN_FREE_GB:g}GB)."
        ),
    )
    parser.add_argument(
        "--suite-timeout-minutes",
        type=float,
        default=0.0,
        help="Optional timeout for each run_full_experiment_suite.py stage; 0 disables the watchdog.",
    )
    parser.add_argument(
        "--claim-profile",
        choices=sorted(CLAIM_PROFILES),
        default="full_system",
        help=(
            "Focused acceptance profile. full_system keeps Track 4 in the gate; "
            "procrustes_control gates the narrower Procrustes/observer/Track5 claim bundle."
        ),
    )
    parser.add_argument(
        "--reuse-run-ids",
        nargs="+",
        default=None,
        help="Skip experiment execution and build focused evidence from these existing experiment run ids.",
    )
    parser.add_argument("--preferred-run-id", default=None)
    parser.add_argument("--hadamard-run-id", default=None)
    parser.add_argument("--riemannian-run-id", default=None)
    parser.add_argument("--synthetic-run-id", default=None)
    parser.add_argument(
        "--reuse-hadamard-run-id",
        default=None,
        help="Reuse an existing Hadamard suite run and continue with missing focused stages.",
    )
    parser.add_argument(
        "--reuse-riemannian-run-id",
        default=None,
        help="Reuse an existing strict-Riemannian suite run and continue with missing focused stages.",
    )
    parser.add_argument(
        "--reuse-synthetic-run-id",
        default=None,
        help="Reuse an existing synthetic suite run and continue with evidence refresh.",
    )
    args = parser.parse_args()
    try:
        args.observer_indices = _parse_observer_indices_arg(args.observer_indices)
    except Exception as exc:
        parser.error(f"invalid --observer-indices: {exc}")
    suite_timeout_seconds = (
        float(args.suite_timeout_minutes) * 60.0
        if float(args.suite_timeout_minutes or 0.0) > 0.0
        else None
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_name = f"focused_proof_{timestamp}"
    bundle_dir = THESIS_ROOT / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    status_path = bundle_dir / "focused_proof_status.json"

    suite_results: List[Dict[str, object]] = []
    if args.reuse_run_ids:
        run_ids = [str(run_id).strip() for run_id in args.reuse_run_ids if str(run_id).strip()]
        allowlist_path = bundle_dir / "focused_run_ids.txt"
        _write_run_id_allowlist(allowlist_path, run_ids)
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "running",
                "stage": "evidence_refresh_reuse",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "runs": suite_results,
                "evidence": None,
                "reused_run_ids": run_ids,
            },
        )
        registry_sync = _sync_results_registry() if args.post_sync_results else None
        evidence = _build_evidence_bundle(
            bundle_dir=bundle_dir,
            run_id_allowlist=allowlist_path,
            control_metric_basis=args.control_metric_basis,
        )
        evidence_acceptance = _focused_evidence_acceptance(bundle_dir, claim_profile=args.claim_profile)
        proof_success = (
            bool(run_ids)
            and int(evidence["returncode"]) == 0
            and bool(evidence_acceptance.get("safe_for_focused_defense", False))
        )
        preferred_run_id = args.preferred_run_id or (run_ids[0] if run_ids else None)
        marker_path = _write_marker(
            bundle_dir=bundle_dir,
            bundle_name=bundle_name,
            run_ids=run_ids,
            preferred_run_id=preferred_run_id,
            hadamard_run_id=args.hadamard_run_id,
            riemannian_run_id=args.riemannian_run_id,
            synthetic_run_id=args.synthetic_run_id,
            activate_current=proof_success,
            evidence_acceptance=evidence_acceptance,
        )
        summary = {
            "bundle_name": bundle_name,
            "bundle_dir": str(bundle_dir),
            "allowlist_path": str(allowlist_path),
            "status_path": str(status_path),
            "marker_path": str(marker_path),
            "accepted_run_ids": run_ids,
            "rejected_runs": [],
            "evidence_acceptance": evidence_acceptance,
            "runs": suite_results,
            "evidence": evidence,
            "registry_sync": registry_sync,
            "reused": True,
            "orchestration_contract": _focused_orchestration_contract(
                bundle_name=bundle_name,
                stage="complete",
                suite_results=suite_results,
                evidence=evidence,
                run_ids=run_ids,
                reuse=True,
            ),
        }
        summary_path = bundle_dir / "focused_proof_bundle.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "success" if proof_success else "failed",
                "stage": "complete",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "summary_path": str(summary_path),
                "runs": suite_results,
                "evidence": evidence,
                "evidence_acceptance": evidence_acceptance,
                "registry_sync": registry_sync,
                "reused_run_ids": run_ids,
            },
        )
        print("\n[FOCUSED] Reused-run bundle summary:")
        print(json.dumps(summary, indent=2))
        return 0 if proof_success else 1

    _write_status(
        status_path,
        {
            "bundle_name": bundle_name,
            "status": "running",
            "stage": "hadamard_main",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bundle_dir": str(bundle_dir),
            "runs": suite_results,
            "evidence": None,
        },
    )

    if args.reuse_hadamard_run_id:
        hadamard = _reused_suite_result(
            run_id=args.reuse_hadamard_run_id,
            track5_mode="hadamard_strict",
            synthetic=False,
            limit=args.main_limit,
        )
    else:
        hadamard = _run_suite(
            limit=args.main_limit,
            kernels=args.kernels,
            channels=args.channels,
            corpora=args.corpora,
            seeds=args.seeds,
            track5_mode="hadamard_strict",
            verify=args.verify,
            post_sync_results=args.post_sync_results,
            control_metric_basis=args.control_metric_basis,
            observer_bundle_scope=args.observer_bundle_scope,
            observer_indices=args.observer_indices,
            observer_index_policy=args.observer_index_policy,
            observer_index_count=args.observer_index_count,
            no_cache=args.no_cache,
            min_free_gb=args.min_free_gb,
            timeout_seconds=suite_timeout_seconds,
        )
    suite_results.append(hadamard)
    _write_status(
        status_path,
        {
            "bundle_name": bundle_name,
            "status": "running" if int(hadamard["returncode"]) == 0 else "retrying",
            "stage": "hadamard_main_complete",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bundle_dir": str(bundle_dir),
            "runs": suite_results,
            "evidence": None,
        },
    )

    if int(hadamard["returncode"]) != 0 and int(args.fallback_limit) != int(args.main_limit):
        print(f"[FOCUSED][WARN] Main hadamard bundle failed at limit={args.main_limit}; retrying at limit={args.fallback_limit}")
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "retrying",
                "stage": "hadamard_fallback",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "runs": suite_results,
                "evidence": None,
            },
        )
        hadamard = _run_suite(
            limit=args.fallback_limit,
            kernels=args.kernels,
            channels=args.channels,
            corpora=args.corpora,
            seeds=args.seeds,
            track5_mode="hadamard_strict",
            verify=args.verify,
            post_sync_results=args.post_sync_results,
            control_metric_basis=args.control_metric_basis,
            observer_bundle_scope=args.observer_bundle_scope,
            observer_indices=args.observer_indices,
            observer_index_policy=args.observer_index_policy,
            observer_index_count=args.observer_index_count,
            no_cache=args.no_cache,
            min_free_gb=args.min_free_gb,
            timeout_seconds=suite_timeout_seconds,
        )
        suite_results.append(hadamard)
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "running" if int(hadamard["returncode"]) == 0 else "failed",
                "stage": "hadamard_fallback_complete",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "runs": suite_results,
                "evidence": None,
            },
        )

    riemannian: Optional[Dict[str, object]] = None
    if not args.skip_riemannian:
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "running",
                "stage": "riemannian_main",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "runs": suite_results,
                "evidence": None,
            },
        )
        riemannian_limit = (
            args.main_limit
            if int(hadamard["returncode"]) == 0
            and int(hadamard.get("limit", args.main_limit) or args.main_limit) == args.main_limit
            else args.fallback_limit
        )
        if args.reuse_riemannian_run_id:
            riemannian = _reused_suite_result(
                run_id=args.reuse_riemannian_run_id,
                track5_mode="riemannian_strict",
                synthetic=False,
                limit=riemannian_limit,
            )
        else:
            riemannian = _run_suite(
                limit=riemannian_limit,
                kernels=args.kernels,
                channels=args.channels,
                corpora=args.corpora,
                seeds=args.seeds,
                track5_mode="riemannian_strict",
                verify=args.verify,
                post_sync_results=args.post_sync_results,
                control_metric_basis=args.control_metric_basis,
                observer_bundle_scope=args.observer_bundle_scope,
                observer_indices=args.observer_indices,
                observer_index_policy=args.observer_index_policy,
                observer_index_count=args.observer_index_count,
                no_cache=args.no_cache,
                min_free_gb=args.min_free_gb,
                timeout_seconds=suite_timeout_seconds,
            )
        suite_results.append(riemannian)
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "running" if int(riemannian["returncode"]) == 0 else "retrying",
                "stage": "riemannian_main_complete",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "runs": suite_results,
                "evidence": None,
            },
        )
        if int(riemannian["returncode"]) != 0 and int(riemannian.get("limit", args.main_limit)) != args.fallback_limit:
            print(f"[FOCUSED][WARN] Main riemannian bundle failed at limit={args.main_limit}; retrying at limit={args.fallback_limit}")
            _write_status(
                status_path,
                {
                    "bundle_name": bundle_name,
                    "status": "retrying",
                    "stage": "riemannian_fallback",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bundle_dir": str(bundle_dir),
                    "runs": suite_results,
                    "evidence": None,
                },
            )
            riemannian = _run_suite(
                limit=args.fallback_limit,
                kernels=args.kernels,
                channels=args.channels,
                corpora=args.corpora,
                seeds=args.seeds,
                track5_mode="riemannian_strict",
                verify=args.verify,
                post_sync_results=args.post_sync_results,
                control_metric_basis=args.control_metric_basis,
                observer_bundle_scope=args.observer_bundle_scope,
                observer_indices=args.observer_indices,
                observer_index_policy=args.observer_index_policy,
                observer_index_count=args.observer_index_count,
                no_cache=args.no_cache,
                min_free_gb=args.min_free_gb,
                timeout_seconds=suite_timeout_seconds,
            )
            suite_results.append(riemannian)
            _write_status(
                status_path,
                {
                    "bundle_name": bundle_name,
                    "status": "running" if int(riemannian["returncode"]) == 0 else "failed",
                    "stage": "riemannian_fallback_complete",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bundle_dir": str(bundle_dir),
                    "runs": suite_results,
                    "evidence": None,
                },
            )

    synthetic: Optional[Dict[str, object]] = None
    if not args.skip_synthetic:
        _write_status(
            status_path,
            {
                "bundle_name": bundle_name,
                "status": "running",
                "stage": "synthetic_bundle",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_dir": str(bundle_dir),
                "runs": suite_results,
                "evidence": None,
            },
        )
        per_cluster = max(1, int(args.synthetic_limit) // 4)
        if args.reuse_synthetic_run_id:
            synthetic = _reused_suite_result(
                run_id=args.reuse_synthetic_run_id,
                track5_mode="hadamard_strict",
                synthetic=True,
                limit=args.synthetic_limit,
            )
        else:
            synthetic = _run_suite(
                limit=args.synthetic_limit,
                kernels=args.kernels,
                channels=args.channels,
                corpora=args.corpora,
                seeds=args.seeds,
                track5_mode="hadamard_strict",
                synthetic=True,
                synthetic_n_articles=per_cluster,
                synthetic_clusters=4,
                verify=args.verify,
                post_sync_results=args.post_sync_results,
                control_metric_basis=args.control_metric_basis,
                observer_bundle_scope=args.observer_bundle_scope,
                observer_indices=args.observer_indices,
                observer_index_policy=args.observer_index_policy,
                observer_index_count=args.observer_index_count,
                no_cache=args.no_cache,
                min_free_gb=args.min_free_gb,
                timeout_seconds=suite_timeout_seconds,
            )
        suite_results.append(synthetic)

    run_ids = _successful_run_ids(suite_results)
    if not run_ids:
        run_ids = [
            str(result["run_id"])
            for result in suite_results
            if isinstance(result.get("run_id"), str) and str(result["run_id"]).strip()
        ]
    allowlist_path = bundle_dir / "focused_run_ids.txt"
    _write_run_id_allowlist(allowlist_path, run_ids)

    _write_status(
        status_path,
        {
            "bundle_name": bundle_name,
            "status": "running",
            "stage": "evidence_refresh",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bundle_dir": str(bundle_dir),
            "runs": suite_results,
            "evidence": None,
        },
    )
    registry_sync = _sync_results_registry() if args.post_sync_results else None
    evidence = _build_evidence_bundle(
        bundle_dir=bundle_dir,
        run_id_allowlist=allowlist_path,
        control_metric_basis=args.control_metric_basis,
    )

    evidence_acceptance = _focused_evidence_acceptance(bundle_dir, claim_profile=args.claim_profile)
    proof_success = (
        bool(run_ids)
        and all(int(item["returncode"]) == 0 for item in [*suite_results, evidence])
        and bool(evidence_acceptance.get("safe_for_focused_defense", False))
    )
    marker_path = _write_marker(
        bundle_dir=bundle_dir,
        bundle_name=bundle_name,
        run_ids=run_ids,
        preferred_run_id=str(hadamard.get("run_id") or "") or None,
        hadamard_run_id=str(hadamard.get("run_id") or "") or None,
        riemannian_run_id=(str(riemannian.get("run_id") or "") or None) if riemannian else None,
        synthetic_run_id=(str(synthetic.get("run_id") or "") or None) if synthetic else None,
        activate_current=proof_success,
        evidence_acceptance=evidence_acceptance,
    )

    summary = {
        "bundle_name": bundle_name,
        "bundle_dir": str(bundle_dir),
        "allowlist_path": str(allowlist_path),
        "status_path": str(status_path),
        "marker_path": str(marker_path),
        "accepted_run_ids": run_ids,
        "rejected_runs": _failed_run_summaries(suite_results),
        "evidence_acceptance": evidence_acceptance,
        "runs": suite_results,
        "evidence": evidence,
        "registry_sync": registry_sync,
        "orchestration_contract": _focused_orchestration_contract(
            bundle_name=bundle_name,
            stage="complete",
            suite_results=suite_results,
            evidence=evidence,
            run_ids=run_ids,
            skip_riemannian=args.skip_riemannian,
            skip_synthetic=args.skip_synthetic,
        ),
    }
    summary_path = bundle_dir / "focused_proof_bundle.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_status(
        status_path,
        {
            "bundle_name": bundle_name,
            "status": "success" if proof_success else "failed",
            "stage": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bundle_dir": str(bundle_dir),
            "summary_path": str(summary_path),
            "runs": suite_results,
            "evidence": evidence,
            "evidence_acceptance": evidence_acceptance,
            "registry_sync": registry_sync,
        },
    )

    print("\n[FOCUSED] Bundle summary:")
    print(json.dumps(summary, indent=2))
    return 0 if proof_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
