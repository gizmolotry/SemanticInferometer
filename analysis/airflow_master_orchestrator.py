"""
Master DAG + CLI orchestrator for end-to-end experiment execution.

Design goal:
- Wrap existing scripts/tools (do not rewrite pipeline logic).
- Capture durable run records (commands, logs, manifests).
- Keep rollback scope small and behavior explicit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from airflow.decorators import dag, task  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dag = None
    task = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = ROOT / "analysis" / "orchestration_records"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from analysis.airflow_ablation_orchestrator import run_ablation_matrix
    from analysis.freeze_viz_tuple import FreezeConfig, freeze_run_snapshot, verify_snapshot
except Exception:  # pragma: no cover - allow direct script invocation fallback
    from airflow_ablation_orchestrator import run_ablation_matrix  # type: ignore
    from freeze_viz_tuple import FreezeConfig, freeze_run_snapshot, verify_snapshot  # type: ignore


@dataclass(frozen=True)
class MasterConfig:
    run_id: str
    record_root: Path
    suite_args: List[str]
    run_procrustes: bool
    procrustes_data_dir: Optional[Path]
    viz_experiment_dir: Optional[Path]
    viz_output_html: Optional[Path]
    run_ablation: bool
    ablation_scalar_bins: int
    freeze_snapshot: bool
    freeze_snapshot_name: Optional[str]


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run_command(
    command: List[str],
    *,
    cwd: Path,
    log_path: Path,
) -> Dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    payload = [
        f"$ {' '.join(command)}",
        "",
        "=== STDOUT ===",
        proc.stdout or "",
        "",
        "=== STDERR ===",
        proc.stderr or "",
    ]
    log_path.write_text("\n".join(payload), encoding="utf-8")
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": int(proc.returncode),
        "log": str(log_path),
    }


def _snapshot_experiment_dirs() -> Dict[str, List[str]]:
    roots = [ROOT / "outputs" / "experiments" / "runs", ROOT]
    snapshot: Dict[str, List[str]] = {}
    for r in roots:
        if not r.exists():
            continue
        dirs = sorted([p.name for p in r.glob("experiments_*") if p.is_dir()])
        snapshot[str(r)] = dirs
    return snapshot


def _detect_new_experiment_dir(
    before: Dict[str, List[str]],
    after: Dict[str, List[str]],
) -> Optional[Path]:
    candidates: List[Path] = []
    for root_str, dirs_after in after.items():
        dirs_before = set(before.get(root_str, []))
        for d in dirs_after:
            if d not in dirs_before:
                candidates.append(Path(root_str) / d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _write_record(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_master_orchestration(config: MasterConfig) -> Path:
    run_dir = config.record_root / config.run_id
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, object] = {
        "run_id": config.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    before = _snapshot_experiment_dirs()
    suite_cmd = ["python", "run_full_experiment_suite.py", *config.suite_args]
    suite_result = _run_command(
        suite_cmd,
        cwd=ROOT,
        log_path=logs_dir / "01_run_full_experiment_suite.log",
    )
    record["steps"]["run_full_experiment_suite"] = suite_result
    if int(suite_result["exit_code"]) != 0:
        _write_record(run_dir / "run_record.json", record)
        raise RuntimeError("run_full_experiment_suite.py failed")

    after = _snapshot_experiment_dirs()
    discovered_experiment_dir = _detect_new_experiment_dir(before, after)
    record["discovered_experiment_dir"] = (
        str(discovered_experiment_dir) if discovered_experiment_dir else None
    )

    if config.run_procrustes and config.procrustes_data_dir is not None:
        out_dir = config.procrustes_data_dir / "analysis_outputs"
        cmd = [
            "python",
            "procrustes_alignment.py",
            "--data-dir",
            str(config.procrustes_data_dir),
            "--output-dir",
            str(out_dir),
        ]
        result = _run_command(
            cmd,
            cwd=ROOT,
            log_path=logs_dir / "02_procrustes_alignment.log",
        )
        record["steps"]["procrustes_alignment"] = result
        if int(result["exit_code"]) != 0:
            _write_record(run_dir / "run_record.json", record)
            raise RuntimeError("procrustes_alignment.py failed")

    source_html: Optional[Path] = None
    source_log: Optional[Path] = None
    if config.viz_experiment_dir is not None and config.viz_output_html is not None:
        source_html = config.viz_output_html
        cmd = [
            "python",
            "-m",
            "analysis.MONOLITH_VIZ",
            str(config.viz_experiment_dir),
            "-o",
            str(config.viz_output_html),
        ]
        result = _run_command(
            cmd,
            cwd=ROOT,
            log_path=logs_dir / "03_monolith_viz.log",
        )
        record["steps"]["monolith_viz"] = result
        if int(result["exit_code"]) != 0:
            _write_record(run_dir / "run_record.json", record)
            raise RuntimeError("analysis.MONOLITH_VIZ failed")
        source_log = Path(result["log"])

    if config.run_ablation and config.viz_experiment_dir is not None:
        base_csv = config.viz_experiment_dir / "MONOLITH_DATA.csv"
        if not base_csv.exists():
            _write_record(run_dir / "run_record.json", record)
            raise FileNotFoundError(f"Ablation base csv not found: {base_csv}")
        out = run_ablation_matrix(
            base_csv=base_csv,
            output_dir=run_dir / "ablation_outputs",
            scalar_bins=config.ablation_scalar_bins,
        )
        record["steps"]["ablation_matrix"] = {"summary_json": str(out)}

    if config.freeze_snapshot and source_html is not None and source_log is not None:
        snapshot_name = config.freeze_snapshot_name or f"{config.run_id}_snapshot"
        snap_dir = freeze_run_snapshot(
            FreezeConfig(
                snapshot_name=snapshot_name,
                snapshot_root=ROOT / "analysis" / "locked_runs",
                source_html=source_html,
                source_log=source_log,
                source_viz_code=ROOT / "analysis" / "MONOLITH_VIZ.py",
                source_viz_code_git=ROOT / "analysis" / "MONOLITH_VIZ.py",
                input_dir=config.viz_experiment_dir if config.viz_experiment_dir else ROOT / "outputs",
                viz_code_commit="master_orchestrator",
                command="python -m analysis.MONOLITH_VIZ ...",
            )
        )
        manifest = snap_dir / "RUN_MANIFEST.json"
        verify_snapshot(manifest)
        record["steps"]["freeze_snapshot"] = {
            "snapshot_dir": str(snap_dir),
            "manifest": str(manifest),
        }

    _write_record(run_dir / "run_record.json", record)
    return run_dir / "run_record.json"


if dag is not None and task is not None:  # pragma: no cover - exercised in Airflow
    @dag(
        dag_id="monolith_master_orchestrator_v1",
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["monolith", "orchestrator", "ablation", "records"],
    )
    def monolith_master_orchestrator_v1():
        @task
        def run_pipeline(
            limit: int = 80,
            mode: str = "enhanced",
            run_procrustes: bool = False,
            procrustes_data_dir: str = "",
            viz_experiment_dir: str = "outputs/honest_matern",
            viz_output_html: str = "outputs/honest_matern/MONOLITH_DEBUG_latest.html",
            run_ablation: bool = True,
            ablation_scalar_bins: int = 8,
            freeze_snapshot: bool = True,
        ) -> str:
            run_id = f"master_{_utc_now_slug()}"
            suite_args = ["--limit", str(limit), "--mode", mode]
            cfg = MasterConfig(
                run_id=run_id,
                record_root=DEFAULT_RECORD_ROOT,
                suite_args=suite_args,
                run_procrustes=run_procrustes,
                procrustes_data_dir=Path(procrustes_data_dir) if procrustes_data_dir else None,
                viz_experiment_dir=Path(viz_experiment_dir) if viz_experiment_dir else None,
                viz_output_html=Path(viz_output_html) if viz_output_html else None,
                run_ablation=run_ablation,
                ablation_scalar_bins=ablation_scalar_bins,
                freeze_snapshot=freeze_snapshot,
                freeze_snapshot_name=None,
            )
            out = run_master_orchestration(cfg)
            return str(out)

        run_pipeline()

    monolith_master_orchestrator_v1_dag = monolith_master_orchestrator_v1()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master orchestration wrapper for suite + viz + ablation + snapshot.")
    parser.add_argument("--run-id", default=f"master_{_utc_now_slug()}")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--suite-args", nargs="*", default=["--limit", "80", "--mode", "enhanced"])
    parser.add_argument("--run-procrustes", action="store_true")
    parser.add_argument("--procrustes-data-dir", type=Path, default=None)
    parser.add_argument("--viz-experiment-dir", type=Path, default=Path("outputs/honest_matern"))
    parser.add_argument("--viz-output-html", type=Path, default=Path("outputs/honest_matern/MONOLITH_DEBUG_latest.html"))
    parser.add_argument("--run-ablation", action="store_true", default=True)
    parser.add_argument("--no-run-ablation", action="store_false", dest="run_ablation")
    parser.add_argument("--ablation-scalar-bins", type=int, default=8)
    parser.add_argument("--freeze-snapshot", action="store_true", default=True)
    parser.add_argument("--no-freeze-snapshot", action="store_false", dest="freeze_snapshot")
    parser.add_argument("--freeze-snapshot-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = MasterConfig(
        run_id=args.run_id,
        record_root=args.record_root,
        suite_args=args.suite_args,
        run_procrustes=args.run_procrustes,
        procrustes_data_dir=args.procrustes_data_dir,
        viz_experiment_dir=args.viz_experiment_dir,
        viz_output_html=args.viz_output_html,
        run_ablation=args.run_ablation,
        ablation_scalar_bins=args.ablation_scalar_bins,
        freeze_snapshot=args.freeze_snapshot,
        freeze_snapshot_name=args.freeze_snapshot_name,
    )
    out = run_master_orchestration(cfg)
    print(f"[master-orchestrator] run record: {out}")
