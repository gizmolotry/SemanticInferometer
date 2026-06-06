"""
Master DAG + CLI orchestrator for end-to-end experiment execution.

Design goal:
- Wrap existing scripts/tools (do not rewrite pipeline logic).
- Capture durable run records (commands, logs, manifests).
- Keep rollback scope small and behavior explicit.

Scope note:
- The Airflow ablation helper remains a sidecar for CSV-level analysis.
- The canonical lab/ablation execution path lives in `core/master_ablation.py`.
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
DEFAULT_REVIEW_PACKET_ROOT = ROOT / "outputs" / "review_packets"
DEFAULT_FOCUSED_REVIEW_PACKET_DIR = DEFAULT_REVIEW_PACKET_ROOT / "focused_paper_evidence_20260522_MAX10"
DEFAULT_TRACK4_REVIEW_PACKET_DIR = DEFAULT_REVIEW_PACKET_ROOT / "track4_soft_terrain_diagnostic_20260525"
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
    lab_ablation_preset: Optional[str]
    lab_ablation_sweep: Optional[str]
    lab_ablation_output_dir: Optional[Path]
    lab_ablation_max_articles: Optional[int]
    lab_ablation_device: str
    run_procrustes: bool
    procrustes_data_dir: Optional[Path]
    viz_experiment_dir: Optional[Path]
    viz_output_html: Optional[Path]
    run_ablation: bool
    ablation_scalar_bins: int
    freeze_snapshot: bool
    freeze_snapshot_name: Optional[str]
    run_review_packet_generation: bool = False
    focused_evidence_dir: Optional[Path] = None
    track4_diagnostic_dir: Optional[Path] = None
    track4_note_path: Optional[Path] = None
    review_packet_output_root: Optional[Path] = None
    review_packet_max_files: int = 10
    run_review_packet_verification: bool = False
    focused_review_packet_dir: Optional[Path] = None
    track4_review_packet_dir: Optional[Path] = None
    review_packet_verification_output_dir: Optional[Path] = None


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


def _run_lab_ablation(config: MasterConfig, run_dir: Path) -> Optional[Dict[str, object]]:
    if not config.lab_ablation_preset and not config.lab_ablation_sweep:
        return None

    from core.master_ablation import ABLATION_PRESETS, AblationConfig, AblationRunner

    output_dir = config.lab_ablation_output_dir or (run_dir / "lab_ablation_dag")
    run_name_parts = [config.run_id]
    if config.lab_ablation_preset:
        run_name_parts.append(config.lab_ablation_preset)
    if config.lab_ablation_sweep:
        run_name_parts.append(config.lab_ablation_sweep)
    lab_config = AblationConfig(
        run_name="__".join(run_name_parts),
        output_dir=str(output_dir),
        max_articles=config.lab_ablation_max_articles or 80,
        device=config.lab_ablation_device,
    )
    if config.lab_ablation_preset:
        if config.lab_ablation_preset not in ABLATION_PRESETS:
            raise ValueError(
                f"Unknown lab ablation preset: {config.lab_ablation_preset}. "
                f"Choose from {sorted(ABLATION_PRESETS)}"
            )
        for key, value in ABLATION_PRESETS[config.lab_ablation_preset].items():
            setattr(lab_config, key, value)

    runner = AblationRunner(lab_config)
    if config.lab_ablation_sweep:
        result = runner.run_sweep(config.lab_ablation_sweep)
        result_path = output_dir / f"sweep_{config.lab_ablation_sweep}_summary.json"
        mode = "sweep"
    else:
        result = runner.run_single(lab_config)
        result_path = output_dir / lab_config.run_name / "ablation_manifest.json"
        mode = "single"

    return {
        "status": "success",
        "mode": mode,
        "preset": config.lab_ablation_preset,
        "sweep": config.lab_ablation_sweep,
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "run_status": (
            result.get("run_status")
            if isinstance(result, dict)
            else "completed"
        ),
    }


def _run_review_packet_generation(config: MasterConfig, run_dir: Path) -> Dict[str, object]:
    """Build compact focused-paper and Track 4 review packets from existing evidence."""
    from scripts.build_focused_paper_packet import (
        DEFAULT_EVIDENCE_DIR,
        build_packet as build_focused_packet,
    )
    from scripts.build_track4_diagnostic_packet import (
        DEFAULT_DIAGNOSTIC_DIR,
        DEFAULT_NOTE,
        build_packet as build_track4_packet,
    )

    output_root = config.review_packet_output_root or (run_dir / "review_packets")
    output_root.mkdir(parents=True, exist_ok=True)

    focused_evidence_dir = config.focused_evidence_dir or DEFAULT_EVIDENCE_DIR
    track4_diagnostic_dir = config.track4_diagnostic_dir or DEFAULT_DIAGNOSTIC_DIR
    track4_note_path = config.track4_note_path or DEFAULT_NOTE
    focused_packet_dir = output_root / f"{Path(focused_evidence_dir).name}_MAX{config.review_packet_max_files}"
    track4_packet_dir = output_root / f"{Path(track4_diagnostic_dir).name}_track4_MAX{config.review_packet_max_files}"
    track4_terrain_summary = Path(track4_diagnostic_dir) / "track4_terrain_semantics_diagnostics_summary.json"

    focused_summary = build_focused_packet(
        evidence_dir=Path(focused_evidence_dir),
        out_dir=focused_packet_dir,
        track4_terrain_diagnostics_summary=track4_terrain_summary if track4_terrain_summary.exists() else None,
        max_files=int(config.review_packet_max_files),
    )
    track4_summary = build_track4_packet(
        diagnostic_dir=Path(track4_diagnostic_dir),
        out_dir=track4_packet_dir,
        note_path=Path(track4_note_path),
        max_files=int(config.review_packet_max_files),
    )

    return {
        "status": "success",
        "focused_evidence_dir": str(focused_evidence_dir),
        "track4_diagnostic_dir": str(track4_diagnostic_dir),
        "track4_note_path": str(track4_note_path),
        "output_root": str(output_root),
        "max_files": int(config.review_packet_max_files),
        "focused_packet_dir": str(focused_packet_dir),
        "track4_packet_dir": str(track4_packet_dir),
        "focused_packet": focused_summary,
        "track4_packet": {
            "packet_dir": track4_summary.get("packet_dir"),
            "file_count": track4_summary.get("file_count"),
            "manifest": str(Path(track4_packet_dir) / "artifact_manifest.json"),
        },
    }


def _run_review_packet_verification(config: MasterConfig, run_dir: Path) -> Dict[str, object]:
    """Verify generated review packets without rebuilding heavyweight evidence."""
    from scripts.verify_all_review_packets import verify_all_packets

    output_dir = config.review_packet_verification_output_dir or (run_dir / "review_packet_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    focused_dir = config.focused_review_packet_dir or DEFAULT_FOCUSED_REVIEW_PACKET_DIR
    track4_dir = config.track4_review_packet_dir or DEFAULT_TRACK4_REVIEW_PACKET_DIR

    summary = verify_all_packets(
        packet_dirs=[focused_dir, track4_dir],
        root=DEFAULT_REVIEW_PACKET_ROOT,
        include_legacy=False,
        fail_on_skip=True,
    )
    summary_path = output_dir / "review_packet_verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    by_type = {str(row.get("packet_type")): row for row in summary.get("packets", []) if isinstance(row, dict)}
    focused_row = by_type.get("focused_paper_evidence", {})
    track4_row = by_type.get("track4_terrain_diagnostic", {})
    return {
        "status": "success" if bool(summary["pass"]) else "failed",
        "summary_json": str(summary_path),
        "focused_packet_dir": str(focused_dir),
        "track4_packet_dir": str(track4_dir),
        "pass": bool(summary["pass"]),
        "focused_pass": bool(focused_row.get("pass")),
        "track4_pass": bool(track4_row.get("pass")),
    }


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

    lab_ablation_result = _run_lab_ablation(config, run_dir)
    if lab_ablation_result is not None:
        record["steps"]["lab_ablation_dag"] = lab_ablation_result

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

    effective_viz_experiment_dir = config.viz_experiment_dir or discovered_experiment_dir
    record["effective_viz_experiment_dir"] = (
        str(effective_viz_experiment_dir) if effective_viz_experiment_dir else None
    )

    source_html: Optional[Path] = None
    source_log: Optional[Path] = None
    if effective_viz_experiment_dir is not None and config.viz_output_html is not None:
        source_html = config.viz_output_html
        cmd = [
            "python",
            "-m",
            "analysis.MONOLITH_VIZ",
            str(effective_viz_experiment_dir),
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

    if config.run_ablation and effective_viz_experiment_dir is not None:
        base_csv = effective_viz_experiment_dir / "MONOLITH_DATA.csv"
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
                input_dir=effective_viz_experiment_dir if effective_viz_experiment_dir else ROOT / "outputs",
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

    if config.run_review_packet_generation:
        generation_result = _run_review_packet_generation(config, run_dir)
        record["steps"]["review_packet_generation"] = generation_result
        generated_focused_dir = Path(str(generation_result["focused_packet_dir"]))
        generated_track4_dir = Path(str(generation_result["track4_packet_dir"]))
        config = MasterConfig(
            **{
                **config.__dict__,
                "focused_review_packet_dir": config.focused_review_packet_dir or generated_focused_dir,
                "track4_review_packet_dir": config.track4_review_packet_dir or generated_track4_dir,
            }
        )

    if config.run_review_packet_verification:
        review_result = _run_review_packet_verification(config, run_dir)
        record["steps"]["review_packet_verification"] = review_result
        if not bool(review_result.get("pass")):
            _write_record(run_dir / "run_record.json", record)
            raise RuntimeError("review packet verification failed")

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
            lab_ablation_preset: str = "",
            lab_ablation_sweep: str = "",
            lab_ablation_output_dir: str = "",
            lab_ablation_max_articles: int = 80,
            lab_ablation_device: str = "cpu",
            run_procrustes: bool = False,
            procrustes_data_dir: str = "",
            viz_experiment_dir: str = "",
            viz_output_html: str = "outputs/honest_matern/MONOLITH_DEBUG_latest.html",
            run_ablation: bool = True,
            ablation_scalar_bins: int = 8,
            freeze_snapshot: bool = True,
            run_review_packet_generation: bool = False,
            focused_evidence_dir: str = "",
            track4_diagnostic_dir: str = "",
            track4_note_path: str = "",
            review_packet_output_root: str = "",
            review_packet_max_files: int = 10,
            run_review_packet_verification: bool = False,
            focused_review_packet_dir: str = "",
            track4_review_packet_dir: str = "",
            review_packet_verification_output_dir: str = "",
        ) -> str:
            run_id = f"master_{_utc_now_slug()}"
            suite_args = ["--limit", str(limit), "--mode", mode]
            cfg = MasterConfig(
                run_id=run_id,
                record_root=DEFAULT_RECORD_ROOT,
                suite_args=suite_args,
                lab_ablation_preset=lab_ablation_preset or None,
                lab_ablation_sweep=lab_ablation_sweep or None,
                lab_ablation_output_dir=Path(lab_ablation_output_dir) if lab_ablation_output_dir else None,
                lab_ablation_max_articles=lab_ablation_max_articles,
                lab_ablation_device=lab_ablation_device,
                run_procrustes=run_procrustes,
                procrustes_data_dir=Path(procrustes_data_dir) if procrustes_data_dir else None,
                viz_experiment_dir=Path(viz_experiment_dir) if viz_experiment_dir else None,
                viz_output_html=Path(viz_output_html) if viz_output_html else None,
                run_ablation=run_ablation,
                ablation_scalar_bins=ablation_scalar_bins,
                freeze_snapshot=freeze_snapshot,
                freeze_snapshot_name=None,
                run_review_packet_generation=run_review_packet_generation,
                focused_evidence_dir=Path(focused_evidence_dir) if focused_evidence_dir else None,
                track4_diagnostic_dir=Path(track4_diagnostic_dir) if track4_diagnostic_dir else None,
                track4_note_path=Path(track4_note_path) if track4_note_path else None,
                review_packet_output_root=Path(review_packet_output_root) if review_packet_output_root else None,
                review_packet_max_files=review_packet_max_files,
                run_review_packet_verification=run_review_packet_verification,
                focused_review_packet_dir=Path(focused_review_packet_dir) if focused_review_packet_dir else None,
                track4_review_packet_dir=Path(track4_review_packet_dir) if track4_review_packet_dir else None,
                review_packet_verification_output_dir=(
                    Path(review_packet_verification_output_dir)
                    if review_packet_verification_output_dir
                    else None
                ),
            )
            out = run_master_orchestration(cfg)
            return str(out)

        run_pipeline()

    monolith_master_orchestrator_v1_dag = monolith_master_orchestrator_v1()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master orchestration wrapper for suite + viz + ablation + snapshot.")
    parser.add_argument("--run-id", default=f"master_{_utc_now_slug()}")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--suite-args", nargs="*", default=None)
    parser.add_argument(
        "--suite-arg",
        action="append",
        default=[],
        help="Repeatable child argument for run_full_experiment_suite.py, e.g. --suite-arg=--limit --suite-arg=60.",
    )
    parser.add_argument("--lab-ablation-preset", type=str, default=None)
    parser.add_argument("--lab-ablation-sweep", type=str, default=None)
    parser.add_argument("--lab-ablation-output-dir", type=Path, default=None)
    parser.add_argument("--lab-ablation-max-articles", type=int, default=80)
    parser.add_argument("--lab-ablation-device", type=str, default="cpu")
    parser.add_argument("--run-procrustes", action="store_true")
    parser.add_argument("--procrustes-data-dir", type=Path, default=None)
    parser.add_argument("--viz-experiment-dir", type=Path, default=None)
    parser.add_argument("--viz-output-html", type=Path, default=Path("outputs/honest_matern/MONOLITH_DEBUG_latest.html"))
    parser.add_argument("--run-ablation", action="store_true", default=True)
    parser.add_argument("--no-run-ablation", action="store_false", dest="run_ablation")
    parser.add_argument("--ablation-scalar-bins", type=int, default=8)
    parser.add_argument("--freeze-snapshot", action="store_true", default=True)
    parser.add_argument("--no-freeze-snapshot", action="store_false", dest="freeze_snapshot")
    parser.add_argument("--freeze-snapshot-name", type=str, default=None)
    parser.add_argument("--run-review-packet-generation", action="store_true")
    parser.add_argument("--focused-evidence-dir", type=Path, default=None)
    parser.add_argument("--track4-diagnostic-dir", type=Path, default=None)
    parser.add_argument("--track4-note-path", type=Path, default=None)
    parser.add_argument("--review-packet-output-root", type=Path, default=None)
    parser.add_argument("--review-packet-max-files", type=int, default=10)
    parser.add_argument("--run-review-packet-verification", action="store_true")
    parser.add_argument("--focused-review-packet-dir", type=Path, default=None)
    parser.add_argument("--track4-review-packet-dir", type=Path, default=None)
    parser.add_argument("--review-packet-verification-output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    suite_args = args.suite_arg or args.suite_args or ["--limit", "80", "--mode", "enhanced"]
    cfg = MasterConfig(
        run_id=args.run_id,
        record_root=args.record_root,
        suite_args=suite_args,
        lab_ablation_preset=args.lab_ablation_preset,
        lab_ablation_sweep=args.lab_ablation_sweep,
        lab_ablation_output_dir=args.lab_ablation_output_dir,
        lab_ablation_max_articles=args.lab_ablation_max_articles,
        lab_ablation_device=args.lab_ablation_device,
        run_procrustes=args.run_procrustes,
        procrustes_data_dir=args.procrustes_data_dir,
        viz_experiment_dir=args.viz_experiment_dir,
        viz_output_html=args.viz_output_html,
        run_ablation=args.run_ablation,
        ablation_scalar_bins=args.ablation_scalar_bins,
        freeze_snapshot=args.freeze_snapshot,
        freeze_snapshot_name=args.freeze_snapshot_name,
        run_review_packet_generation=args.run_review_packet_generation,
        focused_evidence_dir=args.focused_evidence_dir,
        track4_diagnostic_dir=args.track4_diagnostic_dir,
        track4_note_path=args.track4_note_path,
        review_packet_output_root=args.review_packet_output_root,
        review_packet_max_files=args.review_packet_max_files,
        run_review_packet_verification=args.run_review_packet_verification,
        focused_review_packet_dir=args.focused_review_packet_dir,
        track4_review_packet_dir=args.track4_review_packet_dir,
        review_packet_verification_output_dir=args.review_packet_verification_output_dir,
    )
    out = run_master_orchestration(cfg)
    print(f"[master-orchestrator] run record: {out}")
