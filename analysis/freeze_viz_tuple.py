from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REQUIRED_INPUTS = [
    "MONOLITH_DATA.csv",
    "phantom_verdicts.json",
    "walker_states.json",
    "spectral_probe_magnitudes.npy",
    "spectral_u_axis.npy",
    "features.npy",
    "validation.json",
    "run_meta.json",
    "variance_tracking.json",
    "walker_paths.npz",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path, relative_to: Path) -> Dict[str, object]:
    return {
        "file": str(path.relative_to(relative_to)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


@dataclass
class FreezeConfig:
    snapshot_name: str
    snapshot_root: Path
    source_html: Path
    source_log: Path
    source_viz_code: Path
    source_viz_code_git: Path
    input_dir: Path
    viz_code_commit: str
    command: str


def freeze_run_snapshot(cfg: FreezeConfig) -> Path:
    snap_dir = cfg.snapshot_root / cfg.snapshot_name
    snap_dir.mkdir(parents=True, exist_ok=True)

    html_dst = snap_dir / cfg.source_html.name
    log_dst = snap_dir / cfg.source_log.name
    viz_dst = snap_dir / "MONOLITH_VIZ_runtime.py"
    viz_git_dst = snap_dir / "MONOLITH_VIZ_git.py"

    html_dst.write_bytes(cfg.source_html.read_bytes())
    log_dst.write_bytes(cfg.source_log.read_bytes())
    viz_dst.write_bytes(cfg.source_viz_code.read_bytes())
    viz_git_dst.write_bytes(cfg.source_viz_code_git.read_bytes())

    input_records: List[Dict[str, object]] = []
    for name in REQUIRED_INPUTS:
        p = cfg.input_dir / name
        if p.exists():
            input_records.append(_file_record(p, cfg.input_dir))

    manifest = {
        "snapshot_name": cfg.snapshot_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "viz_code_commit": cfg.viz_code_commit,
        "command": cfg.command,
        "source": {
            "html": str(cfg.source_html).replace("\\", "/"),
            "log": str(cfg.source_log).replace("\\", "/"),
            "input_dir": str(cfg.input_dir).replace("\\", "/"),
        },
        "snapshot_files": {
            "html": _file_record(html_dst, snap_dir),
            "log": _file_record(log_dst, snap_dir),
            "viz_runtime": _file_record(viz_dst, snap_dir),
            "viz_git": _file_record(viz_git_dst, snap_dir),
        },
        "inputs": input_records,
    }

    (snap_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (snap_dir / "INPUT_HASHES.json").write_text(json.dumps(input_records, indent=2), encoding="utf-8")
    return snap_dir


def verify_snapshot(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snap_dir = manifest_path.parent

    for key in ("html", "log", "viz_runtime", "viz_git"):
        rec = manifest["snapshot_files"][key]
        p = snap_dir / rec["file"]
        if not p.exists():
            raise FileNotFoundError(f"Missing snapshot file: {p}")
        actual = _sha256(p)
        if actual != rec["sha256"]:
            raise ValueError(f"Hash mismatch for {p.name}: expected {rec['sha256']}, got {actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze and verify a known-good MONOLITH run snapshot.")
    parser.add_argument("--snapshot-name", required=True)
    parser.add_argument("--snapshot-root", type=Path, default=Path("analysis/locked_runs"))
    parser.add_argument("--source-html", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--source-viz-code", type=Path, required=True)
    parser.add_argument("--source-viz-code-git", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--viz-code-commit", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_only:
        if args.manifest is None:
            raise ValueError("--verify-only requires --manifest")
        verify_snapshot(args.manifest)
        print(f"[freeze] verified snapshot manifest: {args.manifest}")
    else:
        snap = freeze_run_snapshot(
            FreezeConfig(
                snapshot_name=args.snapshot_name,
                snapshot_root=args.snapshot_root,
                source_html=args.source_html,
                source_log=args.source_log,
                source_viz_code=args.source_viz_code,
                source_viz_code_git=args.source_viz_code_git,
                input_dir=args.input_dir,
                viz_code_commit=args.viz_code_commit,
                command=args.command,
            )
        )
        manifest_path = snap / "RUN_MANIFEST.json"
        verify_snapshot(manifest_path)
        print(f"[freeze] snapshot written and verified: {snap}")
