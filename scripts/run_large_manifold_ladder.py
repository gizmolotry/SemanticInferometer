#!/usr/bin/env python3
"""Launch run_experiments.py through an increasing set of article limits."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_experiments.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "large_manifold_ladder"
DEFAULT_LIMITS = [50, 100, 250, 500, 1000]
MANIFEST_NAME = "large_manifold_ladder_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_limits(values: Sequence[int]) -> List[int]:
    limits: List[int] = []
    seen = set()
    for value in values:
        if value <= 0:
            raise argparse.ArgumentTypeError("--limits values must be positive integers")
        if value not in seen:
            limits.append(value)
            seen.add(value)
    if not limits:
        raise argparse.ArgumentTypeError("--limits must include at least one value")
    return limits


def _expand_extra_flags(values: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for value in values:
        expanded.extend(shlex.split(value, posix=(os.name != "nt")))
    return expanded


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage_slug(limit: int) -> str:
    return f"limit_{limit}"


def _build_stage_command(args: argparse.Namespace, limit: int, stage_output_root: Path) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--corpus",
        args.corpus,
        "--mode",
        args.mode,
        "--seeds",
        str(args.seed),
        "--limit",
        str(limit),
        "--output-root",
        str(stage_output_root),
    ]
    if args.kernel:
        command.extend(["--kernel-type", args.kernel])
    if args.nli_cache_dir:
        cache_path = Path(args.nli_cache_dir).resolve() / f"{_stage_slug(limit)}_nli_cache.pt"
        command.extend(["--nli-cache-path", str(cache_path)])
    command.extend(args.extra_flags)
    return command


def _write_log(
    log_path: Path,
    *,
    command: Sequence[str],
    start: str,
    end: str,
    status: str,
    returncode: int | None,
    stdout: str = "",
    stderr: str = "",
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        f"start: {start}",
        f"end: {end}",
        f"status: {status}",
        f"returncode: {returncode}",
        "command:",
        " ".join(shlex.quote(part) for part in command),
        "",
        "stdout:",
        stdout,
        "",
        "stderr:",
        stderr,
        "",
    ]
    log_path.write_text("\n".join(sections), encoding="utf-8")


def _append_log_footer(
    log_path: Path,
    *,
    end: str,
    status: str,
    returncode: int | None,
) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("=" * 80)
        handle.write("\n")
        handle.write(f"end: {end}\n")
        handle.write(f"status: {status}\n")
        handle.write(f"returncode: {returncode}\n")


def _run_stage(
    *,
    args: argparse.Namespace,
    limit: int,
    stage_index: int,
    output_root: Path,
) -> dict:
    stage_name = _stage_slug(limit)
    stage_output_root = output_root / "stages" / stage_name
    log_path = output_root / "logs" / f"{stage_name}.log"
    command = _build_stage_command(args, limit, stage_output_root)
    start = _utc_now()

    if args.dry_run:
        end = _utc_now()
        _write_log(
            log_path,
            command=command,
            start=start,
            end=end,
            status="dry_run",
            returncode=0,
            stdout="DRY RUN: command was not executed.\n",
        )
        return {
            "stage": stage_index,
            "limit": limit,
            "start": start,
            "end": end,
            "status": "dry_run",
            "returncode": 0,
            "command": command,
            "output_root": str(stage_output_root),
            "log_path": str(log_path),
        }

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(f"start: {start}\n")
            handle.write("status: running\n")
            handle.write("command:\n")
            handle.write(" ".join(shlex.quote(part) for part in command))
            handle.write("\n\n")
            handle.write("combined stdout/stderr:\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        end = _utc_now()
        status = "succeeded" if completed.returncode == 0 else "failed"
        _append_log_footer(
            log_path,
            end=end,
            status=status,
            returncode=completed.returncode,
        )
        return {
            "stage": stage_index,
            "limit": limit,
            "start": start,
            "end": end,
            "status": status,
            "returncode": completed.returncode,
            "command": command,
            "output_root": str(stage_output_root),
            "log_path": str(log_path),
        }
    except OSError as exc:
        end = _utc_now()
        _write_log(
            log_path,
            command=command,
            start=start,
            end=end,
            status="error",
            returncode=None,
            stderr=f"{type(exc).__name__}: {exc}\n",
        )
        return {
            "stage": stage_index,
            "limit": limit,
            "start": start,
            "end": end,
            "status": "error",
            "returncode": None,
            "command": command,
            "output_root": str(stage_output_root),
            "log_path": str(log_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run run_experiments.py through staged --limit values, stopping on "
            "the first failed stage."
        )
    )
    parser.add_argument("--corpus", default="real", help="Corpus passed to run_experiments.py")
    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default=DEFAULT_LIMITS,
        help=f"Article limits to run in order (default: {' '.join(map(str, DEFAULT_LIMITS))})",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Optional kernel passed as run_experiments.py --kernel-type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed passed as the sole --seeds value")
    parser.add_argument(
        "--mode",
        default="cls_dirichlet",
        help="Experiment mode passed to run_experiments.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for stage outputs, logs, and the JSON manifest",
    )
    parser.add_argument(
        "--nli-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for per-stage NLI caches consumed by run_experiments.py",
    )
    parser.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help=(
            "Additional run_experiments.py flag(s). Repeat as needed. For values "
            "starting with '-', use --extra-flag=--flag or quote a flag/value pair."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and logs without launching run_experiments.py",
    )
    args = parser.parse_args()
    args.limits = _parse_limits(args.limits)
    args.extra_flags = _expand_extra_flags(args.extra_flag)
    args.output_root = args.output_root.resolve()
    if args.nli_cache_dir is not None:
        args.nli_cache_dir = args.nli_cache_dir.resolve()
    return args


def main() -> int:
    args = _parse_args()
    if not RUNNER.exists():
        print(f"Missing runner: {RUNNER}", file=sys.stderr)
        return 2

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_NAME
    manifest = {
        "script": str(Path(__file__).resolve()),
        "runner": str(RUNNER),
        "repo_root": str(REPO_ROOT),
        "created_at": _utc_now(),
        "updated_at": None,
        "dry_run": bool(args.dry_run),
        "status": "running",
        "corpus": args.corpus,
        "limits": args.limits,
        "kernel": args.kernel,
        "seed": args.seed,
        "mode": args.mode,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "nli_cache_dir": str(args.nli_cache_dir) if args.nli_cache_dir else None,
        "extra_flags": args.extra_flags,
        "stages": [],
    }
    _write_json(manifest_path, manifest)

    exit_code = 0
    for index, limit in enumerate(args.limits, start=1):
        stage = _run_stage(args=args, limit=limit, stage_index=index, output_root=output_root)
        manifest["stages"].append(stage)
        manifest["updated_at"] = _utc_now()
        if stage["status"] not in {"succeeded", "dry_run"}:
            manifest["status"] = "failed"
            exit_code = int(stage["returncode"] or 1)
            _write_json(manifest_path, manifest)
            print(f"Stage {index} failed at limit={limit}; see {stage['log_path']}", file=sys.stderr)
            return exit_code
        _write_json(manifest_path, manifest)

    manifest["status"] = "dry_run" if args.dry_run else "succeeded"
    manifest["updated_at"] = _utc_now()
    _write_json(manifest_path, manifest)
    print(f"Manifest written to {manifest_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
