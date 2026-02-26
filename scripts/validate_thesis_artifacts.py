#!/usr/bin/env python3
"""Validate baseline thesis/release scaffolding artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_paths(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not is_nonempty_file(path):
            errors.append(f"Missing or empty: {path.as_posix()}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate required thesis/release artifacts for a release folder."
    )
    parser.add_argument(
        "--release-dir",
        default="thesis_release/2026-02-26",
        help="Release directory to validate (default: %(default)s)",
    )
    args = parser.parse_args()

    release_dir = Path(args.release_dir)

    required = [
        Path(".gitignore"),
        Path("MAINTAINABILITY_PLAN.md"),
        release_dir / "README.md",
        Path("notes/physarum_walk_update_report.txt"),
        Path("notes/test_write_shell.txt"),
    ]

    errors = validate_paths(required)

    root_should_be_absent = [
        Path("physarum_walk_update_report.txt"),
        Path("test_write_shell.txt"),
    ]
    for root_file in root_should_be_absent:
        if root_file.exists():
            errors.append(
                f"Expected moved out of repository root: {root_file.as_posix()}"
            )

    if errors:
        print("VALIDATION FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("VALIDATION PASSED")
    for path in required:
        print(f"- OK: {path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
