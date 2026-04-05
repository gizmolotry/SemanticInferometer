#!/usr/bin/env python3
"""Report oversized Python modules to support incremental modularization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 1200


def py_files():
    for path in ROOT.rglob("*.py"):
        if any(part in {"archive", "outputs", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        yield path


def main() -> int:
    oversized = []
    for path in py_files():
        lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
        if lines >= THRESHOLD:
            oversized.append((lines, path))

    oversized.sort(reverse=True)
    if not oversized:
        print("No oversized modules detected.")
        return 0

    print(f"Oversized modules (>= {THRESHOLD} lines):")
    for lines, path in oversized:
        rel = path.relative_to(ROOT).as_posix()
        print(f"- {rel}: {lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
