#!/usr/bin/env python3
"""Verify integrity invariants for a Track 4 diagnostics packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _packet_file_relpaths(packet_dir: Path) -> List[str]:
    if not packet_dir.exists():
        return []
    return sorted(
        str(path.relative_to(packet_dir)).replace("\\", "/")
        for path in packet_dir.rglob("*")
        if path.is_file()
    )


def _safe_packet_path(packet_dir: Path, relpath: str) -> Optional[Path]:
    if Path(relpath).is_absolute():
        return None
    root = packet_dir.resolve()
    path = (packet_dir / relpath).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def _manifest_failure(packet_dir: Path, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "packet_dir": str(packet_dir),
        "pass": False,
        "failure_reasons": [reason],
        "actual_packet_file_count": 0,
        "verified_copied_file_count": 0,
        "verified_copied_files": [],
    }


def verify_packet(packet_dir: Path, max_files: Optional[int] = None) -> Dict[str, Any]:
    packet_dir = Path(packet_dir)
    manifest_path = packet_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return _manifest_failure(packet_dir, "artifact_manifest_missing")
    try:
        manifest = _load_json_object(manifest_path)
    except Exception:
        return _manifest_failure(packet_dir, "artifact_manifest_unreadable")

    failures: List[str] = []
    verified: List[Dict[str, Any]] = []
    actual_packet_files = _packet_file_relpaths(packet_dir)
    actual_packet_file_count = len(actual_packet_files)
    manifest_fingerprint = {
        "name": "artifact_manifest.json",
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
    }

    if manifest.get("schema_version") != "1.0":
        failures.append("schema_version_invalid")
    if manifest.get("file_count") != actual_packet_file_count:
        failures.append("file_count_mismatch")

    packet_files = manifest.get("packet_files")
    if not isinstance(packet_files, list):
        failures.append("packet_files_missing")
        packet_file_names: List[str] = []
    else:
        packet_file_names = [str(name) for name in packet_files if str(name)]
        if len(packet_file_names) != len(set(packet_file_names)):
            failures.append("packet_files_duplicate")
        if "artifact_manifest.json" not in packet_file_names:
            failures.append("artifact_manifest_not_listed")
        for name in sorted(set(actual_packet_files) - set(packet_file_names)):
            failures.append(f"unlisted_packet_file:{name}")
        for name in sorted(set(packet_file_names) - set(actual_packet_files)):
            failures.append(f"listed_file_missing:{name}")

    cap = max_files if max_files is not None else manifest.get("max_files")
    if isinstance(cap, int):
        if actual_packet_file_count > cap:
            failures.append("packet_file_count_exceeds_max")
    else:
        failures.append("max_files_missing")
        cap = None

    copied_artifacts = manifest.get("copied_artifacts")
    if not isinstance(copied_artifacts, list):
        failures.append("copied_artifacts_missing")
        copied_artifacts = []

    fingerprints = manifest.get("packet_file_fingerprints")
    if not isinstance(fingerprints, list):
        failures.append("packet_file_fingerprints_missing")
        fingerprints = []
    fingerprint_names: List[str] = []
    verified_files: List[Dict[str, Any]] = []
    expected_fingerprinted = set(actual_packet_files) - {"artifact_manifest.json"}
    for row in fingerprints:
        if not isinstance(row, dict):
            failures.append("malformed_packet_fingerprint")
            continue
        relpath = str(row.get("name") or "")
        if relpath:
            fingerprint_names.append(relpath)
        path = _safe_packet_path(packet_dir, relpath)
        if not relpath or path is None:
            failures.append(f"packet_fingerprint_path_invalid:{relpath or '<blank>'}")
            continue
        expected_bytes = row.get("bytes")
        expected_sha = str(row.get("sha256") or "")
        if not path.exists():
            failures.append(f"missing_file:{relpath}")
            verified_files.append(
                {
                    "name": relpath,
                    "pass": False,
                    "expected_bytes": expected_bytes,
                    "actual_bytes": None,
                    "expected_sha256": expected_sha,
                    "actual_sha256": "",
                }
            )
            continue
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        file_pass = actual_bytes == expected_bytes and actual_sha == expected_sha
        if not file_pass:
            failures.append(f"fingerprint_mismatch:{relpath}")
        verified_files.append(
            {
                "name": relpath,
                "pass": file_pass,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
        )
    if len(fingerprint_names) != len(set(fingerprint_names)):
        failures.append("packet_file_fingerprints_duplicate")
    fingerprint_name_set = set(fingerprint_names)
    for name in sorted(fingerprint_name_set - expected_fingerprinted):
        failures.append(f"unexpected_fingerprint:{name}")
    for name in sorted(expected_fingerprinted - fingerprint_name_set):
        failures.append(f"missing_fingerprint:{name}")

    copied_relpaths: List[str] = []
    for row in copied_artifacts:
        if not isinstance(row, dict):
            failures.append("malformed_copied_artifact")
            continue
        name = str(row.get("name") or "")
        relpath = str(row.get("packet_relpath") or "")
        status = str(row.get("status") or "")
        if status != "copied":
            failures.append(f"copied_artifact_not_copied:{name or '<blank>'}")
            continue
        if not relpath:
            failures.append(f"copied_artifact_relpath_missing:{name or '<blank>'}")
            continue
        copied_relpaths.append(relpath)

        path = _safe_packet_path(packet_dir, relpath)
        if path is None:
            failures.append(f"copied_artifact_relpath_outside_packet:{relpath}")
            continue

        expected_bytes = row.get("bytes")
        expected_sha = str(row.get("sha256") or "")
        if expected_bytes is None or not expected_sha:
            failures.append(f"copied_file_fingerprint_missing:{relpath}")
            continue
        if not path.exists():
            failures.append(f"copied_file_missing:{relpath}")
            verified.append(
                {
                    "name": name,
                    "packet_relpath": relpath,
                    "pass": False,
                    "expected_bytes": expected_bytes,
                    "actual_bytes": None,
                    "expected_sha256": expected_sha,
                    "actual_sha256": "",
                }
            )
            continue

        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        file_pass = actual_bytes == expected_bytes and actual_sha == expected_sha
        if not file_pass:
            failures.append(f"copied_file_tampered:{relpath}")
        verified.append(
            {
                "name": name,
                "packet_relpath": relpath,
                "pass": file_pass,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
        )

    if len(copied_relpaths) != len(set(copied_relpaths)):
        failures.append("copied_artifacts_duplicate")

    return {
        "schema_version": "1.0",
        "packet_dir": str(packet_dir),
        "pass": not failures,
        "failure_reasons": failures,
        "manifest_fingerprint": manifest_fingerprint,
        "actual_packet_file_count": actual_packet_file_count,
        "max_files": cap,
        "verified_file_count": len(verified_files),
        "verified_files": verified_files,
        "verified_copied_file_count": len(verified),
        "verified_copied_files": verified,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = verify_packet(args.packet_dir, max_files=args.max_files)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(payload.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
