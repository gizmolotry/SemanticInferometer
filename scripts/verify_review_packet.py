#!/usr/bin/env python3
"""Verify integrity invariants recorded in a focused review packet digest."""

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


def _claim_ids_from_profile_rows(rows: Any) -> List[str]:
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("claim_id"))
        for row in rows
        if isinstance(row, dict) and bool(str(row.get("claim_id") or ""))
    ]


def _expected_supported_exploratory_claims(
    *,
    claim_matrix: Dict[str, Any],
    core_claims: Sequence[str],
) -> List[str]:
    core_claim_set = {str(claim_id) for claim_id in core_claims if claim_id}
    claims = claim_matrix.get("claims")
    if not isinstance(claims, list):
        return []
    expected: List[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "")
        if (
            claim_id
            and bool(claim.get("pass"))
            and bool(claim.get("thesis_safe"))
            and claim_id not in core_claim_set
        ):
            expected.append(claim_id)
    return expected


def _list_packet_files(packet_dir: Path) -> List[str]:
    if not packet_dir.exists():
        return []
    return sorted(path.name for path in packet_dir.iterdir() if path.is_file())


def verify_packet(packet_dir: Path) -> Dict[str, Any]:
    packet_dir = Path(packet_dir)
    digest_path = packet_dir / "review_digest.json"
    if not digest_path.exists():
        return {
            "schema_version": "1.0",
            "packet_dir": str(packet_dir),
            "pass": False,
            "failure_reasons": ["review_digest_missing"],
            "verified_files": [],
        }
    try:
        digest = _load_json_object(digest_path)
    except Exception:
        return {
            "schema_version": "1.0",
            "packet_dir": str(packet_dir),
            "pass": False,
            "failure_reasons": ["review_digest_unreadable"],
            "verified_files": [],
        }
    fingerprints = digest.get("packet_file_fingerprints")
    if not isinstance(fingerprints, list):
        return {
            "schema_version": "1.0",
            "packet_dir": str(packet_dir),
            "pass": False,
            "failure_reasons": ["packet_file_fingerprints_missing"],
            "verified_files": [],
        }

    failures: List[str] = []
    verified: List[Dict[str, Any]] = []
    digest_fingerprint = {
        "name": "review_digest.json",
        "bytes": digest_path.stat().st_size,
        "sha256": _sha256(digest_path),
    }
    if digest.get("packet_schema_version") != "1.0":
        failures.append("packet_schema_version_invalid")

    packet_files = digest.get("packet_files")
    if not isinstance(packet_files, list):
        packet_files = []
        failures.append("packet_files_missing")
    packet_file_names = [str(name) for name in packet_files if str(name)]
    if len(packet_file_names) != len(set(packet_file_names)):
        failures.append("packet_files_duplicate")
    if "review_digest.json" not in packet_file_names:
        failures.append("review_digest_not_listed")
    expected_count = digest.get("packet_file_count")
    if expected_count != len(packet_file_names):
        failures.append("packet_file_count_mismatch")

    actual_packet_files = _list_packet_files(packet_dir)
    for name in sorted(set(actual_packet_files) - set(packet_file_names)):
        failures.append(f"unlisted_packet_file:{name}")
    for name in sorted(set(packet_file_names) - set(actual_packet_files)):
        failures.append(f"listed_file_missing:{name}")

    fingerprint_names: List[str] = []
    for row in fingerprints:
        if not isinstance(row, dict):
            failures.append("malformed_fingerprint_row")
            continue
        name = str(row.get("name") or "")
        if name:
            fingerprint_names.append(name)
        expected_bytes = row.get("bytes")
        expected_sha = str(row.get("sha256") or "")
        path = packet_dir / name
        if not name or not path.exists():
            failures.append(f"missing_file:{name or '<blank>'}")
            continue
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        file_pass = actual_bytes == expected_bytes and actual_sha == expected_sha
        if not file_pass:
            failures.append(f"fingerprint_mismatch:{name}")
        verified.append(
            {
                "name": name,
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
    expected_fingerprinted = set(packet_file_names) - {"review_digest.json"}
    for name in sorted(fingerprint_name_set - expected_fingerprinted):
        failures.append(f"unexpected_fingerprint:{name}")
    for name in sorted(expected_fingerprinted - fingerprint_name_set):
        failures.append(f"missing_fingerprint:{name}")

    claim_matrix_path = packet_dir / "claim_matrix.json"
    if claim_matrix_path.exists():
        try:
            claim_matrix = _load_json_object(claim_matrix_path)
        except Exception:
            failures.append("claim_matrix_unreadable")
            claim_matrix = {}
        expected_supported = _expected_supported_exploratory_claims(
            claim_matrix=claim_matrix,
            core_claims=[str(claim_id) for claim_id in digest.get("core_claims", []) or []],
        )
        actual_supported = [
            str(claim_id)
            for claim_id in digest.get("supported_exploratory_claims", []) or []
            if str(claim_id)
        ]
        if actual_supported != expected_supported:
            failures.append("supported_exploratory_claims_mismatch")

    profile_path = packet_dir / "paper_claim_profile.json"
    if profile_path.exists():
        try:
            profile = _load_json_object(profile_path)
        except Exception:
            failures.append("paper_claim_profile_unreadable")
            profile = {}
        if bool(digest.get("publication_ready", False)) != bool(profile.get("publication_ready", False)):
            failures.append("publication_ready_mismatch")
        if list(digest.get("core_claims", []) or []) != _claim_ids_from_profile_rows(profile.get("core_claims")):
            failures.append("core_claims_mismatch")
        if list(digest.get("blocked_core_claims", []) or []) != _claim_ids_from_profile_rows(
            profile.get("blocked_core_claims")
        ):
            failures.append("blocked_core_claims_mismatch")

    return {
        "schema_version": "1.0",
        "packet_dir": str(packet_dir),
        "pass": not failures,
        "failure_reasons": failures,
        "digest_fingerprint": digest_fingerprint,
        "actual_packet_file_count": len(actual_packet_files),
        "verified_file_count": len(verified),
        "verified_files": verified,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = verify_packet(args.packet_dir)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(payload.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
