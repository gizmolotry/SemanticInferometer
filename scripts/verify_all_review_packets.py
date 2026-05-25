#!/usr/bin/env python3
"""Verify all supported review packet types with one command.

The focused paper packet and the Track 4 diagnostic packet intentionally have
different manifests. This wrapper routes each packet to its native verifier and
emits one aggregate result without adding files inside capped packet folders.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_ROOT = REPO_ROOT / "outputs" / "review_packets"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _discover_packet_dirs(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _packet_kind(packet_dir: Path, *, include_legacy: bool = False) -> Dict[str, Any]:
    packet_dir = Path(packet_dir)
    if (packet_dir / "review_digest.json").exists():
        return {
            "packet_type": "focused_paper_evidence",
            "verifier": "verify_review_packet",
            "sentinel": "review_digest.json",
            "supported": True,
        }
    manifest_path = packet_dir / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = _load_json_object(manifest_path)
        if isinstance(manifest.get("packet_file_fingerprints"), list):
            return {
                "packet_type": "track4_terrain_diagnostic",
                "verifier": "verify_track4_diagnostic_packet",
                "sentinel": "artifact_manifest.json",
                "supported": True,
            }
        if include_legacy:
            return {
                "packet_type": "legacy_artifact_manifest",
                "verifier": "verify_track4_diagnostic_packet",
                "sentinel": "artifact_manifest.json",
                "supported": True,
                "legacy": True,
            }
        return {
            "packet_type": "unsupported_artifact_manifest",
            "verifier": None,
            "sentinel": "artifact_manifest.json",
            "supported": False,
            "skip_reason": "unsupported_artifact_manifest_schema",
        }
    return {
        "packet_type": "unknown",
        "verifier": None,
        "sentinel": "",
        "supported": False,
        "skip_reason": "no_supported_packet_sentinel",
    }


def _verification_row(packet_dir: Path, *, include_legacy: bool = False) -> Dict[str, Any]:
    kind = _packet_kind(packet_dir, include_legacy=include_legacy)
    if not bool(kind.get("supported")):
        return {
            "packet_dir": str(packet_dir),
            "status": "skipped",
            **kind,
        }

    if kind.get("verifier") == "verify_review_packet":
        from scripts.verify_review_packet import verify_packet

        payload = verify_packet(packet_dir)
        fingerprint = payload.get("digest_fingerprint", {})
    else:
        from scripts.verify_track4_diagnostic_packet import verify_packet

        payload = verify_packet(packet_dir)
        fingerprint = payload.get("manifest_fingerprint", {})

    return {
        "packet_dir": str(packet_dir),
        "status": "verified",
        **kind,
        "pass": bool(payload.get("pass")),
        "failure_reasons": payload.get("failure_reasons", []),
        "actual_packet_file_count": payload.get("actual_packet_file_count"),
        "verified_file_count": payload.get("verified_file_count"),
        "fingerprint_name": fingerprint.get("name") if isinstance(fingerprint, dict) else None,
        "fingerprint_sha256": fingerprint.get("sha256") if isinstance(fingerprint, dict) else None,
        "verification": payload,
    }


def verify_all_packets(
    *,
    packet_dirs: Optional[Sequence[Path]] = None,
    root: Path = DEFAULT_PACKET_ROOT,
    include_legacy: bool = False,
    fail_on_skip: bool = False,
) -> Dict[str, Any]:
    candidates = [Path(path) for path in packet_dirs] if packet_dirs else _discover_packet_dirs(root)
    rows = [_verification_row(path, include_legacy=include_legacy) for path in candidates]
    verified = [row for row in rows if row.get("status") == "verified"]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    failed = [row for row in verified if not bool(row.get("pass"))]
    failure_reasons: List[str] = []
    if not verified:
        failure_reasons.append("no_packets_verified")
    for row in failed:
        failure_reasons.append(f"packet_failed:{row.get('packet_dir')}")
    if fail_on_skip and skipped:
        failure_reasons.append("packet_skipped")
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "pass": not failure_reasons,
        "packet_count": len(rows),
        "verified_packet_count": len(verified),
        "skipped_packet_count": len(skipped),
        "failed_packet_count": len(failed),
        "failure_reasons": failure_reasons,
        "packets": verified,
        "skipped": skipped,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--packet-dir", type=Path, action="append", default=[])
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = verify_all_packets(
        packet_dirs=args.packet_dir or None,
        root=args.root,
        include_legacy=bool(args.include_legacy),
        fail_on_skip=bool(args.fail_on_skip),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(payload.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
