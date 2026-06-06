import json
from pathlib import Path

from scripts.build_focused_paper_packet import build_packet as build_focused_packet
from scripts.build_track4_diagnostic_packet import build_packet as build_track4_packet
from scripts.verify_all_review_packets import verify_all_packets
from test_focused_paper_packet import _write_minimal_evidence_bundle
from test_track4_diagnostic_packet import _write_diagnostic_source


def _build_packet_pair(tmp_path: Path) -> tuple[Path, Path]:
    evidence_dir = tmp_path / "focused_evidence"
    focused_dir = tmp_path / "focused_packet"
    _write_minimal_evidence_bundle(evidence_dir)
    build_focused_packet(
        evidence_dir=evidence_dir,
        out_dir=focused_dir,
        track4_basis_summary=None,
    )

    diagnostic_source = tmp_path / "track4_diagnostics"
    diagnostic_source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(diagnostic_source)
    track4_dir = tmp_path / "track4_packet"
    build_track4_packet(
        diagnostic_dir=diagnostic_source,
        out_dir=track4_dir,
        note_path=note,
        max_files=10,
    )
    return focused_dir, track4_dir


def test_verify_all_review_packets_routes_supported_packet_types(tmp_path: Path):
    focused_dir, track4_dir = _build_packet_pair(tmp_path)

    payload = verify_all_packets(packet_dirs=[focused_dir, track4_dir], root=tmp_path)

    assert payload["pass"] is True
    assert payload["packet_count"] == 2
    assert payload["verified_packet_count"] == 2
    assert payload["skipped_packet_count"] == 0
    by_type = {row["packet_type"]: row for row in payload["packets"]}
    assert by_type["focused_paper_evidence"]["verifier"] == "verify_review_packet"
    assert by_type["focused_paper_evidence"]["sentinel"] == "review_digest.json"
    assert by_type["focused_paper_evidence"]["verified_file_count"] == 8
    assert by_type["track4_terrain_diagnostic"]["verifier"] == "verify_track4_diagnostic_packet"
    assert by_type["track4_terrain_diagnostic"]["sentinel"] == "artifact_manifest.json"
    assert by_type["track4_terrain_diagnostic"]["verified_file_count"] == 9


def test_verify_all_review_packets_aggregates_child_failures(tmp_path: Path):
    focused_dir, track4_dir = _build_packet_pair(tmp_path)
    target = track4_dir / "evidence" / "soft_terrain_work_coupling.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["tampered"] = True
    target.write_text(json.dumps(data), encoding="utf-8")

    payload = verify_all_packets(packet_dirs=[focused_dir, track4_dir], root=tmp_path)

    assert payload["pass"] is False
    assert payload["failed_packet_count"] == 1
    assert any(reason.startswith("packet_failed:") for reason in payload["failure_reasons"])
    track4 = [row for row in payload["packets"] if row["packet_type"] == "track4_terrain_diagnostic"][0]
    assert "fingerprint_mismatch:evidence/soft_terrain_work_coupling.json" in track4["failure_reasons"]


def test_verify_all_review_packets_skips_legacy_or_unknown_packets_by_default(tmp_path: Path):
    focused_dir, track4_dir = _build_packet_pair(tmp_path)
    legacy_dir = tmp_path / "legacy_packet"
    legacy_dir.mkdir()
    (legacy_dir / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "old_shape": True}),
        encoding="utf-8",
    )
    unknown_dir = tmp_path / "unknown_packet"
    unknown_dir.mkdir()
    (unknown_dir / "README.md").write_text("not a known packet", encoding="utf-8")

    payload = verify_all_packets(packet_dirs=[focused_dir, track4_dir, legacy_dir, unknown_dir], root=tmp_path)

    assert payload["pass"] is True
    assert payload["verified_packet_count"] == 2
    assert payload["skipped_packet_count"] == 2
    reasons = {row["skip_reason"] for row in payload["skipped"]}
    assert reasons == {"unsupported_artifact_manifest_schema", "no_supported_packet_sentinel"}
    assert {Path(row["packet_dir"]).name for row in payload["packets"]} == {
        focused_dir.name,
        track4_dir.name,
    }


def test_verify_all_review_packets_can_fail_on_skipped_packets(tmp_path: Path):
    focused_dir, track4_dir = _build_packet_pair(tmp_path)
    unknown_dir = tmp_path / "unknown_packet"
    unknown_dir.mkdir()

    payload = verify_all_packets(packet_dirs=[focused_dir, track4_dir, unknown_dir], root=tmp_path, fail_on_skip=True)

    assert payload["pass"] is False
    assert "packet_skipped" in payload["failure_reasons"]
