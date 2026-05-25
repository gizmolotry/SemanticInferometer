import json
from pathlib import Path

from scripts.build_focused_paper_packet import build_packet
from scripts.verify_review_packet import verify_packet
from test_focused_paper_packet import _write_json, _write_minimal_evidence_bundle


def test_verify_review_packet_passes_on_generated_packet(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    packet_dir = tmp_path / "packet"
    track4_basis = tmp_path / "track4_focused_basis_validation_summary.json"
    _write_minimal_evidence_bundle(evidence_dir)
    _write_json(track4_basis, {"summary_type": "track4_focused_basis_validation"})
    build_packet(evidence_dir=evidence_dir, out_dir=packet_dir, track4_basis_summary=track4_basis)

    payload = verify_packet(packet_dir)

    assert payload["pass"] is True
    assert payload["failure_reasons"] == []
    assert payload["verified_file_count"] == 9
    assert payload["actual_packet_file_count"] == 10
    assert payload["digest_fingerprint"]["name"] == "review_digest.json"


def test_verify_review_packet_detects_tampered_file(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    packet_dir = tmp_path / "packet"
    _write_minimal_evidence_bundle(evidence_dir)
    build_packet(evidence_dir=evidence_dir, out_dir=packet_dir, track4_basis_summary=None)
    target = packet_dir / "claim_matrix.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["tampered"] = True
    target.write_text(json.dumps(payload), encoding="utf-8")

    verification = verify_packet(packet_dir)

    assert verification["pass"] is False
    assert "fingerprint_mismatch:claim_matrix.json" in verification["failure_reasons"]


def test_verify_review_packet_detects_tampered_digest_claims(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    packet_dir = tmp_path / "packet"
    _write_minimal_evidence_bundle(evidence_dir)
    build_packet(evidence_dir=evidence_dir, out_dir=packet_dir, track4_basis_summary=None)
    digest_path = packet_dir / "review_digest.json"
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    digest["supported_exploratory_claims"] = [
        *digest["supported_exploratory_claims"],
        "track4_traversal_validity",
    ]
    digest_path.write_text(json.dumps(digest), encoding="utf-8")

    verification = verify_packet(packet_dir)

    assert verification["pass"] is False
    assert "supported_exploratory_claims_mismatch" in verification["failure_reasons"]


def test_verify_review_packet_detects_digest_file_list_drift(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    packet_dir = tmp_path / "packet"
    _write_minimal_evidence_bundle(evidence_dir)
    build_packet(evidence_dir=evidence_dir, out_dir=packet_dir, track4_basis_summary=None)
    _write_json(packet_dir / "extra.json", {"not": "listed"})
    digest_path = packet_dir / "review_digest.json"
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    digest["packet_file_count"] = 999
    digest_path.write_text(json.dumps(digest), encoding="utf-8")

    verification = verify_packet(packet_dir)

    assert verification["pass"] is False
    assert "packet_file_count_mismatch" in verification["failure_reasons"]
    assert "unlisted_packet_file:extra.json" in verification["failure_reasons"]


def test_verify_review_packet_fails_without_digest(tmp_path: Path):
    payload = verify_packet(tmp_path / "missing")

    assert payload["pass"] is False
    assert payload["failure_reasons"] == ["review_digest_missing"]
