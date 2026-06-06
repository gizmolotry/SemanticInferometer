import json
import zipfile
from pathlib import Path

from scripts.build_track4_diagnostic_packet import build_packet, packet_file_count, write_packet_zip
from scripts.verify_track4_diagnostic_packet import verify_packet as verify_track4_packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_diagnostic_source(root: Path) -> None:
    _write_json(
        root / "track4_terrain_semantics_diagnostics_summary.json",
        {
            "claim_boundary": {
                "terrain_construct_distinctness": True,
                "walker_sensitivity_detected": True,
                "native_work_decomposition_available": False,
                "terrain_specificity_supported": True,
                "soft_terrain_work_coupling_supported": True,
                "pooled_soft_terrain_specificity_supported": False,
                "matched_soft_terrain_specificity_supported": True,
                "targeted_event_pair_status": "NO_DATA",
            }
        },
    )
    _write_json(root / "terrain_construct_validity.json", {"zone_count": 4, "work_range": 12.0})
    _write_json(
        root / "real_vs_control_terrain_specificity.json",
        {
            "real_minus_control_terrain_safe_rate": 0.37,
            "real_minus_control_mean_score": 0.18,
        },
    )
    (root / "terrain_contrast_matrix.csv").write_text(
        "contrast,observation_count,mean_work_gap_abs\nBridge_vs_Swamp,3,2.4\n",
        encoding="utf-8",
    )
    _write_json(
        root / "terrain_contrast_matrix.json",
        {"ranked_contrasts": [{"contrast": "Bridge_vs_Swamp", "mean_work_gap_abs": 2.4}]},
    )
    _write_json(
        root / "soft_terrain_work_coupling.json",
        {
            "real_minus_control_barrier_work_corr": 0.03,
            "real": {"corr_soft_barrier_mass_work": 0.296},
            "matched_cell_specificity": {
                "matched_cell_pass_rate": 0.778,
                "supporting_cell_count": 7,
                "usable_matched_cell_count": 9,
                "median_excess_corr_real_minus_control": 0.634,
            },
        },
    )
    _write_json(root / "walker_sensitivity_matrix.json", {"score_range": 0.19})
    _write_json(
        root / "work_decomposition_summary.json",
        {"confound_correlations": {"work_vs_path_edge_count": 0.5}},
    )
    _write_json(root / "targeted_event_pair_results.json", {"status": "NO_DATA"})


def test_build_track4_diagnostic_packet_stays_under_ten_files(tmp_path: Path):
    source = tmp_path / "diagnostics"
    source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(source)

    packet_dir = tmp_path / "packet"
    manifest = build_packet(diagnostic_dir=source, out_dir=packet_dir, note_path=note, max_files=10)

    assert packet_file_count(packet_dir) <= 10
    assert manifest["file_count"] <= 10
    assert (packet_dir / "README.md").exists()
    assert (packet_dir / "artifact_manifest.json").exists()
    assert (packet_dir / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md").exists()
    assert (packet_dir / "evidence" / "track4_terrain_semantics_diagnostics_summary.json").exists()
    assert (packet_dir / "evidence" / "soft_terrain_work_coupling.json").exists()
    assert not (packet_dir / "evidence" / "terrain_contrast_matrix.csv").exists()
    assert "artifact_manifest.json" in manifest["packet_files"]
    assert "README.md" in manifest["packet_files"]
    packet_fingerprints = {
        row["name"]: row
        for row in manifest["packet_file_fingerprints"]
    }
    assert "artifact_manifest.json" not in packet_fingerprints
    assert "README.md" in packet_fingerprints
    copied_soft = [
        row
        for row in manifest["copied_artifacts"]
        if row["packet_relpath"] == "evidence/soft_terrain_work_coupling.json"
    ][0]
    assert copied_soft["bytes"] > 0
    assert len(copied_soft["sha256"]) == 64
    readme = (packet_dir / "README.md").read_text(encoding="utf-8")
    assert "Terrain specificity supported: `true`" in readme
    assert "Pooled soft terrain specificity supported: `false`" in readme
    assert "Matched soft terrain specificity supported: `true`" in readme
    assert "Matched supporting cells: `7` / `9`" in readme
    assert "not a broad semantic-terrain ontology" in readme
    assert "Top contrast" not in readme


def test_write_track4_diagnostic_packet_zip_preserves_layout(tmp_path: Path):
    packet_dir = tmp_path / "packet"
    (packet_dir / "evidence").mkdir(parents=True)
    (packet_dir / "README.md").write_text("readme", encoding="utf-8")
    (packet_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    (packet_dir / "evidence" / "x.json").write_text("{}", encoding="utf-8")

    zip_path = write_packet_zip(packet_dir)

    assert zip_path == packet_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == [
            "README.md",
            "artifact_manifest.json",
            "evidence/x.json",
        ]


def test_verify_track4_diagnostic_packet_passes_on_generated_packet(tmp_path: Path):
    source = tmp_path / "diagnostics"
    source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(source)

    packet_dir = tmp_path / "packet"
    build_packet(diagnostic_dir=source, out_dir=packet_dir, note_path=note, max_files=10)

    payload = verify_track4_packet(packet_dir)

    assert payload["pass"] is True
    assert payload["failure_reasons"] == []
    assert payload["actual_packet_file_count"] == 10
    assert payload["max_files"] == 10
    assert payload["verified_file_count"] == 9
    assert payload["verified_copied_file_count"] == 8
    assert payload["manifest_fingerprint"]["name"] == "artifact_manifest.json"


def test_verify_track4_diagnostic_packet_detects_missing_copied_file(tmp_path: Path):
    source = tmp_path / "diagnostics"
    source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(source)
    packet_dir = tmp_path / "packet"
    build_packet(diagnostic_dir=source, out_dir=packet_dir, note_path=note, max_files=10)
    (packet_dir / "evidence" / "soft_terrain_work_coupling.json").unlink()

    payload = verify_track4_packet(packet_dir)

    assert payload["pass"] is False
    assert "missing_file:evidence/soft_terrain_work_coupling.json" in payload["failure_reasons"]
    assert "copied_file_missing:evidence/soft_terrain_work_coupling.json" in payload["failure_reasons"]
    assert "listed_file_missing:evidence/soft_terrain_work_coupling.json" in payload["failure_reasons"]


def test_verify_track4_diagnostic_packet_detects_tampered_copied_file(tmp_path: Path):
    source = tmp_path / "diagnostics"
    source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(source)
    packet_dir = tmp_path / "packet"
    build_packet(diagnostic_dir=source, out_dir=packet_dir, note_path=note, max_files=10)
    target = packet_dir / "evidence" / "soft_terrain_work_coupling.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["tampered"] = True
    target.write_text(json.dumps(payload), encoding="utf-8")

    verification = verify_track4_packet(packet_dir)

    assert verification["pass"] is False
    assert "fingerprint_mismatch:evidence/soft_terrain_work_coupling.json" in verification["failure_reasons"]
    assert "copied_file_tampered:evidence/soft_terrain_work_coupling.json" in verification["failure_reasons"]


def test_verify_track4_diagnostic_packet_detects_tampered_readme(tmp_path: Path):
    source = tmp_path / "diagnostics"
    source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(source)
    packet_dir = tmp_path / "packet"
    build_packet(diagnostic_dir=source, out_dir=packet_dir, note_path=note, max_files=10)
    (packet_dir / "README.md").write_text("tampered", encoding="utf-8")

    verification = verify_track4_packet(packet_dir)

    assert verification["pass"] is False
    assert "fingerprint_mismatch:README.md" in verification["failure_reasons"]


def test_verify_track4_diagnostic_packet_detects_cap_drift(tmp_path: Path):
    source = tmp_path / "diagnostics"
    source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(source)
    packet_dir = tmp_path / "packet"
    build_packet(diagnostic_dir=source, out_dir=packet_dir, note_path=note, max_files=10)
    (packet_dir / "extra.txt").write_text("extra", encoding="utf-8")

    payload = verify_track4_packet(packet_dir)

    assert payload["pass"] is False
    assert "packet_file_count_exceeds_max" in payload["failure_reasons"]
    assert "unlisted_packet_file:extra.txt" in payload["failure_reasons"]
