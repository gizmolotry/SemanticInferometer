import json
import zipfile
from pathlib import Path

from scripts.build_track4_diagnostic_packet import build_packet, packet_file_count, write_packet_zip


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
    readme = (packet_dir / "README.md").read_text(encoding="utf-8")
    assert "Terrain specificity supported: `true`" in readme
    assert "Pooled soft terrain specificity supported: `false`" in readme
    assert "Matched soft terrain specificity supported: `true`" in readme
    assert "Matched supporting cells: `7` / `9`" in readme
    assert "not a broad semantic-terrain ontology" in readme
    assert "Bridge_vs_Swamp" in readme


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
