from pathlib import Path

from scripts.build_thesis_evidence import main
from thesis_test_support import build_canonical_fixture


def test_build_thesis_evidence_cli_validates_focused_selection_before_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    fixture = build_canonical_fixture(fixture_root)
    out_dir = tmp_path / "invalid_bundle"
    missing_manifest = tmp_path / "missing" / "experiment_manifest.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_thesis_evidence.py",
            "--runs-dir",
            str(fixture["runs_dir"]),
            "--methods-path",
            str(fixture["methods"]),
            "--results-path",
            str(fixture["results"]),
            "--suite-manifest",
            str(missing_manifest),
            "--output-dir",
            str(out_dir),
        ],
    )

    assert main() == 1
    assert not out_dir.exists()


def test_build_thesis_evidence_cli_writes_after_publication_profile_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    fixture = build_canonical_fixture(fixture_root)
    out_dir = tmp_path / "valid_bundle"

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_thesis_evidence.py",
            "--runs-dir",
            str(fixture["runs_dir"]),
            "--methods-path",
            str(fixture["methods"]),
            "--results-path",
            str(fixture["results"]),
            "--suite-manifest",
            str(fixture["suite_manifest"]),
            "--suite-manifest",
            str(fixture["runs_dir"] / "experiments_20260504_030303" / "experiment_manifest.json"),
            "--output-dir",
            str(out_dir),
            "--control-metric-basis",
            "comprehensive",
            "--publication-profile",
        ],
    )

    assert main() == 0
    assert (out_dir / "scientific_validation_summary.json").exists()
    assert (out_dir / "paper_claim_profile.json").exists()
