import json
import sys
import types
from pathlib import Path

import pytest

from analysis.airflow_ablation_orchestrator import AblationConfig, compute_ablation_cell, run_ablation_matrix
from analysis.airflow_master_orchestrator import (
    MasterConfig,
    _parse_args,
    _run_lab_ablation,
    _run_review_packet_generation,
    _run_review_packet_verification,
    run_master_orchestration,
)
from scripts.build_focused_paper_packet import build_packet as build_focused_packet
from scripts.build_track4_diagnostic_packet import build_packet as build_track4_packet
from test_focused_paper_packet import _write_minimal_evidence_bundle
from test_track4_diagnostic_packet import _write_diagnostic_source


def test_master_orchestrator_can_call_lower_level_ablation_dag(monkeypatch, tmp_path: Path):
    fake_master = types.ModuleType("core.master_ablation")
    fake_master.ABLATION_PRESETS = {"full_mature": {"kernel_type": "matern"}}

    class FakeAblationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.kernel_type = kwargs.get("kernel_type", "imq")

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run_single(self, config):
            out_dir = Path(config.output_dir) / config.run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "ablation_manifest.json").write_text('{"run_status":"completed"}', encoding="utf-8")
            return {"run_status": "completed", "kernel_type": config.kernel_type}

    fake_master.AblationConfig = FakeAblationConfig
    fake_master.AblationRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "core.master_ablation", fake_master)

    cfg = MasterConfig(
        run_id="master_test",
        record_root=tmp_path / "records",
        suite_args=[],
        lab_ablation_preset="full_mature",
        lab_ablation_sweep=None,
        lab_ablation_output_dir=tmp_path / "lab",
        lab_ablation_max_articles=12,
        lab_ablation_device="cpu",
        run_procrustes=False,
        procrustes_data_dir=None,
        viz_experiment_dir=None,
        viz_output_html=None,
        run_ablation=False,
        ablation_scalar_bins=8,
        freeze_snapshot=False,
        freeze_snapshot_name=None,
    )

    result = _run_lab_ablation(cfg, tmp_path / "record")

    assert result is not None
    assert result["status"] == "success"
    assert result["mode"] == "single"
    assert result["preset"] == "full_mature"
    assert (tmp_path / "lab" / "master_test__full_mature" / "ablation_manifest.json").exists()


def test_master_orchestrator_suite_arg_accepts_child_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "airflow_master_orchestrator.py",
            "--suite-arg=--limit",
            "--suite-arg=2",
            "--suite-arg=--mode",
            "--suite-arg=enhanced",
        ],
    )

    args = _parse_args()

    assert args.suite_arg == ["--limit", "2", "--mode", "enhanced"]


def test_master_orchestrator_accepts_review_packet_verification_args(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "airflow_master_orchestrator.py",
            "--run-review-packet-verification",
            "--focused-review-packet-dir",
            str(tmp_path / "focused"),
            "--track4-review-packet-dir",
            str(tmp_path / "track4"),
            "--review-packet-verification-output-dir",
            str(tmp_path / "verify"),
        ],
    )

    args = _parse_args()

    assert args.run_review_packet_verification is True
    assert args.focused_review_packet_dir == tmp_path / "focused"
    assert args.track4_review_packet_dir == tmp_path / "track4"
    assert args.review_packet_verification_output_dir == tmp_path / "verify"


def test_master_orchestrator_accepts_review_packet_generation_args(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "airflow_master_orchestrator.py",
            "--run-review-packet-generation",
            "--focused-evidence-dir",
            str(tmp_path / "evidence"),
            "--track4-diagnostic-dir",
            str(tmp_path / "track4_diagnostics"),
            "--track4-note-path",
            str(tmp_path / "track4_note.md"),
            "--review-packet-output-root",
            str(tmp_path / "packets"),
            "--review-packet-max-files",
            "10",
        ],
    )

    args = _parse_args()

    assert args.run_review_packet_generation is True
    assert args.focused_evidence_dir == tmp_path / "evidence"
    assert args.track4_diagnostic_dir == tmp_path / "track4_diagnostics"
    assert args.track4_note_path == tmp_path / "track4_note.md"
    assert args.review_packet_output_root == tmp_path / "packets"
    assert args.review_packet_max_files == 10


def _review_packet_config(
    tmp_path: Path,
    *,
    focused_dir: Path,
    track4_dir: Path,
    generation: bool = False,
    focused_evidence_dir: Path | None = None,
    track4_diagnostic_dir: Path | None = None,
    track4_note_path: Path | None = None,
) -> MasterConfig:
    return MasterConfig(
        run_id="packet_verify",
        record_root=tmp_path / "records",
        suite_args=[],
        lab_ablation_preset=None,
        lab_ablation_sweep=None,
        lab_ablation_output_dir=None,
        lab_ablation_max_articles=12,
        lab_ablation_device="cpu",
        run_procrustes=False,
        procrustes_data_dir=None,
        viz_experiment_dir=None,
        viz_output_html=None,
        run_ablation=False,
        ablation_scalar_bins=8,
        freeze_snapshot=False,
        freeze_snapshot_name=None,
        run_review_packet_generation=generation,
        focused_evidence_dir=focused_evidence_dir,
        track4_diagnostic_dir=track4_diagnostic_dir,
        track4_note_path=track4_note_path,
        review_packet_output_root=tmp_path / "generated_packets",
        review_packet_max_files=10,
        run_review_packet_verification=True,
        focused_review_packet_dir=focused_dir,
        track4_review_packet_dir=track4_dir,
        review_packet_verification_output_dir=tmp_path / "packet_verification",
    )


def _build_review_packet_pair(tmp_path: Path) -> tuple[Path, Path]:
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


def _build_review_packet_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence_dir = tmp_path / "focused_evidence"
    _write_minimal_evidence_bundle(evidence_dir)
    diagnostic_source = tmp_path / "track4_diagnostics"
    diagnostic_source.mkdir()
    note = tmp_path / "TRACK4_SOFT_TERRAIN_DIAGNOSTIC_20260525.md"
    note.write_text("# Track 4 note\n", encoding="utf-8")
    _write_diagnostic_source(diagnostic_source)
    return evidence_dir, diagnostic_source, note


def test_master_orchestrator_generates_review_packets(tmp_path: Path):
    evidence_dir, diagnostic_source, note = _build_review_packet_sources(tmp_path)
    cfg = _review_packet_config(
        tmp_path,
        focused_dir=tmp_path / "unused_focused",
        track4_dir=tmp_path / "unused_track4",
        generation=True,
        focused_evidence_dir=evidence_dir,
        track4_diagnostic_dir=diagnostic_source,
        track4_note_path=note,
    )

    result = _run_review_packet_generation(cfg, tmp_path / "record")

    focused_packet = Path(str(result["focused_packet_dir"]))
    track4_packet = Path(str(result["track4_packet_dir"]))
    assert result["status"] == "success"
    assert (focused_packet / "review_digest.json").exists()
    assert (track4_packet / "artifact_manifest.json").exists()
    assert result["max_files"] == 10


def test_master_orchestrator_generation_feeds_verification(monkeypatch, tmp_path: Path):
    evidence_dir, diagnostic_source, note = _build_review_packet_sources(tmp_path)
    cfg = _review_packet_config(
        tmp_path,
        focused_dir=tmp_path / "unused_focused",
        track4_dir=tmp_path / "unused_track4",
        generation=True,
        focused_evidence_dir=evidence_dir,
        track4_diagnostic_dir=diagnostic_source,
        track4_note_path=note,
    )
    cfg = MasterConfig(**{**cfg.__dict__, "focused_review_packet_dir": None, "track4_review_packet_dir": None})

    def fake_run_command(command, *, cwd, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ fake\n", encoding="utf-8")
        return {"command": command, "cwd": str(cwd), "exit_code": 0, "log": str(log_path)}

    monkeypatch.setattr("analysis.airflow_master_orchestrator._run_command", fake_run_command)
    record_path = run_master_orchestration(cfg)
    payload = json.loads(record_path.read_text(encoding="utf-8"))

    assert payload["steps"]["review_packet_generation"]["status"] == "success"
    assert payload["steps"]["review_packet_verification"]["pass"] is True


def test_master_orchestrator_verifies_review_packets(tmp_path: Path):
    focused_dir, track4_dir = _build_review_packet_pair(tmp_path)
    cfg = _review_packet_config(tmp_path, focused_dir=focused_dir, track4_dir=track4_dir)

    result = _run_review_packet_verification(cfg, tmp_path / "record")

    summary_path = Path(str(result["summary_json"]))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["pass"] is True
    assert payload["pass"] is True
    assert payload["verified_packet_count"] == 2
    by_type = {row["packet_type"]: row for row in payload["packets"]}
    assert by_type["focused_paper_evidence"]["pass"] is True
    assert by_type["track4_terrain_diagnostic"]["pass"] is True
    assert by_type["track4_terrain_diagnostic"]["verified_file_count"] == 9


def test_master_orchestrator_review_packet_verification_reports_failures(tmp_path: Path):
    focused_dir, track4_dir = _build_review_packet_pair(tmp_path)
    (track4_dir / "README.md").write_text("tampered", encoding="utf-8")
    cfg = _review_packet_config(tmp_path, focused_dir=focused_dir, track4_dir=track4_dir)

    result = _run_review_packet_verification(cfg, tmp_path / "record")

    summary_path = Path(str(result["summary_json"]))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["pass"] is False
    by_type = {row["packet_type"]: row for row in payload["packets"]}
    assert by_type["focused_paper_evidence"]["pass"] is True
    assert by_type["track4_terrain_diagnostic"]["pass"] is False
    assert "fingerprint_mismatch:README.md" in by_type["track4_terrain_diagnostic"]["failure_reasons"]


def test_airflow_csv_sidecar_rejects_missing_required_columns(tmp_path: Path):
    base_csv = tmp_path / "MONOLITH_DATA.csv"
    base_csv.write_text("density,label\n0.5,a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stress"):
        compute_ablation_cell(
            AblationConfig(base_csv=base_csv, output_dir=tmp_path / "out"),
            stress_mode="raw_l2",
            zone_rule="fixed_0_5_threshold",
        )


def test_airflow_csv_sidecar_handles_constant_stress_without_nan(tmp_path: Path):
    base_csv = tmp_path / "MONOLITH_DATA.csv"
    base_csv.write_text(
        "density,stress\n0.2,7.0\n0.8,7.0\n",
        encoding="utf-8",
    )

    cell = compute_ablation_cell(
        AblationConfig(base_csv=base_csv, output_dir=tmp_path / "out", scalar_bins=4),
        stress_mode="normalized_l2",
        zone_rule="fixed_0_5_threshold",
    )

    assert cell["stress_view_min"] == 0.5
    assert cell["stress_view_max"] == 0.5
    assert cell["scalar_bin_coverage"] > 0.0


def test_airflow_csv_sidecar_writes_summary_bundle(tmp_path: Path):
    base_csv = tmp_path / "MONOLITH_DATA.csv"
    base_csv.write_text(
        "density,stress\n0.2,0.8\n0.8,0.2\n0.9,0.9\n0.1,0.1\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    summary_path = run_ablation_matrix(base_csv, out_dir, scalar_bins=4)

    assert summary_path.exists()
    assert (out_dir / "manifold_ablation_summary.csv").exists()
    assert len(list(out_dir.glob("ablation_cell_*.json"))) == 4
