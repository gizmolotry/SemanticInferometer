import json
from pathlib import Path

from scripts.audit_track2_foundation import audit_track2_foundation, write_audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_evidence_dir(path: Path, *, integrated_supported: bool = True) -> None:
    _write_json(
        path / "variance_separation_summary.json",
        {
            "primary_basis": "comprehensive_results",
            "primary_metric": "simple_variance",
            "primary_pass": integrated_supported,
            "mean_primary_abs_log_ratio": 0.33 if integrated_supported else 0.01,
            "required_kernels_evaluated": ["rbf", "matern", "imq"],
            "per_kernel": {
                "rbf": {"passes": integrated_supported},
                "matern": {"passes": integrated_supported},
                "imq": {"passes": integrated_supported},
            },
        },
    )
    _write_json(
        path / "ablation_matrix.json",
        {
            "required_modes_present": True,
            "by_track5_mode": {
                "hadamard_strict": {"n_runs": 3, "pass_rate": 1.0},
                "riemannian_strict": {"n_runs": 3, "pass_rate": 1.0},
            },
            "records": [],
        },
    )
    _write_json(
        path / "scientific_validation_summary.json",
        {
            "synthetic_recoverability": {
                "thesis_safe": True,
                "mean_nmi": 0.71,
                "std_nmi": 0.02,
                "mean_ari": 0.44,
                "std_ari": 0.03,
            }
        },
    )


def _write_track4_summary(path: Path, *, track2_supported: bool = False) -> None:
    track2_real_safe = True if track2_supported else False
    track2_control_safe = False if track2_supported else True
    rows = [
        {
            "corpus": "real",
            "basis": "track2",
            "safe_for_thesis_claim": track2_real_safe,
            "basis_probe_score": 0.90 if track2_supported else 0.60,
        },
        {
            "corpus": "control_random",
            "basis": "track2",
            "safe_for_thesis_claim": track2_control_safe,
            "basis_probe_score": 0.50 if track2_supported else 0.80,
        },
        {
            "corpus": "real",
            "basis": "logits_flat",
            "safe_for_thesis_claim": True,
            "basis_probe_score": 0.92,
        },
        {
            "corpus": "control_random",
            "basis": "logits_flat",
            "safe_for_thesis_claim": False,
            "basis_probe_score": 0.50,
        },
    ]
    _write_json(path, {"rows": rows})


def test_track2_walker_failure_does_not_mark_whole_system_busted(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    track4_summary = tmp_path / "track4_summary.json"
    _write_evidence_dir(evidence_dir, integrated_supported=True)
    _write_track4_summary(track4_summary, track2_supported=False)

    report = audit_track2_foundation(evidence_dir=evidence_dir, track4_summary_path=track4_summary)

    assert report["foundation_status"] == "NOT_BUSTED_TRACK4_LOCAL_FAILURE"
    assert report["whole_system_busted_by_current_evidence"] is False
    assert report["track4_track2_basis_failure_is_local"] is True
    assert report["direct_track2_necessity_tested"] is False
    assert report["checks"]["integrated_geometry_signal"]["supported"] is True
    assert report["checks"]["track4_track2_basis_signal"]["supported"] is False


def test_track2_foundation_audit_marks_system_at_risk_when_integrated_geometry_fails(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    track4_summary = tmp_path / "track4_summary.json"
    _write_evidence_dir(evidence_dir, integrated_supported=False)
    _write_track4_summary(track4_summary, track2_supported=False)

    report = audit_track2_foundation(evidence_dir=evidence_dir, track4_summary_path=track4_summary)

    assert report["foundation_status"] == "FOUNDATION_AT_RISK"
    assert report["whole_system_busted_by_current_evidence"] is True
    assert report["track4_track2_basis_failure_is_local"] is False


def test_track2_foundation_audit_writes_json_and_csv(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    track4_summary = tmp_path / "track4_summary.json"
    output_dir = tmp_path / "out"
    _write_evidence_dir(evidence_dir, integrated_supported=True)
    _write_track4_summary(track4_summary, track2_supported=True)

    report = audit_track2_foundation(evidence_dir=evidence_dir, track4_summary_path=track4_summary)
    written = write_audit(report, output_dir)

    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    saved = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert saved["foundation_status"] == "FOUNDATION_SUPPORTED_BY_CURRENT_EVIDENCE"
    csv_text = Path(written["csv"]).read_text(encoding="utf-8")
    assert "integrated_geometry_signal" in csv_text
