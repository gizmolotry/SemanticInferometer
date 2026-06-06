from pathlib import Path

from analysis.verification.scientific_summaries import summarize_terrain_incremental_signal
from analysis.verification.thesis_evidence import build_thesis_evidence
from thesis_test_support import (
    build_canonical_fixture,
    write_csv,
    write_hidden_groups,
    write_synthetic_terrain_incremental_fixture,
)


def test_terrain_incremental_signal_passes_when_cross_zone_work_gaps_exceed_same_zone_gaps(tmp_path: Path):
    run_dir = tmp_path / "synthetic_run"
    write_synthetic_terrain_incremental_fixture(run_dir)

    summary = summarize_terrain_incremental_signal(run_dir)

    assert summary["status"] == "OK"
    assert summary["safe_for_thesis_claim"] is True
    assert summary["mixed_label_count"] == 2
    assert summary["same_label_cross_terrain_pair_count"] >= 3
    assert summary["cross_minus_same_work_gap"] > 0.10
    assert summary["cliffs_delta_cross_gt_same"] > 0.0


def test_terrain_incremental_signal_rejects_labels_that_are_just_terrain_aliases(tmp_path: Path):
    run_dir = tmp_path / "synthetic_run"
    rows = [
        {"index": 0, "zone": "Bridge", "w_actual": 10.0, "density": 0.9, "stress": 0.1, "x": 0.0, "y": 0.0},
        {"index": 1, "zone": "Bridge", "w_actual": 11.0, "density": 0.8, "stress": 0.1, "x": 1.0, "y": 0.0},
        {"index": 2, "zone": "Bridge", "w_actual": 12.0, "density": 0.7, "stress": 0.1, "x": 2.0, "y": 0.0},
        {"index": 3, "zone": "Void", "w_actual": 40.0, "density": 0.1, "stress": 0.9, "x": 3.0, "y": 0.0},
        {"index": 4, "zone": "Void", "w_actual": 41.0, "density": 0.1, "stress": 0.8, "x": 4.0, "y": 0.0},
        {"index": 5, "zone": "Void", "w_actual": 42.0, "density": 0.1, "stress": 0.7, "x": 5.0, "y": 0.0},
    ]
    write_csv(
        run_dir / "MONOLITH_DATA.csv",
        rows,
        fieldnames=["index", "zone", "w_actual", "density", "stress", "x", "y"],
    )
    write_hidden_groups(run_dir / "labels" / "hidden_groups.csv", {
        0: "Bridge",
        1: "Bridge",
        2: "Bridge",
        3: "Void",
        4: "Void",
        5: "Void",
    })

    summary = summarize_terrain_incremental_signal(run_dir)

    assert summary["status"] == "INVALID"
    assert summary["safe_for_thesis_claim"] is False
    assert "no hidden label contains multiple terrain zones" in summary["failure_reasons"]
    assert summary["warnings"]


def test_thesis_evidence_emits_terrain_incremental_signal_claim(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, kernels=["rbf"], channels=["cls"], seeds=[42])

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claims = {claim["claim_id"]: claim for claim in payloads["claim_matrix"]["claims"]}
    assert claims["terrain_incremental_signal"]["thesis_safe"] is True
    summary = payloads["terrain_incremental_signal_summary"]
    assert summary["aggregate"]["n_runs"] == 1
    assert summary["aggregate"]["mean_cross_minus_same_work_gap"] > 0.10
