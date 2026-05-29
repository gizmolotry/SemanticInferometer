from pathlib import Path

from analysis.verification.thesis_evidence import build_thesis_evidence
from thesis_test_support import build_canonical_fixture, write_relativity_deltas


def test_observer_relativity_flags_zero_rotation_placeholder_behavior(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    relativity_path = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "relativity_deltas.json"
    )
    write_relativity_deltas(
        relativity_path,
        mean_coord_delta=0.0,
        max_coord_delta=0.0,
        rotation_deg=0.0,
        path_flip_count=0,
        placeholder=True,
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    observer_summary = payloads["observer_relativity_summary"]
    assert observer_summary["aggregate"]["pass_rate"] < 1.0
    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "observer_relativity"
    )
    assert not claim["thesis_safe"]
    failures = payloads["scientific_validation_summary"]["failure_modes"]["records"]
    assert any(record["failure_type"] == "zero_rotation_observer" for record in failures)


def test_observer_relativity_prefers_null_equivalence_path_flips_over_translation_only_zero(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    relativity_path = (
        fixture["runs_dir"]
        / "experiments_20260504_010101"
        / "rbf"
        / "cls"
        / "real"
        / "relativity_deltas.json"
    )
    relativity_path.write_text(
        """
{
  "status": "OK",
  "synthetic_placeholder": false,
  "observer_count": 1,
  "summary": {"mean_coord_delta": 0.4, "max_coord_delta": 0.4},
  "observers": [
    {
      "observer_id": 0,
      "null_observer_equivalence": {
        "equivalent": false,
        "max_coord_delta": 0.4,
        "path_flip_count": 4,
        "axis_rotation_deg": 12.0
      },
      "axis_delta": {"rotation_deg": 12.0},
      "translation_only_comparison": {"d_path_flip_count": 0}
    }
  ]
}
        """.strip(),
        encoding="utf-8",
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    claim = next(
        claim for claim in payloads["claim_matrix"]["claims"]
        if claim["claim_id"] == "observer_relativity"
    )
    assert claim["thesis_safe"] is True
    summary = payloads["observer_relativity_summary"]
    record = next(row for row in summary["records"] if row["kernel"] == "rbf" and row["channel"] == "cls")
    assert record["max_path_flip_count"] == 4
