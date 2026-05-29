from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_track4_action_graph_runs import summarize


def _write_summary(
    path: Path,
    mean_action: float,
    corpus_prefix: str,
    *,
    observer_transport_penalty: float | None = None,
    hysteresis_penalty: float | None = None,
) -> Path:
    run_dir = path / f"{corpus_prefix}_demo_20260521"
    run_dir.mkdir(parents=True)
    record = {
        "metric": mean_action / 2.0,
        "shear_penalty": 1.0,
        "stress_penalty": 2.0,
        "curvature_penalty": 0.5,
    }
    if observer_transport_penalty is not None:
        record["observer_transport_penalty"] = observer_transport_penalty
    if hysteresis_penalty is not None:
        record["hysteresis_penalty"] = hysteresis_penalty
    payload = {
        "n_articles": 6,
        "path_count": 2,
        "reached_count": 2,
        "mean_action": mean_action,
        "median_action": mean_action,
        "max_action": mean_action + 1.0,
        "records": [record],
        "terrain_zone_counts": {"Bridge": 3, "Void": 3},
        "anchor_zones": ["Bridge", "Void"],
    }
    out = run_dir / "track4_action_summary.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_action_graph_probe_summary_reports_direction_free_engineering_lead(tmp_path: Path) -> None:
    real = _write_summary(tmp_path, 10.0, "real")
    random = _write_summary(tmp_path, 4.0, "control_random")
    shuffled = _write_summary(tmp_path, 6.0, "control_shuffled")

    payload = summarize([real, random, shuffled])

    assert payload["summary_type"] == "track4_action_graph_probe"
    assert payload["claim_scope"] == "engineering_probe_not_thesis_claim"
    assert payload["safe_for_thesis_claim"] is False
    assert payload["action_separation_candidate"] is True
    assert payload["real_mean_action"] == 10.0
    assert payload["control_mean_action"] == 5.0
    assert payload["real_over_control_mean_action"] == 2.0
    assert {row["corpus"] for row in payload["rows"]} == {"real", "control_random", "control_shuffled"}


def test_action_graph_probe_summary_reports_observer_hysteresis_means_when_present(tmp_path: Path) -> None:
    real = _write_summary(
        tmp_path,
        10.0,
        "real",
        observer_transport_penalty=3.0,
        hysteresis_penalty=0.75,
    )
    random = _write_summary(
        tmp_path,
        4.0,
        "control_random",
        observer_transport_penalty=1.0,
        hysteresis_penalty=0.25,
    )
    shuffled = _write_summary(
        tmp_path,
        6.0,
        "control_shuffled",
        observer_transport_penalty=2.0,
        hysteresis_penalty=0.50,
    )

    payload = summarize([real, random, shuffled])
    rows_by_corpus = {row["corpus"]: row for row in payload["rows"]}

    assert payload["observer_simplex_contract_supported"] is True
    assert payload["observer_simplex_row_count"] == 3
    assert payload["real_mean_observer_transport_penalty"] == 3.0
    assert payload["control_mean_observer_transport_penalty"] == 1.5
    assert payload["real_mean_hysteresis_penalty"] == 0.75
    assert payload["control_mean_hysteresis_penalty"] == 0.375
    assert rows_by_corpus["real"]["mean_observer_transport_penalty"] == 3.0
    assert rows_by_corpus["real"]["mean_hysteresis_penalty"] == 0.75
    assert all(row["observer_simplex_contract_supported"] is True for row in payload["rows"])
