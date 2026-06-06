from __future__ import annotations

import json
from pathlib import Path

import run_full_experiment_suite as suite
from analysis.regression.precompute_observer_artifacts import (
    _render_focused,
    _emit_relativity_sidecars,
    _remap_payload_relativity_sidecars,
    _validate_focused_view_state,
)
from analysis.verification import scientific_summaries


def test_remap_payload_relativity_sidecars_writes_article_id_files_without_losing_collisions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rel_dir = run_dir / "relativity_cache"
    rel_dir.mkdir(parents=True)
    for prefix in ("state", "delta"):
        for row_idx in (0, 1):
            (rel_dir / f"{prefix}_{row_idx}.json").write_text(
                json.dumps(
                    {
                        "observer_id": row_idx,
                        "paths": [f"observer_{row_idx}/MONOLITH.html"],
                        "provenance": {"source": "observer_payload_relativity_v1"},
                    }
                ),
                encoding="utf-8",
            )

    result = _remap_payload_relativity_sidecars(run_dir, {0: 1, 1: 10})

    assert result["status"] == "OK"
    assert not (rel_dir / "state_0.json").exists()
    assert not (rel_dir / "delta_0.json").exists()
    state_one = json.loads((rel_dir / "state_1.json").read_text(encoding="utf-8"))
    state_ten = json.loads((rel_dir / "state_10.json").read_text(encoding="utf-8"))
    delta_ten = json.loads((rel_dir / "delta_10.json").read_text(encoding="utf-8"))
    assert state_one["observer_id"] == 1
    assert state_one["paths"] == ["observer_1/MONOLITH.html"]
    assert state_ten["observer_id"] == 10
    assert state_ten["paths"] == ["observer_10/MONOLITH.html"]
    assert state_ten["provenance"]["observer_row_index"] == 1
    assert state_ten["provenance"]["observer_article_idx"] == 10
    assert delta_ten["observer_id"] == 10


def test_emit_relativity_sidecars_marks_invalid_summary_not_ok(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "MONOLITH_DATA.csv").write_text("index,title\n7,Article\n", encoding="utf-8")

    monkeypatch.setattr(suite, "_emit_relativity_defaults", lambda *_args, **_kwargs: {"mode": "suite-default"})
    monkeypatch.setattr(suite, "_emit_relativity_deltas_json", lambda run_dir_arg: Path(run_dir_arg) / "relativity_deltas.json")

    def _write_invalid_summary(run_dir_arg: Path) -> Path:
        out = Path(run_dir_arg) / "observer_relativity_summary.json"
        out.write_text(
            json.dumps(
                {
                    "status": "INVALID",
                    "safe_for_thesis_claim": False,
                    "failure_reasons": ["observer relativity produced no nonzero coordinate displacement"],
                }
            ),
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr(scientific_summaries, "write_observer_relativity_summary", _write_invalid_summary)

    result = _emit_relativity_sidecars(run_dir, observer_indices=[7])

    assert result["status"] == "invalid"
    assert result["summary_status"] == "INVALID"
    assert result["summary_safe_for_thesis_claim"] is False
    assert result["summary_failure_reasons"] == ["observer relativity produced no nonzero coordinate displacement"]


def test_render_focused_rejects_failed_local_materialization(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    import run_full_experiment_suite as suite_mod

    monkeypatch.setattr(suite_mod, "_ensure_observer_universes_materialized", lambda *_args, **_kwargs: {"status": "failed", "reason": "boom"})

    try:
        _render_focused(run_dir, run_dir / "observer_7" / "MONOLITH.html", observer_idx=7, strict=False)
    except RuntimeError as exc:
        assert "requires local observer materialization" in str(exc)
    else:
        raise AssertionError("_render_focused should fail when local materialization fails")


def test_validate_focused_view_state_rejects_global_or_sidecar_fallback(tmp_path: Path) -> None:
    output = tmp_path / "observer_7" / "MONOLITH.html"
    output.parent.mkdir(parents=True)
    output.with_suffix(".view_state.json").write_text(
        json.dumps(
            {
                "observer_focus": {
                    "idx": 7,
                    "recenter_mode": "sidecar_coordinate_override",
                    "local_track_recompute_active": False,
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        _validate_focused_view_state(output, observer_idx=7)
    except RuntimeError as exc:
        assert "degraded to non-local observer geometry" in str(exc)
    else:
        raise AssertionError("_validate_focused_view_state should reject non-local focused artifacts")
