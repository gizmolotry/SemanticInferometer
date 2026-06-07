from __future__ import annotations

import json
from pathlib import Path

from scripts.run_track4_engineering_ledger import build_ledger


def _artifact(
    path: Path,
    *,
    top_config_id: str,
    top_gap: float,
    top_relative_gap: float,
    feature_raw_gap: float,
    feature_rks_gap: float,
    norm_z_gap: float,
    norm_global_gap: float,
    pair_farthest_gap: float,
    pair_nearest_gap: float,
    pair_mixed_gap: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic_type": "observer_transport_engineering_ablation",
        "screen": {"row_count": 2, "config_summaries": []},
        "confirm": {
            "row_count": 2,
            "axis_summary": {
                "feature_source": {
                    "raw_cls": {"mean_real_control_gap": feature_raw_gap, "mean_engineering_score": feature_raw_gap},
                    "rks": {"mean_real_control_gap": feature_rks_gap, "mean_engineering_score": feature_rks_gap},
                },
                "normalization": {
                    "zscore_per_slice": {"mean_real_control_gap": norm_z_gap},
                    "global_zscore": {"mean_real_control_gap": norm_global_gap},
                },
                "pair_mode": {
                    "farthest": {"mean_real_control_gap": pair_farthest_gap},
                    "nearest": {"mean_real_control_gap": pair_nearest_gap},
                    "mixed": {"mean_real_control_gap": pair_mixed_gap},
                },
            },
            "config_summaries": [
                {
                    "config_id": top_config_id,
                    "config": {
                        "feature_source": "raw_cls" if top_config_id.startswith("raw_cls") else "rks",
                        "normalization": "zscore_per_slice",
                        "pair_mode": "farthest" if "farthest" in top_config_id else "mixed",
                    },
                    "real_minus_control_mean_excess_holonomy_action": top_gap,
                    "real_minus_control_mean_relative_holonomy": top_relative_gap,
                }
            ],
        },
        "failures": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scale_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic_type": "observer_slice_transport_scale_suite",
        "runs": [
            {"corpus": "real", "kernel": "rbf", "cell_id": "real", "field_sources": {"density_source": "default_one"}},
            {
                "corpus": "control_random",
                "kernel": "matern",
                "cell_id": "control_random",
                "field_sources": {"density_source": "default_one"},
            },
            {
                "corpus": "synthetic",
                "kernel": "rbf",
                "cell_id": "rbf_seed42",
                "field_sources": {"density_source": "dirichlet_curvature_participation_ratio"},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_track4_engineering_ledger_writes_six_points(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    big = tmp_path / "big.json"
    actual = tmp_path / "actual.json"
    no_far = tmp_path / "no_far.json"
    scale = tmp_path / "scale.json"
    _artifact(
        baseline,
        top_config_id="rks|dim=512|norm=none|pairs=mixed|w=default_action|null=zero",
        top_gap=0.02,
        top_relative_gap=0.08,
        feature_raw_gap=0.01,
        feature_rks_gap=0.02,
        norm_z_gap=0.02,
        norm_global_gap=0.02,
        pair_farthest_gap=0.0,
        pair_nearest_gap=0.01,
        pair_mixed_gap=0.02,
    )
    _artifact(
        big,
        top_config_id="raw_cls|dim=512|norm=zscore_per_slice|pairs=farthest|w=switch_heavy|null=zero",
        top_gap=40.0,
        top_relative_gap=0.6,
        feature_raw_gap=20.0,
        feature_rks_gap=5.0,
        norm_z_gap=24.0,
        norm_global_gap=21.0,
        pair_farthest_gap=20.0,
        pair_nearest_gap=0.3,
        pair_mixed_gap=0.2,
    )
    _artifact(
        actual,
        top_config_id="raw_cls|dim=512|norm=zscore_per_slice|pairs=farthest|w=switch_heavy|null=independent_article_shuffle",
        top_gap=39.0,
        top_relative_gap=0.35,
        feature_raw_gap=18.0,
        feature_rks_gap=6.0,
        norm_z_gap=14.0,
        norm_global_gap=18.0,
        pair_farthest_gap=15.0,
        pair_nearest_gap=0.0,
        pair_mixed_gap=0.0,
    )
    _artifact(
        no_far,
        top_config_id="rks|dim=128|norm=zscore_per_slice|pairs=nearest|w=switch_heavy|null=dimension_signflip_by_slice",
        top_gap=0.9,
        top_relative_gap=0.04,
        feature_raw_gap=-0.1,
        feature_rks_gap=0.3,
        norm_z_gap=0.29,
        norm_global_gap=-0.1,
        pair_farthest_gap=0.0,
        pair_nearest_gap=0.38,
        pair_mixed_gap=0.28,
    )
    _scale_artifact(scale)

    payload = build_ledger(
        baseline_path=baseline,
        big_path=big,
        actual_null_path=actual,
        no_farthest_path=no_far,
        scale_artifact_paths=[scale],
        output_dir=tmp_path / "out",
    )

    assert payload["summary_type"] == "track4_engineering_ledger"
    assert payload["summary"]["entry_count"] == 6
    ids = {entry["point_id"] for entry in payload["entries"]}
    assert "track4_density_wiring" in ids
    density = next(entry for entry in payload["entries"] if entry["point_id"] == "track4_density_wiring")
    assert density["status"] == "blocked_by_flat_real_control_density"
    assert density["engineering_safe"] is False
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["csv"]).exists()
    claim_matrix = json.loads(Path(payload["artifacts"]["claim_matrix"]).read_text(encoding="utf-8"))
    assert len(claim_matrix["claims"]) == 6
    assert all(claim["claim_type"] == "engineering_hypothesis" for claim in claim_matrix["claims"])
