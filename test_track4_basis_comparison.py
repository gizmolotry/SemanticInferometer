import json

from scripts.compile_track4_basis_comparison import compile_basis_comparison


def _write_summary(path, feature_key, robust_score, peak_score):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "feature_key": feature_key,
                "condition_count": 2,
                "compact_artifacts": True,
                "method_recommendation": {
                    "decision": "keep_robust_default_and_report_peak_ablation",
                    "recommended_default_proposal_mode": "metric_softmax",
                    "best_peak_proposal_mode": "committor_guided",
                    "best_robust_mode": {
                        "mean_score": robust_score,
                        "std_score": 0.1,
                        "safe_rate": 1.0,
                        "mean_closed_loop_rate": 0.8,
                    },
                    "best_peak_condition": {
                        "proposal_quality_score": peak_score,
                        "k_neighbors": 10,
                        "seed": 42,
                        "temperature": 0.75,
                        "gamma": 5.0,
                        "path_shape_entropy_norm": 1.0,
                        "path_edge_entropy_norm": 0.8,
                        "reactive_flux_total": 0.01,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_compile_track4_basis_comparison_orders_by_robust_score(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_summary(first, "cls_stacked", 0.8, 0.9)
    _write_summary(second, "logits_flat", 0.9, 0.91)

    report = compile_basis_comparison([first, second], tmp_path / "out")

    assert report["status"] == "OK"
    assert [row["feature_key"] for row in report["ranked_bases"]] == ["logits_flat", "cls_stacked"]
    assert (tmp_path / "out" / "track4_basis_comparison.json").exists()
    assert (tmp_path / "out" / "track4_basis_comparison.csv").exists()
