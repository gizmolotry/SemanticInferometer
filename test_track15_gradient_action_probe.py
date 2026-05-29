import numpy as np

from core.track4_action_graph import ActionGraphConfig
from scripts.run_track15_gradient_action_probe import run_probe


def _feature_tensor(offset: float, *, n_items: int = 8, n_bots: int = 3, hidden: int = 4) -> np.ndarray:
    rng = np.random.default_rng(123)
    base = rng.normal(loc=offset, scale=0.1, size=(n_items, n_bots, hidden)).astype(np.float32)
    trend = np.linspace(0.0, 1.0, n_items, dtype=np.float32).reshape(n_items, 1, 1)
    return base + trend


def test_track15_gradient_action_probe_runs_on_npz_features(tmp_path):
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        forward_delta__real=_feature_tensor(0.4),
        forward_delta__shuffled=_feature_tensor(0.2),
        forward_delta__random=_feature_tensor(0.1),
        anchor_gradient_delta__real=_feature_tensor(0.3),
        anchor_gradient_delta__shuffled=_feature_tensor(0.05),
        anchor_gradient_delta__random=_feature_tensor(0.15),
    )

    summary = run_probe(
        features_path=features_path,
        output_dir=tmp_path / "out",
        methods=("forward_delta", "anchor_gradient_delta"),
        corpora=("real", "shuffled", "random"),
        null_corpus="shuffled",
        config=ActionGraphConfig(
            k_neighbors=3,
            action_mode="null_calibrated_hysteresis",
            max_paths=4,
            target_count_per_anchor=1,
        ),
    )

    assert summary["summary_type"] == "track15_gradient_action_probe"
    assert len(summary["rows"]) == 6
    assert {row["method"] for row in summary["rows"]} == {"forward_delta", "anchor_gradient_delta"}
    assert summary["method_scores"]["forward_delta"]["mean_action_abs_log_ratio_stochastic_controls"] is not None
    assert (tmp_path / "out" / "track15_gradient_action_probe_summary.json").exists()
    assert (tmp_path / "out" / "track15_gradient_action_rows.csv").exists()
    assert (tmp_path / "out" / "track15_gradient_action_comparisons.csv").exists()
