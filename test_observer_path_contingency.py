from __future__ import annotations

import numpy as np

from core.observer_slice_transport import (
    ObserverSliceTransportConfig,
    summarize_observer_path_contingency,
)


def test_observer_path_contingency_records_enablers_and_blockers() -> None:
    slices = {
        "observer_a": np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "observer_b": np.asarray(
            [
                [0.0, 0.0],
                [1.1, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "observer_c": np.asarray(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    summary = summarize_observer_path_contingency(
        slices,
        article_pairs=[(0, 1), (1, 2), (0, 2)],
        config=ObserverSliceTransportConfig(),
        availability_action_cutoff=1.25,
        consensus_fraction=0.75,
        stable_cv_threshold=0.25,
    )

    assert summary["summary_type"] == "observer_path_contingency_summary"
    assert summary["status"] == "OK"
    assert summary["path_class_counts"]["observer_contingent_path"] == 2
    assert summary["path_class_counts"]["universal_barrier"] == 1
    first = next(row for row in summary["records"] if row["source_idx"] == 0 and row["target_idx"] == 1)
    assert first["path_class"] == "observer_contingent_path"
    assert first["enabled_by_observers"] == ["observer_a", "observer_b"]
    assert first["blocked_by_observers"] == ["observer_c"]
    assert first["path_requires_observer_subset"] is True
    remove_c = next(row for row in first["leave_one_out"] if row["removed_observer"] == "observer_c")
    assert remove_c["removal_effect"] == "unblocks_consensus"
    assert remove_c["path_class_without_observer"] == "consensus_path"
    assert summary["leave_one_out_observer_effects"]["observer_c"]["unblocks_consensus"] >= 1
    assert "null_observer_contingent_rate" in summary
    assert "observer_subset_required_rate" in summary


def test_subset_required_rate_counts_consensus_supported_by_subset() -> None:
    slices = {
        "observer_a": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "observer_b": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "observer_c": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "observer_d": np.asarray([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64),
    }

    summary = summarize_observer_path_contingency(
        slices,
        article_pairs=[(0, 1)],
        availability_action_cutoff=1.25,
        consensus_fraction=0.75,
        stable_cv_threshold=10.0,
    )

    assert summary["records"][0]["path_class"] == "consensus_path"
    assert summary["records"][0]["path_requires_observer_subset"] is True
    assert summary["observer_contingent_rate"] == 0.0
    assert summary["observer_subset_required_rate"] == 1.0


def test_path_contingency_rejects_invalid_thresholds() -> None:
    slices = {
        "observer_a": np.asarray([[0.0], [1.0]], dtype=np.float64),
        "observer_b": np.asarray([[0.0], [1.0]], dtype=np.float64),
    }

    try:
        summarize_observer_path_contingency(slices, article_pairs=[(0, 1)], availability_quantile=1.5)
    except ValueError as exc:
        assert "availability_quantile" in str(exc)
    else:
        raise AssertionError("expected invalid availability_quantile to fail")
