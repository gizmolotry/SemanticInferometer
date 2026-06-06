import json
from pathlib import Path

import numpy as np
import torch

from analysis.verification.verify_run import check_control_ordering, discover_all_layers, resolve_validation_path


def _write_observer_payload(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        "provenance": {
            "basis_hash": "basis-hash",
            "crn_seed": 12345,
            "alpha": 1.0,
            "weights_hash": "weights-hash",
        },
        "meta": {
            "provenance": {
                "basis_hash": "basis-hash",
                "crn_seed": 12345,
                "alpha": 1.0,
                "weights_hash": "weights-hash",
            }
        },
    }
    torch.save(payload, path)


def test_discover_all_layers_skips_relativity_cache_and_observer_global(tmp_path):
    exp_dir = tmp_path / "experiments_20260403_000000"
    real_dir = exp_dir / "matern" / "cls" / "real"
    rel_dir = real_dir / "relativity_cache"
    real_dir.mkdir(parents=True, exist_ok=True)
    rel_dir.mkdir(parents=True, exist_ok=True)

    _write_observer_payload(real_dir / "observer_42.pt")
    _write_observer_payload(real_dir / "observer_global.pt")
    _write_observer_payload(rel_dir / "observer_0.pt")

    layers = discover_all_layers(exp_dir)

    assert len(layers) == 1
    layer = layers[0]
    assert "real" in layer["artifacts"]
    assert "relativity_cache" not in layer["artifacts"]
    assert set(layer["artifacts"]["real"].keys()) == {"42"}


def test_resolve_validation_path_prefers_real_leaf_validation(tmp_path):
    layer_dir = tmp_path / "matern" / "cls"
    real_validation = layer_dir / "real" / "validation.json"
    control_validation = layer_dir / "control_random" / "validation.json"
    real_validation.parent.mkdir(parents=True, exist_ok=True)
    control_validation.parent.mkdir(parents=True, exist_ok=True)
    real_validation.write_text(json.dumps({"nmi": 0.72}, indent=2), encoding="utf-8")
    control_validation.write_text(json.dumps({"nmi": None}, indent=2), encoding="utf-8")

    resolved = resolve_validation_path(layer_dir, corpus="real")

    assert resolved == real_validation


def test_check_control_ordering_uses_control_metric_fallback_for_collapsed_primary_scores(tmp_path):
    layer_dir = tmp_path / "matern" / "cls"
    (layer_dir / "real").mkdir(parents=True, exist_ok=True)
    (layer_dir / "real" / "control_metrics.comprehensive_results.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "procrustes_min_control_ratio": 2.4,
                    "simple_variance_stochastic_ratio": 1.3,
                    "separates_count": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "real": {
            "42": {
                "features": np.array([[0.0, 0.0], [1.0e-5, 0.0]], dtype=np.float64),
            }
        },
        "control_shuffled": {
            "42": {
                "features": np.array([[0.0, 0.0], [2.0e-5, 0.0]], dtype=np.float64),
            }
        },
        "control_random": {
            "42": {
                "features": np.array([[0.0, 0.0], [3.0e-5, 0.0]], dtype=np.float64),
            }
        },
        "control_constant": {
            "42": {
                "features": np.ones((2, 2), dtype=np.float64),
            }
        },
    }

    result = check_control_ordering(artifacts, is_comparable=True, layer_dir=layer_dir)

    assert result["pass"] is True
    assert result["values"]["numeric_collapse_detected"] is True
    assert result["values"]["used_fallback"] is True
    assert result["values"]["fallback"]["metric"] == "comprehensive_control_metrics"


def test_check_control_ordering_keeps_failure_when_fallback_is_not_supportive(tmp_path):
    layer_dir = tmp_path / "matern" / "cls"
    (layer_dir / "real").mkdir(parents=True, exist_ok=True)
    (layer_dir / "real" / "control_metrics.comprehensive_results.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "procrustes_min_control_ratio": 0.9,
                    "simple_variance_stochastic_ratio": 0.95,
                    "separates_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "real": {
            "42": {
                "features": np.array([[0.0, 0.0], [1.0e-5, 0.0]], dtype=np.float64),
            }
        },
        "control_shuffled": {
            "42": {
                "features": np.array([[0.0, 0.0], [2.0e-5, 0.0]], dtype=np.float64),
            }
        },
        "control_random": {
            "42": {
                "features": np.array([[0.0, 0.0], [3.0e-5, 0.0]], dtype=np.float64),
            }
        },
        "control_constant": {
            "42": {
                "features": np.ones((2, 2), dtype=np.float64),
            }
        },
    }

    result = check_control_ordering(artifacts, is_comparable=True, layer_dir=layer_dir)

    assert result["pass"] is False
    assert result["values"]["numeric_collapse_detected"] is True
    assert result["values"]["used_fallback"] is False
    assert result["values"]["fallback"]["pass"] is False
