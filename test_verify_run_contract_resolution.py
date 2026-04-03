import json
from pathlib import Path

import numpy as np
import torch

from analysis.verification.verify_run import discover_all_layers, resolve_validation_path


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
