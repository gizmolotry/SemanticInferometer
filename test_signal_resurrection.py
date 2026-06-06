from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from core.complete_pipeline import BeliefTransformerPipeline, initialize_full_pipeline


def _load_articles(limit: int = 30) -> list[dict]:
    corpus_path = Path("data/control_corpus.jsonl")
    if not corpus_path.exists():
        pytest.skip("control_corpus.jsonl missing")
    with corpus_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for i, line in enumerate(f) if i < limit]


def _run_signal_resurrection_probe() -> dict:
    articles = _load_articles()
    components = initialize_full_pipeline(
        random_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        kernel_type="matern",
        use_cls_tokens=True,
        use_dirichlet_fusion=True,
        dirichlet_rks_dim=256,
        dirichlet_n_observers=8,
        dirichlet_alpha=1.0,
        dirichlet_hidden_dim=1536,
        mix_in_rkhs=True,
        geometry_mode="rks",
        normalize_features=True,
    )

    pipeline = BeliefTransformerPipeline(
        components=components,
        random_seed=42,
        enable_provenance=False,
    )
    result = pipeline.process_month(articles=articles, month_name="signal_test")
    diagnostics = result.get("diagnostics", {}) or {}
    phase_space = diagnostics.get("phase_space", {}) or {}
    singularity_counts = result.get("singularity_counts", {}) or {}
    summary = {
        "n_articles": len(articles),
        "has_integrated_vectors": "integrated_vectors" in result,
        "integrated_dim": int(phase_space.get("integrated_dim", 0) or 0),
        "n_particles": int(phase_space.get("n_particles", 0) or 0),
        "consensus": int(singularity_counts.get("consensus", 0) or 0),
        "structural_singularity": int(singularity_counts.get("structural_singularity", 0) or 0),
        "noise": int(singularity_counts.get("noise", 0) or 0),
        "ideological_barrier": int(singularity_counts.get("ideological_barrier", 0) or 0),
        "mean_liar_score": float(result.get("mean_liar_score", 0.0) or 0.0),
        "n_liars": int(result.get("n_liars", 0) or 0),
    }
    return summary


def test_signal_resurrection_verification_contract():
    summary = _run_signal_resurrection_probe()

    assert summary["n_articles"] == 30
    assert summary["has_integrated_vectors"]
    assert summary["integrated_dim"] > 0
    assert summary["n_particles"] > 0
    assert summary["consensus"] + summary["structural_singularity"] + summary["noise"] + summary["ideological_barrier"] == summary["n_particles"]
    assert math.isfinite(summary["mean_liar_score"])


if __name__ == "__main__":
    result = _run_signal_resurrection_probe()
    print("=" * 60)
    print("SIGNAL RESURRECTION VERIFICATION TEST")
    print("=" * 60)
    print(json.dumps(result, indent=2))
