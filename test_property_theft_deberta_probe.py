from pathlib import Path

import numpy as np

import scripts.property_theft_deberta_probe as probe
from scripts.property_theft_microprobe import _articles


def test_property_theft_deberta_probe_reports_lift_without_loading_model(monkeypatch, tmp_path: Path):
    articles = _articles()

    def fake_tfidf(_texts):
        # Topic-heavy baseline: property and theft remain close within the same event.
        return np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.1],
                [1.0, 0.1],
                [1.0, 0.2],
                [1.0, 0.2],
                [1.0, 0.0],
                [1.0, 0.15],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.9],
                [0.0, 0.9],
            ],
            dtype=np.float32,
        )

    def fake_deberta(_articles, *, model_name, max_length, cache_path, force=False):
        vectors = []
        for article in articles:
            if article.frame == "property":
                vectors.append([1.0, 0.0, 0.0])
            elif article.frame == "theft":
                vectors.append([0.0, 1.0, 0.0])
            elif article.frame == "bridge":
                vectors.append([0.6, 0.6, 0.0])
            else:
                vectors.append([0.8, 0.8, 0.5])
        arr = np.asarray(vectors, dtype=np.float32)
        return {
            "logits_flat": arr,
            "cls_stacked": arr,
            "bot_norms": arr,
            "spectral_pc1": arr[:, :1],
        }

    monkeypatch.setattr(probe, "_tfidf_features", fake_tfidf)
    monkeypatch.setattr(probe, "_extract_deberta_features", fake_deberta)

    summary = probe.run_property_theft_deberta_probe(
        tmp_path / "probe",
        model_name="fake-deberta",
    )

    assert summary["passes_microprobe"] is True
    assert summary["primary_cross_minus_same_cosine"] > 0.0
    assert summary["primary_lift_over_tfidf"] > 0.0
    assert (tmp_path / "probe" / "property_theft_deberta_summary.json").exists()
    assert (tmp_path / "probe" / "property_theft_corpus.jsonl").exists()
    assert (tmp_path / "probe" / "labels" / "hidden_groups.csv").exists()
