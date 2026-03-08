from __future__ import annotations

import numpy as np
import pytest

from analysis import waterfall_viz as wv


def _make_data(n: int = 12, d: int = 5) -> wv.WaterfallData:
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(n, d))
    return wv.WaterfallData(
        n_articles=n,
        t2_kernels={"z_rbf": feats},
    )


def test_tsne_method_uses_tsne_projection_when_available(monkeypatch):
    data = _make_data()

    monkeypatch.setattr(wv, "HAS_SKLEARN", True)
    monkeypatch.setattr(wv, "HAS_TSNE", True)

    class FailPCA:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, x):
            raise AssertionError("PCA should not run when method=tsne and TSNE is available")

    class StubTSNE:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, x):
            return np.full((x.shape[0], 2), 7.0, dtype=float)

    monkeypatch.setattr(wv, "PCA", FailPCA)
    monkeypatch.setattr(wv, "TSNE", StubTSNE)

    out = wv.compute_projections(data, method="tsne")

    assert out.cp2_2d is not None
    assert out.cp2_2d.shape == (data.n_articles, 2)
    assert np.all(out.cp2_2d == 7.0)


def test_tsne_method_warns_and_explicitly_falls_back_to_pca_when_tsne_unavailable(monkeypatch, capsys):
    data = _make_data()

    monkeypatch.setattr(wv, "HAS_SKLEARN", True)
    monkeypatch.setattr(wv, "HAS_TSNE", False)

    class StubPCA:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, x):
            return np.full((x.shape[0], 2), 3.0, dtype=float)

    monkeypatch.setattr(wv, "PCA", StubPCA)

    out = wv.compute_projections(data, method="tsne")
    captured = capsys.readouterr()

    assert "WARNING: --method tsne requested" in captured.out
    assert "falling back to PCA explicitly" in captured.out
    assert out.cp2_2d is not None
    assert np.all(out.cp2_2d == 3.0)


def test_tsne_method_errors_when_sklearn_missing(monkeypatch):
    data = _make_data()
    monkeypatch.setattr(wv, "HAS_SKLEARN", False)

    with pytest.raises(RuntimeError, match="--method tsne requested but scikit-learn is unavailable"):
        wv.compute_projections(data, method="tsne")
