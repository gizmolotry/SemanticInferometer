import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.source_proxy_validation import main, summarize_results, validate_run_dir


def _write_source_fixture(run_dir: Path, *, clustered: bool = True) -> None:
    run_dir.mkdir(parents=True)
    sources = ["source_a"] * 6 + ["source_b"] * 6 + ["source_c"] * 6
    rows = [
        {
            "index": idx,
            "source": source,
            "publication": source,
            "title": f"{source}-{idx}",
        }
        for idx, source in enumerate(sources)
    ]
    pd.DataFrame(rows).to_csv(run_dir / "article_metadata.csv", index=False)
    if clustered:
        centers = {
            "source_a": np.array([0.0, 0.0, 0.0]),
            "source_b": np.array([8.0, 0.0, 0.0]),
            "source_c": np.array([0.0, 8.0, 0.0]),
        }
        features = np.vstack(
            [
                centers[source] + np.array([0.01 * (idx % 6), 0.0, 0.0])
                for idx, source in enumerate(sources)
            ]
        )
    else:
        rng = np.random.default_rng(123)
        features = rng.normal(size=(len(sources), 3))
    np.save(run_dir / "features.npy", features.astype(np.float32))


def test_source_proxy_validation_passes_clustered_repeated_sources(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_source_fixture(run_dir)

    payload = validate_run_dir(
        run_dir,
        min_source_count=3,
        min_sources=3,
        min_articles=18,
        permutations=40,
        min_effect=0.1,
        min_nn_excess=0.1,
        max_pvalue=0.10,
    )

    assert payload["status"] == "OK"
    assert payload["pass"] is True
    assert payload["thesis_safe"] is True
    assert payload["observed"]["same_source_mean_distance"] < payload["observed"]["different_source_mean_distance"]
    assert payload["p_values"]["source_distance_effect"] <= 0.10
    assert payload["p_values"]["nearest_neighbor_source_excess"] <= 0.10


def test_source_proxy_validation_rejects_sparse_source_replication(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"index": idx, "source": f"source_{idx}", "publication": f"source_{idx}"}
            for idx in range(12)
        ]
    ).to_csv(run_dir / "article_metadata.csv", index=False)
    np.save(run_dir / "features.npy", np.eye(12, dtype=np.float32))

    payload = validate_run_dir(run_dir, min_source_count=3, min_sources=3, min_articles=9)

    assert payload["status"] == "INSUFFICIENT_SOURCE_REPLICATION"
    assert payload["pass"] is False
    assert "insufficient_repeated_source_articles" in payload["failure_reasons"]
    assert "insufficient_repeated_source_count" in payload["failure_reasons"]


def test_source_proxy_summary_is_not_safe_when_any_run_is_insufficient(tmp_path: Path):
    good = tmp_path / "good"
    sparse = tmp_path / "sparse"
    _write_source_fixture(good)
    sparse.mkdir()
    pd.DataFrame([{"source": f"s{i}"} for i in range(8)]).to_csv(sparse / "article_metadata.csv", index=False)
    np.save(sparse / "features.npy", np.eye(8, dtype=np.float32))

    rows = [
        validate_run_dir(good, permutations=20, max_pvalue=0.15),
        validate_run_dir(sparse),
    ]
    summary = summarize_results(rows)

    assert summary["ok_run_count"] == 1
    assert summary["passing_run_count"] == 1
    assert summary["insufficient_source_run_count"] == 1
    assert summary["thesis_safe"] is False


def test_source_proxy_cli_writes_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    out = tmp_path / "summary.json"
    _write_source_fixture(run_dir)

    code = main(
        [
            "--run-dir",
            str(run_dir),
            "--permutations",
            "20",
            "--max-pvalue",
            "0.15",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["diagnostic_type"] == "source_proxy_validation"
    assert payload["thesis_safe"] is True
    assert payload["runs"][0]["pass"] is True
