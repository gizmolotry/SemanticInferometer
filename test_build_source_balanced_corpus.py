import json
from pathlib import Path

import pytest

from scripts.build_source_balanced_corpus import build_source_balanced_slice, main


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture_rows():
    rows = []
    for source, count in {
        "alpha.example": 6,
        "beta.example": 5,
        "gamma.example": 4,
        "delta.example": 3,
    }.items():
        for idx in range(count):
            rows.append(
                {
                    "id": f"{source}-{idx}",
                    "publisher": source,
                    "published_at": f"2024-12-{idx + 1:02d}",
                    "content": f"{source} article body {idx}",
                    "title": f"{source} title {idx}",
                }
            )
    return rows


def test_build_source_balanced_slice_selects_top_repeated_sources(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, _fixture_rows())

    rows, manifest = build_source_balanced_slice(
        corpus,
        n_sources=3,
        articles_per_source=4,
        selection="spread",
    )

    assert len(rows) == 12
    assert manifest["selected_sources"] == ["alpha.example", "beta.example", "gamma.example"]
    assert manifest["selected_source_counts"] == {
        "alpha.example": 4,
        "beta.example": 4,
        "gamma.example": 4,
    }
    assert all(row["source"] == row["publisher"] == row["source_proxy_label"] for row in rows)
    assert all("source_balanced_selection_policy" in row for row in rows)


def test_build_source_balanced_slice_respects_explicit_sources(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, _fixture_rows())

    rows, manifest = build_source_balanced_slice(
        corpus,
        n_sources=2,
        articles_per_source=3,
        sources=["gamma.example", "alpha.example"],
    )

    assert len(rows) == 6
    assert manifest["selected_sources"] == ["gamma.example", "alpha.example"]
    assert [row["source"] for row in rows[:3]] == ["gamma.example"] * 3


def test_build_source_balanced_slice_rejects_insufficient_sources(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, _fixture_rows())

    with pytest.raises(ValueError, match="eligible sources"):
        build_source_balanced_slice(corpus, n_sources=5, articles_per_source=4)


def test_build_source_balanced_cli_writes_jsonl_and_manifest(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    out = tmp_path / "balanced.jsonl"
    manifest_out = tmp_path / "balanced.manifest.json"
    _write_jsonl(corpus, _fixture_rows())

    code = main(
        [
            "--input",
            str(corpus),
            "--out",
            str(out),
            "--manifest-out",
            str(manifest_out),
            "--n-sources",
            "2",
            "--articles-per-source",
            "5",
        ]
    )

    assert code == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(written) == 10
    assert manifest["selected_source_count"] == 2
    assert manifest["selected_article_count"] == 10
    assert manifest["claim_boundary"]["source_labels_reserved_for_downstream_evaluation"] is True
