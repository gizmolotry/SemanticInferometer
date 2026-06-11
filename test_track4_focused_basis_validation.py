import json
from pathlib import Path

import scripts.run_track4_focused_basis_validation as focused_basis
from scripts.run_track4_focused_basis_validation import (
    CorpusSpec,
    aggregate_validation,
    build_validation_cells,
    resolve_corpus_specs,
    write_validation_summary,
)


def test_resolve_corpus_specs_uses_existing_real_controls_and_synthetic_microprobe(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    microprobe_dir = tmp_path / "outputs" / "microprobes" / "property_theft" / "deberta_20260515"
    data_dir.mkdir(parents=True)
    microprobe_dir.mkdir(parents=True)
    for name in ("real_corpus", "control_shuffled", "control_random"):
        (data_dir / f"{name}.jsonl").write_text(json.dumps({"title": name, "text": "fixture"}) + "\n", encoding="utf-8")
    (microprobe_dir / "property_theft_corpus.jsonl").write_text(
        json.dumps({"title": "synthetic", "text": "fixture"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(focused_basis, "ROOT", tmp_path)

    specs = resolve_corpus_specs(["real", "control_shuffled", "control_random", "synthetic_microprobe"])
    by_name = {spec.name: spec for spec in specs}

    assert by_name["real"].corpus_arg == "real"
    assert by_name["control_shuffled"].corpus_arg == "control_shuffled"
    assert by_name["control_random"].corpus_arg == "control_random"
    assert by_name["synthetic_microprobe"].kind == "synthetic"
    assert Path(by_name["synthetic_microprobe"].corpus_arg).exists()


def test_build_validation_cells_creates_cross_product_with_cache_paths(tmp_path: Path):
    corpora = [
        CorpusSpec(name="real", corpus_arg="real", kind="real"),
        CorpusSpec(name="control_random", corpus_arg="control_random", kind="control_random"),
    ]

    cells = build_validation_cells(
        output_root=tmp_path,
        corpora=corpora,
        kernels=["rbf", "imq"],
        seeds=[42, 420],
    )

    assert len(cells) == 8
    assert cells[0].output_dir == tmp_path / "real" / "rbf" / "seed_42"
    assert cells[0].nli_cache_path == tmp_path / "_nli_cache" / "real_seed42.pt"
    assert cells[-1].output_dir == tmp_path / "control_random" / "imq" / "seed_420"


def test_aggregate_validation_keeps_instrumentation_separate_from_terrain_validity():
    cell_summaries = [
        {
            "status": "OK",
            "claim_boundary": {
                "instrumentation_supported": True,
                "terrain_validity_supported": False,
                "basis_superiority_supported": False,
                "recommended_basis": "logits_flat",
                "basis_score_margin": 0.001,
            },
        }
    ]
    rows = [
        {
            "corpus": "real",
            "corpus_kind": "real",
            "basis": "logits_flat",
            "basis_probe_score": 0.76,
            "safe_for_thesis_claim": False,
            "closed_loop_rate": 0.8,
            "primary_zone_count": 2,
            "mean_work_integral": 10.0,
            "failure_reasons": ["track 4 path-touched terrain coverage fewer than three terrain zones"],
        },
        {
            "corpus": "control_random",
            "corpus_kind": "control_random",
            "basis": "track2",
            "basis_probe_score": 0.70,
            "safe_for_thesis_claim": False,
            "closed_loop_rate": 0.9,
            "primary_zone_count": 2,
            "mean_work_integral": 8.0,
            "failure_reasons": [],
        },
    ]

    aggregate = aggregate_validation(cell_summaries=cell_summaries, rows=rows)

    assert aggregate["claim_boundary"]["instrumentation_supported"] is True
    assert aggregate["claim_boundary"]["terrain_validity_supported"] is False
    assert aggregate["claim_boundary"]["basis_superiority_supported"] is False
    assert aggregate["basis_summary"]["logits_flat"]["mean_score"] == 0.76
    assert aggregate["basis_summary"]["track2"]["mean_score"] == 0.70
    assert "track 4 path-touched terrain coverage fewer than three terrain zones" in aggregate["common_failure_reasons"]


def test_write_validation_summary_emits_json_csv_and_boundary(tmp_path: Path):
    corpus = CorpusSpec(name="real", corpus_arg="real", kind="real")
    cells = build_validation_cells(output_root=tmp_path, corpora=[corpus], kernels=["rbf"], seeds=[42])
    rows = [
        {
            "corpus": "real",
            "corpus_kind": "real",
            "kernel": "rbf",
            "seed": 42,
            "basis": "track2",
            "basis_probe_score": 0.8,
            "safe_for_thesis_claim": True,
            "closed_loop_rate": 1.0,
            "primary_zone_count": 3,
            "mean_work_integral": 12.0,
            "failure_reasons": [],
        }
    ]
    cell_summaries = [
        {
            "status": "OK",
            "corpus": "real",
            "kernel": "rbf",
            "seed": 42,
            "claim_boundary": {
                "instrumentation_supported": True,
                "terrain_validity_supported": True,
                "basis_superiority_supported": False,
            },
        }
    ]

    summary_path = write_validation_summary(
        tmp_path,
        cells=cells,
        cell_summaries=cell_summaries,
        rows=rows,
        bases=["track2"],
        kernels=["rbf"],
        seeds=[42],
        limit=60,
    )

    assert summary_path.exists()
    assert (tmp_path / "track4_focused_basis_validation_rows.csv").exists()
    assert (tmp_path / "track4_focused_basis_validation_claim_boundary.json").exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["claim_boundary"]["instrumentation_supported"] is True
