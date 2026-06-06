from pathlib import Path
import json

from analysis.verification.thesis_evidence import build_thesis_evidence
from thesis_test_support import build_canonical_fixture, write_synthetic_manifest


def test_canonical_run_freeze_detects_missing_protocol_coverage(tmp_path: Path):
    fixture = build_canonical_fixture(
        tmp_path,
        kernels=["rbf", "laplacian", "rq"],
        channels=["cls"],
        corpora=["real", "control_constant", "control_shuffled"],
        seeds=[42, 420],
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    freeze = payloads["scientific_validation_summary"]["canonical_freeze"]
    assert not freeze["thesis_safe"]
    assert "imq" in freeze["missing_protocol_coverage"]["kernels"]
    assert "logits" in freeze["missing_protocol_coverage"]["channels"]
    assert "control_random" in freeze["missing_protocol_coverage"]["corpora"]
    assert 4200 in freeze["missing_protocol_coverage"]["seeds"]


def test_canonical_run_freeze_detects_missing_synthetic_protocol_coverage(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    freeze = payloads["scientific_validation_summary"]["canonical_freeze"]
    assert freeze["thesis_safe"]

    synthetic_manifest = fixture["synthetic_manifest"]
    assert synthetic_manifest is not None
    write_synthetic_manifest(
        synthetic_manifest,
        output_dir=synthetic_manifest.parent,
        kernels=["rbf", "laplacian", "rq"],
        seeds=[42, 420, 4200],
    )

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    freeze = payloads["scientific_validation_summary"]["canonical_freeze"]
    assert not freeze["thesis_safe"]
    assert "imq" in freeze["missing_synthetic_protocol_coverage"]["kernels"]


def test_canonical_run_freeze_requires_protocol_cross_product_cells(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path)
    for manifest_path in fixture["runs_dir"].rglob("experiment_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("experiment_type", "")).lower() == "synthetic":
            continue
        manifest["experiments"] = [
            experiment for experiment in manifest["experiments"]
            if not (
                experiment.get("kernel") == "imq"
                and experiment.get("channel") == "logits"
                and experiment.get("corpus") == "control_random"
            )
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    freeze = payloads["scientific_validation_summary"]["canonical_freeze"]
    assert not freeze["thesis_safe"]
    assert freeze["missing_protocol_coverage"] == {
        "kernels": [],
        "channels": [],
        "corpora": [],
        "seeds": [],
    }
    assert freeze["missing_protocol_cells_count"] == 3
    assert {
        "kernel": "imq",
        "channel": "logits",
        "corpus": "control_random",
        "seed": 42,
    } in freeze["missing_protocol_cells_sample"]


def test_canonical_run_freeze_requires_manifest_limit_to_meet_protocol(tmp_path: Path):
    fixture = build_canonical_fixture(tmp_path, methods_limit=120, manifest_limit=30)

    payloads = build_thesis_evidence(
        runs_dir=fixture["runs_dir"],
        methods_path=fixture["methods"],
        results_path=fixture["results"],
    )

    freeze = payloads["scientific_validation_summary"]["canonical_freeze"]
    assert not freeze["thesis_safe"]
    assert freeze["protocol_article_limit"] == 120
    assert freeze["under_protocol_limit_count"] == 2
    assert {
        "run_id": "experiments_20260504_010101",
        "manifest": str(fixture["suite_manifest"]),
        "observed_limit": 30,
        "required_limit": 120,
    } in freeze["under_protocol_limit_runs"]
