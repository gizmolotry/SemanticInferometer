from __future__ import annotations

import json
from pathlib import Path

from scripts.run_track2_component_ablation_microtest import (
    CLAIM_SCOPE,
    DIAGNOSTIC_TYPE,
    build_artifact,
    build_records,
    build_synthetic_fixture,
    default_thresholds,
    detect_microtest_effect,
    kernel_label_metrics,
    write_artifact,
)


def _records_by_variant() -> dict[str, dict]:
    return {record["variant"]: record for record in build_records()}


def test_kernel_metrics_detect_label_geometry() -> None:
    fixture = build_synthetic_fixture()
    records = _records_by_variant()

    track2_topic = records["track2_only"]["topic"]
    track15_stance = records["track15_only"]["stance"]

    assert len(fixture.topic_labels) == 24
    assert track2_topic["same_different_similarity_separation"] > 0.50
    assert track2_topic["nearest_neighbor_label_agreement"] == 1.0
    assert track15_stance["same_different_similarity_separation"] > 0.80
    assert track15_stance["nearest_neighbor_label_agreement"] == 1.0


def test_expected_synthetic_component_ablation_relationships() -> None:
    records = _records_by_variant()
    thresholds = default_thresholds()

    track2 = records["track2_only"]
    track15 = records["track15_only"]
    hadamard = records["hadamard_track2_track15"]
    no_track2 = records["no_track2_null"]

    assert track2["topic_separation"] > no_track2["topic_separation"]
    assert (
        track2["topic"]["nearest_neighbor_label_agreement"]
        > track15["topic"]["nearest_neighbor_label_agreement"]
    )
    assert (
        track15["stance"]["nearest_neighbor_label_agreement"]
        > track2["stance"]["nearest_neighbor_label_agreement"]
    )
    assert hadamard["joint_separation"] > 0.0
    assert hadamard["joint"]["nearest_neighbor_label_agreement"] == 1.0
    assert detect_microtest_effect(list(records.values()), thresholds)


def test_kernel_label_metrics_do_not_require_signed_embedding_coordinates() -> None:
    fixture = build_synthetic_fixture()
    records = _records_by_variant()
    metrics = kernel_label_metrics(
        records["hadamard_track2_track15"]["joint"]["same_similarity_mean"]
        * __import__("torch").eye(len(fixture.joint_labels), dtype=fixture.track2.dtype),
        fixture.joint_labels,
    )

    assert set(metrics) == {
        "same_similarity_mean",
        "different_similarity_mean",
        "same_different_similarity_separation",
        "nearest_neighbor_label_agreement",
    }


def test_json_artifact_schema_and_write(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_path.name == "track2_component_ablation_microtest.json"
    assert artifact["schema_version"] == "1.0"
    assert artifact["diagnostic_type"] == DIAGNOSTIC_TYPE
    assert artifact["claim_scope"] == CLAIM_SCOPE
    assert artifact["metric_basis"] == "kernel_similarity"
    assert artifact["foundation_audit_consumable"] is False
    assert artifact["component_level_claim_safe"] is False
    assert artifact["safe_for_thesis_claim"] is False
    assert artifact["supports_followup_priority"] is True
    assert artifact["failure_reasons"] == []
    assert "real corpus" in artifact["interpretation"]
    assert artifact["microtest_effect_detected"] is True
    assert {record["variant"] for record in artifact["records"]} == {
        "track2_only",
        "track15_only",
        "hadamard_track2_track15",
        "no_track2_null",
    }
    assert artifact["variants_evaluated"] == [
        "track2_only",
        "track15_only",
        "hadamard_track2_track15",
        "no_track2_null",
    ]
    assert set(artifact["thresholds"]) == set(default_thresholds())


def test_build_artifact_accepts_explicit_records() -> None:
    records = build_records()
    artifact = build_artifact(records)

    assert artifact["records"] == records
    assert artifact["microtest_effect_detected"] is True
