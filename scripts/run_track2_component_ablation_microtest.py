from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hadamard_fusion import HadamardFusion, HadamardFusionConfig


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "track2_component_ablation_microtest"
CLAIM_SCOPE = "microtest_only_not_foundation_ablation"


@dataclass(frozen=True)
class SyntheticFixture:
    track2: torch.Tensor
    track15: torch.Tensor
    topic_labels: list[str]
    stance_labels: list[str]
    joint_labels: list[str]


def build_synthetic_fixture() -> SyntheticFixture:
    """Create a small deterministic fixture with separable topic and stance factors."""
    rows: list[list[float]] = []
    stress_rows: list[list[float]] = []
    topic_labels: list[str] = []
    stance_labels: list[str] = []
    joint_labels: list[str] = []

    topic_basis = torch.eye(4, dtype=torch.float64) * 3.0
    stance_axis = {
        "pro": torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "con": torch.tensor([-2.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    }

    for topic_idx in range(4):
        for stance_idx, stance in enumerate(["pro", "con"]):
            for replicate in range(3):
                replicate_offset = (replicate - 1) * 0.04
                topic_vector = topic_basis[topic_idx].clone()
                topic_vector += torch.tensor(
                    [
                        replicate_offset,
                        -0.5 * replicate_offset,
                        0.25 * replicate_offset,
                        0.0,
                    ],
                    dtype=torch.float64,
                )

                shear = torch.tensor(
                    [
                        0.0,
                        0.08 * (replicate - 1),
                        0.04 * stance_idx,
                        0.02 * replicate * (1 if stance == "pro" else -1),
                    ],
                    dtype=torch.float64,
                )
                stance_vector = stance_axis[stance] + shear

                rows.append(topic_vector.tolist())
                stress_rows.append(stance_vector.tolist())
                topic_label = f"topic_{topic_idx}"
                topic_labels.append(topic_label)
                stance_labels.append(stance)
                joint_labels.append(f"{topic_label}_{stance}")

    return SyntheticFixture(
        track2=torch.tensor(rows, dtype=torch.float64),
        track15=torch.tensor(stress_rows, dtype=torch.float64),
        topic_labels=topic_labels,
        stance_labels=stance_labels,
        joint_labels=joint_labels,
    )


def _upper_triangle_label_separation(kernel: torch.Tensor, labels: list[str]) -> dict[str, float]:
    same: list[float] = []
    different: list[float] = []
    n_items = len(labels)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            value = float(kernel[i, j])
            if labels[i] == labels[j]:
                same.append(value)
            else:
                different.append(value)

    same_mean = sum(same) / len(same) if same else 0.0
    different_mean = sum(different) / len(different) if different else 0.0
    return {
        "same_similarity_mean": same_mean,
        "different_similarity_mean": different_mean,
        "same_different_similarity_separation": same_mean - different_mean,
    }


def _nearest_neighbor_label_agreement(kernel: torch.Tensor, labels: list[str]) -> float:
    n_items = len(labels)
    if n_items <= 1:
        return 0.0

    hits = 0
    for i in range(n_items):
        row = kernel[i].clone()
        row[i] = -torch.inf
        nearest = int(torch.argmax(row).item())
        hits += int(labels[i] == labels[nearest])
    return hits / n_items


def kernel_label_metrics(kernel: torch.Tensor, labels: list[str]) -> dict[str, float]:
    metrics = _upper_triangle_label_separation(kernel, labels)
    metrics["nearest_neighbor_label_agreement"] = _nearest_neighbor_label_agreement(kernel, labels)
    return metrics


def _variant_record(name: str, kernel: torch.Tensor, fixture: SyntheticFixture) -> dict[str, Any]:
    topic_metrics = kernel_label_metrics(kernel, fixture.topic_labels)
    stance_metrics = kernel_label_metrics(kernel, fixture.stance_labels)
    joint_metrics = kernel_label_metrics(kernel, fixture.joint_labels)
    return {
        "variant": name,
        "n_samples": len(fixture.topic_labels),
        "topic": topic_metrics,
        "stance": stance_metrics,
        "joint": joint_metrics,
        "topic_separation": topic_metrics["same_different_similarity_separation"],
        "stance_separation": stance_metrics["same_different_similarity_separation"],
        "joint_separation": joint_metrics["same_different_similarity_separation"],
    }


def build_records(fixture: SyntheticFixture | None = None) -> list[dict[str, Any]]:
    fixture = fixture or build_synthetic_fixture()
    config = HadamardFusionConfig(
        rks_dim=fixture.track2.shape[1],
        spectral_dim=fixture.track15.shape[1],
        output_dim=4,
        sigma_rks=1.0,
        sigma_spectral=1.0,
        kernel_floor=1e-12,
    )
    fusion = HadamardFusion(config)
    k_track2, k_track15 = fusion.compute_kernel_matrices(fixture.track2, fixture.track15)
    k_hadamard, _, _ = fusion.hadamard_product(k_track2, k_track15)
    k_null_track2 = torch.ones_like(k_track2)
    k_no_track2, _, _ = fusion.hadamard_product(k_null_track2, k_track15)

    variants = [
        ("track2_only", k_track2),
        ("track15_only", k_track15),
        ("hadamard_track2_track15", k_hadamard),
        ("no_track2_null", k_no_track2),
    ]
    return [_variant_record(name, kernel, fixture) for name, kernel in variants]


def _by_variant(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["variant"]): record for record in records}


def default_thresholds() -> dict[str, float]:
    return {
        "min_topic_drop_when_track2_removed": 0.25,
        "min_track2_topic_advantage_over_track15": 0.25,
        "min_track15_stance_advantage_over_track2": 0.25,
        "min_hadamard_joint_separation": 0.10,
        "min_hadamard_joint_nn_agreement": 0.95,
    }


def detect_microtest_effect(records: list[dict[str, Any]], thresholds: dict[str, float]) -> bool:
    variants = _by_variant(records)
    track2 = variants["track2_only"]
    track15 = variants["track15_only"]
    hadamard = variants["hadamard_track2_track15"]
    no_track2 = variants["no_track2_null"]

    return all(
        [
            track2["topic_separation"] - no_track2["topic_separation"]
            >= thresholds["min_topic_drop_when_track2_removed"],
            track2["topic"]["nearest_neighbor_label_agreement"]
            - track15["topic"]["nearest_neighbor_label_agreement"]
            >= thresholds["min_track2_topic_advantage_over_track15"],
            track15["stance"]["nearest_neighbor_label_agreement"]
            - track2["stance"]["nearest_neighbor_label_agreement"]
            >= thresholds["min_track15_stance_advantage_over_track2"],
            hadamard["joint_separation"] >= thresholds["min_hadamard_joint_separation"],
            hadamard["joint"]["nearest_neighbor_label_agreement"]
            >= thresholds["min_hadamard_joint_nn_agreement"],
        ]
    )


def build_artifact(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = records or build_records()
    thresholds = default_thresholds()
    microtest_effect_detected = detect_microtest_effect(records, thresholds)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "metric_basis": "kernel_similarity",
        "foundation_audit_consumable": False,
        "component_level_claim_safe": False,
        "safe_for_thesis_claim": False,
        "supports_followup_priority": True,
        "interpretation": (
            "This is a deterministic component microtest. It can verify that the "
            "Track 5 kernel assembly can preserve separable Track 2 topic geometry "
            "and Track 1.5 stance/shear geometry in a controlled fixture, but it "
            "does not establish Track 2 necessity on the real corpus."
        ),
        "failure_reasons": [] if microtest_effect_detected else ["microtest_thresholds_not_met"],
        "records": records,
        "thresholds": thresholds,
        "variants_evaluated": [str(record["variant"]) for record in records],
        "microtest_effect_detected": microtest_effect_detected,
    }


def write_artifact(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    artifact_path = output_dir / "track2_component_ablation_microtest.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return ROOT / "outputs" / "track2_component_ablation_microtest" / f"current_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for the JSON artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_path = write_artifact(args.output_dir)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    status = "effect_detected" if artifact["microtest_effect_detected"] else "effect_not_detected"
    print(f"status={status}")
    print(f"artifact_path={artifact_path}")
    return 0 if artifact["microtest_effect_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
