#!/usr/bin/env python3
"""Build a Track 4 engineering ledger from observer-transport ablations.

The ledger is deliberately engineering-scoped.  It does not promote stronger
paper claims; it records which concrete system changes are supported by recent
ablation artifacts and which ones are blocked by missing/inert pipeline fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "1.0"
SUMMARY_TYPE = "track4_engineering_ledger"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "track4_engineering_ledger" / "latest"
DEFAULT_BASELINE = (
    ROOT
    / "outputs"
    / "observer_transport_engineering_ablation"
    / "baseline_current_20260531_v1"
    / "observer_transport_engineering_ablation.json"
)
DEFAULT_BIG = (
    ROOT
    / "outputs"
    / "observer_transport_engineering_ablation"
    / "big_20260531_v1"
    / "observer_transport_engineering_ablation.json"
)
DEFAULT_ACTUAL_NULL = (
    ROOT
    / "outputs"
    / "observer_transport_engineering_ablation"
    / "actual_null_20260531_v1"
    / "observer_transport_engineering_ablation.json"
)
DEFAULT_NO_FARTHEST = (
    ROOT
    / "outputs"
    / "observer_transport_engineering_ablation"
    / "no_farthest_actual_null_20260531_v1"
    / "observer_transport_engineering_ablation.json"
)
DEFAULT_SCALE_ARTIFACTS = [
    ROOT
    / "outputs"
    / "observer_slice_transport_scale_suite"
    / "default_big_20260530_v3_fast"
    / "observer_slice_transport_scale_suite.json",
    ROOT
    / "outputs"
    / "observer_slice_transport_scale_suite"
    / "real_control_deep_20260530_v2"
    / "observer_slice_transport_scale_suite.json",
]


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _top_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    confirm = artifact.get("confirm") if isinstance(artifact.get("confirm"), Mapping) else {}
    screen = artifact.get("screen") if isinstance(artifact.get("screen"), Mapping) else {}
    summaries = confirm.get("config_summaries") or screen.get("config_summaries") or []
    if not summaries:
        return {}
    return dict(summaries[0]) if isinstance(summaries[0], Mapping) else {}


def _axis_stats(artifact: Mapping[str, Any], axis: str, value: str) -> dict[str, Any]:
    confirm = artifact.get("confirm") if isinstance(artifact.get("confirm"), Mapping) else {}
    screen = artifact.get("screen") if isinstance(artifact.get("screen"), Mapping) else {}
    axis_summary = confirm.get("axis_summary") or screen.get("axis_summary") or {}
    axis_rows = axis_summary.get(axis) if isinstance(axis_summary, Mapping) else {}
    row = axis_rows.get(value) if isinstance(axis_rows, Mapping) else {}
    return dict(row) if isinstance(row, Mapping) else {}


def _artifact_meta(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "diagnostic_type": payload.get("diagnostic_type") or payload.get("summary_type"),
        "screen_rows": ((payload.get("screen") or {}).get("row_count") if isinstance(payload.get("screen"), Mapping) else None),
        "confirm_rows": (
            (payload.get("confirm") or {}).get("row_count") if isinstance(payload.get("confirm"), Mapping) else None
        ),
        "failure_count": len(payload.get("failures") or []) if isinstance(payload.get("failures"), list) else None,
    }


def _density_source_audit(scale_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in scale_payloads:
        for run in payload.get("runs") or []:
            if not isinstance(run, Mapping):
                continue
            field_sources = run.get("field_sources") if isinstance(run.get("field_sources"), Mapping) else {}
            rows.append(
                {
                    "corpus": run.get("corpus"),
                    "kernel": run.get("kernel"),
                    "cell_id": run.get("cell_id"),
                    "density_source": field_sources.get("density_source"),
                    "stress_source": field_sources.get("stress_source"),
                    "payload_path": run.get("payload_path"),
                }
            )
    payload_key_rows = [_payload_density_key_audit(row.get("payload_path")) for row in rows[:24]]
    by_corpus: dict[str, list[str]] = {}
    for row in rows:
        by_corpus.setdefault(str(row.get("corpus") or "unknown"), []).append(str(row.get("density_source")))
    flat_real_control = all(
        set(by_corpus.get(corpus, [])) <= {"default_one"}
        for corpus in ("real", "control_random", "control_shuffled", "control_constant")
        if corpus in by_corpus
    )
    synthetic_sources = sorted(set(by_corpus.get("synthetic", [])))
    return {
        "row_count": len(rows),
        "density_sources_by_corpus": {key: sorted(set(values)) for key, values in sorted(by_corpus.items())},
        "real_control_density_flat_default_one": bool(flat_real_control),
        "synthetic_density_sources": synthetic_sources,
        "rows": rows[:24],
        "payload_density_key_audit": payload_key_rows,
    }


def _payload_density_key_audit(raw_path: Any) -> dict[str, Any]:
    path = Path(str(raw_path)) if raw_path else Path("__missing__")
    if not path.exists():
        return {"payload_path": str(raw_path), "status": "missing"}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"payload_path": str(path), "status": "unreadable", "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {"payload_path": str(path), "status": "not_mapping"}
    topology = payload.get("T3_topology")
    topology_keys = sorted(topology.keys()) if isinstance(topology, Mapping) else []
    top_keys = sorted(
        key
        for key in ("track3_density_rho", "dirichlet_curvature", "dirichlet_fused_std", "density", "rho")
        if key in payload
    )
    current_scale_reader_source = "default_one"
    if not isinstance(topology, Mapping) and topology is not None:
        current_scale_reader_source = "T3_topology_inverse_norm"
    elif isinstance(payload.get("dirichlet_curvature"), Mapping) and payload["dirichlet_curvature"].get(
        "participation_ratio"
    ) is not None:
        current_scale_reader_source = "dirichlet_curvature_participation_ratio"

    current_action_replay_source = "none"
    if isinstance(topology, Mapping):
        for key in ("dirichlet_fused_std", "fused_std", "density", "rho"):
            if key in topology:
                current_action_replay_source = f"T3_topology.{key}"
                break
    return {
        "payload_path": str(path),
        "status": "ok",
        "top_level_density_keys": top_keys,
        "t3_topology_type": type(topology).__name__ if topology is not None else None,
        "t3_topology_keys": topology_keys,
        "current_scale_suite_density_reader_source": current_scale_reader_source,
        "current_action_graph_replay_density_source": current_action_replay_source,
        "has_real_density_available_for_scale_suite_patch": bool(
            isinstance(topology, Mapping)
            and any(key in topology for key in ("dirichlet_fused_std", "fused_std", "density", "rho"))
        ),
    }


def _entry(
    *,
    point_id: str,
    title: str,
    status: str,
    engineering_safe: bool,
    evidence: Mapping[str, Any],
    inference: str,
    engineering_action: str,
    interpretation_trap: str,
    next_test: str,
    artifacts: Sequence[str],
) -> dict[str, Any]:
    return {
        "point_id": point_id,
        "title": title,
        "status": status,
        "engineering_safe": bool(engineering_safe),
        "evidence": dict(evidence),
        "inference": inference,
        "engineering_action": engineering_action,
        "interpretation_trap": interpretation_trap,
        "next_test": next_test,
        "artifacts": list(artifacts),
    }


def build_ledger(
    *,
    baseline_path: Path,
    big_path: Path,
    actual_null_path: Path,
    no_farthest_path: Path,
    scale_artifact_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_json(baseline_path)
    big = _load_json(big_path)
    actual_null = _load_json(actual_null_path)
    no_farthest = _load_json(no_farthest_path)
    scale_payloads = [_load_json(path) for path in scale_artifact_paths if Path(path).exists()]

    baseline_top = _top_summary(baseline)
    big_top = _top_summary(big)
    actual_top = _top_summary(actual_null)
    no_far_top = _top_summary(no_farthest)
    density_audit = _density_source_audit(scale_payloads)

    baseline_gap = _safe_float(baseline_top.get("real_minus_control_mean_excess_holonomy_action"))
    baseline_relative_gap = _safe_float(baseline_top.get("real_minus_control_mean_relative_holonomy"))
    actual_gap = _safe_float(actual_top.get("real_minus_control_mean_excess_holonomy_action"))
    actual_relative_gap = _safe_float(actual_top.get("real_minus_control_mean_relative_holonomy"))
    no_far_gap = _safe_float(no_far_top.get("real_minus_control_mean_excess_holonomy_action"))
    big_gap = _safe_float(big_top.get("real_minus_control_mean_excess_holonomy_action"))

    actual_raw = _axis_stats(actual_null, "feature_source", "raw_cls")
    actual_rks = _axis_stats(actual_null, "feature_source", "rks")
    no_far_raw = _axis_stats(no_farthest, "feature_source", "raw_cls")
    no_far_rks = _axis_stats(no_farthest, "feature_source", "rks")
    actual_zscore = _axis_stats(actual_null, "normalization", "zscore_per_slice")
    actual_global_zscore = _axis_stats(actual_null, "normalization", "global_zscore")
    no_far_zscore = _axis_stats(no_farthest, "normalization", "zscore_per_slice")
    actual_farthest = _axis_stats(actual_null, "pair_mode", "farthest")
    no_far_nearest = _axis_stats(no_farthest, "pair_mode", "nearest")
    no_far_mixed = _axis_stats(no_farthest, "pair_mode", "mixed")

    entries = [
        _entry(
            point_id="track4_true_null_calibration",
            title="Promote true null-calibrated excess action",
            status="supported_for_real_control_and_required_for_reporting",
            engineering_safe=bool(actual_gap is not None and baseline_gap is not None and actual_gap > baseline_gap),
            evidence={
                "baseline_zero_null_gap": baseline_gap,
                "baseline_zero_null_relative_gap": baseline_relative_gap,
                "strict_actual_null_top_gap": actual_gap,
                "strict_actual_null_top_relative_gap": actual_relative_gap,
                "strict_actual_null_top_config": actual_top.get("config_id"),
                "baseline_top_config": baseline_top.get("config_id"),
            },
            inference=(
                "Zero-null excess was mostly raw holonomy. Destructive nulls sharply suppress control floors "
                "while preserving a large real/control gap for long-range observer transport."
            ),
            engineering_action=(
                "Add null chart generation and calibrated excess fields to the primary observer-slice "
                "transport ledger, then stop treating zero-null raw holonomy as the main score."
            ),
            interpretation_trap=(
                "The best strict-null long-range config has weak synthetic pass rate, so this is an "
                "engineering signal for real/control rupture rather than a universal semantic-label claim."
            ),
            next_test="Run shared, independent, sign-flip, and observer-permutation nulls as first-class Track 4 cells.",
            artifacts=[str(baseline_path), str(actual_null_path)],
        ),
        _entry(
            point_id="track4_dual_engine_split",
            title="Split Track 4 into local navigation and long-range observer rupture",
            status="supported",
            engineering_safe=bool(actual_gap is not None and no_far_gap is not None and actual_gap > max(no_far_gap, 1e-12)),
            evidence={
                "strict_farthest_enabled_gap": actual_gap,
                "strict_no_farthest_gap": no_far_gap,
                "gap_ratio_farthest_to_no_farthest": (
                    float(actual_gap / max(no_far_gap, 1e-12))
                    if actual_gap is not None and no_far_gap is not None
                    else None
                ),
                "big_lenient_top_gap": big_gap,
                "no_farthest_top_config": no_far_top.get("config_id"),
            },
            inference=(
                "Removing farthest pairs does not kill the effect, but it changes its scale by orders of "
                "magnitude. A single generic walker is mixing two physical questions."
            ),
            engineering_action=(
                "Create separate Track 4 modes: local_navigation for neighborhood/mixed paths and "
                "observer_rupture for landmark/farthest cross-chart transitions."
            ),
            interpretation_trap=(
                "Do not describe farthest-pair rupture as ordinary local traversal; it is a different "
                "operator over the observer atlas."
            ),
            next_test="Expose both engines in the DAG and require each to pass its own matched null suite.",
            artifacts=[str(actual_null_path), str(no_farthest_path), str(big_path)],
        ),
        _entry(
            point_id="track4_feature_routing_raw_cls_and_rks",
            title="Route raw observer CLS to rupture and RKS to local traversal",
            status="supported_but_mode_specific",
            engineering_safe=True,
            evidence={
                "strict_actual_null_raw_cls_mean_gap": actual_raw.get("mean_real_control_gap"),
                "strict_actual_null_rks_mean_gap": actual_rks.get("mean_real_control_gap"),
                "no_farthest_raw_cls_mean_gap": no_far_raw.get("mean_real_control_gap"),
                "no_farthest_rks_mean_gap": no_far_rks.get("mean_real_control_gap"),
                "strict_top_feature_source": (actual_top.get("config") or {}).get("feature_source"),
                "no_farthest_feature_axis_winner": "rks"
                if (_safe_float(no_far_rks.get("mean_engineering_score")) or -1e9)
                > (_safe_float(no_far_raw.get("mean_engineering_score")) or -1e9)
                else "raw_cls",
            },
            inference=(
                "Raw observer CLS dominates long-range rupture; RKS becomes the better feature family "
                "once farthest jumps are removed."
            ),
            engineering_action=(
                "Do not replace Track 2 globally. Add a Track 4 feature router: RKS coordinates for local "
                "navigation, normalized raw observer CLS for observer-slice rupture."
            ),
            interpretation_trap=(
                "Raw CLS beating RKS in one transport regime does not falsify Track 2; it says RKS and raw "
                "observer states answer different action questions."
            ),
            next_test="Run a hybrid action graph with per-edge mode selection and compare against pure raw/RKS branches.",
            artifacts=[str(actual_null_path), str(no_farthest_path)],
        ),
        _entry(
            point_id="track4_chart_normalization",
            title="Normalize observer charts before transport",
            status="supported",
            engineering_safe=True,
            evidence={
                "strict_actual_null_global_zscore_mean_gap": actual_global_zscore.get("mean_real_control_gap"),
                "strict_actual_null_zscore_per_slice_mean_gap": actual_zscore.get("mean_real_control_gap"),
                "no_farthest_zscore_per_slice_mean_gap": no_far_zscore.get("mean_real_control_gap"),
                "strict_top_normalization": (actual_top.get("config") or {}).get("normalization"),
                "no_farthest_top_normalization": (no_far_top.get("config") or {}).get("normalization"),
            },
            inference=(
                "Z-scored charts repeatedly outperform raw chart coordinates. This points to scale/covariance "
                "mismatch across observer slices, not merely lack of signal."
            ),
            engineering_action=(
                "Add an explicit observer-chart calibration stage before transport: at minimum global_zscore "
                "and zscore_per_slice branches with provenance in the ledger."
            ),
            interpretation_trap=(
                "Normalization can inflate long-distance action magnitudes; judge it by calibrated real/control "
                "gap and null behavior, not raw action size."
            ),
            next_test="Add per-observer-pair z-scored holonomy to suppress dominant slice pairs.",
            artifacts=[str(actual_null_path), str(no_farthest_path)],
        ),
        _entry(
            point_id="track4_density_wiring",
            title="Fix Track 3 density wiring before trusting density-weighted action",
            status="blocked_by_flat_real_control_density",
            engineering_safe=False,
            evidence={
                "density_audit": density_audit,
                "real_control_density_flat_default_one": density_audit.get("real_control_density_flat_default_one"),
                "synthetic_density_sources": density_audit.get("synthetic_density_sources"),
            },
            inference=(
                "Real/control observer-slice runs often have density_source=default_one, while synthetic can "
                "use Track 3 curvature. Density-weight sweeps therefore do not currently test a real Track 3 field "
                "on real/control."
            ),
            engineering_action=(
                "Repair density export/loader contracts so Track 4 receives canonical Track 3 rho for real, "
                "synthetic, and controls. Then rerun density-only and stress+density action branches."
            ),
            interpretation_trap=(
                "A density-heavy branch scoring well or poorly right now may be an artifact of a flat density "
                "vector, not evidence about terrain."
            ),
            next_test="Assert every real/control Track 4 replay reports a non-default density_source and nonzero density variance.",
            artifacts=[str(path) for path in scale_artifact_paths],
        ),
        _entry(
            point_id="track4_explicit_pair_policies",
            title="Make pair/path proposal policy explicit",
            status="supported",
            engineering_safe=True,
            evidence={
                "strict_actual_null_farthest_axis_gap": actual_farthest.get("mean_real_control_gap"),
                "no_farthest_nearest_axis_gap": no_far_nearest.get("mean_real_control_gap"),
                "no_farthest_mixed_axis_gap": no_far_mixed.get("mean_real_control_gap"),
                "strict_top_pair_mode": (actual_top.get("config") or {}).get("pair_mode"),
                "no_farthest_top_pair_mode": (no_far_top.get("config") or {}).get("pair_mode"),
            },
            inference=(
                "Pair policy is not a sampling detail; it changes the physical meaning and magnitude of Track 4."
            ),
            engineering_action=(
                "Replace the single mixed pair sampler with ledgered pair policies: local_nearest, mixed_local, "
                "distance_stratified, and landmark/farthest rupture."
            ),
            interpretation_trap=(
                "A farthest-pair pass should not be compared directly to nearest-path local traversal without "
                "declaring the policy."
            ),
            next_test="Use fixed shared pair lists across real/control for each pair policy to match geometric difficulty.",
            artifacts=[str(actual_null_path), str(no_farthest_path)],
        ),
    ]

    engineering_safe_entries = [entry for entry in entries if bool(entry.get("engineering_safe"))]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "summary_type": SUMMARY_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "artifacts_used": {
            "baseline": _artifact_meta(baseline_path, baseline),
            "big": _artifact_meta(big_path, big),
            "actual_null": _artifact_meta(actual_null_path, actual_null),
            "no_farthest": _artifact_meta(no_farthest_path, no_farthest),
            "scale_artifacts": [
                _artifact_meta(path, payload)
                for path, payload in zip(scale_artifact_paths, scale_payloads)
            ],
        },
        "summary": {
            "entry_count": len(entries),
            "engineering_safe_count": len(engineering_safe_entries),
            "blocked_count": len([entry for entry in entries if str(entry.get("status", "")).startswith("blocked")]),
            "highest_priority_changes": [
                "promote true null-calibrated excess action",
                "split Track 4 into local_navigation and observer_rupture modes",
                "route raw CLS to rupture and RKS to local navigation",
                "add observer-chart normalization before transport",
                "repair Track 3 density wiring before interpreting density-weighted action",
                "make pair-policy a first-class DAG dimension",
            ],
        },
        "claim_boundary": {
            "safe_claim": "these are engineering recommendations supported by ablation artifacts",
            "unsafe_claim": "these entries do not prove ideological correctness or literal terrain semantics",
        },
        "entries": entries,
    }

    json_path = output_dir / "track4_engineering_ledger.json"
    csv_path = output_dir / "track4_engineering_ledger.csv"
    claim_path = output_dir / "claim_matrix.json"
    csv_fields = [
        "point_id",
        "title",
        "status",
        "engineering_safe",
        "inference",
        "engineering_action",
        "interpretation_trap",
        "next_test",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({field: entry.get(field) for field in csv_fields})

    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": payload["generated_at_utc"],
        "summary_type": "track4_engineering_claim_matrix",
        "claim_scope": "engineering_repair_not_thesis_truth_claims",
        "claims": [
            {
                "claim_id": entry["point_id"],
                "claim_type": "engineering_hypothesis",
                "pass": bool(entry["engineering_safe"]),
                "engineering_safe": bool(entry["engineering_safe"]),
                "thesis_safe": False,
                "status": entry["status"],
                "artifact_family": "track4_engineering_ledger.json",
                "failure_reasons": [] if bool(entry["engineering_safe"]) else [entry["status"]],
            }
            for entry in entries
        ],
    }
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--big", type=Path, default=DEFAULT_BIG)
    parser.add_argument("--actual-null", type=Path, default=DEFAULT_ACTUAL_NULL)
    parser.add_argument("--no-farthest", type=Path, default=DEFAULT_NO_FARTHEST)
    parser.add_argument("--scale-artifact", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scale_paths = args.scale_artifact or DEFAULT_SCALE_ARTIFACTS
    payload = build_ledger(
        baseline_path=args.baseline,
        big_path=args.big,
        actual_null_path=args.actual_null,
        no_farthest_path=args.no_farthest,
        scale_artifact_paths=scale_paths,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "artifacts": payload["artifacts"],
                "summary": payload["summary"],
                "entries": [
                    {
                        "point_id": entry["point_id"],
                        "status": entry["status"],
                        "engineering_safe": entry["engineering_safe"],
                    }
                    for entry in payload["entries"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
