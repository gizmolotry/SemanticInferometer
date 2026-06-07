#!/usr/bin/env python3
"""Probe gauge/coordinate sensitivity of observer-slice transport.

This diagnostic asks whether Track 4 transport is stable under harmless global
coordinate changes and where it is vulnerable to arbitrary observer-chart
alignment.  Global translations and global orthogonal rotations should preserve
Euclidean action.  Per-slice transforms are intentionally harsher: if they
change the result dramatically, Track 4 needs chart calibration/gauge fixing
before observer-switch distances are treated as physical.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observer_slice_transport import ObserverSliceTransportConfig  # noqa: E402
from scripts.run_observer_slice_transport_scale_suite import (  # noqa: E402
    _fast_transport_summary,
    _infer_run_identity,
    _load_payload,
    _safe_float,
)
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _config_id,
    _dedupe_paths,
    _null_slices,
    _payloads_for_mode,
    _prepare_slices,
    _select_pairs_by_mode,
    _stable_seed,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_gauge_invariance_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_gauge_invariance_probe" / "latest"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _default_configs() -> list[dict[str, Any]]:
    return [
        {
            "label": "current_baseline",
            "feature_source": "rks",
            "projection_dim_cap": 512,
            "normalization": "none",
            "pair_mode": "mixed",
            "null_mode": "zero",
            "weight_profile": "default_action",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.0,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "rupture_raw_cls_farthest",
            "feature_source": "raw_cls",
            "projection_dim_cap": 512,
            "normalization": "zscore_per_slice",
            "pair_mode": "farthest",
            "null_mode": "independent_article_shuffle",
            "weight_profile": "switch_heavy",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.5,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
        {
            "label": "local_rks_nearest",
            "feature_source": "rks",
            "projection_dim_cap": 128,
            "normalization": "zscore_per_slice",
            "pair_mode": "nearest",
            "null_mode": "dimension_signflip_by_slice",
            "weight_profile": "switch_heavy",
            "semantic_weight": 1.0,
            "observer_switch_weight": 1.5,
            "stress_weight": 0.25,
            "density_weight": 0.15,
        },
    ]


def _action_config(config: Mapping[str, Any]) -> ObserverSliceTransportConfig:
    return ObserverSliceTransportConfig(
        semantic_weight=float(config["semantic_weight"]),
        observer_switch_weight=float(config["observer_switch_weight"]),
        stress_weight=float(config["stress_weight"]),
        density_weight=float(config["density_weight"]),
    )


def _orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mat = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs.reshape(1, -1)).astype(np.float64)


def _transform_slices(
    slices: Mapping[str, np.ndarray],
    *,
    mode: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {name: np.asarray(coords, dtype=np.float64) for name, coords in slices.items()}
    names = list(arrays)
    dim = int(next(iter(arrays.values())).shape[1])
    rng = np.random.default_rng(int(seed))
    if mode == "identity":
        return {name: coords.copy() for name, coords in arrays.items()}, {"transform_mode": mode}
    if mode == "global_translation":
        shift = rng.normal(scale=3.0, size=(1, dim))
        return {name: coords + shift for name, coords in arrays.items()}, {
            "transform_mode": mode,
            "shift_norm": float(np.linalg.norm(shift)),
        }
    if mode == "global_orthogonal":
        q = _orthogonal(dim, seed)
        return {name: coords @ q for name, coords in arrays.items()}, {"transform_mode": mode}
    if mode == "global_scalar_scale":
        scale = 2.75
        return {name: coords * scale for name, coords in arrays.items()}, {
            "transform_mode": mode,
            "scale": scale,
        }
    if mode == "global_anisotropic_scale":
        scales = np.exp(rng.normal(loc=0.0, scale=0.45, size=(1, dim)))
        return {name: coords * scales for name, coords in arrays.items()}, {
            "transform_mode": mode,
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
        }
    if mode == "per_slice_translation":
        return {
            name: coords + rng.normal(scale=3.0, size=(1, dim))
            for name, coords in arrays.items()
        }, {"transform_mode": mode}
    if mode == "per_slice_orthogonal":
        return {
            name: coords @ _orthogonal(dim, seed + 1009 * idx)
            for idx, (name, coords) in enumerate(arrays.items())
        }, {"transform_mode": mode}
    if mode == "small_jitter":
        stacked = np.concatenate(list(arrays.values()), axis=0)
        scale = max(float(np.std(stacked)), 1e-9) * 0.02
        return {
            name: coords + rng.normal(scale=scale, size=coords.shape)
            for name, coords in arrays.items()
        }, {"transform_mode": mode, "noise_scale": scale}
    raise ValueError(f"unknown transform mode: {mode}")


def _calibrate_summary(summary: Mapping[str, Any], null_summary: Optional[Mapping[str, Any]], *, null_mode: str) -> dict[str, Any]:
    raw = _safe_float(summary.get("mean_holonomy_action"))
    raw_relative = _safe_float(summary.get("mean_relative_holonomy"))
    null = _safe_float((null_summary or {}).get("mean_holonomy_action")) if null_summary else 0.0
    null_relative = _safe_float((null_summary or {}).get("mean_relative_holonomy")) if null_summary else 0.0
    return {
        **dict(summary),
        "null_mode": null_mode,
        "mean_null_holonomy_action": null,
        "mean_null_relative_holonomy": null_relative,
        "mean_calibrated_excess_holonomy_action": None if raw is None or null is None else float(raw - null),
        "mean_calibrated_relative_holonomy": None
        if raw_relative is None or null_relative is None
        else float(raw_relative - null_relative),
        "null_record_count": (null_summary or {}).get("record_count", 0) if null_summary else 0,
    }


def _run_summary(
    slices: Mapping[str, np.ndarray],
    *,
    article_pairs: Sequence[tuple[int, int]],
    density: np.ndarray,
    stress: np.ndarray,
    config: Mapping[str, Any],
    payload_path: Path,
    transform_mode: str,
    random_seed: int,
) -> dict[str, Any]:
    action_config = _action_config(config)
    summary = _fast_transport_summary(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=action_config,
        top_records=0,
    )
    # Keep the null draw fixed across coordinate transforms so deltas measure
    # gauge sensitivity, not a different random null permutation.
    null_seed = _stable_seed(payload_path, _config_id(config), "gauge", random_seed)
    null_chart = _null_slices(slices, mode=str(config["null_mode"]), random_seed=null_seed)
    null_summary = None
    if null_chart is not None:
        null_summary = _fast_transport_summary(
            null_chart,
            article_pairs=article_pairs,
            density=density,
            stress=stress,
            config=action_config,
            top_records=0,
        )
    return _calibrate_summary(summary, null_summary, null_mode=str(config["null_mode"]))


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    transform_modes: Sequence[str],
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
) -> list[dict[str, Any]]:
    identity = _infer_run_identity(payload_path)
    slices, density, stress, metadata = _prepare_slices(
        payload_path,
        payload,
        config,
        device=device,
        normalization=str(config["normalization"]),
        article_cap=article_cap,
        random_seed=random_seed,
    )
    reference = np.mean(np.stack(list(slices.values()), axis=1), axis=1)
    article_pairs, pair_meta = _select_pairs_by_mode(
        reference,
        pair_mode=str(config["pair_mode"]),
        max_pairs=max_pairs,
        neighbor_count=neighbor_count,
        random_seed=random_seed,
        pair_dim_cap=pair_dim_cap,
    )
    rows: list[dict[str, Any]] = []
    for mode in transform_modes:
        transformed, transform_meta = _transform_slices(
            slices,
            mode=mode,
            seed=_stable_seed(payload_path, _config_id(config), mode, random_seed),
        )
        transport = _run_summary(
            transformed,
            article_pairs=article_pairs,
            density=density,
            stress=stress,
            config=config,
            payload_path=payload_path,
            transform_mode=mode,
            random_seed=random_seed,
        )
        rows.append(
            {
                **identity,
                "config_label": config.get("label"),
                "config_id": _config_id(config),
                "config": {key: value for key, value in dict(config).items() if key != "label"},
                "transform_mode": mode,
                "transform": transform_meta,
                "n_articles": int(next(iter(slices.values())).shape[0]),
                "n_slices": len(slices),
                "projection": metadata.get("projection"),
                "normalization": metadata.get("normalization"),
                "field_sources": metadata.get("field_sources"),
                "pair_selection": pair_meta,
                "transport": transport,
            }
        )
    return rows


def _corpus_metric(rows: Sequence[Mapping[str, Any]], corpus: str, metric: str) -> Optional[float]:
    return _mean((row.get("transport") or {}).get(metric) for row in rows if row.get("corpus") == corpus)


def _gap(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    real = _corpus_metric(rows, "real", metric)
    controls = _mean(
        _corpus_metric(rows, corpus, metric)
        for corpus in ("control_random", "control_shuffled", "control_constant")
    )
    return float(real - controls) if real is not None and controls is not None else None


def _summarize_config_transform(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("config_label")), str(row.get("transform_mode")))].append(row)
    summaries: list[dict[str, Any]] = []
    identity_by_config: dict[str, dict[str, Any]] = {}
    for (config_label, mode), group_rows in grouped.items():
        first = group_rows[0]
        summary = {
            "config_label": config_label,
            "config_id": first.get("config_id"),
            "config": first.get("config"),
            "transform_mode": mode,
            "payload_count": len(group_rows),
            "real_minus_control_gap": _gap(group_rows, "mean_calibrated_excess_holonomy_action"),
            "real_minus_control_relative_gap": _gap(group_rows, "mean_calibrated_relative_holonomy"),
            "real_mean_calibrated_excess": _corpus_metric(
                group_rows, "real", "mean_calibrated_excess_holonomy_action"
            ),
            "control_mean_calibrated_excess": _mean(
                _corpus_metric(group_rows, corpus, "mean_calibrated_excess_holonomy_action")
                for corpus in ("control_random", "control_shuffled", "control_constant")
            ),
            "synthetic_mean_calibrated_excess": _corpus_metric(
                group_rows, "synthetic", "mean_calibrated_excess_holonomy_action"
            ),
        }
        summaries.append(summary)
        if mode == "identity":
            identity_by_config[config_label] = summary
    for summary in summaries:
        identity = identity_by_config.get(str(summary["config_label"]), {})
        base_gap = _safe_float(identity.get("real_minus_control_gap"))
        base_rel = _safe_float(identity.get("real_minus_control_relative_gap"))
        gap = _safe_float(summary.get("real_minus_control_gap"))
        rel = _safe_float(summary.get("real_minus_control_relative_gap"))
        summary["delta_vs_identity_gap"] = None if gap is None or base_gap is None else float(gap - base_gap)
        summary["delta_vs_identity_relative_gap"] = None if rel is None or base_rel is None else float(rel - base_rel)
        summary["ratio_vs_identity_gap"] = (
            None if gap is None or base_gap is None or abs(base_gap) <= 1e-12 else float(gap / base_gap)
        )
        summary["rigid_invariance_expected"] = summary["transform_mode"] in {
            "identity",
            "global_translation",
            "global_orthogonal",
        }
        summary["rigid_invariance_pass"] = bool(
            not summary["rigid_invariance_expected"]
            or summary["transform_mode"] == "identity"
            or (
                summary["delta_vs_identity_relative_gap"] is not None
                and abs(float(summary["delta_vs_identity_relative_gap"])) <= 0.02
            )
        )
        summary["gauge_vulnerability_flag"] = bool(
            summary["transform_mode"] in {"per_slice_translation", "per_slice_orthogonal", "global_anisotropic_scale"}
            and summary["ratio_vs_identity_gap"] is not None
            and abs(float(summary["ratio_vs_identity_gap"])) > 2.0
        )
    return sorted(
        summaries,
        key=lambda row: (str(row.get("config_label")), str(row.get("transform_mode"))),
    )


def build_gauge_invariance_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    transform_modes: Sequence[str],
    device_name: str,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    configs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    selected_configs = [dict(config) for config in (configs or _default_configs())]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for payload_path in payload_paths:
        try:
            payload = _load_payload(payload_path)
        except Exception as exc:
            failures.append({"payload_path": str(payload_path), "error": str(exc)})
            continue
        for config in selected_configs:
            try:
                rows.extend(
                    _run_one(
                        payload_path,
                        payload,
                        config,
                        transform_modes=transform_modes,
                        device=device,
                        max_pairs=max_pairs,
                        neighbor_count=neighbor_count,
                        random_seed=random_seed,
                        article_cap=article_cap,
                        pair_dim_cap=pair_dim_cap,
                    )
                )
            except Exception as exc:
                failures.append({"payload_path": str(payload_path), "config_label": config.get("label"), "error": str(exc)})
    summaries = _summarize_config_transform(rows)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_gauge_invariance_claim_matrix",
        "claim_scope": "engineering_coordinate_sensitivity_not_thesis_truth_claim",
        "claims": [
            {
                "claim_id": f"gauge_{summary.get('config_label')}_{summary.get('transform_mode')}",
                "claim_type": "engineering_hypothesis",
                "pass": bool(summary.get("rigid_invariance_pass")) if summary.get("rigid_invariance_expected") else True,
                "engineering_safe": bool(summary.get("rigid_invariance_pass"))
                if summary.get("rigid_invariance_expected")
                else True,
                "thesis_safe": False,
                "gauge_vulnerability_flag": bool(summary.get("gauge_vulnerability_flag")),
                "point_estimate": summary.get("real_minus_control_gap"),
                "artifact_family": "observer_gauge_invariance_probe.json",
            }
            for summary in summaries
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at_utc": claim_matrix["generated_at_utc"],
        "output_dir": str(output_dir),
        "device": str(device),
        "config": {
            "payload_count": len(payload_paths),
            "transform_modes": list(transform_modes),
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "Track 4 coordinate sensitivity under explicit chart transforms",
            "unsafe_claim": "arbitrary per-slice gauge transforms are semantically valid without chart alignment",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "config_transform_summaries": summaries,
            "interpretation": (
                "Rigid global transforms should be stable. Large per-slice transform changes imply chart-gauge vulnerability."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_gauge_invariance_probe.json"
    csv_path = output_dir / "observer_gauge_invariance_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_label",
                "transform_mode",
                "real_minus_control_gap",
                "real_minus_control_relative_gap",
                "delta_vs_identity_gap",
                "delta_vs_identity_relative_gap",
                "ratio_vs_identity_gap",
                "rigid_invariance_pass",
                "gauge_vulnerability_flag",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in writer.fieldnames})
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="append", type=Path, default=[])
    parser.add_argument("--payload-mode", choices=("representative", "defaults", "real_controls"), default="defaults")
    parser.add_argument("--no-mode-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transform-modes",
        nargs="+",
        default=[
            "identity",
            "global_translation",
            "global_orthogonal",
            "global_scalar_scale",
            "global_anisotropic_scale",
            "per_slice_translation",
            "per_slice_orthogonal",
            "small_jitter",
        ],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_gauge_invariance_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        transform_modes=[str(mode) for mode in args.transform_modes],
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "config_transform_summaries": artifact["summary"]["config_transform_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
