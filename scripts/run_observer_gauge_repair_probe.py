#!/usr/bin/env python3
"""Test whether chart-gauge alignment repairs observer-slice transport.

The gauge invariance probe asks where Track 4 is coordinate-sensitive.  This
probe tests an engineering repair: when observer slices are independently
translated or rotated, can we align them back to a shared chart before computing
cross-observer paths?
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

from scripts.run_observer_gauge_invariance_probe import (  # noqa: E402
    _default_configs,
    _gap,
    _json_safe,
    _run_summary,
    _transform_slices,
)
from scripts.run_observer_slice_transport_scale_suite import (  # noqa: E402
    _infer_run_identity,
    _load_payload,
    _safe_float,
)
from scripts.run_observer_transport_engineering_ablation import (  # noqa: E402
    _config_id,
    _dedupe_paths,
    _payloads_for_mode,
    _prepare_slices,
    _select_pairs_by_mode,
    _stable_seed,
)


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_TYPE = "observer_gauge_repair_probe"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "observer_gauge_repair_probe" / "latest"


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_safe_float(raw) for raw in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the rotation that best maps centered source onto centered target."""

    cross = np.asarray(source, dtype=np.float64).T @ np.asarray(target, dtype=np.float64)
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return u @ vt


def _farthest_indices(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n_items = int(pts.shape[0])
    if count >= n_items:
        return np.arange(n_items, dtype=np.int64)
    center = np.mean(pts, axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(pts - center, axis=1)))
    chosen = [first]
    min_dist = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(1, count):
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(pts - pts[next_idx], axis=1))
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def _select_anchor_indices(
    reference_slices: Mapping[str, np.ndarray],
    *,
    anchor_count: Optional[int],
    strategy: str,
    random_seed: int,
) -> np.ndarray:
    refs = {name: np.asarray(coords, dtype=np.float64) for name, coords in reference_slices.items()}
    names = list(refs)
    n_items = int(next(iter(refs.values())).shape[0])
    if anchor_count is None or int(anchor_count) <= 0 or int(anchor_count) >= n_items:
        return np.arange(n_items, dtype=np.int64)
    count = max(1, min(int(anchor_count), n_items))
    stack = np.stack([refs[name] for name in names], axis=1)
    mean_chart = np.mean(stack, axis=1)
    rng = np.random.default_rng(int(random_seed))
    if strategy == "random":
        return np.sort(rng.choice(n_items, size=count, replace=False)).astype(np.int64)
    if strategy == "stride":
        return np.unique(np.linspace(0, n_items - 1, count, dtype=np.int64))
    if strategy == "farthest_mean":
        return _farthest_indices(mean_chart, count)
    if strategy == "high_observer_disagreement":
        disagreement = np.linalg.norm(np.var(stack, axis=1), axis=1)
        order = np.argsort(disagreement)[::-1]
        return np.sort(order[:count]).astype(np.int64)
    if strategy == "leverage":
        centered = mean_chart - np.mean(mean_chart, axis=0, keepdims=True)
        try:
            u, _, _ = np.linalg.svd(centered, full_matrices=False)
            leverage = np.sum(u * u, axis=1)
        except np.linalg.LinAlgError:
            leverage = np.linalg.norm(centered, axis=1)
        return np.sort(np.argsort(leverage)[::-1][:count]).astype(np.int64)
    if strategy == "hybrid_disagreement_farthest":
        disagreement = np.linalg.norm(np.var(stack, axis=1), axis=1)
        first_count = max(1, count // 2)
        chosen = list(np.argsort(disagreement)[::-1][:first_count])
        min_dist = np.min(
            np.stack([np.linalg.norm(mean_chart - mean_chart[idx], axis=1) for idx in chosen], axis=0),
            axis=0,
        )
        while len(set(chosen)) < count:
            for idx in chosen:
                min_dist[int(idx)] = -1.0
            chosen.append(int(np.argmax(min_dist)))
            min_dist = np.minimum(min_dist, np.linalg.norm(mean_chart - mean_chart[chosen[-1]], axis=1))
        return np.asarray(sorted(set(chosen))[:count], dtype=np.int64)
    raise ValueError(f"unknown anchor strategy: {strategy}")


def _repair_slices(
    transformed: Mapping[str, np.ndarray],
    reference_slices: Mapping[str, np.ndarray],
    *,
    mode: str,
    anchor_count: Optional[int],
    anchor_strategy: str,
    random_seed: int,
    anchor_indices: Optional[np.ndarray] = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {name: np.asarray(coords, dtype=np.float64) for name, coords in transformed.items()}
    refs = {name: np.asarray(coords, dtype=np.float64) for name, coords in reference_slices.items()}
    names = list(arrays)
    if anchor_indices is None:
        anchor_idx = _select_anchor_indices(
            refs,
            anchor_count=anchor_count,
            strategy=str(anchor_strategy),
            random_seed=random_seed,
        )
    else:
        anchor_idx = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
        anchor_idx = np.unique(anchor_idx)
        if anchor_idx.size == 0:
            raise ValueError("explicit anchor_indices must not be empty")

    if mode == "none":
        return {name: coords.copy() for name, coords in arrays.items()}, {
            "repair_mode": mode,
            "anchor_count": int(anchor_idx.size),
            "anchor_strategy": str(anchor_strategy),
            "explicit_anchor_indices": anchor_indices is not None,
        }
    if mode == "center_only":
        repaired: dict[str, np.ndarray] = {}
        for name in names:
            src = arrays[name]
            tgt = refs[name]
            src_mean = np.mean(src[anchor_idx], axis=0, keepdims=True)
            tgt_mean = np.mean(tgt[anchor_idx], axis=0, keepdims=True)
            repaired[name] = (src - src_mean) + tgt_mean
        return repaired, {
            "repair_mode": mode,
            "anchor_count": int(anchor_idx.size),
            "anchor_strategy": str(anchor_strategy),
            "explicit_anchor_indices": anchor_indices is not None,
        }
    if mode == "procrustes_to_original_slice":
        repaired = {}
        residuals = []
        for name in names:
            src = arrays[name]
            tgt = refs[name]
            src_mean = np.mean(src[anchor_idx], axis=0, keepdims=True)
            tgt_mean = np.mean(tgt[anchor_idx], axis=0, keepdims=True)
            rotation = _orthogonal_procrustes(src[anchor_idx] - src_mean, tgt[anchor_idx] - tgt_mean)
            aligned = (src - src_mean) @ rotation + tgt_mean
            repaired[name] = aligned
            residuals.append(float(np.linalg.norm(aligned[anchor_idx] - tgt[anchor_idx]) / max(1, int(anchor_idx.size))))
        return repaired, {
            "repair_mode": mode,
            "anchor_count": int(anchor_idx.size),
            "anchor_strategy": str(anchor_strategy),
            "explicit_anchor_indices": anchor_indices is not None,
            "mean_anchor_residual": float(np.mean(residuals)) if residuals else None,
        }
    if mode == "shared_mean_procrustes":
        reference = np.mean(np.stack([refs[name] for name in names], axis=1), axis=1)
        repaired = {}
        residuals = []
        for name in names:
            src = arrays[name]
            src_mean = np.mean(src[anchor_idx], axis=0, keepdims=True)
            ref_mean = np.mean(reference[anchor_idx], axis=0, keepdims=True)
            rotation = _orthogonal_procrustes(src[anchor_idx] - src_mean, reference[anchor_idx] - ref_mean)
            aligned = (src - src_mean) @ rotation + ref_mean
            repaired[name] = aligned
            residuals.append(float(np.linalg.norm(aligned[anchor_idx] - reference[anchor_idx]) / max(1, int(anchor_idx.size))))
        return repaired, {
            "repair_mode": mode,
            "anchor_count": int(anchor_idx.size),
            "anchor_strategy": str(anchor_strategy),
            "explicit_anchor_indices": anchor_indices is not None,
            "mean_anchor_residual": float(np.mean(residuals)) if residuals else None,
        }
    raise ValueError(f"unknown repair mode: {mode}")


def _run_one(
    payload_path: Path,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    transform_modes: Sequence[str],
    repair_modes: Sequence[str],
    device: torch.device,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    anchor_count: Optional[int],
    anchor_strategy: str,
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
    baseline = _run_summary(
        slices,
        article_pairs=article_pairs,
        density=density,
        stress=stress,
        config=config,
        payload_path=payload_path,
        transform_mode="identity",
        random_seed=random_seed,
    )
    rows: list[dict[str, Any]] = []
    for transform_mode in transform_modes:
        transformed, transform_meta = _transform_slices(
            slices,
            mode=transform_mode,
            seed=_stable_seed(payload_path, _config_id(config), transform_mode, random_seed),
        )
        for repair_mode in repair_modes:
            repaired, repair_meta = _repair_slices(
                transformed,
                slices,
                mode=repair_mode,
                anchor_count=anchor_count,
                anchor_strategy=anchor_strategy,
                random_seed=_stable_seed(payload_path, _config_id(config), transform_mode, repair_mode, random_seed),
            )
            transport = _run_summary(
                repaired,
                article_pairs=article_pairs,
                density=density,
                stress=stress,
                config=config,
                payload_path=payload_path,
                transform_mode=transform_mode,
                random_seed=random_seed,
            )
            rows.append(
                {
                    **identity,
                    "config_label": config.get("label"),
                    "config_id": _config_id(config),
                    "config": {key: value for key, value in dict(config).items() if key != "label"},
                    "transform_mode": transform_mode,
                    "repair_mode": repair_mode,
                    "transform": transform_meta,
                    "repair": repair_meta,
                    "n_articles": int(next(iter(slices.values())).shape[0]),
                    "n_slices": len(slices),
                    "projection": metadata.get("projection"),
                    "normalization": metadata.get("normalization"),
                    "field_sources": metadata.get("field_sources"),
                    "pair_selection": pair_meta,
                    "identity_transport": baseline,
                    "transport": transport,
                }
            )
    return rows


def _corpus_metric(rows: Sequence[Mapping[str, Any]], corpus: str, metric: str) -> Optional[float]:
    return _mean((row.get("transport") or {}).get(metric) for row in rows if row.get("corpus") == corpus)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row.get("config_label")), str(row.get("transform_mode")), str(row.get("repair_mode")))
        ].append(row)
    summaries: list[dict[str, Any]] = []
    identity_by_config: dict[str, float] = {}
    for row in rows:
        if row.get("transform_mode") == "identity" and row.get("repair_mode") == "none":
            label = str(row.get("config_label"))
            identity_by_config.setdefault(label, _gap([r for r in rows if r.get("config_label") == label and r.get("transform_mode") == "identity" and r.get("repair_mode") == "none"], "mean_calibrated_excess_holonomy_action") or 0.0)
    damaged_by_config_transform: dict[tuple[str, str], float] = {}
    for (label, transform_mode, repair_mode), group_rows in grouped.items():
        first = group_rows[0]
        gap = _gap(group_rows, "mean_calibrated_excess_holonomy_action")
        rel_gap = _gap(group_rows, "mean_calibrated_relative_holonomy")
        summary = {
            "config_label": label,
            "config_id": first.get("config_id"),
            "config": first.get("config"),
            "transform_mode": transform_mode,
            "repair_mode": repair_mode,
            "payload_count": len(group_rows),
            "real_minus_control_gap": gap,
            "real_minus_control_relative_gap": rel_gap,
            "real_mean_calibrated_excess": _corpus_metric(group_rows, "real", "mean_calibrated_excess_holonomy_action"),
            "control_mean_calibrated_excess": _mean(
                _corpus_metric(group_rows, corpus, "mean_calibrated_excess_holonomy_action")
                for corpus in ("control_random", "control_shuffled", "control_constant")
            ),
            "synthetic_mean_calibrated_excess": _corpus_metric(
                group_rows, "synthetic", "mean_calibrated_excess_holonomy_action"
            ),
            "mean_anchor_residual": _mean((row.get("repair") or {}).get("mean_anchor_residual") for row in group_rows),
        }
        summaries.append(summary)
        if repair_mode == "none":
            damaged_by_config_transform[(label, transform_mode)] = gap if gap is not None else 0.0
    for summary in summaries:
        label = str(summary["config_label"])
        transform_mode = str(summary["transform_mode"])
        identity_gap = identity_by_config.get(label)
        damaged_gap = damaged_by_config_transform.get((label, transform_mode))
        gap = _safe_float(summary.get("real_minus_control_gap"))
        summary["identity_gap"] = identity_gap
        summary["damaged_gap"] = damaged_gap
        summary["gap_error_vs_identity"] = (
            None if gap is None or identity_gap is None else float(abs(gap - identity_gap))
        )
        damaged_error = None if damaged_gap is None or identity_gap is None else float(abs(damaged_gap - identity_gap))
        summary["damaged_gap_error_vs_identity"] = damaged_error
        summary["recovery_fraction"] = (
            None
            if damaged_error is None or gap is None or identity_gap is None or damaged_error <= 1e-12
            else float(1.0 - (abs(gap - identity_gap) / damaged_error))
        )
        gap_error = _safe_float(summary.get("gap_error_vs_identity"))
        summary["repair_evaluated"] = bool(summary["repair_mode"] != "none")
        summary["repair_pass"] = bool(
            summary["repair_mode"] == "none"
            or (
                summary["transform_mode"] == "identity"
                and gap_error is not None
                and abs(gap_error) <= 1e-9
            )
            or (
                summary["recovery_fraction"] is not None
                and float(summary["recovery_fraction"]) >= 0.75
            )
        )
    return sorted(
        summaries,
        key=lambda row: (str(row.get("config_label")), str(row.get("transform_mode")), str(row.get("repair_mode"))),
    )


def build_gauge_repair_suite(
    *,
    payload_paths: Sequence[Path],
    output_dir: Path,
    transform_modes: Sequence[str],
    repair_modes: Sequence[str],
    device_name: str,
    max_pairs: int,
    neighbor_count: int,
    random_seed: int,
    article_cap: Optional[int],
    pair_dim_cap: Optional[int],
    anchor_count: Optional[int],
    anchor_strategy: str = "random",
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
                        repair_modes=repair_modes,
                        device=device,
                        max_pairs=max_pairs,
                        neighbor_count=neighbor_count,
                        random_seed=random_seed,
                        article_cap=article_cap,
                        pair_dim_cap=pair_dim_cap,
                        anchor_count=anchor_count,
                        anchor_strategy=anchor_strategy,
                    )
                )
            except Exception as exc:
                failures.append({"payload_path": str(payload_path), "config_label": config.get("label"), "error": str(exc)})
    summaries = _summarize(rows)
    claim_matrix = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_type": "observer_gauge_repair_claim_matrix",
        "claim_scope": "engineering_chart_alignment_repair_not_semantic_truth_claim",
        "claims": [
            {
                "claim_id": f"repair_{summary.get('config_label')}_{summary.get('transform_mode')}_{summary.get('repair_mode')}",
                "claim_type": "engineering_hypothesis",
                "pass": bool(summary.get("repair_pass")),
                "engineering_safe": bool(summary.get("repair_pass")),
                "thesis_safe": False,
                "point_estimate": summary.get("recovery_fraction"),
                "artifact_family": "observer_gauge_repair_probe.json",
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
            "repair_modes": list(repair_modes),
            "max_pairs": int(max_pairs),
            "neighbor_count": int(neighbor_count),
            "random_seed": int(random_seed),
            "article_cap": article_cap,
            "pair_dim_cap": pair_dim_cap,
            "anchor_count": anchor_count,
            "anchor_strategy": anchor_strategy,
            "config_labels": [config.get("label") for config in selected_configs],
        },
        "claim_boundary": {
            "safe_claim": "known article correspondences can be used to align observer charts before path computation",
            "unsafe_claim": "alignment proves the semantic interpretation of every path without external validation",
        },
        "summary": {
            "run_count": len(rows),
            "failure_count": len(failures),
            "config_transform_repair_summaries": summaries,
            "interpretation": (
                "A high recovery_fraction means chart alignment restores the original Track 4 signal after gauge damage."
            ),
        },
        "runs": rows,
        "failures": failures,
    }
    json_path = output_dir / "observer_gauge_repair_probe.json"
    csv_path = output_dir / "observer_gauge_repair_probe.csv"
    claim_path = output_dir / "claim_matrix.json"
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path), "claim_matrix": str(claim_path)}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claim_path.write_text(json.dumps(_json_safe(claim_matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "config_label",
            "transform_mode",
            "repair_mode",
            "real_minus_control_gap",
            "identity_gap",
            "damaged_gap",
            "gap_error_vs_identity",
            "recovery_fraction",
            "repair_evaluated",
            "repair_pass",
            "mean_anchor_residual",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})
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
        default=["identity", "per_slice_translation", "per_slice_orthogonal"],
    )
    parser.add_argument(
        "--repair-modes",
        nargs="+",
        default=["none", "center_only", "procrustes_to_original_slice", "shared_mean_procrustes"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--article-cap", type=int, default=500)
    parser.add_argument("--pair-dim-cap", type=int, default=64)
    parser.add_argument("--anchor-count", type=int, default=0)
    parser.add_argument(
        "--anchor-strategy",
        default="random",
        choices=(
            "random",
            "stride",
            "farthest_mean",
            "high_observer_disagreement",
            "leverage",
            "hybrid_disagreement_farthest",
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = [] if args.no_mode_payloads else _payloads_for_mode(str(args.payload_mode))
    payloads.extend(args.payload)
    payloads = _dedupe_paths(payloads)
    if not payloads:
        raise SystemExit("No observer_global.pt payloads selected")
    artifact = build_gauge_repair_suite(
        payload_paths=payloads,
        output_dir=args.output_dir,
        transform_modes=[str(mode) for mode in args.transform_modes],
        repair_modes=[str(mode) for mode in args.repair_modes],
        device_name=str(args.device),
        max_pairs=int(args.max_pairs),
        neighbor_count=int(args.neighbor_count),
        random_seed=int(args.random_seed),
        article_cap=int(args.article_cap) if int(args.article_cap) > 0 else None,
        pair_dim_cap=int(args.pair_dim_cap) if int(args.pair_dim_cap) > 0 else None,
        anchor_count=int(args.anchor_count) if int(args.anchor_count) > 0 else None,
        anchor_strategy=str(args.anchor_strategy),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "artifacts": artifact["artifacts"],
                    "run_count": artifact["summary"]["run_count"],
                    "failure_count": artifact["summary"]["failure_count"],
                    "summaries": artifact["summary"]["config_transform_repair_summaries"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not artifact["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
