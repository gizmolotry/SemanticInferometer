"""Focused Track 4 basis validation across corpora, kernels, and seeds.

This script deliberately reuses the existing full-pipeline Track 4 basis probe
runner. It adds only matrix orchestration and aggregate claim-boundary reporting.
The goal is to test whether Track 4 terrain/basis claims survive outside the
tiny property/theft probe without inventing a second experiment stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_track4_pipeline_basis_probe import (  # noqa: E402
    ProbeConfig,
    probe_claim_boundary,
    run_probe,
)


DEFAULT_BASES = ("track2", "logits_flat")
DEFAULT_KERNELS = ("rbf", "matern", "imq")
DEFAULT_SEEDS = (42, 420, 4200)
DEFAULT_CORPORA = ("real", "control_shuffled", "control_random", "synthetic_microprobe")


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    corpus_arg: str
    kind: str


@dataclass(frozen=True)
class ValidationCell:
    corpus: CorpusSpec
    kernel: str
    seed: int
    output_dir: Path
    nli_cache_path: Path


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def resolve_corpus_specs(
    requested: Sequence[str],
    *,
    synthetic_corpus: Path | None = None,
) -> List[CorpusSpec]:
    """Resolve corpus labels into run_experiments.py corpus arguments."""
    data_dir = ROOT / "data"
    property_theft = ROOT / "outputs" / "microprobes" / "property_theft" / "deberta_20260515" / "property_theft_corpus.jsonl"
    synthetic_path = synthetic_corpus if synthetic_corpus is not None else property_theft

    specs: List[CorpusSpec] = []
    for raw_name in requested:
        name = str(raw_name).strip()
        if not name:
            continue
        if name in {"real", "control_shuffled", "control_random", "control_constant"}:
            expected = data_dir / f"{name}.jsonl" if name != "real" else data_dir / "real_corpus.jsonl"
            if not expected.exists():
                raise FileNotFoundError(f"Requested corpus '{name}' is missing expected file: {expected}")
            specs.append(CorpusSpec(name=name, corpus_arg=name, kind=name))
        elif name in {"synthetic", "synthetic_microprobe", "property_theft"}:
            if synthetic_path is None or not Path(synthetic_path).exists():
                raise FileNotFoundError(
                    "Synthetic corpus requested but no synthetic corpus path exists. "
                    "Pass --synthetic-corpus PATH."
                )
            specs.append(
                CorpusSpec(
                    name="synthetic_microprobe" if name != "property_theft" else "property_theft",
                    corpus_arg=str(Path(synthetic_path)),
                    kind="synthetic",
                )
            )
        else:
            path = Path(name)
            if not path.exists():
                raise ValueError(f"Unknown corpus '{name}' and path does not exist.")
            specs.append(CorpusSpec(name=path.stem, corpus_arg=str(path), kind="custom"))

    # Keep first occurrence of each resolved display name.
    deduped: Dict[str, CorpusSpec] = {}
    for spec in specs:
        deduped.setdefault(spec.name, spec)
    return list(deduped.values())


def build_validation_cells(
    *,
    output_root: Path,
    corpora: Sequence[CorpusSpec],
    kernels: Sequence[str],
    seeds: Sequence[int],
) -> List[ValidationCell]:
    cells: List[ValidationCell] = []
    cache_dir = Path(output_root) / "_nli_cache"
    for corpus in corpora:
        for kernel in kernels:
            for seed in seeds:
                cell_dir = Path(output_root) / corpus.name / str(kernel) / f"seed_{int(seed)}"
                cache_path = cache_dir / f"{corpus.name}_seed{int(seed)}.pt"
                cells.append(
                    ValidationCell(
                        corpus=corpus,
                        kernel=str(kernel),
                        seed=int(seed),
                        output_dir=cell_dir,
                        nli_cache_path=cache_path,
                    )
                )
    return cells


def _row_from_probe_row(cell: ValidationCell, probe_row: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "corpus": cell.corpus.name,
        "corpus_kind": cell.corpus.kind,
        "kernel": cell.kernel,
        "seed": int(cell.seed),
        "basis": probe_row.get("basis"),
        "observer_exists": bool(probe_row.get("observer_exists")),
        "effective_basis": probe_row.get("effective_basis"),
        "basis_embedding_dim": probe_row.get("basis_embedding_dim"),
        "basis_probe_score": _safe_float(probe_row.get("basis_probe_score")),
        "safe_for_thesis_claim": bool(probe_row.get("safe_for_thesis_claim", False)),
        "traversal_status": probe_row.get("traversal_status"),
        "path_count": probe_row.get("path_count"),
        "closed_loop_rate": _safe_float(probe_row.get("closed_loop_rate")),
        "mean_work_integral": _safe_float(probe_row.get("mean_work_integral")),
        "path_shape_entropy_norm": _safe_float(probe_row.get("path_shape_entropy_norm")),
        "path_edge_entropy_norm": _safe_float(probe_row.get("path_edge_entropy_norm")),
        "primary_zone_count": probe_row.get("primary_zone_count"),
        "reactive_flux_total": _safe_float(probe_row.get("reactive_flux_total")),
        "failure_reasons": probe_row.get("failure_reasons", []),
        "run_dir": probe_row.get("run_dir"),
    }
    return row


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "corpus",
        "corpus_kind",
        "kernel",
        "seed",
        "basis",
        "observer_exists",
        "effective_basis",
        "basis_embedding_dim",
        "basis_probe_score",
        "safe_for_thesis_claim",
        "traversal_status",
        "path_count",
        "closed_loop_rate",
        "mean_work_integral",
        "path_shape_entropy_norm",
        "path_edge_entropy_norm",
        "primary_zone_count",
        "reactive_flux_total",
        "run_dir",
        "failure_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["failure_reasons"] = "; ".join(str(item) for item in row.get("failure_reasons", []))
            writer.writerow({key: out.get(key) for key in fieldnames})


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(v) for v in (_safe_float(value) for value in values) if v is not None]
    return float(np.mean(finite)) if finite else None


def aggregate_validation(
    *,
    cell_summaries: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    min_basis_score_margin: float = 0.05,
) -> Dict[str, Any]:
    cell_boundaries = [summary.get("claim_boundary", {}) for summary in cell_summaries]
    completed_cells = [summary for summary in cell_summaries if summary.get("status") in {"OK", "SKIPPED_EXISTING"}]
    failed_cells = [summary for summary in cell_summaries if summary.get("status") not in {"OK", "SKIPPED_EXISTING"}]
    real_rows = [row for row in rows if row.get("corpus") == "real"]
    control_rows = [row for row in rows if str(row.get("corpus", "")).startswith("control_")]
    synthetic_rows = [row for row in rows if row.get("corpus_kind") == "synthetic"]

    basis_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        basis_groups.setdefault(str(row.get("basis")), []).append(row)
    basis_summary = {
        basis: {
            "row_count": len(group),
            "mean_score": _mean(row.get("basis_probe_score") for row in group),
            "terrain_safe_rate": (
                sum(1 for row in group if bool(row.get("safe_for_thesis_claim"))) / float(len(group))
                if group else 0.0
            ),
            "mean_closed_loop_rate": _mean(row.get("closed_loop_rate") for row in group),
            "mean_primary_zone_count": _mean(row.get("primary_zone_count") for row in group),
            "mean_work_integral": _mean(row.get("mean_work_integral") for row in group),
        }
        for basis, group in sorted(basis_groups.items())
    }

    cell_recommendations: Dict[str, int] = {}
    margins: List[float] = []
    for boundary in cell_boundaries:
        recommended = boundary.get("recommended_basis")
        if recommended:
            cell_recommendations[str(recommended)] = cell_recommendations.get(str(recommended), 0) + 1
        margin = _safe_float(boundary.get("basis_score_margin"))
        if margin is not None:
            margins.append(float(margin))

    real_terrain_safe_rate = (
        sum(1 for row in real_rows if bool(row.get("safe_for_thesis_claim"))) / float(len(real_rows))
        if real_rows else 0.0
    )
    control_terrain_safe_rate = (
        sum(1 for row in control_rows if bool(row.get("safe_for_thesis_claim"))) / float(len(control_rows))
        if control_rows else 0.0
    )
    instrumentation_supported = bool(cell_boundaries) and all(
        bool(boundary.get("instrumentation_supported")) for boundary in cell_boundaries
    )
    terrain_validity_supported = bool(real_rows) and real_terrain_safe_rate >= 0.67
    basis_superiority_supported = bool(cell_boundaries) and all(
        bool(boundary.get("basis_superiority_supported")) for boundary in cell_boundaries
    )

    failure_reasons = sorted(
        {
            str(reason)
            for row in rows
            for reason in (row.get("failure_reasons") or [])
            if str(reason).strip()
        }
    )

    claim_boundary = {
        "instrumentation_supported": instrumentation_supported,
        "terrain_validity_supported": terrain_validity_supported,
        "basis_superiority_supported": basis_superiority_supported,
        "basis_recommendation_counts": cell_recommendations,
        "mean_basis_score_margin": float(np.mean(margins)) if margins else None,
        "minimum_basis_score_margin": float(min_basis_score_margin),
        "real_terrain_safe_rate": real_terrain_safe_rate,
        "control_terrain_safe_rate": control_terrain_safe_rate,
        "safe_language": (
            "This matrix supports Track 4 instrumentation if true. Terrain validity "
            "requires real runs to clear path-touched zone and Bridge/Void effect checks."
        ),
    }

    return {
        "schema_version": "1.0",
        "cell_count": len(cell_summaries),
        "completed_cell_count": len(completed_cells),
        "failed_cell_count": len(failed_cells),
        "row_count": len(rows),
        "real_row_count": len(real_rows),
        "control_row_count": len(control_rows),
        "synthetic_row_count": len(synthetic_rows),
        "basis_summary": basis_summary,
        "cell_recommendations": cell_recommendations,
        "claim_boundary": claim_boundary,
        "common_failure_reasons": failure_reasons,
        "failed_cells": failed_cells,
    }


def write_validation_summary(
    output_root: Path,
    *,
    cells: Sequence[ValidationCell],
    cell_summaries: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    bases: Sequence[str],
    kernels: Sequence[str],
    seeds: Sequence[int],
    limit: int,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_validation(cell_summaries=cell_summaries, rows=rows)
    payload = {
        "schema_version": "1.0",
        "experiment_type": "track4_focused_basis_validation",
        "limit": int(limit),
        "bases": list(bases),
        "kernels": list(kernels),
        "seeds": [int(seed) for seed in seeds],
        "cells": [
            {
                "corpus": cell.corpus.name,
                "corpus_kind": cell.corpus.kind,
                "kernel": cell.kernel,
                "seed": int(cell.seed),
                "output_dir": str(cell.output_dir),
                "nli_cache_path": str(cell.nli_cache_path),
            }
            for cell in cells
        ],
        "cell_summaries": list(cell_summaries),
        "rows": list(rows),
        "aggregate": aggregate,
    }
    summary_path = output_root / "track4_focused_basis_validation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_rows_csv(output_root / "track4_focused_basis_validation_rows.csv", rows)
    (output_root / "track4_focused_basis_validation_claim_boundary.json").write_text(
        json.dumps(aggregate["claim_boundary"], indent=2),
        encoding="utf-8",
    )
    return summary_path


def run_validation(
    *,
    output_root: Path,
    corpora: Sequence[CorpusSpec],
    kernels: Sequence[str],
    seeds: Sequence[int],
    bases: Sequence[str],
    limit: int,
    mode: str = "cls_logits",
    rks_dim: int = 512,
    proposal_mode: str = "metric_softmax",
    adaptive_tpt_connectivity: bool = True,
    walker_temperature: float = 0.75,
    walker_gamma: float = 5.0,
    walker_k_neighbors: int = 8,
    dry_run: bool = False,
    skip_existing: bool = False,
    max_cells: int | None = None,
) -> Dict[str, Any]:
    cells = build_validation_cells(output_root=output_root, corpora=corpora, kernels=kernels, seeds=seeds)
    if max_cells is not None:
        cells = cells[: int(max_cells)]

    all_rows: List[Dict[str, Any]] = []
    cell_summaries: List[Dict[str, Any]] = []
    for cell in cells:
        config = ProbeConfig(
            corpus=Path(cell.corpus.corpus_arg),
            output_root=cell.output_dir,
            mode=mode,
            seed=int(cell.seed),
            limit=int(limit),
            kernel_type=cell.kernel,
            rks_dim=int(rks_dim),
            nli_cache_path=cell.nli_cache_path,
            proposal_mode=proposal_mode,
            adaptive_tpt_connectivity=bool(adaptive_tpt_connectivity),
            walker_temperature=float(walker_temperature),
            walker_gamma=float(walker_gamma),
            walker_k_neighbors=int(walker_k_neighbors),
        )
        status = "OK"
        error = None
        try:
            result = run_probe(config, bases, dry_run=dry_run, skip_existing=skip_existing)
        except Exception as exc:
            status = "FAILED"
            error = str(exc)
            result = {"rows": [], "summary_path": None, "run_records": []}

        scored_rows = result.get("ranked_rows") or result.get("rows", [])
        cell_rows = [_row_from_probe_row(cell, row) for row in scored_rows]
        all_rows.extend(cell_rows)
        boundary = result.get("claim_boundary") or probe_claim_boundary(scored_rows)
        cell_summaries.append(
            {
                "status": status,
                "error": error,
                "corpus": cell.corpus.name,
                "corpus_kind": cell.corpus.kind,
                "kernel": cell.kernel,
                "seed": int(cell.seed),
                "summary_path": result.get("summary_path"),
                "run_records": result.get("run_records", []),
                "claim_boundary": boundary,
            }
        )

    summary_path = write_validation_summary(
        output_root,
        cells=cells,
        cell_summaries=cell_summaries,
        rows=all_rows,
        bases=bases,
        kernels=kernels,
        seeds=seeds,
        limit=limit,
    )
    return {
        "summary_path": str(summary_path),
        "row_count": len(all_rows),
        "cell_count": len(cells),
        "failed_cell_count": sum(1 for cell in cell_summaries if cell.get("status") == "FAILED"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--corpora", nargs="+", default=list(DEFAULT_CORPORA))
    parser.add_argument("--synthetic-corpus", type=Path, default=None)
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--seeds", nargs="+", default=[str(seed) for seed in DEFAULT_SEEDS])
    parser.add_argument("--bases", nargs="+", default=list(DEFAULT_BASES))
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--mode", default="cls_logits")
    parser.add_argument("--rks-dim", type=int, default=512)
    parser.add_argument("--track4-proposal-mode", default="metric_softmax")
    parser.add_argument("--track4-adaptive-tpt-connectivity", action="store_true", default=True)
    parser.add_argument("--no-track4-adaptive-tpt-connectivity", action="store_false", dest="track4_adaptive_tpt_connectivity")
    parser.add_argument("--walker-temperature", type=float, default=0.75)
    parser.add_argument("--walker-gamma", type=float, default=5.0)
    parser.add_argument("--walker-k-neighbors", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    corpora = resolve_corpus_specs(args.corpora, synthetic_corpus=args.synthetic_corpus)
    seeds = [int(seed) for seed in args.seeds]
    result = run_validation(
        output_root=args.output_root,
        corpora=corpora,
        kernels=[str(kernel) for kernel in args.kernels],
        seeds=seeds,
        bases=[str(basis) for basis in args.bases],
        limit=int(args.limit),
        mode=str(args.mode),
        rks_dim=int(args.rks_dim),
        proposal_mode=str(args.track4_proposal_mode),
        adaptive_tpt_connectivity=bool(args.track4_adaptive_tpt_connectivity),
        walker_temperature=float(args.walker_temperature),
        walker_gamma=float(args.walker_gamma),
        walker_k_neighbors=int(args.walker_k_neighbors),
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
        max_cells=args.max_cells,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
