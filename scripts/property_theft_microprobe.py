#!/usr/bin/env python3
"""Build a tiny "property is theft" semantic environment.

The probe is intentionally small and synthetic. It does not try to validate the
full model. It tests one reviewer-facing idea from the paper: articles can be
topically close while perspectivally difficult to traverse.

The environment creates same-event articles split across a property-rights frame
and a theft/dispossession frame, then reports whether cross-frame pairs have
larger work gaps than same-frame pairs under the existing terrain summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.verification.scientific_summaries import (  # noqa: E402
    summarize_terrain_incremental_signal,
    summarize_track4_traversal,
)
from core.physarum_walk import SemanticWalker  # noqa: E402


@dataclass(frozen=True)
class MicroArticle:
    index: int
    hidden_label: str
    frame: str
    zone: str
    title: str
    content: str
    embedding: Sequence[float]
    density: float
    stress: float
    work: float
    x: float
    y: float


def _articles() -> List[MicroArticle]:
    """Return a hand-controlled toy semantic field.

    Embedding dimensions:
    0 topic_land, 1 property_rights, 2 theft_dispossession,
    3 legal_register, 4 moral_injury, 5 bureaucratic_neutrality.
    """
    return [
        MicroArticle(
            0,
            "settlement_land_case",
            "bridge",
            "Bridge",
            "Land registry records are disputed after a settlement expansion notice",
            "The notice describes deeds, claims, residents, and a pending legal review.",
            [1.00, 0.35, 0.35, 0.75, 0.35, 0.80],
            0.92,
            0.10,
            12.0,
            -0.20,
            0.00,
        ),
        MicroArticle(
            1,
            "settlement_land_case",
            "property",
            "Tightrope",
            "Property deeds establish lawful ownership of the hillside parcel",
            "The article frames the dispute as a question of title, registry procedure, and lawful transfer.",
            [1.00, 1.00, 0.05, 0.88, 0.05, 0.25],
            0.34,
            0.22,
            18.0,
            1.00,
            0.35,
        ),
        MicroArticle(
            2,
            "settlement_land_case",
            "property",
            "Tightrope",
            "Court filing says ownership documents support the settlement claim",
            "The language emphasizes private title, contract, registration, and procedural compliance.",
            [1.00, 0.96, 0.08, 0.92, 0.08, 0.28],
            0.36,
            0.20,
            19.0,
            1.20,
            -0.20,
        ),
        MicroArticle(
            3,
            "settlement_land_case",
            "theft",
            "Void",
            "Families describe the same land transfer as confiscation",
            "The article frames the transfer as seizure, dispossession, coercion, and loss of home.",
            [1.00, 0.04, 1.00, 0.20, 0.95, 0.10],
            0.18,
            0.92,
            65.0,
            -1.05,
            0.45,
        ),
        MicroArticle(
            4,
            "settlement_land_case",
            "theft",
            "Void",
            "Residents call the property ruling legalized theft",
            "The language emphasizes removal, stolen inheritance, unequal law, and moral injury.",
            [1.00, 0.07, 0.96, 0.24, 1.00, 0.12],
            0.20,
            0.90,
            67.0,
            -1.25,
            -0.25,
        ),
        MicroArticle(
            5,
            "settlement_land_case",
            "bridge",
            "Bridge",
            "Mediators seek compensation and shared access after land dispute",
            "The article combines legal title, resident claims, access rights, and negotiated remedy.",
            [1.00, 0.52, 0.48, 0.72, 0.55, 0.70],
            0.88,
            0.18,
            14.0,
            0.00,
            0.75,
        ),
        MicroArticle(
            6,
            "settlement_land_case",
            "collision",
            "Swamp",
            "Lawful deed or dispossession: both sides cite rights over the parcel",
            "The article maximally activates both property and theft frames without resolving the conflict.",
            [1.00, 0.88, 0.86, 0.86, 0.88, 0.20],
            0.78,
            0.82,
            46.0,
            0.00,
            -0.80,
        ),
        MicroArticle(
            7,
            "tax_collection_case",
            "property",
            "Tightrope",
            "Small business owners contest a new asset levy",
            "The article frames tax collection as a dispute about retained earnings and ownership.",
            [0.15, 0.88, 0.10, 0.72, 0.10, 0.45],
            0.35,
            0.24,
            20.0,
            2.60,
            0.30,
        ),
        MicroArticle(
            8,
            "tax_collection_case",
            "property",
            "Tightrope",
            "Opponents say the levy violates property expectations",
            "The language emphasizes entitlement, assets, planning, and formal legality.",
            [0.12, 0.92, 0.08, 0.70, 0.09, 0.42],
            0.34,
            0.25,
            21.0,
            2.80,
            -0.25,
        ),
        MicroArticle(
            9,
            "tax_collection_case",
            "theft",
            "Void",
            "Activists call the levy extraction from workers",
            "The article frames the same fiscal act as coercive taking and institutionalized theft.",
            [0.12, 0.08, 0.90, 0.24, 0.84, 0.18],
            0.20,
            0.86,
            51.0,
            1.60,
            0.40,
        ),
        MicroArticle(
            10,
            "tax_collection_case",
            "theft",
            "Void",
            "The levy is described as wage confiscation",
            "The language emphasizes extraction, coercion, loss, and class injury.",
            [0.10, 0.05, 0.94, 0.20, 0.88, 0.16],
            0.18,
            0.88,
            52.0,
            1.45,
            -0.30,
        ),
    ]


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    denom = max(float(np.linalg.norm(av) * np.linalg.norm(bv)), 1e-12)
    return float(1.0 - float(np.dot(av, bv) / denom))


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _pair_summary(articles: Sequence[MicroArticle]) -> Dict[str, Any]:
    same_frame_work_gaps: List[float] = []
    cross_frame_work_gaps: List[float] = []
    same_frame_cosine: List[float] = []
    cross_frame_cosine: List[float] = []
    rows: List[Dict[str, Any]] = []
    for left, right in combinations(articles, 2):
        if left.hidden_label != right.hidden_label:
            continue
        if {left.frame, right.frame} - {"property", "theft"}:
            continue
        work_gap = abs(float(left.work) - float(right.work))
        cosine_distance = _cosine_distance(left.embedding, right.embedding)
        same_frame = left.frame == right.frame
        if same_frame:
            same_frame_work_gaps.append(work_gap)
            same_frame_cosine.append(cosine_distance)
        else:
            cross_frame_work_gaps.append(work_gap)
            cross_frame_cosine.append(cosine_distance)
        rows.append(
            {
                "hidden_label": left.hidden_label,
                "left_article_idx": left.index,
                "right_article_idx": right.index,
                "left_frame": left.frame,
                "right_frame": right.frame,
                "same_frame": same_frame,
                "work_gap": work_gap,
                "cosine_distance": cosine_distance,
            }
        )
    same_work = _mean(same_frame_work_gaps)
    cross_work = _mean(cross_frame_work_gaps)
    same_cos = _mean(same_frame_cosine)
    cross_cos = _mean(cross_frame_cosine)
    return {
        "same_topic_same_frame_pair_count": len(same_frame_work_gaps),
        "same_topic_cross_frame_pair_count": len(cross_frame_work_gaps),
        "mean_same_frame_work_gap": same_work,
        "mean_cross_frame_work_gap": cross_work,
        "cross_minus_same_frame_work_gap": (
            float(cross_work - same_work)
            if same_work is not None and cross_work is not None
            else None
        ),
        "mean_same_frame_cosine_distance": same_cos,
        "mean_cross_frame_cosine_distance": cross_cos,
        "cross_minus_same_frame_cosine_distance": (
            float(cross_cos - same_cos)
            if same_cos is not None and cross_cos is not None
            else None
        ),
        "pairs": rows,
    }


def _identity_basis(x: torch.Tensor) -> torch.Tensor:
    return x


def build_property_theft_microprobe(output_dir: Path, *, run_walker: bool = True) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = _articles()

    monolith_rows = [
        {
            "index": article.index,
            "bt_uid": f"property_theft_{article.index}",
            "source": "microprobe",
            "publication": "microprobe",
            "author": "synthetic",
            "hidden_label": article.hidden_label,
            "frame": article.frame,
            "label": article.hidden_label,
            "title": article.title,
            "content_preview": article.content,
            "density": article.density,
            "stress": article.stress,
            "z_height": article.stress,
            "zone": article.zone,
            "color_code": "#00F0FF" if article.zone == "Bridge" else "#FFFFCC" if article.zone == "Tightrope" else "#FF00FF" if article.zone == "Swamp" else "#111111",
            "verdict": "MECHANICAL",
            "delta": article.stress,
            "d_spectral": _cosine_distance(article.embedding, articles[0].embedding),
            "w_actual": article.work,
            "x": article.x,
            "y": article.y,
        }
        for article in articles
    ]
    _write_csv(
        output_dir / "MONOLITH_DATA.csv",
        monolith_rows,
        fieldnames=[
            "index",
            "bt_uid",
            "source",
            "publication",
            "author",
            "hidden_label",
            "frame",
            "label",
            "title",
            "content_preview",
            "density",
            "stress",
            "z_height",
            "zone",
            "color_code",
            "verdict",
            "delta",
            "d_spectral",
            "w_actual",
            "x",
            "y",
        ],
    )
    _write_csv(
        output_dir / "labels" / "hidden_groups.csv",
        [
            {"article_id": article.index, "group_topic": article.hidden_label, "frame": article.frame}
            for article in articles
        ],
        fieldnames=["article_id", "group_topic", "frame"],
    )

    walker_result: Dict[str, Any] = {"status": "SKIPPED"}
    if run_walker:
        embeddings = torch.tensor([article.embedding for article in articles], dtype=torch.float32)
        density = torch.tensor([article.density for article in articles], dtype=torch.float32)
        stress = torch.tensor([article.stress for article in articles], dtype=torch.float32)
        coords = torch.tensor([[article.x, article.y] for article in articles], dtype=torch.float32)
        walker = SemanticWalker(
            embeddings=embeddings,
            rks_basis=_identity_basis,
            temperature=0.35,
            track3_density=density,
            metric_stress=stress,
            article_coords_2d=coords,
        )
        walker_result = walker.run_stress_triggered_cyclic_walk(
            max_steps=90,
            gamma=5.0,
            k_neighbors=4,
            start_seed=42,
            output_dir=str(output_dir),
        )

    terrain_incremental = summarize_terrain_incremental_signal(output_dir)
    track4_summary = summarize_track4_traversal(output_dir)
    pair_summary = _pair_summary(articles)
    summary = {
        "schema_version": "1.0",
        "probe_name": "property_is_theft_microprobe",
        "output_dir": str(output_dir),
        "article_count": len(articles),
        "semantic_basis": [
            "topic_land",
            "property_rights",
            "theft_dispossession",
            "legal_register",
            "moral_injury",
            "bureaucratic_neutrality",
        ],
        "claim_under_test": (
            "Same-event cross-frame pairs should be harder to traverse than "
            "same-event same-frame pairs."
        ),
        "articles": [asdict(article) for article in articles],
        "pair_summary": pair_summary,
        "terrain_incremental_signal": terrain_incremental,
        "track4_traversal": track4_summary,
        "walker_result": {
            "catalyst_indices": walker_result.get("catalyst_indices"),
            "catalyst_zones": walker_result.get("catalyst_zones"),
            "anchor_summaries": walker_result.get("anchor_summaries"),
            "cognitive_horizon": walker_result.get("cognitive_horizon"),
            "markov_observables": walker_result.get("markov_observables"),
        },
        "passes_microprobe": bool(
            terrain_incremental.get("safe_for_thesis_claim")
            and (pair_summary.get("cross_minus_same_frame_work_gap") or 0.0) > 0.0
            and (pair_summary.get("cross_minus_same_frame_cosine_distance") or 0.0) > 0.0
        ),
    }
    (output_dir / "property_theft_microprobe_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "track4_traversal_summary.json").write_text(
        json.dumps(track4_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "terrain_incremental_signal_summary.json").write_text(
        json.dumps(terrain_incremental, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/microprobes/property_theft/latest",
        help="Directory to receive MONOLITH_DATA.csv, cyclic_paths.npz, and summaries.",
    )
    parser.add_argument("--no-walker", action="store_true", help="Skip the Track 4 walker export.")
    args = parser.parse_args()

    summary = build_property_theft_microprobe(
        Path(args.output_dir),
        run_walker=not args.no_walker,
    )
    pair = summary["pair_summary"]
    terrain = summary["terrain_incremental_signal"]
    print("Property/Theft microprobe complete")
    print(f"- output_dir: {summary['output_dir']}")
    print(f"- passes_microprobe: {summary['passes_microprobe']}")
    print(f"- cross_minus_same_frame_work_gap: {pair['cross_minus_same_frame_work_gap']:.3f}")
    print(f"- cross_minus_same_frame_cosine_distance: {pair['cross_minus_same_frame_cosine_distance']:.3f}")
    print(f"- terrain_incremental_status: {terrain['status']}")
    print(f"- terrain_cross_minus_same_work_gap: {terrain['cross_minus_same_work_gap']:.3f}")
    return 0 if summary["passes_microprobe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
