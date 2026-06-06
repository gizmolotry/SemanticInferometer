#!/usr/bin/env python3
"""Build a source-balanced JSONL slice for real-corpus proxy validation.

The source-proxy validation harness needs repeated outlets. The focused proof
leaves intentionally sampled broad source diversity, so this helper creates a
small, deterministic repeated-source slice from existing JSONL corpora without
changing the pipeline itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data_utils import extract_article_text


SOURCE_COLUMNS = ("source", "publisher", "publication", "outlet")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _normalize_source(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "null", "unknown", "missing"}:
        return ""
    return text.removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")


def _source_for_row(row: Dict[str, Any]) -> Tuple[str, str]:
    for column in SOURCE_COLUMNS:
        value = _normalize_source(row.get(column))
        if value:
            return value, column
    return "", ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            source, source_column = _source_for_row(row)
            if not source:
                continue
            try:
                text = extract_article_text(row)
            except Exception:
                continue
            if not text.strip():
                continue
            item = dict(row)
            item["_source_balancer_source"] = source
            item["_source_balancer_source_column"] = source_column
            rows.append(item)
    return rows


def _sort_group(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("published_at") or row.get("date") or row.get("fetched_at") or ""),
            str(row.get("id") or row.get("url") or row.get("title") or ""),
        ),
    )


def _spread_indices(n: int, k: int) -> List[int]:
    if k > n:
        raise ValueError(f"Cannot select {k} rows from group of {n}")
    if k == n:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _select_from_group(
    rows: Sequence[Dict[str, Any]],
    *,
    articles_per_source: int,
    selection: str,
    seed: int,
) -> List[Dict[str, Any]]:
    ordered = _sort_group(rows)
    if selection == "first":
        return ordered[:articles_per_source]
    if selection == "last":
        return ordered[-articles_per_source:]
    if selection == "spread":
        return [ordered[idx] for idx in _spread_indices(len(ordered), articles_per_source)]
    if selection == "random":
        import random

        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(ordered)), articles_per_source))
        return [ordered[idx] for idx in indices]
    raise ValueError(f"Unknown selection mode: {selection}")


def _parse_sources(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    parsed: List[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            source = _normalize_source(part)
            if source and source not in parsed:
                parsed.append(source)
    return parsed


def build_source_balanced_slice(
    input_path: Path,
    *,
    n_sources: int = 8,
    articles_per_source: int = 10,
    sources: Optional[Sequence[str]] = None,
    selection: str = "spread",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_jsonl(input_path)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_columns: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        source = str(row["_source_balancer_source"])
        groups[source].append(row)
        source_columns[source][str(row["_source_balancer_source_column"])] += 1

    requested_sources = _parse_sources(sources)
    eligible = {
        source: group
        for source, group in groups.items()
        if len(group) >= int(articles_per_source)
    }
    if requested_sources:
        missing = [source for source in requested_sources if source not in eligible]
        if missing:
            raise ValueError(
                "Requested sources lack enough usable articles: "
                + ", ".join(missing)
                + f" (need {articles_per_source} each)"
            )
        chosen_sources = requested_sources[: int(n_sources)]
    else:
        chosen_sources = [
            source
            for source, _count in sorted(
                ((source, len(group)) for source, group in eligible.items()),
                key=lambda item: (-item[1], item[0]),
            )[: int(n_sources)]
        ]

    if len(chosen_sources) < int(n_sources):
        raise ValueError(
            f"Only {len(chosen_sources)} eligible sources found; need {n_sources} "
            f"with at least {articles_per_source} articles each."
        )

    selected: List[Dict[str, Any]] = []
    for source in chosen_sources:
        picked = _select_from_group(
            eligible[source],
            articles_per_source=int(articles_per_source),
            selection=selection,
            seed=int(seed),
        )
        for within_source_idx, row in enumerate(picked):
            out = dict(row)
            out["source"] = source
            out["publisher"] = source
            out["source_proxy_label"] = source
            out["source_balanced_within_source_index"] = within_source_idx
            out["source_balanced_selection_policy"] = selection
            out.pop("_source_balancer_source", None)
            out.pop("_source_balancer_source_column", None)
            selected.append(out)

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "selection_policy": selection,
        "seed": int(seed),
        "requested_n_sources": int(n_sources),
        "articles_per_source": int(articles_per_source),
        "selected_article_count": len(selected),
        "selected_source_count": len(chosen_sources),
        "selected_sources": chosen_sources,
        "selected_source_counts": dict(Counter(row["source"] for row in selected)),
        "source_columns_seen": {
            source: dict(source_columns[source])
            for source in chosen_sources
        },
        "eligible_source_count": len(eligible),
        "raw_usable_article_count": len(rows),
        "raw_source_count": len(groups),
        "claim_boundary": {
            "source_balanced_slice_for_proxy_validation": True,
            "does_not_use_source_labels_as_model_inputs": True,
            "source_labels_reserved_for_downstream_evaluation": True,
        },
    }
    return selected, manifest


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--n-sources", type=int, default=8)
    parser.add_argument("--articles-per-source", type=int, default=10)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--selection", choices=["spread", "first", "last", "random"], default="spread")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows, manifest = build_source_balanced_slice(
        args.input,
        n_sources=args.n_sources,
        articles_per_source=args.articles_per_source,
        sources=args.sources,
        selection=args.selection,
        seed=args.seed,
    )
    write_jsonl(rows, args.out)
    manifest_out = args.manifest_out or args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest["output_path"] = str(args.out)
    manifest_out.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
