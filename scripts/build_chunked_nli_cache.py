#!/usr/bin/env python3
"""Build restartable NLI caches for large manifold runs.

This script isolates DeBERTa extraction into chunk subprocesses. Each chunk writes
its own `.pt` file, so large runs can resume after CUDA faults instead of losing
all completed extraction work. The final assembled cache matches the
`run_experiments.py --nli-cache-path` contract:

    {"nli_pairs": [...], "n_articles": N, "channel": "cls"}
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENTS = REPO_ROOT / "run_experiments.py"
MANIFEST_NAME = "chunked_nli_cache_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_articles(corpus: str, limit: int | None) -> List[Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_experiments

    return run_experiments.load_corpus(corpus, limit=limit)


def _chunk_path(chunks_dir: Path, start: int, end: int) -> Path:
    return chunks_dir / f"chunk_{start:06d}_{end:06d}.pt"


def _log_path(chunks_dir: Path, start: int, end: int) -> Path:
    return chunks_dir / "logs" / f"chunk_{start:06d}_{end:06d}.log"


def _chunk_is_valid(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    pairs = payload.get("nli_pairs")
    return isinstance(pairs, list) and len(pairs) == int(expected_count)


def _run_worker(args: argparse.Namespace) -> int:
    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA-only worker requested, but torch.cuda.is_available() is false.")

    articles = _load_articles(args.corpus, limit=args.limit)
    chunk_articles = articles[int(args.worker_start):int(args.worker_end)]
    out = Path(args.worker_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        torch.save(
            {
                "nli_pairs": [],
                "n_articles": 0,
                "channel": "cls",
                "worker_start": int(args.worker_start),
                "worker_end": int(args.worker_end),
                "dry_run": True,
            },
            out,
        )
        return 0

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.nli_extraction import NLIExtractor

    extractor = NLIExtractor(
        device=str(args.device),
        use_cls_tokens=True,
        paragraph_aware=True,
        paragraph_weights=[0.5, 0.3, 0.2],
        max_length=int(args.max_length),
    )
    nli_pairs = extractor.extract_nli_pairs(chunk_articles)
    torch.save(
        {
            "nli_pairs": nli_pairs,
            "n_articles": len(nli_pairs),
            "channel": "cls",
            "worker_start": int(args.worker_start),
            "worker_end": int(args.worker_end),
            "created_at": _utc_now(),
        },
        out,
    )
    return 0


def _worker_command(args: argparse.Namespace, start: int, end: int, chunk_out: Path) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--corpus",
        args.corpus,
        "--limit",
        str(args.limit),
        "--chunks-dir",
        str(args.chunks_dir),
        "--worker-start",
        str(start),
        "--worker-end",
        str(end),
        "--worker-out",
        str(chunk_out),
        "--device",
        str(args.device),
        "--max-length",
        str(args.max_length),
    ] + (["--dry-run"] if args.dry_run else [])


def _run_chunk(args: argparse.Namespace, start: int, end: int, chunk_out: Path) -> Dict[str, Any]:
    expected = end - start
    log = _log_path(args.chunks_dir, start, end)
    if args.dry_run:
        command = _worker_command(args, start, end, chunk_out)
        start_time = _utc_now()
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "\n".join(
                [
                    f"start: {start_time}",
                    f"end: {_utc_now()}",
                    "status: dry_run",
                    "returncode: 0",
                    "command:",
                    " ".join(shlex.quote(part) for part in command),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "start": start,
            "end": end,
            "count": expected,
            "status": "dry_run",
            "chunk_path": str(chunk_out),
            "log_path": str(log),
            "returncode": 0,
        }
    if _chunk_is_valid(chunk_out, expected):
        return {
            "start": start,
            "end": end,
            "count": expected,
            "status": "skipped_existing",
            "chunk_path": str(chunk_out),
            "log_path": str(log),
            "returncode": 0,
        }

    command = _worker_command(args, start, end, chunk_out)
    log.parent.mkdir(parents=True, exist_ok=True)
    stage_started = _utc_now()
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"start: {stage_started}\n")
        handle.write("command:\n")
        handle.write(" ".join(shlex.quote(part) for part in command))
        handle.write("\n\ncombined stdout/stderr:\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        ended = _utc_now()
        status = "succeeded" if completed.returncode == 0 and _chunk_is_valid(chunk_out, expected) else "failed"
        handle.write("\n")
        handle.write("=" * 80)
        handle.write("\n")
        handle.write(f"end: {ended}\n")
        handle.write(f"status: {status}\n")
        handle.write(f"returncode: {completed.returncode}\n")

    return {
        "start": start,
        "end": end,
        "count": expected,
        "status": status,
        "chunk_path": str(chunk_out),
        "log_path": str(log),
        "returncode": int(completed.returncode),
        "started_at": stage_started,
        "ended_at": ended,
    }


def _assemble_cache(cache_path: Path, chunk_paths: Sequence[Path], expected_n: int) -> Dict[str, Any]:
    all_pairs: List[Any] = []
    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        pairs = payload.get("nli_pairs")
        if not isinstance(pairs, list):
            raise RuntimeError(f"Invalid chunk payload: {path}")
        all_pairs.extend(pairs)
    if len(all_pairs) != int(expected_n):
        raise RuntimeError(f"Assembled {len(all_pairs)} NLI pairs, expected {expected_n}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "nli_pairs": all_pairs,
            "n_articles": int(expected_n),
            "channel": "cls",
            "created_at": _utc_now(),
            "source": "scripts/build_chunked_nli_cache.py",
            "chunk_count": len(chunk_paths),
        },
        cache_path,
    )
    return {
        "cache_path": str(cache_path),
        "n_articles": int(expected_n),
        "chunk_count": len(chunk_paths),
        "status": "succeeded",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a chunked CLS NLI cache for large manifold runs.")
    parser.add_argument("--corpus", default="real")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--cache-path", type=Path, required=False)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.chunks_dir = args.chunks_dir.resolve()
    if args.cache_path is None:
        args.cache_path = args.chunks_dir / f"{Path(args.corpus).stem}_limit_{args.limit}_cls_nli_cache.pt"
    args.cache_path = args.cache_path.resolve()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _run_worker(args)

    articles = _load_articles(args.corpus, limit=args.limit)
    n_articles = len(articles)
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.chunks_dir / MANIFEST_NAME
    chunks = [
        (start, min(start + int(args.chunk_size), n_articles))
        for start in range(0, n_articles, int(args.chunk_size))
    ]
    manifest: Dict[str, Any] = {
        "created_at": _utc_now(),
        "updated_at": None,
        "status": "running",
        "corpus": args.corpus,
        "limit": int(args.limit),
        "n_articles": int(n_articles),
        "chunk_size": int(args.chunk_size),
        "device": str(args.device),
        "max_length": int(args.max_length),
        "dry_run": bool(args.dry_run),
        "cache_path": str(args.cache_path),
        "chunks_dir": str(args.chunks_dir),
        "manifest_path": str(manifest_path),
        "chunks": [],
        "assembly": None,
    }
    _write_json(manifest_path, manifest)

    chunk_paths: List[Path] = []
    for start, end in chunks:
        chunk_out = _chunk_path(args.chunks_dir, start, end)
        attempt_payload = None
        for attempt in range(int(args.max_retries) + 1):
            attempt_payload = _run_chunk(args, start, end, chunk_out)
            attempt_payload["attempt"] = attempt + 1
            if attempt_payload["status"] in {"succeeded", "skipped_existing", "dry_run"}:
                break
        assert attempt_payload is not None
        manifest["chunks"].append(attempt_payload)
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        if attempt_payload["status"] not in {"succeeded", "skipped_existing", "dry_run"}:
            manifest["status"] = "failed"
            _write_json(manifest_path, manifest)
            print(f"Chunk failed: {start}:{end}; see {attempt_payload['log_path']}", file=sys.stderr)
            return int(attempt_payload.get("returncode") or 1)
        chunk_paths.append(chunk_out)

    if not args.dry_run:
        manifest["assembly"] = _assemble_cache(args.cache_path, chunk_paths, n_articles)
    else:
        manifest["assembly"] = {"status": "dry_run", "cache_path": str(args.cache_path)}
    manifest["status"] = "dry_run" if args.dry_run else "succeeded"
    manifest["updated_at"] = _utc_now()
    _write_json(manifest_path, manifest)
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
