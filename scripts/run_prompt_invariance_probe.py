#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_prompt_bank(path: Path) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        prompts = payload.get("prompts") or payload.get("hypotheses") or []
        if isinstance(prompts, list):
            return [str(item).strip() for item in prompts if str(item).strip()]
    raise ValueError(f"Unsupported prompt bank format: {path}")


def _fingerprint(prompts: Sequence[str]) -> str:
    raw = json.dumps(list(prompts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _parse_bank(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), Path(path)
    path = Path(raw)
    return path.stem, path


def run_probe(*, prompt_banks: Sequence[Tuple[str, Path]], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    fingerprints = set()
    prompt_counts = set()
    for name, path in prompt_banks:
        try:
            prompts = _load_prompt_bank(path)
            fingerprint = _fingerprint(prompts)
            rows.append(
                {
                    "bank_name": name,
                    "path": str(path),
                    "status": "OK",
                    "prompt_count": len(prompts),
                    "fingerprint": fingerprint,
                }
            )
            fingerprints.add(fingerprint)
            prompt_counts.add(len(prompts))
        except Exception as exc:
            rows.append(
                {
                    "bank_name": name,
                    "path": str(path),
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append(f"bank_load_failed:{name}")
    if len(rows) < 3:
        failures.append("fewer_than_three_prompt_banks")
    if len(fingerprints) < max(1, len(rows)):
        failures.append("prompt_banks_not_distinct")
    if len(prompt_counts) > 1:
        failures.append("prompt_bank_prompt_counts_differ")
    payload = {
        "schema_version": "1.0",
        "summary_type": "prompt_invariance_probe",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank_count": len(rows),
        "distinct_bank_count": len(fingerprints),
        "prompt_counts": sorted(prompt_counts),
        "prompt_invariance_pass": not failures,
        "failure_reasons": failures,
        "claim_boundary": {
            "safe_claim": "prompt bank inventory is registered",
            "unsafe_claim": "semantic invariance is proven without rerunning extraction under alternate banks",
        },
        "banks": rows,
    }
    out = output_dir / "prompt_invariance_summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["artifact"] = str(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-bank",
        action="append",
        default=[],
        help="Prompt bank as name=path or path. Repeat for invariance banks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "prompt_invariance_probe" / "latest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    banks = [_parse_bank(raw) for raw in args.prompt_bank]
    if not banks:
        banks = [("canonical", ROOT / "probe_hypotheses.json")]
    payload = run_probe(prompt_banks=banks, output_dir=args.output_dir)
    print(json.dumps({"prompt_invariance_pass": payload["prompt_invariance_pass"], "failure_reasons": payload["failure_reasons"], "artifact": payload["artifact"]}, indent=2))


if __name__ == "__main__":
    main()
