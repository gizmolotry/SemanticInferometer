#!/usr/bin/env python3
"""Report completion status for Track 4 observer-state replay roots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_BRANCHES = ("baseline_raw_action", "null_calibrated_hysteresis", "richer_walker_state")
DEFAULT_BASES = ("track2", "integrated")
DEFAULT_CORPORA = ("real", "control_random", "control_shuffled", "control_constant")
DEFAULT_KERNELS = ("rbf", "matern", "imq")
DEFAULT_SEEDS = (42, 420, 4200)
REAL_ABLATIONS = ("observer_disabled", "observer_shuffled", "zero_hysteresis")
DEFAULT_RUN_SUFFIX = "unknown"


def _expected_names(
    *,
    branches: Sequence[str],
    bases: Sequence[str],
    corpora: Sequence[str],
    kernels: Sequence[str],
    seeds: Sequence[int],
    run_suffix: str = DEFAULT_RUN_SUFFIX,
) -> List[str]:
    names: List[str] = []
    for branch in branches:
        for basis in bases:
            for seed in seeds:
                for kernel in kernels:
                    for corpus in corpora:
                        names.append(f"{branch}_{corpus}_{kernel}_seed{seed}_{basis}_full_{run_suffix}")
                    names.extend(
                        f"{branch}_real_{kernel}_seed{seed}_{basis}_{ablation}_{run_suffix}"
                        for ablation in REAL_ABLATIONS
                    )
    return names


def _expected_names_from_inventory(inventory: Dict[str, Any], *, run_suffix: str = DEFAULT_RUN_SUFFIX) -> List[str]:
    """Build the expected replay matrix from the runner manifest.

    The replay runner writes the exact leaves, bases, and action-branch specs it
    intended to execute. Prefer that manifest over reconstructing a grid from
    defaults so status reports do not invent missing cells for subset runs.
    """

    branches = [str(item) for item in inventory.get("action_branches") or []]
    bases = [str(item) for item in inventory.get("bases") or []]
    leaves = list(inventory.get("leaves") or [])
    branch_specs = dict(inventory.get("action_branch_specs") or {})
    names: List[str] = []
    for branch in branches:
        spec = dict(branch_specs.get(branch) or {})
        branch_bases = [str(spec["force_basis"])] if spec.get("force_basis") else bases
        for basis in branch_bases:
            for leaf in leaves:
                corpus = str(leaf.get("corpus", ""))
                kernel = str(leaf.get("kernel", ""))
                seed = leaf.get("seed")
                if not corpus or not kernel or seed is None:
                    continue
                names.append(f"{branch}_{corpus}_{kernel}_seed{int(seed)}_{basis}_full_{run_suffix}")
                if corpus == "real":
                    names.extend(
                        f"{branch}_real_{kernel}_seed{int(seed)}_{basis}_{ablation}_{run_suffix}"
                        for ablation in REAL_ABLATIONS
                    )
    return names


def _infer_run_suffix(names: Iterable[str]) -> str:
    suffix_counts: Dict[str, int] = {}
    for name in names:
        suffix = name.rsplit("_", 1)[-1]
        if len(suffix) == 8 and suffix.isdigit():
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    if not suffix_counts:
        return DEFAULT_RUN_SUFFIX
    return sorted(suffix_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _infer_summary_key(run_name: str, branches: Sequence[str] = DEFAULT_BRANCHES) -> Tuple[str, str, str]:
    branch = "other"
    for candidate in sorted((str(item) for item in branches), key=len, reverse=True):
        if run_name.startswith(candidate):
            branch = candidate
            break
    basis = "integrated" if "_integrated_" in run_name else "track2" if "_track2_" in run_name else "unknown"
    variant = "full"
    for ablation in REAL_ABLATIONS:
        if f"_{ablation}_" in run_name:
            variant = ablation
            break
    return branch, basis, variant


def summarize_replay_root(
    root: Path,
    *,
    branches: Sequence[str] = DEFAULT_BRANCHES,
    bases: Sequence[str] = DEFAULT_BASES,
    corpora: Sequence[str] = DEFAULT_CORPORA,
    kernels: Sequence[str] = DEFAULT_KERNELS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    inventory_path: Optional[Path] = None,
    prefer_inventory: bool = True,
    run_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root)
    inventory_payload: Optional[Dict[str, Any]] = None
    inventory_source: Optional[Path] = None
    if prefer_inventory:
        candidate_inventory = Path(inventory_path) if inventory_path else root / "observer_state_matrix_inventory.json"
        if candidate_inventory.exists():
            inventory_payload = json.loads(candidate_inventory.read_text(encoding="utf-8"))
            inventory_source = candidate_inventory

    completed_paths = sorted(root.rglob("track4_action_summary.json"))
    completed_names = {path.parent.name for path in completed_paths}
    newest_completed_path = max(completed_paths, key=lambda path: path.stat().st_mtime) if completed_paths else None
    oldest_completed_path = min(completed_paths, key=lambda path: path.stat().st_mtime) if completed_paths else None
    newest_mtime = newest_completed_path.stat().st_mtime if newest_completed_path else None
    oldest_mtime = oldest_completed_path.stat().st_mtime if oldest_completed_path else None
    elapsed_minutes = (
        (float(newest_mtime) - float(oldest_mtime)) / 60.0
        if newest_mtime is not None and oldest_mtime is not None and newest_mtime > oldest_mtime
        else None
    )
    throughput_per_hour = (
        (len(completed_paths) / (elapsed_minutes / 60.0))
        if elapsed_minutes is not None and elapsed_minutes > 0
        else None
    )
    inventory_suffix = None
    if inventory_payload is not None:
        raw_suffix = inventory_payload.get("run_suffix")
        if raw_suffix is not None and str(raw_suffix).strip():
            inventory_suffix = str(raw_suffix).strip()
    selected_run_suffix = str(run_suffix or inventory_suffix or _infer_run_suffix(completed_names))
    if inventory_payload is not None:
        expected_names = _expected_names_from_inventory(inventory_payload, run_suffix=selected_run_suffix)
        expected_source = "inventory"
        active_branches = [str(item) for item in inventory_payload.get("action_branches") or branches]
    else:
        expected_names = _expected_names(
            branches=branches,
            bases=bases,
            corpora=corpora,
            kernels=kernels,
            seeds=seeds,
            run_suffix=selected_run_suffix,
        )
        expected_source = "arguments"
        active_branches = [str(item) for item in branches]

    expected = set(expected_names)
    completed_expected = completed_names & expected
    missing = sorted(expected - completed_names)
    extra = sorted(completed_names - expected)
    by_group: Dict[str, int] = {}
    for name in completed_names:
        branch, basis, variant = _infer_summary_key(name, active_branches)
        key = f"{branch}|{basis}|{variant}"
        by_group[key] = by_group.get(key, 0) + 1
    inventory_warning = None
    if inventory_payload is not None and bool(inventory_payload.get("dry_run")) and completed_names:
        inventory_warning = "inventory_records_dry_run_but_completed_summaries_exist"
    return {
        "schema_version": "1.0",
        "summary_type": "track4_replay_status",
        "root": str(root),
        "expected_source": expected_source,
        "inventory_path": str(inventory_source) if inventory_source else None,
        "inventory_dry_run": bool(inventory_payload.get("dry_run")) if inventory_payload else None,
        "inventory_skip_existing": bool(inventory_payload.get("skip_existing")) if inventory_payload else None,
        "inventory_leaf_count": int(inventory_payload.get("leaf_count", 0)) if inventory_payload else None,
        "inventory_warning": inventory_warning,
        "run_suffix": selected_run_suffix,
        "expected_count": len(expected),
        "completed_count": len(completed_expected),
        "raw_completed_count": len(completed_names),
        "last_completed_summary": str(newest_completed_path) if newest_completed_path else None,
        "last_completed_at_unix": newest_mtime,
        "minutes_since_last_completion": (
            (time.time() - float(newest_mtime)) / 60.0 if newest_mtime is not None else None
        ),
        "completed_throughput_per_hour": throughput_per_hour,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "completion_ratio": (len(completed_expected) / float(len(expected))) if expected else 0.0,
        "complete": not missing,
        "by_group": dict(sorted(by_group.items())),
        "missing_sample": missing[:50],
        "extra_sample": extra[:50],
    }


def _csv_or_default(values: Optional[Sequence[str]], defaults: Sequence[str]) -> List[str]:
    if not values:
        return list(defaults)
    out: List[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out


def _csv_int_or_default(values: Optional[Sequence[str]], defaults: Sequence[int]) -> List[int]:
    return [int(value) for value in _csv_or_default(values, [str(item) for item in defaults])]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--branch", action="append", default=None)
    parser.add_argument("--basis", action="append", default=None)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--kernel", action="append", default=None)
    parser.add_argument("--seed", action="append", default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional replay directory suffix such as 20260521. Defaults to the most common completed suffix.",
    )
    parser.add_argument(
        "--no-inventory",
        action="store_true",
        help="Ignore observer_state_matrix_inventory.json and reconstruct the expected grid from CLI/default arguments.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return exit code 1 when expected replay cells are still missing.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = summarize_replay_root(
        args.root,
        branches=_csv_or_default(args.branch, DEFAULT_BRANCHES),
        bases=_csv_or_default(args.basis, DEFAULT_BASES),
        corpora=_csv_or_default(args.corpus, DEFAULT_CORPORA),
        kernels=_csv_or_default(args.kernel, DEFAULT_KERNELS),
        seeds=_csv_int_or_default(args.seed, DEFAULT_SEEDS),
        inventory_path=args.inventory,
        prefer_inventory=not bool(args.no_inventory),
        run_suffix=args.run_suffix,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.require_complete and not bool(payload.get("complete")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
