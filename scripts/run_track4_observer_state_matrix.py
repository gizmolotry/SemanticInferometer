#!/usr/bin/env python3
"""Run Track 4 observer-state action replay over existing observer artifacts.

This is a replay harness only. It does not rerun DeBERTa or regenerate observer
artifacts; it finds existing ``observer_<seed>.pt`` leaves, runs the least-action
graph for requested bases, and writes observer-state ablation summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_track4_action_graph_runs import summarize_observer_state_ablation


KNOWN_CORPORA = ("real", "control_constant", "control_random", "control_shuffled", "synthetic_microprobe")
KNOWN_KERNELS = ("rbf", "matern", "imq", "rq", "laplacian")
DEFAULT_ACTION_BRANCHES = (
    "baseline_raw_action",
    "track2_default",
    "null_calibrated_hysteresis",
    "richer_walker_state",
    "separated_action_channels",
    "virtual_transition_states",
    "path_ensemble_tpt",
    "per_basis_gates",
)
BRANCH_PRESETS: Dict[str, Sequence[str]] = {
    "all": DEFAULT_ACTION_BRANCHES,
    "repair3": (
        "baseline_raw_action",
        "null_calibrated_hysteresis",
        "richer_walker_state",
    ),
}


@dataclass(frozen=True)
class ActionBranchSpec:
    name: str
    description: str
    branch_type: str
    core_action_mode: str = "baseline"
    observer_state_mode: str = "artifact"
    observer_shuffle_seed: int = 42
    virtual_interpolation_steps: int = 0
    force_basis: str | None = None
    k_neighbors_delta: int = 0
    observer_transport_weight: float = 1.0
    hysteresis_weight: float = 0.5
    shear_weight: float = 1.0
    stress_weight: float = 2.0
    void_weight: float = 1.25
    curvature_weight: float = 1.0
    target_count_per_anchor: int = 1
    max_paths_multiplier: float = 1.0


ACTION_BRANCH_SPECS: Dict[str, ActionBranchSpec] = {
    "baseline_raw_action": ActionBranchSpec(
        name="baseline_raw_action",
        description="Baseline core action replay.",
        branch_type="baseline",
        core_action_mode="baseline",
    ),
    "track2_default": ActionBranchSpec(
        name="track2_default",
        description="Stable thesis-safe default: force Track 2 action basis.",
        branch_type="basis_policy",
        core_action_mode="baseline",
        force_basis="track2",
    ),
    "null_calibrated_hysteresis": ActionBranchSpec(
        name="null_calibrated_hysteresis",
        description="Null-calibrated action replay with a fixed null observer shuffle seed.",
        branch_type="calibration",
        core_action_mode="null_calibrated",
        observer_shuffle_seed=0,
        hysteresis_weight=0.75,
    ),
    "richer_walker_state": ActionBranchSpec(
        name="richer_walker_state",
        description="Richer state-mode replay with observer transport, directed hysteresis, curvature, and virtual rhetorical intermediates.",
        branch_type="state_model",
        core_action_mode="richer_state",
        virtual_interpolation_steps=3,
        observer_transport_weight=1.5,
        hysteresis_weight=1.0,
        curvature_weight=2.0,
    ),
    "separated_action_channels": ActionBranchSpec(
        name="separated_action_channels",
        description="Keep action channels interpretable; branch summary reports component separation before fusion.",
        branch_type="channel_ablation",
        core_action_mode="baseline",
        shear_weight=0.75,
        stress_weight=1.5,
    ),
    "virtual_transition_states": ActionBranchSpec(
        name="virtual_transition_states",
        description="Virtual transition-state replay using interpolation-step metadata plus expanded graph support.",
        branch_type="graph_densification_proxy",
        core_action_mode="virtual_transition",
        virtual_interpolation_steps=3,
        k_neighbors_delta=6,
        curvature_weight=0.5,
    ),
    "path_ensemble_tpt": ActionBranchSpec(
        name="path_ensemble_tpt",
        description="Approximate path ensemble/TPT evidence with more targets per anchor and larger path budget.",
        branch_type="ensemble_probe",
        core_action_mode="baseline",
        target_count_per_anchor=2,
        max_paths_multiplier=2.0,
    ),
    "per_basis_gates": ActionBranchSpec(
        name="per_basis_gates",
        description="No physics change; verifies strict basis/seed/kernel gates cannot be hidden by pooling.",
        branch_type="gate_policy",
        core_action_mode="baseline",
    ),
}


@dataclass(frozen=True)
class ObserverLeaf:
    path: Path
    corpus: str
    kernel: str
    seed: int


def _parse_seed(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("observer_"):
        raw = stem.split("observer_", 1)[1]
        try:
            return int(raw)
        except Exception:
            return None
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("seed_", 1)[1])
            except Exception:
                continue
    return None


def _parse_leaf(path: Path) -> ObserverLeaf | None:
    parts = [part.lower() for part in path.parts]
    corpus = next((part for part in parts if part in KNOWN_CORPORA), None)
    kernel = next((part for part in parts if part in KNOWN_KERNELS), None)
    seed = _parse_seed(path)
    if corpus is None or kernel is None or seed is None:
        return None
    return ObserverLeaf(path=path, corpus=corpus, kernel=kernel, seed=int(seed))


def discover_observer_leaves(input_root: Path) -> List[ObserverLeaf]:
    leaves: Dict[tuple[str, str, int], ObserverLeaf] = {}
    for path in Path(input_root).rglob("observer_*.pt"):
        if "relativity_cache" in {part.lower() for part in path.parts}:
            continue
        leaf = _parse_leaf(path)
        if leaf is None:
            continue
        key = (leaf.corpus, leaf.kernel, leaf.seed)
        current = leaves.get(key)
        if current is None or _artifact_preference(leaf.path) > _artifact_preference(current.path):
            leaves[key] = leaf
    return sorted(leaves.values(), key=lambda leaf: (leaf.seed, leaf.kernel, leaf.corpus, str(leaf.path)))


def _artifact_preference(path: Path) -> int:
    text = str(path).lower()
    if "track4_basis_track2" in text:
        return 3
    if "track4_basis_logits_flat" in text:
        return 2
    return 1


def filter_leaves(
    leaves: Iterable[ObserverLeaf],
    *,
    corpora: Sequence[str],
    kernels: Sequence[str],
    seeds: Sequence[int],
) -> List[ObserverLeaf]:
    corpus_set = {str(item) for item in corpora}
    kernel_set = {str(item) for item in kernels}
    seed_set = {int(item) for item in seeds}
    return [
        leaf
        for leaf in leaves
        if leaf.corpus in corpus_set and leaf.kernel in kernel_set and int(leaf.seed) in seed_set
    ]


def build_replay_command(
    leaf: ObserverLeaf,
    *,
    output_dir: Path,
    basis: str,
    action_branch: str = "baseline_raw_action",
    core_action_mode: str = "baseline",
    observer_state_mode: str = "artifact",
    observer_shuffle_seed: int = 42,
    virtual_interpolation_steps: int = 0,
    hysteresis_weight: float = 0.5,
    observer_transport_weight: float = 1.0,
    shear_weight: float = 1.0,
    stress_weight: float = 2.0,
    void_weight: float = 1.25,
    curvature_weight: float = 1.0,
    max_paths: int = 9,
    k_neighbors: int = 10,
    target_count_per_anchor: int = 1,
) -> List[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_track4_action_graph.py"),
        "--observer-path",
        str(leaf.path),
        "--output-dir",
        str(output_dir),
        "--action-branch",
        str(action_branch),
        "--core-action-mode",
        str(core_action_mode),
        "--basis",
        str(basis),
        "--observer-state-mode",
        str(observer_state_mode),
        "--observer-shuffle-seed",
        str(int(observer_shuffle_seed)),
        "--k-neighbors",
        str(int(k_neighbors)),
        "--stress-weight",
        str(float(stress_weight)),
        "--shear-weight",
        str(float(shear_weight)),
        "--observer-transport-weight",
        str(float(observer_transport_weight)),
        "--hysteresis-weight",
        str(float(hysteresis_weight)),
        "--void-weight",
        str(float(void_weight)),
        "--curvature-weight",
        str(float(curvature_weight)),
        "--max-paths",
        str(int(max_paths)),
        "--target-count-per-anchor",
        str(int(target_count_per_anchor)),
    ]
    if str(core_action_mode).strip().lower() in {"null_calibrated", "null_calibrated_hysteresis"}:
        command.append("--null-calibrated-hysteresis")
    if str(core_action_mode).strip().lower() in {"richer_state", "richer_walker_state"}:
        command.append("--richer-walker-state")
    if int(virtual_interpolation_steps) > 0:
        command.extend(["--use-virtual-transitions", "--virtual-interpolation-steps", str(int(virtual_interpolation_steps))])
    if str(core_action_mode).strip().lower() in {"virtual_transition", "virtual_transition_states"}:
        command.append("--virtual-transition-states")
    return command


def _run_command(command: Sequence[str], *, dry_run: bool) -> int:
    if dry_run:
        print("DRY_RUN " + " ".join(str(part) for part in command))
        return 0
    completed = subprocess.run(list(command), cwd=ROOT)
    return int(completed.returncode)


def _run_replays(
    leaves: Sequence[ObserverLeaf],
    *,
    output_root: Path,
    bases: Sequence[str],
    action_branches: Sequence[str],
    dry_run: bool,
    skip_existing: bool,
    max_paths: int,
    run_suffix: str,
) -> List[Path]:
    summaries: List[Path] = []
    real_leaves = [leaf for leaf in leaves if leaf.corpus == "real"]
    for branch_name in action_branches:
        spec = ACTION_BRANCH_SPECS[str(branch_name)]
        branch_bases = [spec.force_basis] if spec.force_basis else list(bases)
        for basis in branch_bases:
            k_neighbors = max(2, 10 + int(spec.k_neighbors_delta))
            branch_max_paths = max(1, int(round(float(max_paths) * float(spec.max_paths_multiplier))))
            for leaf in leaves:
                out_dir = (
                    output_root
                    / f"{branch_name}_{leaf.corpus}_{leaf.kernel}_seed{leaf.seed}_{basis}_full_{run_suffix}"
                )
                summary_path = out_dir / "track4_action_summary.json"
                if skip_existing and summary_path.exists():
                    print(f"SKIP_EXISTING {summary_path}")
                    summaries.append(summary_path)
                    continue
                command = build_replay_command(
                    leaf,
                    output_dir=out_dir,
                    basis=str(basis),
                    action_branch=branch_name,
                    core_action_mode=spec.core_action_mode,
                    observer_state_mode=spec.observer_state_mode,
                    observer_shuffle_seed=spec.observer_shuffle_seed,
                    virtual_interpolation_steps=spec.virtual_interpolation_steps,
                    hysteresis_weight=spec.hysteresis_weight,
                    observer_transport_weight=spec.observer_transport_weight,
                    shear_weight=spec.shear_weight,
                    stress_weight=spec.stress_weight,
                    void_weight=spec.void_weight,
                    curvature_weight=spec.curvature_weight,
                    max_paths=branch_max_paths,
                    k_neighbors=k_neighbors,
                    target_count_per_anchor=spec.target_count_per_anchor,
                )
                if _run_command(command, dry_run=dry_run) != 0:
                    raise RuntimeError(f"replay failed for {leaf.path} ({branch_name})")
                summaries.append(summary_path)
            for real_leaf in real_leaves:
                for ablation, mode, hysteresis_weight in (
                    ("observer_disabled", "disabled", spec.hysteresis_weight),
                    ("observer_shuffled", "shuffled", spec.hysteresis_weight),
                    ("zero_hysteresis", "artifact", 0.0),
                ):
                    out_dir = (
                        output_root
                        / f"{branch_name}_real_{real_leaf.kernel}_seed{real_leaf.seed}_{basis}_{ablation}_{run_suffix}"
                    )
                    summary_path = out_dir / "track4_action_summary.json"
                    if skip_existing and summary_path.exists():
                        print(f"SKIP_EXISTING {summary_path}")
                        summaries.append(summary_path)
                        continue
                    shuffle_seed = (
                        int(spec.observer_shuffle_seed)
                        if branch_name == "null_calibrated_hysteresis"
                        else 9000 + int(real_leaf.seed)
                    )
                    command = build_replay_command(
                        real_leaf,
                        output_dir=out_dir,
                        basis=str(basis),
                        action_branch=branch_name,
                        core_action_mode=spec.core_action_mode,
                        observer_state_mode=mode,
                        observer_shuffle_seed=shuffle_seed,
                        virtual_interpolation_steps=spec.virtual_interpolation_steps,
                        hysteresis_weight=hysteresis_weight,
                        observer_transport_weight=spec.observer_transport_weight,
                        shear_weight=spec.shear_weight,
                        stress_weight=spec.stress_weight,
                        void_weight=spec.void_weight,
                        curvature_weight=spec.curvature_weight,
                        max_paths=branch_max_paths,
                        k_neighbors=k_neighbors,
                        target_count_per_anchor=spec.target_count_per_anchor,
                    )
                    if _run_command(command, dry_run=dry_run) != 0:
                        raise RuntimeError(f"replay failed for {real_leaf.path} ({branch_name}/{ablation})")
                    summaries.append(summary_path)
    return summaries


def _write_summary(
    paths: Sequence[Path],
    output_root: Path,
    required_kernels: Sequence[str],
    required_seeds: Sequence[int],
    required_bases: Sequence[str],
) -> Path:
    summary = summarize_observer_state_ablation(
        paths,
        required_kernels=tuple(required_kernels),
        required_seeds=tuple(required_seeds),
        required_bases=tuple(required_bases),
    )
    out_dir = output_root / "summary_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track4_observer_state_ablation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _write_branch_comparison(
    paths: Sequence[Path],
    output_root: Path,
    *,
    action_branches: Sequence[str],
    required_kernels: Sequence[str],
    required_seeds: Sequence[int],
    required_bases: Sequence[str],
) -> Path:
    branch_summaries: Dict[str, Dict[str, object]] = {}
    for branch in action_branches:
        branch_paths = [path for path in paths if path.exists() and f"{branch}_" in path.parent.name]
        branch_required_bases = (
            [ACTION_BRANCH_SPECS[branch].force_basis]
            if ACTION_BRANCH_SPECS[branch].force_basis
            else list(required_bases)
        )
        branch_summary = summarize_observer_state_ablation(
            branch_paths,
            required_kernels=tuple(required_kernels),
            required_seeds=tuple(required_seeds),
            required_bases=tuple(str(basis) for basis in branch_required_bases if basis),
        )
        branch_summaries[branch] = {
            "branch_spec": ACTION_BRANCH_SPECS[branch].__dict__,
            "summary": branch_summary,
        }
        branch_out = output_root / f"summary_{branch}"
        branch_out.mkdir(parents=True, exist_ok=True)
        (branch_out / "track4_observer_state_ablation_summary.json").write_text(
            json.dumps(branch_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    ranked = sorted(
        (
            {
                "action_branch": branch,
                "safe_for_thesis_claim": bool(payload["summary"].get("safe_for_thesis_claim")),
                "point_estimate": (
                    payload["summary"].get("claim_evaluation", {}).get("point_estimate")
                    if isinstance(payload.get("summary"), dict)
                    else None
                ),
                "failure_reasons": (
                    payload["summary"].get("claim_evaluation", {}).get("failure_reasons")
                    if isinstance(payload.get("summary"), dict)
                    else []
                ),
                "branch_type": ACTION_BRANCH_SPECS[branch].branch_type,
            }
            for branch, payload in branch_summaries.items()
        ),
        key=lambda row: (
            not bool(row["safe_for_thesis_claim"]),
            -float(row["point_estimate"] or 0.0),
            str(row["action_branch"]),
        ),
    )
    payload = {
        "schema_version": "1.0",
        "summary_type": "track4_observer_state_action_branch_comparison",
        "action_branches": list(action_branches),
        "required_kernels": list(required_kernels),
        "required_seeds": list(required_seeds),
        "required_bases": list(required_bases),
        "ranked_branches": ranked,
        "branch_summaries": branch_summaries,
    }
    out_path = output_root / "track4_action_branch_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--kernel", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--basis", action="append", default=None)
    parser.add_argument(
        "--action-branch",
        action="append",
        default=None,
        choices=DEFAULT_ACTION_BRANCHES,
        help="Track 4 engineering branch to replay. Repeatable; overrides --branch-preset when supplied.",
    )
    parser.add_argument(
        "--branch-preset",
        choices=tuple(BRANCH_PRESETS),
        default="all",
        help="Named action branch set. Use repair3 for baseline/null_calibrated/richer.",
    )
    parser.add_argument("--max-paths", type=int, default=9)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing track4_action_summary.json files instead of rerunning those cells.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Suffix appended to replay output directories. Defaults to current UTC date (YYYYMMDD).",
    )
    return parser.parse_args()


def _unique_or_default(values: Sequence[str] | None, defaults: Sequence[str]) -> List[str]:
    raw = list(values) if values else list(defaults)
    out: List[str] = []
    for value in raw:
        value_s = str(value)
        if value_s not in out:
            out.append(value_s)
    return out


def _unique_int_or_default(values: Sequence[int] | None, defaults: Sequence[int]) -> List[int]:
    raw = list(values) if values else list(defaults)
    out: List[int] = []
    for value in raw:
        value_i = int(value)
        if value_i not in out:
            out.append(value_i)
    return out


def _resolve_action_branches(action_branch: Sequence[str] | None, branch_preset: str) -> List[str]:
    defaults = BRANCH_PRESETS[str(branch_preset)]
    return _unique_or_default(action_branch, defaults)


def main() -> int:
    args = parse_args()
    corpora = _unique_or_default(args.corpus, ("real", "control_random", "control_shuffled"))
    kernels = _unique_or_default(args.kernel, ("rbf", "matern", "imq"))
    seeds = _unique_int_or_default(args.seed, (42, 420, 4200))
    bases = _unique_or_default(args.basis, ("track2", "integrated"))
    action_branches = _resolve_action_branches(args.action_branch, args.branch_preset)
    run_suffix = str(args.run_suffix or datetime.now(timezone.utc).strftime("%Y%m%d")).strip()
    leaves = filter_leaves(
        discover_observer_leaves(args.input_root),
        corpora=corpora,
        kernels=kernels,
        seeds=seeds,
    )
    if not leaves:
        raise FileNotFoundError(f"no observer leaves matched {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    inventory_filename = (
        "observer_state_matrix_inventory.dry_run.json"
        if bool(args.dry_run)
        else "observer_state_matrix_inventory.json"
    )
    inventory_path = args.output_root / inventory_filename
    inventory_path.write_text(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "leaf_count": len(leaves),
                "leaves": [leaf.__dict__ | {"path": str(leaf.path)} for leaf in leaves],
                "bases": bases,
                "branch_preset": str(args.branch_preset),
                "action_branches": action_branches,
                "run_suffix": run_suffix,
                "action_branch_specs": {name: ACTION_BRANCH_SPECS[name].__dict__ for name in action_branches},
                "action_mode_knobs": {
                    name: {
                        "core_action_mode": ACTION_BRANCH_SPECS[name].core_action_mode,
                        "observer_state_mode": ACTION_BRANCH_SPECS[name].observer_state_mode,
                        "observer_shuffle_seed": ACTION_BRANCH_SPECS[name].observer_shuffle_seed,
                        "virtual_interpolation_steps": ACTION_BRANCH_SPECS[name].virtual_interpolation_steps,
                        "k_neighbors_delta": ACTION_BRANCH_SPECS[name].k_neighbors_delta,
                        "observer_transport_weight": ACTION_BRANCH_SPECS[name].observer_transport_weight,
                        "hysteresis_weight": ACTION_BRANCH_SPECS[name].hysteresis_weight,
                        "curvature_weight": ACTION_BRANCH_SPECS[name].curvature_weight,
                    }
                    for name in action_branches
                },
                "dry_run": bool(args.dry_run),
                "skip_existing": bool(args.skip_existing),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summaries = _run_replays(
        leaves,
        output_root=args.output_root,
        bases=bases,
        action_branches=action_branches,
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
        max_paths=int(args.max_paths),
        run_suffix=run_suffix,
    )
    summary_path = None
    if not args.dry_run:
        existing_summaries = [path for path in summaries if path.exists()]
        summary_path = _write_summary(
            existing_summaries,
            args.output_root,
            required_kernels=kernels,
            required_seeds=seeds,
            required_bases=bases,
        )
        branch_comparison_path = _write_branch_comparison(
            existing_summaries,
            args.output_root,
            action_branches=action_branches,
            required_kernels=kernels,
            required_seeds=seeds,
            required_bases=bases,
        )
    print("status=OK")
    print(f"leaf_count={len(leaves)}")
    print(f"inventory={inventory_path}")
    if summary_path is not None:
        print(f"summary={summary_path}")
        print(f"branch_comparison={branch_comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
