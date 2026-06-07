#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _latest(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    matches = [path for path in root.glob(pattern) if path.exists()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _run(cmd: Sequence[str], *, cwd: Path = ROOT, allow_fail: bool = True) -> Dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = {
        "cmd": [str(part) for part in cmd],
        "returncode": int(proc.returncode),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(json.dumps(payload, indent=2))
    return payload


def _status_counts(open_matrix: Path) -> Dict[str, int]:
    payload = _load_json(open_matrix)
    return {str(k): int(v) for k, v in (payload.get("status_counts") or {}).items()}


def run_series(
    *,
    output_dir: Path,
    synthetic_root: Path,
    kernels: Sequence[str],
    seeds: Sequence[int],
    base_claim_matrix: Optional[Path],
    skip_recenter: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: List[Dict[str, Any]] = []

    prompt_dir = output_dir / "prompt_invariance"
    commands.append(
        _run(
            [
                sys.executable,
                "scripts/run_prompt_invariance_probe.py",
                "--prompt-bank",
                "canonical=probe_hypotheses.json",
                "--prompt-bank",
                "paraphrase_a=config/prompt_banks/probe_hypotheses_paraphrase_a.json",
                "--prompt-bank",
                "paraphrase_b=config/prompt_banks/probe_hypotheses_paraphrase_b.json",
                "--output-dir",
                str(prompt_dir),
            ]
        )
    )
    prompt_summary = prompt_dir / "prompt_invariance_summary.json"

    track3_artifacts: Dict[str, Path] = {}
    for variant in ("raw", "action_smoothed", "action_calibrated"):
        out = output_dir / "track3_density" / variant
        commands.append(
            _run(
                [
                    sys.executable,
                    "scripts/run_track3_density_validation.py",
                    "--synthetic-root",
                    str(synthetic_root),
                    "--output-dir",
                    str(out),
                    "--kernels",
                    *[str(k) for k in kernels],
                    "--seeds",
                    *[str(seed) for seed in seeds],
                    "--variant",
                    variant,
                ]
            )
        )
        track3_artifacts[variant] = out / "track3_density_validation.json"
    track3_payloads = {name: _load_json(path) for name, path in track3_artifacts.items()}
    best_track3_name, _ = max(
        track3_payloads.items(),
        key=lambda item: (
            bool(item[1].get("track3_density_semantic_pass")),
            float(item[1].get("pass_rate") or 0.0),
        ),
    )
    best_track3 = track3_artifacts[best_track3_name]

    property_dir = output_dir / "property_theft_scaled"
    commands.append(
        _run(
            [
                sys.executable,
                "scripts/run_property_theft_scale_probe.py",
                "--output-dir",
                str(property_dir),
                "--max-pairs",
                "10",
            ]
        )
    )
    property_summary = property_dir / "observer_slice_transport_summary.json"

    viz_dir = output_dir / "visualization_readability"
    commands.append(
        _run(
            [
                sys.executable,
                "scripts/run_visualization_readability_validation.py",
                "--transport-summary",
                str(property_summary),
                "--output-dir",
                str(viz_dir),
                "--min-records",
                "40",
            ]
        )
    )
    visualization_summary = viz_dir / "visualization_readability_validation.json"

    terrain_dir = output_dir / "terrain_action_ablation"
    commands.append(
        _run(
            [
                sys.executable,
                "scripts/run_action_terrain_ablation.py",
                "--synthetic-root",
                str(synthetic_root),
                "--output-dir",
                str(terrain_dir),
                "--kernels",
                *[str(k) for k in kernels],
                "--seeds",
                *[str(seed) for seed in seeds],
                "--base-claim-matrix",
                str(base_claim_matrix) if base_claim_matrix else "",
            ]
            if base_claim_matrix
            else [
                sys.executable,
                "scripts/run_action_terrain_ablation.py",
                "--synthetic-root",
                str(synthetic_root),
                "--output-dir",
                str(terrain_dir),
                "--kernels",
                *[str(k) for k in kernels],
                "--seeds",
                *[str(seed) for seed in seeds],
            ]
        )
    )
    terrain_claim_matrix = terrain_dir / "claim_matrix.json"

    recenter_summary = _latest(ROOT / "outputs" / "observer_recenter_robustness_suite", "**/observer_recenter_robustness_suite.json")
    if not skip_recenter:
        recenter_dir = output_dir / "observer_recenter_robustness"
        commands.append(
            _run(
                [
                    sys.executable,
                    "scripts/run_observer_recenter_robustness_suite.py",
                    "--synthetic-root",
                    str(synthetic_root),
                    "--output-dir",
                    str(recenter_dir),
                    "--kernels",
                    *[str(k) for k in kernels],
                    "--seeds",
                    *[str(seed) for seed in seeds],
                    "--baselines",
                    "translation_only",
                    "artifact_view",
                    "raw_track2_pca",
                    "cls_mean_pca",
                    "local_track_recompute",
                    "local_track_recompute:graph_whitened_hybrid",
                    "local_track_recompute:source_proxy_metric",
                    "--label-column",
                    "perspective_tag",
                    "--label-mode",
                    "ideological",
                ],
                allow_fail=True,
            )
        )
        recenter_summary = recenter_dir / "observer_recenter_robustness_suite.json"

    independent_summary: Optional[Path] = None
    first_cell = Path(synthetic_root) / f"{kernels[0]}_seed{int(seeds[0])}"
    if first_cell.exists():
        independent_dir = output_dir / "independent_label_proxy"
        commands.append(
            _run(
                [
                    sys.executable,
                    "scripts/run_independent_label_validation.py",
                    "--run-dir",
                    str(first_cell),
                    "--output-dir",
                    str(independent_dir),
                    "--permutations",
                    "80",
                    "--max-pvalue",
                    "0.15",
                    "--allow-proxy-as-independent",
                ],
                allow_fail=True,
            )
        )
        independent_summary = independent_dir / "independent_label_validation_summary.json"

    open_dir = output_dir / "open_hypothesis_suite"
    open_cmd = [
        sys.executable,
        "scripts/run_open_hypothesis_suite.py",
        "--repo-root",
        str(ROOT),
        "--output-dir",
        str(open_dir),
        "--claim-matrix",
        str(terrain_claim_matrix),
        "--recentering-summary",
        str(recenter_summary),
        "--property-theft-transport-summary",
        str(property_summary),
        "--prompt-invariance-summary",
        str(prompt_summary),
        "--track3-density-validation",
        str(best_track3),
        "--visualization-validation",
        str(visualization_summary),
    ]
    if independent_summary and independent_summary.exists():
        open_cmd.extend(["--independent-label-summary", str(independent_summary)])
    commands.append(_run(open_cmd, allow_fail=True))

    final_matrix = open_dir / "open_hypothesis_matrix.json"
    summary = {
        "schema_version": "1.0",
        "summary_type": "open_hypothesis_ablation_series",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_root": str(synthetic_root),
        "kernels": list(kernels),
        "seeds": [int(seed) for seed in seeds],
        "commands": commands,
        "artifacts": {
            "prompt_invariance_summary": str(prompt_summary),
            "track3_density_variants": {name: str(path) for name, path in track3_artifacts.items()},
            "best_track3_density_variant": best_track3_name,
            "best_track3_density_validation": str(best_track3),
            "property_theft_transport_summary": str(property_summary),
            "visualization_validation": str(visualization_summary),
            "terrain_ablation_summary": str(terrain_dir / "action_terrain_ablation_summary.json"),
            "terrain_claim_matrix": str(terrain_claim_matrix),
            "recentering_summary": str(recenter_summary) if recenter_summary else None,
            "independent_label_summary": str(independent_summary) if independent_summary else None,
            "open_hypothesis_matrix": str(final_matrix),
        },
        "status_counts": _status_counts(final_matrix),
    }
    out = output_dir / "open_hypothesis_ablation_series_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "thesis_validation" / "open_hypothesis_ablation_series_latest",
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=ROOT / "outputs" / "experiments" / "runs" / "experiments_20260506_192553" / "synthetic",
    )
    parser.add_argument("--kernels", nargs="+", default=["rbf", "matern", "imq"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 420, 4200])
    parser.add_argument(
        "--base-claim-matrix",
        type=Path,
        default=_latest(ROOT / "outputs" / "thesis_validation", "**/claim_matrix.json"),
    )
    parser.add_argument("--skip-recenter", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_series(
        output_dir=args.output_dir,
        synthetic_root=args.synthetic_root,
        kernels=[str(k) for k in args.kernels],
        seeds=[int(seed) for seed in args.seeds],
        base_claim_matrix=args.base_claim_matrix,
        skip_recenter=bool(args.skip_recenter),
    )
    print(
        json.dumps(
            {
                "status_counts": payload["status_counts"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
