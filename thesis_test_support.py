from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_methods(path: Path, *, limit: int = 500) -> None:
    path.write_text(
        "\n".join(
            [
                "# Methods (Canonical Thesis Protocol)",
                "",
                "- Runner: `run_full_experiment_suite.py`",
                "- Mode: `enhanced`",
                "- Kernels: `rbf` `laplacian` `rq` `imq`",
                "- Seeds: `42` `420` `4200`",
                "- Channels: `logits` `cls`",
                "- Corpora: `real` `control_constant` `control_shuffled` `control_random`",
                f"- Article limit per corpus: `{limit}`",
                "",
                "```powershell",
                f"python run_full_experiment_suite.py --mode enhanced --seeds 42 420 4200 --kernels rbf laplacian rq imq --channels logits cls --corpora real control_constant control_shuffled control_random --limit {limit}",
                "```",
                "",
                "```powershell",
                "python run_full_experiment_suite.py --synthetic --mode enhanced --seeds 42 420 4200 --kernels rbf laplacian rq imq --channels cls --limit 60",
                "```",
            ]
        ),
        encoding="utf-8",
    )


def write_results(path: Path, run_ids: Iterable[str]) -> None:
    lines = [
        "# Results",
        "",
        "## Canonical Run Registry",
        "",
    ]
    for run_id in run_ids:
        lines.append(f"- {run_id}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_baseline_meta(path: Path, *, kernel: str, channel: str, track5_mode: Optional[str] = None) -> None:
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "cache_version": "1.0",
        "dataset_hash": "dataset-hash",
        "code_hash_or_commit": "code-hash",
        "weights_hash": "weights-hash",
        "kernel_params": {"kernel": kernel, "channel": channel},
        "rks_dim": 2048,
        "crn_seed": 12345,
        "alpha": 1.0,
        "timestamp_utc": "2026-05-04T00:00:00+00:00",
        "verification_status": "VERIFIED",
        "provenance_source": "observer_payload",
    }
    if track5_mode:
        payload["track5_assembly_mode"] = track5_mode
    write_json(path, payload)


def write_verification_report(path: Path, *, global_pass: bool = True, seed_stability: float = 0.12) -> None:
    payload = {
        "run_id": "run_1",
        "timestamp": "2026-05-04T00:00:00+00:00",
        "layers": [
            {
                "layer_id": "rbf/cls",
                "layer_name": "cls",
                "status": "VERIFIED" if global_pass else "UNVERIFIED",
                "checks": [
                    {"name": "crn_locked", "pass": global_pass},
                    {"name": "seed_stability", "pass": global_pass, "value": seed_stability},
                ],
                "fail_reasons": [],
            }
        ],
        "global_pass": global_pass,
        "verification_status": "VERIFIED" if global_pass else "UNVERIFIED",
        "dataset_hash": "dataset_fixture_hash",
        "code_hash_or_commit": "code_fixture_hash",
        "weights_hash": "weights_fixture_hash",
        "kernel_params": {"kernel": "rbf", "channel": "cls"},
        "crn_seed": 12345,
    }
    write_json(path, payload)


def write_control_metrics(
    path: Path,
    *,
    procrustes_ratio: float = 1.6,
    separates_count: int = 3,
    distance_corr_ratio: float = 0.75,
    simple_variance_real: float = 0.80,
    simple_variance_shuffled: float = 1.00,
    simple_variance_random: float = 1.05,
    placeholder: bool = False,
    metric_basis: Optional[str] = None,
) -> None:
    stochastic_mean = (simple_variance_shuffled + simple_variance_random) / 2.0
    simple_variance_ratio = (
        simple_variance_real / stochastic_mean if abs(stochastic_mean) > 1e-12 else None
    )
    payload = {
        "status": "OK",
        "synthetic_placeholder": placeholder,
        "metrics": {
            "procrustes_ratio": procrustes_ratio,
            "procrustes_min_control_ratio": procrustes_ratio,
            "distance_corr_ratio": distance_corr_ratio,
            "simple_variance_ratio": simple_variance_ratio,
            "simple_variance_stochastic_ratio": simple_variance_ratio,
            "simple_variance_real": simple_variance_real,
            "simple_variance_control_avg": (0.0 + simple_variance_shuffled + simple_variance_random) / 3.0,
            "simple_variance_stochastic_control_mean": stochastic_mean,
            "separates_count": separates_count,
            "consensus_pct": 35.0,
            "residual_pct": 65.0,
        },
        "controls": {
            "Real": {
                "procrustes": {"std": 0.05},
                "distance_corr": {"std": 0.08},
                "simple_variance": {"mean": simple_variance_real, "std": 0.03},
                "seeds": [42, 420, 4200],
            },
            "Constant": {
                "simple_variance": {"mean": 0.0, "std": 0.0},
                "seeds": [42, 420, 4200],
            },
            "Shuffled": {
                "simple_variance": {"mean": simple_variance_shuffled, "std": 0.04},
                "seeds": [42, 420, 4200],
            },
            "Random": {
                "simple_variance": {"mean": simple_variance_random, "std": 0.04},
                "seeds": [42, 420, 4200],
            },
        },
        "procrustes_real_vs_controls": {
            "available": True,
            "min_ratio_real_over_control": procrustes_ratio,
            "rows": [
                {
                    "control": "Constant",
                    "real_mean": 1.0,
                    "control_mean": 1.0 / (procrustes_ratio + 0.10),
                    "ratio_real_over_control": procrustes_ratio + 0.10,
                },
                {
                    "control": "Shuffled",
                    "real_mean": 1.0,
                    "control_mean": 1.0 / procrustes_ratio,
                    "ratio_real_over_control": procrustes_ratio,
                },
                {
                    "control": "Random",
                    "real_mean": 1.0,
                    "control_mean": 1.0 / (procrustes_ratio + 0.05),
                    "ratio_real_over_control": procrustes_ratio + 0.05,
                },
            ],
        },
    }
    if metric_basis:
        payload["metric_basis"] = metric_basis
    write_json(path, payload)


def write_relativity_deltas(
    path: Path,
    *,
    mean_coord_delta: float = 0.33,
    max_coord_delta: float = 0.41,
    rotation_deg: float = 21.0,
    path_flip_count: int = 6,
    placeholder: bool = False,
) -> None:
    payload = {
        "status": "OK",
        "synthetic_placeholder": placeholder,
        "observer_count": 2,
        "summary": {
            "mean_coord_delta": mean_coord_delta,
            "max_coord_delta": max_coord_delta,
        },
        "observers": [
            {
                "observer_id": 0,
                "null_observer_equivalence": {
                    "equivalent": False,
                    "path_flip_count": path_flip_count,
                    "axis_rotation_deg": rotation_deg,
                },
                "axis_delta": {"rotation_deg": rotation_deg},
                "translation_only_comparison": {"d_path_flip_count": path_flip_count},
            }
        ],
    }
    write_json(path, payload)


def write_ablation_summary(
    path: Path,
    *,
    status: str = "OK",
    placeholder: bool = False,
    stage_1: Optional[float] = 0.2,
    stage_2: Optional[float] = 0.5,
    stage_3: Optional[float] = 0.7,
    delta: Optional[float] = 0.3,
) -> None:
    payload = {
        "status": status,
        "synthetic_placeholder": placeholder,
        "metrics": {
            "stage_1_alignment_score": stage_1,
            "stage_2_alignment_score": stage_2,
            "stage_3_survival_rate": stage_3,
            "delta_alignment_score": delta,
        },
        "reason": "ablation flow not executed" if placeholder else None,
    }
    write_json(path, payload)


def write_view_state(path: Path, *, survival_rate: float = 0.72, mean_work: float = 71.0) -> None:
    payload = {
        "metrics": {
            "walker_survival_rate": survival_rate,
            "walker_mean_action": mean_work,
        }
    }
    write_json(path, payload)


def write_walker_states(path: Path, *, closed_count: int = 2, total: int = 3, work: float = 70.0) -> None:
    rows = []
    for idx in range(total):
        rows.append(
            {
                "closed_loop": idx < closed_count,
                "work_integral": work + idx,
                "label": "closed_loop" if idx < closed_count else "shattered",
                "steps": 150,
            }
        )
    write_json(path, rows)


def write_monolith_data(path: Path, *, collapsed: bool = False) -> None:
    rows = [
        {"index": 0, "bt_uid": "a0", "title": "Bridge", "zone": "Bridge", "density": 0.9, "stress": 0.1, "w_actual": 40.0, "x": 0.0, "y": 0.0},
        {"index": 1, "bt_uid": "a1", "title": "Swamp", "zone": "Swamp", "density": 0.8, "stress": 0.8, "w_actual": 88.0, "x": 1.0 if not collapsed else 0.0, "y": 0.4},
        {"index": 2, "bt_uid": "a2", "title": "Tightrope", "zone": "Tightrope", "density": 0.1, "stress": 0.2, "w_actual": 62.0, "x": 2.0 if not collapsed else 0.0, "y": 0.8},
        {"index": 3, "bt_uid": "a3", "title": "Void", "zone": "Void", "density": 0.0, "stress": 0.95, "w_actual": 120.0, "x": 3.0 if not collapsed else 0.0, "y": 1.2},
    ]
    write_csv(
        path,
        rows,
        fieldnames=["index", "bt_uid", "title", "zone", "density", "stress", "w_actual", "x", "y"],
    )


def write_hidden_groups(path: Path, labels: Dict[int, str]) -> None:
    write_csv(
        path,
        [
            {"article_id": int(article_id), "group_topic": str(label)}
            for article_id, label in sorted(labels.items())
        ],
        fieldnames=["article_id", "group_topic"],
    )


def write_synthetic_terrain_incremental_fixture(run_dir: Path) -> None:
    rows = []
    labels: Dict[int, str] = {}
    specs = [
        ("ClusterA", "Bridge", 10.0, 0.90, 0.10),
        ("ClusterA", "Bridge", 11.0, 0.88, 0.12),
        ("ClusterA", "Bridge", 12.0, 0.86, 0.14),
        ("ClusterA", "Void", 40.0, 0.10, 0.92),
        ("ClusterA", "Void", 41.0, 0.12, 0.90),
        ("ClusterA", "Void", 42.0, 0.14, 0.88),
        ("ClusterB", "Swamp", 20.0, 0.82, 0.80),
        ("ClusterB", "Swamp", 21.0, 0.80, 0.78),
        ("ClusterB", "Swamp", 22.0, 0.78, 0.76),
        ("ClusterB", "Tightrope", 50.0, 0.18, 0.20),
        ("ClusterB", "Tightrope", 51.0, 0.20, 0.18),
        ("ClusterB", "Tightrope", 52.0, 0.22, 0.16),
    ]
    for idx, (label, zone, work, density, stress) in enumerate(specs):
        labels[idx] = label
        rows.append(
            {
                "index": idx,
                "bt_uid": f"s{idx}",
                "title": f"{label} {zone} {idx}",
                "zone": zone,
                "density": density,
                "stress": stress,
                "w_actual": work,
                "x": float(idx),
                "y": float(idx % 3),
            }
        )
    write_csv(
        run_dir / "MONOLITH_DATA.csv",
        rows,
        fieldnames=["index", "bt_uid", "title", "zone", "density", "stress", "w_actual", "x", "y"],
    )
    write_hidden_groups(run_dir / "labels" / "hidden_groups.csv", labels)


def write_track4_observer_state_action_summary(path: Path) -> None:
    rows = []
    for kernel in ("rbf", "matern", "imq"):
        rows.extend(
            [
                {
                    "run_name": f"real_{kernel}_full_20260521",
                    "summary_path": str(path.parent / f"real_{kernel}" / "track4_action_summary.json"),
                    "corpus": "real",
                    "kernel": kernel,
                    "ablation": "full",
                    "observer_state_mode": "enabled",
                    "mean_action": 12.0,
                    "mean_observer_transport_penalty": 3.0,
                    "mean_hysteresis_penalty": 1.2,
                    "observer_simplex_contract_supported": True,
                },
                {
                    "run_name": f"control_random_{kernel}_full_20260521",
                    "summary_path": str(path.parent / f"control_random_{kernel}" / "track4_action_summary.json"),
                    "corpus": "control_random",
                    "kernel": kernel,
                    "ablation": "full",
                    "observer_state_mode": "enabled",
                    "mean_action": 6.0,
                    "mean_observer_transport_penalty": 1.0,
                    "mean_hysteresis_penalty": 0.4,
                    "observer_simplex_contract_supported": True,
                },
                {
                    "run_name": f"control_shuffled_{kernel}_shuffled_20260521",
                    "summary_path": str(path.parent / f"control_shuffled_{kernel}" / "track4_action_summary.json"),
                    "corpus": "control_shuffled",
                    "kernel": kernel,
                    "ablation": "shuffled",
                    "observer_state_mode": "shuffled",
                    "mean_action": 5.0,
                    "mean_observer_transport_penalty": 0.8,
                    "mean_hysteresis_penalty": 0.3,
                    "observer_simplex_contract_supported": True,
                },
                {
                    "run_name": f"real_{kernel}_observer_disabled_20260521",
                    "summary_path": str(path.parent / f"real_{kernel}_observer_disabled" / "track4_action_summary.json"),
                    "corpus": "real",
                    "kernel": kernel,
                    "ablation": "observer_disabled",
                    "observer_state_mode": "disabled",
                    "mean_action": 8.0,
                    "mean_observer_transport_penalty": 0.0,
                    "mean_hysteresis_penalty": 0.0,
                    "observer_simplex_contract_supported": True,
                },
                {
                    "run_name": f"real_{kernel}_zero_hysteresis_20260521",
                    "summary_path": str(path.parent / f"real_{kernel}_zero_hysteresis" / "track4_action_summary.json"),
                    "corpus": "real",
                    "kernel": kernel,
                    "ablation": "zero_hysteresis",
                    "observer_state_mode": "zero_hysteresis",
                    "mean_action": 9.0,
                    "mean_observer_transport_penalty": 2.5,
                    "mean_hysteresis_penalty": 0.0,
                    "observer_simplex_contract_supported": True,
                },
            ]
        )
    payload = {
        "schema_version": "1.0",
        "summary_type": "track4_observer_state_ablation",
        "claim_scope": "exploratory_track4",
        "safe_for_thesis_claim": True,
        "pass": True,
        "thesis_safe": True,
        "row_count": len(rows),
        "rows": rows,
        "required_kernels": ["rbf", "matern", "imq"],
        "kernels_present": ["rbf", "matern", "imq"],
        "missing_required_kernels": [],
        "required_kernels_present": True,
        "observer_simplex_row_count": 6,
        "observer_simplex_contract_supported": True,
        "comparisons": {
            "real_vs_control": {
                "mean_action": {"real": 12.0, "control": 6.0, "gap": 6.0, "ratio": 2.0},
                "mean_observer_transport_penalty": {"real": 3.0, "control": 1.0, "gap": 2.0, "ratio": 3.0},
                "mean_hysteresis_penalty": {"real": 1.2, "control": 0.4, "gap": 0.8, "ratio": 3.0},
            }
        },
        "baseline_comparisons": {
            "observer_disabled": {
                "mean_observer_transport_penalty": {
                    "full_real": 3.0,
                    "baseline": 0.0,
                    "gap_vs_full_real": -3.0,
                    "ratio_full_real_over_baseline": None,
                }
            },
            "zero_hysteresis": {
                "mean_hysteresis_penalty": {
                    "full_real": 1.2,
                    "baseline": 0.0,
                    "gap_vs_full_real": -1.2,
                    "ratio_full_real_over_baseline": None,
                }
            },
            "shuffled": {
                "mean_hysteresis_penalty": {
                    "full_real": 1.2,
                    "baseline": 0.3,
                    "gap_vs_full_real": -0.9,
                    "ratio_full_real_over_baseline": 4.0,
                }
            },
        },
        "claim_evaluation": {
            "claim_id": "track4_observer_state_action_separation",
            "claim_scope": "exploratory_track4",
            "pass": True,
            "thesis_safe": True,
            "safe_for_thesis_claim": True,
            "failure_reasons": [],
            "required_kernels_present": True,
            "observer_simplex_contract_supported": True,
            "real_control_separation_pass": True,
            "shuffled_hysteresis_baseline_pass": True,
            "observer_disabled_transport_baseline_pass": True,
            "point_estimate": 2.0,
            "effect_direction": "real_observer_state_action_gt_controls",
        },
    }
    write_json(path, payload)


def write_suite_manifest(
    path: Path,
    *,
    run_id: str,
    kernels: list[str],
    channels: list[str],
    corpora: list[str],
    seeds: list[int],
    base_run_dir: Path,
    status_overrides: Optional[Dict[tuple[str, str, str], str]] = None,
    limit: int = 500,
) -> None:
    experiments = []
    status_overrides = status_overrides or {}
    for kernel in kernels:
        for channel in channels:
            for corpus in corpora:
                status = status_overrides.get((kernel, channel, corpus), "success")
                experiments.append(
                    {
                        "kernel": kernel,
                        "channel": channel,
                        "corpus": corpus,
                        "seeds": seeds,
                        "status": status,
                        "output_dir": str((base_run_dir / kernel / channel / corpus).relative_to(path.parents[4])),
                    }
                )
    payload = {
        "timestamp": "2026-05-04T00:00:00+00:00",
        "config": {
            "kernels": kernels,
            "channels": channels,
            "corpora": corpora,
            "seeds": seeds,
            "limit": limit,
        },
        "experiments": experiments,
    }
    write_json(path, payload)


def write_synthetic_manifest(
    path: Path,
    *,
    output_dir: Path,
    kernels: list[str],
    seeds: list[int],
    nmi: float = 0.62,
    ari: float = 0.28,
) -> None:
    results = []
    for kernel in kernels:
        for seed in seeds:
            results.append(
                {
                    "run_key": f"{kernel}_seed{seed}",
                    "run_dir": str((output_dir / "synthetic" / f"{kernel}_seed{seed}").relative_to(path.parents[4])),
                    "kernel": kernel,
                    "seed": seed,
                    "nmi": nmi,
                    "ari": ari,
                    "status": "success",
                }
            )
    payload = {
        "experiment_type": "synthetic",
        "output_dir": str(output_dir.relative_to(path.parents[4])),
        "timestamp": "2026-05-04T00:00:00+00:00",
        "synthetic_result": {
            "status": "success",
            "summary": {
                "n_runs": len(results),
                "successful_runs": len(results),
                "nmi_scores": [nmi for _ in results],
                "ari_scores": [ari for _ in results],
                "by_kernel": {kernel: {"nmi": [nmi], "ari": [ari]} for kernel in kernels},
                "by_seed": {str(seed): {"nmi": [nmi], "ari": [ari]} for seed in seeds},
                "mean_nmi": nmi,
                "std_nmi": 0.0,
                "mean_ari": ari,
                "std_ari": 0.0,
            },
            "results": results,
            "output_dir": str((output_dir / "synthetic").relative_to(path.parents[4])),
        },
    }
    write_json(path, payload)
    write_json(output_dir / "synthetic" / "synthetic_summary.json", payload["synthetic_result"])


def build_canonical_fixture(
    root: Path,
    *,
    kernels: Optional[list[str]] = None,
    channels: Optional[list[str]] = None,
    corpora: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    track5_mode: str = "hadamard_strict",
    ablation_modes: Optional[list[str]] = None,
    include_synthetic: bool = True,
    methods_limit: int = 500,
    manifest_limit: int = 500,
) -> Dict[str, Path]:
    kernels = kernels or ["rbf", "laplacian", "rq", "imq"]
    channels = channels or ["logits", "cls"]
    corpora = corpora or ["real", "control_constant", "control_shuffled", "control_random"]
    seeds = seeds or [42, 420, 4200]
    ablation_modes = ablation_modes or ["hadamard_strict", "riemannian_strict"]

    methods_path = root / "METHODS.md"
    results_path = root / "RESULTS.md"
    runs_dir = root / "outputs" / "experiments" / "runs"
    suite_run_dir = runs_dir / "experiments_20260504_010101"
    suite_manifest = suite_run_dir / "experiment_manifest.json"

    write_methods(methods_path, limit=methods_limit)
    ablation_run_ids = ["experiments_20260504_010101", "experiments_20260504_030303", "experiments_20260504_040404"]
    run_ids = ablation_run_ids[:len(ablation_modes)]
    if include_synthetic:
        run_ids.append("experiments_20260504_020202")
    write_results(results_path, run_ids)
    for idx, mode in enumerate(ablation_modes):
        run_dir = suite_run_dir if idx == 0 else runs_dir / ablation_run_ids[idx]
        write_suite_manifest(
            run_dir / "experiment_manifest.json",
            run_id=run_dir.name,
            kernels=kernels,
            channels=channels,
            corpora=corpora,
            seeds=seeds,
            base_run_dir=run_dir,
            limit=manifest_limit,
        )

        for kernel in kernels:
            for channel in channels:
                for corpus in corpora:
                    leaf_dir = run_dir / kernel / channel / corpus
                    write_baseline_meta(
                        leaf_dir / "baseline_meta.json",
                        kernel=kernel,
                        channel=channel,
                        track5_mode=mode if idx > 0 else track5_mode if track5_mode else mode,
                    )
                    write_verification_report(leaf_dir / "verification_report.json")
                    write_control_metrics(leaf_dir / "control_metrics.json")
                    if corpus == "real":
                        write_control_metrics(
                            leaf_dir / "control_metrics.comprehensive_results.json",
                            simple_variance_real=1.35,
                            simple_variance_shuffled=1.0,
                            simple_variance_random=1.0,
                            metric_basis="comprehensive_results",
                        )
                    if corpus == "real":
                        write_relativity_deltas(leaf_dir / "relativity_deltas.json")
                        write_ablation_summary(
                            leaf_dir / "ablation_summary.json",
                            stage_2=0.5 if "hadamard" in mode else 0.6,
                            stage_3=0.72 if "hadamard" in mode else 0.76,
                            delta=0.28 if "hadamard" in mode else 0.34,
                        )
                        write_view_state(leaf_dir / "MONOLITH.view_state.json", survival_rate=0.72, mean_work=71.0)
                        write_walker_states(leaf_dir / "walker_states.json", closed_count=3, total=4, work=70.0)
                    else:
                        write_view_state(leaf_dir / "MONOLITH.view_state.json", survival_rate=0.18, mean_work=18.0)
                        write_walker_states(leaf_dir / "walker_states.json", closed_count=0, total=4, work=18.0)
                    write_monolith_data(leaf_dir / "MONOLITH_DATA.csv")

    synthetic_manifest = None
    if include_synthetic:
        synthetic_dir = runs_dir / "experiments_20260504_020202"
        synthetic_manifest = synthetic_dir / "experiment_manifest.json"
        write_synthetic_manifest(
            synthetic_manifest,
            output_dir=synthetic_dir,
            kernels=kernels,
            seeds=seeds,
        )
        for kernel in kernels:
            for seed in seeds:
                write_synthetic_terrain_incremental_fixture(
                    synthetic_dir / "synthetic" / f"{kernel}_seed{seed}"
                )

    write_track4_observer_state_action_summary(
        root
        / "outputs"
        / "track4_action_graph"
        / "fixture_observer_state"
        / "track4_observer_state_ablation_summary.json"
    )

    return {
        "root": root,
        "methods": methods_path,
        "results": results_path,
        "runs_dir": runs_dir,
        "suite_manifest": suite_manifest,
        "synthetic_manifest": synthetic_manifest,
    }
