from __future__ import annotations

import json
from pathlib import Path

from scripts.run_track4_observer_state_matrix import (
    ACTION_BRANCH_SPECS,
    ObserverLeaf,
    build_replay_command,
    discover_observer_leaves,
    filter_leaves,
    main,
    _run_replays,
    _unique_int_or_default,
    _unique_or_default,
    _write_branch_comparison,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    return path


def _write_action_summary(
    root: Path,
    *,
    branch: str,
    corpus: str,
    kernel: str = "rbf",
    seed: int = 42,
    basis: str = "track2",
    ablation: str = "full",
    mean_action: float = 12.0,
    observer_transport_penalty: float = 3.0,
    hysteresis_penalty: float = 1.2,
) -> Path:
    run_dir = root / f"{branch}_{corpus}_{kernel}_seed{seed}_{basis}_{ablation}_20260521"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kernel": kernel,
        "seed": seed,
        "basis": basis,
        "action_branch": branch,
        "ablation": ablation,
        "observer_state_mode": "enabled",
        "n_articles": 8,
        "path_count": 2,
        "reached_count": 2,
        "mean_action": mean_action,
        "records": [
            {
                "metric": mean_action / 2.0,
                "observer_transport_penalty": observer_transport_penalty,
                "hysteresis_penalty": hysteresis_penalty,
                "shear_penalty": 0.25,
                "stress_penalty": 0.5,
                "curvature_penalty": 0.1,
            }
        ],
    }
    path = run_dir / "track4_action_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_discover_observer_leaves_dedupes_artifact_basis_by_track2_preference(tmp_path: Path) -> None:
    root = tmp_path / "seed42_kernel_slice"
    logits = _touch(root / "real" / "rbf" / "seed_42" / "track4_basis_logits_flat" / "observer_42.pt")
    track2 = _touch(root / "real" / "rbf" / "seed_42" / "track4_basis_track2" / "observer_42.pt")
    _touch(root / "real" / "rbf" / "seed_42" / "track4_basis_track2" / "relativity_cache" / "observer_42.pt")
    _touch(root / "control_random" / "imq" / "seed_420" / "track4_basis_track2" / "observer_420.pt")

    leaves = discover_observer_leaves(root)

    assert len(leaves) == 2
    real_leaf = next(leaf for leaf in leaves if leaf.corpus == "real")
    assert real_leaf.path == track2
    assert real_leaf.path != logits
    assert real_leaf.kernel == "rbf"
    assert real_leaf.seed == 42


def test_filter_leaves_applies_corpus_kernel_seed_matrix() -> None:
    leaves = [
        ObserverLeaf(path=Path("a"), corpus="real", kernel="rbf", seed=42),
        ObserverLeaf(path=Path("b"), corpus="real", kernel="matern", seed=420),
        ObserverLeaf(path=Path("c"), corpus="control_random", kernel="rbf", seed=42),
    ]

    filtered = filter_leaves(
        leaves,
        corpora=("real", "control_random"),
        kernels=("rbf",),
        seeds=(42,),
    )

    assert [(leaf.corpus, leaf.kernel, leaf.seed) for leaf in filtered] == [
        ("real", "rbf", 42),
        ("control_random", "rbf", 42),
    ]


def test_build_replay_command_threads_observer_state_knobs(tmp_path: Path) -> None:
    leaf = ObserverLeaf(path=tmp_path / "observer_42.pt", corpus="real", kernel="rbf", seed=42)

    command = build_replay_command(
        leaf,
        output_dir=tmp_path / "out",
        basis="integrated",
        action_branch="richer_walker_state",
        observer_state_mode="shuffled",
        observer_shuffle_seed=9042,
        hysteresis_weight=0.0,
        observer_transport_weight=1.5,
        max_paths=5,
    )
    joined = " ".join(command)

    assert "run_track4_action_graph.py" in joined
    assert "--basis integrated" in joined
    assert "--action-branch richer_walker_state" in joined
    assert "--observer-state-mode shuffled" in joined
    assert "--observer-shuffle-seed 9042" in joined
    assert "--observer-transport-weight 1.5" in joined
    assert "--hysteresis-weight 0.0" in joined
    assert "--max-paths 5" in joined


def test_build_replay_command_threads_repair_branch_action_mode_flags(tmp_path: Path) -> None:
    leaf = ObserverLeaf(path=tmp_path / "observer_42.pt", corpus="real", kernel="rbf", seed=42)

    expected = {
        "null_calibrated_hysteresis": {
            "--core-action-mode": "null_calibrated",
            "--null-calibrated-hysteresis": None,
        },
        "richer_walker_state": {
            "--core-action-mode": "richer_state",
            "--richer-walker-state": None,
        },
        "virtual_transition_states": {
            "--core-action-mode": "virtual_transition",
            "--virtual-transition-states": None,
            "--virtual-interpolation-steps": "3",
        },
    }

    for branch, expected_flags in expected.items():
        spec = ACTION_BRANCH_SPECS[branch]
        command = build_replay_command(
            leaf,
            output_dir=tmp_path / branch,
            basis="track2",
            action_branch=branch,
            observer_state_mode=spec.observer_state_mode,
            observer_shuffle_seed=spec.observer_shuffle_seed,
            hysteresis_weight=spec.hysteresis_weight,
            observer_transport_weight=spec.observer_transport_weight,
            max_paths=5,
            k_neighbors=10 + spec.k_neighbors_delta,
            core_action_mode=spec.core_action_mode,
            virtual_interpolation_steps=spec.virtual_interpolation_steps,
        )
        joined = " ".join(command)

        assert f"--action-branch {branch}" in joined
        for flag, value in expected_flags.items():
            assert flag in command
            if value is not None:
                assert f"{flag} {value}" in joined


def test_matrix_runner_normalizes_append_defaults_without_duplicates() -> None:
    assert _unique_or_default(None, ("track2", "integrated")) == ["track2", "integrated"]
    assert _unique_or_default(["track2", "integrated", "track2"], ("logits_flat",)) == ["track2", "integrated"]
    assert _unique_int_or_default(None, (42,)) == [42]
    assert _unique_int_or_default([42, 420, 42, 4200], (1,)) == [42, 420, 4200]


def test_matrix_runner_defaults_to_required_robustness_seeds(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "artifacts"
    output_root = tmp_path / "matrix_out"
    for seed in (42, 420, 4200):
        _touch(input_root / "real" / "rbf" / f"seed_{seed}" / "track4_basis_track2" / f"observer_{seed}.pt")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_track4_observer_state_matrix.py",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert main() == 0

    assert not (output_root / "observer_state_matrix_inventory.json").exists()
    inventory = json.loads((output_root / "observer_state_matrix_inventory.dry_run.json").read_text(encoding="utf-8"))
    assert inventory["leaf_count"] == 3
    assert [leaf["seed"] for leaf in inventory["leaves"]] == [42, 420, 4200]
    assert inventory["action_branches"] == [
        "baseline_raw_action",
        "track2_default",
        "null_calibrated_hysteresis",
        "richer_walker_state",
        "separated_action_channels",
        "virtual_transition_states",
        "path_ensemble_tpt",
        "per_basis_gates",
    ]
    assert inventory["skip_existing"] is False


def test_run_replays_skip_existing_reuses_summary_without_rerun(tmp_path: Path, monkeypatch) -> None:
    leaf = ObserverLeaf(
        path=_touch(tmp_path / "artifacts" / "control_random" / "rbf" / "seed_42" / "observer_42.pt"),
        corpus="control_random",
        kernel="rbf",
        seed=42,
    )
    expected_summary = (
        tmp_path
        / "out"
        / "baseline_raw_action_control_random_rbf_seed42_track2_full_20260521"
        / "track4_action_summary.json"
    )
    _write_action_summary(
        tmp_path / "out",
        branch="baseline_raw_action",
        corpus="control_random",
        kernel="rbf",
        seed=42,
        basis="track2",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("existing summary should have been reused")

    monkeypatch.setattr(
        "scripts.run_track4_observer_state_matrix._run_command",
        fail_if_called,
    )

    summaries = _run_replays(
        [leaf],
        output_root=tmp_path / "out",
        bases=("track2",),
        action_branches=("baseline_raw_action",),
        dry_run=False,
        skip_existing=True,
        max_paths=1,
        run_suffix="20260521",
    )

    assert summaries == [expected_summary]


def test_branch_comparison_writes_per_branch_summaries_with_branch_basis_policy(tmp_path: Path) -> None:
    paths: list[Path] = []
    for branch, bases in {
        "baseline_raw_action": ("track2", "integrated"),
        "track2_default": ("track2",),
    }.items():
        for basis in bases:
            paths.extend(
                [
                    _write_action_summary(
                        tmp_path,
                        branch=branch,
                        corpus="real",
                        basis=basis,
                        mean_action=12.0,
                        observer_transport_penalty=3.0,
                        hysteresis_penalty=1.2,
                    ),
                    _write_action_summary(
                        tmp_path,
                        branch=branch,
                        corpus="control_random",
                        basis=basis,
                        mean_action=6.0,
                        observer_transport_penalty=1.0,
                        hysteresis_penalty=0.4,
                    ),
                    _write_action_summary(
                        tmp_path,
                        branch=branch,
                        corpus="real",
                        basis=basis,
                        ablation="observer_disabled",
                        mean_action=8.0,
                        observer_transport_penalty=0.0,
                        hysteresis_penalty=0.0,
                    ),
                    _write_action_summary(
                        tmp_path,
                        branch=branch,
                        corpus="real",
                        basis=basis,
                        ablation="observer_shuffled",
                        mean_action=5.0,
                        observer_transport_penalty=0.8,
                        hysteresis_penalty=0.3,
                    ),
                ]
            )

    out_path = _write_branch_comparison(
        paths,
        tmp_path / "out",
        action_branches=("baseline_raw_action", "track2_default"),
        required_kernels=("rbf",),
        required_seeds=(42,),
        required_bases=("track2", "integrated"),
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary_type"] == "track4_observer_state_action_branch_comparison"
    assert [row["action_branch"] for row in payload["ranked_branches"]] == [
        "baseline_raw_action",
        "track2_default",
    ]
    track2_summary = payload["branch_summaries"]["track2_default"]["summary"]
    assert track2_summary["required_bases"] == ["track2"]
    assert track2_summary["safe_for_thesis_claim"] is True
    assert (tmp_path / "out" / "summary_track2_default" / "track4_observer_state_ablation_summary.json").exists()
