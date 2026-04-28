import json
from pathlib import Path

import numpy as np
import torch

from core.complete_pipeline import (
    SPECTRAL_CLS_NORMALIZATION_CONTRACT,
    _canonicalize_cls_per_bot_for_spectral,
    _classify_track5_semantic_verdicts,
    _construct_spectral_poles,
    _map_track4_state_to_track5_verdict,
    _serialize_rks_basis_state,
)
from core.dirichlet_fusion import SharedRKSBasis
from core.metric_fusion import calculate_unified_metric
from core.physarum_walk import compute_corpus_walker_resistance, compute_walker_resistance


def test_canonical_cls_per_bot_contract_is_magnitude_preserving():
    cls_entries = [
        torch.tensor([[3.0, 4.0], [0.0, 5.0]], dtype=torch.float32),
        torch.tensor([[6.0, 8.0], [8.0, 15.0]], dtype=torch.float32),
    ]

    stacked, as_list, contract = _canonicalize_cls_per_bot_for_spectral(
        cls_entries,
        normalize_features_flag=True,
    )

    assert stacked is not None
    assert as_list is not None
    assert contract["contract"] == SPECTRAL_CLS_NORMALIZATION_CONTRACT
    assert contract["requested_normalize_features"] is True
    assert contract["applied_l2_normalization"] is False
    assert torch.allclose(stacked, torch.stack(cls_entries, dim=0))
    assert stacked[0].norm(dim=-1).max().item() > 1.0
    assert torch.equal(as_list[0], stacked[0])
    assert torch.equal(as_list[1], stacked[1])


def test_construct_spectral_poles_falls_back_for_missing_positive_bucket():
    G = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]],
        dtype=torch.float32,
    )
    mags = torch.tensor([[-2.0, -1.0, 0.0]], dtype=torch.float32)

    poles = _construct_spectral_poles(G, mags)

    assert bool(poles["fallback_used"][0].item()) is True
    assert poles["fallback_state"][0] == "fallback_positive_sign_bucket_empty"
    assert torch.allclose(poles["emb_pos"][0], G[0, 2])  # argmax(mags) == 2
    assert torch.isfinite(poles["emb_neg"]).all()


def test_construct_spectral_poles_explicit_both_empty_fallback():
    G = torch.tensor(
        [[[2.0, 2.0], [5.0, 5.0], [7.0, 7.0]]],
        dtype=torch.float32,
    )
    mags = torch.zeros((1, 3), dtype=torch.float32)

    poles = _construct_spectral_poles(G, mags)

    assert bool(poles["fallback_used"][0].item()) is True
    assert poles["fallback_state"][0] == "fallback_both_sign_buckets_empty"
    assert torch.allclose(poles["emb_pos"][0], G[0, 0])
    assert torch.allclose(poles["emb_neg"][0], G[0, 1])


def test_track4_panic_bypass_preserves_native_states():
    assert _map_track4_state_to_track5_verdict("phantom", phantom_ratio=0.1) == "PHANTOM"
    assert _map_track4_state_to_track5_verdict("honest", phantom_ratio=9.0) == "HONEST"
    assert _map_track4_state_to_track5_verdict("tautology", phantom_ratio=9.0) == "TAUTOLOGY"
    assert _map_track4_state_to_track5_verdict("Type 2 Rupture", phantom_ratio=0.1) == "TAUTOLOGY"
    assert _map_track4_state_to_track5_verdict("Type 1 Rupture", phantom_ratio=0.1) == "PHANTOM"
    assert _map_track4_state_to_track5_verdict("rupture", phantom_ratio=0.1) == "PHANTOM"


def test_metric_fusion_prefers_track5_verdict_ledger():
    run_dir = Path(".pytest_local_tmp/test_metric_fusion_prefers_track5_verdict_ledger")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "features.npy", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(run_dir / "spectral_u_axis.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(run_dir / "walker_work_integrals.npy", np.array([5.0, 50.0], dtype=np.float32))
    np.save(run_dir / "d_spectral.npy", np.array([1.0, 2.0], dtype=np.float32))
    np.save(run_dir / "spectral_probe_magnitudes.npy", np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32))
    (run_dir / "article_metadata.csv").write_text(
        "index,bt_uid,title\n0,u0,A0\n1,u1,A1\n",
        encoding="utf-8",
    )
    (run_dir / "validation.json").write_text('{"nmi": 0.5}', encoding="utf-8")
    (run_dir / "walker_states.json").write_text(
        json.dumps([
            {"label": "phantom", "status": "SUCCESS"},
            {"label": "phantom", "status": "SUCCESS"},
        ]),
        encoding="utf-8",
    )
    (run_dir / "phantom_verdicts.json").write_text(
        json.dumps([
            {"verdict": "PHANTOM"},
            {"verdict": "RUPTURE"},
        ]),
        encoding="utf-8",
    )

    df = calculate_unified_metric(
        embeddings_path=run_dir / "features.npy",
        gradients_path=run_dir / "spectral_u_axis.npy",
        metadata_path=run_dir / "article_metadata.csv",
        output_path=run_dir / "MONOLITH_DATA.csv",
        knn_k=1,
    )

    assert list(df["verdict"]) == ["PHANTOM", "PHANTOM"]


def test_track5_semantic_classifier_uses_article_delta_and_anomalies():
    work = np.array([1.0, 2.0, 9.0, 0.5], dtype=np.float32)
    disp = np.array([100.0, 40.0, 50.0, 1e-6], dtype=np.float32)
    records = [
        {"label": "phantom", "anomaly_kind": "none"},
        {"label": "phantom", "anomaly_kind": "none"},
        {"label": "phantom", "anomaly_kind": "kinetic_break"},
        {"label": "tautology", "anomaly_kind": "trapped_stall"},
    ]

    verdicts, stats = _classify_track5_semantic_verdicts(work, disp, records)

    assert verdicts.tolist() == ["HONEST", "HONEST", "PHANTOM", "TAUTOLOGY"]
    assert stats["threshold_mode"] in {"kmeans_2cluster", "median_fallback"}


def test_track5_semantic_classifier_emits_lower_tail_tautologies_when_supported():
    delta = np.array(
        [0.030, 0.032, 0.034, 0.036, 0.038, 0.040, 0.043, 0.046, 0.050, 0.054,
         0.058, 0.062, 0.070, 0.078, 0.086, 0.094, 0.110, 0.128, 0.160, 0.220],
        dtype=np.float32,
    )
    work = delta.copy()
    disp = np.ones_like(delta, dtype=np.float32)
    records = [{"label": "phantom", "anomaly_kind": "none"} for _ in range(len(delta))]

    verdicts, stats = _classify_track5_semantic_verdicts(work, disp, records)

    assert "TAUTOLOGY" in verdicts.tolist()
    assert "HONEST" in verdicts.tolist()
    assert "PHANTOM" in verdicts.tolist()
    assert stats["tautology_cutoff"] is not None
    assert stats["tautology_cutoff"] < stats["honest_cutoff"]


def test_serialize_rks_basis_state_preserves_replay_fields():
    basis = SharedRKSBasis(input_dim=4, output_dim=6, seed=7, kernel_type="matern", nu=2.5, roughness=5)
    basis.set_sigma(1.75)

    state = _serialize_rks_basis_state(basis)

    assert state is not None
    assert state["input_dim"] == 4
    assert state["output_dim"] == 6
    assert state["seed"] == 7
    assert state["kernel_type"] == "matern"
    assert state["nu"] == 2.5
    assert state["roughness"] == 5
    assert state["sigma"] == 1.75
    assert torch.is_tensor(state["omega"])
    assert torch.is_tensor(state["b"])
    assert state["omega"].shape == (4, 6)
    assert state["b"].shape == (6,)


def test_compute_walker_resistance_observer_cost_changes_path_metrics():
    cls_per_bot = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.2, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, 0.2, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5, 0.2],
            [0.0, 0.0, 0.0, 2.0],
            [0.2, 0.0, 0.0, 1.5],
        ],
        dtype=torch.float32,
    )
    basis = SharedRKSBasis(input_dim=4, output_dim=8, seed=11, kernel_type="rbf")
    basis.set_sigma(1.0)
    observer_axis = torch.tensor([1.0, -0.5, 0.25, 0.0], dtype=torch.float32)

    torch.manual_seed(123)
    baseline = compute_walker_resistance(
        cls_per_bot=cls_per_bot,
        rks_basis=basis,
        n_walkers=8,
        n_steps=6,
        observer_cost_strength=0.0,
    )

    torch.manual_seed(123)
    conditioned = compute_walker_resistance(
        cls_per_bot=cls_per_bot,
        rks_basis=basis,
        n_walkers=8,
        n_steps=6,
        observer_axis=observer_axis,
        observer_cost_strength=1.5,
    )

    assert conditioned["observer_cost_strength"] == 1.5
    assert not np.isclose(
        float(baseline["work_integral"]),
        float(conditioned["work_integral"]),
        atol=1e-5,
    )
    base_path = np.asarray(baseline["path_xyz"], dtype=np.float32)
    conditioned_path = np.asarray(conditioned["path_xyz"], dtype=np.float32)
    assert base_path.shape == conditioned_path.shape
    assert not np.allclose(base_path, conditioned_path, atol=1e-5)
    assert "observer_penalty" in conditioned["step_diagnostics"][0]
    assert "observer_similarity" in conditioned["step_diagnostics"][0]


def test_compute_corpus_walker_resistance_returns_article_neighbor_contract():
    embeddings = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [1.1, 1.0, 0.0, 0.0],
            [0.1, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    density = torch.full((4, 4), 0.5, dtype=torch.float32)
    stress = torch.tensor([0.0, 0.4, 0.9, 0.2], dtype=torch.float32)
    basis = SharedRKSBasis(input_dim=4, output_dim=4, seed=13, kernel_type="rbf")
    basis.set_sigma(1.0)

    result = compute_corpus_walker_resistance(
        embeddings=embeddings,
        rks_basis=basis,
        track3_density=density,
        z_coordinates=stress,
        metric_stress=stress,
        article_coords_2d=embeddings[:, :2],
        article_ids=["a0", "a1", "a2", "a3"],
        n_walkers=4,
        max_steps=12,
        k_neighbors=2,
        start_seed=7,
    )

    assert result["walker_output"].shape == embeddings.shape
    assert len(result["work_integrals"]) == 4
    assert len(result["states"]) == 4
    assert len(result["state_records"]) == 4
    assert len(result["path_records"]) == 4
    assert len(result["step_diagnostics"]) == 4
    assert len(result["catalyst_indices"]) <= 3
    assert len(result["anchor_summaries"]) == len(result["catalyst_indices"])
    assert all(record["bt_uid"].startswith("a") for record in result["path_records"])
    assert all(np.asarray(record["path_xyz"], dtype=np.float32).ndim == 2 for record in result["path_records"])
    assert set(result["states"]).issubset({"closed_loop", "open_loop"})
    assert all("closed_loop" in record for record in result["state_records"])
    assert all("work_integral" in record for record in result["state_records"])
    assert all("closed_loop" in record for record in result["path_records"])
    assert all("work_integral" in record for record in result["path_records"])
    non_empty_steps = [record["steps"] for record in result["step_diagnostics"] if record.get("steps")]
    assert non_empty_steps
    first_step = non_empty_steps[0][0]
    assert "metric_distance" in first_step
    assert "density_midpoint" in first_step
    assert "shear_projection" in first_step
