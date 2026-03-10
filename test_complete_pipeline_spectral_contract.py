import torch

from core.complete_pipeline import (
    SPECTRAL_CLS_NORMALIZATION_CONTRACT,
    _canonicalize_cls_per_bot_for_spectral,
    _construct_spectral_poles,
    _map_track4_state_to_track5_verdict,
)


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
    assert _map_track4_state_to_track5_verdict("Type 2 Rupture", phantom_ratio=0.1) == "RUPTURE"
