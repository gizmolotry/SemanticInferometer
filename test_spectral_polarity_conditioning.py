import math

import torch

from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig


def _make_probe_matrix() -> torch.Tensor:
    # Dominant axis + secondary axis to produce a non-trivial singular spectrum.
    return torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_adaptive_sigma_is_clamped_to_configured_floor():
    cfg = SpectralPolarityConfig(
        adaptive_sigma_min=1e-3,
        adaptive_sigma_max_samples=256,
    )
    sp = SpectralPolarity(cfg)
    # Near-identical points force tiny pairwise distances.
    g = torch.full((1, 8, 4), 1.0, dtype=torch.float32)
    g[:, 1:, :] += 1e-9
    sigma = sp._compute_adaptive_sigma(g)
    assert sigma >= cfg.adaptive_sigma_min


def test_compute_batch_handles_tiny_and_non_finite_sigma_values():
    cfg = SpectralPolarityConfig(
        sigma_values=(0.0, float("nan"), 1.0),
        use_adaptive_sigma=False,
        sigma_scale_epsilon=1e-4,
        multi_scale_enabled=True,
    )
    sp = SpectralPolarity(cfg)
    result = sp.compute_batch([_make_probe_matrix()])
    assert torch.isfinite(result.evr).all()
    assert torch.isfinite(result.singular_values).all()
    assert torch.isfinite(result.u_axis).all()
    assert torch.isfinite(result.evr_per_scale).all()


def test_weighting_modes_are_explicit_and_change_behavior():
    emb_a = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    emb_b = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    singular_values = torch.tensor([[2.0]], dtype=torch.float32)
    vt = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)

    weighted_sigma = SpectralPolarity(
        SpectralPolarityConfig(weighted_sigma_mode="sigma")
    ).compute_weighted_spectral_distance(emb_a, emb_b, singular_values, vt)
    weighted_sigma_sq = SpectralPolarity(
        SpectralPolarityConfig(weighted_sigma_mode="sigma_squared")
    ).compute_weighted_spectral_distance(emb_a, emb_b, singular_values, vt)
    assert torch.allclose(weighted_sigma, torch.tensor([math.sqrt(2.0)]))
    assert torch.allclose(weighted_sigma_sq, torch.tensor([2.0]))

    whitened_sigma = SpectralPolarity(
        SpectralPolarityConfig(whitened_sigma_mode="sigma")
    ).compute_whitened_spectral_distance(emb_a, emb_b, singular_values, vt)
    whitened_sigma_sq = SpectralPolarity(
        SpectralPolarityConfig(whitened_sigma_mode="sigma_squared")
    ).compute_whitened_spectral_distance(emb_a, emb_b, singular_values, vt)
    assert torch.allclose(whitened_sigma, torch.tensor([0.5]))
    assert torch.allclose(whitened_sigma_sq, torch.tensor([0.25]))


def test_dynamic_k_threshold_ratio_is_configurable_in_compute_batch():
    g = _make_probe_matrix()

    default_result = SpectralPolarity(
        SpectralPolarityConfig(
            multi_scale_enabled=False,
            dynamic_k_threshold_ratio=0.1,
        )
    ).compute_batch([g])
    strict_result = SpectralPolarity(
        SpectralPolarityConfig(
            multi_scale_enabled=False,
            dynamic_k_threshold_ratio=0.3,
        )
    ).compute_batch([g])

    assert default_result.dynamic_k.item() == 2
    assert strict_result.dynamic_k.item() == 1
