"""
Integration coverage for the current phantom-path differential stack.

This test intentionally targets the live APIs rather than the older ASTER-shaped
prototype classes that no longer exist in the codebase.
"""

from __future__ import annotations

import numpy as np
import torch

from core.hott_sidecar import HoTTSidecar, HoTTSidecarConfig
from core.phase_space_integrator import IntegratorConfig, PhaseSpaceIntegrator
from core.spectral_polarity import SpectralPolarity, SpectralPolarityConfig


def test_spectral_distance_is_nonnegative():
    polarity = SpectralPolarity(SpectralPolarityConfig())
    batch_size, n_bots, hidden = 3, 8, 16
    cls_per_bot_list = [torch.randn(n_bots, hidden) for _ in range(batch_size)]
    u_axis = torch.randn(batch_size, hidden)

    distance = polarity.compute_spectral_distance(cls_per_bot_list, u_axis)

    assert distance.shape == (batch_size,)
    assert torch.all(distance >= 0)
    assert torch.isfinite(distance).all()


def test_phase_space_integrator_computes_phantom_differential_payloads():
    integrator = PhaseSpaceIntegrator(
        IntegratorConfig(track5_assembly_mode="hadamard_strict")
    )

    d_spectral = torch.tensor([0.5, 2.0, 0.25], dtype=torch.float32)
    w_actual = [0.2, 5.0, 9.0]
    walker_states = ["elastic", "trapped", "broken"]
    blinker_variance = [0.05, 0.7, 0.95]

    verdicts = integrator.compute_phantom_differential(
        d_spectral=d_spectral,
        w_actual=w_actual,
        walker_states=walker_states,
        blinker_variance=blinker_variance,
    )

    assert len(verdicts) == 3
    assert {payload["verdict"] for payload in verdicts}.issubset(
        {"TAUTOLOGY", "HONEST", "PHANTOM", "RUPTURE"}
    )
    assert {payload["terrain_state"] for payload in verdicts}.issubset(
        {"BRIDGE", "SWAMP", "TIGHTROPE", "VOID"}
    )
    assert all(payload["delta"] >= 0 for payload in verdicts)


def test_hott_sidecar_accepts_phantom_verdicts_from_integrator():
    integrator = PhaseSpaceIntegrator(
        IntegratorConfig(track5_assembly_mode="hadamard_strict")
    )
    sidecar = HoTTSidecar(HoTTSidecarConfig())

    d_spectral = torch.tensor([0.5, 2.0], dtype=torch.float32)
    w_actual = [0.6, 25.0]
    walker_states = ["elastic", "broken"]
    phantom_payloads = integrator.compute_phantom_differential(
        d_spectral=d_spectral,
        w_actual=w_actual,
        walker_states=walker_states,
        blinker_variance=[0.2, 0.9],
    )

    proofs = sidecar.prove_batch(
        article_ids=["a0", "a1"],
        evr_batch=np.array([0.8, 0.9], dtype=np.float32),
        dipole_valid_batch=np.array([True, True]),
        n_persistent_scales_batch=np.array([3, 4], dtype=np.int32),
        work_integrals=w_actual,
        walker_states=walker_states,
        probe_magnitudes_batch=np.array(
            [[0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.6, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        phantom_verdicts=[payload["verdict"] for payload in phantom_payloads],
    )

    assert len(proofs) == 2
    assert proofs[0].article_id == "a0"
    assert proofs[1].article_id == "a1"
    assert all(0.0 <= proof.confidence <= 1.0 for proof in proofs)
