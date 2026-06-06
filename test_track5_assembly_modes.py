from __future__ import annotations

import torch

from core.complete_pipeline import BeliefTransformerPipeline
from core.phase_space_integrator import IntegratorConfig, PhaseSpaceIntegrator
from core.pipeline_config import (
    PipelineRuntimeConfig,
    TRACK5_ASSEMBLY_MODE_HADAMARD,
    TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN,
)


def _minimal_components(track5_assembly_mode: str | None = None) -> dict:
    components = {
        "nli_extractor": object(),
        "gru_model": None,
        "rks_map": None,
        "attention_model": None,
        "recorder": None,
    }
    if track5_assembly_mode is not None:
        components["track5_assembly_mode"] = track5_assembly_mode
    return components


def _build_track_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    n_samples = 6
    logits = torch.randn(n_samples, 24)
    antagonism = torch.randn(n_samples, 8)
    hologram = torch.randn(n_samples, 8)
    blinker = torch.rand(n_samples, 8) + 0.25
    walker = torch.randn(n_samples, 5)
    return logits, antagonism, hologram, blinker, walker


def _fit_integrator(include_antagonism: bool = False) -> PhaseSpaceIntegrator:
    logits, antagonism, hologram, blinker, walker = _build_track_tensors()
    integrator = PhaseSpaceIntegrator(IntegratorConfig())
    track_samples = {
        "logits": logits,
        "hologram": hologram,
        "blinker": blinker,
        "walker": walker,
    }
    if include_antagonism:
        track_samples["antagonism"] = antagonism
    integrator.fit(track_samples)
    return integrator


def test_pipeline_runtime_config_threads_track5_mode() -> None:
    runtime_cfg = PipelineRuntimeConfig.from_mode_config({"track5_assembly_mode": "riemannian"})

    assert runtime_cfg.track5_assembly_mode == TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN
    assert runtime_cfg.to_initialize_kwargs()["track5_assembly_mode"] == TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN


def test_pipeline_defaults_to_hadamard_track5_mode() -> None:
    pipeline = BeliefTransformerPipeline(_minimal_components())

    assert pipeline.track5_assembly_mode == TRACK5_ASSEMBLY_MODE_HADAMARD


def test_integrator_explicit_hadamard_mode_matches_legacy_flag() -> None:
    logits, antagonism, hologram, blinker, walker = _build_track_tensors()
    integrator = _fit_integrator(include_antagonism=False)

    legacy_particles = integrator.integrate(
        logits=logits,
        antagonism=antagonism,
        hologram=hologram,
        blinker=blinker,
        walker=walker,
        use_hadamard_fusion=True,
    )
    explicit_particles = integrator.integrate(
        logits=logits,
        antagonism=antagonism,
        hologram=hologram,
        blinker=blinker,
        walker=walker,
        track5_assembly_mode=TRACK5_ASSEMBLY_MODE_HADAMARD,
    )

    assert len(legacy_particles) == len(explicit_particles) == logits.shape[0]
    for legacy, explicit in zip(legacy_particles, explicit_particles):
        assert torch.allclose(legacy.vector, explicit.vector, atol=1e-6)
        assert legacy.hadamard_diagnostics["assembly_mode"] == TRACK5_ASSEMBLY_MODE_HADAMARD


def test_integrator_supports_strict_riemannian_track5_mode() -> None:
    logits, antagonism, hologram, blinker, walker = _build_track_tensors()
    integrator = _fit_integrator(include_antagonism=True)

    particles = integrator.integrate(
        logits=logits,
        antagonism=antagonism,
        hologram=hologram,
        blinker=blinker,
        walker=walker,
        track5_assembly_mode=TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN,
    )

    assert len(particles) == logits.shape[0]
    assert all("hologram" in particle.track_contributions for particle in particles)
    assert all("antagonism" not in particle.track_contributions for particle in particles)
    assert all("blinker" not in particle.track_contributions for particle in particles)
    assert all(particle.hadamard_diagnostics["assembly_mode"] == TRACK5_ASSEMBLY_MODE_STRICT_RIEMANNIAN for particle in particles)

    norms = torch.stack([particle.vector.norm() for particle in particles])
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
