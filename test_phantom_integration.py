"""
Integration test for Phantom Path Differential implementation.
Tests all modified components to ensure they work together correctly.
"""

import torch
import numpy as np
from core.complete_pipeline import ASTERPipeline
from core.phase_space_integrator import PhaseSpaceIntegrator
from core.spectral_polarity import compute_spectral_distance
from core.hott_sidecar import HoTTVerifier

def test_spectral_distance():
    """Test the new compute_spectral_distance function."""
    print("Testing compute_spectral_distance...")

    # Create sample phase space points
    batch_size, seq_len, dim = 2, 4, 8
    phase_A = torch.randn(batch_size, seq_len, dim)
    phase_B = torch.randn(batch_size, seq_len, dim)

    # Compute spectral distance
    distance = compute_spectral_distance(phase_A, phase_B)

    assert distance.shape == (batch_size,), f"Expected shape (batch_size,), got {distance.shape}"
    assert torch.all(distance >= 0), "Spectral distance should be non-negative"
    assert not torch.any(torch.isnan(distance)), "Spectral distance contains NaN"
    assert not torch.any(torch.isinf(distance)), "Spectral distance contains Inf"

    print(f" Spectral distance shape: {distance.shape}")
    print(f" Distance range: [{distance.min().item():.4f}, {distance.max().item():.4f}]")
    print()

def test_phase_space_integrator_phantom():
    """Test PhaseSpaceIntegrator with phantom differential computation."""
    print("Testing PhaseSpaceIntegrator phantom methods...")

    batch_size, seq_len, dim = 2, 4, 8
    integrator = PhaseSpaceIntegrator(dim=dim)

    # Test phase evolution
    phase_0 = torch.randn(batch_size, seq_len, dim)
    belief_0 = torch.randn(batch_size, seq_len, dim)
    h_t = torch.randn(batch_size, seq_len, dim)

    phase_1 = integrator(phase_0, belief_0, h_t)

    assert phase_1.shape == phase_0.shape, f"Phase shape mismatch: {phase_1.shape} vs {phase_0.shape}"
    print(f" Phase evolution output shape: {phase_1.shape}")

    # Test phantom differential computation
    differential = integrator.compute_phantom_differential(phase_0, phase_1)

    assert differential.shape == (batch_size,), f"Expected differential shape (batch_size,), got {differential.shape}"
    assert torch.all(differential >= 0), "Phantom differential should be non-negative"
    assert not torch.any(torch.isnan(differential)), "Phantom differential contains NaN"
    assert not torch.any(torch.isinf(differential)), "Phantom differential contains Inf"

    print(f" Phantom differential shape: {differential.shape}")
    print(f" Differential range: [{differential.min().item():.4f}, {differential.max().item():.4f}]")
    print()

def test_hott_verifier_phantom_parameter():
    """Test HoTTVerifier with new phantom_verdict parameter."""
    print("Testing HoTTVerifier with phantom_verdict parameter...")

    batch_size, seq_len, dim = 2, 4, 8
    verifier = HoTTVerifier(dim=dim)

    # Test inputs
    belief_mu = torch.randn(batch_size, seq_len, dim)
    belief_prec = torch.rand(batch_size, seq_len, dim, dim) + torch.eye(dim) * 0.1
    h_t = torch.randn(batch_size, seq_len, dim)

    # Test without phantom verdict
    output1 = verifier(belief_mu, belief_prec, h_t)
    assert 'verdict' in output1, "Output should contain 'verdict' key"
    assert 'coherence_score' in output1, "Output should contain 'coherence_score' key"
    print(f" HoTTVerifier output keys (no phantom): {list(output1.keys())}")

    # Test with phantom verdict
    phantom_diff = torch.rand(batch_size)
    output2 = verifier(belief_mu, belief_prec, h_t, phantom_verdict=phantom_diff)
    assert 'verdict' in output2, "Output should contain 'verdict' key"
    assert 'coherence_score' in output2, "Output should contain 'coherence_score' key"
    print(f" HoTTVerifier output keys (with phantom): {list(output2.keys())}")
    print()

def test_complete_pipeline_integration():
    """Test the complete ASTER pipeline with all phantom components."""
    print("Testing complete ASTER pipeline integration...")

    # Initialize pipeline
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    n_layers = 2

    pipeline = ASTERPipeline(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )

    # Create sample input
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Run forward pass
    output = pipeline(input_ids)

    # Verify output structure
    assert 'logits' in output, "Output should contain 'logits'"
    assert 'belief_mu' in output, "Output should contain 'belief_mu'"
    assert 'belief_prec' in output, "Output should contain 'belief_prec'"
    assert 'hott_verdict' in output, "Output should contain 'hott_verdict'"

    # Verify new phantom components
    assert 'fog_gate' in output, "Output should contain 'fog_gate' (phantom path gating)"
    assert 'phantom_differential' in output, "Output should contain 'phantom_differential'"

    print(f" Pipeline output keys: {list(output.keys())}")
    print(f" Logits shape: {output['logits'].shape}")
    print(f" Fog gate shape: {output['fog_gate'].shape}")
    print(f" Phantom differential shape: {output['phantom_differential'].shape}")

    # Verify fog gate properties
    fog_gate = output['fog_gate']
    assert torch.all((fog_gate >= 0) & (fog_gate <= 1)), "Fog gate should be in [0, 1]"
    print(f" Fog gate range: [{fog_gate.min().item():.4f}, {fog_gate.max().item():.4f}]")

    # Verify phantom differential properties
    phantom_diff = output['phantom_differential']
    assert torch.all(phantom_diff >= 0), "Phantom differential should be non-negative"
    print(f" Phantom differential range: [{phantom_diff.min().item():.4f}, {phantom_diff.max().item():.4f}]")

    print()

def main():
    print("=" * 70)
    print("ASTER v3.2 Phantom Path Differential - Integration Test")
    print("=" * 70)
    print()

    try:
        test_spectral_distance()
        test_phase_space_integrator_phantom()
        test_hott_verifier_phantom_parameter()
        test_complete_pipeline_integration()

        print("=" * 70)
        print(" ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("   compute_spectral_distance() working correctly")
        print("   PhaseSpaceIntegrator.compute_phantom_differential() working correctly")
        print("   HoTTVerifier phantom_verdict parameter working correctly")
        print("   Complete pipeline fog gating and phantom differential working correctly")
        print("   All outputs have correct shapes and valid ranges")
        print()

        return 0

    except Exception as e:
        print("=" * 70)
        print(" INTEGRATION TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())



