import unittest

import torch

from core.hadamard_fusion import HadamardFusion, HadamardFusionConfig
from core.physarum_walk import SemanticWalker


class TestCoreStability(unittest.TestCase):
    def test_hadamard_softening_recovers_connectivity(self):
        torch.manual_seed(7)
        n, d = 32, 24
        t2 = torch.randn(n, d)
        t15 = torch.randn(n, d)

        strict = HadamardFusion(HadamardFusionConfig(hadamard_softening=0.0))
        soft = HadamardFusion(HadamardFusionConfig(hadamard_softening=0.15))

        k2, k15 = strict.compute_kernel_matrices(t2, t15)
        k_strict, _, _ = strict.hadamard_product(k2, k15)
        k_soft, _, _ = soft.hadamard_product(k2, k15)

        self.assertTrue(torch.all(k_soft >= k_strict))
        self.assertLessEqual(float(k_soft.max()), 1.0)
        self.assertGreater(float(k_soft.mean()), float(k_strict.mean()))

    def test_work_integral_is_finite_in_extreme_voids(self):
        torch.manual_seed(11)
        n_bots, dim = 8, 16
        embeddings = torch.randn(n_bots, dim)
        gradients = torch.randn(n_bots, dim) * 1e6
        walker = SemanticWalker(
            embeddings=embeddings,
            gradients=gradients,
            rks_basis=lambda x: x,
            u_axis=None,
        )

        n_steps, n_walkers = 6, 5
        raw = torch.rand(n_steps + 1, n_walkers, n_bots)
        trajectory_weights = raw / raw.sum(dim=-1, keepdim=True)

        work, path_length, spectral_distance, divergence_ratio = walker.compute_work_integral(
            trajectory_weights
        )

        self.assertTrue(torch.isfinite(work).all())
        self.assertTrue(torch.isfinite(path_length).all())
        self.assertTrue(torch.isfinite(spectral_distance).all())
        self.assertTrue(torch.isfinite(divergence_ratio).all())
        self.assertTrue((work >= 0).all())
        # Friction is capped at 1e3 in compute_work_integral.
        self.assertTrue((work <= 1000.0 * path_length + 1e-4).all())


if __name__ == "__main__":
    unittest.main()
