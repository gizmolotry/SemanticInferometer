import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from core.hadamard_fusion import HadamardFusion, HadamardFusionConfig
from core.physarum_walk import SemanticWalker, compute_corpus_walker_resistance


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

    def test_catalyst_selection_falls_back_to_spatially_distinct_anchors(self):
        embeddings = torch.randn(5, 4)
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
            ],
            dtype=torch.float32,
        )
        walker = SemanticWalker(
            embeddings=embeddings,
            rks_basis=lambda x: x,
            article_coords_2d=coords,
            metric_stress=torch.tensor([10.0, 9.0, 0.0, 0.0, 0.0]),
            track3_density=torch.ones(5),
        )

        catalysts = walker.select_catalysts()

        self.assertEqual(len(catalysts), 3)
        self.assertEqual(len(set(catalysts)), 3)
        self.assertIn(0, catalysts)
        self.assertIn(4, catalysts)

    def test_catalyst_selection_is_zone_constrained_before_farthest_fill(self):
        embeddings = torch.randn(6, 4)
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],
                [15.0, 0.0],
                [20.0, 0.0],
            ],
            dtype=torch.float32,
        )
        walker = SemanticWalker(
            embeddings=embeddings,
            rks_basis=lambda x: x,
            article_coords_2d=coords,
            track3_density=torch.tensor([0.0, 0.05, 0.95, 0.95, 0.05, 0.90]),
            metric_stress=torch.tensor([1.0, 0.9, 0.05, 0.9, 0.05, 1.0]),
        )

        catalysts = walker.select_catalysts()
        selected_zones = walker._last_catalyst_selection["selected_zones"]

        self.assertEqual(len(catalysts), 3)
        self.assertEqual(len(set(catalysts)), 3)
        self.assertIn("Void", selected_zones)
        self.assertIn("Bridge", selected_zones)
        self.assertEqual(len(set(selected_zones)), 3)

    def test_cyclic_walk_exports_markov_observables_without_breaking_legacy_keys(self):
        embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        walker = SemanticWalker(
            embeddings=embeddings,
            rks_basis=lambda x: x,
            article_coords_2d=embeddings[:, :2],
            track3_density=torch.tensor([0.0, 0.25, 0.9, 0.95, 0.1]),
            metric_stress=torch.tensor([1.0, 0.8, 0.1, 0.9, 0.2]),
            temperature=0.75,
        )

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = walker.run_stress_triggered_cyclic_walk(
                max_steps=12,
                k_neighbors=4,
                output_dir=str(out_dir),
                start_seed=17,
            )
            with np.load(out_dir / "cyclic_paths.npz", allow_pickle=True) as payload:
                for key in ("work_integral", "closed_loop", "path_anchor_idx", "path_is_hot", "anchor_indices", "path_indices"):
                    self.assertIn(key, payload.files)
                for key in ("committor_to_void", "mfpt_to_bridge", "mfpt_to_void", "reactive_flux_edges", "reactive_flux_values"):
                    self.assertIn(key, payload.files)
                committor = np.asarray(payload["committor_to_void"], dtype=float)
            self.assertTrue(np.isfinite(committor).any())
            self.assertGreaterEqual(float(np.nanmin(committor)), 0.0)
            self.assertLessEqual(float(np.nanmax(committor)), 1.0)
            self.assertEqual(result["markov_observables"]["status"], "OK")
            self.assertTrue((out_dir / "track4_markov_summary.json").exists())

    def test_cyclic_walk_proposal_modes_are_additive_lab_switches(self):
        embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )
        modes = ("metric_softmax", "stress_biased", "committor_guided", "deterministic_low_cost")

        for mode in modes:
            walker = SemanticWalker(
                embeddings=embeddings,
                rks_basis=lambda x: x,
                article_coords_2d=embeddings[:, :2],
                track3_density=torch.tensor([0.0, 0.15, 0.95, 0.9, 0.1, 0.8]),
                metric_stress=torch.tensor([1.0, 0.85, 0.1, 0.9, 0.2, 0.05]),
                temperature=0.75,
            )
            with TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                result = walker.run_stress_triggered_cyclic_walk(
                    max_steps=10,
                    k_neighbors=4,
                    output_dir=str(out_dir),
                    start_seed=19,
                    proposal_mode=mode,
                    feature_basis="logits_flat",
                )
                with np.load(out_dir / "cyclic_paths.npz", allow_pickle=True) as payload:
                    self.assertIn("path_proposal_mode", payload.files)
                    self.assertIn("path_feature_basis", payload.files)
                    exported_modes = {str(value) for value in payload["path_proposal_mode"].tolist()}
                    exported_basis = {str(value) for value in payload["path_feature_basis"].tolist()}
                self.assertEqual(result["proposal_mode"], mode)
                self.assertEqual(result["feature_basis"], "logits_flat")
                self.assertEqual(exported_modes, {mode})
                self.assertEqual(exported_basis, {"logits_flat"})

        walker = SemanticWalker(embeddings=embeddings, rks_basis=lambda x: x)
        with self.assertRaises(ValueError):
            walker.run_stress_triggered_cyclic_walk(proposal_mode="not_a_mode")

    def test_cyclic_walk_can_adapt_neighbor_graph_for_tpt_flux(self):
        embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [10.2, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        walker = SemanticWalker(
            embeddings=embeddings,
            rks_basis=lambda x: x,
            article_coords_2d=embeddings[:, :2],
            track3_density=torch.tensor([0.95, 0.2, 0.1, 0.05, 0.05, 0.9]),
            metric_stress=torch.tensor([0.05, 0.1, 0.8, 0.95, 0.9, 0.9]),
            temperature=0.75,
        )

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = walker.run_stress_triggered_cyclic_walk(
                max_steps=8,
                k_neighbors=1,
                output_dir=str(out_dir),
                start_seed=23,
                adaptive_tpt_connectivity=True,
            )

            summary = result["markov_observables"]["summary"]
            self.assertTrue(summary["adaptive_tpt_connectivity"])
            self.assertGreaterEqual(result["effective_k_neighbors"], result["requested_k_neighbors"])
            self.assertEqual(summary["effective_k_neighbors"], result["effective_k_neighbors"])
            self.assertIn("connectivity_repair_applied", summary)

    def test_corpus_walker_resistance_preserves_track4_lab_knobs(self):
        embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )

        with TemporaryDirectory() as tmp:
            result = compute_corpus_walker_resistance(
                embeddings=embeddings,
                rks_basis=lambda x: x,
                track3_density=torch.tensor([0.0, 0.15, 0.95, 0.9, 0.1, 0.8]),
                metric_stress=torch.tensor([1.0, 0.85, 0.1, 0.9, 0.2, 0.05]),
                article_coords_2d=embeddings[:, :2],
                article_ids=[f"a{i}" for i in range(embeddings.shape[0])],
                temperature=0.75,
                max_steps=8,
                k_neighbors=4,
                start_seed=29,
                output_dir=tmp,
                proposal_mode="committor_guided",
                adaptive_tpt_connectivity=True,
                feature_basis="bot_norms",
            )

            self.assertEqual(result["proposal_mode"], "committor_guided")
            self.assertEqual(result["feature_basis"], "bot_norms")
            self.assertTrue(result["adaptive_tpt_connectivity"])
            self.assertIn("effective_k_neighbors", result)
            self.assertTrue(all(record["proposal_mode"] == "committor_guided" for record in result["state_records"]))
            self.assertTrue(all(record["proposal_mode"] == "committor_guided" for record in result["path_records"]))
            self.assertTrue(all(record["feature_basis"] == "bot_norms" for record in result["state_records"]))
            self.assertTrue(all(record["feature_basis"] == "bot_norms" for record in result["path_records"]))


if __name__ == "__main__":
    unittest.main()
