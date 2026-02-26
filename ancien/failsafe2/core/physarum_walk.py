"""
core/physarum_walk.py

Track 4: The Path — Riemannian Walker Integration

The Semantic Walker Engine (Monte Carlo Semantic Exploration).
Simulates 'Semantic Walkers' traversing the framing landscape to map
Bonds (Valleys) and Cracks (Walls).

ASTER v3.2 Integration:
- Projects landscape onto gauge-fixed u_axis from Track 1.5
- Computes 1D potential along the compass direction
- Integrates work W = ∫ √(g_μν ẋ^μ ẋ^ν) dt along path
- Classifies: Elastic (W≈0), Trapped (W>0 finite), Broken (W→∞)

NOTE: This is distinct from the AtmosphericAnnealer in dirichlet_fusion.py,
which operates at the alpha-sweep / crack-bond topology level.
This module operates at the embedding level, using MCMC walkers to explore
the bot-weight simplex via gradient-informed Metropolis steps.
"""
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class WalkerState(Enum):
    """
    Classification of walker traversal based on DIVERGENCE RATIO (Δ).

    ASTER v3.2 PATH SYSTEM (Riemannian Web Router):
    Δ = W_actual / d_spectral (terrain-invariant efficiency metric)

    The 4-State Classification:
    ---------------------------
    - TAUTOLOGY: Δ → 0 (or d_spectral ≈ 0)
      Walker didn't actually move. Spinning in place. Null path.
      Example: "Israel says Israel is right" — no semantic displacement.

    - HONEST: 1.0 ≤ Δ < threshold_honest
      Direct geodesic path. May climb steep terrain, but efficient.
      Example: Controversial but substantive claim with clear evidence.

    - PHANTOM: threshold_honest ≤ Δ < threshold_rupture
      High-energy loop/detour to bypass obstruction. Creates fake path.
      The Spin captures GEOMETRY (how the path curves), not just cost.
      Example: Claim that requires mental gymnastics to connect A→B.

    - RUPTURE: Δ ≥ threshold_rupture (or Δ → ∞)
      Impossible path. Walker hit void/wall. Topological barrier.
      Example: Claim that contradicts established facts with no bridge.

    NO WIND. Pure slope/gravity physics based on terrain gradient (∇Φ).
    """
    TAUTOLOGY = "tautology"  # Δ → 0: no real movement
    HONEST = "honest"        # 1.0 ≤ Δ < 15: efficient direct path
    PHANTOM = "phantom"      # 15 ≤ Δ < 25: inefficient spin/detour
    RUPTURE = "rupture"      # Δ ≥ 25: impossible/blocked


@dataclass
class WalkerResult:
    """Result of walker integration along u_axis."""
    # [N] — integrated work along projected path
    work_integral: torch.Tensor
    # [N] — divergence ratio Δ = W / d_spectral (terrain-invariant)
    divergence_ratio: torch.Tensor
    # [N] — spectral distance (straight-line in embedding space)
    spectral_distance: torch.Tensor
    # [N] — walker state classification
    state: torch.Tensor  # 0=honest, 1=phantom, 2=rupture
    # [N] — final walker position (mean across walkers)
    final_position: torch.Tensor
    # [N, D] — projected trajectory endpoint in RKS space
    trajectory_endpoint: torch.Tensor
    # Thresholds used for classification (now Δ-based)
    honest_threshold: float   # Δ < this = honest
    rupture_threshold: float  # Δ > this = rupture


class SemanticWalker:
    """
    MCMC walker that explores the bot-weight simplex.

    ASTER v3.2 RIEMANNIAN WEB ROUTER
    ================================
    Pure slope/gravity physics. NO WIND.

    The walker feels only the terrain gradient (∇Φ), not an external force.
    This is the correct physical model: gravity pulls downhill, friction
    resists motion, but there's no "wind" pushing the walker around.

    Physics Model:
    - Work integral W = ∫ (1/ρ) ds  (pure terrain friction)
    - Divergence ratio Δ = W / d_spectral (efficiency metric)
    - Classification based on Δ: TAUTOLOGY / HONEST / PHANTOM / RUPTURE

    The u_axis from Track 1.5 is used for:
    - Computing directional energy (projection onto compass)
    - Starting walkers at opposing poles
    - NOT as a "wind" force (that was the legacy bug)
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        gradients: torch.Tensor,
        rks_basis,
        temperature: float = 0.5,
        u_axis: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            embeddings: [8, H] — positions of the 8 bots in embedding space
            gradients: [8, H] — gradient vectors (deviation from mean)
            rks_basis: SharedRKSBasis — projector to map trajectory to geometry
            temperature: Plasticity. Higher = walkers climb walls easier.
            u_axis: [H] — gauge-fixed compass direction from Track 1.5.
                   Used for directional energy and pole initialization.
                   NOT used as wind (legacy bug removed).
        """
        self.embeddings = embeddings.float()
        self.gradients = gradients.float()
        self.kernel = rks_basis
        self.T = temperature
        self.n_bots = embeddings.shape[0]
        self.u_axis = u_axis.float() if u_axis is not None else None

        # DIVERGENCE RATIO (Δ) THRESHOLDS for state classification
        # ASTER v3.2 PATH SYSTEM: Pure slope physics, no wind
        #
        # Δ = W_actual / d_spectral (terrain-invariant efficiency)
        #
        # The 4-State Classification:
        # - TAUTOLOGY: Δ < 1.0 or d_spectral ≈ 0 (null path, no displacement)
        # - HONEST: 1.0 ≤ Δ < 15.0 (direct geodesic, efficient)
        # - PHANTOM: 15.0 ≤ Δ < 25.0 (high-energy loop, spin)
        # - RUPTURE: Δ ≥ 25.0 (impossible, topological barrier)
        #
        # Calibrated for W = ∫ (1/ρ) ds in 256-1536D embedding space.
        self.tautology_threshold = 1.0   # Below = tautology (null path)
        self.honest_threshold = 15.0     # Below = honest (efficient)
        self.rupture_threshold = 25.0    # Above = rupture (blocked)

        # Spectral distance threshold for tautology detection
        self.min_spectral_distance = 0.1

    def _compute_energy(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute energy at a point in the bot-weight simplex.

        If u_axis is provided, energy = |projection onto u_axis|.
        Otherwise, energy = norm of fused gradient (legacy mode).
        """
        # Linear mix of gradients based on walker position
        fused_grad = torch.matmul(weights, self.gradients)  # [..., H]

        if self.u_axis is not None:
            # Project onto compass direction: energy = |fused_grad · u_axis|
            # High energy = strong alignment with polarization axis
            energy = torch.abs((fused_grad * self.u_axis).sum(dim=-1))
        else:
            # Legacy: energy = norm of fused gradient
            energy = torch.norm(fused_grad, p=2, dim=-1)

        return energy

    def _compute_potential(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D potential along u_axis.

        Potential = signed projection of fused embedding onto u_axis.
        Positive pole = high potential, negative pole = low potential.
        """
        if self.u_axis is None:
            # No u_axis: use energy as potential
            return self._compute_energy(weights)

        # Fused embedding position
        fused_emb = torch.matmul(weights, self.embeddings)  # [..., H]
        # Signed projection onto compass
        potential = (fused_emb * self.u_axis).sum(dim=-1)
        return potential

    def run_swarm(
        self,
        n_walkers: int = 100,
        n_steps: int = 50,
        start_seed: Optional[int] = None,
        start_from_poles: bool = False,
    ) -> torch.Tensor:
        """
        Release a swarm of walkers to explore the simplex.

        Args:
            n_walkers: Number of parallel walkers
            n_steps: Number of MCMC steps
            start_seed: Random seed for reproducibility
            start_from_poles: If True, half start from positive pole, half from negative

        Returns:
            trajectory_weights: [n_steps+1, n_walkers, n_bots] — weight paths
        """
        if start_seed is not None:
            torch.manual_seed(start_seed)

        if start_from_poles and self.u_axis is not None:
            # Start half walkers at positive pole, half at negative
            # Determine pole bots by projecting bot gradients onto u_axis
            bot_projections = (self.gradients * self.u_axis).sum(dim=-1)  # [8]
            pos_bot = bot_projections.argmax().item()
            neg_bot = bot_projections.argmin().item()

            # Create starting weights
            current_weights = torch.zeros(n_walkers, self.n_bots)
            half = n_walkers // 2
            # Positive pole starters
            current_weights[:half, pos_bot] = 0.8
            current_weights[:half] += 0.2 / self.n_bots
            # Negative pole starters
            current_weights[half:, neg_bot] = 0.8
            current_weights[half:] += 0.2 / self.n_bots
            current_weights = current_weights / current_weights.sum(dim=-1, keepdim=True)
        else:
            # Spawn at center (neutral uniform weights)
            current_weights = torch.ones(n_walkers, self.n_bots) / self.n_bots

        current_energy = self._compute_energy(current_weights)

        # History: [Steps, Walkers, Weights]
        trajectory_weights = [current_weights.clone()]
        trajectory_energies = [current_energy.clone()]

        # The Walk Loop
        for t in range(n_steps):
            # A. Propose a step (perturb weights)
            noise = torch.randn_like(current_weights) * 0.1
            proposal = torch.abs(current_weights + noise)
            proposal = proposal / proposal.sum(dim=-1, keepdim=True)

            # B. Check the wall (calculate energy)
            proposal_energy = self._compute_energy(proposal)

            # C. Metropolis criterion
            delta_E = proposal_energy - current_energy
            acceptance_prob = torch.exp(-delta_E / self.T)
            dice_roll = torch.rand(n_walkers)
            mask_accept = dice_roll < acceptance_prob

            # D. Update state
            mask_expanded = mask_accept.unsqueeze(-1).expand_as(current_weights)
            current_weights = torch.where(mask_expanded, proposal, current_weights)
            current_energy = torch.where(mask_accept, proposal_energy, current_energy)

            trajectory_weights.append(current_weights.clone())
            trajectory_energies.append(current_energy.clone())

        return torch.stack(trajectory_weights)

    def _compute_density(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute local semantic density ρ at a point in the simplex.

        Higher density = more argument support = easier traversal.
        Density is inversely related to gradient magnitude (high gradient = low density).

        SAFETY: Density is clamped to a minimum epsilon (1e-6) to prevent
        singularities (infinite cost) when the walker steps into a void.
        This forces "High Cost" paths rather than crashes.

        Args:
            weights: [..., n_bots] position in bot-weight simplex

        Returns:
            density: [...] local density (> epsilon, always finite)
        """
        DENSITY_EPSILON = 1e-6  # Minimum density to prevent 1/ρ → ∞

        # Fused gradient magnitude at this position
        fused_grad = torch.matmul(weights, self.gradients)  # [..., H]
        grad_magnitude = torch.norm(fused_grad, p=2, dim=-1)  # [...]
        # Density = 1 / (1 + grad_magnitude) — bounded (0, 1]
        density = 1.0 / (1.0 + grad_magnitude)

        # SAFETY CLAMP: Prevent singularities
        # If density → 0, clamp to epsilon so Cost = 1/ε = 1,000,000 (high but finite)
        density = torch.clamp(density, min=DENSITY_EPSILON)

        return density

    def compute_work_integral(
        self,
        trajectory_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RIEMANNIAN WEB ROUTER — Pure Slope Physics
        Computes Work (W), Spectral Distance (d), and Divergence Ratio (Δ).

        NO WIND. Only terrain friction (1/ρ).

        Physics Model:
        ==============
        W = ∫ (1/ρ) ds

        Where:
        - 1/ρ = terrain friction (inverse semantic density)
        - ds = infinitesimal path length

        This is gravity/slope physics:
        - High density (ρ → 1): easy traverse, low friction
        - Low density (ρ → 0): hard traverse, high friction (void)
        - The walker feels the terrain, not an external wind

        Divergence Ratio (Δ) = W / d_spectral:
        - Δ < 1.0: TAUTOLOGY (spinning in place)
        - 1.0 ≤ Δ < 15: HONEST (direct geodesic)
        - 15 ≤ Δ < 25: PHANTOM (high-energy loop)
        - Δ ≥ 25: RUPTURE (blocked)

        Args:
            trajectory_weights: [n_steps+1, n_walkers, n_bots]

        Returns:
            work: [n_walkers] — integrated work per walker (W_actual)
            path_length: [n_walkers] — total path length in embedding space
            spectral_distance: [n_walkers] — straight-line distance start→end
            divergence_ratio: [n_walkers] — Δ = W / d_spectral
        """
        n_steps_plus_1, n_walkers, n_bots = trajectory_weights.shape

        # =========================================
        # 1. GEOMETRY: Positions in Embedding Space
        # =========================================
        positions = torch.matmul(trajectory_weights, self.embeddings)

        # Step-wise displacements [n_steps, n_walkers, H]
        deltas = positions[1:] - positions[:-1]

        # Step-wise distances (Euclidean arc length ds)
        step_distances = torch.norm(deltas, p=2, dim=-1)  # [n_steps, n_walkers]

        # =========================================
        # 2. TERRAIN FRICTION: Cost = 1/ρ (no wind!)
        # =========================================
        midpoint_weights = (trajectory_weights[1:] + trajectory_weights[:-1]) / 2
        density = self._compute_density(midpoint_weights)  # [n_steps, n_walkers]
        terrain_friction = 1.0 / density.clamp(min=1e-9)  # Inverse density

        # =========================================
        # 3. WORK INTEGRAL: W = ∫ (1/ρ) ds
        # =========================================
        # Pure slope physics — no wind term
        step_work = terrain_friction * step_distances
        work = step_work.sum(dim=0)  # [n_walkers]
        path_length = step_distances.sum(dim=0)  # [n_walkers]

        # =========================================
        # 4. DIVERGENCE RATIO: Δ = W / d_spectral
        # =========================================
        # Spectral distance = straight-line Euclidean from start to end
        start_pos = positions[0]   # [n_walkers, H]
        end_pos = positions[-1]    # [n_walkers, H]
        spectral_distance = torch.norm(end_pos - start_pos, p=2, dim=-1)

        # Δ = W / d_spectral (terrain-invariant efficiency metric)
        EPSILON = 1e-6
        divergence_ratio = work / (spectral_distance + EPSILON)

        return work, path_length, spectral_distance, divergence_ratio

    def classify_state(
        self,
        divergence_ratio: torch.Tensor,
        spectral_distance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify walker state based on DIVERGENCE RATIO (Δ).

        ASTER v3.2 PATH SYSTEM — 4-State Classification:
        =================================================

        State 0: TAUTOLOGY (Δ < 1.0 or d_spectral ≈ 0)
          Null path. Walker didn't actually move. Spinning in place.
          Example: "Israel says Israel is right" — no semantic displacement.

        State 1: HONEST (1.0 ≤ Δ < honest_threshold)
          Direct geodesic path. May climb steep terrain, but efficient.
          The "hard truth" — controversial but substantive.

        State 2: PHANTOM (honest_threshold ≤ Δ < rupture_threshold)
          High-energy loop/detour. Creates fake path via spin.
          Captures the GEOMETRY of evasion, not just cost.

        State 3: RUPTURE (Δ ≥ rupture_threshold)
          Impossible path. Topological barrier. Walker hit void.

        Args:
            divergence_ratio: [n_walkers] — Δ = W / d_spectral
            spectral_distance: [n_walkers] — straight-line displacement

        Returns:
            state: [n_walkers] — 0=tautology, 1=honest, 2=phantom, 3=rupture
        """
        # Default: HONEST (state 1)
        state = torch.ones_like(divergence_ratio, dtype=torch.long)

        # State 0: TAUTOLOGY (Δ < 1.0 or d_spectral too small)
        is_tautology = (divergence_ratio < self.tautology_threshold) | \
                       (spectral_distance < self.min_spectral_distance)
        state = torch.where(is_tautology, torch.zeros_like(state), state)

        # State 2: PHANTOM (honest_threshold ≤ Δ < rupture_threshold)
        state = torch.where(
            divergence_ratio >= self.honest_threshold,
            2 * torch.ones_like(state),
            state
        )

        # State 3: RUPTURE (Δ ≥ rupture_threshold)
        state = torch.where(
            divergence_ratio >= self.rupture_threshold,
            3 * torch.ones_like(state),
            state
        )

        return state

    def project_trajectory(self, trajectory_weights: torch.Tensor) -> torch.Tensor:
        """
        Map the weight path (simplex) to the geometry path (RKS manifold).

        Args:
            trajectory_weights: [n_steps+1, n_walkers, n_bots]

        Returns:
            projected: [n_steps+1, n_walkers, D] — RKS-projected positions
        """
        steps, walkers, bots = trajectory_weights.shape
        flat_weights = trajectory_weights.view(-1, bots)
        fused_emb = torch.matmul(flat_weights, self.embeddings)
        projected = self.kernel(fused_emb)
        return projected.view(steps, walkers, -1)

    def run_full_integration(
        self,
        n_walkers: int = 20,
        n_steps: int = 10,
        start_seed: Optional[int] = None,
    ) -> WalkerResult:
        """
        Run complete walker integration and return structured result.

        This is the main entry point for Track 4 integration.

        ASTER v3.2 CALIBRATION:
        Now returns divergence_ratio (Δ) for terrain-invariant classification.
        The state classification uses Δ instead of raw work W.

        Returns:
            WalkerResult with work integral, divergence ratio, state, and trajectory info
        """
        # Run the swarm
        trajectory = self.run_swarm(
            n_walkers=n_walkers,
            n_steps=n_steps,
            start_seed=start_seed,
            start_from_poles=self.u_axis is not None,
        )

        # Compute work integral with calibration metrics
        work, path_length, spectral_distance, divergence_ratio = self.compute_work_integral(trajectory)

        # Classify states using DIVERGENCE RATIO and SPECTRAL DISTANCE
        # Now includes TAUTOLOGY detection (4-state system)
        states = self.classify_state(divergence_ratio, spectral_distance)

        # Project trajectory endpoint
        projected = self.project_trajectory(trajectory)
        endpoint = projected[-1].mean(dim=0)  # [D] mean across walkers

        # Final position in embedding space
        final_weights = trajectory[-1].mean(dim=0)  # [n_bots]
        final_position = torch.matmul(final_weights, self.embeddings)  # [H]

        # Aggregate: mean values across walkers
        mean_work = work.mean()
        mean_divergence = divergence_ratio.mean()
        mean_spectral_dist = spectral_distance.mean()
        majority_state = states.mode().values

        return WalkerResult(
            work_integral=mean_work,
            divergence_ratio=mean_divergence,
            spectral_distance=mean_spectral_dist,
            state=majority_state,
            final_position=final_position,
            trajectory_endpoint=endpoint,
            honest_threshold=self.honest_threshold,
            rupture_threshold=self.rupture_threshold,
        )


def compute_walker_resistance(
    cls_per_bot: torch.Tensor,
    rks_basis,
    u_axis: Optional[torch.Tensor] = None,
    temperature: float = 0.5,
    n_walkers: int = 20,
    n_steps: int = 10,
) -> Dict[str, Any]:
    """
    Compute walker resistance for a single article.

    ASTER v3.2 RIEMANNIAN WEB ROUTER
    ================================
    Pure slope physics. NO WIND.

    Returns divergence_ratio (Δ) for terrain-invariant classification.
    Uses 4-state system: TAUTOLOGY / HONEST / PHANTOM / RUPTURE.

    Args:
        cls_per_bot: [8, H] — per-bot embeddings
        rks_basis: SharedRKSBasis for projection
        u_axis: [H] — compass direction from Track 1.5 (for pole init, NOT wind)
        temperature: Walker plasticity
        n_walkers: Number of MCMC walkers
        n_steps: Number of steps per walker

    Returns:
        Dict with:
        - work_integral: Raw work W (terrain-sensitive)
        - divergence_ratio: Δ = W / d_spectral (terrain-invariant)
        - spectral_distance: Straight-line distance start→end
        - state: "tautology" / "honest" / "phantom" / "rupture"
        - state_code: 0 / 1 / 2 / 3
        - walker_output: [D] trajectory endpoint for Track 5
        - final_position: [H] final position in embedding space
    """
    # Compute gradients as deviation from mean
    bot_grads = cls_per_bot - cls_per_bot.mean(dim=0, keepdim=True)

    explorer = SemanticWalker(
        embeddings=cls_per_bot,
        gradients=bot_grads,
        rks_basis=rks_basis,
        temperature=temperature,
        u_axis=u_axis,
    )

    result = explorer.run_full_integration(
        n_walkers=n_walkers,
        n_steps=n_steps,
    )

    # 4-state system: TAUTOLOGY / HONEST / PHANTOM / RUPTURE
    state_names = ["tautology", "honest", "phantom", "rupture"]
    state_code = result.state.item()
    state_name = state_names[min(state_code, 3)]  # Safety clamp

    return {
        "work_integral": result.work_integral.item(),
        "divergence_ratio": result.divergence_ratio.item(),
        "spectral_distance": result.spectral_distance.item(),
        "state": state_name,
        "state_code": state_code,
        "walker_output": result.trajectory_endpoint,
        "final_position": result.final_position,
        # Thresholds for reference
        "tautology_threshold": explorer.tautology_threshold,
        "honest_threshold": explorer.honest_threshold,
        "rupture_threshold": explorer.rupture_threshold,
    }
