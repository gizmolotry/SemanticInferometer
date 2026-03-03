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
- Classifies: TAUTOLOGY / HONEST / PHANTOM / RUPTURE (4-state system)

ASTER v3.2 HYSTERESIS (Path Memory):
=====================================
True ant-colony / pheromone trail logic for TIME-DEPENDENT LOGIC.

When enable_hysteresis=True:
- Paths that are traversed become easier to traverse again (rut formation)
- Memory decays over time (the "snow fills back in")
- Creates "Highways" over repeated swarm traversals
- Return path B→A may have different cost than A→B (Geometric Hysteresis)

Physics Model (Dissipative Langevin with Memory):
    dv = -γv dt - ∇V(x) dt + σdW_t
    V_effective = V_static * exp(-memory * sensitivity)

Parameters:
- memory_decay (0.95): How fast memory fades per step
- reinforcement_rate (0.1): How much each traversal reinforces the path
- memory_sensitivity (1.0): How strongly memory affects energy costs

The memory_tensor[i,j] tracks "rut depth" for bot_i → bot_j transitions.
High memory = low effective energy barrier = easier traversal.

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
from .thermo_config import ThermodynamicConfig


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
    state: torch.Tensor  # 0=tautology, 1=honest, 2=phantom, 3=rupture
    # [N] — final walker position (mean across walkers)
    final_position: torch.Tensor
    # [N, D] — projected trajectory endpoint in RKS space
    trajectory_endpoint: torch.Tensor
    # Thresholds used for classification (now Δ-based)
    honest_threshold: float   # Δ < this = honest
    rupture_threshold: float  # Δ > this = rupture
    # HYSTERESIS (Path Memory) statistics
    hysteresis_stats: Optional[Dict[str, Any]] = None
    memory_matrix: Optional[torch.Tensor] = None  # [n_bots, n_bots] rut depths
    # Energy Tank (Track 4 Thermodynamics)
    energy_survival_rate: Optional[float] = None  # Fraction of walkers that didn't exhaust budget


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

    ASTER v3.2 HYSTERESIS (Path Memory):
    ====================================
    True ant-colony / pheromone trail logic. Paths that are traversed
    become easier to traverse again (memory reinforcement). Memory decays
    over time (the "snow fills back in").

    This creates TIME-DEPENDENT LOGIC:
    - Return path B→A may have different cost than A→B
    - Swarms form "Highways" over repeated traversals
    - Geometric Hysteresis = path history affects future costs
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        gradients: torch.Tensor,
        rks_basis,
        temperature: float = 0.5,
        u_axis: Optional[torch.Tensor] = None,
        anisotropy_strength: float = 2.0,
        # Hysteresis parameters (ASTER v3.2)
        enable_hysteresis: bool = True,
        memory_decay: float = 0.95,
        reinforcement_rate: float = 0.1,
        memory_sensitivity: float = 1.0,
        thermo_config: Optional[ThermodynamicConfig] = None,
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
            enable_hysteresis: Enable path memory (ant colony pheromones)
            memory_decay: How fast memory fades (0.95 = 5% decay per step)
            reinforcement_rate: How much each traversal reinforces the path
            memory_sensitivity: How strongly memory affects energy costs
        """
        self.embeddings = embeddings.float()
        self.gradients = gradients.float()
        self.kernel = rks_basis
        self.T = temperature
        self.n_bots = embeddings.shape[0]
        self.u_axis = u_axis.float() if u_axis is not None else None
        self.thermo_config = thermo_config or ThermodynamicConfig()

        # =========================================
        # HYSTERESIS STATE (Path Memory)
        # =========================================
        # Memory tensor tracks "rut depth" for each bot→bot transition
        # Shape: [n_bots, n_bots] — transition matrix
        # memory[i,j] = accumulated traversals from bot_i → bot_j
        self.enable_hysteresis = enable_hysteresis
        self.memory_decay = memory_decay
        self.reinforcement_rate = reinforcement_rate
        self.memory_sensitivity = memory_sensitivity

        # Initialize memory as zeros (fresh snow, no ruts)
        self.memory_tensor = torch.zeros(self.n_bots, self.n_bots)

        # Track hysteresis statistics
        self.hysteresis_stats = {
            "total_reinforcements": 0,
            "max_rut_depth": 0.0,
            "highway_count": 0,  # Paths with memory > threshold
        }

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
        self.tautology_threshold = self.thermo_config.tautology_work_threshold
        self.honest_threshold = 15.0     # Below = honest (efficient)
        self.rupture_threshold = 25.0    # Above = rupture (blocked)

        # Spectral distance threshold for tautology detection
        self.min_spectral_distance = self.thermo_config.tautology_disp_threshold

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

    def _compute_transition_memory(
        self,
        prev_weights: torch.Tensor,  # [n_walkers, n_bots]
        next_weights: torch.Tensor,  # [n_walkers, n_bots]
    ) -> torch.Tensor:
        """
        Compute memory bonus for a transition (pheromone trail strength).

        HYSTERESIS LOGIC:
        The memory tensor tracks "rut depth" for bot→bot transitions.
        When walkers shift weight from bot_i to bot_j, they benefit from
        previous traversals along that direction.

        Returns:
            memory_bonus: [n_walkers] — accumulated memory along transition direction
        """
        if not self.enable_hysteresis:
            return torch.zeros(prev_weights.shape[0])

        # Compute weight deltas: which bots gained/lost weight?
        deltas = next_weights - prev_weights  # [n_walkers, n_bots]

        # Identify sources (lost weight) and sinks (gained weight)
        # Use outer product to get transition matrix per walker
        # sources[i] * sinks[j] = strength of i→j transition

        # Clamp to get only positive changes
        sources = torch.clamp(-deltas, min=0)  # [n_walkers, n_bots] weight lost
        sinks = torch.clamp(deltas, min=0)      # [n_walkers, n_bots] weight gained

        # Transition strength: outer product sources × sinks
        # For each walker: [n_bots] × [n_bots] → [n_bots, n_bots]
        # Then dot with memory_tensor to get total memory bonus

        # Efficient batch computation:
        # memory_bonus[w] = sum_{i,j} sources[w,i] * sinks[w,j] * memory[i,j]
        # = sources[w] @ memory @ sinks[w].T (but we want scalar per walker)

        # Reshape for batch matmul: [n_walkers, 1, n_bots] @ [n_bots, n_bots] @ [n_walkers, n_bots, 1]
        memory_bonus = torch.einsum('wi,ij,wj->w', sources, self.memory_tensor, sinks)

        return memory_bonus

    def _update_memory(
        self,
        prev_weights: torch.Tensor,  # [n_walkers, n_bots]
        next_weights: torch.Tensor,  # [n_walkers, n_bots]
        accepted_mask: torch.Tensor,  # [n_walkers] bool
    ) -> None:
        """
        Reinforce memory for accepted transitions (pheromone deposit).

        HYSTERESIS UPDATE:
        When walkers successfully traverse from prev → next, we strengthen
        the memory of that transition. This makes future traversals easier.

        The memory decays globally each step (snow fills back in).
        """
        if not self.enable_hysteresis:
            return

        # Compute weight deltas for accepted transitions only
        deltas = next_weights - prev_weights  # [n_walkers, n_bots]

        # Mask out rejected transitions
        accepted_deltas = deltas * accepted_mask.unsqueeze(-1).float()

        # Sources and sinks
        sources = torch.clamp(-accepted_deltas, min=0)  # Weight lost
        sinks = torch.clamp(accepted_deltas, min=0)      # Weight gained

        # Reinforcement: sum over walkers of outer(sources, sinks)
        # reinforcement[i,j] = sum_w sources[w,i] * sinks[w,j]
        reinforcement = torch.einsum('wi,wj->ij', sources, sinks)

        # Apply reinforcement
        self.memory_tensor += self.reinforcement_rate * reinforcement

        # Decay entire memory (the "snow fills back in")
        self.memory_tensor *= self.memory_decay

        # Update statistics
        self.hysteresis_stats["total_reinforcements"] += int(accepted_mask.sum().item())
        self.hysteresis_stats["max_rut_depth"] = float(self.memory_tensor.max().item())
        self.hysteresis_stats["highway_count"] = int((self.memory_tensor > 0.5).sum().item())

    def reset_memory(self) -> None:
        """Reset the memory tensor (fresh snow)."""
        self.memory_tensor.zero_()
        self.hysteresis_stats = {
            "total_reinforcements": 0,
            "max_rut_depth": 0.0,
            "highway_count": 0,
        }

    def get_memory_matrix(self) -> torch.Tensor:
        """Return the current memory tensor (for visualization)."""
        return self.memory_tensor.clone()

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Release a swarm of walkers to explore the simplex.

        Args:
            n_walkers: Number of parallel walkers
            n_steps: Number of MCMC steps
            start_seed: Random seed for reproducibility
            start_from_poles: If True, half start from positive pole, half from negative

        Returns:
            trajectory_weights: [n_steps+1, n_walkers, n_bots] — weight paths
            broken_mask: [n_walkers] True where kinetic energy was exhausted before target
            reached_target: [n_walkers] True where walker reached target region
        """
        if start_seed is not None:
            torch.manual_seed(start_seed)

        target_weights = None
        target_reach_threshold = 0.25

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

            # Target for each walker is the opposite pole.
            target_weights = torch.zeros_like(current_weights)
            target_weights[:half, neg_bot] = 0.8
            target_weights[:half] += 0.2 / self.n_bots
            target_weights[half:, pos_bot] = 0.8
            target_weights[half:] += 0.2 / self.n_bots
            target_weights = target_weights / target_weights.sum(dim=-1, keepdim=True)
        else:
            # Spawn at center (neutral uniform weights)
            current_weights = torch.ones(n_walkers, self.n_bots) / self.n_bots

        current_energy = self._compute_energy(current_weights)
        energy_tank = torch.full(
            (n_walkers,),
            float(self.thermo_config.walker_energy_budget),
            dtype=current_weights.dtype,
            device=current_weights.device,
        )
        broken_mask = torch.zeros(n_walkers, dtype=torch.bool, device=current_weights.device)
        reached_target = torch.zeros(n_walkers, dtype=torch.bool, device=current_weights.device)

        # History: [Steps, Walkers, Weights]
        trajectory_weights = [current_weights.clone()]
        trajectory_energies = [current_energy.clone()]

        # The Walk Loop (with Hysteresis)
        for t in range(n_steps):
            # A. Propose a step (perturb weights)
            noise = torch.randn_like(current_weights) * self.thermo_config.noise_sigma
            
            # RESTORE WIND FORCE: Push walkers across the manifold using the u_axis field.
            # We guide walkers towards their opposing pole target if defined by the field.
            wind_drift = 0
            if self.u_axis is not None and target_weights is not None:
                wind_drift = (target_weights - current_weights) * 0.1

            proposal = torch.abs(current_weights + noise + wind_drift)
            proposal = proposal / proposal.sum(dim=-1, keepdim=True)

            # B. Check the wall (calculate energy)
            proposal_energy = self._compute_energy(proposal)

            # C. Metropolis criterion with HYSTERESIS DISCOUNT
            delta_E = proposal_energy - current_energy

            # HYSTERESIS: Memory discounts the effective energy barrier
            # Traversed paths have lower effective cost (ruts in the snow)
            if self.enable_hysteresis:
                memory_bonus = self._compute_transition_memory(current_weights, proposal)
                # Effective delta_E is reduced by memory (easier to follow ruts)
                effective_delta_E = delta_E * torch.exp(-memory_bonus * self.memory_sensitivity)
            else:
                effective_delta_E = delta_E

            acceptance_prob = torch.exp(-effective_delta_E / self.T)
            dice_roll = torch.rand(n_walkers)
            mask_accept = dice_roll < acceptance_prob

            # Dead walkers cannot move any further.
            mask_accept = mask_accept & (~broken_mask)

            # D. Update memory BEFORE state (reinforce accepted transitions)
            if self.enable_hysteresis:
                self._update_memory(current_weights, proposal, mask_accept)

            # E. Update state
            mask_expanded = mask_accept.unsqueeze(-1).expand_as(current_weights)
            next_weights = torch.where(mask_expanded, proposal, current_weights)
            next_energy = torch.where(mask_accept, proposal_energy, current_energy)

            # Kinetic energy tank: deduct terrain-friction work for accepted moves.
            current_pos = torch.matmul(current_weights, self.embeddings)
            next_pos = torch.matmul(next_weights, self.embeddings)
            step_distance = torch.norm(next_pos - current_pos, p=2, dim=-1)

            midpoint_weights = (current_weights + next_weights) / 2.0
            local_density = self._compute_density(midpoint_weights)
            local_friction = self.thermo_config.friction_coefficient / local_density.clamp(
                min=self.thermo_config.density_clamp_min
            )
            energy_cost = local_friction * step_distance
            energy_tank = energy_tank - torch.where(mask_accept, energy_cost, torch.zeros_like(energy_cost))

            if target_weights is not None:
                to_target = torch.norm(next_weights - target_weights, p=2, dim=-1)
                reached_target = reached_target | (to_target <= target_reach_threshold)

            newly_broken = (energy_tank <= 0) & (~reached_target)
            broken_mask = broken_mask | newly_broken

            current_weights = next_weights
            current_energy = next_energy

            trajectory_weights.append(current_weights.clone())
            trajectory_energies.append(current_energy.clone())

        return torch.stack(trajectory_weights), broken_mask, reached_target

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
        DENSITY_EPSILON = self.thermo_config.density_clamp_min

        # Fused gradient magnitude at this position
        fused_grad = torch.matmul(weights, self.gradients)  # [..., H]
        grad_magnitude = torch.norm(fused_grad, p=2, dim=-1)  # [...]
        # Clamp gradient norm to prevent instability
        grad_magnitude = torch.clamp(grad_magnitude, max=1000.0)
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
        # 2. TERRAIN FRICTION: Cost = 1/rho (no wind!)
        # =========================================
        midpoint_weights = (trajectory_weights[1:] + trajectory_weights[:-1]) / 2
        density = self._compute_density(midpoint_weights)  # [n_steps, n_walkers]

        # Physical friction model: inverse density.
        # Clamp to keep extreme voids finite and avoid INF work explosions.
        DENSITY_EPSILON = self.thermo_config.density_clamp_min
        MAX_FRICTION = 1e3
        terrain_friction = (
            self.thermo_config.friction_coefficient / density.clamp(min=DENSITY_EPSILON)
        ).clamp(max=MAX_FRICTION)

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
        Classify walker state based on ADAPTIVE PERCENTILES of efficiency.

        ASTER v3.2 PATH SYSTEM — 4-State Classification (ADAPTIVE):
        ===========================================================

        Efficiency = displacement / path_length (1.0 = perfect straight line)
        We compute this as: spectral_distance / divergence_ratio
        (since divergence_ratio = work / spectral_distance ≈ path_length / spectral_distance)

        State 0: TAUTOLOGY (Bottom 10% efficiency or d_spectral ≈ 0)
          Null path. Walker didn't actually move. Spinning in place.
          Example: "Israel says Israel is right" — no semantic displacement.

        State 1: HONEST (Top 20% efficiency, above 80th percentile)
          Direct geodesic path. May climb steep terrain, but efficient.
          The "hard truth" — controversial but substantive.

        State 2: PHANTOM (Middle range, between tautology and honest)
          High-energy loop/detour. Creates fake path via spin.
          Captures the GEOMETRY of evasion, not just cost.

        State 3: RUPTURE (Detected separately via stalls or spectral distance)
          Impossible path. Topological barrier. Walker hit void.

        Args:
            divergence_ratio: [n_walkers] — Δ = W / d_spectral
            spectral_distance: [n_walkers] — straight-line displacement

        Returns:
            state: [n_walkers] — 0=tautology, 1=honest, 2=phantom, 3=rupture
        """
        # Compute efficiency: displacement / path_length (higher = more efficient)
        # Since divergence_ratio = work/spectral ≈ path/spectral,
        # efficiency ≈ spectral / work = 1 / divergence_ratio
        EPSILON = 1e-8
        efficiency = spectral_distance / (divergence_ratio + EPSILON)

        # Adaptive thresholds from batch distribution
        # Tautology: Bottom 10% efficiency (least efficient, spinning in place)
        taut_thresh = torch.quantile(efficiency, 0.10)
        # Honest: Top 20% efficiency (most efficient, direct paths)
        honest_thresh = torch.quantile(efficiency, 0.80)

        # Default: PHANTOM (middle range, state 2)
        state = 2 * torch.ones_like(efficiency, dtype=torch.long)

        # State 0: TAUTOLOGY (bottom 10% efficiency OR no displacement)
        is_tautology = (efficiency <= taut_thresh) | \
                       (spectral_distance < self.min_spectral_distance)
        state = torch.where(is_tautology, torch.zeros_like(state), state)

        # State 1: HONEST (top 20% efficiency)
        state = torch.where(efficiency >= honest_thresh, torch.ones_like(state), state)

        # State 3: RUPTURE (adaptive tail of divergence distribution).
        # Use per-swarm high-divergence tail so ruptures remain detectable even when
        # absolute delta scales shift across datasets.
        rupture_quantile = torch.quantile(divergence_ratio, 0.90)
        rupture_cutoff = min(float(self.rupture_threshold), float(rupture_quantile))
        is_rupture = divergence_ratio >= rupture_cutoff
        state = torch.where(is_rupture, 3 * torch.ones_like(state), state)

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
        # Run the swarm and track per-walker kinetic failures.
        trajectory, broken_mask, reached_target = self.run_swarm(
            n_walkers=n_walkers,
            n_steps=n_steps,
            start_seed=start_seed,
            start_from_poles=self.u_axis is not None,
        )

        # Compute work integral with calibration metrics
        work, path_length, spectral_distance, divergence_ratio = self.compute_work_integral(trajectory)

        # Topological death: walker ran max_steps and stayed near origin.
        trapped_distance_threshold = self.thermo_config.tautology_disp_threshold
        trapped_mask = (~broken_mask) & (~reached_target) & (spectral_distance < trapped_distance_threshold)
        survived = (~broken_mask) & (~trapped_mask)
        energy_survival_rate = float(survived.float().mean())

        # Classify states using DIVERGENCE RATIO and SPECTRAL DISTANCE
        # Now includes TAUTOLOGY detection (4-state system)
        states = self.classify_state(divergence_ratio, spectral_distance)

        # Kinetic death => RUPTURE bucket in 4-state map.
        states = torch.where(broken_mask, torch.tensor(3, dtype=torch.long, device=states.device), states)

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
        # Aggregate walker states to an article-level state.
        # Preserve rupture evidence instead of erasing it via pure majority vote.
        broken_fraction = broken_mask.float().mean()
        trapped_fraction = trapped_mask.float().mean()
        rupture_fraction = (states == 3).float().mean()
        if broken_fraction >= 0.20:
            majority_state = torch.tensor(4, dtype=torch.long, device=states.device)  # broken
        elif trapped_fraction >= 0.20:
            majority_state = torch.tensor(5, dtype=torch.long, device=states.device)  # trapped
        elif rupture_fraction >= 0.20:
            majority_state = torch.tensor(3, dtype=torch.long, device=states.device)
        else:
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
            # HYSTERESIS: Include path memory state
            hysteresis_stats=self.hysteresis_stats.copy() if self.enable_hysteresis else None,
            memory_matrix=self.get_memory_matrix() if self.enable_hysteresis else None,
            # ENERGY TANK: fraction of walkers that survived the terrain
            energy_survival_rate=energy_survival_rate,
        )


def compute_walker_resistance(
    cls_per_bot: torch.Tensor,
    rks_basis,
    u_axis: Optional[torch.Tensor] = None,
    temperature: float = 0.5,
    n_walkers: int = 20,
    n_steps: int = 10,
    # HYSTERESIS parameters (ASTER v3.2)
    enable_hysteresis: bool = True,
    memory_decay: float = 0.95,
    reinforcement_rate: float = 0.1,
    memory_sensitivity: float = 1.0,
    existing_memory: Optional[torch.Tensor] = None,  # [n_bots, n_bots] to continue from
    thermo_config: Optional[ThermodynamicConfig] = None,
) -> Dict[str, Any]:
    """
    Compute walker resistance for a single article.

    ASTER v3.2 RIEMANNIAN WEB ROUTER
    ================================
    Pure slope physics. NO WIND.

    Returns divergence_ratio (Δ) for terrain-invariant classification.
    Uses 4-state system: TAUTOLOGY / HONEST / PHANTOM / RUPTURE.

    HYSTERESIS (Path Memory):
    When enable_hysteresis=True, the walker tracks traversed paths and
    makes them easier to traverse again (ant colony / pheromone logic).
    Pass existing_memory to continue building on previous walker runs.

    Args:
        cls_per_bot: [8, H] — per-bot embeddings
        rks_basis: SharedRKSBasis for projection
        u_axis: [H] — compass direction from Track 1.5 (for pole init, NOT wind)
        temperature: Walker plasticity
        n_walkers: Number of MCMC walkers
        n_steps: Number of steps per walker
        enable_hysteresis: Enable path memory (default True)
        memory_decay: How fast memory fades (0.95 = 5% decay per step)
        reinforcement_rate: How much each traversal reinforces the path
        memory_sensitivity: How strongly memory affects energy costs
        existing_memory: [n_bots, n_bots] memory tensor to continue from

    Returns:
        Dict with:
        - work_integral: Raw work W (terrain-sensitive)
        - divergence_ratio: Δ = W / d_spectral (terrain-invariant)
        - spectral_distance: Straight-line distance start→end
        - status: "SUCCESS" / "BROKEN" / "TRAPPED" (thermodynamic/topologic outcome)
        - state: "tautology" / "honest" / "phantom" / "rupture"
        - state_code: 0 / 1 / 2 / 3
        - walker_output: [D] trajectory endpoint for Track 5
        - final_position: [H] final position in embedding space
        - hysteresis_stats: Path memory statistics (if enabled)
        - memory_matrix: [n_bots, n_bots] final memory state (for chaining)
    """
    # Compute gradients as deviation from mean
    bot_grads = cls_per_bot - cls_per_bot.mean(dim=0, keepdim=True)

    explorer = SemanticWalker(
        embeddings=cls_per_bot,
        gradients=bot_grads,
        rks_basis=rks_basis,
        temperature=temperature,
        u_axis=u_axis,
        enable_hysteresis=enable_hysteresis,
        memory_decay=memory_decay,
        reinforcement_rate=reinforcement_rate,
        memory_sensitivity=memory_sensitivity,
        thermo_config=thermo_config,
    )

    # If continuing from existing memory, load it
    if existing_memory is not None and enable_hysteresis:
        explorer.memory_tensor = existing_memory.clone()

    result = explorer.run_full_integration(
        n_walkers=n_walkers,
        n_steps=n_steps,
    )

    # 4-state system: TAUTOLOGY / HONEST / PHANTOM / RUPTURE
    state_names = ["tautology", "honest", "phantom", "rupture", "Type 1 Rupture", "Type 2 Rupture"]
    state_code = result.state.item()
    state_name = state_names[min(state_code, len(state_names) - 1)]  # Safety clamp
    if state_name == "Type 1 Rupture":
        walker_status = "FAILED"
    elif state_name == "Type 2 Rupture":
        walker_status = "FAILED"
    elif state_name == "rupture":
        walker_status = "FAILED"
    else:
        walker_status = "SUCCESS"

    output = {
        "status": walker_status,
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

    # HYSTERESIS: Include path memory state
    if enable_hysteresis:
        output["hysteresis_stats"] = result.hysteresis_stats
        output["memory_matrix"] = result.memory_matrix  # For chaining across articles

    return output
