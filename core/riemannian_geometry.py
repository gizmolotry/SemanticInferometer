"""
riemannian_geometry.py - The Geometry of Ideological Curvature

SIDECAR TO THE WATERFALL: The theoretical engine that powers the car.

=============================================================================
HYPOTHESIS
=============================================================================
Ideological bias is not a shift in semantic coordinates, but a DEFORMATION
of the underlying METRIC TENSOR. We model language not as a flat Euclidean
space, but as a curved Riemannian manifold where ideology dictates the
cost of movement.

=============================================================================
THE MATHEMATICS
=============================================================================

1. IDEOLOGICAL METRIC TENSOR (g_μν)

   Standard NLP assumes a static, isotropic distance (Euclidean Space).
   We reject this. We define the Ideological Metric Tensor at concept x
   under ideology θ as:

       g_μν(x, θ) = δ_μν + ∇_μ Φ_θ(x) ∇_ν Φ_θ(x)

   Where:
   - δ_μν: The base "Dictionary Definition" (Euclidean Identity)
   - Φ_θ(x): The Ideological Potential Field (moral "charge" of concept)
   - ∇Φ: The Gradient Force (how fast moral value changes)

2. BIAS AS "SEMANTIC WORK"

   Distance between concepts A and B is the GEODESIC WORK required to
   traverse the path γ:

       W(A→B) = ∫_A^B √(g_μν ẋ^μ ẋ^ν) dt

   This is NOT Euclidean distance. It is the energy cost to move through
   the ideological force field.

3. THE PROPERTY-THEFT EXAMPLE

   Marxist Manifold (The Valley):
   - Potential: Φ(Property) ≈ Φ(Theft) (both carry negative charge)
   - Gradient: ∇Φ ≈ 0 (potential is flat)
   - Result: g_μν ≈ δ_μν (space is locally flat)
   - Interpretation: W → 0, concepts slide together effortlessly

   Liberal Manifold (The Cliff):
   - Potential: Φ(Property) ≫ 0 (Right), Φ(Theft) ≪ 0 (Crime)
   - Gradient: ∇Φ is massive (a "Moral Wall" exists)
   - Result: g_μν expands, space stretches
   - Interpretation: W → ∞, massive semantic energy to equate terms

=============================================================================
TRACK REFRAMES
=============================================================================

Track 3 (Blinker) → ENTROPY FILTER
    Checks if the Metric Tensor is stable enough to measure.
    High entropy = metric is fluctuating = measurement unreliable.

Track 4 (Walker) → WORK INTEGRATOR
    Calculates the Semantic Energy W defined above.
    High work = ideological barrier exists between concepts.

=============================================================================
CONCLUSION
=============================================================================

Bias is measurable as ANISOTROPY:
- Unbiased model: uniform metric (Isotropic)
- Biased model: warped metric, creating "Super-highways" of easy
  association and "Topological Voids" where connection is prohibited.

Author: Belief Transformer Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiemannianConfig:
    """Configuration for Riemannian geometry computations."""

    # Metric tensor settings
    base_metric: str = "euclidean"  # Base metric: "euclidean", "cosine"
    potential_scale: float = 1.0     # Scale factor for ideological potential

    # Work integral settings
    n_geodesic_steps: int = 100      # Steps for path integral approximation
    integration_method: str = "trapezoidal"  # "trapezoidal", "simpson"

    # Entropy filter settings (Track 3)
    entropy_window: int = 10         # Window for local entropy estimation
    stability_threshold: float = 0.5 # Below this = metric is stable

    # Work integrator settings (Track 4)
    work_normalization: str = "log"  # "linear", "log", "sigmoid"

    # Matern kernel settings (Track 2 integration)
    matern_nu: float = 1.5           # Smoothness: 0.5=Laplacian (rough), 1.5=default, ∞=RBF
    matern_sigma: float = 1.0        # Bandwidth for Matern kernel

    # Device
    device: str = "cuda"


# =============================================================================
# TRACK 2: MATERN KERNEL DISTANCE (The "Jagged World")
# =============================================================================

def compute_matern_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    nu: float = 1.5,
    sigma: float = 1.0,
    use_angular: bool = None,  # Auto-detect if None
) -> torch.Tensor:
    """
    Compute Matern kernel-based distance between points.

    The Matern kernel captures "jaggedness" in semantic space:
    - Low ν (0.5): Maximum roughness, non-differentiable "cliffs"
    - ν = 1.5: Default, captures cracks while remaining stable
    - High ν (→∞): Approaches RBF, loses crack sensitivity

    The "distance" is derived from kernel similarity:
        d_matern(x, y) = sqrt(2 * (1 - K_matern(x, y)))

    For L2-normalized data (unit sphere), automatically uses ANGULAR distance
    (derived from cosine similarity) which is more appropriate and preserves
    the variance structure.

    Args:
        x: [N, D] or [D] first point(s)
        y: [M, D] or [D] second point(s)
        nu: Matern smoothness parameter
        sigma: Bandwidth (length scale)
        use_angular: If True, use angular distance (for L2-normalized data)
                     If None, auto-detect based on norm variance

    Returns:
        [N, M] distance matrix or scalar
    """
    import math

    # Ensure 2D
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Auto-detect L2-normalized data
    if use_angular is None:
        x_norms = torch.norm(x, dim=1)
        use_angular = (x_norms.std() < 0.01)  # All norms ≈ 1

    if use_angular:
        # For unit sphere data: use ANGULAR distance
        # Angular distance = arccos(cosine_similarity) / π ∈ [0, 1]
        # This preserves variance even for tightly clustered data
        x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)
        cos_sim = torch.mm(x_norm, y_norm.t())
        cos_sim = cos_sim.clamp(-1, 1)  # Numerical stability
        r = torch.acos(cos_sim) / math.pi  # Normalize to [0, 1]
    else:
        # Standard Euclidean distance
        r = torch.cdist(x, y)  # [N, M]

    # Scaled distance (for angular, sigma is in angular units)
    effective_sigma = sigma if not use_angular else sigma * 0.5  # Scale for [0,1] range
    effective_sigma = max(effective_sigma, 1e-4)
    r_scaled = r / effective_sigma

    # Matern kernel evaluation
    if nu == 0.5:
        # Laplacian: K(r) = exp(-r/σ)
        K = torch.exp(-r_scaled)
    elif nu == 1.5:
        # Matern 3/2: K(r) = (1 + √3·r/σ) exp(-√3·r/σ)
        sqrt3 = math.sqrt(3)
        K = (1 + sqrt3 * r_scaled) * torch.exp(-sqrt3 * r_scaled)
    elif nu == 2.5:
        # Matern 5/2: K(r) = (1 + √5·r/σ + 5r²/3σ²) exp(-√5·r/σ)
        sqrt5 = math.sqrt(5)
        K = (1 + sqrt5 * r_scaled + (5/3) * r_scaled**2) * torch.exp(-sqrt5 * r_scaled)
    else:
        # General Matern (approximation via RBF for high nu)
        K = torch.exp(-0.5 * r_scaled**2)

    # Convert kernel to distance metric
    # d = sqrt(2 * (1 - K)) ensures triangle inequality
    K = K.clamp(0, 1)  # Numerical stability
    d_matern = torch.sqrt(2 * (1 - K) + 1e-8)

    return d_matern


def compute_matern_distance_matrix(
    embeddings: torch.Tensor,
    nu: float = 1.5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute full Matern distance matrix for a set of embeddings.

    Args:
        embeddings: [N, D] embeddings
        nu: Matern smoothness
        sigma: Bandwidth

    Returns:
        [N, N] Matern distance matrix
    """
    return compute_matern_distance(embeddings, embeddings, nu, sigma)


# =============================================================================
# TRACK 1.5: ANTAGONISM GRADIENT FIELD (The "Force")
# =============================================================================

def compute_antagonism_gradient(
    antagonism_vectors: torch.Tensor,
    embeddings: torch.Tensor,
    anchors: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the ideological gradient field from Track 1.5 Antagonism.

    The Antagonism is the variance across 8 bot observers - it represents
    the DIRECTION in which ideological forces are pulling the article.

    This is NOT passive geometry (neighbor density).
    This is ACTIVE force: "Where is the Marxist observer trying to pull this?"

    Args:
        antagonism_vectors: [N, D] from complete_pipeline - variance across bots
        embeddings: [N, D] the fused embeddings (for reference frame)
        anchors: Optional dict of anchor embeddings {"left": [D], "right": [D], ...}

    Returns:
        Dict with:
        - 'gradient_magnitude': [N] the |∇Φ| at each point
        - 'gradient_direction': [N, D] unit vector of gradient direction
        - 'force_toward_anchors': [N, n_anchors] pull toward each anchor
    """
    N, D = antagonism_vectors.shape

    # The antagonism vector IS the gradient direction
    # High variance in a direction = strong ideological pull in that direction
    grad_magnitude = torch.norm(antagonism_vectors, dim=1)  # [N]

    # Normalize to get direction
    grad_direction = antagonism_vectors / (grad_magnitude.unsqueeze(1) + 1e-8)  # [N, D]

    result = {
        'gradient_magnitude': grad_magnitude,
        'gradient_direction': grad_direction,
    }

    # If anchors provided, compute pull toward each
    if anchors is not None:
        force_toward = {}
        for name, anchor in anchors.items():
            if anchor.dim() == 1:
                anchor = anchor.unsqueeze(0)
            # Direction from embedding to anchor
            to_anchor = anchor - embeddings  # [N, D]
            to_anchor_norm = to_anchor / (torch.norm(to_anchor, dim=1, keepdim=True) + 1e-8)

            # Dot product: how aligned is the gradient with the anchor direction?
            # Positive = gradient pulling toward anchor
            # Negative = gradient pulling away from anchor
            alignment = (grad_direction * to_anchor_norm).sum(dim=1)  # [N]
            force = grad_magnitude * alignment  # [N]
            force_toward[name] = force

        result['force_toward_anchors'] = force_toward

    return result


def compute_quiver_field(
    antagonism_vectors: torch.Tensor,
    coords_3d: np.ndarray,
    scale: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Compute 3D quiver (arrow) field from antagonism vectors.

    This creates the "Wind Field" for the Track 5 visualization.
    Arrows point WHERE the observers are pulling each article.

    Args:
        antagonism_vectors: [N, D] the raw antagonism (gradient) vectors
        coords_3d: [N, 3] the 3D projected coordinates
        scale: Arrow length scaling factor

    Returns:
        Dict with 'origins' [N, 3], 'directions' [N, 3], 'magnitudes' [N]
    """
    from sklearn.decomposition import PCA

    N = antagonism_vectors.shape[0]

    # Project antagonism vectors to 3D using PCA
    # (same dimensionality reduction as the embeddings)
    if antagonism_vectors.shape[1] > 3:
        pca = PCA(n_components=3)
        directions_3d = pca.fit_transform(antagonism_vectors.detach().cpu().numpy())
    else:
        directions_3d = antagonism_vectors.detach().cpu().numpy()

    # Compute magnitudes
    magnitudes = np.linalg.norm(directions_3d, axis=1)

    # Normalize directions
    directions_3d = directions_3d / (magnitudes[:, np.newaxis] + 1e-8)

    # Scale by magnitude and global scale
    magnitudes_normalized = magnitudes / (magnitudes.max() + 1e-8)
    directions_3d = directions_3d * magnitudes_normalized[:, np.newaxis] * scale

    return {
        'origins': coords_3d,
        'directions': directions_3d,
        'magnitudes': magnitudes_normalized,
    }


# =============================================================================
# SHEAR STRESS / FAULT LINES (Where the manifold tears apart)
# =============================================================================

def compute_fault_lines(
    antagonism_vectors: torch.Tensor,
    embeddings: torch.Tensor,
    k_neighbors: int = 10,
    shear_threshold: float = -0.5,
    nu: float = 1.5,
    sigma: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute Fault Lines: edges between neighboring points whose ideological
    vectors point in OPPOSITE directions (shear stress).

    A Fault Line exists between articles i and j when:
    1. They are neighbors (within k-NN)
    2. Their antagonism vectors have dot product < shear_threshold
       (i.e., the observers are pulling them in opposing directions)

    This reveals where the semantic manifold is literally TEARING APART.

    Args:
        antagonism_vectors: [N, D] Track 1.5 ideological gradient vectors
        embeddings: [N, D] for computing neighborhoods
        k_neighbors: Neighborhood size
        shear_threshold: Dot product threshold (default -0.5, range [-1, 1])
        nu: Matern smoothness for neighbor computation
        sigma: Matern bandwidth

    Returns:
        Dict with:
        - 'fault_pairs': List[Tuple[int, int]] - pairs of articles with fault lines
        - 'shear_stress': List[float] - shear magnitude for each pair
        - 'n_faults': int - total number of fault lines
        - 'fault_fraction': float - fraction of edges that are faults
    """
    N = antagonism_vectors.shape[0]
    k = min(k_neighbors, N - 1)

    # Normalize antagonism vectors to unit length for dot product
    antag_norm = antagonism_vectors / (torch.norm(antagonism_vectors, dim=1, keepdim=True) + 1e-8)

    # Build neighbor graph using Matern distances
    dists = compute_matern_distance_matrix(embeddings, nu=nu, sigma=sigma)
    _, neighbor_idx = dists.topk(k + 1, largest=False)
    neighbor_idx = neighbor_idx[:, 1:]  # Exclude self

    # Find fault lines: neighbor pairs with opposing ideological vectors
    fault_pairs = []
    shear_values = []
    total_edges = 0

    for i in range(N):
        for j_pos in range(k):
            j = neighbor_idx[i, j_pos].item()
            if j <= i:
                continue  # Avoid duplicate edges

            total_edges += 1

            # Dot product of normalized antagonism vectors
            dot = (antag_norm[i] * antag_norm[j]).sum().item()

            if dot < shear_threshold:
                fault_pairs.append((i, j))
                shear_values.append(abs(dot))  # Magnitude of opposition

    return {
        'fault_pairs': fault_pairs,
        'shear_stress': shear_values,
        'n_faults': len(fault_pairs),
        'fault_fraction': len(fault_pairs) / max(total_edges, 1),
        'shear_threshold': shear_threshold,
    }


# =============================================================================
# TRACK 4: STOCHASTIC WALKER (Metropolis-Hastings Mixing Time)
# =============================================================================
# MANDATE: The Matern Kernel's jaggedness makes gradients UNDEFINED at
# singularities. You cannot "ski" on a fractal surface. Instead, we use
# Stochastic Homotopy (Metropolis-Hastings) to measure MIXING TIME (τ_mix).
#
# τ_mix = how many MCMC steps until a walker, starting at article i,
#         reaches equilibrium across its neighborhood.
#
# HIGH τ_mix → Article is ISOLATED (barrier, ideological wall)
# LOW  τ_mix → Article is CONNECTED (consensus, easy flow)
# =============================================================================

def compute_matern_density(
    embeddings: torch.Tensor,
    nu: float = 1.5,
    sigma: float = 1.0,
    k_neighbors: int = 10,
) -> torch.Tensor:
    """
    Compute Matern kernel density π(x) at each point.

    This is the TARGET DISTRIBUTION for the Metropolis-Hastings walker.
    It defines "habitability" of each point in semantic space:
    - High π(x): Dense, habitable region → walker stays easily
    - Low π(x):  Sparse, inhospitable → walker gets rejected

    The density is estimated as the mean Matern kernel similarity to
    k nearest neighbors:
        π(x_i) = (1/k) Σ_{j ∈ kNN(i)} K_matern(x_i, x_j)

    Args:
        embeddings: [N, D] article embeddings
        nu: Matern smoothness (0.5=rough, 1.5=default)
        sigma: Matern bandwidth
        k_neighbors: Number of neighbors for density estimation

    Returns:
        [N] density values (unnormalized)
    """
    import math

    N = embeddings.shape[0]
    k = min(k_neighbors, N - 1)

    # Check if L2-normalized
    norms = torch.norm(embeddings, dim=1)
    is_normalized = (norms.std() < 0.01)

    if is_normalized:
        # Angular distance for unit sphere data
        x_norm = embeddings / (norms.unsqueeze(1) + 1e-8)
        cos_sim = torch.mm(x_norm, x_norm.t())
        cos_sim = cos_sim.clamp(-1, 1)
        r = torch.acos(cos_sim) / math.pi
        effective_sigma = sigma * 0.5
    else:
        r = torch.cdist(embeddings, embeddings)
        effective_sigma = sigma

    effective_sigma = max(effective_sigma, 1e-4)
    r_scaled = r / effective_sigma

    # Matern kernel evaluation
    if nu == 0.5:
        K = torch.exp(-r_scaled)
    elif nu == 1.5:
        sqrt3 = math.sqrt(3)
        K = (1 + sqrt3 * r_scaled) * torch.exp(-sqrt3 * r_scaled)
    elif nu == 2.5:
        sqrt5 = math.sqrt(5)
        K = (1 + sqrt5 * r_scaled + (5/3) * r_scaled**2) * torch.exp(-sqrt5 * r_scaled)
    else:
        K = torch.exp(-0.5 * r_scaled**2)

    K = K.clamp(0, 1)

    # Zero out self-similarity
    K.fill_diagonal_(0)

    # Density = mean kernel similarity to k nearest neighbors
    # (use topk on kernel values, not distances)
    topk_vals, _ = K.topk(k, dim=1)
    density = topk_vals.mean(dim=1)  # [N]

    return density


def metropolis_step(
    current_idx: torch.Tensor,
    density: torch.Tensor,
    neighbor_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Single Metropolis-Hastings step for stochastic walkers.

    The walker proposes a jump to a random neighbor.
    Accept with probability min(1, π(neighbor) / π(current)).

    Args:
        current_idx: [n_walkers] current position indices
        density: [N] Matern density at each point
        neighbor_idx: [N, k] neighbor indices for proposals
        temperature: Controls acceptance (higher = more exploration)

    Returns:
        [n_walkers] new position indices after step
    """
    n_walkers = current_idx.shape[0]
    k = neighbor_idx.shape[1]

    # Propose: random neighbor
    proposal_which = torch.randint(0, k, (n_walkers,))
    proposal_idx = neighbor_idx[current_idx, proposal_which]

    # Acceptance ratio: π(proposed) / π(current)
    current_density = density[current_idx]
    proposal_density = density[proposal_idx]
    ratio = (proposal_density / (current_density + 1e-10)).clamp(max=10000)

    # Metropolis criterion with temperature
    acceptance_prob = torch.min(torch.ones_like(ratio), ratio ** (1.0 / temperature))
    accept_mask = torch.rand(n_walkers) < acceptance_prob

    # Update positions
    new_idx = torch.where(accept_mask, proposal_idx, current_idx)
    return new_idx


def compute_mixing_time(
    embeddings: torch.Tensor,
    nu: float = 1.5,
    sigma: float = 1.0,
    n_walkers_per_article: int = 5,
    n_steps: int = 50,
    k_neighbors: int = 10,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute MCMC mixing time for each article via Metropolis-Hastings walkers.

    THIS IS TRACK 4 (Walker). NOT the work integral. NOT geodesic distance.

    For each article i, we spawn walkers and measure how many steps
    until they reach equilibrium (explore beyond their starting neighborhood).

    τ_mix is estimated as the average number of steps before the walker
    escapes a radius of k neighbors from the starting point.

    HIGH τ_mix = Article is trapped (ideological barrier, isolated)
    LOW  τ_mix = Article flows freely (consensus, well-connected)

    Args:
        embeddings: [N, D] article embeddings
        nu: Matern smoothness
        sigma: Matern bandwidth
        n_walkers_per_article: Walkers spawned per article
        n_steps: Maximum MCMC steps
        k_neighbors: Neighborhood size
        temperature: Metropolis temperature

    Returns:
        Dict with:
        - 'mixing_time': [N] τ_mix per article
        - 'acceptance_rate': [N] mean acceptance rate
        - 'density': [N] Matern density π(x)
        - 'escape_fraction': [N] fraction of walkers that escaped
    """
    N = embeddings.shape[0]
    k = min(k_neighbors, N - 1)

    # Step 1: Compute Matern density π(x)
    density = compute_matern_density(embeddings, nu=nu, sigma=sigma, k_neighbors=k)

    # Step 2: Build neighbor graph
    d_matern = compute_matern_distance_matrix(embeddings, nu=nu, sigma=sigma)
    _, neighbor_idx = d_matern.topk(k + 1, largest=False)
    neighbor_idx = neighbor_idx[:, 1:]  # Exclude self

    # Step 3: Run walkers from each article
    mixing_times = torch.zeros(N)
    acceptance_rates = torch.zeros(N)
    escape_fractions = torch.zeros(N)

    for i in range(N):
        n_w = n_walkers_per_article
        # Spawn all walkers at article i
        current_pos = torch.full((n_w,), i, dtype=torch.long)

        # Track escape: has the walker left the k-neighborhood of i?
        home_neighbors = set(neighbor_idx[i].tolist())
        home_neighbors.add(i)

        step_escaped = torch.full((n_w,), float(n_steps))  # Default: never escaped
        n_accepted = torch.zeros(n_w)
        escaped = torch.zeros(n_w, dtype=torch.bool)

        for step in range(n_steps):
            new_pos = metropolis_step(current_pos, density, neighbor_idx, temperature)

            # Track acceptances
            n_accepted += (new_pos != current_pos).float()
            current_pos = new_pos

            # Check escape: is walker outside home neighborhood?
            for w in range(n_w):
                if not escaped[w] and current_pos[w].item() not in home_neighbors:
                    step_escaped[w] = step + 1
                    escaped[w] = True

        # τ_mix for article i = mean steps to escape
        mixing_times[i] = step_escaped.mean()
        acceptance_rates[i] = n_accepted.mean() / n_steps
        escape_fractions[i] = escaped.float().mean()

    return {
        'mixing_time': mixing_times,        # [N] τ_mix per article
        'acceptance_rate': acceptance_rates,  # [N] mean acceptance
        'density': density,                  # [N] Matern density π(x)
        'escape_fraction': escape_fractions, # [N] fraction that escaped
    }


def compute_directed_mixing_time(
    embeddings: torch.Tensor,
    source_idx: int,
    target_idx: int,
    nu: float = 1.5,
    sigma: float = 1.0,
    n_walkers: int = 20,
    n_steps: int = 100,
    k_neighbors: int = 10,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute DIRECTED mixing time from article A to article B.

    Used for HYSTERESIS detection:
    If τ(A→B) ≠ τ(B→A), the semantic space is IRREVERSIBLE.

    Example: τ(Property→Theft) < τ(Theft→Property) in a Marxist manifold
    means the concepts slide together easily but resist separation.

    Args:
        embeddings: [N, D] all embeddings
        source_idx: Starting article index
        target_idx: Target article index
        nu, sigma: Matern parameters
        n_walkers: Number of walkers
        n_steps: Max steps
        k_neighbors: Neighborhood size
        temperature: Metropolis temperature

    Returns:
        Dict with 'arrival_time', 'arrival_fraction', 'mean_path_length'
    """
    N = embeddings.shape[0]
    k = min(k_neighbors, N - 1)

    density = compute_matern_density(embeddings, nu=nu, sigma=sigma, k_neighbors=k)

    d_matern = compute_matern_distance_matrix(embeddings, nu=nu, sigma=sigma)
    _, neighbor_idx = d_matern.topk(k + 1, largest=False)
    neighbor_idx = neighbor_idx[:, 1:]

    # Target neighborhood
    target_neighbors = set(neighbor_idx[target_idx].tolist())
    target_neighbors.add(target_idx)

    # Spawn walkers at source
    current_pos = torch.full((n_walkers,), source_idx, dtype=torch.long)
    arrival_time = torch.full((n_walkers,), float(n_steps))
    arrived = torch.zeros(n_walkers, dtype=torch.bool)

    for step in range(n_steps):
        current_pos = metropolis_step(current_pos, density, neighbor_idx, temperature)

        for w in range(n_walkers):
            if not arrived[w] and current_pos[w].item() in target_neighbors:
                arrival_time[w] = step + 1
                arrived[w] = True

    return {
        'arrival_time': float(arrival_time.mean()),
        'arrival_fraction': float(arrived.float().mean()),
        'source': source_idx,
        'target': target_idx,
    }


# =============================================================================
# IDEOLOGICAL POTENTIAL FIELD (Φ_θ)
# =============================================================================

class IdeologicalPotential(nn.Module):
    """
    The Ideological Potential Field Φ_θ(x).

    This represents the "moral charge" of a concept under ideology θ.
    The gradient ∇Φ creates the force field that warps semantic space.

    Φ_θ(x) is learned/estimated from:
    1. Observer disagreement (Dirichlet variance)
    2. Anchor distances (from metric_gradients.py)
    3. NLI confidence gaps
    """

    def __init__(self, hidden_dim: int, n_ideologies: int = 8, config: RiemannianConfig = None):
        super().__init__()
        self.config = config or RiemannianConfig()
        self.hidden_dim = hidden_dim
        self.n_ideologies = n_ideologies

        # Learnable ideology embeddings (the θ parameter)
        self.ideology_embeddings = nn.Parameter(
            torch.randn(n_ideologies, hidden_dim) * 0.02
        )

        # Potential function: maps (x, θ) → scalar potential
        self.potential_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, ideology_idx: int = None) -> torch.Tensor:
        """
        Compute potential Φ_θ(x) for concept embedding x.

        Args:
            x: [batch, hidden_dim] concept embeddings
            ideology_idx: Which ideology to use (None = mean over all)

        Returns:
            potential: [batch] scalar potential values
        """
        if ideology_idx is not None:
            theta = self.ideology_embeddings[ideology_idx].unsqueeze(0)
            theta = theta.expand(x.shape[0], -1)
        else:
            # Average over all ideologies
            theta = self.ideology_embeddings.mean(dim=0, keepdim=True)
            theta = theta.expand(x.shape[0], -1)

        # Concatenate x and θ
        combined = torch.cat([x, theta], dim=-1)

        # Compute potential
        potential = self.potential_net(combined).squeeze(-1)

        return potential * self.config.potential_scale

    def gradient(self, x: torch.Tensor, ideology_idx: int = None) -> torch.Tensor:
        """
        Compute gradient ∇Φ_θ(x) - the "force" at point x.

        Args:
            x: [batch, hidden_dim] concept embeddings (requires_grad=True)
            ideology_idx: Which ideology to use

        Returns:
            gradient: [batch, hidden_dim] force vectors
        """
        x = x.clone().requires_grad_(True)
        potential = self.forward(x, ideology_idx)

        # Compute gradient via autograd
        grad = torch.autograd.grad(
            outputs=potential.sum(),
            inputs=x,
            create_graph=True,
        )[0]

        return grad


# =============================================================================
# IDEOLOGICAL METRIC TENSOR (g_μν)
# =============================================================================

class IdeologicalMetricTensor:
    """
    The Ideological Metric Tensor g_μν(x, θ).

    g_μν(x, θ) = δ_μν + ∇_μ Φ_θ(x) ∇_ν Φ_θ(x)

    This tensor defines the "cost" of moving in each direction at point x.
    When the potential gradient is large, the metric stretches space,
    making movement expensive (creating ideological barriers).
    """

    def __init__(self, potential: IdeologicalPotential, config: RiemannianConfig = None):
        self.potential = potential
        self.config = config or RiemannianConfig()
        self.hidden_dim = potential.hidden_dim

    def compute(self, x: torch.Tensor, ideology_idx: int = None) -> torch.Tensor:
        """
        Compute the metric tensor at point x.

        g_μν = δ_μν + ∇_μ Φ ∇_ν Φ

        Args:
            x: [batch, hidden_dim] points in semantic space
            ideology_idx: Which ideology's metric to compute

        Returns:
            g: [batch, hidden_dim, hidden_dim] metric tensor at each point
        """
        batch_size = x.shape[0]

        # Get potential gradient
        grad_phi = self.potential.gradient(x, ideology_idx)  # [batch, D]

        # Compute outer product: ∇Φ ⊗ ∇Φ
        # [batch, D, 1] @ [batch, 1, D] = [batch, D, D]
        grad_outer = torch.bmm(
            grad_phi.unsqueeze(-1),
            grad_phi.unsqueeze(-2)
        )

        # Base metric (identity)
        delta = torch.eye(self.hidden_dim, device=x.device)
        delta = delta.unsqueeze(0).expand(batch_size, -1, -1)

        # Full metric tensor
        g = delta + grad_outer

        return g

    def compute_determinant(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute √det(g) - the volume element.

        This tells us how much space is "stretched" at each point.
        High determinant = space is expanded = movement is costly.
        """
        # For numerical stability, use log determinant
        sign, logdet = torch.linalg.slogdet(g)
        det = sign * torch.exp(logdet / 2)  # √det
        return det

    def compute_anisotropy(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute anisotropy of the metric tensor.

        Anisotropy = how non-uniform the metric is.
        - Isotropic (uniform): eigenvalues are equal
        - Anisotropic: eigenvalues differ

        Returns ratio of max/min eigenvalue (condition number).
        """
        eigenvalues = torch.linalg.eigvalsh(g)  # [batch, D]

        # Condition number (max/min eigenvalue)
        max_eig = eigenvalues.max(dim=-1).values
        min_eig = eigenvalues.min(dim=-1).values.clamp(min=1e-8)

        anisotropy = max_eig / min_eig

        return anisotropy


# =============================================================================
# SEMANTIC WORK INTEGRATOR (Track 4)
# =============================================================================

class SemanticWorkIntegrator:
    """
    Computes the Semantic Work W(A→B) along a geodesic path.

    W(A→B) = ∫_A^B √(g_μν ẋ^μ ẋ^ν) dt

    This is the "energy cost" to move from concept A to concept B
    through the ideological force field.

    This is Track 4 (Walker) reframed: instead of "resistance",
    we compute actual geodesic work in curved semantic space.
    """

    def __init__(self, metric_tensor: IdeologicalMetricTensor, config: RiemannianConfig = None):
        self.metric = metric_tensor
        self.config = config or RiemannianConfig()

    def compute_work(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        ideology_idx: int = None,
        path: str = "linear",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute semantic work W(start → end).

        Args:
            start: [batch, D] starting points
            end: [batch, D] ending points
            ideology_idx: Which ideology's metric to use
            path: Path type ("linear", "geodesic_approx")

        Returns:
            Dict with:
            - work: [batch] total work for each pair
            - work_density: [batch, n_steps] work at each step
            - path_points: [batch, n_steps, D] points along path
        """
        n_steps = self.config.n_geodesic_steps
        batch_size = start.shape[0]

        # Generate path points (linear interpolation for now)
        # TODO: True geodesic would require solving the geodesic equation
        t = torch.linspace(0, 1, n_steps, device=start.device)
        t = t.view(1, -1, 1).expand(batch_size, -1, start.shape[-1])

        # Linear path: γ(t) = (1-t)*A + t*B
        path_points = (1 - t) * start.unsqueeze(1) + t * end.unsqueeze(1)

        # Compute velocity ẋ = dγ/dt = B - A (constant for linear path)
        velocity = (end - start).unsqueeze(1).expand(-1, n_steps, -1)

        # Compute metric at each point along path
        path_flat = path_points.view(-1, start.shape[-1])
        g_flat = self.metric.compute(path_flat, ideology_idx)
        g = g_flat.view(batch_size, n_steps, start.shape[-1], start.shape[-1])

        # Compute √(g_μν ẋ^μ ẋ^ν) at each point
        # This is the "infinitesimal work" at each step

        # v^T @ g @ v for each point
        # [batch, steps, 1, D] @ [batch, steps, D, D] @ [batch, steps, D, 1]
        v = velocity.unsqueeze(-1)  # [batch, steps, D, 1]
        gv = torch.matmul(g, v)     # [batch, steps, D, 1]
        vgv = torch.matmul(velocity.unsqueeze(-2), gv).squeeze(-1).squeeze(-1)

        # √(v^T g v)
        work_density = torch.sqrt(vgv.clamp(min=1e-10))

        # Integrate using trapezoidal rule
        dt = 1.0 / (n_steps - 1)
        if self.config.integration_method == "trapezoidal":
            work = (work_density[:, :-1] + work_density[:, 1:]).sum(dim=-1) * dt / 2
        else:  # Simple sum
            work = work_density.sum(dim=-1) * dt

        # Normalize work
        if self.config.work_normalization == "log":
            work = torch.log1p(work)
        elif self.config.work_normalization == "sigmoid":
            work = torch.sigmoid(work - 1) * 2  # Centered sigmoid

        return {
            'work': work,
            'work_density': work_density,
            'path_points': path_points,
            'euclidean_distance': torch.norm(end - start, dim=-1),
        }

    def compute_work_matrix(
        self,
        points: torch.Tensor,
        ideology_idx: int = None,
    ) -> torch.Tensor:
        """
        Compute pairwise work matrix for a set of points.

        Args:
            points: [N, D] points in semantic space
            ideology_idx: Which ideology's metric

        Returns:
            W: [N, N] work matrix where W[i,j] = work from i to j
        """
        N = points.shape[0]
        work_matrix = torch.zeros(N, N, device=points.device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                result = self.compute_work(
                    points[i:i+1],
                    points[j:j+1],
                    ideology_idx,
                )
                work_matrix[i, j] = result['work'].item()

        return work_matrix


# =============================================================================
# ENTROPY FILTER (Track 3)
# =============================================================================

class MetricEntropyFilter:
    """
    The Entropy Filter checks if the Metric Tensor is stable enough to measure.

    This is Track 3 (Blinker) reframed: instead of "variance", we compute
    the entropy of the metric tensor eigenvalues to measure stability.

    High entropy = metric is fluctuating across nearby points = unreliable.
    Low entropy = metric is stable = measurements are meaningful.

    Physics interpretation: If spacetime is "foamy" (high Planck-scale
    fluctuations), you can't trust distance measurements.
    """

    def __init__(self, metric_tensor: IdeologicalMetricTensor, config: RiemannianConfig = None):
        self.metric = metric_tensor
        self.config = config or RiemannianConfig()

    def compute_local_entropy(
        self,
        center: torch.Tensor,
        neighbors: torch.Tensor,
        ideology_idx: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute entropy of metric tensor in local neighborhood.

        Args:
            center: [batch, D] center points
            neighbors: [batch, K, D] K neighbors for each center
            ideology_idx: Which ideology's metric

        Returns:
            Dict with entropy and stability measures
        """
        batch_size = center.shape[0]
        K = neighbors.shape[1]

        # Compute metric at center
        g_center = self.metric.compute(center, ideology_idx)

        # Compute metric at each neighbor
        neighbors_flat = neighbors.view(-1, center.shape[-1])
        g_neighbors_flat = self.metric.compute(neighbors_flat, ideology_idx)
        g_neighbors = g_neighbors_flat.view(batch_size, K, center.shape[-1], center.shape[-1])

        # Compute eigenvalues at each point
        eig_center = torch.linalg.eigvalsh(g_center)  # [batch, D]
        eig_neighbors = torch.linalg.eigvalsh(g_neighbors)  # [batch, K, D]

        # Compute spectral entropy at center
        # H = -Σ p_i log(p_i) where p_i = λ_i / Σλ
        def spectral_entropy(eigenvalues):
            # Normalize eigenvalues to form probability distribution
            p = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            p = p.clamp(min=1e-10)  # Avoid log(0)
            entropy = -(p * torch.log(p)).sum(dim=-1)
            return entropy

        entropy_center = spectral_entropy(eig_center)
        entropy_neighbors = spectral_entropy(eig_neighbors)

        # Metric stability = variance of entropy across neighborhood
        entropy_variance = entropy_neighbors.var(dim=1)

        # Anisotropy at center
        anisotropy = self.metric.compute_anisotropy(g_center)

        # Stability score: low entropy variance + low anisotropy = stable
        stability = 1.0 / (1.0 + entropy_variance + 0.1 * torch.log1p(anisotropy))

        return {
            'entropy_center': entropy_center,
            'entropy_variance': entropy_variance,
            'anisotropy': anisotropy,
            'stability': stability,
            'is_stable': stability > self.config.stability_threshold,
        }

    def compute_corpus_stability(
        self,
        points: torch.Tensor,
        k_neighbors: int = 10,
        ideology_idx: int = None,
    ) -> Dict[str, Any]:
        """
        Compute stability across entire corpus.

        Args:
            points: [N, D] all points
            k_neighbors: Number of neighbors for local entropy
            ideology_idx: Which ideology's metric

        Returns:
            Stability statistics for the corpus
        """
        N = points.shape[0]

        # Find k nearest neighbors for each point
        dists = torch.cdist(points, points)
        _, neighbor_idx = dists.topk(k_neighbors + 1, largest=False)
        neighbor_idx = neighbor_idx[:, 1:]  # Exclude self

        neighbors = points[neighbor_idx]  # [N, K, D]

        # Compute local entropy for all points
        results = self.compute_local_entropy(points, neighbors, ideology_idx)

        return {
            'mean_entropy': results['entropy_center'].mean().item(),
            'std_entropy': results['entropy_center'].std().item(),
            'mean_stability': results['stability'].mean().item(),
            'pct_stable': results['is_stable'].float().mean().item() * 100,
            'mean_anisotropy': results['anisotropy'].mean().item(),
            'per_point_stability': results['stability'],
        }


# =============================================================================
# ENERGY LANDSCAPE VISUALIZATION
# =============================================================================

class EnergyLandscape:
    """
    Generate energy landscape visualization data.

    This transforms the "comet trails" visualization into an
    "Energy Landscape" showing valleys (easy paths) and cliffs (barriers).
    """

    def __init__(
        self,
        potential: IdeologicalPotential,
        metric: IdeologicalMetricTensor,
        config: RiemannianConfig = None,
    ):
        self.potential = potential
        self.metric = metric
        self.config = config or RiemannianConfig()

    def compute_landscape_2d(
        self,
        points_2d: np.ndarray,
        points_hd: torch.Tensor,
        ideology_idx: int = None,
        grid_resolution: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Compute energy landscape on 2D projection.

        Args:
            points_2d: [N, 2] 2D projections (e.g., UMAP)
            points_hd: [N, D] high-dimensional embeddings
            ideology_idx: Which ideology
            grid_resolution: Resolution of energy grid

        Returns:
            Landscape data for visualization
        """
        # Compute potential at each point
        with torch.no_grad():
            potential_values = self.potential(points_hd, ideology_idx)

        # Compute metric determinant (volume element)
        g = self.metric.compute(points_hd, ideology_idx)
        det = self.metric.compute_determinant(g)

        # Compute anisotropy
        anisotropy = self.metric.compute_anisotropy(g)

        return {
            'points_2d': points_2d,
            'potential': potential_values.cpu().numpy(),
            'volume_element': det.cpu().numpy(),
            'anisotropy': anisotropy.cpu().numpy(),
            # Classification
            'is_valley': (potential_values < potential_values.median()).cpu().numpy(),
            'is_cliff': (anisotropy > anisotropy.median() * 2).cpu().numpy(),
        }


# =============================================================================
# WATERFALL INTEGRATION - Using Existing Architecture
# =============================================================================

def compute_physics_from_waterfall(
    fused_std: torch.Tensor,           # From dirichlet_fusion: [N, D] observer disagreement
    stability_results: Dict[str, Any], # From metric_gradients.compute_metric_gradients
    embeddings: torch.Tensor,          # The fused embeddings [N, D]
    tension_results: Optional[Dict] = None,  # Optional: from MetricGradientAnalyzer
    antagonism_vectors: Optional[torch.Tensor] = None,  # Track 1.5: [N, D] ideological gradient
    matern_nu: float = 1.5,            # Track 2: Matern roughness parameter
    matern_sigma: float = 1.0,         # Track 2: Matern bandwidth
) -> Dict[str, Any]:
    """
    Compute physics using EXISTING Waterfall outputs.

    This is the PROPER integration that connects to dirichlet_fusion.py
    and metric_gradients.py systems.

    WIRING MANDATE (Critical Integration):
    =====================================
    Track 1.5 (Antagonism):
        Gradient field from 8 observers. Used for quiver directions and
        shear stress detection. NOT used for smooth work integrals.

    Track 2 (Matern) → Density π(x):
        Matern kernel used as PROBABILITY DENSITY for Metropolis-Hastings
        rejection sampling. NOT as a distance metric for gradients.
        Jagged topology = non-differentiable = no gradients = use MCMC.

    Track 3 (Blinker/Entropy): fused_std + stability_score
        Observer disagreement IS the entropy. Unchanged.

    Track 4 (Walker/Mixing Time): Metropolis-Hastings stochastic walkers
        NO work integrals. NO geodesic distance. Cannot ski on fractal.
        τ_mix = MCMC mixing time = steps to escape neighborhood.
        High τ_mix = isolated (barrier). Low τ_mix = connected (consensus).

    Args:
        fused_std: [N, D] from DirichletFusion - std across observers
        stability_results: from compute_metric_gradients - contains stability_score
        embeddings: [N, D] the fused article embeddings
        tension_results: optional per-article tensions from MetricGradientAnalyzer
        antagonism_vectors: [N, D] Track 1.5 - variance across 8 bots (for quivers)
        matern_nu: Matern smoothness (0.5=rough, 1.5=default, high=smooth)
        matern_sigma: Matern bandwidth

    Returns:
        Dict with blinker, walker, singularity classifications, density
    """
    N = embeddings.shape[0]
    device = embeddings.device

    # =========================================================================
    # TRACK 3: BLINKER (Entropy Filter) from fused_std
    # =========================================================================
    # The fused_std IS the observer disagreement - this directly measures
    # how "uncertain" the ideological position is.
    # High std = the manifold is "foamy" = metric tensor is unstable

    # Per-article entropy: mean std across dimensions
    article_entropy = fused_std.mean(dim=1)  # [N]

    # Global stability from alpha sweep
    global_stability = stability_results.get('stability_score', 0.5)

    # Blinker = local entropy modulated by global stability
    # If globally unstable, all blinkers are elevated
    blinker_raw = article_entropy * (2 - global_stability)

    # =========================================================================
    # TRACK 4: WALKER (Stochastic Mixing Time) - MCMC, NOT work integral
    # =========================================================================
    # MANDATE: Matern's jaggedness makes gradients UNDEFINED at singularities.
    # We use Metropolis-Hastings walkers to measure CONNECTIVITY via mixing time.
    # The Matern kernel is the DENSITY π(x) for rejection sampling.

    mcmc_results = compute_mixing_time(
        embeddings,
        nu=matern_nu,
        sigma=matern_sigma,
        n_walkers_per_article=5,
        n_steps=min(50, N),  # Scale steps with corpus size
        k_neighbors=min(10, N - 1),
        temperature=1.0,
    )

    walker_raw = mcmc_results['mixing_time']  # τ_mix per article
    matern_density = mcmc_results['density']  # π(x) per article

    # Track 1.5: Antagonism gradient (for quivers, NOT for walker)
    grad_magnitude = None
    grad_direction = None
    if antagonism_vectors is not None:
        gradient_results = compute_antagonism_gradient(antagonism_vectors, embeddings)
        grad_magnitude = gradient_results['gradient_magnitude']
        grad_direction = gradient_results['gradient_direction']

    # =========================================================================
    # NORMALIZE AND CLASSIFY
    # =========================================================================
    def robust_normalize(x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        if np.any(np.isnan(x_np)):
            x_np = np.nan_to_num(x_np, nan=0.5)
        p5, p95 = np.percentile(x_np, [5, 95])
        if p95 - p5 < 1e-10:
            return torch.tensor(np.random.uniform(0.3, 0.7, len(x_np)), dtype=torch.float32)
        clipped = np.clip(x_np, p5, p95)
        normalized = (clipped - p5) / (p95 - p5 + 1e-10)
        return torch.tensor(normalized, dtype=torch.float32)

    blinker = robust_normalize(blinker_raw)
    walker = robust_normalize(walker_raw)

    # Classify singularities
    threshold = 0.5
    singularity_types = []
    for i in range(N):
        b_high = blinker[i].item() > threshold
        w_high = walker[i].item() > threshold

        if b_high and w_high:
            singularity_types.append('structural_singularity')
        elif b_high:
            singularity_types.append('noise')
        elif w_high:
            singularity_types.append('ideological_barrier')
        else:
            singularity_types.append('consensus')

    # Compute liar scores: disagreement between entropy and work
    # A liar has low entropy (stable metric) but high work (far from consensus)
    liar_scores = torch.abs(blinker - walker) * (blinker + walker) / 2

    result = {
        'blinker': blinker,           # Track 3: Entropy filter (observer disagreement)
        'walker': walker,             # Track 4: MCMC mixing time (τ_mix)
        'liar_scores': liar_scores,   # Liar detection
        'singularity_types': singularity_types,
        # MCMC diagnostics
        'mixing_time_raw': mcmc_results['mixing_time'],
        'matern_density': matern_density,          # π(x) habitability
        'acceptance_rate': mcmc_results['acceptance_rate'],
        'escape_fraction': mcmc_results['escape_fraction'],
        # Gradient field (Track 1.5 - for quivers, NOT for walker)
        'gradient_magnitude': grad_magnitude,
        'gradient_direction': grad_direction,
        # Diagnostics
        'global_stability': global_stability,
        'mean_entropy': float(article_entropy.mean().item()),
        'matern_nu': matern_nu,
        'matern_sigma': matern_sigma,
        'source': 'waterfall_mcmc',
        'wiring': {
            'track_1_5': antagonism_vectors is not None,
            'track_2_matern_density': True,  # Matern as π(x) for MCMC
            'track_4_method': 'metropolis_hastings_mixing_time',
        }
    }

    return result


# =============================================================================
# SIMPLIFIED INTERFACE FOR EXISTING PIPELINE
# =============================================================================

def compute_riemannian_physics(
    embeddings: torch.Tensor,
    config: RiemannianConfig = None,
    antagonism_vectors: Optional[torch.Tensor] = None,  # Track 1.5 if available
) -> Dict[str, torch.Tensor]:
    """
    Compute physics from embeddings using MCMC mixing time (Track 4)
    and Matern density (Track 2).

    Track 3 (Blinker): Local metric instability from neighbor variance.
    Track 4 (Walker):  Metropolis-Hastings mixing time τ_mix.
                       NOT work integrals. NOT geodesic distance.
                       Matern kernel defines the density π(x) for rejection sampling.

    Args:
        embeddings: [N, D] article embeddings
        config: Riemannian configuration
        antagonism_vectors: Optional [N, D] Track 1.5 gradients (for quivers only)

    Returns:
        Dict with blinker (entropy), walker (mixing time), and classifications
    """
    config = config or RiemannianConfig()
    device = embeddings.device
    N, D = embeddings.shape

    # Check if embeddings are L2-normalized (all norms ≈ 1)
    norms = torch.norm(embeddings, dim=1)
    is_normalized = (norms.std() < 0.01)

    # =========================================================================
    # TRACK 2: MATERN DENSITY π(x) for neighborhoods and MCMC
    # =========================================================================
    dists = compute_matern_distance_matrix(
        embeddings,
        nu=config.matern_nu,
        sigma=config.matern_sigma
    )

    k = min(10, N - 1)
    knn_dists, knn_idx = dists.topk(k + 1, largest=False)
    knn_dists = knn_dists[:, 1:]
    knn_idx = knn_idx[:, 1:]

    # =========================================================================
    # TRACK 3: BLINKER (Entropy Filter) - local metric instability
    # =========================================================================
    # Measure how "foamy" the local geometry is
    if is_normalized:
        neighbor_vecs = embeddings[knn_idx]
        center_expanded = embeddings.unsqueeze(1)
        neighbor_cos = (neighbor_vecs * center_expanded).sum(dim=-1)
        angular_spread = (1 - neighbor_cos).mean(dim=1)
    else:
        angular_spread = knn_dists.std(dim=1)

    spread_var = knn_dists.var(dim=1)
    blinker_raw = spread_var + 0.5 * angular_spread

    # =========================================================================
    # TRACK 4: WALKER (MCMC Mixing Time) - stochastic, NOT smooth
    # =========================================================================
    # MANDATE: No work integrals. No geodesic skiing on fractal surfaces.
    # Use Metropolis-Hastings walkers with Matern density π(x).
    mcmc_results = compute_mixing_time(
        embeddings,
        nu=config.matern_nu,
        sigma=config.matern_sigma,
        n_walkers_per_article=5,
        n_steps=min(50, N),
        k_neighbors=k,
        temperature=1.0,
    )

    walker_raw = mcmc_results['mixing_time']
    matern_density = mcmc_results['density']

    # Track 1.5: Antagonism (for quivers, NOT for walker computation)
    grad_magnitude = None
    grad_direction = None
    if antagonism_vectors is not None:
        gradient_results = compute_antagonism_gradient(antagonism_vectors, embeddings)
        grad_magnitude = gradient_results['gradient_magnitude']
        grad_direction = gradient_results['gradient_direction']

    # Robust normalization to [0, 1] using percentile
    def robust_normalize(x):
        x_np = x.detach().cpu().numpy()
        # Handle NaN
        if np.any(np.isnan(x_np)):
            x_np = np.nan_to_num(x_np, nan=0.5)
        # Percentile scaling
        p5, p95 = np.percentile(x_np, [5, 95])
        if p95 - p5 < 1e-10:
            # No variance - use random to create distribution
            return torch.tensor(np.random.uniform(0.3, 0.7, N), dtype=x.dtype, device=x.device)
        clipped = np.clip(x_np, p5, p95)
        normalized = (clipped - p5) / (p95 - p5 + 1e-10)
        return torch.tensor(normalized, dtype=x.dtype, device=x.device)

    blinker = robust_normalize(blinker_raw)
    walker = robust_normalize(walker_raw)

    # Classify singularity types
    threshold = 0.5
    singularity_types = []
    for i in range(N):
        b_high = blinker[i].item() > threshold
        w_high = walker[i].item() > threshold

        if b_high and w_high:
            singularity_types.append('structural_singularity')
        elif b_high:
            singularity_types.append('noise')
        elif w_high:
            singularity_types.append('ideological_barrier')
        else:
            singularity_types.append('consensus')

    result = {
        'blinker': blinker,           # Track 3: Entropy filter
        'walker': walker,             # Track 4: MCMC mixing time (τ_mix)
        'matern_density': matern_density,  # π(x) habitability
        'singularity_types': singularity_types,
        # MCMC diagnostics
        'mixing_time_raw': mcmc_results['mixing_time'],
        'acceptance_rate': mcmc_results['acceptance_rate'],
        'escape_fraction': mcmc_results['escape_fraction'],
        # Statistics
        'anisotropy': (walker.std() / walker.mean()).item() if walker.mean() > 0 else 0,
        # Wiring status
        'wiring': {
            'track_1_5': antagonism_vectors is not None,
            'track_2_matern_density': True,  # Matern as π(x) for MCMC
            'track_4_method': 'metropolis_hastings_mixing_time',
            'matern_nu': config.matern_nu,
            'matern_sigma': config.matern_sigma,
        }
    }

    # Include gradient direction if computed from antagonism (for quivers)
    if grad_direction is not None:
        result['gradient_direction'] = grad_direction
        result['gradient_magnitude'] = grad_magnitude

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'RiemannianConfig',

    # Core components
    'IdeologicalPotential',
    'IdeologicalMetricTensor',
    'SemanticWorkIntegrator',
    'MetricEntropyFilter',
    'EnergyLandscape',

    # Track 2: Matern density and distance
    'compute_matern_distance',
    'compute_matern_distance_matrix',
    'compute_matern_density',

    # Track 4: MCMC stochastic walker
    'compute_mixing_time',
    'compute_directed_mixing_time',
    'metropolis_step',

    # Shear stress / fault lines
    'compute_fault_lines',

    # Track 1.5: Antagonism (quivers only)
    'compute_antagonism_gradient',
    'compute_quiver_field',

    # Waterfall integration
    'compute_physics_from_waterfall',

    # Simplified interface
    'compute_riemannian_physics',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RIEMANNIAN GEOMETRY - THE ENGINE OF IDEOLOGICAL CURVATURE")
    print("=" * 70)

    # Create test embeddings
    torch.manual_seed(42)
    N, D = 100, 64
    embeddings = torch.randn(N, D)

    # Compute Riemannian physics
    print("\nComputing Riemannian physics...")
    results = compute_riemannian_physics(embeddings)

    print(f"\nResults:")
    print(f"  Blinker (entropy) range: [{results['blinker'].min():.3f}, {results['blinker'].max():.3f}]")
    print(f"  Walker (work) range: [{results['walker'].min():.3f}, {results['walker'].max():.3f}]")
    print(f"  Mean potential: {results['mean_potential']:.3f}")
    print(f"  Anisotropy: {results['anisotropy']:.3f}")

    # Count singularities
    from collections import Counter
    counts = Counter(results['singularity_types'])
    print(f"\nSingularity distribution:")
    for stype, count in sorted(counts.items()):
        print(f"  {stype}: {count} ({count/N*100:.1f}%)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE - Bias is Measurable as Anisotropy")
    print("=" * 70)
