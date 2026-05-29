"""
core/physarum_walk.py

Track 4: corpus-level thermodynamic walkers over article neighborhoods.

Primary path:
- Walkers move between article nodes in Track 2 space.
- Transition probability follows exp(-DeltaZ / tau).
- Walkers accumulate fatigue until they hit a cognitive horizon.
- Retreat adds a homing penalty back to the anchor article.

Legacy path:
- The older per-article bot-simplex walker is kept as a compatibility
  helper for focused replay and existing tests.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .hadamard_fusion import ConformalMetric
from .thermo_config import ThermodynamicConfig

TRACK4_PROPOSAL_MODES = {
    "metric_softmax",
    "stress_biased",
    "committor_guided",
    "deterministic_low_cost",
}


def _normalize_proposal_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "metric_softmax").strip().lower().replace("-", "_")
    aliases = {
        "default": "metric_softmax",
        "baseline": "metric_softmax",
        "current": "metric_softmax",
        "softmax": "metric_softmax",
        "metric": "metric_softmax",
        "stress": "stress_biased",
        "stress_bias": "stress_biased",
        "committor": "committor_guided",
        "tpt": "committor_guided",
        "low_cost": "deterministic_low_cost",
        "deterministic": "deterministic_low_cost",
        "greedy": "deterministic_low_cost",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRACK4_PROPOSAL_MODES:
        valid = ", ".join(sorted(TRACK4_PROPOSAL_MODES))
        raise ValueError(f"Unknown Track 4 proposal_mode={mode!r}; expected one of: {valid}")
    return normalized

def _to_float_tensor(value: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to(device=reference.device, dtype=torch.float32)
    return torch.as_tensor(value, device=reference.device, dtype=torch.float32)


def _reduce_density(track3_density: Optional[torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
    if track3_density is None:
        return torch.ones(embeddings.shape[0], device=embeddings.device, dtype=torch.float32)
    density = _to_float_tensor(track3_density, embeddings)
    if density is None:
        return torch.ones(embeddings.shape[0], device=embeddings.device, dtype=torch.float32)
    if density.ndim > 1:
        density = ConformalMetric().compute_density(density)
    density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    if density.shape[0] != embeddings.shape[0]:
        density = torch.ones(embeddings.shape[0], device=embeddings.device, dtype=torch.float32)
    return density.clamp(min=1e-6, max=1.0)


def _build_knn_graph(embeddings: torch.Tensor, k_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n_articles = int(embeddings.shape[0])
    if n_articles <= 1:
        empty = torch.empty((n_articles, 0), dtype=torch.long, device=embeddings.device)
        dmat = torch.zeros((n_articles, n_articles), dtype=torch.float32, device=embeddings.device)
        return empty, dmat
    k = min(max(int(k_neighbors), 1), n_articles - 1)
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    neighbors = torch.topk(distance_matrix, k=k + 1, largest=False).indices[:, 1:]
    return neighbors, distance_matrix


def _estimate_shear_field(
    embeddings: torch.Tensor,
    scalar_stress: torch.Tensor,
    distance_matrix: torch.Tensor,
    k_neighbors: int,
) -> torch.Tensor:
    n_articles = int(embeddings.shape[0])
    if n_articles <= 1:
        return torch.zeros_like(embeddings)

    k = min(max(int(k_neighbors), 1), n_articles - 1)
    neighbor_idx = torch.topk(distance_matrix, k=k + 1, largest=False).indices[:, 1:]
    shear = torch.zeros_like(embeddings)

    for idx in range(n_articles):
        nbrs = neighbor_idx[idx]
        if nbrs.numel() == 0:
            continue
        dx = embeddings[nbrs] - embeddings[idx]
        ds = scalar_stress[nbrs] - scalar_stress[idx]
        dist_sq = (dx * dx).sum(dim=1).clamp(min=1e-6)
        weights = 1.0 / torch.sqrt(dist_sq)
        local_grad = (dx * (ds / dist_sq).unsqueeze(1)) * weights.unsqueeze(1)
        shear[idx] = local_grad.sum(dim=0) / weights.sum().clamp(min=1e-6)

    return torch.nan_to_num(shear, nan=0.0, posinf=0.0, neginf=0.0)


def _build_metric_distance_matrix(
    embeddings: torch.Tensor,
    rho: torch.Tensor,
    shear_vectors: torch.Tensor,
    euclidean_distance_matrix: torch.Tensor,
) -> torch.Tensor:
    euclidean_sq = euclidean_distance_matrix.pow(2)
    ux = shear_vectors @ embeddings.T
    ux_diag = torch.diagonal(ux)
    shear_projection = 0.5 * (ux - ux_diag.unsqueeze(1) + ux_diag.unsqueeze(0) - ux.T)
    rho_mid = 0.5 * (rho.unsqueeze(1) + rho.unsqueeze(0))
    metric_sq = euclidean_sq / rho_mid.clamp(min=1e-6) + shear_projection.pow(2)
    metric_dist = torch.sqrt(metric_sq.clamp(min=0.0))
    metric_dist.fill_diagonal_(0.0)
    return torch.nan_to_num(metric_dist, nan=0.0, posinf=0.0, neginf=0.0)


def _build_metric_graph(
    embeddings: torch.Tensor,
    rho: torch.Tensor,
    scalar_stress: torch.Tensor,
    k_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, euclidean_distance_matrix = _build_knn_graph(embeddings, k_neighbors=k_neighbors)
    shear_vectors = _estimate_shear_field(
        embeddings=embeddings,
        scalar_stress=scalar_stress,
        distance_matrix=euclidean_distance_matrix,
        k_neighbors=k_neighbors,
    )
    metric_distance_matrix = _build_metric_distance_matrix(
        embeddings=embeddings,
        rho=rho,
        shear_vectors=shear_vectors,
        euclidean_distance_matrix=euclidean_distance_matrix,
    )
    n_articles = int(embeddings.shape[0])
    if n_articles <= 1:
        empty = torch.empty((n_articles, 0), dtype=torch.long, device=embeddings.device)
        return empty, metric_distance_matrix, euclidean_distance_matrix, shear_vectors
    k = min(max(int(k_neighbors), 1), n_articles - 1)
    neighbors = torch.topk(metric_distance_matrix, k=k + 1, largest=False).indices[:, 1:]
    return neighbors, metric_distance_matrix, euclidean_distance_matrix, shear_vectors


def _metric_horizon(metric_distance_matrix: torch.Tensor, neighbors: torch.Tensor, default: float = 1.0) -> float:
    if neighbors.numel() == 0:
        return float(default)
    edge_costs = metric_distance_matrix.gather(1, neighbors)
    finite = edge_costs[torch.isfinite(edge_costs) & (edge_costs > 0)]
    if finite.numel() <= 0:
        return float(default)
    return float(torch.quantile(finite, 0.70).item())


def _safe_quantile(values: List[float], q: float, default: float) -> float:
    finite = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float(default)
    return float(np.quantile(finite, q))


def _unit_interval_tensor(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() <= 0:
        return values
    lo = torch.min(values)
    hi = torch.max(values)
    span = (hi - lo).clamp(min=1e-9)
    if float((hi - lo).item()) <= 1e-9:
        return torch.zeros_like(values)
    return ((values - lo) / span).clamp(0.0, 1.0)


def _zone_from_density_stress(density: float, stress: float) -> str:
    if density >= 0.5 and stress < 0.5:
        return "Bridge"
    if density >= 0.5 and stress >= 0.5:
        return "Swamp"
    if density < 0.5 and stress < 0.5:
        return "Tightrope"
    return "Void"


def _safe_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _legacy_observer_penalty(
    weights: torch.Tensor,
    embeddings: torch.Tensor,
    observer_axis: Optional[torch.Tensor],
    observer_cost_strength: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = weights.shape[0]
    if observer_axis is None or observer_cost_strength <= 0.0:
        ones = torch.ones(n, device=weights.device, dtype=torch.float32)
        zeros = torch.zeros(n, device=weights.device, dtype=torch.float32)
        return ones, zeros
    axis = observer_axis.to(device=weights.device, dtype=torch.float32)
    fused = torch.matmul(weights, embeddings)
    similarity = torch.nn.functional.cosine_similarity(fused, axis.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    penalty = 1.0 + float(observer_cost_strength) * torch.clamp(1.0 - similarity, min=0.0)
    return penalty, similarity


def _legacy_compute_energy(
    weights: torch.Tensor,
    embeddings: torch.Tensor,
    gradients: torch.Tensor,
    u_axis: Optional[torch.Tensor],
    observer_axis: Optional[torch.Tensor],
    observer_cost_strength: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_grad = torch.matmul(weights, gradients)
    if u_axis is not None:
        axis = u_axis.to(device=weights.device, dtype=torch.float32)
        energy = torch.abs((fused_grad * axis.unsqueeze(0)).sum(dim=-1))
    else:
        energy = torch.norm(fused_grad, p=2, dim=-1)
    penalty, similarity = _legacy_observer_penalty(weights, embeddings, observer_axis, observer_cost_strength)
    if observer_axis is not None and observer_cost_strength > 0.0:
        energy = energy + torch.clamp(penalty - 1.0, min=0.0)
    return energy, penalty, similarity


def compute_walker_resistance(
    cls_per_bot: torch.Tensor,
    rks_basis,
    u_axis: Optional[torch.Tensor] = None,
    observer_axis: Optional[torch.Tensor] = None,
    observer_cost_strength: float = 0.0,
    temperature: float = 0.5,
    n_walkers: int = 20,
    n_steps: int = 10,
    thermo_config: Optional[ThermodynamicConfig] = None,
    start_seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Legacy per-article simplex walker kept for focused replay compatibility.
    """
    embeddings = cls_per_bot.detach().to(dtype=torch.float32)
    gradients = embeddings - embeddings.mean(dim=0, keepdim=True)
    thermo = thermo_config or ThermodynamicConfig()
    if start_seed is not None:
        torch.manual_seed(int(start_seed))

    n_bots = int(embeddings.shape[0])
    weights = torch.ones((int(n_walkers), n_bots), device=embeddings.device, dtype=torch.float32) / max(n_bots, 1)
    trajectory = [weights.clone()]
    step_work_rows: List[torch.Tensor] = []
    step_diagnostics: List[Dict[str, Any]] = []
    cumulative_work = torch.zeros(int(n_walkers), device=embeddings.device, dtype=torch.float32)

    for _step in range(int(max(n_steps, 1))):
        noise = torch.randn_like(weights) * float(thermo.noise_sigma)
        proposal = torch.abs(weights + noise)
        proposal = proposal / proposal.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        current_energy, current_penalty, current_similarity = _legacy_compute_energy(
            weights,
            embeddings,
            gradients,
            u_axis,
            observer_axis,
            observer_cost_strength,
        )
        proposal_energy, proposal_penalty, proposal_similarity = _legacy_compute_energy(
            proposal,
            embeddings,
            gradients,
            u_axis,
            observer_axis,
            observer_cost_strength,
        )
        accept_prob = torch.exp(-(proposal_energy - current_energy) / max(float(temperature), 1e-6))
        accept = torch.rand(int(n_walkers), device=embeddings.device) < accept_prob
        weights = torch.where(accept.unsqueeze(-1), proposal, weights)

        prev_pos = torch.matmul(trajectory[-1], embeddings)
        next_pos = torch.matmul(weights, embeddings)
        mid = (trajectory[-1] + weights) / 2.0
        fused_mid_grad = torch.matmul(mid, gradients)
        density = 1.0 / (1.0 + torch.norm(fused_mid_grad, p=2, dim=-1).clamp(max=1000.0))
        mean_penalty, mean_similarity = _legacy_observer_penalty(mid, embeddings, observer_axis, observer_cost_strength)
        local_friction = (float(thermo.friction_coefficient) / density.clamp(min=float(thermo.density_clamp_min))) * mean_penalty
        dist = torch.norm(next_pos - prev_pos, p=2, dim=-1)
        step_work = local_friction * dist
        cumulative_work = cumulative_work + step_work

        trajectory.append(weights.clone())
        step_work_rows.append(step_work)
        step_diagnostics.append(
            {
                "local_friction": float(local_friction.mean().item()),
                "step_work": float(step_work.mean().item()),
                "cumulative_work": float(cumulative_work.mean().item()),
                "observer_penalty": float(mean_penalty.mean().item()),
                "observer_similarity": float(mean_similarity.mean().item()),
            }
        )

    traj = torch.stack(trajectory, dim=0)
    path_positions = torch.matmul(traj, embeddings)
    path_xyz = rks_basis(path_positions.reshape(-1, path_positions.shape[-1])).view(path_positions.shape[0], path_positions.shape[1], -1)
    work = torch.stack(step_work_rows, dim=0).sum(dim=0) if step_work_rows else torch.zeros(int(n_walkers), device=embeddings.device)
    spectral_distance = torch.norm(path_positions[-1] - path_positions[0], p=2, dim=-1)
    divergence_ratio = work / spectral_distance.clamp(min=1e-6)

    return {
        "status": "LEGACY_PATH",
        "work_integral": float(work.mean().item()) if work.numel() else 0.0,
        "divergence_ratio": float(divergence_ratio.mean().item()) if divergence_ratio.numel() else 0.0,
        "spectral_distance": float(spectral_distance.mean().item()) if spectral_distance.numel() else 0.0,
        "closed_loop": False,
        "walker_output": path_xyz[-1].mean(dim=0),
        "path_xyz": path_xyz.mean(dim=1),
        "final_position": path_positions[-1].mean(dim=0),
        "observer_cost_strength": float(observer_cost_strength),
        "step_diagnostics": step_diagnostics,
    }


class SemanticWalker:
    """
    Manuscript-aligned Track 4 walker over the article manifold.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        rks_basis,
        temperature: float = 0.5,
        track3_density: Optional[torch.Tensor] = None,
        z_coordinates: Optional[torch.Tensor] = None,
        metric_stress: Optional[torch.Tensor] = None,
        article_coords_2d: Optional[torch.Tensor] = None,
        thermo_config: Optional[ThermodynamicConfig] = None,
        gradients: Optional[torch.Tensor] = None,
        u_axis: Optional[torch.Tensor] = None,
    ):
        self.embeddings = embeddings.detach().to(dtype=torch.float32)
        self.kernel = rks_basis
        self.temperature = float(temperature)
        self.thermo_config = thermo_config or ThermodynamicConfig()
        self.gradients = _to_float_tensor(gradients, self.embeddings)
        if self.gradients is None:
            self.gradients = self.embeddings - self.embeddings.mean(dim=0, keepdim=True)
        self.u_axis = _to_float_tensor(u_axis, self.embeddings)
        self.track3_density = _reduce_density(track3_density, self.embeddings)
        self.z_coordinates = _to_float_tensor(z_coordinates, self.embeddings)
        self.metric_stress = _to_float_tensor(metric_stress, self.embeddings)
        if self.metric_stress is None:
            if self.z_coordinates is not None:
                self.metric_stress = self.z_coordinates.abs().clone()
            else:
                self.metric_stress = torch.zeros(self.embeddings.shape[0], device=self.embeddings.device, dtype=torch.float32)
        elif self.metric_stress.ndim > 1:
            self.metric_stress = self.metric_stress.norm(dim=-1)
        if self.z_coordinates is None:
            self.z_coordinates = self.metric_stress.clone()
        elif self.z_coordinates.ndim > 1:
            self.z_coordinates = self.z_coordinates.norm(dim=-1)
        self.article_coords_2d = _to_float_tensor(article_coords_2d, self.embeddings)
        self._last_catalyst_selection: Dict[str, Any] = {}

    def _selection_coords(self) -> torch.Tensor:
        if self.article_coords_2d is not None and self.article_coords_2d.ndim == 2 and self.article_coords_2d.shape[0] == self.embeddings.shape[0]:
            return self.article_coords_2d
        if self.embeddings.shape[1] >= 2:
            return self.embeddings[:, :2]
        if self.embeddings.shape[1] == 1:
            zeros = torch.zeros((self.embeddings.shape[0], 1), device=self.embeddings.device, dtype=torch.float32)
            return torch.cat([self.embeddings[:, :1], zeros], dim=1)
        return torch.zeros((self.embeddings.shape[0], 2), device=self.embeddings.device, dtype=torch.float32)

    def _terrain_fields(self) -> Dict[str, Any]:
        density = _unit_interval_tensor(self.track3_density.reshape(-1))
        stress = _unit_interval_tensor(torch.abs(self.metric_stress.reshape(-1)))
        n_articles = int(self.embeddings.shape[0])
        if density.shape[0] != n_articles:
            density = torch.zeros(n_articles, device=self.embeddings.device, dtype=torch.float32)
        if stress.shape[0] != n_articles:
            stress = torch.zeros(n_articles, device=self.embeddings.device, dtype=torch.float32)

        labels = [
            _zone_from_density_stress(float(density[idx].item()), float(stress[idx].item()))
            for idx in range(n_articles)
        ]
        zone_scores = torch.zeros(n_articles, device=self.embeddings.device, dtype=torch.float32)
        for idx, label in enumerate(labels):
            d = density[idx]
            s = stress[idx]
            if label == "Bridge":
                score = d * (1.0 - s)
            elif label == "Swamp":
                score = d * s
            elif label == "Tightrope":
                score = (1.0 - d) * (1.0 - s)
            else:
                score = (1.0 - d) * s
            zone_scores[idx] = torch.clamp(score, min=0.0, max=1.0)
        return {
            "density": density,
            "stress": stress,
            "labels": labels,
            "zone_scores": zone_scores,
        }

    def _candidate_distance_score(self, candidate: int, selected: List[int], coords: torch.Tensor) -> float:
        if not selected:
            return 1.0
        selected_tensor = torch.as_tensor(selected, device=coords.device, dtype=torch.long)
        distances = torch.norm(coords[int(candidate)].unsqueeze(0) - coords[selected_tensor], p=2, dim=1)
        max_span = torch.norm(coords.max(dim=0).values - coords.min(dim=0).values, p=2).clamp(min=1e-6)
        return float((distances.min() / max_span).clamp(0.0, 1.0).item())

    def _pick_zone_candidate(
        self,
        zone: str,
        labels: List[str],
        zone_scores: torch.Tensor,
        coords: torch.Tensor,
        selected: List[int],
        *,
        distance_weight: float = 0.35,
    ) -> Optional[int]:
        candidates = [idx for idx, label in enumerate(labels) if label == zone and idx not in selected]
        if not candidates:
            return None
        best_idx = None
        best_score = -1.0
        for idx in candidates:
            score = (1.0 - distance_weight) * float(zone_scores[idx].item())
            score += distance_weight * self._candidate_distance_score(idx, selected, coords)
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        return best_idx

    def _compute_base_friction(self, weights: torch.Tensor) -> torch.Tensor:
        fused_grad = torch.matmul(weights, self.gradients)
        if self.u_axis is not None:
            axis = self.u_axis.to(device=weights.device, dtype=torch.float32)
            grad_magnitude = torch.abs((fused_grad * axis.unsqueeze(0)).sum(dim=-1))
        else:
            grad_magnitude = torch.norm(fused_grad, p=2, dim=-1)
        grad_magnitude = torch.clamp(grad_magnitude, max=1000.0)
        density = 1.0 / (1.0 + grad_magnitude)
        density = torch.clamp(density, min=float(self.thermo_config.density_clamp_min))
        return 1.0 / density

    def _compute_effective_friction(self, base_friction: torch.Tensor) -> torch.Tensor:
        return torch.clamp(base_friction, min=0.0, max=1000.0)

    def compute_work_integral(
        self,
        trajectory_weights: torch.Tensor,
        precomputed_step_work: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compatibility helper for older stability tests and offline diagnostics.

        Operates in the bot-weight simplex using the current embedding/gradient
        payload, while keeping friction finite in low-density voids.
        """
        positions = torch.matmul(trajectory_weights, self.embeddings)
        deltas = positions[1:] - positions[:-1]
        step_distances = torch.norm(deltas, p=2, dim=-1)

        if precomputed_step_work is not None:
            step_work = precomputed_step_work
        else:
            midpoint_weights = (trajectory_weights[1:] + trajectory_weights[:-1]) / 2.0
            base_friction = self._compute_base_friction(midpoint_weights)
            terrain_friction = self._compute_effective_friction(base_friction)
            step_work = terrain_friction * step_distances

        work = step_work.sum(dim=0)
        path_length = step_distances.sum(dim=0)
        spectral_distance = torch.norm(positions[-1] - positions[0], p=2, dim=-1)
        divergence_ratio = work / spectral_distance.clamp(min=1e-6)
        return work, path_length, spectral_distance, divergence_ratio

    def select_catalysts(self) -> List[int]:
        n_articles = int(self.embeddings.shape[0])
        if n_articles <= 0:
            self._last_catalyst_selection = {
                "policy": "zone_constrained_farthest_first_v1",
                "coverage_status": "NO_ARTICLES",
                "selected_indices": [],
                "selected_zones": [],
                "available_zone_counts": {},
            }
            return []
        if n_articles <= 3:
            terrain = self._terrain_fields()
            selected = list(range(n_articles))
            self._last_catalyst_selection = {
                "policy": "zone_constrained_farthest_first_v1",
                "coverage_status": "LIMITED_SMALL_CORPUS",
                "selected_indices": selected,
                "selected_zones": [terrain["labels"][idx] for idx in selected],
                "available_zone_counts": {
                    zone: int(terrain["labels"].count(zone))
                    for zone in sorted(set(terrain["labels"]))
                },
                "density": [float(terrain["density"][idx].item()) for idx in selected],
                "stress": [float(terrain["stress"][idx].item()) for idx in selected],
                "terrain_labels": terrain["labels"],
            }
            return list(range(n_articles))
        target_count = min(3, n_articles)
        coords = self._selection_coords()
        terrain = self._terrain_fields()
        labels: List[str] = terrain["labels"]
        zone_scores: torch.Tensor = terrain["zone_scores"]
        available_zone_counts = {
            zone: int(labels.count(zone))
            for zone in sorted(set(labels))
        }
        catalysts: List[int] = []

        required_priority = ["Void", "Bridge"]
        for zone in required_priority:
            if len(catalysts) >= target_count:
                break
            picked = self._pick_zone_candidate(zone, labels, zone_scores, coords, catalysts, distance_weight=0.25)
            if picked is not None:
                catalysts.append(picked)

        remaining_zones = [
            zone for zone in ("Swamp", "Tightrope", "Void", "Bridge")
            if zone in available_zone_counts and zone not in {labels[idx] for idx in catalysts}
        ]
        while len(catalysts) < target_count and remaining_zones:
            best_zone = None
            best_candidate = None
            best_score = -1.0
            for zone in remaining_zones:
                candidate = self._pick_zone_candidate(zone, labels, zone_scores, coords, catalysts, distance_weight=0.45)
                if candidate is None:
                    continue
                score = 0.6 * float(zone_scores[candidate].item())
                score += 0.4 * self._candidate_distance_score(candidate, catalysts, coords)
                if score > best_score:
                    best_score = score
                    best_zone = zone
                    best_candidate = candidate
            if best_candidate is None or best_zone is None:
                break
            catalysts.append(int(best_candidate))
            remaining_zones = [zone for zone in remaining_zones if zone != best_zone]

        while len(catalysts) < target_count:
            remaining = [idx for idx in range(n_articles) if idx not in catalysts]
            if not remaining:
                break
            if catalysts:
                remaining_tensor = torch.as_tensor(remaining, device=coords.device, dtype=torch.long)
                selected = coords[torch.as_tensor(catalysts, device=coords.device, dtype=torch.long)]
                distances = torch.cdist(coords[remaining_tensor], selected, p=2)
                candidate = int(remaining[int(torch.argmax(distances.min(dim=1).values).item())])
            else:
                candidate = int(torch.argmax(zone_scores).item())
            catalysts.append(candidate)

        selected_zones = [labels[idx] for idx in catalysts]
        required_present = all(zone not in available_zone_counts or zone in selected_zones for zone in required_priority)
        distinct_available = min(target_count, len(available_zone_counts))
        coverage_status = "OK"
        if len(set(selected_zones)) < distinct_available or not required_present:
            coverage_status = "LIMITED_ZONE_COVERAGE"
        self._last_catalyst_selection = {
            "policy": "zone_constrained_farthest_first_v1",
            "coverage_status": coverage_status,
            "selected_indices": [int(idx) for idx in catalysts],
            "selected_zones": selected_zones,
            "available_zone_counts": available_zone_counts,
            "density": [float(terrain["density"][idx].item()) for idx in catalysts],
            "stress": [float(terrain["stress"][idx].item()) for idx in catalysts],
            "terrain_labels": labels,
        }
        return catalysts

    def _build_transition_matrix(self, neighbors: torch.Tensor, metric_distance_matrix: torch.Tensor) -> np.ndarray:
        n_articles = int(self.embeddings.shape[0])
        transition = np.zeros((n_articles, n_articles), dtype=np.float64)
        temperature = max(float(self.temperature), 1e-6)
        for row_idx in range(n_articles):
            candidate_tensor = neighbors[row_idx]
            if candidate_tensor.numel() == 0:
                transition[row_idx, row_idx] = 1.0
                continue
            candidate_indices = candidate_tensor.detach().cpu().numpy().astype(np.int64)
            costs = metric_distance_matrix[row_idx, candidate_tensor].detach().cpu().numpy().astype(np.float64)
            logits = np.exp(-np.clip(costs / temperature, 0.0, 50.0))
            total = float(np.sum(logits))
            if not np.isfinite(total) or total <= 0.0:
                transition[row_idx, row_idx] = 1.0
                continue
            transition[row_idx, candidate_indices] = logits / total
        return transition

    def _mfpt_to_target(self, transition: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
        n_articles = transition.shape[0]
        mfpt = np.full(n_articles, np.nan, dtype=np.float64)
        if target_indices.size == 0:
            return mfpt
        target_mask = np.zeros(n_articles, dtype=bool)
        target_mask[target_indices] = True
        mfpt[target_mask] = 0.0
        interior = np.where(~target_mask)[0]
        if interior.size == 0:
            return mfpt
        pii = transition[np.ix_(interior, interior)]
        matrix = np.eye(interior.size, dtype=np.float64) - pii
        rhs = np.ones(interior.size, dtype=np.float64)
        mfpt[interior] = np.maximum(_safe_solve(matrix, rhs), 0.0)
        return mfpt

    def _dominant_flux_path(
        self,
        net_flux: np.ndarray,
        source_indices: np.ndarray,
        sink_indices: np.ndarray,
    ) -> List[int]:
        if source_indices.size == 0 or sink_indices.size == 0:
            return []
        node_flux = np.sum(net_flux, axis=1)
        source = int(source_indices[int(np.argmax(node_flux[source_indices]))])
        sinks = set(int(idx) for idx in sink_indices.tolist())
        path = [source]
        visited = {source}
        current = source
        for _ in range(max(1, net_flux.shape[0])):
            if current in sinks:
                break
            row = net_flux[current].copy()
            for seen in visited:
                row[seen] = 0.0
            next_idx = int(np.argmax(row))
            if float(row[next_idx]) <= 0.0:
                break
            path.append(next_idx)
            visited.add(next_idx)
            current = next_idx
            if current in sinks:
                break
        return path

    def _has_directed_reachability(
        self,
        transition: np.ndarray,
        source_indices: np.ndarray,
        sink_indices: np.ndarray,
    ) -> bool:
        if source_indices.size == 0 or sink_indices.size == 0:
            return False
        adjacency = transition > 0.0
        sinks = set(int(idx) for idx in sink_indices.tolist())
        frontier = [int(idx) for idx in source_indices.tolist()]
        visited = set(frontier)
        while frontier:
            current = frontier.pop()
            if current in sinks:
                return True
            for next_idx in np.flatnonzero(adjacency[current]):
                nxt = int(next_idx)
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)
        return False

    def _compute_markov_observables(
        self,
        neighbors: torch.Tensor,
        metric_distance_matrix: torch.Tensor,
        terrain_labels: List[str],
    ) -> Dict[str, Any]:
        n_articles = int(self.embeddings.shape[0])
        transition = self._build_transition_matrix(neighbors, metric_distance_matrix)
        bridge_indices = np.asarray([idx for idx, label in enumerate(terrain_labels) if label == "Bridge"], dtype=np.int32)
        void_indices = np.asarray([idx for idx, label in enumerate(terrain_labels) if label == "Void"], dtype=np.int32)
        bridge_to_void_reachable = self._has_directed_reachability(transition, bridge_indices, void_indices)
        void_to_bridge_reachable = self._has_directed_reachability(transition, void_indices, bridge_indices)
        empty_float = np.full(n_articles, np.nan, dtype=np.float32)
        if bridge_indices.size == 0 or void_indices.size == 0:
            return {
                "status": "NO_BOUNDARY_SETS",
                "committor_to_void": empty_float,
                "mfpt_to_bridge": empty_float.copy(),
                "mfpt_to_void": empty_float.copy(),
                "reactive_flux_edges": np.empty((0, 2), dtype=np.int32),
                "reactive_flux_values": np.empty((0,), dtype=np.float32),
                "reactive_flux_node_throughput": np.zeros(n_articles, dtype=np.float32),
                "dominant_reactive_path_indices": np.empty((0,), dtype=np.int32),
                "bridge_indices": bridge_indices,
                "void_indices": void_indices,
                "summary": {
                    "status": "NO_BOUNDARY_SETS",
                    "bridge_count": int(bridge_indices.size),
                    "void_count": int(void_indices.size),
                    "bridge_to_void_reachable": bool(bridge_to_void_reachable),
                    "void_to_bridge_reachable": bool(void_to_bridge_reachable),
                    "reactive_flux_total": 0.0,
                    "dominant_reactive_path_length": 0,
                },
            }

        boundary = np.concatenate([bridge_indices, void_indices]).astype(np.int32)
        boundary_mask = np.zeros(n_articles, dtype=bool)
        boundary_mask[boundary] = True
        q = np.zeros(n_articles, dtype=np.float64)
        q[void_indices] = 1.0
        interior = np.where(~boundary_mask)[0]
        if interior.size:
            pii = transition[np.ix_(interior, interior)]
            pib = transition[np.ix_(interior, boundary)]
            matrix = np.eye(interior.size, dtype=np.float64) - pii
            rhs = pib @ q[boundary]
            q[interior] = _safe_solve(matrix, rhs)
        q = np.clip(np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        mfpt_to_bridge = self._mfpt_to_target(transition, bridge_indices)
        mfpt_to_void = self._mfpt_to_target(transition, void_indices)

        stationary = np.ones(n_articles, dtype=np.float64) / max(n_articles, 1)
        for _ in range(500):
            next_stationary = stationary @ transition
            if np.max(np.abs(next_stationary - stationary)) <= 1e-10:
                stationary = next_stationary
                break
            stationary = next_stationary
        stationary = np.clip(np.nan_to_num(stationary, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        if float(stationary.sum()) > 0.0:
            stationary = stationary / stationary.sum()

        q_backward_approx = 1.0 - q
        raw_flux = stationary[:, None] * transition * q_backward_approx[:, None] * q[None, :]
        net_flux = np.maximum(raw_flux - raw_flux.T, 0.0)
        edge_rows, edge_cols = np.nonzero(net_flux > 0.0)
        edge_values = net_flux[edge_rows, edge_cols]
        if edge_values.size:
            order = np.argsort(edge_values)[::-1]
            edge_rows = edge_rows[order]
            edge_cols = edge_cols[order]
            edge_values = edge_values[order]
        flux_edges = np.stack([edge_rows, edge_cols], axis=1).astype(np.int32) if edge_values.size else np.empty((0, 2), dtype=np.int32)
        node_throughput = (net_flux.sum(axis=0) + net_flux.sum(axis=1)).astype(np.float32)
        dominant_path = self._dominant_flux_path(net_flux, bridge_indices, void_indices)

        finite_void_mfpt = mfpt_to_void[np.isfinite(mfpt_to_void)]
        finite_bridge_mfpt = mfpt_to_bridge[np.isfinite(mfpt_to_bridge)]
        summary = {
            "status": "OK",
            "bridge_count": int(bridge_indices.size),
            "void_count": int(void_indices.size),
            "committor_mean": float(np.mean(q)) if q.size else None,
            "mfpt_to_bridge_mean": float(np.mean(finite_bridge_mfpt)) if finite_bridge_mfpt.size else None,
            "mfpt_to_void_mean": float(np.mean(finite_void_mfpt)) if finite_void_mfpt.size else None,
            "reactive_flux_total": float(np.sum(edge_values)) if edge_values.size else 0.0,
            "reactive_flux_edge_count": int(edge_values.size),
            "dominant_reactive_path_length": int(len(dominant_path)),
            "dominant_reactive_path_indices": [int(idx) for idx in dominant_path],
            "transition_temperature": float(max(float(self.temperature), 1e-6)),
            "bridge_to_void_reachable": bool(bridge_to_void_reachable),
            "void_to_bridge_reachable": bool(void_to_bridge_reachable),
        }
        return {
            "status": "OK",
            "committor_to_void": q.astype(np.float32),
            "mfpt_to_bridge": mfpt_to_bridge.astype(np.float32),
            "mfpt_to_void": mfpt_to_void.astype(np.float32),
            "reactive_flux_edges": flux_edges,
            "reactive_flux_values": edge_values.astype(np.float32),
            "reactive_flux_node_throughput": node_throughput,
            "dominant_reactive_path_indices": np.asarray(dominant_path, dtype=np.int32),
            "bridge_indices": bridge_indices,
            "void_indices": void_indices,
            "summary": summary,
        }

    def _markov_supports_bridge_void_flux(self, markov_observables: Dict[str, Any]) -> bool:
        summary = markov_observables.get("summary") or {}
        if str(markov_observables.get("status", "")).upper() != "OK":
            return False
        if int(summary.get("bridge_count") or 0) <= 0 or int(summary.get("void_count") or 0) <= 0:
            return False
        if summary.get("bridge_to_void_reachable") is False:
            return False
        return float(summary.get("reactive_flux_total") or 0.0) > 0.0

    def _connectivity_repair_candidates(self, requested_k: int, n_articles: int) -> List[int]:
        max_k = max(1, int(n_articles) - 1)
        requested = min(max(int(requested_k), 1), max_k)
        raw = [
            requested + 1,
            max(requested + 1, 8),
            max(requested + 1, 10),
            max(requested + 1, int(np.ceil(0.25 * max_k))),
            max(requested + 1, int(np.ceil(0.50 * max_k))),
            max_k,
        ]
        candidates = sorted({min(max(int(k), 1), max_k) for k in raw if int(k) > requested})
        return candidates

    def _simulate_anchor(
        self,
        anchor_idx: int,
        neighbors: torch.Tensor,
        metric_distance_matrix: torch.Tensor,
        euclidean_distance_matrix: torch.Tensor,
        shear_vectors: torch.Tensor,
        n_walkers: int,
        max_steps: int,
        gamma: float,
        hot_temperature_multiplier: float,
        rng: np.random.Generator,
        s_max: float,
        proposal_mode: str = "metric_softmax",
        committor_to_void: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        proposal_mode = _normalize_proposal_mode(proposal_mode)
        walker_runs: List[Dict[str, Any]] = []
        stress_unit = _unit_interval_tensor(torch.abs(self.metric_stress.reshape(-1)))
        committor_arr: Optional[np.ndarray] = None
        if committor_to_void is not None:
            candidate = np.asarray(committor_to_void, dtype=np.float64).reshape(-1)
            if candidate.shape[0] == self.embeddings.shape[0]:
                committor_arr = np.clip(
                    np.nan_to_num(candidate, nan=0.5, posinf=1.0, neginf=0.0),
                    0.0,
                    1.0,
                )

        for walker_idx in range(max(int(n_walkers), 1)):
            is_hot = walker_idx < 2
            current_idx = int(anchor_idx)
            current_temp = self.temperature * (float(hot_temperature_multiplier) if is_hot else 1.0)
            current_temp = max(current_temp, 1e-6)
            path = [current_idx]
            cumulative_work = 0.0
            max_anchor_distance = 0.0
            in_retreat = False
            step_diagnostics: List[Dict[str, Any]] = []

            for _step in range(max(int(max_steps), 1)):
                candidate_tensor = neighbors[current_idx]
                if candidate_tensor.numel() == 0:
                    break

                candidate_indices = candidate_tensor.detach().cpu().numpy()
                step_delta = metric_distance_matrix[current_idx, candidate_tensor].detach().cpu().numpy()
                logits = np.exp(-np.clip(step_delta / current_temp, 0.0, 50.0))
                anchor_dist = metric_distance_matrix[anchor_idx, candidate_tensor].detach().cpu().numpy()
                proposal_note = proposal_mode
                if proposal_mode == "deterministic_low_cost":
                    effective_cost = np.asarray(step_delta, dtype=np.float64)
                    if in_retreat:
                        effective_cost = effective_cost + float(gamma) * np.asarray(anchor_dist, dtype=np.float64)
                    next_idx = int(candidate_indices[int(np.argmin(effective_cost))])
                else:
                    if proposal_mode == "stress_biased" and not in_retreat:
                        candidate_stress = stress_unit[candidate_tensor].detach().cpu().numpy().astype(np.float64)
                        logits *= np.exp(np.clip(1.25 * candidate_stress, -5.0, 5.0))
                    elif proposal_mode == "committor_guided" and committor_arr is not None and not in_retreat:
                        q_current = float(committor_arr[current_idx])
                        q_candidates = committor_arr[candidate_indices.astype(np.int64)]
                        if np.isfinite(q_current) and np.isfinite(q_candidates).any():
                            direction = 1.0 if q_current < 0.5 else -1.0
                            logits *= np.exp(np.clip(1.75 * direction * (q_candidates - q_current), -5.0, 5.0))
                    elif proposal_mode == "committor_guided" and committor_arr is None:
                        proposal_note = "committor_guided_no_boundary_fallback"

                    if in_retreat:
                        logits *= np.exp(-np.clip(float(gamma) * anchor_dist, 0.0, 50.0))
                    prob_sum = float(logits.sum())
                    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
                        break

                    probs = logits / prob_sum
                    next_idx = int(rng.choice(candidate_indices, p=probs))
                step_work = float(metric_distance_matrix[current_idx, next_idx].item())
                cumulative_work += step_work
                max_anchor_distance = max(max_anchor_distance, float(metric_distance_matrix[anchor_idx, next_idx].item()))

                step_vector = (self.embeddings[next_idx] - self.embeddings[current_idx]).detach().cpu().numpy().astype(np.float32)
                rho_mid = 0.5 * float(self.track3_density[current_idx].item() + self.track3_density[next_idx].item())
                shear_mid = 0.5 * (shear_vectors[current_idx] + shear_vectors[next_idx])
                shear_projection = float(torch.dot(shear_mid, self.embeddings[next_idx] - self.embeddings[current_idx]).item())
                step_diagnostics.append(
                    {
                        "step_axis_idx": int(next_idx),
                        "step_axis_vector": step_vector,
                        "local_friction": float(step_work),
                        "step_work": float(step_work),
                        "cumulative_work": float(cumulative_work),
                        "debt_axis": float(metric_distance_matrix[anchor_idx, next_idx].item()),
                        "metric_distance": float(step_work),
                        "euclidean_distance": float(euclidean_distance_matrix[current_idx, next_idx].item()),
                        "density_midpoint": float(rho_mid),
                        "shear_projection": float(shear_projection),
                        "proposal_mode": proposal_note,
                        "memory_integral": 0.0,
                        "event_active": bool(in_retreat),
                        "event_severity": float(step_work / max(s_max, 1e-6)),
                    }
                )

                current_idx = next_idx
                path.append(current_idx)

                if (not in_retreat) and cumulative_work >= s_max:
                    in_retreat = True

                if in_retreat and current_idx == anchor_idx and len(path) > 1:
                    break

            closed_loop = bool(in_retreat and current_idx == anchor_idx and len(path) > 1)
            walker_runs.append(
                {
                    "is_hot": bool(is_hot),
                    "closed_loop": closed_loop,
                    "path_indices": path,
                    "work_integral": float(cumulative_work),
                    "spectral_distance": float(max_anchor_distance),
                    "divergence_ratio": float(cumulative_work / max(max_anchor_distance, 1e-6)),
                    "proposal_mode": proposal_mode,
                    "step_diagnostics": step_diagnostics,
                }
            )

        successful = [run for run in walker_runs if run["closed_loop"]]
        cold_successful = [run for run in successful if not run["is_hot"]]
        if cold_successful:
            selected = min(cold_successful, key=lambda run: run["work_integral"])
        elif successful:
            selected = min(successful, key=lambda run: run["work_integral"])
        else:
            selected = max(
                walker_runs,
                key=lambda run: (run["spectral_distance"], run["work_integral"], len(run["path_indices"])),
            )

        path_tensor = self.embeddings[selected["path_indices"]]
        centroid = path_tensor.mean(dim=0) if path_tensor.numel() > 0 else self.embeddings[anchor_idx]
        return {
            "anchor_idx": int(anchor_idx),
            "work_integral": float(selected["work_integral"]),
            "spectral_distance": float(selected["spectral_distance"]),
            "divergence_ratio": float(selected["divergence_ratio"]),
            "path_indices": list(selected["path_indices"]),
            "path_xyz": path_tensor.detach().cpu().numpy().astype(np.float32),
            "walker_output": centroid.detach().clone(),
            "final_position": self.embeddings[selected["path_indices"][-1]].detach().clone(),
            "step_diagnostics": selected["step_diagnostics"],
            "closed_loop": bool(selected["closed_loop"]),
            "proposal_mode": proposal_mode,
            "all_runs": walker_runs,
        }

    def _export_cyclic_paths(
        self,
        swarm_records: List[Dict[str, Any]],
        path_anchor_idx: List[int],
        path_is_hot: List[bool],
        output_dir: Optional[str],
        markov_observables: Optional[Dict[str, Any]] = None,
        feature_basis: str = "track2",
    ) -> None:
        if not output_dir or not swarm_records:
            return

        path_xyz = [np.asarray(record["path_xyz"], dtype=np.float32) for record in swarm_records]
        path_indices = [np.asarray(record.get("path_indices", []), dtype=np.int32) for record in swarm_records]
        work_integral = np.asarray([float(record["work_integral"]) for record in swarm_records], dtype=np.float32)
        closed_loop = np.asarray([bool(record["closed_loop"]) for record in swarm_records], dtype=np.bool_)
        path_proposal_mode = np.asarray(
            [str(record.get("proposal_mode", "metric_softmax")) for record in swarm_records],
            dtype=object,
        )
        path_feature_basis = np.asarray(
            [str(record.get("feature_basis", feature_basis)) for record in swarm_records],
            dtype=object,
        )
        path_anchor_idx_arr = np.asarray([int(idx) for idx in path_anchor_idx], dtype=np.int32)
        path_is_hot_arr = np.asarray([bool(flag) for flag in path_is_hot], dtype=np.bool_)
        selection = self._last_catalyst_selection or {}
        terrain_labels = [str(label) for label in selection.get("terrain_labels", [])]
        path_anchor_terrain = np.asarray(
            [
                terrain_labels[int(idx)] if terrain_labels and 0 <= int(idx) < len(terrain_labels) else ""
                for idx in path_anchor_idx
            ],
            dtype=object,
        )

        anchor_index_to_runs: Dict[int, List[Dict[str, Any]]] = {}
        for record, anchor_idx, is_hot in zip(swarm_records, path_anchor_idx, path_is_hot):
            mechanical_record = dict(record)
            mechanical_record["is_hot_walker"] = bool(is_hot)
            anchor_index_to_runs.setdefault(int(anchor_idx), []).append(mechanical_record)

        anchor_indices = np.asarray(sorted(anchor_index_to_runs.keys()), dtype=np.int32)
        anchor_terrain_label = np.asarray(
            [
                terrain_labels[int(idx)] if terrain_labels and 0 <= int(idx) < len(terrain_labels) else ""
                for idx in anchor_indices.tolist()
            ],
            dtype=object,
        )
        anchor_summaries: List[str] = []
        for anchor_idx in anchor_indices.tolist():
            anchor_runs = anchor_index_to_runs.get(anchor_idx, [])
            hot_runs = [run for run in anchor_runs if bool(run["is_hot_walker"])]
            cold_runs = [run for run in anchor_runs if not bool(run["is_hot_walker"])]
            hot_survived = sum(1 for run in hot_runs if bool(run["closed_loop"]))
            cold_survived = sum(1 for run in cold_runs if bool(run["closed_loop"]))
            anchor_summaries.append(
                f"{hot_survived}/{max(len(hot_runs), 1)} Hot Walkers Tunneled | "
                f"{cold_survived}/{max(len(cold_runs), 1)} Cold Walkers Closed Loop"
            )

        os.makedirs(output_dir, exist_ok=True)
        save_payload: Dict[str, Any] = {
            "path_xyz": np.array(path_xyz, dtype=object),
            "path_indices": np.array(path_indices, dtype=object),
            "work_integral": work_integral,
            "closed_loop": closed_loop,
            "path_proposal_mode": path_proposal_mode,
            "path_feature_basis": path_feature_basis,
            "path_anchor_idx": path_anchor_idx_arr,
            "path_is_hot": path_is_hot_arr,
            "anchor_indices": anchor_indices,
            "anchor_summary": np.array(anchor_summaries, dtype=object),
            "anchor_terrain_label": anchor_terrain_label,
            "path_anchor_terrain_label": path_anchor_terrain,
            "anchor_selection_policy": np.asarray(
                [str(selection.get("policy", "zone_constrained_farthest_first_v1"))],
                dtype=object,
            ),
            "anchor_selection_metadata": np.asarray(
                [json.dumps(selection, sort_keys=True)],
                dtype=object,
            ),
        }
        if markov_observables:
            for key in (
                "committor_to_void",
                "mfpt_to_bridge",
                "mfpt_to_void",
                "reactive_flux_edges",
                "reactive_flux_values",
                "reactive_flux_node_throughput",
                "dominant_reactive_path_indices",
                "bridge_indices",
                "void_indices",
            ):
                if key in markov_observables:
                    save_payload[key] = markov_observables[key]
            save_payload["track4_markov_status"] = np.asarray(
                [str(markov_observables.get("status", "UNKNOWN"))],
                dtype=object,
            )

        np.savez_compressed(os.path.join(output_dir, "cyclic_paths.npz"), **save_payload)
        if markov_observables:
            summary = dict(markov_observables.get("summary") or {})
            summary.setdefault("status", str(markov_observables.get("status", "UNKNOWN")))
            with open(os.path.join(output_dir, "track4_markov_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)

    def run_stress_triggered_cyclic_walk(
        self,
        article_ids: Optional[List[str]] = None,
        max_steps: int = 150,
        gamma: float = 5.0,
        k_neighbors: int = 10,
        start_seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        proposal_mode: str = "metric_softmax",
        adaptive_tpt_connectivity: bool = False,
        feature_basis: str = "track2",
    ) -> Dict[str, Any]:
        proposal_mode = _normalize_proposal_mode(proposal_mode)
        n_articles = int(self.embeddings.shape[0])
        if n_articles == 0:
            return {
                "catalyst_indices": [],
                "swarm_records": [],
                "swarm_anchor_idx": [],
                "swarm_is_hot": [],
                "anchor_summaries": [],
                "proposal_mode": proposal_mode,
                "effective_k_neighbors": 0,
            }

        rng = np.random.default_rng(int(start_seed) if start_seed is not None else 0)
        effective_k_neighbors = min(max(int(k_neighbors), 1), max(n_articles - 1, 1))

        def _build_graph_and_markov(k_value: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, Dict[str, Any]]:
            local_neighbors, local_metric_distance_matrix, local_euclidean_distance_matrix, local_shear_vectors = _build_metric_graph(
                embeddings=self.embeddings,
                rho=self.track3_density,
                scalar_stress=self.metric_stress,
                k_neighbors=k_value,
            )
            local_s_max = _metric_horizon(local_metric_distance_matrix, local_neighbors, default=1.0)
            if local_s_max <= 0.0:
                local_s_max = 1.0
            local_markov = self._compute_markov_observables(
                neighbors=local_neighbors,
                metric_distance_matrix=local_metric_distance_matrix,
                terrain_labels=terrain_labels,
            )
            local_markov.setdefault("summary", {})
            local_markov["summary"]["requested_k_neighbors"] = int(k_neighbors)
            local_markov["summary"]["effective_k_neighbors"] = int(k_value)
            local_markov["summary"]["adaptive_tpt_connectivity"] = bool(adaptive_tpt_connectivity)
            return (
                local_neighbors,
                local_metric_distance_matrix,
                local_euclidean_distance_matrix,
                local_shear_vectors,
                float(local_s_max),
                local_markov,
            )

        catalyst_indices = self.select_catalysts()[:3]
        selection = self._last_catalyst_selection or {}
        terrain_labels = list(selection.get("terrain_labels") or self._terrain_fields()["labels"])
        (
            neighbors,
            metric_distance_matrix,
            euclidean_distance_matrix,
            shear_vectors,
            s_max,
            markov_observables,
        ) = _build_graph_and_markov(effective_k_neighbors)
        if adaptive_tpt_connectivity and not self._markov_supports_bridge_void_flux(markov_observables):
            initial_summary = dict(markov_observables.get("summary") or {})
            for candidate_k in self._connectivity_repair_candidates(effective_k_neighbors, n_articles):
                (
                    candidate_neighbors,
                    candidate_metric_distance_matrix,
                    candidate_euclidean_distance_matrix,
                    candidate_shear_vectors,
                    candidate_s_max,
                    candidate_markov,
                ) = _build_graph_and_markov(candidate_k)
                if self._markov_supports_bridge_void_flux(candidate_markov):
                    neighbors = candidate_neighbors
                    metric_distance_matrix = candidate_metric_distance_matrix
                    euclidean_distance_matrix = candidate_euclidean_distance_matrix
                    shear_vectors = candidate_shear_vectors
                    s_max = candidate_s_max
                    markov_observables = candidate_markov
                    effective_k_neighbors = int(candidate_k)
                    markov_observables["summary"]["connectivity_repair_applied"] = True
                    markov_observables["summary"]["initial_markov_summary"] = initial_summary
                    break
            else:
                markov_observables.setdefault("summary", {})
                markov_observables["summary"]["connectivity_repair_applied"] = False
                markov_observables["summary"]["connectivity_repair_failed"] = True
                markov_observables["summary"]["initial_markov_summary"] = initial_summary
        else:
            markov_observables.setdefault("summary", {})
            markov_observables["summary"]["connectivity_repair_applied"] = False
            markov_observables["summary"]["connectivity_repair_failed"] = False
        markov_observables["summary"]["requested_k_neighbors"] = int(k_neighbors)
        markov_observables["summary"]["effective_k_neighbors"] = int(effective_k_neighbors)
        markov_observables["summary"]["adaptive_tpt_connectivity"] = bool(adaptive_tpt_connectivity)
        swarm_records: List[Dict[str, Any]] = []
        swarm_anchor_idx: List[int] = []
        swarm_is_hot: List[bool] = []
        anchor_summaries: List[str] = []

        for anchor_idx in catalyst_indices:
            anchor_runs = self._simulate_anchor(
                anchor_idx=int(anchor_idx),
                neighbors=neighbors,
                metric_distance_matrix=metric_distance_matrix,
                euclidean_distance_matrix=euclidean_distance_matrix,
                shear_vectors=shear_vectors,
                n_walkers=5,
                max_steps=max_steps,
                gamma=gamma,
                hot_temperature_multiplier=5.0,
                rng=rng,
                s_max=s_max,
                proposal_mode=proposal_mode,
                committor_to_void=markov_observables.get("committor_to_void"),
            )["all_runs"]

            hot_survived = 0
            cold_survived = 0
            for run in anchor_runs:
                swarm_records.append(
                    {
                        "path_xyz": self.embeddings[run["path_indices"]].detach().cpu().numpy().astype(np.float32),
                        "path_indices": np.asarray(run["path_indices"], dtype=np.int32),
                        "work_integral": float(run["work_integral"]),
                        "closed_loop": bool(run["closed_loop"]),
                        "proposal_mode": str(run.get("proposal_mode", proposal_mode)),
                        "feature_basis": str(feature_basis),
                    }
                )
                swarm_anchor_idx.append(int(anchor_idx))
                swarm_is_hot.append(bool(run["is_hot"]))
                if bool(run["is_hot"]) and bool(run["closed_loop"]):
                    hot_survived += 1
                if (not bool(run["is_hot"])) and bool(run["closed_loop"]):
                    cold_survived += 1

            anchor_summaries.append(
                f"{hot_survived}/2 Hot Walkers Tunneled | {cold_survived}/3 Cold Walkers Closed Loop"
            )

        self._export_cyclic_paths(
            swarm_records,
            swarm_anchor_idx,
            swarm_is_hot,
            output_dir,
            markov_observables=markov_observables,
            feature_basis=feature_basis,
        )
        return {
            "catalyst_indices": catalyst_indices,
            "catalyst_zones": [terrain_labels[int(idx)] for idx in catalyst_indices if 0 <= int(idx) < len(terrain_labels)],
            "catalyst_selection": selection,
            "swarm_records": swarm_records,
            "swarm_anchor_idx": swarm_anchor_idx,
            "swarm_is_hot": swarm_is_hot,
            "anchor_summaries": anchor_summaries,
            "cognitive_horizon": float(s_max),
            "proposal_mode": proposal_mode,
            "feature_basis": str(feature_basis),
            "requested_k_neighbors": int(k_neighbors),
            "effective_k_neighbors": int(effective_k_neighbors),
            "adaptive_tpt_connectivity": bool(adaptive_tpt_connectivity),
            "markov_observables": {
                "status": markov_observables.get("status"),
                "summary": markov_observables.get("summary", {}),
            },
        }

    def compute_corpus_walk(
        self,
        article_ids: Optional[List[str]] = None,
        n_walkers: int = 10,
        max_steps: int = 150,
        gamma: float = 5.0,
        k_neighbors: int = 10,
        hot_temperature_multiplier: float = 5.0,
        start_seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        proposal_mode: str = "metric_softmax",
        adaptive_tpt_connectivity: bool = False,
        feature_basis: str = "track2",
    ) -> Dict[str, Any]:
        proposal_mode = _normalize_proposal_mode(proposal_mode)
        n_articles = int(self.embeddings.shape[0])
        if n_articles == 0:
            empty = torch.empty((0, 0), dtype=torch.float32)
            return {
                "walker_output": empty,
                "work_integrals": [],
                "states": [],
                "state_records": [],
                "path_records": [],
                "step_diagnostics": [],
            }

        rng = np.random.default_rng(int(start_seed) if start_seed is not None else 0)
        neighbors, metric_distance_matrix, euclidean_distance_matrix, shear_vectors = _build_metric_graph(
            embeddings=self.embeddings,
            rho=self.track3_density,
            scalar_stress=self.metric_stress,
            k_neighbors=k_neighbors,
        )
        s_max = _metric_horizon(metric_distance_matrix, neighbors, default=1.0)
        if s_max <= 0.0:
            s_max = 1.0

        records = [
            self._simulate_anchor(
                anchor_idx=article_idx,
                neighbors=neighbors,
                metric_distance_matrix=metric_distance_matrix,
                euclidean_distance_matrix=euclidean_distance_matrix,
                shear_vectors=shear_vectors,
                n_walkers=n_walkers,
                max_steps=max_steps,
                gamma=gamma,
                hot_temperature_multiplier=hot_temperature_multiplier,
                rng=rng,
                s_max=s_max,
                proposal_mode=proposal_mode,
            )
            for article_idx in range(n_articles)
        ]

        walker_output = torch.stack(
            [record["walker_output"].to(device=self.embeddings.device, dtype=torch.float32) for record in records],
            dim=0,
        )
        work_integrals = [float(record["work_integral"]) for record in records]
        closed_loop_flags = [bool(record["closed_loop"]) for record in records]
        states = ["closed_loop" if flag else "open_loop" for flag in closed_loop_flags]
        state_records = []
        path_records = []
        step_diagnostics = []

        for article_idx, record in enumerate(records):
            state_records.append(
                {
                    "label": "closed_loop" if bool(record["closed_loop"]) else "open_loop",
                    "raw_state": "closed_loop" if bool(record["closed_loop"]) else "open_loop",
                    "closed_loop": bool(record["closed_loop"]),
                    "work_integral": float(record["work_integral"]),
                    "proposal_mode": str(record.get("proposal_mode", proposal_mode)),
                    "feature_basis": str(feature_basis),
                    "steps": int(max_steps),
                }
            )
            path_records.append(
                {
                    "article_idx": int(article_idx),
                    "bt_uid": article_ids[article_idx] if article_ids and article_idx < len(article_ids) else f"article_{article_idx}",
                    "path_xyz": record["path_xyz"],
                    "work_integral": float(record["work_integral"]),
                    "closed_loop": bool(record["closed_loop"]),
                    "proposal_mode": str(record.get("proposal_mode", proposal_mode)),
                    "feature_basis": str(feature_basis),
                    "step_diagnostics": record["step_diagnostics"],
                }
            )
            step_diagnostics.append(
                {
                    "article_idx": int(article_idx),
                    "bt_uid": article_ids[article_idx] if article_ids and article_idx < len(article_ids) else f"article_{article_idx}",
                    "steps": record["step_diagnostics"],
                }
            )

        anchor_swarm = self.run_stress_triggered_cyclic_walk(
            article_ids=article_ids,
            max_steps=max_steps,
            gamma=gamma,
            k_neighbors=k_neighbors,
            start_seed=start_seed,
            output_dir=output_dir,
            proposal_mode=proposal_mode,
            adaptive_tpt_connectivity=adaptive_tpt_connectivity,
            feature_basis=feature_basis,
        )
        return {
            "walker_output": walker_output,
            "work_integrals": work_integrals,
            "states": states,
            "state_records": state_records,
            "path_records": path_records,
            "step_diagnostics": step_diagnostics,
            "catalyst_indices": anchor_swarm.get("catalyst_indices", []),
            "catalyst_zones": anchor_swarm.get("catalyst_zones", []),
            "catalyst_selection": anchor_swarm.get("catalyst_selection", {}),
            "anchor_summaries": anchor_swarm.get("anchor_summaries", []),
            "cognitive_horizon": float(s_max),
            "proposal_mode": proposal_mode,
            "feature_basis": str(feature_basis),
            "adaptive_tpt_connectivity": bool(adaptive_tpt_connectivity),
            "requested_k_neighbors": anchor_swarm.get("requested_k_neighbors", int(k_neighbors)),
            "effective_k_neighbors": anchor_swarm.get("effective_k_neighbors", int(k_neighbors)),
            "markov_observables": anchor_swarm.get("markov_observables", {}),
        }


def compute_corpus_walker_resistance(
    embeddings: torch.Tensor,
    rks_basis,
    track3_density: Optional[torch.Tensor] = None,
    z_coordinates: Optional[torch.Tensor] = None,
    metric_stress: Optional[torch.Tensor] = None,
    article_coords_2d: Optional[torch.Tensor] = None,
    article_ids: Optional[List[str]] = None,
    temperature: float = 0.5,
    n_walkers: int = 10,
    max_steps: int = 150,
    gamma: float = 5.0,
    k_neighbors: int = 10,
    hot_temperature_multiplier: float = 5.0,
    thermo_config: Optional[ThermodynamicConfig] = None,
    start_seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    proposal_mode: str = "metric_softmax",
    adaptive_tpt_connectivity: bool = False,
    feature_basis: str = "track2",
) -> Dict[str, Any]:
    walker = SemanticWalker(
        embeddings=embeddings,
        rks_basis=rks_basis,
        temperature=temperature,
        track3_density=track3_density,
        z_coordinates=z_coordinates,
        metric_stress=metric_stress,
        article_coords_2d=article_coords_2d,
        thermo_config=thermo_config,
    )
    return walker.compute_corpus_walk(
        article_ids=article_ids,
        n_walkers=n_walkers,
        max_steps=max_steps,
        gamma=gamma,
        k_neighbors=k_neighbors,
        hot_temperature_multiplier=hot_temperature_multiplier,
        start_seed=start_seed,
        output_dir=output_dir,
        proposal_mode=proposal_mode,
        adaptive_tpt_connectivity=adaptive_tpt_connectivity,
        feature_basis=feature_basis,
    )


def run_cyclic_physarum_ablation(
    embeddings: torch.Tensor,
    rks_basis,
    track3_density: Optional[torch.Tensor],
    z_coordinates: Optional[torch.Tensor],
    metric_stress: Optional[torch.Tensor],
    article_coords_2d: Optional[torch.Tensor],
    output_dir: str,
    n_walkers: int = 10,
    max_steps: int = 1000,
) -> None:
    compute_corpus_walker_resistance(
        embeddings=embeddings,
        rks_basis=rks_basis,
        track3_density=track3_density,
        z_coordinates=z_coordinates,
        metric_stress=metric_stress,
        article_coords_2d=article_coords_2d,
        n_walkers=n_walkers,
        max_steps=max_steps,
        output_dir=output_dir,
    )
