"""Track 4 least-action traversal over the reified metric graph.

This module is intentionally separate from the legacy cyclic walker.  The
cyclic walker asks how stochastic walkers behave on the article graph.  The
action graph asks a more mechanical question: what is the minimum work required
to move between two semantic landmarks when Track 2 geometry, Track 1.5 shear,
and Track 3 density jointly define edge costs?
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .physarum_walk import (
    _build_metric_graph,
    _reduce_density,
    _to_float_tensor,
    _unit_interval_tensor,
    _zone_from_density_stress,
)


@dataclass(frozen=True)
class ActionGraphConfig:
    """Configuration for least-action Track 4 traversal."""

    k_neighbors: int = 8
    action_mode: str = "baseline"
    stress_weight: float = 1.5
    shear_weight: float = 1.0
    observer_transport_weight: float = 1.0
    hysteresis_weight: float = 0.5
    void_weight: float = 1.0
    curvature_weight: float = 0.75
    max_paths: int = 12
    target_count_per_anchor: int = 1
    use_virtual_transitions: bool = False
    virtual_interpolation_steps: int = 0
    work_bucket_count: int = 6
    richer_state_memory_weight: float = 0.05


@dataclass(frozen=True)
class ActionGraphFields:
    """Metric graph fields used by the least-action solver."""

    embeddings: torch.Tensor
    rho: torch.Tensor
    stress: torch.Tensor
    shear_features: torch.Tensor
    shear_distance: torch.Tensor
    observer_simplex: torch.Tensor
    null_observer_simplex: Optional[torch.Tensor]
    observer_transport_distance: torch.Tensor
    observer_kl_forward: torch.Tensor
    null_observer_kl_forward: Optional[torch.Tensor]
    stress_unit: torch.Tensor
    density_unit: torch.Tensor
    terrain_labels: List[str]
    neighbors: torch.Tensor
    metric_distance: torch.Tensor
    euclidean_distance: torch.Tensor
    shear_vectors: torch.Tensor
    original_node_count: int
    virtual_node_metadata: Dict[int, Dict[str, Any]]
    neighbor_support: Optional[List[List[int]]] = None


def _as_tensor(value: Any, *, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D array/tensor, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one article")
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_observer_simplex(value: Any | None, reference: torch.Tensor) -> torch.Tensor:
    """Return an [N, B] observer-mixture simplex.

    If no observer data is available, every article receives the same one-point
    simplex so observer transport contributes zero cost.  Existing artifacts can
    pass per-bot magnitudes, annealing weights, or any nonnegative/probability
    matrix with one row per article.
    """

    n_items = int(reference.shape[0])
    if value is None:
        return torch.ones((n_items, 1), dtype=torch.float32, device=reference.device)
    simplex = _to_float_tensor(value, reference)
    if simplex is None:
        return torch.ones((n_items, 1), dtype=torch.float32, device=reference.device)
    if simplex.ndim == 3:
        simplex = simplex.norm(dim=-1)
    if simplex.ndim == 1:
        simplex = simplex.reshape(n_items, 1) if simplex.shape[0] == n_items else simplex.reshape(1, -1).repeat(n_items, 1)
    if simplex.ndim != 2 or simplex.shape[0] != n_items:
        raise ValueError(f"observer_simplex must have shape [N, B], got {tuple(simplex.shape)} for N={n_items}")
    simplex = torch.nan_to_num(simplex, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(simplex < 0.0):
        simplex = torch.softmax(simplex, dim=-1)
    else:
        row_sums = simplex.sum(dim=-1, keepdim=True)
        zero_rows = row_sums <= 1e-9
        simplex = simplex / row_sums.clamp(min=1e-9)
        if bool(zero_rows.any().item()):
            simplex[zero_rows.reshape(-1)] = 1.0 / float(simplex.shape[1])
    return simplex.clamp(min=1e-8)


def _directed_kl_matrix(simplex: torch.Tensor) -> torch.Tensor:
    """Compute KL(next || current) for every directed edge pair."""

    eps = 1e-8
    p_current = simplex.clamp(min=eps)
    p_next = simplex.clamp(min=eps)
    log_next = torch.log(p_next)
    log_current = torch.log(p_current)
    kl = (p_next.unsqueeze(0) * (log_next.unsqueeze(0) - log_current.unsqueeze(1))).sum(dim=-1)
    kl.fill_diagonal_(0.0)
    return torch.nan_to_num(kl.clamp(min=0.0), nan=0.0, posinf=0.0, neginf=0.0)


def build_action_graph_fields(
    embeddings: Any,
    *,
    track3_density: Any | None = None,
    metric_stress: Any | None = None,
    observer_simplex: Any | None = None,
    null_observer_simplex: Any | None = None,
    k_neighbors: int = 8,
) -> ActionGraphFields:
    """Build Track 4 metric fields from already-computed track artifacts."""

    embedding_tensor = _as_tensor(embeddings, name="embeddings")
    rho = _reduce_density(_to_float_tensor(track3_density, embedding_tensor), embedding_tensor)

    raw_stress = _to_float_tensor(metric_stress, embedding_tensor)
    if raw_stress is None:
        stress = torch.zeros(embedding_tensor.shape[0], dtype=torch.float32, device=embedding_tensor.device)
        shear_features = torch.zeros((embedding_tensor.shape[0], 1), dtype=torch.float32, device=embedding_tensor.device)
    elif raw_stress.ndim > 1:
        if raw_stress.shape[0] != embedding_tensor.shape[0]:
            raise ValueError(
                f"metric_stress length must match embeddings; got {raw_stress.shape[0]} vs {embedding_tensor.shape[0]}"
            )
        shear_features = torch.nn.functional.normalize(raw_stress, p=2, dim=-1)
        stress = raw_stress.norm(dim=-1)
    else:
        stress = raw_stress.reshape(-1)
        shear_features = torch.zeros((embedding_tensor.shape[0], 1), dtype=torch.float32, device=embedding_tensor.device)
    if stress.shape[0] != embedding_tensor.shape[0]:
        raise ValueError(
            f"metric_stress length must match embeddings; got {stress.shape[0]} vs {embedding_tensor.shape[0]}"
        )
    stress = torch.nan_to_num(stress.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    shear_features = torch.nan_to_num(shear_features, nan=0.0, posinf=0.0, neginf=0.0)
    shear_distance = torch.cdist(shear_features, shear_features, p=2)
    shear_distance.fill_diagonal_(0.0)
    simplex = _normalize_observer_simplex(observer_simplex, embedding_tensor)
    null_simplex = None
    null_observer_kl_forward = None
    if null_observer_simplex is not None:
        null_simplex = _normalize_observer_simplex(null_observer_simplex, embedding_tensor)
    observer_transport_distance = 0.5 * torch.cdist(simplex, simplex, p=1)
    observer_kl_forward = _directed_kl_matrix(simplex)
    if null_simplex is not None:
        null_observer_kl_forward = _directed_kl_matrix(null_simplex)
    stress_unit = _unit_interval_tensor(torch.abs(stress))
    density_unit = _unit_interval_tensor(rho)
    terrain_labels = [
        _zone_from_density_stress(float(density_unit[idx].item()), float(stress_unit[idx].item()))
        for idx in range(int(embedding_tensor.shape[0]))
    ]
    neighbors, metric_distance, euclidean_distance, shear_vectors = _build_metric_graph(
        embeddings=embedding_tensor,
        rho=rho,
        scalar_stress=stress,
        k_neighbors=k_neighbors,
    )
    return ActionGraphFields(
        embeddings=embedding_tensor,
        rho=rho,
        stress=stress,
        shear_features=shear_features,
        shear_distance=shear_distance,
        observer_simplex=simplex,
        observer_transport_distance=observer_transport_distance,
        observer_kl_forward=observer_kl_forward,
        stress_unit=stress_unit,
        density_unit=density_unit,
        terrain_labels=terrain_labels,
        neighbors=neighbors,
        metric_distance=metric_distance,
        euclidean_distance=euclidean_distance,
        shear_vectors=shear_vectors,
        null_observer_simplex=null_simplex,
        null_observer_kl_forward=null_observer_kl_forward,
        original_node_count=int(embedding_tensor.shape[0]),
        virtual_node_metadata={},
    )


def _normalized_action_mode(config: ActionGraphConfig | None) -> str:
    raw = str((config or ActionGraphConfig()).action_mode or "baseline").strip().lower()
    aliases = {
        "baseline_raw_action": "baseline",
        "track2_default": "baseline",
        "null_calibrated": "null_calibrated_hysteresis",
        "calibrated": "null_calibrated_hysteresis",
        "richer_walker_state": "richer_state",
        "stateful": "richer_state",
        "virtual_transition": "virtual_transition_states",
    }
    return aliases.get(raw, raw)


def _uses_null_calibrated_hysteresis(config: ActionGraphConfig | None) -> bool:
    return _normalized_action_mode(config) == "null_calibrated_hysteresis"


def _uses_richer_state(config: ActionGraphConfig | None) -> bool:
    return _normalized_action_mode(config) == "richer_state"


def _uses_virtual_transitions(config: ActionGraphConfig | None) -> bool:
    cfg = config or ActionGraphConfig()
    mode = _normalized_action_mode(cfg)
    return bool(cfg.use_virtual_transitions) or mode == "virtual_transition_states"


def _renormalize_simplex_rows(simplex: torch.Tensor) -> torch.Tensor:
    simplex = torch.nan_to_num(simplex, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    row_sums = simplex.sum(dim=-1, keepdim=True)
    simplex = simplex / row_sums.clamp(min=1e-9)
    zero_rows = row_sums <= 1e-9
    if bool(zero_rows.any().item()):
        simplex[zero_rows.reshape(-1)] = 1.0 / float(simplex.shape[1])
    return simplex.clamp(min=1e-8)


def _append_virtual_transition_nodes(fields: ActionGraphFields, config: ActionGraphConfig) -> ActionGraphFields:
    """Densify existing graph edges with interpolated virtual transition states."""

    steps = max(int(config.virtual_interpolation_steps), 0)
    if steps <= 0 or not _uses_virtual_transitions(config):
        return fields

    n_original = int(fields.embeddings.shape[0])
    edge_pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for source in range(n_original):
        for target in _neighbor_indices(fields, source):
            if target < 0 or target >= n_original or target == source:
                continue
            pair = (min(source, target), max(source, target))
            if pair not in seen:
                seen.add(pair)
                edge_pairs.append(pair)
    if not edge_pairs:
        return fields

    embeddings = [fields.embeddings]
    rho_values = [fields.rho]
    stress_values = [fields.stress]
    shear_features = [fields.shear_features]
    observer_values = [fields.observer_simplex]
    null_observer_values: List[torch.Tensor] = []
    if fields.null_observer_simplex is not None:
        null_observer_values.append(fields.null_observer_simplex)

    support: List[set[int]] = [set() for _ in range(n_original)]
    metadata: Dict[int, Dict[str, Any]] = {}
    next_idx = n_original

    for source, target in edge_pairs:
        chain = [source]
        for step in range(1, steps + 1):
            lam = float(step) / float(steps + 1)
            embeddings.append((1.0 - lam) * fields.embeddings[source : source + 1] + lam * fields.embeddings[target : target + 1])
            rho_values.append((1.0 - lam) * fields.rho[source : source + 1] + lam * fields.rho[target : target + 1])
            stress_values.append(
                (1.0 - lam) * fields.stress[source : source + 1] + lam * fields.stress[target : target + 1]
            )
            shear_features.append(
                (1.0 - lam) * fields.shear_features[source : source + 1]
                + lam * fields.shear_features[target : target + 1]
            )
            observer_values.append(
                _renormalize_simplex_rows(
                    (1.0 - lam) * fields.observer_simplex[source : source + 1]
                    + lam * fields.observer_simplex[target : target + 1]
                )
            )
            if fields.null_observer_simplex is not None:
                null_observer_values.append(
                    _renormalize_simplex_rows(
                        (1.0 - lam) * fields.null_observer_simplex[source : source + 1]
                        + lam * fields.null_observer_simplex[target : target + 1]
                    )
                )
            metadata[next_idx] = {
                "kind": "virtual_transition",
                "source_idx": int(source),
                "target_idx": int(target),
                "lambda": float(lam),
                "edge_key": f"{source}->{target}",
            }
            chain.append(next_idx)
            support.append(set())
            next_idx += 1
        chain.append(target)
        for left, right in zip(chain[:-1], chain[1:]):
            support[left].add(right)
            support[right].add(left)

    expanded_embeddings = torch.cat(embeddings, dim=0)
    expanded_rho = torch.cat(rho_values, dim=0)
    expanded_stress = torch.cat(stress_values, dim=0)
    expanded_shear_features = torch.cat(shear_features, dim=0)
    expanded_simplex = _renormalize_simplex_rows(torch.cat(observer_values, dim=0))
    expanded_null_simplex = None
    expanded_null_kl = None
    if null_observer_values:
        expanded_null_simplex = _renormalize_simplex_rows(torch.cat(null_observer_values, dim=0))
        expanded_null_kl = _directed_kl_matrix(expanded_null_simplex)

    stress_unit = _unit_interval_tensor(torch.abs(expanded_stress))
    density_unit = _unit_interval_tensor(expanded_rho)
    terrain_labels = [
        _zone_from_density_stress(float(density_unit[idx].item()), float(stress_unit[idx].item()))
        for idx in range(int(expanded_embeddings.shape[0]))
    ]
    neighbors, metric_distance, euclidean_distance, shear_vectors = _build_metric_graph(
        embeddings=expanded_embeddings,
        rho=expanded_rho,
        scalar_stress=expanded_stress,
        k_neighbors=config.k_neighbors,
    )
    return ActionGraphFields(
        embeddings=expanded_embeddings,
        rho=expanded_rho,
        stress=expanded_stress,
        shear_features=torch.nan_to_num(expanded_shear_features, nan=0.0, posinf=0.0, neginf=0.0),
        shear_distance=torch.cdist(expanded_shear_features, expanded_shear_features, p=2),
        observer_simplex=expanded_simplex,
        null_observer_simplex=expanded_null_simplex,
        observer_transport_distance=0.5 * torch.cdist(expanded_simplex, expanded_simplex, p=1),
        observer_kl_forward=_directed_kl_matrix(expanded_simplex),
        null_observer_kl_forward=expanded_null_kl,
        stress_unit=stress_unit,
        density_unit=density_unit,
        terrain_labels=terrain_labels,
        neighbors=neighbors,
        metric_distance=metric_distance,
        euclidean_distance=euclidean_distance,
        shear_vectors=shear_vectors,
        original_node_count=n_original,
        virtual_node_metadata=metadata,
        neighbor_support=[sorted(items) for items in support],
    )


def _edge_support_costs(fields: ActionGraphFields) -> List[float]:
    costs: List[float] = []
    for source in range(int(fields.embeddings.shape[0])):
        for target in _neighbor_indices(fields, source):
            if source < target:
                value = float(fields.metric_distance[source, target].item())
                if math.isfinite(value) and value > 0.0:
                    costs.append(value)
    return costs


def _work_bucket_width(fields: ActionGraphFields) -> float:
    costs = _edge_support_costs(fields)
    if not costs:
        return 1.0
    return max(float(np.median(np.asarray(costs, dtype=np.float64))) * 3.0, 1e-6)


def _work_bucket(accumulated_work: float, fields: ActionGraphFields, config: ActionGraphConfig) -> int:
    count = max(int(config.work_bucket_count), 1)
    if count <= 1:
        return 0
    return min(count - 1, max(0, int(float(accumulated_work) / _work_bucket_width(fields))))


def edge_action_components(
    fields: ActionGraphFields,
    source: int,
    target: int,
    previous: int | None = None,
    *,
    config: ActionGraphConfig | None = None,
    accumulated_work_bucket: int = 0,
) -> Dict[str, float]:
    """Compute the local action terms for one directed edge."""

    cfg = config or ActionGraphConfig()
    i = int(source)
    j = int(target)
    metric = float(fields.metric_distance[i, j].item())
    euclidean = float(fields.euclidean_distance[i, j].item())
    rho_mid = 0.5 * (float(fields.rho[i].item()) + float(fields.rho[j].item()))
    stress_mid = 0.5 * (float(fields.stress_unit[i].item()) + float(fields.stress_unit[j].item()))
    shear_distance = float(fields.shear_distance[i, j].item())
    local_scale = max(metric, euclidean, 1e-6)
    shear_penalty = float(cfg.shear_weight) * shear_distance * local_scale
    observer_transport_penalty = (
        float(cfg.observer_transport_weight) * float(fields.observer_transport_distance[i, j].item()) * local_scale
    )
    hysteresis_penalty = float(cfg.hysteresis_weight) * float(fields.observer_kl_forward[i, j].item()) * local_scale
    null_hysteresis_penalty = 0.0
    if fields.null_observer_kl_forward is not None:
        null_hysteresis_penalty = (
            float(cfg.hysteresis_weight) * float(fields.null_observer_kl_forward[i, j].item()) * local_scale
        )
    excess_hysteresis_penalty = float(hysteresis_penalty - null_hysteresis_penalty)
    positive_excess_hysteresis_penalty = float(max(excess_hysteresis_penalty, 0.0))
    calibrated_hysteresis_penalty = (
        positive_excess_hysteresis_penalty if _uses_null_calibrated_hysteresis(cfg) else hysteresis_penalty
    )
    density_penalty = float(cfg.void_weight) * (1.0 - max(min(rho_mid, 1.0), 0.0)) * metric
    stress_penalty = float(cfg.stress_weight) * stress_mid * metric
    curvature_penalty = 0.0

    if previous is not None and int(previous) != i:
        prev = int(previous)
        v1 = fields.embeddings[i] - fields.embeddings[prev]
        v2 = fields.embeddings[j] - fields.embeddings[i]
        denom = torch.norm(v1, p=2) * torch.norm(v2, p=2)
        if float(denom.item()) > 1e-9:
            cos_theta = float(torch.dot(v1, v2).item() / denom.item())
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            turn = 1.0 - cos_theta
            curvature_penalty = float(cfg.curvature_weight) * turn * local_scale

    work_memory_penalty = 0.0
    if _uses_richer_state(cfg):
        bucket_fraction = 0.0
        bucket_count = max(int(cfg.work_bucket_count), 1)
        if bucket_count > 1:
            bucket_fraction = max(min(float(accumulated_work_bucket) / float(bucket_count - 1), 1.0), 0.0)
        stateful_terms = observer_transport_penalty + calibrated_hysteresis_penalty + curvature_penalty
        work_memory_penalty = float(cfg.richer_state_memory_weight) * bucket_fraction * stateful_terms

    total = (
        metric
        + shear_penalty
        + observer_transport_penalty
        + calibrated_hysteresis_penalty
        + density_penalty
        + stress_penalty
        + curvature_penalty
        + work_memory_penalty
    )
    return {
        "metric": float(metric),
        "euclidean": float(euclidean),
        "shear_penalty": float(shear_penalty),
        "observer_transport_penalty": float(observer_transport_penalty),
        "hysteresis_penalty": float(hysteresis_penalty),
        "null_hysteresis_penalty": float(null_hysteresis_penalty),
        "excess_hysteresis_penalty": float(excess_hysteresis_penalty),
        "positive_excess_hysteresis_penalty": float(positive_excess_hysteresis_penalty),
        "calibrated_hysteresis_penalty": float(calibrated_hysteresis_penalty),
        "density_penalty": float(density_penalty),
        "stress_penalty": float(stress_penalty),
        "curvature_penalty": float(curvature_penalty),
        "work_memory_penalty": float(work_memory_penalty),
        "action": float(total),
    }


def path_action_components(
    fields: ActionGraphFields,
    path_indices: Sequence[int],
    *,
    config: ActionGraphConfig | None = None,
) -> Dict[str, float]:
    """Sum action terms along a path."""

    totals = {
        "metric": 0.0,
        "euclidean": 0.0,
        "density_penalty": 0.0,
        "shear_penalty": 0.0,
        "observer_transport_penalty": 0.0,
        "hysteresis_penalty": 0.0,
        "null_hysteresis_penalty": 0.0,
        "excess_hysteresis_penalty": 0.0,
        "positive_excess_hysteresis_penalty": 0.0,
        "calibrated_hysteresis_penalty": 0.0,
        "stress_penalty": 0.0,
        "curvature_penalty": 0.0,
        "work_memory_penalty": 0.0,
        "action": 0.0,
    }
    path = [int(idx) for idx in path_indices]
    running_action = 0.0
    for offset in range(1, len(path)):
        previous = path[offset - 2] if offset >= 2 else None
        bucket = _work_bucket(running_action, fields, config or ActionGraphConfig()) if _uses_richer_state(config) else 0
        components = edge_action_components(
            fields,
            path[offset - 1],
            path[offset],
            previous,
            config=config,
            accumulated_work_bucket=bucket,
        )
        for key in totals:
            totals[key] += float(components[key])
        running_action += float(components["action"])
    totals["steps"] = float(max(len(path) - 1, 0))
    return totals


def _neighbor_indices(fields: ActionGraphFields, current: int) -> List[int]:
    """Return undirected kNN support for a node.

    The legacy walker uses directed top-k rows, but least-action traversal is a
    metric-graph question.  If either endpoint selected the other as a local
    neighbor, the edge is valid support for an action path.
    """

    if fields.neighbor_support is not None:
        if 0 <= int(current) < len(fields.neighbor_support):
            return [int(idx) for idx in fields.neighbor_support[int(current)] if int(idx) != int(current)]
        return []
    row = set(int(idx) for idx in fields.neighbors[int(current)].detach().cpu().numpy().astype(np.int64).tolist())
    incoming = torch.where(fields.neighbors == int(current))[0].detach().cpu().numpy().astype(np.int64).tolist()
    row.update(int(idx) for idx in incoming)
    row.discard(int(current))
    return sorted(row)


def least_action_path(
    fields: ActionGraphFields,
    source: int,
    target: int,
    *,
    config: ActionGraphConfig | None = None,
) -> Dict[str, Any]:
    """Find a minimum-action path using Dijkstra over (previous,current) states."""

    cfg = config or ActionGraphConfig()
    richer_state = _uses_richer_state(cfg)
    bucket_width = _work_bucket_width(fields) if richer_state else 1.0
    n_items = int(fields.embeddings.shape[0])
    start = int(source)
    goal = int(target)
    if not (0 <= start < n_items and 0 <= goal < n_items):
        raise ValueError(f"source/target out of range: source={source}, target={target}, n={n_items}")
    if start == goal:
        components = path_action_components(fields, [start], config=cfg)
        return {
            "source_idx": start,
            "target_idx": goal,
            "path_indices": [start],
            "reached": True,
            **components,
        }

    queue: List[Tuple[float, int, int, int]] = [(0.0, -1, start, 0)]
    best: Dict[Tuple[int, int, int], float] = {(-1, start, 0): 0.0}
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    final_state: Tuple[int, int, int] | None = None

    while queue:
        cost, previous, current, work_bucket = heapq.heappop(queue)
        state = (previous, current, work_bucket)
        if cost > best.get(state, math.inf) + 1e-12:
            continue
        if current == goal:
            final_state = state
            break
        for nxt in _neighbor_indices(fields, current):
            components = edge_action_components(
                fields,
                current,
                nxt,
                None if previous < 0 else previous,
                config=cfg,
                accumulated_work_bucket=work_bucket if richer_state else 0,
            )
            next_cost = cost + float(components["action"])
            if richer_state:
                next_bucket = min(
                    max(int(cfg.work_bucket_count), 1) - 1,
                    max(0, int(float(next_cost) / max(bucket_width, 1e-6))),
                )
            else:
                next_bucket = 0
            next_state = (current, nxt, next_bucket)
            if next_cost + 1e-12 < best.get(next_state, math.inf):
                best[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(queue, (next_cost, current, nxt, next_bucket))

    if final_state is None:
        return {
            "source_idx": start,
            "target_idx": goal,
            "path_indices": [start],
            "reached": False,
            "metric": math.inf,
            "euclidean": math.inf,
            "density_penalty": math.inf,
            "shear_penalty": math.inf,
            "observer_transport_penalty": math.inf,
            "hysteresis_penalty": math.inf,
            "null_hysteresis_penalty": math.inf,
            "excess_hysteresis_penalty": math.inf,
            "positive_excess_hysteresis_penalty": math.inf,
            "calibrated_hysteresis_penalty": math.inf,
            "stress_penalty": math.inf,
            "curvature_penalty": math.inf,
            "work_memory_penalty": math.inf,
            "action": math.inf,
            "steps": 0.0,
        }

    rev: List[int] = [final_state[1]]
    state = final_state
    while state in parent:
        state = parent[state]
        rev.append(state[1])
    path = list(reversed(rev))
    components = path_action_components(fields, path, config=cfg)
    return {
        "source_idx": start,
        "target_idx": goal,
        "path_indices": path,
        "reached": True,
        **components,
    }


def select_action_anchors(fields: ActionGraphFields, count: int = 3) -> List[int]:
    """Select high-leverage anchors with zone diversity and farthest-first spread."""

    n_items = int(fields.embeddings.shape[0])
    if n_items <= 0:
        return []
    count = min(max(int(count), 1), n_items)
    influence = fields.stress_unit + (1.0 - fields.density_unit)
    labels = fields.terrain_labels
    selected: List[int] = []

    for zone in ("Void", "Bridge"):
        candidates = [idx for idx, label in enumerate(labels) if label == zone]
        if candidates and len(selected) < count:
            selected.append(max(candidates, key=lambda idx: float(influence[idx].item())))

    while len(selected) < count:
        remaining = [idx for idx in range(n_items) if idx not in selected]
        if not remaining:
            break
        if not selected:
            selected.append(max(remaining, key=lambda idx: float(influence[idx].item())))
            continue
        selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=fields.embeddings.device)
        remaining_tensor = torch.as_tensor(remaining, dtype=torch.long, device=fields.embeddings.device)
        spread = torch.cdist(fields.embeddings[remaining_tensor], fields.embeddings[selected_tensor], p=2).min(dim=1).values
        spread_unit = _unit_interval_tensor(spread)
        scores = torch.as_tensor(
            [float(influence[idx].item()) for idx in remaining],
            dtype=torch.float32,
            device=fields.embeddings.device,
        )
        scores = 0.55 * _unit_interval_tensor(scores) + 0.45 * spread_unit
        selected.append(int(remaining[int(torch.argmax(scores).item())]))
    return selected


def select_action_targets(fields: ActionGraphFields, anchor_idx: int, count: int = 1) -> List[int]:
    """Pick opposite-zone targets for least-action traversal."""

    opposite = {
        "Bridge": ["Void", "Swamp", "Tightrope"],
        "Void": ["Bridge", "Tightrope", "Swamp"],
        "Swamp": ["Tightrope", "Bridge", "Void"],
        "Tightrope": ["Swamp", "Void", "Bridge"],
    }
    anchor = int(anchor_idx)
    labels = fields.terrain_labels
    target_zones = opposite.get(labels[anchor], ["Void", "Bridge", "Swamp", "Tightrope"])
    selected: List[int] = []
    for zone in target_zones:
        candidates = [idx for idx, label in enumerate(labels) if label == zone and idx != anchor and idx not in selected]
        if not candidates:
            continue
        candidate_tensor = torch.as_tensor(candidates, dtype=torch.long, device=fields.embeddings.device)
        distances = fields.metric_distance[anchor, candidate_tensor]
        selected.append(int(candidates[int(torch.argmax(distances).item())]))
        if len(selected) >= count:
            return selected
    if len(selected) < count:
        remaining = [idx for idx in range(len(labels)) if idx != anchor and idx not in selected]
        remaining.sort(key=lambda idx: float(fields.metric_distance[anchor, idx].item()), reverse=True)
        selected.extend(remaining[: max(0, int(count) - len(selected))])
    return selected[:count]


def run_action_graph(
    embeddings: Any,
    *,
    track3_density: Any | None = None,
    metric_stress: Any | None = None,
    observer_simplex: Any | None = None,
    null_observer_simplex: Any | None = None,
    action_branch: str | None = None,
    richer_walker_state: bool = False,
    virtual_transition_states: bool = False,
    virtual_interpolation_steps: int | None = None,
    config: ActionGraphConfig | None = None,
    anchors: Sequence[int] | None = None,
    targets_by_anchor: Mapping[int, Sequence[int]] | None = None,
) -> Dict[str, Any]:
    """Run least-action paths for selected anchors and targets."""

    cfg = config or ActionGraphConfig()
    branch_label = str(action_branch or cfg.action_mode or "baseline_raw_action")
    mode = _normalized_action_mode(ActionGraphConfig(action_mode=branch_label))
    if action_branch is None:
        mode = _normalized_action_mode(cfg)
        branch_label = {
            "baseline": "baseline_raw_action",
            "null_calibrated_hysteresis": "null_calibrated_hysteresis",
            "richer_state": "richer_walker_state",
            "virtual_transition_states": "virtual_transition_states",
        }.get(mode, str(cfg.action_mode or "baseline_raw_action"))
    if richer_walker_state:
        mode = "richer_state"
        branch_label = "richer_walker_state"
    use_virtual = bool(virtual_transition_states or cfg.use_virtual_transitions or mode == "virtual_transition_states")
    virtual_steps = (
        int(virtual_interpolation_steps)
        if virtual_interpolation_steps is not None
        else int(cfg.virtual_interpolation_steps)
    )
    if use_virtual and virtual_steps <= 0:
        virtual_steps = 1
    cfg = replace(
        cfg,
        action_mode=mode,
        use_virtual_transitions=use_virtual,
        virtual_interpolation_steps=virtual_steps,
    )
    fields = build_action_graph_fields(
        embeddings,
        track3_density=track3_density,
        metric_stress=metric_stress,
        observer_simplex=observer_simplex,
        null_observer_simplex=null_observer_simplex,
        k_neighbors=cfg.k_neighbors,
    )
    selection_fields = fields
    selected_anchors = [int(idx) for idx in anchors] if anchors is not None else select_action_anchors(selection_fields, count=3)
    selected_targets_by_anchor: Dict[int, List[int]] = {}
    for anchor in selected_anchors:
        selected_targets_by_anchor[int(anchor)] = (
            [int(idx) for idx in targets_by_anchor.get(int(anchor), [])]
            if targets_by_anchor is not None and int(anchor) in targets_by_anchor
            else select_action_targets(selection_fields, int(anchor), count=cfg.target_count_per_anchor)
        )
    if _uses_virtual_transitions(cfg):
        fields = _append_virtual_transition_nodes(fields, cfg)
    records: List[Dict[str, Any]] = []
    for anchor in selected_anchors:
        target_list = selected_targets_by_anchor.get(int(anchor), [])
        for target in target_list:
            if len(records) >= int(cfg.max_paths):
                break
            record = least_action_path(fields, int(anchor), int(target), config=cfg)
            touched = [fields.terrain_labels[idx] for idx in record["path_indices"]]
            virtual_metadata = [
                {"path_offset": int(offset), **fields.virtual_node_metadata[int(idx)]}
                for offset, idx in enumerate(record["path_indices"])
                if int(idx) in fields.virtual_node_metadata
            ]
            record.update(
                {
                    "action_branch": branch_label,
                    "action_mode": _normalized_action_mode(cfg),
                    "state_mode": "richer_walker_state" if _uses_richer_state(cfg) else "article_node",
                    "source_zone": fields.terrain_labels[int(anchor)],
                    "target_zone": fields.terrain_labels[int(target)],
                    "touched_zones": touched,
                    "unique_touched_zones": sorted(set(touched)),
                    "path_node_types": [
                        "virtual" if int(idx) in fields.virtual_node_metadata else "article"
                        for idx in record["path_indices"]
                    ],
                    "uses_virtual_nodes": bool(virtual_metadata),
                    "virtual_node_count_in_path": int(len(virtual_metadata)),
                    "virtual_transition_metadata": virtual_metadata,
                }
            )
            records.append(record)

    finite_actions = [float(row["action"]) for row in records if bool(row.get("reached")) and np.isfinite(row.get("action"))]
    reached_records = [row for row in records if bool(row.get("reached"))]

    def _mean_record_term(key: str) -> float | None:
        values = []
        for row in reached_records:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else None

    summary = {
        "schema_version": "1.0",
        "track4_engine": "least_action_metric_graph",
        "action_branch": branch_label,
        "action_mode": _normalized_action_mode(cfg),
        "n_articles": int(fields.original_node_count),
        "n_total_nodes": int(fields.embeddings.shape[0]),
        "virtual_node_count": int(len(fields.virtual_node_metadata)),
        "virtual_transitions_enabled": bool(len(fields.virtual_node_metadata) > 0),
        "path_count": int(len(records)),
        "reached_count": int(sum(1 for row in records if bool(row.get("reached")))),
        "mean_action": float(np.mean(finite_actions)) if finite_actions else None,
        "median_action": float(np.median(finite_actions)) if finite_actions else None,
        "max_action": float(np.max(finite_actions)) if finite_actions else None,
        "mean_hysteresis_penalty": _mean_record_term("hysteresis_penalty"),
        "mean_null_hysteresis_penalty": _mean_record_term("null_hysteresis_penalty"),
        "mean_excess_hysteresis_penalty": _mean_record_term("excess_hysteresis_penalty"),
        "mean_positive_excess_hysteresis_penalty": _mean_record_term("positive_excess_hysteresis_penalty"),
        "mean_calibrated_hysteresis_penalty": _mean_record_term("calibrated_hysteresis_penalty"),
        "terrain_zone_counts": {
            zone: int(fields.terrain_labels.count(zone))
            for zone in sorted(set(fields.terrain_labels))
        },
        "anchors": selected_anchors,
        "anchor_zones": [fields.terrain_labels[idx] for idx in selected_anchors],
        "config": {
            "k_neighbors": int(cfg.k_neighbors),
            "action_mode": _normalized_action_mode(cfg),
            "stress_weight": float(cfg.stress_weight),
            "shear_weight": float(cfg.shear_weight),
            "observer_transport_weight": float(cfg.observer_transport_weight),
            "hysteresis_weight": float(cfg.hysteresis_weight),
            "void_weight": float(cfg.void_weight),
            "curvature_weight": float(cfg.curvature_weight),
            "max_paths": int(cfg.max_paths),
            "target_count_per_anchor": int(cfg.target_count_per_anchor),
            "use_virtual_transitions": bool(cfg.use_virtual_transitions),
            "virtual_interpolation_steps": int(cfg.virtual_interpolation_steps),
            "work_bucket_count": int(cfg.work_bucket_count),
            "richer_state_memory_weight": float(cfg.richer_state_memory_weight),
        },
    }
    return {
        "fields": fields,
        "records": records,
        "summary": summary,
    }


def _object_array(rows: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(rows), dtype=object)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def export_action_graph(result: Mapping[str, Any], output_dir: str | Path) -> Dict[str, str]:
    """Export least-action paths as NPZ plus a compact JSON summary."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields: ActionGraphFields = result["fields"]
    records: List[Dict[str, Any]] = list(result.get("records") or [])
    max_dim = int(fields.embeddings.shape[1])
    trajectories = [fields.embeddings[row["path_indices"]].detach().cpu().numpy().astype(np.float32) for row in records]
    npz_path = out / "least_action_paths.npz"
    np.savez_compressed(
        npz_path,
        trajectory_coordinates=_object_array(trajectories),
        path_indices=_object_array([row["path_indices"] for row in records]),
        action_integral=np.asarray([float(row["action"]) for row in records], dtype=np.float32),
        metric_integral=np.asarray([float(row["metric"]) for row in records], dtype=np.float32),
        density_penalty_integral=np.asarray([float(row["density_penalty"]) for row in records], dtype=np.float32),
        shear_penalty_integral=np.asarray([float(row["shear_penalty"]) for row in records], dtype=np.float32),
        observer_transport_penalty_integral=np.asarray(
            [float(row["observer_transport_penalty"]) for row in records],
            dtype=np.float32,
        ),
        hysteresis_penalty_integral=np.asarray([float(row["hysteresis_penalty"]) for row in records], dtype=np.float32),
        null_hysteresis_penalty_integral=np.asarray(
            [float(row["null_hysteresis_penalty"]) for row in records],
            dtype=np.float32,
        ),
        excess_hysteresis_penalty_integral=np.asarray(
            [float(row["excess_hysteresis_penalty"]) for row in records],
            dtype=np.float32,
        ),
        positive_excess_hysteresis_penalty_integral=np.asarray(
            [float(row["positive_excess_hysteresis_penalty"]) for row in records],
            dtype=np.float32,
        ),
        calibrated_hysteresis_penalty_integral=np.asarray(
            [float(row["calibrated_hysteresis_penalty"]) for row in records],
            dtype=np.float32,
        ),
        stress_penalty_integral=np.asarray([float(row["stress_penalty"]) for row in records], dtype=np.float32),
        curvature_penalty_integral=np.asarray([float(row["curvature_penalty"]) for row in records], dtype=np.float32),
        work_memory_penalty_integral=np.asarray([float(row["work_memory_penalty"]) for row in records], dtype=np.float32),
        reached=np.asarray([bool(row["reached"]) for row in records], dtype=np.bool_),
        source_idx=np.asarray([int(row["source_idx"]) for row in records], dtype=np.int32),
        target_idx=np.asarray([int(row["target_idx"]) for row in records], dtype=np.int32),
        source_zone=np.asarray([str(row["source_zone"]) for row in records], dtype=object),
        target_zone=np.asarray([str(row["target_zone"]) for row in records], dtype=object),
        touched_zones=_object_array([row["touched_zones"] for row in records]),
        path_node_types=_object_array([row.get("path_node_types", []) for row in records]),
        uses_virtual_nodes=np.asarray([bool(row.get("uses_virtual_nodes")) for row in records], dtype=np.bool_),
        virtual_transition_metadata=_object_array([row.get("virtual_transition_metadata", []) for row in records]),
        embedding_dim=np.asarray([max_dim], dtype=np.int32),
    )
    summary_path = out / "track4_action_summary.json"
    summary = _json_safe(dict(result.get("summary") or {}))
    summary["artifact"] = str(npz_path)
    summary["records"] = [
        _json_safe({key: value for key, value in row.items() if key != "path_indices"})
        | {"path_indices": [int(idx) for idx in row["path_indices"]]}
        for row in records
    ]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"npz": str(npz_path), "summary": str(summary_path)}
