"""
complete_pipeline.py  end-to-end orchestration for the Belief Transformer / bias-geometry experiments.

This file is intentionally "pipeline glue": it wires together the NLI extractor, optional temporal modeling,
kernel feature maps, attention aggregation, and provenance tracking. It is designed to be:

1) Reproducible
   - deterministic seeds per observer
   - explicit device handling
   - stable, traceable output artifacts

2) Traceable (thesis-ready)
   - every article gets a stable UID (`bt_uid`) so arrays, plots, and sidecar metadata can be joined without
     relying on fragile "list index" assumptions.
   - timestamps are preserved and normalized (epoch + ISO) so downstream timeline/map visualizations
     can be built reliably, even when raw inputs mix formats.

3) Safe to iterate
   - the pipeline never mutates provenance into model inputs
   - optional caching is explicitly surfaced via diagnostics
   - ordering transforms (temporal sort / restoration) are recorded so you can explain them in writing.

Key invariants this pipeline tries to uphold:

A) Index stability
   You should be able to re-run the same month/corpus and still match points across:
     - embedding matrices
     - UMAP/PCA plots
     - polygon/line overlays
     - per-article metadata panels
   without hand-waving about "some list changed".

B) Timestamp integrity
   If the source dataset contains `published_at` (or any supported timestamp field), we:
     - keep the raw value,
     - parse it to a UTC epoch (seconds),
     - produce a normalized ISO string,
     - expose per-article coverage stats.

C) No silent collapse
   Exact duplicates can be detected and optionally deduplicated, but by default we preserve records and
   instead disambiguate colliding UIDs with deterministic suffixes. This prevents accidental "clumps"
   caused by repeated identical rows while still letting you diagnose the data problem explicitly.

Supported timestamp fields (checked in order):
  - published_at
  - timestamp
  - date
  - created_at
  - updated_at

If none parse cleanly, we keep the raw values and fall back to a stable order tie-breaker (`bt_uid`, then
original index). This keeps temporal sorting deterministic rather than "whatever Python happened to do".

"""

import os
import json
import time
import hashlib
import math
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import datetime
from email.utils import parsedate_to_datetime

from .nli_extraction import NLIExtractor, RepKind, validate_operation
from .temporal_gru import TemporalGRU
from .rks_feature_map import RKSFeatureMap, SharedBasis
from .cross_article_attention import CrossArticleAttention
from .pca_removal import remove_top_pca_component, fit_whitening_matrix, apply_whitening_fixed
from .attention_recorder import AttentionRecorder
from .provenance_tracker import ProvenanceTracker, ProvenanceEntry

# Try to import Dirichlet fusion (optional, for new pipeline)
try:
    from .dirichlet_fusion import DirichletFusion, DirichletFusionConfig
    HAS_DIRICHLET_FUSION = True
except ImportError:
    HAS_DIRICHLET_FUSION = False
    DirichletFusion = None
    DirichletFusionConfig = None

# Try to import PhaseSpaceIntegrator (Track 5 - Final Exit Gate)
try:
    from .phase_space_integrator import (
        PhaseSpaceIntegrator,
        IntegratorConfig,
        IntegratedParticle,
    )
    HAS_PHASE_SPACE = True
except ImportError:
    HAS_PHASE_SPACE = False
    PhaseSpaceIntegrator = None
    IntegratorConfig = None
    IntegratedParticle = None

# --------------------------------------------------------------------------------------
# GRU Hard Disable Flag
# --------------------------------------------------------------------------------------
# The GRU expects a specific input dimension (8192) but receives 12288 from the CLS
# channel. This is a dimension mismatch that needs architectural changes to fix.
# For now, we hard-disable the GRU to allow experiments to proceed.
_GRU_HARD_DISABLED = True


# --------------------------------------------------------------------------------------
# Information Preservation Utilities (Thesis-Grade Provenance)
# --------------------------------------------------------------------------------------

def get_git_hash() -> str:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return 'unknown'


def get_git_dirty() -> bool:
    """Check if git repo has uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        pass
    return True


def build_structured_output_path(
    output_dir: Path,
    kernel_type: str,
    channel: str,
    corpus_name: str,
    seed: int,
) -> Path:
    """
    Build output path with kernel/channel/corpus structure.
    
    Structure: output_dir/kernel/channel/corpus/observer_{seed}.pt
    """
    structured_dir = output_dir / kernel_type / channel / corpus_name
    structured_dir.mkdir(parents=True, exist_ok=True)
    return structured_dir / f"observer_{seed}.pt"


def compute_variance_stats(tensor: torch.Tensor, name: str) -> Dict[str, float]:
    """Compute variance statistics for a tensor (for stage tracking)."""
    if tensor is None or tensor.numel() == 0:
        return {}
    with torch.no_grad():
        flat = tensor.reshape(-1).float()
        stats = {
            f'{name}_mean': float(flat.mean().item()),
            f'{name}_std': float(flat.std().item()),
            f'{name}_var': float(flat.var().item()),
            f'{name}_min': float(flat.min().item()),
            f'{name}_max': float(flat.max().item()),
        }
        if tensor.dim() >= 2:
            stats[f'{name}_norm_mean'] = float(tensor.float().norm(dim=-1).mean().item())
        return stats


class VarianceTracker:
    """Track variance at each pipeline stage for information preservation."""
    
    def __init__(self):
        self.stages = {}
        self.metadata = {}
    
    def record_stage(self, name: str, tensor):
        """Record variance stats for a pipeline stage. Accepts torch.Tensor or numpy.ndarray."""
        if tensor is None:
            return
        # Handle both numpy arrays and torch tensors
        if isinstance(tensor, np.ndarray):
            if tensor.size > 0:
                tensor_torch = torch.from_numpy(tensor)
                self.stages[name] = compute_variance_stats(tensor_torch, name)
                self.stages[name]['shape'] = list(tensor.shape)
        elif hasattr(tensor, 'numel') and tensor.numel() > 0:
            self.stages[name] = compute_variance_stats(tensor, name)
            self.stages[name]['shape'] = list(tensor.shape)
    
    def record_metadata(self, key: str, value):
        """Record arbitrary metadata."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict:
        """Export all tracked data."""
        return {'stages': self.stages, 'metadata': self.metadata}


# --------------------------------------------------------------------------------------
# Constitutional Contract Enforcement
# --------------------------------------------------------------------------------------

def _get_channel_rep_kind(channel_name: str, use_cls_tokens: bool) -> RepKind:
    """
    Determine RepKind for a given channel.
    
    Rules:
    - 'logits' channel -> LOGITS_RAW (never kernelized)
    - 'cls' or 'main' with use_cls_tokens -> CLS_VIEWS
    - 'main' without use_cls_tokens -> LOGITS_RAW
    """
    channel_lower = channel_name.lower()
    if channel_lower == "logits":
        return RepKind.LOGITS_RAW
    elif use_cls_tokens:
        return RepKind.CLS_VIEWS
    else:
        return RepKind.LOGITS_RAW


def _check_operation_allowed(rep_kind: RepKind, operation: str, warn_only: bool = False) -> bool:
    """
    Check if an operation is allowed for a rep_kind.

    Args:
        rep_kind: The representation kind
        operation: Operation name (e.g., 'rks', 'pca_removal', 'procrustes')
        warn_only: If True, just warn; if False, return False for forbidden ops

    Returns:
        True if operation should proceed, False if it should be skipped
    """
    try:
        if validate_operation(rep_kind, operation):
            return True
    except ValueError as e:
        msg = f"[CONSTITUTIONAL CONTRACT] Skipping '{operation}' for {rep_kind.value}: {e}"
        print(f"WARNING: {msg}")
        return False

    msg = f"[CONSTITUTIONAL CONTRACT] Skipping '{operation}' for {rep_kind.value} (forbidden operation)"
    print(f"WARNING: {msg}")
    return False
# --------------------------------------------------------------------------------------

_TS_KEYS_PRIORITY = ("published_at", "timestamp", "date", "created_at", "updated_at")


def _safe_str(x) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def parse_timestamp_to_utc(value):
    """
    Parse a timestamp-like value into:
      - dt_utc: timezone-aware datetime in UTC (or None)
      - epoch_s: int seconds since epoch (or None)
      - iso_utc: ISO 8601 string in UTC with 'Z' suffix (or None)

    Accepts:
      - int/float epoch (seconds or milliseconds)
      - ISO 8601 strings (with or without timezone)
      - RFC 2822 / HTTP-date strings (via email.utils)
      - date-only strings YYYY-MM-DD

    Never raises: returns (None, None, None) if parsing fails.
    """
    if value is None:
        return None, None, None

    # Epoch numeric
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            # Heuristic: > 1e12 implies milliseconds
            if v > 1e12:
                v = v / 1000.0
            dt = datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc)
            epoch_s = int(dt.timestamp())
            iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            return dt, epoch_s, iso
        except Exception:
            return None, None, None

    s = _safe_str(value).strip()
    if not s:
        return None, None, None

    # Pure digits
    if s.isdigit():
        try:
            v = float(s)
            if v > 1e12:
                v = v / 1000.0
            dt = datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc)
            epoch_s = int(dt.timestamp())
            iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            return dt, epoch_s, iso
        except Exception:
            pass

    # ISO-ish
    try:
        s_iso = s.replace("Z", "+00:00")  # fromisoformat can't read 'Z'
        dt = datetime.datetime.fromisoformat(s_iso)
        if dt.tzinfo is None:
            # Treat naive timestamps as UTC (document this in thesis!)
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        dt_utc = dt.astimezone(datetime.timezone.utc)
        epoch_s = int(dt_utc.timestamp())
        iso = dt_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return dt_utc, epoch_s, iso
    except Exception:
        pass

    # Date-only
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            dt = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
            epoch_s = int(dt.timestamp())
            iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            return dt, epoch_s, iso
    except Exception:
        pass

    # RFC 2822 / HTTP-date
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        dt_utc = dt.astimezone(datetime.timezone.utc)
        epoch_s = int(dt_utc.timestamp())
        iso = dt_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return dt_utc, epoch_s, iso
    except Exception:
        return None, None, None


def extract_article_timestamp(article: dict):
    """
    Return:
      (source_key, raw_value, dt_utc, epoch_s, iso_utc)

    Where source_key is the field name we chose from _TS_KEYS_PRIORITY.
    """
    for k in _TS_KEYS_PRIORITY:
        if k in article and article.get(k) is not None:
            raw = article.get(k)
            dt, epoch_s, iso = parse_timestamp_to_utc(raw)
            return k, raw, dt, epoch_s, iso
    return None, None, None, None, None


def stable_article_uid(article: dict, fallback_index: int, _collision_counter: dict):
    """
    Produce a stable per-article identifier.

    Strategy:
      1) Prefer explicit IDs if present (id/uid/guid/doc_id/article_id).
      2) Else hash a canonical string built from URL + title + source + timestamp.
      3) If collisions occur (exact duplicates), append a deterministic suffix _{n}.

    This ensures downstream joins do NOT depend on list ordering.
    """
    for k in ("bt_uid", "uid", "id", "guid", "doc_id", "article_id"):
        v = article.get(k)
        if v:
            base = _safe_str(v)
            break
    else:
        url = article.get("url") or article.get("link") or ""
        title = article.get("title") or article.get("headline") or ""
        source = article.get("source") or article.get("publisher") or ""
        _, raw_ts, _, _, _ = extract_article_timestamp(article)
        ts = _safe_str(raw_ts)
        base_str = f"url={url}|title={title}|source={source}|ts={ts}"
        base = hashlib.sha1(base_str.encode("utf-8", "ignore")).hexdigest()[:16]

    # Collision handling: preserve duplicates, but make them joinable deterministically
    n = _collision_counter.get(base, 0)
    _collision_counter[base] = n + 1
    return base if n == 0 else f"{base}_{n}"


class MetricTracker:
    """Track metrics with mean/std/min/max aggregation."""
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(float(value))

    def summary(self):
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
        return summary



# --------------------------------------------------------------------------------------
# Kernel geometry helpers ("adult mode"): exact kernel PCA + Nystrm approximation
# --------------------------------------------------------------------------------------

def _pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
    """Return NxN matrix of squared Euclidean distances."""
    # torch.cdist returns Euclidean distances; square it
    return torch.cdist(X, X, p=2) ** 2


def _pairwise_l1_dists(X: torch.Tensor) -> torch.Tensor:
    """Return NxN matrix of L1 distances."""
    # Use built-in cdist for p=1 (L1) to avoid OOM on moderate N
    return torch.cdist(X, X, p=1)


def _median_heuristic_sigma_from_d2(d2: torch.Tensor) -> float:
    """Median heuristic for RBF sigma using squared distances."""
    # Use upper triangle (excluding diagonal) to avoid zeros dominating
    n = d2.shape[0]
    if n < 2:
        return 1.0
    tri = d2[torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)]
    if tri.numel() == 0:
        return 1.0
    med = torch.median(tri).item()
    # For RBF: exp(-d^2/(2*sigma^2)); using median(d) is typical. Here we have median(d^2).
    sigma = max(1e-6, math.sqrt(max(1e-12, med)) / math.sqrt(2.0))
    return float(sigma)


def _compute_kernel_matrix(
    X: torch.Tensor,
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute an NxN kernel matrix K for features X.

    Supported kernels:
      - rbf: exp(-||x-y||^2 / (2*sigma^2))  (sigma via median heuristic if None)
      - laplacian: exp(-||x-y||_1 / sigma)  (sigma via median heuristic on L1 if None)
      - rq (rational quadratic): (1 + ||x-y||^2 / (2*alpha*sigma^2))^{-alpha} where alpha=gamma
      - imq (inverse multiquadric): (1 + ||x-y||^2 / sigma^2)^{-1/2}
    """
    kt = (kernel_type or "rbf").lower().strip()

    if kt == "laplacian":
        d1 = _pairwise_l1_dists(X)
        if sigma is None:
            # median heuristic on L1
            n = d1.shape[0]
            tri = d1[torch.triu(torch.ones_like(d1, dtype=torch.bool), diagonal=1)]
            med = torch.median(tri).item() if tri.numel() else 1.0
            sigma = float(max(1e-6, med))
        K = torch.exp(-d1 / float(sigma))
        return K

    d2 = _pairwise_sq_dists(X)

    if kt == "rbf":
        if sigma is None:
            sigma = _median_heuristic_sigma_from_d2(d2)
        K = torch.exp(-d2 / (2.0 * (float(sigma) ** 2)))
        return K

    if kt == "rq":
        # gamma here plays role of alpha (shape). sigma controls scale.
        alpha = float(max(1e-6, gamma))
        if sigma is None:
            sigma = _median_heuristic_sigma_from_d2(d2)
        K = (1.0 + d2 / (2.0 * alpha * (float(sigma) ** 2))).pow(-alpha)
        return K

    if kt == "imq":
        if sigma is None:
            sigma = _median_heuristic_sigma_from_d2(d2)
        K = (1.0 + d2 / (float(sigma) ** 2)).pow(-0.5)
        return K

    # Fallback: treat as RBF
    if sigma is None:
        sigma = _median_heuristic_sigma_from_d2(d2)
    return torch.exp(-d2 / (2.0 * (float(sigma) ** 2)))


def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center kernel matrix."""
    n = K.shape[0]
    if n < 2:
        return K
    one_n = torch.ones((n, n), device=K.device, dtype=K.dtype) / float(n)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def _kernel_pca_from_kernel(K: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Kernel PCA embedding from centered kernel matrix.

    Returns Y: [N, n_components] with coordinates sqrt(lambda)*v.
    """
    n = K.shape[0]
    if n == 0:
        return K
    # eigh for symmetric matrices; returns ascending eigenvalues
    evals, evecs = torch.linalg.eigh(K)
    # take top components
    k = int(min(n_components, n - 1, evals.numel()))
    if k <= 0:
        return torch.zeros((n, 1), device=K.device, dtype=K.dtype)
    idx = torch.argsort(evals, descending=True)[:k]
    L = evals[idx].clamp_min(0)
    V = evecs[:, idx]
    Y = V * torch.sqrt(L).unsqueeze(0)
    return Y


def kernel_pca_embed(
    X: torch.Tensor,
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    sigma: Optional[float] = None,
    n_components: int = 128,
    center: bool = True,
) -> torch.Tensor:
    """Convenience: compute K then (optionally) center then kernel PCA."""
    K = _compute_kernel_matrix(X, kernel_type=kernel_type, gamma=gamma, sigma=sigma)
    if center:
        K = _center_kernel(K)
    return _kernel_pca_from_kernel(K, n_components=n_components)


def nystrom_kernel_pca_embed(
    X: torch.Tensor,
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    sigma: Optional[float] = None,
    n_components: int = 128,
    m_landmarks: int = 256,
    center: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """
    Nystrm approximation for kernel PCA.

    Strategy:
      - sample m landmark indices (uniform)
      - compute C = K(X, landmarks) [N,m]
      - compute W = K(landmarks, landmarks) [m,m]
      - approximate K  C W^{-1} C^T (optionally centered approximately by centering the implicit K)

    For the embedding, we compute the eigendecomposition of W and project.

    Note: This is a pragmatic implementation for moderate N. For very large N, you'd want
    more careful centering + numerical stabilization.
    """
    n = X.shape[0]
    m = int(min(max(1, m_landmarks), n))
    if n == 0:
        return torch.zeros((0, n_components), device=X.device, dtype=X.dtype)

    # Sample landmarks deterministically
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)[:m]
    perm = perm.to(X.device)

    X_l = X[perm]

    # Compute C and W
    # Compute full pairwise between X and X_l by concatenating and slicing would be expensive.
    # We'll compute squared distances explicitly for RBF-like kernels.
    kt = (kernel_type or "rbf").lower().strip()

    # For simplicity, reuse kernel computations by building pairwise with cdist.
    if kt == "laplacian":
        # L1 distances between all points and landmarks
        diff = X.unsqueeze(1) - X_l.unsqueeze(0)  # [N,m,D]
        d1 = diff.abs().sum(dim=-1)
        if sigma is None:
            # estimate sigma on landmarks
            d1_ll = _pairwise_l1_dists(X_l)
            tri = d1_ll[torch.triu(torch.ones_like(d1_ll, dtype=torch.bool), diagonal=1)]
            med = torch.median(tri).item() if tri.numel() else 1.0
            sigma = float(max(1e-6, med))
        C = torch.exp(-d1 / float(sigma))
        W = _compute_kernel_matrix(X_l, kernel_type="laplacian", gamma=gamma, sigma=sigma)
    else:
        d2 = torch.cdist(X, X_l, p=2) ** 2  # [N,m]
        if sigma is None:
            # estimate sigma from landmark-landmark distances
            d2_ll = _pairwise_sq_dists(X_l)
            sigma = _median_heuristic_sigma_from_d2(d2_ll)
        if kt == "rbf":
            C = torch.exp(-d2 / (2.0 * (float(sigma) ** 2)))
        elif kt == "rq":
            alpha = float(max(1e-6, gamma))
            C = (1.0 + d2 / (2.0 * alpha * (float(sigma) ** 2))).pow(-alpha)
        elif kt == "imq":
            C = (1.0 + d2 / (float(sigma) ** 2)).pow(-0.5)
        else:
            C = torch.exp(-d2 / (2.0 * (float(sigma) ** 2)))
        W = _compute_kernel_matrix(X_l, kernel_type=kt, gamma=gamma, sigma=sigma)

    # Stabilize W
    eps = 1e-6
    W = (W + W.T) * 0.5
    W = W + eps * torch.eye(m, device=W.device, dtype=W.dtype)

    # Approximate centered K by centering C and W (approx)
    if center:
        # Center using landmark means as a proxy
        one_n = torch.ones((n, 1), device=C.device, dtype=C.dtype) / float(n)
        one_m = torch.ones((m, 1), device=C.device, dtype=C.dtype) / float(m)

        C_mean_row = (one_n.T @ C)  # [1,m]
        C = C - one_n @ C_mean_row  # center rows

        W_mean_row = (one_m.T @ W)  # [1,m]
        W_mean_col = (W @ one_m)    # [m,1]
        W_mean_all = (one_m.T @ W @ one_m)  # [1,1]
        W = W - one_m @ W_mean_row - W_mean_col @ one_m.T + one_m @ W_mean_all @ one_m.T

    # Eigendecompose W
    evals, evecs = torch.linalg.eigh(W)
    idx = torch.argsort(evals, descending=True)

    # Keep positive components
    evals = evals[idx].clamp_min(1e-12)
    evecs = evecs[:, idx]

    k = int(min(n_components, m, evals.numel()))
    evals_k = evals[:k]
    evecs_k = evecs[:, :k]

    # Nystrm feature map: Phi = C * evecs_k * diag(1/sqrt(evals_k))
    Phi = C @ (evecs_k / torch.sqrt(evals_k).unsqueeze(0))
    # Then embed like PCA coordinates: multiply by sqrt(evals_k)
    Y = Phi * torch.sqrt(evals_k).unsqueeze(0)
    return Y

def initialize_full_pipeline(
    random_seed: int = 42,
    device: str = "cuda",
    embedding_dim: int = 24,
    hidden_dim: int = 256,
    output_dim: int = 2048,
    use_gru: bool = True,
    use_rks: bool = True,
    use_attention: bool = True,
    use_contrastive: bool = False,
    use_cls_tokens: bool = False,
    normalize_features: bool = True,
    pca_remove: bool = True,
    global_pca_component: bool = True,
    # New (no CLI sprawl): geometry + comparison modes
    geometry_mode: str = "rks",  # "rks" | "kernel_pca" | "nystrom" | "none"
    adult_kernel: str = "rbf",
    adult_gamma: float = 1.0,
    adult_sigma: Optional[float] = None,
    adult_center: bool = True,
    adult_nystrom_m: int = 256,
    # NEW: Dirichlet fusion parameters
    use_dirichlet_fusion: bool = False,
    dirichlet_alpha: float = 1.0,
    dirichlet_n_observers: int = 50,
    dirichlet_rks_dim: int = 2048,
    dirichlet_basis_path: Optional[str] = None,
    dirichlet_weights_path: Optional[str] = None,
    dirichlet_basis_seed: int = 42,
    dirichlet_crn_seed: int = 12345,
    dirichlet_hidden_dim: int = 1536,  # Must match NLI model hidden size (DeBERTa-large=1536)
    # NEW: Force logits-raw mode (constitutional contract)
    force_logits_raw: bool = False,
    compare_logits_vs_cli: bool = False,
    rks_output_dim: Optional[int] = None,
    # NEW: KernelContext integration (Phase 1)
    kernel_ctx: Optional[Any] = None,  # KernelContext instance
    kernel_type: str = "rbf",          # Fallback if no kernel_ctx
    kernel_bandwidth: Optional[float] = None,  # Fallback sigma
    mix_in_rkhs: bool = False,         # Mode B for Dirichlet fusion
):
    """
    Initialize the full belief transformer pipeline components.

    This is the "module factory"  it builds the objects that `BeliefTransformerPipeline`
    uses in `process_month()`.

    Parameters
    ----------
    random_seed : int
        Controls torch/NumPy RNG for reproducibility inside this observer instance.
    device : str
        Torch device string, e.g. 'cuda' or 'cpu'.
    embedding_dim : int
        Dimension of base embeddings (e.g., DeBERTa pooled size).
    hidden_dim : int
        Hidden size for the TemporalGRU (if enabled).
    output_dim : int
        Final representation dim after the temporal stage.
    use_gru : bool
        If True, run TemporalGRU over the corpus in timestamp order (then restore original order).
    use_rks : bool
        If True, apply Random Kitchen Sinks feature map over the (optionally temporal) features.
    use_attention : bool
        If True, run cross-article attention aggregation on top of RKS features.
    use_contrastive : bool
        If True, enable contrastive normalization in attention/RKS stages where supported.
    use_cls_tokens : bool
        If True, use CLS+logits stacking mode (8192D output from NLI, PCA applied to CLS during extraction).
    normalize_features : bool
        If True, L2-normalize features at key boundaries to stabilize geometry metrics.
    pca_remove : bool
        If True, remove top principal component(s). Disabled automatically if use_cls_tokens=True.
    global_pca_component : bool
        If True, compute the PCA component globally (across the month) rather than per-batch.

    Returns
    -------
    dict
        Dictionary of initialized components.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # PCA rule (hard invariant): never PCA logits in the pipeline.
    # CLS PCA (if enabled) happens inside the NLI extractor, and only touches CLS-derived features.
    # [FIX] Overrides removed to allow manual control.
    # if not use_cls_tokens:
    #     pca_remove = False

    nli_extractor = None
    # NLI extractor wiring: filter kwargs so older extractor versions don't crash.
    import inspect

    nli_kwargs = {
        "device": device,
        # Paragraph awareness is a hard invariant for this project.
        "paragraph_aware": True,
        "paragraph_weights": [0.5, 0.3, 0.2],
    }

    # Dual-mode extraction: compute BOTH logits and CLI for later side-by-side pipelines.
    if compare_logits_vs_cli:
        nli_kwargs["extract_cli"] = True
        nli_kwargs["cli_mode"] = "contrastive"

    # CLS channel regime (only if NLIExtractor version supports it)
    if use_cls_tokens:
        nli_kwargs.update({
            "use_cls_tokens": True,
            "projection_dim": 256,
            "apply_pca_to_cls": False,  # Jagged Truth: external backdrop only
            "normalize_before_projection": True,
        })

    sig = inspect.signature(NLIExtractor.__init__)
    filtered = {k: v for k, v in nli_kwargs.items() if k in sig.parameters}
    missing = [k for k in nli_kwargs.keys() if k not in filtered]
    if missing:
        # Don't hard fail; just document the downgrade.
        if use_cls_tokens and any(k in missing for k in ("use_cls_tokens", "projection_dim", "apply_pca_to_cls")):
            print("WARNING: NLIExtractor.__init__ does not accept CLS channel args; falling back to legacy extraction.")
        if compare_logits_vs_cli and "extract_cli" in missing:
            print("WARNING: NLIExtractor.__init__ does not accept extract_cli; logits/CLI comparison disabled.")

    nli_extractor = NLIExtractor(**filtered)

    # If CLS channel is active, infer embedding_dim from extractor output.
    # This prevents GRU/RKS dim mismatches and avoids any implicit CLS+logits stacking assumptions.
    if use_cls_tokens:
        try:
            # ROBUST FIX: Try multiple paths to get CLS dimension
            # Priority: nli_extractor.output_dim > nli_extractor.extractor.cls_dim > fallback
            inferred_dim = None

            # Path 1: NLIExtractor.output_dim (should be cls_dim when use_cls_tokens=True)
            if hasattr(nli_extractor, 'output_dim'):
                inferred_dim = int(nli_extractor.output_dim)

            # Path 2: Directly access UnifiedNLIExtractor.cls_dim
            if inferred_dim is None or inferred_dim <= 24:  # 24 = logits_dim, means wrong path
                inner = getattr(nli_extractor, 'extractor', None)
                if inner and hasattr(inner, 'cls_dim'):
                    inferred_dim = int(inner.cls_dim)

            # Path 3: Compute from hidden_size * n_bots
            if inferred_dim is None or inferred_dim <= 24:
                inner = getattr(nli_extractor, 'extractor', None)
                if inner:
                    h = getattr(inner, 'hidden_size', 1536)
                    b = getattr(inner, 'n_bots', 8)
                    inferred_dim = h * b

            if inferred_dim and inferred_dim > 24:
                embedding_dim = inferred_dim
                print(f"[FIX] CLS embedding_dim correctly inferred as {embedding_dim}")
            else:
                print(f"[WARNING] Could not infer CLS dim, falling back to {embedding_dim}")
        except Exception as e:
            print(f"[WARNING] embedding_dim inference failed: {e}")

        print(f"CLS-only channel enabled: embedding_dim = {embedding_dim}. (Logits are not concatenated.)")


    gru_model = None
    if use_gru:
        if _GRU_HARD_DISABLED:
            print("[GRU] Hard disabled via _GRU_HARD_DISABLED flag - skipping initialization")
        else:
            gru_model = TemporalGRU(
                feature_dim=embedding_dim,
                hidden_dim=hidden_dim,
                seed=random_seed,
            )

    # Effective output dimension for representation stage.
    # - If geometry_mode == "rks": final_dim is the RKS output dim.
    # - If geometry_mode != "rks": final_dim starts as output_dim, but may be capped at runtime (<= N-1 for exact kernel PCA).
    final_dim = int(rks_output_dim or output_dim)

    geometry_mode = (geometry_mode or "rks").lower().strip()
    if geometry_mode not in ("rks", "kernel_pca", "nystrom"):
        print(f"WARNING: Unknown geometry_mode={geometry_mode!r}; defaulting to 'rks'")
        geometry_mode = "rks"

    rks_map = None
    if use_rks and geometry_mode == "rks":
        # Random Kitchen Sinks feature map (explicit finite-dimensional kernel approximation)
        # FIX: Check if GRU was actually created, not just requested (GRU may be hard-disabled)
        rks_map = RKSFeatureMap(
            input_dim=output_dim if gru_model is not None else embedding_dim,
            output_dim=final_dim,
            kernel_type="rbf",
            gamma=1.0,
            random_seed=random_seed,
            device=device,
        )

    # NEW: Build or use provided KernelContext
    built_kernel_ctx = kernel_ctx
    if built_kernel_ctx is None and use_cls_tokens:
        # Build KernelContext from parameters for CLS channel
        try:
            from .kernel_context import KernelContext
            # Use dirichlet_hidden_dim (1536 for DeBERTa-large), NOT hardcoded 768
            cls_hidden_dim = dirichlet_hidden_dim  # 1536 for DeBERTa-large
            if kernel_type.lower() == 'rbf':
                built_kernel_ctx = KernelContext(
                    kernel_type='rbf',
                    input_dim=cls_hidden_dim,  # Use actual hidden dim, not 768
                    rks_dim=dirichlet_rks_dim,
                    seed=dirichlet_basis_seed,
                    bandwidth=kernel_bandwidth or adult_sigma or 1.0,
                )
            elif kernel_type.lower() in ('cosine', 'linear'):
                built_kernel_ctx = KernelContext(
                    kernel_type=kernel_type.lower(),
                    input_dim=cls_hidden_dim,  # Use actual hidden dim, not 768
                    seed=dirichlet_basis_seed,
                )
            if built_kernel_ctx is not None:
                built_kernel_ctx.validate()
                print(f"[PIPELINE] KernelContext: {built_kernel_ctx.canonical_id()}")
        except ImportError:
            print("WARNING: kernel_context.py not found, skipping KernelContext")
        except Exception as e:
            print(f"WARNING: Failed to build KernelContext: {e}")

    # NEW: Initialize Dirichlet fusion for CLS views
    dirichlet_fusion = None
    dirichlet_config = None
    if use_dirichlet_fusion and HAS_DIRICHLET_FUSION:
        if not use_cls_tokens:
            print("WARNING: Dirichlet fusion requires use_cls_tokens=True. Enabling CLS mode.")
            use_cls_tokens = True
        
        dirichlet_config = DirichletFusionConfig(
            n_bots=8,
            hidden_dim=dirichlet_hidden_dim,
            rks_dim=dirichlet_rks_dim,
            n_observers=dirichlet_n_observers,
            alpha=dirichlet_alpha,
            kernel_type=kernel_type,
            basis_seed=dirichlet_basis_seed,
            basis_path=dirichlet_basis_path,
            crn_enabled=True,
            locked_dirichlet_weights=dirichlet_weights_path,
            crn_seed=dirichlet_crn_seed,
            mix_in_rkhs=mix_in_rkhs,  # NEW: Mode A vs Mode B
            kernel_ctx=built_kernel_ctx,  # NEW: Pass kernel context
            # ASTER v3.2: Sequential Annealing (HotCold hysteresis)
            use_annealing=True,
            annealing_schedule=[10.0, 5.0, 2.0, 1.0, 0.5, 0.1],
        )
        dirichlet_fusion = DirichletFusion(dirichlet_config)
        mode_str = "Mode B (map-then-mix)" if mix_in_rkhs else "Mode A (mix-then-map)"
        print(f"[PIPELINE] Dirichlet fusion initialized: {mode_str}, alpha={dirichlet_alpha}, K={dirichlet_n_observers}, annealing=ENABLED")
    elif use_dirichlet_fusion and not HAS_DIRICHLET_FUSION:
        print("WARNING: Dirichlet fusion requested but dirichlet_fusion.py not found. Skipping.")

    # Determine primary rep_kind for contract enforcement
    if force_logits_raw:
        primary_rep_kind = RepKind.LOGITS_RAW
        # Enforce constitutional constraints
        use_rks = False
        pca_remove = False
        geometry_mode = "none"
        print("[CONSTITUTIONAL CONTRACT] force_logits_raw=True: RKS, PCA, geometry disabled")
    elif use_cls_tokens:
        primary_rep_kind = RepKind.CLS_VIEWS
    else:
        primary_rep_kind = RepKind.LOGITS_RAW

    attention_model = None
    if use_attention:
        attention_model = CrossArticleAttention(
            feature_dim=final_dim,
            num_heads=8,
        )

    recorder = AttentionRecorder()

    return {
        "nli_extractor": nli_extractor,
        "gru_model": gru_model,
        "rks_map": rks_map,
        "attention_model": attention_model,
        "recorder": recorder,
        "normalize_features": normalize_features,
        "pca_remove": pca_remove,
        "global_pca_component": global_pca_component,
        "use_cls_tokens": use_cls_tokens,
        "compare_logits_vs_cli": compare_logits_vs_cli,
        "geometry_mode": geometry_mode,
        "adult_kernel": adult_kernel,
        "adult_gamma": adult_gamma,
        "adult_sigma": adult_sigma,
        "adult_center": adult_center,
        "adult_nystrom_m": adult_nystrom_m,
        "final_dim": final_dim,
        # NEW: Dirichlet fusion components
        "dirichlet_fusion": dirichlet_fusion,
        "dirichlet_config": dirichlet_config,
        # NEW: Rep kind for contract enforcement
        "primary_rep_kind": primary_rep_kind,
        "force_logits_raw": force_logits_raw,
        # NEW: KernelContext for provenance
        "kernel_ctx": built_kernel_ctx,
        "mix_in_rkhs": mix_in_rkhs,
    }


class BeliefTransformerPipeline:
    """
    Main pipeline class for multi-step bias geometry processing.

    This class is instantiated per "observer" seed  meaning each run can have its own
    kernel random features and any seeded stochasticity.

    IMPORTANT: `process_month()` returns arrays AND traceable sidecars:
      - `timeline` (per-article timestamps, normalized)
      - `bt_uid_list` (stable IDs for joins)

    That is what keeps downstream map-making + polygon matching from breaking when list
    order changes (or duplicates exist).

    Parameters
    ----------
    components : dict
        Output of `initialize_full_pipeline()`.
    random_seed : int
        Seed for any internal randomness (RKS, dropout-like ops, etc.).
    enable_provenance : bool
        Whether to write provenance entries for each pipeline stage.
    provenance_dir : str | Path | None
        If set, provenance JSON entries will be saved here.
    """

    def __init__(
        self,
        components: Dict[str, Any],
        random_seed: int = 42,
        enable_provenance: bool = True,
        provenance_dir: Optional[str] = None,
    ):
        self.components = components
        self.random_seed = random_seed

        self.nli_extractor = components["nli_extractor"]
        self.gru_model = components["gru_model"]
        self.rks_map = components["rks_map"]
        self.attention_model = components["attention_model"]
        self.recorder = components["recorder"]

        self.normalize_features = components.get("normalize_features", True)
        self.pca_remove = components.get("pca_remove", True)
        self.global_pca_component = components.get("global_pca_component", True)
        self.use_cls_tokens = components.get("use_cls_tokens", False)

        self.compare_logits_vs_cli = components.get("compare_logits_vs_cli", False)

        # Geometry mode: "rks" (Random Kitchen Sinks) or "kernel_pca"/"nystrom" (kernel trick / spectral) or "none"
        self.geometry_mode = (components.get("geometry_mode", "rks") or "rks").lower().strip()
        self.adult_kernel = components.get("adult_kernel", "rbf")
        self.adult_gamma = float(components.get("adult_gamma", 1.0))
        self.adult_sigma = components.get("adult_sigma", None)
        self.adult_center = bool(components.get("adult_center", True))
        self.adult_nystrom_m = int(components.get("adult_nystrom_m", 256))

        # Nominal representation dim (may be capped at runtime for exact kernel PCA).
        self.final_dim = int(components.get("final_dim", 0) or 0)

        # NEW: Dirichlet fusion components
        self.dirichlet_fusion = components.get("dirichlet_fusion", None)
        self.dirichlet_config = components.get("dirichlet_config", None)
        
        # NEW: Rep kind for contract enforcement
        self.primary_rep_kind = components.get("primary_rep_kind", RepKind.LOGITS_RAW)
        self.force_logits_raw = components.get("force_logits_raw", False)
        
        # NEW: KernelContext for provenance
        self.kernel_ctx = components.get("kernel_ctx", None)
        self.mix_in_rkhs = components.get("mix_in_rkhs", False)

        self.enable_provenance = enable_provenance
        self.provenance_dir = provenance_dir
        self.provenance_tracker = None
        if enable_provenance:
            self.provenance_tracker = ProvenanceTracker()

        self._nli_cache = None  # optional cache for repeated runs

    def _sort_by_timestamp(self, articles, bt_uids=None, ts_epoch_s=None):
        """
        Deterministically compute a chronological permutation for the TemporalGRU stage.

        Parameters
        ----------
        articles : list[dict]
            Raw article records.
        bt_uids : list[str] | None
            Stable UIDs (if already computed). If None, they will be computed.
        ts_epoch_s : list[int|None] | None
            Parsed epoch seconds (if already computed). If None, they will be computed.

        Returns
        -------
        sorted_indices : list[int]
            Indices into the original `articles` list that produce chronological order.
        inverse_indices : list[int]
            A list of length N where:
                inverse_indices[orig_idx] = sorted_position
            so that you can restore original order via:
                restored = sorted_tensor[inverse_indices]
        """
        n = len(articles)
        collision_counter = {}

        if bt_uids is None:
            bt_uids = []
            for i, a in enumerate(articles):
                bt_uids.append(stable_article_uid(a, i, collision_counter))
        else:
            # still count collisions for diagnostics parity if caller didn't provide
            for uid in bt_uids:
                collision_counter[uid] = collision_counter.get(uid, 0) + 1

        if ts_epoch_s is None:
            ts_epoch_s = []
            for a in articles:
                _k, _raw, _dt, epoch_s, _iso = extract_article_timestamp(a)
                ts_epoch_s.append(epoch_s)

        # Build sortable tuples (has_ts, epoch_key, uid, orig_idx)
        items = []
        for i in range(n):
            epoch_s = ts_epoch_s[i]
            has_ts = 0 if epoch_s is None else 1
            epoch_key = epoch_s if epoch_s is not None else 2**63 - 1
            items.append((has_ts, epoch_key, bt_uids[i], i))

        items.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
        sorted_indices = [t[3] for t in items]

        inverse_indices = [0] * n
        for sorted_pos, orig_idx in enumerate(sorted_indices):
            inverse_indices[orig_idx] = sorted_pos

        return sorted_indices, inverse_indices

    def process_month(self, articles: List[Dict], month_name: str = "unknown", config: Optional[Dict] = None) -> Dict:
        """
        Process a month of articles through the full pipeline.

        Args:
            articles: List of article dictionaries
            month_name: Name identifier for this processing run
            config: Optional configuration dict with:
                - enable_checkpoints: bool - Enable waterfall checkpoint saving
                - checkpoint_dir: str - Directory to save checkpoints
                - run_social_texture: bool - Run expensive social texture analysis

        Steps:
          1) NLI extraction -> base embeddings (and optional multi-framing)
          2) Temporal GRU over timestamp-sorted sequence (optional)
          3) Random Kitchen Sinks feature map (optional)
          4) Cross-article attention aggregation (optional)
          5) PCA removal (optional)
          6) Return final features + diagnostics + metadata

        Returns a dict with:
          - features: final tensor/ndarray (N x D)
          - diagnostics: coverage, timing, variance, and ordering stats
          - provenance: optional provenance entries
          - article_metadata: per-article payload for dashboards
          - timeline: timestamp-normalized join table
          - bt_uid_list: stable IDs aligned to returned arrays
        """

        # ------------------------------------------------------------------
        # Initialize optional track outputs (prevent UnboundLocalError)
        # ------------------------------------------------------------------
        logit_confidence_t1 = None
        spectral_results = None
        walker_results = None
        phantom_verdicts = None
        hott_proofs = None
        phase_space_results = None
        d_spectral = None
        dirichlet_results = None
        cls_per_bot_list = None

        # ------------------------------------------------------------------
        # Stable IDs + timestamp normalization (thesis-ready joins)
        # ------------------------------------------------------------------
        _uid_collision_counter = {}
        bt_uids = []
        ts_source_keys = []
        ts_raw_values = []
        ts_epoch_s = []
        ts_iso_utc = []

        for i, art in enumerate(articles):
            uid = stable_article_uid(art, i, _uid_collision_counter)
            bt_uids.append(uid)

            src_key, raw_ts, _dt, epoch_s, iso_utc = extract_article_timestamp(art)
            ts_source_keys.append(src_key)
            ts_raw_values.append(raw_ts)
            ts_epoch_s.append(epoch_s)
            ts_iso_utc.append(iso_utc)

        n_with_epoch = sum(1 for x in ts_epoch_s if x is not None)
        timestamp_coverage = {
            "total": len(articles),
            "parseable": n_with_epoch,
            "parseable_fraction": float(n_with_epoch / max(1, len(articles))),
            "fields_seen": sorted({k for k in ts_source_keys if k}),
        }
        uid_collision_stats = {
            "unique_bases": len(_uid_collision_counter),
            "total_records": len(articles),
            "max_collision_count": max(_uid_collision_counter.values()) if _uid_collision_counter else 0,
        }

        # A single, joinable timeline table: index-safe and plot-safe
        timeline = []
        for i in range(len(articles)):
            timeline.append({
                "bt_uid": bt_uids[i],
                "original_index": i,
                "sorted_position": i,
                "timestamp_field": ts_source_keys[i],
                "timestamp_raw": ts_raw_values[i],
                "timestamp_epoch_s": ts_epoch_s[i],
                "timestamp_iso_utc": ts_iso_utc[i],
            })

        # Store article metadata for provenance
        if self.enable_provenance and self.provenance_tracker:
            provenance_metadata = {
                "month": month_name,
                "n_articles": len(articles),
                "articles": {
                    i: {
                        "bt_uid": bt_uids[i],
                        "id": art.get("id", art.get("url", f"article_{i}")),
                        "source": art.get("source", "unknown"),
                        "url": art.get("url", ""),
                        "timestamp_field": ts_source_keys[i],
                        "timestamp_raw": ts_raw_values[i],
                        "timestamp_epoch_s": ts_epoch_s[i],
                        "timestamp_iso_utc": ts_iso_utc[i],
                        "title": art.get("title", ""),
                        "content_preview": art.get("content", "")[:200] if art.get("content") else "",
                    }
                    for i, art in enumerate(articles)
                },
            }

        # Diagnostic tracking
        diagnostics = {
            "month": month_name,
            "n_articles": len(articles),
            "use_gru": self.gru_model is not None,
            "use_rks": self.rks_map is not None,
            "use_attention": self.attention_model is not None,
            "pca_remove": self.pca_remove,
            "steps": [],
            "timing": {},
            "variance": {},
        }

        diagnostics["timestamp_coverage"] = timestamp_coverage
        diagnostics["uid_collision_bases"] = uid_collision_stats

        start_total = time.time()

        # Step 1: NLI extraction
        step_start = time.time()
        if self._nli_cache is not None and len(self._nli_cache) == len(articles):
            nli_pairs = self._nli_cache
            diagnostics["steps"].append("nli_extraction_cached")
        else:
            nli_pairs = self.nli_extractor.extract_nli_pairs(articles)
            self._nli_cache = nli_pairs
            diagnostics["steps"].append("nli_extraction")

        diagnostics["timing"]["nli_extraction"] = time.time() - step_start

        # Convert to tensor(s)
        # Option B: if compare_logits_vs_cli=True, we run identical downstream processing on BOTH
        # (a) raw logits-derived features and (b) CLI-derived features, then return both for plotting.
        base_by_channel: Dict[str, torch.Tensor] = {}

        if self.compare_logits_vs_cli:
            logits_t = torch.stack([pair.get("embedding_logits", pair["embedding"]) for pair in nli_pairs], dim=0)
            cli_t = torch.stack([pair.get("embedding_cli", pair["embedding"]) for pair in nli_pairs], dim=0)
            base_by_channel["logits"] = logits_t
            base_by_channel["cli"] = cli_t
            diagnostics["steps"].append("dual_channel_logits_cli")
        else:
            base_by_channel["main"] = torch.stack([pair["embedding"] for pair in nli_pairs], dim=0)

        # Normalize each channel independently (prevents one modality from dominating by scale)
        if self.normalize_features:
            for k in list(base_by_channel.keys()):
                base_by_channel[k] = F.normalize(base_by_channel[k], dim=1)

        diagnostics.setdefault("channels", {})
        for k, Xk in base_by_channel.items():
            diagnostics["channels"].setdefault(k, {})
            diagnostics["channels"][k]["embedding_variance"] = float(Xk.var().item())

        # ------------------------------------------------------------------
        # Temporal ordering (computed once, shared across channels)
        # ------------------------------------------------------------------
        sorted_indices = None
        inverse_indices = None
        if self.gru_model is not None:
            sorted_indices, inverse_indices = self._sort_by_timestamp(
                articles, bt_uids=bt_uids, ts_epoch_s=ts_epoch_s
            )

            # Update timeline with the actual temporal permutation used for GRU
            for entry in timeline:
                orig_i = entry["original_index"]
                entry["sorted_position"] = inverse_indices[orig_i]

            diagnostics["steps"].append("temporal_sort_applied")
            diagnostics["temporal_sort_indices"] = sorted_indices

        # ------------------------------------------------------------------
        # Shared downstream runner (GRU -> geometry -> attention -> (optional) PCA)
        # ------------------------------------------------------------------
        def _run_channel(channel_name: str, X_in: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
            ch_diag = diagnostics["channels"].setdefault(channel_name, {})
            t0 = time.time()
            
            # NEW: Determine rep_kind for this channel and track it
            channel_rep_kind = _get_channel_rep_kind(channel_name, self.use_cls_tokens)
            ch_diag["rep_kind"] = channel_rep_kind.value
            
            # NEW: Additional outputs for Dirichlet fusion
            channel_extras = {
                "rep_kind": channel_rep_kind,
                "cls_per_bot": None,  # Will be set if CLS_VIEWS
                "curvature": None,    # Will be set if Dirichlet fusion runs
                "provenance": {},
            }

            # Step 2: Optional temporal GRU
            X = X_in
            if self.gru_model is not None:
                t_gru = time.time()
                # Reorder for GRU
                sorted_tensor = X[sorted_indices]

                # TemporalGRU expects [N, D]. If extra structure sneaks in (e.g., [N, B, D]),
                # flatten everything after the article axis to preserve a well-defined feature vector.
                if sorted_tensor.dim() != 2:
                    sorted_tensor = sorted_tensor.reshape(sorted_tensor.shape[0], -1)

                # IMPORTANT: do NOT add a batch dimension here. TemporalGRU.forward() already
                # unsqueezes to [N, 1, D] for torch.nn.GRU. Adding another unsqueeze would create 4D.
                temporal_out = self.gru_model(sorted_tensor)  # (N, D)

                # Defensive: if an older TemporalGRU returns a leading batch axis, remove it.
                if temporal_out.dim() == 3 and temporal_out.shape[0] == 1:
                    temporal_out = temporal_out.squeeze(0)  # (N, D)

                # Restore original order
                inv = torch.tensor(inverse_indices, dtype=torch.long, device=temporal_out.device)
                X = temporal_out[inv]

                ch_diag["timing_temporal_gru"] = time.time() - t_gru
                ch_diag["temporal_variance"] = float(X.var().item())

            # Step 3: Geometry stage
            t_geom = time.time()
            Z = X
            
            # NEW: Constitutional contract check before RKS
            rks_allowed = _check_operation_allowed(channel_rep_kind, 'rks', warn_only=True)
            
            if self.geometry_mode == "rks" and self.rks_map is not None and rks_allowed:
                # Channel-safe RKS: one feature map per (channel, input_dim, kernel, seed)
                if not hasattr(self, "_rks_map_cache"):
                    self._rks_map_cache = {}
                in_dim = int(X.shape[-1])
                kernel_name = getattr(self.rks_map, "kernel_type", "rbf")
                tmpl_cfg = getattr(self.rks_map, "cfg", None)
                try:
                    seed = int(getattr(tmpl_cfg, "random_seed", 0) if tmpl_cfg is not None else 0)
                except Exception:
                    seed = 0
                key = (channel_name, in_dim, kernel_name, seed)
                if key not in self._rks_map_cache:
                    n_f = int(getattr(tmpl_cfg, "n_framings", 8) if tmpl_cfg is not None else 8)
                    base_out = int(getattr(tmpl_cfg, "output_dim", getattr(self, "final_dim", 512)) if tmpl_cfg is not None else getattr(self, "final_dim", 512))
                    # Default policy: logits stay lightweight; CLS/other channels get the full budget
                    if channel_name.lower() == "logits":
                        out_dim = int(getattr(self, "rks_output_dim_logits", min(base_out, 512)))
                    else:
                        out_dim = int(getattr(self, "rks_output_dim_cls", base_out))
                    if out_dim % n_f != 0:
                        out_dim = max(n_f, (out_dim // n_f) * n_f)
                    device_str = str(X.device)
                    self._rks_map_cache[key] = RKSFeatureMap(
                        input_dim=in_dim,
                        output_dim=out_dim,
                        kernel_type=kernel_name,
                        gamma=getattr(tmpl_cfg, "gamma", None) if tmpl_cfg is not None else None,
                        sigma=getattr(tmpl_cfg, "sigma", None) if tmpl_cfg is not None else None,
                        n_framings=n_f,
                        random_seed=int(getattr(tmpl_cfg, "random_seed", 42) if tmpl_cfg is not None else 42),
                        device=device_str,
                        kernel_params=getattr(tmpl_cfg, "kernel_params", None) if tmpl_cfg is not None else None,
                        auto_sigma=bool(getattr(tmpl_cfg, "auto_sigma", False) if tmpl_cfg is not None else False),
                        verbose=bool(getattr(tmpl_cfg, "verbose", False) if tmpl_cfg is not None else False),
                    )
                Z = self._rks_map_cache[key].transform(X)
                diagnostics["steps"].append(f"rks_feature_map:{channel_name}")
            elif self.geometry_mode == "kernel_pca" and _check_operation_allowed(channel_rep_kind, 'kernel_pca', warn_only=True):
                # Exact kernel trick (O(N^2) memory/time)
                n_components = int(self.final_dim or X.shape[0] - 1 or 1)
                Z = kernel_pca_embed(
                    X,
                    kernel_type=self.adult_kernel,
                    gamma=self.adult_gamma,
                    sigma=self.adult_sigma,
                    n_components=n_components,
                    center=self.adult_center,
                )
                diagnostics["steps"].append(f"kernel_pca:{channel_name}")
            elif self.geometry_mode == "nystrom" and _check_operation_allowed(channel_rep_kind, 'nystrom', warn_only=True):
                # Nystrom approximation (sub-quadratic in N for large N)
                n_components = int(self.final_dim or min(256, X.shape[0] - 1) or 1)
                Z = nystrom_kernel_pca_embed(
                    X,
                    kernel_type=self.adult_kernel,
                    gamma=self.adult_gamma,
                    sigma=self.adult_sigma,
                    n_components=n_components,
                    m_landmarks=self.adult_nystrom_m,
                    center=self.adult_center,
                    seed=self.random_seed,
                )
                diagnostics["steps"].append(f"nystrom_kernel_pca:{channel_name}")
            elif self.geometry_mode == "none":
                # No kernel mapping - pass through (used for logits_raw contract)
                Z = X
                diagnostics["steps"].append(f"geometry_none:{channel_name}")
                ch_diag["contract_note"] = "geometry_mode=none, no kernel mapping applied"

            if self.normalize_features:
                Z = F.normalize(Z, dim=1)

            ch_diag["timing_geometry"] = time.time() - t_geom
            ch_diag["geometry_variance"] = float(Z.var().item())
            ch_diag["geometry_dim"] = int(Z.shape[1])

            # Step 4: Optional attention aggregation
            A = Z
            if self.attention_model is not None:
                t_attn = time.time()

                # If adult mode yields a runtime-capped dimension, rebuild attention to match.
                # [FIX] Use local variable to avoid overwriting global state in multi-channel runs
                current_attn = self.attention_model
                
                if A.dim() == 2:
                    cur_d = int(A.shape[1])
                    # Try to detect mismatch without relying on internal attribute names.
                    need_rebuild = False
                    for attr in ("feature_dim", "d_model", "dim"):
                        if hasattr(current_attn, attr):
                            if int(getattr(current_attn, attr)) != cur_d:
                                need_rebuild = True
                            break
                    if need_rebuild:
                        current_attn = CrossArticleAttention(
                            feature_dim=cur_d,
                            num_heads=8,
                        ).to(A.device)

                attn_res = current_attn(A)

                # Support multiple attention module return conventions:
                # - (attn_out, attn_weights)
                # - (attn_out, attn_weights, *extras)
                # - attn_weights only (NxN), in which case attn_out = attn_weights @ A
                # - attn_out only (NxD), in which case weights are unavailable
                attn_out = None
                attn_weights = None

                if isinstance(attn_res, tuple):
                    if len(attn_res) >= 2:
                        attn_out, attn_weights = attn_res[0], attn_res[1]
                    elif len(attn_res) == 1:
                        attn_weights = attn_res[0]
                    else:
                        attn_out = None
                        attn_weights = None
                else:
                    # single tensor result
                    if torch.is_tensor(attn_res):
                        # Heuristic: square matrix -> weights
                        if attn_res.dim() == 2 and attn_res.shape[0] == A.shape[0] and attn_res.shape[1] == A.shape[0]:
                            attn_weights = attn_res
                        else:
                            attn_out = attn_res
                    else:
                        attn_out = attn_res

                if attn_out is None:
                    if attn_weights is None:
                        raise ValueError(
                            f"Attention module returned unsupported value: {type(attn_res)}"
                        )
                    attn_out = attn_weights @ A

                if attn_weights is not None:
                    self.recorder.record(f"{month_name}:{channel_name}", attn_weights)

                A = attn_out
                if self.normalize_features:
                    A = F.normalize(A, dim=1)

                ch_diag["timing_attention"] = time.time() - t_attn
                ch_diag["attention_variance"] = float(A.var().item())

            # Step 5: Optional PCA removal
            # IMPORTANT RULE:
            # - If compare_logits_vs_cli=True, we skip pipeline PCA entirely so the comparison is fair.
            # - If CLS channel is enabled, PCA is handled inside the extractor (CLS only), not here.
            F_out = A
            
            # NEW: Contract check for PCA removal
            pca_allowed = _check_operation_allowed(channel_rep_kind, 'pca_removal', warn_only=True)
            
            if self.compare_logits_vs_cli:
                diagnostics["steps"].append(f"pca_skipped_compare_mode:{channel_name}")
            elif self.use_cls_tokens:
                diagnostics["steps"].append(f"pca_skipped_cls_mode:{channel_name}")
            elif self.pca_remove and pca_allowed:
                t_pca = time.time()
                F_out = remove_top_pca_component(
                    F_out,
                    global_component=self.global_pca_component,
                )
                if self.normalize_features:
                    F_out = F.normalize(F_out, dim=1)
                ch_diag["timing_pca_removal"] = time.time() - t_pca
                ch_diag["final_variance"] = float(F_out.var().item())
                diagnostics["steps"].append(f"pca_removal:{channel_name}")
            elif self.pca_remove and not pca_allowed:
                diagnostics["steps"].append(f"pca_skipped_contract:{channel_name}")
                ch_diag["contract_violation_avoided"] = "pca_removal"
            else:
                ch_diag["final_variance"] = float(F_out.var().item())

            ch_diag["timing_total_channel"] = time.time() - t0
            
            # NEW: Update channel_extras with final provenance
            channel_extras["provenance"]["rep_kind"] = channel_rep_kind.value
            channel_extras["provenance"]["geometry_mode"] = self.geometry_mode
            channel_extras["provenance"]["final_dim"] = int(F_out.shape[-1]) if F_out is not None else 0
            
            return F_out, channel_extras

        # Run all channels
        out_by_channel: Dict[str, torch.Tensor] = {}
        extras_by_channel: Dict[str, Dict[str, Any]] = {}
        for ch, Xch in base_by_channel.items():
            out_by_channel[ch], extras_by_channel[ch] = _run_channel(ch, Xch)
        
        # NEW: Run Dirichlet fusion if enabled and we have CLS views
        # HOLOGRAPHIC LOOP (Axiom 3): whiten  project  sum (per paragraph)
        # Instead of ( wv), we compute  w (Wv)  isomorphic superposition.
        dirichlet_results = None
        if self.dirichlet_fusion is not None and self.use_cls_tokens:
            t_fusion = time.time()

            # Try holographic path first (cls_paragraphs available from Jagged Truth)
            has_paragraphs = any(
                pair.get("cls_paragraphs") is not None for pair in nli_pairs
            )

            if has_paragraphs:
                #  Holographic Loop: whiten each paragraph, project, weighted-sum 
                # Collect a sample for ZCA fitting (first pass)
                all_para_vecs = []
                for pair in nli_pairs:
                    cp = pair.get("cls_paragraphs")  # [n_paras, n_bots, hidden]
                    if cp is not None:
                        all_para_vecs.append(cp.reshape(-1, cp.shape[-1]))
                if all_para_vecs:
                    sample_for_zca = torch.cat(all_para_vecs, dim=0)  # [total, hidden]
                    # Subsample if too large (ZCA only needs ~5000)
                    if sample_for_zca.shape[0] > 5000:
                        idx = torch.randperm(sample_for_zca.shape[0])[:5000]
                        sample_for_zca = sample_for_zca[idx]
                    zca_fit = fit_whitening_matrix(sample_for_zca)
                    zca_mu = zca_fit['mu']
                    zca_W = zca_fit['whitening_matrix']
                else:
                    zca_mu, zca_W = None, None

                # Second pass: whiten  project(RKS via fusion)  weighted sum
                fused_per_article = []
                for pair in nli_pairs:
                    cp = pair.get("cls_paragraphs")   # [n_paras, n_bots, hidden]
                    pw = pair.get("paragraph_weights") # [n_paras]
                    if cp is None:
                        # Fallback: use pre-aggregated cls_per_bot
                        cpb = pair.get("cls_per_bot")
                        if cpb is not None:
                            fused_per_article.append(cpb)  # [n_bots, hidden]
                        continue

                    n_paras, n_bots, hidden = cp.shape
                    if pw is None:
                        pw = torch.ones(n_paras) / n_paras
                    pw = pw.to(cp.device)

                    # Whiten each paragraph's bot vectors
                    if zca_mu is not None and zca_W is not None:
                        cp_whitened = apply_whitening_fixed(
                            cp.reshape(n_paras * n_bots, hidden),
                            zca_mu, zca_W,
                        ).reshape(n_paras, n_bots, hidden)
                    else:
                        cp_whitened = cp

                    # Project each paragraph through Dirichlet fusion, then weighted-sum
                    para_fused = []
                    for p_idx in range(n_paras):
                        # [1, n_bots, hidden]  fusion  [1, rks_dim]
                        p_out = self.dirichlet_fusion(
                            cp_whitened[p_idx].unsqueeze(0),
                            compute_curvature=False,
                        )
                        para_fused.append(p_out["fused"].squeeze(0))  # [rks_dim]

                    # Weighted sum:  w  (Wv)
                    stacked = torch.stack(para_fused, dim=0)  # [n_paras, rks_dim]
                    holographic = torch.einsum('p,pd->d', pw, stacked)  # [rks_dim]
                    fused_per_article.append(holographic)

                if fused_per_article:
                    fused_tensor = torch.stack(fused_per_article, dim=0)  # [N, rks_dim]
                    # Run one final curvature computation on the aggregated result
                    # (use the pre-aggregated path for curvature stats)
                    cls_per_bot_list = [pair.get("cls_per_bot") for pair in nli_pairs if pair.get("cls_per_bot") is not None]
                    if cls_per_bot_list:
                        cls_per_bot = torch.stack(cls_per_bot_list, dim=0)
                        # L2-normalize before Dirichlet/RKS (kernel expects unit vectors)
                        if self.normalize_features:
                            cls_per_bot = F.normalize(cls_per_bot, dim=-1)
                        curv_out = self.dirichlet_fusion(cls_per_bot, compute_curvature=True)
                        curvature_stats = curv_out["curvature"]
                    else:
                        curvature_stats = None

                    # Compute per-article std: use std across feature dims as proxy
                    # for observer variance (since holographic path doesn't have explicit samples)
                    # Shape: [N, D] where each row has the article's feature spread
                    per_article_std = fused_tensor.std(dim=1, keepdim=True).expand_as(fused_tensor)

                    dirichlet_results = {
                        "fused": fused_tensor,
                        "fused_std": per_article_std,  # [N, D] - proper 2D shape
                        "curvature": {
                            "participation_ratio": curvature_stats["participation_ratio"].tolist() if curvature_stats else [],
                            "effective_rank_90": curvature_stats["effective_rank_90"].tolist() if curvature_stats else [],
                            "lambda1_lambda2": curvature_stats["lambda1_lambda2"].tolist() if curvature_stats else [],
                        } if curvature_stats else {},
                        "provenance": {
                            "path": "holographic_loop",
                            "zca_samples": int(sample_for_zca.shape[0]) if all_para_vecs else 0,
                        },
                    }
                    diagnostics["steps"].append("holographic_fusion")
                    diagnostics["dirichlet"] = {
                        "alpha": self.dirichlet_config.alpha if self.dirichlet_config else None,
                        "n_observers": self.dirichlet_config.n_observers if self.dirichlet_config else None,
                        "holographic": True,
                    }
            else:
                # Legacy fallback: pre-aggregated cls_per_bot (no paragraph data)
                cls_per_bot_list = [pair.get("cls_per_bot") for pair in nli_pairs if pair.get("cls_per_bot") is not None]
                if cls_per_bot_list:
                    cls_per_bot = torch.stack(cls_per_bot_list, dim=0)  # [N, 8, hidden]

                    # CRITICAL FIX: L2-normalize cls_per_bot before Dirichlet/RKS
                    # Raw DeBERTa embeddings have magnitude ~20-30, but RKS kernel expects unit vectors
                    # Without this, sigma=0.79 causes exp(-400/1.28)  0 (kernel collapse)
                    if self.normalize_features:
                        cls_per_bot = F.normalize(cls_per_bot, dim=-1)  # Normalize each [hidden] vector

                    # ASTER v3.2: Use Sequential Annealing if enabled
                    if hasattr(self.dirichlet_fusion, 'config') and getattr(self.dirichlet_fusion.config, 'use_annealing', False):
                        fusion_out = self.dirichlet_fusion.forward_annealing(cls_per_bot, compute_curvature=True)
                        dirichlet_results = {
                            "fused": fusion_out["fused"],
                            "fused_std": fusion_out["fused_std"],
                            "consensus_embedding": fusion_out.get("consensus_embedding"),
                            "stability_mask": fusion_out.get("stability_mask"),
                            "atmospheric_alpha": fusion_out.get("atmospheric_alpha"),
                            "adaptive_work": fusion_out.get("adaptive_work"),
                            "curvature": {
                                "participation_ratio": fusion_out["curvature"]["participation_ratio"].tolist(),
                                "effective_rank_90": fusion_out["curvature"]["effective_rank_90"].tolist(),
                                "lambda1_lambda2": fusion_out["curvature"]["lambda1_lambda2"].tolist(),
                            },
                            "provenance": fusion_out["provenance"],
                        }
                        diagnostics["steps"].append("dirichlet_annealing")
                    else:
                        fusion_out = self.dirichlet_fusion(cls_per_bot, compute_curvature=True)
                        dirichlet_results = {
                            "fused": fusion_out["fused"],
                            "fused_std": fusion_out["fused_std"],
                            "curvature": {
                                "participation_ratio": fusion_out["curvature"]["participation_ratio"].tolist(),
                                "effective_rank_90": fusion_out["curvature"]["effective_rank_90"].tolist(),
                                "lambda1_lambda2": fusion_out["curvature"]["lambda1_lambda2"].tolist(),
                            },
                            "provenance": fusion_out["provenance"],
                        }
                        diagnostics["steps"].append("dirichlet_fusion")

                    diagnostics["dirichlet"] = {
                        "alpha": self.dirichlet_config.alpha if self.dirichlet_config else None,
                        "n_observers": self.dirichlet_config.n_observers if self.dirichlet_config else None,
                        "curvature_mean_pr": float(fusion_out["curvature"]["participation_ratio"].mean().item()),
                        "annealing_enabled": getattr(self.dirichlet_fusion.config, 'use_annealing', False) if hasattr(self.dirichlet_fusion, 'config') else False,
                    }

            diagnostics["timing"]["dirichlet_fusion"] = time.time() - t_fusion

        # Choose the canonical "features" output for backward compatibility.
        if self.compare_logits_vs_cli:
            final_features = out_by_channel["logits"]
        else:
            final_features = out_by_channel["main"]

        diagnostics["variance"]["final_variance"] = float(final_features.var().item())
        diagnostics["timing"]["total"] = time.time() - start_total

        # Expose comparison outputs (if present)
        features_cli = out_by_channel.get("cli", None)

        # ====================================================================
        # WATERFALL CHECKPOINT SYSTEM (Data Lineage)
        # ====================================================================
        # Initialize checkpoint manager if output_dir is available
        waterfall_ckpt = None
        if config and config.get("enable_checkpoints", False):
            try:
                from .waterfall_checkpoints import WaterfallCheckpoint
                ckpt_dir = config.get("checkpoint_dir", config.get("output_dir", "."))
                waterfall_ckpt = WaterfallCheckpoint(ckpt_dir, run_id=month_name)

                # T1: Save semantic embeddings (pre-kernel)
                if final_features is not None:
                    embeddings_per_bot = None
                    if cls_per_bot_list:
                        embeddings_per_bot = torch.stack(cls_per_bot_list, dim=0).detach().cpu().numpy()
                    waterfall_ckpt.save_t1_embeddings(
                        embeddings=final_features.detach().cpu().numpy(),
                        embeddings_per_bot=embeddings_per_bot,
                        extraction_method="cls_token" if self.use_cls_tokens else "mean_pool",
                    )

                # T2: Save kernel projections
                if dirichlet_results is not None and "fused" in dirichlet_results:
                    fused = dirichlet_results["fused"]
                    # Safely get sigma from basis (may be SharedRKSBasis or MultiFramingRKS)
                    try:
                        if hasattr(self.dirichlet_fusion, 'basis'):
                            basis = self.dirichlet_fusion.basis
                            if hasattr(basis, 'sigma'):
                                rbf_sigma = float(basis.sigma) if not isinstance(basis.sigma, (list, tuple)) else basis.sigma[0]
                            elif hasattr(basis, 'sigmas'):
                                rbf_sigma = float(basis.sigmas[0]) if basis.sigmas else 1.0
                            elif hasattr(basis, 'kernels') and basis.kernels:
                                # MultiFramingRKS: get sigma from first kernel
                                first_kernel = basis.kernels[0]
                                rbf_sigma = float(getattr(first_kernel, 'sigma', 1.0))
                            else:
                                rbf_sigma = 1.0
                        else:
                            rbf_sigma = 1.0
                    except Exception:
                        rbf_sigma = 1.0

                    waterfall_ckpt.save_t2_kernel_projections(
                        z_rbf=fused.detach().cpu().numpy(),
                        rbf_sigma=rbf_sigma,
                        rks_dim=fused.shape[-1],
                        random_seed=self.random_seed,
                    )
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to initialize checkpoints: {e}")
                import traceback
                traceback.print_exc()
                waterfall_ckpt = None

        # ====================================================================
        # TRACK 5: PHASE-SPACE INTEGRATION (The Final Exit Gate)
        # ====================================================================
        # Fuses all 6 tracks into a single unit-sphere particle per article.
        # Only runs if we have the necessary track outputs.
        phase_space_results = None
        spectral_results = None  # Track 1.5 spectral polarity (populated below)
        social_texture_results = None  # Track 3 extended
        walker_work_integrals = []  # Track 4 work integrals
        walker_states = []  # Track 4 state classifications
        walker_state_records = []  # Track 4 persisted physics metadata
        if HAS_PHASE_SPACE and dirichlet_results is not None:
            t_phase = time.time()
            try:
                # Collect track outputs
                # Track 1: Logits - flatten to [N, 24] if needed
                logits_list = []
                for pair in nli_pairs:
                    lr = pair.get("logits_raw", pair.get("embedding"))
                    if lr is not None:
                        if lr.dim() == 2:  # [8, 3] -> [24]
                            lr = lr.flatten()
                        elif lr.dim() == 1 and lr.shape[0] != 24:
                            lr = lr[:24] if lr.shape[0] > 24 else torch.cat([lr, torch.zeros(24 - lr.shape[0])])
                        logits_list.append(lr)
                logits_t1 = torch.stack(logits_list, dim=0) if logits_list else None

                # T0: Save logit substrate checkpoint
                if waterfall_ckpt is not None and logits_t1 is not None:
                    try:
                        # Reshape to (N, 8, 3) for checkpoint format
                        logits_reshaped = logits_t1.view(-1, 8, 3).detach().cpu().numpy()
                        waterfall_ckpt.save_t0_substrate(logits_reshaped)
                    except Exception as e:
                        print(f"[Checkpoint T0] Warning: Failed to save substrate: {e}")

                # Track 1: Logit Confidence (max softmax probability per probe, averaged)
                logit_confidence_t1 = None
                if logits_t1 is not None:
                    try:
                        # Reshape to [N, 8, 3] (8 probes  3 NLI classes)
                        logits_reshaped = logits_t1.view(-1, 8, 3)
                        # Softmax over class dimension
                        probs = torch.softmax(logits_reshaped, dim=-1)  # [N, 8, 3]
                        # Max probability per probe (confidence = how sure is the model?)
                        max_probs = probs.max(dim=-1).values  # [N, 8]
                        # Average confidence across probes
                        logit_confidence_t1 = max_probs.mean(dim=-1)  # [N]
                        print(f"[Track 1] Logit confidence: mean={logit_confidence_t1.mean():.3f}")
                    except Exception as e:
                        print(f"[Track 1] Logit confidence computation failed: {e}")

                # Track 2: Hologram (fused embeddings after ZCA)
                hologram_t2 = dirichlet_results.get("fused")

                # Track 3: Blinker (Dirichlet variance)
                blinker_t3 = dirichlet_results.get("fused_std")
                if isinstance(blinker_t3, np.ndarray):
                    blinker_t3 = torch.from_numpy(blinker_t3)

                # Track 3 Extended: Social Texture (alpha sweep crack/bond topology)
                # This is computationally expensive, so optional via config
                run_social_texture = config.get("run_social_texture", False) if config else False
                if run_social_texture and cls_per_bot_list and hasattr(self, 'dirichlet_fusion'):
                    try:
                        from .dirichlet_fusion import AtmosphericAnnealer
                        cls_stacked = torch.stack(cls_per_bot_list, dim=0)  # [N, 8, H]
                        annealer = AtmosphericAnnealer(
                            fusion=self.dirichlet_fusion,
                            alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                            n_observers_per_alpha=30,  # Reduced for speed
                            device=str(self.device),
                        )
                        social_texture_results = annealer.analyze(
                            cls_per_bot=cls_stacked,
                            corpus_name=month_name,
                            article_ids=bt_uids,
                            top_k=20,
                            store_full_matrices=False,  # Save memory
                            verbose=True,
                        )
                        print(f"[Track 3] Social texture: {social_texture_results.n_cracks} cracks, "
                              f"{social_texture_results.n_bonds} bonds, "
                              f"{social_texture_results.n_contested} contested "
                              f"(crack_frac={social_texture_results.crack_fraction:.3f})")
                    except Exception as e:
                        import traceback
                        print(f"[Track 3] Social texture failed: {e}")
                        traceback.print_exc()

                # Track 1.5: Spectral Polarity (PCA compass on gradient matrix)
                antagonism_t15 = None
                if cls_per_bot_list and hasattr(self, 'dirichlet_fusion') and self.dirichlet_fusion is not None:
                    try:
                        from .spectral_polarity import SpectralPolarity, SpectralPolarityConfig
                        sp = SpectralPolarity(SpectralPolarityConfig(multi_scale_enabled=True))
                        spectral_results = sp.compute_batch(cls_per_bot_list)
                        # Project directional antagonism through shared RKS basis
                        if hasattr(self.dirichlet_fusion, 'basis') and self.dirichlet_fusion.basis is not None:
                            antagonism_t15 = self.dirichlet_fusion.basis(spectral_results.antagonism)  # [N, D]
                            n_valid = spectral_results.dipole_valid.sum().item()
                            n_total = len(cls_per_bot_list)
                            mean_persist = spectral_results.n_persistent_scales.float().mean().item()
                            print(f"[Track 1.5] Spectral polarity ({spectral_results.dipole_state}): "
                                  f"{antagonism_t15.shape}, EVR={spectral_results.evr.mean():.3f}, "
                                  f"persist={mean_persist:.1f} scales, valid={n_valid}/{n_total}")

                            # T1.5: Save spectral state checkpoint
                            if waterfall_ckpt is not None:
                                waterfall_ckpt.save_t15_spectral(
                                    u_axis=spectral_results.u_axis[0].detach().cpu().numpy(),  # Use first article's u_axis as representative
                                    evr=float(spectral_results.evr.mean().item()),
                                    singular_values=spectral_results.singular_values[0].detach().cpu().numpy() if hasattr(spectral_results, 'singular_values') else np.zeros(8),
                                    probe_magnitudes=spectral_results.probe_magnitudes.detach().cpu().numpy(),
                                    dipole_valid=spectral_results.dipole_valid.detach().cpu().numpy(),
                                    n_persistent_scales=int(spectral_results.n_persistent_scales.float().mean().item()),
                                )
                    except Exception as e:
                        import traceback
                        print(f"[Track 1.5] Spectral polarity failed: {e}")
                        traceback.print_exc()

                # Track 3 Gate: Skip walker on Fog regions (high variance)
                # Use percentile-based fog detection: only top 10% variance = FOG
                FOG_PERCENTILE = 0.90  # Only top 10% get FOG
                fog_mask = torch.zeros(len(cls_per_bot_list), dtype=torch.bool) if cls_per_bot_list else torch.zeros(0, dtype=torch.bool)
                if blinker_t3 is not None and len(cls_per_bot_list) > 0:
                    blinker_magnitude = blinker_t3.norm(dim=-1)
                    fog_threshold = torch.quantile(blinker_magnitude, FOG_PERCENTILE).clamp(min=1e-9)
                    fog_mask = blinker_magnitude > fog_threshold

                # Track 4: Walker (MCMC exploration resistance with u_axis)
                walker_t4 = None
                if cls_per_bot_list and hasattr(self, 'dirichlet_fusion') and self.dirichlet_fusion is not None:
                    try:
                        from .physarum_walk import compute_walker_resistance
                        cls_stacked = torch.stack(cls_per_bot_list, dim=0)  # [N, 8, hidden]
                        walker_outputs = []
                        rks_dim = self.dirichlet_fusion.basis.output_dim if hasattr(self.dirichlet_fusion.basis, 'output_dim') else 2048
                        device = cls_stacked.device

                        # Get u_axis from spectral results if available
                        u_axes = None
                        if spectral_results is not None:
                            u_axes = spectral_results.u_axis  # [N, H]

                        for i in range(len(cls_stacked)): # Iterate through all articles
                            bot_embs = cls_stacked[i]  # [8, hidden]
                            u_axis_i = u_axes[i] if u_axes is not None else None

                            result_t4 = compute_walker_resistance(
                                cls_per_bot=bot_embs,
                                rks_basis=self.dirichlet_fusion.basis,
                                u_axis=u_axis_i,
                                temperature=0.5,
                                n_walkers=20,
                                n_steps=10,
                            )
                            walker_outputs.append(result_t4["walker_output"])  # [D]
                            walker_work_integrals.append(result_t4["work_integral"])
                            walker_states.append(result_t4["state"]) # State name (string)
                            walker_energy_budget = 50.0
                            if hasattr(self, "thermo_config") and getattr(self, "thermo_config", None) is not None:
                                walker_energy_budget = float(getattr(self.thermo_config, "walker_energy_budget", 50.0))
                            walker_state_records.append({
                                "label": result_t4.get("state", "unknown"),
                                "status": result_t4.get("status", "SUCCESS"),
                                "energy_rem": float(walker_energy_budget - float(result_t4["work_integral"])),
                                "steps": 10,
                                "terrain_state": None,
                            })

                        if walker_outputs:
                            walker_t4 = torch.stack(walker_outputs, dim=0)  # [N, D]

                            # Count states (new 4-state system)
                            state_counts = {
                                "tautology": 0,
                                "honest": 0,
                                "phantom": 0,
                                "rupture": 0,
                                "broken": 0,
                                "trapped": 0,
                            }
                            # walker_states is a list of strings like "tautology"
                            for s in walker_states:
                                if s in state_counts: # Defensive check
                                    state_counts[s] += 1
                                else:
                                    print(f"WARNING: Unknown walker state '{s}' encountered.")
                            
                            mean_work = np.mean(walker_work_integrals) # Calculate mean from the list
                            print(f"[Track 4] Walker: W={mean_work:.3f}, "
                                  f"states={{T:{state_counts['tautology']}, H:{state_counts['honest']}, P:{state_counts['phantom']}, R:{state_counts['rupture']}, B:{state_counts['broken']}, Tr:{state_counts['trapped']}}}")
                            print(f"[Track 4] Walker: W={mean_work:.3f}, "
                                  f"states={{T:{state_counts['tautology']}, H:{state_counts['honest']}, P:{state_counts['phantom']}, R:{state_counts['rupture']}, B:{state_counts['broken']}, Tr:{state_counts['trapped']}}}")
                    except Exception as e:
                        import traceback
                        print(f"[Track 4] Walker computation failed: {e}")
                        traceback.print_exc()

                # Compute WEIGHTED spectral distance for Phantom Differential (ASTER v3.2)
                # Uses signal-weighted subspace instead of flat Euclidean
                d_spectral = None
                if spectral_results is not None and cls_per_bot_list:
                    from .spectral_polarity import SpectralPolarity, SpectralPolarityConfig
                    sp = SpectralPolarity(SpectralPolarityConfig())

                    # Stack embeddings to get pole embeddings for weighted distance
                    G = torch.stack(cls_per_bot_list, dim=0)  # [N, 8, H]
                    # Use probe extremes as pole embeddings (max/min projection articles)
                    # For each article, emb_start = mean of positive probes, emb_end = mean of negative probes
                    probe_mags = spectral_results.probe_magnitudes  # [N, 8]
                    pos_mask = probe_mags > 0  # [N, 8]
                    neg_mask = probe_mags <= 0  # [N, 8]

                    # Compute weighted mean of positive/negative probe embeddings
                    # Shape: [N, 8, H] * [N, 8, 1] -> [N, H]
                    pos_weights = (probe_mags * pos_mask.float()).unsqueeze(-1)  # [N, 8, 1]
                    neg_weights = (-probe_mags * neg_mask.float()).unsqueeze(-1)  # [N, 8, 1]

                    pos_sum = pos_weights.sum(dim=1).clamp(min=1e-9)  # [N, 1]
                    neg_sum = neg_weights.sum(dim=1).clamp(min=1e-9)  # [N, 1]

                    emb_pos = (G * pos_weights).sum(dim=1) / pos_sum  # [N, H]
                    emb_neg = (G * neg_weights).sum(dim=1) / neg_sum  # [N, H]

                    # Compute weighted spectral distance using dynamic K
                    d_spectral = sp.compute_weighted_spectral_distance(
                        emb_a=emb_pos,
                        emb_b=emb_neg,
                        singular_values=spectral_results.singular_values,
                        Vt=spectral_results.u_basis,
                        dynamic_k=spectral_results.dynamic_k,
                    )
                    
                    # 3+2+1 OVERHAUL: Compute WHITENED distance and PHANTOM RATIO
                    d_whitened = sp.compute_whitened_spectral_distance(
                        emb_a=emb_pos,
                        emb_b=emb_neg,
                        singular_values=spectral_results.singular_values,
                        Vt=spectral_results.u_basis,
                        dynamic_k=spectral_results.dynamic_k,
                    )
                    phantom_ratio = d_whitened / d_spectral.clamp(min=1e-9)

                    # Log dynamic K statistics
                    mean_k = spectral_results.dynamic_k.float().mean().item()
                    print(f"[Track 1.5] Weighted spectral distance: mean_d={d_spectral.mean():.3f}, "
                          f"dynamic_K={mean_k:.1f}")

                # Compute Phantom Differential verdicts (Panic Function)
                phantom_verdicts = []
                if d_spectral is not None and walker_work_integrals:
                    from .phase_space_integrator import PhaseSpaceIntegrator, IntegratorConfig
                    # Default behavior: bypass panic-function verdict override and preserve
                    # native Track 4 walker states as Track 5 verdicts.
                    use_panic_differential = os.environ.get("BT_USE_PANIC_DIFFERENTIAL", "0").strip() == "1"
                    if use_panic_differential:
                        if not hasattr(self, '_phase_integrator') or self._phase_integrator is None:
                            self._phase_integrator = PhaseSpaceIntegrator(IntegratorConfig())

                        # Compute blinker variance per article for terrain classification
                        blinker_variance = None
                        if blinker_t3 is not None:
                            # blinker_t3 is [N, D] fused_std - compute magnitude per article
                            blinker_variance = blinker_t3.norm(dim=-1).tolist()  # List[float]

                        phantom_verdicts = self._phase_integrator.compute_phantom_differential(
                            d_spectral, walker_work_integrals, walker_states,
                            blinker_variance=blinker_variance
                        )
                        # Log verdict summary with TAUTOLOGY
                        verdict_counts = {"TAUTOLOGY": 0, "HONEST": 0, "PHANTOM": 0, "RUPTURE": 0, "BROKEN": 0, "TRAPPED": 0}
                        for v in phantom_verdicts:
                            verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1
                        print(f"[Panic Function] Verdicts: T={verdict_counts['TAUTOLOGY']}, "
                              f"H={verdict_counts['HONEST']}, P={verdict_counts['PHANTOM']}, "
                              f"R={verdict_counts['RUPTURE']}")
                    else:
                        EPS = 1e-9
                        work_arr = np.array(walker_work_integrals, dtype=float)
                        d_arr = np.array([float(d_spectral[i]) if i < len(d_spectral) else 0.0
                                          for i in range(len(walker_work_integrals))], dtype=float)
                        delta_arr = work_arr / np.maximum(d_arr, EPS)
                        rupture_work_cutoff = float(np.quantile(work_arr, 0.90))
                        honest_delta_cutoff = float(np.quantile(delta_arr, 0.80))
                        tautology_delta_cutoff = float(np.quantile(delta_arr, 0.10))
                        for i, (w, s) in enumerate(zip(walker_work_integrals, walker_states)):
                            ratio = float(phantom_ratio[i]) if i < len(phantom_ratio) else 1.0
                            
                            # 3+2+1 OVERHAUL: Family B (Topological Breaks) have priority
                            if s == "broken":
                                verdict = "RUPTURE" # Map to Rupture for HUD, UI handles Laser
                            elif s == "trapped":
                                verdict = "TAUTOLOGY" # Map to Tautology for HUD, UI handles Stall
                            # Family A (Spectral Paths) based on Ratio
                            elif ratio > 2.0: # ratio >> 1
                                verdict = "PHANTOM"
                            elif ratio < 0.5: # ratio << 1
                                verdict = "TAUTOLOGY"
                            else: # ratio approx 1
                                verdict = "HONEST"
                            phantom_verdicts.append({
                                "verdict": verdict,
                                "delta": delta,
                                "d_spectral": d,
                                "w_actual": float(w),
                                "confidence": 1.0,
                                "walker_state": s,
                            })
                        verdict_counts = {"TAUTOLOGY": 0, "HONEST": 0, "PHANTOM": 0, "RUPTURE": 0, "BROKEN": 0, "TRAPPED": 0}
                        for v in phantom_verdicts:
                            verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1
                        print(f"[Track 5] Using Track 4 states (panic bypass): "
                              f"T={verdict_counts['TAUTOLOGY']}, H={verdict_counts['HONEST']}, "
                              f"P={verdict_counts['PHANTOM']}, R={verdict_counts['RUPTURE']}, "
                              f"B={verdict_counts['BROKEN']}, Tr={verdict_counts['TRAPPED']}")

                # Bind per-article terrain class to walker state records for waterfall persistence.
                if walker_state_records and phantom_verdicts and len(walker_state_records) == len(phantom_verdicts):
                    for i, rec in enumerate(walker_state_records):
                        rec["terrain_state"] = phantom_verdicts[i].get("terrain_state")

                # Create or use cached integrator
                # ASTER v3.2: Also refit if is_fit=False (may have failed previously)
                needs_fit = (
                    not hasattr(self, '_phase_integrator') or
                    self._phase_integrator is None or
                    not self._phase_integrator.is_fit
                )
                if needs_fit:
                    integrator_config = IntegratorConfig()
                    self._phase_integrator = PhaseSpaceIntegrator(integrator_config)

                    # Fit on current batch (in production, fit on 5k sample corpus)
                    track_samples = {}
                    if logits_t1 is not None:
                        track_samples['logits'] = logits_t1
                    if hologram_t2 is not None:
                        track_samples['hologram'] = hologram_t2
                    if blinker_t3 is not None:
                        track_samples['blinker'] = blinker_t3

                    if track_samples:
                        self._phase_integrator.fit(track_samples)

                # Integrate particles (ASTER v3.2: Hadamard fusion enabled)
                print(f"[Track 5] Integrator is_fit={self._phase_integrator.is_fit}, "
                      f"hologram={hologram_t2 is not None}, antagonism={antagonism_t15 is not None}, "
                      f"blinker={blinker_t3 is not None}")
                if self._phase_integrator.is_fit:
                    particles = self._phase_integrator.integrate(
                        logits=logits_t1,
                        antagonism=antagonism_t15,
                        hologram=hologram_t2,
                        blinker=blinker_t3,
                        walker=walker_t4,
                        article_ids=bt_uids,
                        use_hadamard_fusion=True,  # ASTER v3.2: K_final = K_rks * K_spectral
                    )

                    if particles:
                        # Attach phantom verdicts to particles if available
                        if phantom_verdicts and len(phantom_verdicts) == len(particles):
                            for i, p in enumerate(particles):
                                v = phantom_verdicts[i]
                                p.phantom_verdict = v["verdict"]
                                p.phantom_delta = v["delta"]
                                p.d_spectral = v.get("d_spectral", v.get("d", 0.0))
                                p.w_actual = v.get("w_actual", v.get("w", 0.0))
                                p.phantom_confidence = v.get("confidence", 0.0)
                                p.walker_state = v.get("walker_state", "unknown")

                        # Stack integrated vectors
                        integrated_vectors = torch.stack([p.vector for p in particles], dim=0)

                        # Compute phantom verdict counts (ASTER v3.2: includes TAUTOLOGY)
                        phantom_counts = {
                            "TAUTOLOGY": sum(1 for p in particles if p.phantom_verdict == "TAUTOLOGY"),
                            "HONEST": sum(1 for p in particles if p.phantom_verdict == "HONEST"),
                            "PHANTOM": sum(1 for p in particles if p.phantom_verdict == "PHANTOM"),
                            "RUPTURE": sum(1 for p in particles if p.phantom_verdict == "RUPTURE"),
                        }

                        phase_space_results = {
                            "integrated_vectors": integrated_vectors,
                            "particles": particles,
                            "singularity_counts": {
                                "structural_singularity": sum(1 for p in particles if p.singularity_type == "structural_singularity"),
                                "noise": sum(1 for p in particles if p.singularity_type == "noise"),
                                "ideological_barrier": sum(1 for p in particles if p.singularity_type == "ideological_barrier"),
                                "consensus": sum(1 for p in particles if p.singularity_type == "consensus"),
                            },
                            "phantom_counts": phantom_counts,
                            "mean_liar_score": float(np.mean([p.liar_score for p in particles])),
                            "n_liars": sum(1 for p in particles if p.liar_score > 0.5),
                        }
                        diagnostics["steps"].append("phase_space_integration")
                        diagnostics["phase_space"] = {
                            "n_particles": len(particles),
                            "integrated_dim": int(integrated_vectors.shape[-1]),
                            "singularity_counts": phase_space_results["singularity_counts"],
                            "phantom_counts": phantom_counts,
                        }

                        # ASTER v3.2: Hadamard fusion diagnostics
                        if particles and particles[0].hadamard_diagnostics is not None:
                            diagnostics["phase_space"]["hadamard_fusion"] = particles[0].hadamard_diagnostics

            except Exception as e:
                diagnostics.setdefault("warnings", []).append(f"phase_space_error: {e}")

            diagnostics["timing"]["phase_space"] = time.time() - t_phase

        # Provenance log (optional)
        provenance_entries = []
        if self.enable_provenance and self.provenance_tracker is not None:
            # Record the "canonical" channel as the primary artifact.
            try:
                prov_entry = ProvenanceEntry(
                    month=month_name,
                    seed=self.random_seed,
                    mode=self.geometry_mode, # [FIX] Use geometry_mode, not mode
                    embedding_type=("logits" if self.compare_logits_vs_cli else "main"),
                    n_articles=int(final_features.shape[0]),
                    dim=int(final_features.shape[1]),
                    metadata=provenance_metadata,
                    timestamp=time.time(),
                )
                self.provenance_tracker.add_entry(prov_entry)
                provenance_entries = [prov_entry.to_dict()]
            except Exception as e:
                # Don't crash runs due to provenance bookkeeping
                diagnostics.setdefault("warnings", []).append(f"provenance_error: {e}")

        # Article metadata (index-stable)
        metadata = []
        for i, art in enumerate(articles):
            meta = {
                "index": i,
                "bt_uid": bt_uids[i] if bt_uids is not None else None,
                "published_at": art.get("published_at", None) if isinstance(art, dict) else None,
                "source": art.get("source", None) if isinstance(art, dict) else None,
                "title": art.get("title", None) if isinstance(art, dict) else None,
            }
            metadata.append(meta)

        out = {
            "month": month_name,
            "features": final_features.detach().cpu().numpy(),
            # Compatibility alias (older runners saved result['embeddings'])
            "embeddings": final_features.detach().cpu().numpy(),
            "diagnostics": diagnostics,
            "provenance": provenance_entries,
            "article_metadata": metadata,
            "timeline": timeline,
            "bt_uid_list": bt_uids,
            # NEW: Channel extras with rep_kind and provenance
            "channel_extras": extras_by_channel,
            # NEW: Primary rep_kind for downstream contract checks
            "rep_kind": self.primary_rep_kind.value,
        }

        if features_cli is not None:
            out["features_cli"] = features_cli.detach().cpu().numpy()
            out["embeddings_cli"] = features_cli.detach().cpu().numpy()
        
        # NEW: Include Dirichlet fusion results if computed
        if dirichlet_results is not None:
            out["dirichlet_fused"] = dirichlet_results["fused"].detach().cpu().numpy()
            out["dirichlet_fused_std"] = dirichlet_results["fused_std"].detach().cpu().numpy()
            out["dirichlet_curvature"] = dirichlet_results["curvature"]
            out["dirichlet_provenance"] = dirichlet_results["provenance"]

        # NEW: Include Phase-Space Integration results (Track 5)
        if phase_space_results is not None:
            out["integrated_vectors"] = phase_space_results["integrated_vectors"].detach().cpu().numpy()
            out["singularity_counts"] = phase_space_results["singularity_counts"]
            out["phantom_counts"] = phase_space_results.get("phantom_counts", {})
            out["mean_liar_score"] = phase_space_results["mean_liar_score"]
            out["n_liars"] = phase_space_results["n_liars"]
            # Serialize particles for downstream use
            out["particles"] = [p.to_dict() for p in phase_space_results["particles"]]

        # NEW: Include Track 1 Logit Confidence
        if logit_confidence_t1 is not None:
            out["logit_confidence"] = logit_confidence_t1.detach().cpu().numpy()

        # NEW: Include Spectral Polarity results (Track 1.5 compass)
        if spectral_results is not None:
            out["spectral_evr"] = spectral_results.evr.detach().cpu().numpy()
            out["spectral_probe_magnitudes"] = spectral_results.probe_magnitudes.detach().cpu().numpy()
            out["spectral_dipole_valid"] = spectral_results.dipole_valid.detach().cpu().numpy()
            out["spectral_n_persistent_scales"] = spectral_results.n_persistent_scales.detach().cpu().numpy()
            out["spectral_evr_per_scale"] = spectral_results.evr_per_scale.detach().cpu().numpy()
            out["spectral_dipole_state"] = spectral_results.dipole_state
            out["spectral_u_axis"] = spectral_results.u_axis.detach().cpu().numpy()

        # NEW: Include Walker (Track 4) work integrals and states
        if walker_work_integrals:
            out["walker_work_integrals"] = np.array(walker_work_integrals)
            # Flight-data-recorder schema: keep both semantic label and physics status.
            out["walker_states"] = walker_state_records if walker_state_records else walker_states

        # NEW: Include Phantom Differential verdicts (ASTER v3.2)
        if phantom_verdicts:
            out["phantom_verdicts"] = phantom_verdicts
        if d_spectral is not None:
            out["d_spectral"] = d_spectral.detach().cpu().numpy()

        # NEW: Include Social Texture (Track 3 extended) crack/bond topology
        if social_texture_results is not None:
            out["social_texture"] = {
                "n_cracks": social_texture_results.n_cracks,
                "n_bonds": social_texture_results.n_bonds,
                "n_contested": social_texture_results.n_contested,
                "crack_fraction": social_texture_results.crack_fraction,
                "bond_fraction": social_texture_results.bond_fraction,
                "mean_crack_score": social_texture_results.mean_crack_score,
                "mean_bond_score": social_texture_results.mean_bond_score,
                "alphas_tested": social_texture_results.alphas_tested,
            }

        # T3: Save topology checkpoint (bonds, cracks, ruptures)
        if waterfall_ckpt is not None:
            try:
                # Compute bond/crack matrices from Dirichlet results if available
                n_articles = final_features.shape[0]
                bond_matrix = np.eye(n_articles)  # Default: self-bonds only
                crack_matrix = np.zeros((n_articles, n_articles))
                rupture_pairs = []

                # Extract from social texture if available
                if social_texture_results is not None:
                    # Use social texture crack/bond info
                    pass  # social_texture_results has aggregated stats, not full matrices

                # Extract from walker states for ruptures
                if walker_states:
                    for i, state in enumerate(walker_states):
                        if state == "broken":
                            # Mark as self-rupture (W=)
                            rupture_pairs.append((i, i))

                nmi_score = 0.0
                ari_score = 0.0

                waterfall_ckpt.save_t3_topology(
                    bond_matrix=bond_matrix,
                    crack_matrix=crack_matrix,
                    rupture_pairs=rupture_pairs,
                    dirichlet_fused=dirichlet_results["fused"].detach().cpu().numpy() if dirichlet_results else None,
                    walker_work=np.array(walker_work_integrals) if walker_work_integrals else None,
                    nmi_score=nmi_score,
                    ari_score=ari_score,
                )
            except Exception as e:
                print(f"[Checkpoint T3] Warning: Failed to save topology: {e}")

        # ====================================================================
        # TRACK 6: HOTT SIDECAR (The Proof Engine)
        # ====================================================================
        # Converts geometric features into formal HoTT proofs.
        # Vertical constraint: read-only, no feedback to lower tracks.
        run_hott_proofs = True  # Always run HoTT proofs when spectral data is available
        if run_hott_proofs and 'spectral_evr' in out:
            try:
                from .hott_sidecar import prove_from_pipeline_results
                hott_proofs, hott_summary = prove_from_pipeline_results(out)
                out["hott_proofs"] = [p.to_dict() for p in hott_proofs]
                out["hott_summary"] = hott_summary

                # Print summary
                print(f"[Track 6] HoTT Sidecar: {hott_summary['n_proofs']} proofs generated")
                print(f"  Equivalence: {hott_summary['equivalence_rate']*100:.1f}%, "
                      f"Non-equiv: {hott_summary['non_equivalence_rate']*100:.1f}%, "
                      f"Obstruction: {hott_summary['obstruction_rate']*100:.1f}%, "
                      f"Undecidable: {hott_summary['undecidable_rate']*100:.1f}%")
                print(f"  Topological ruptures: {hott_summary['topological_ruptures']}, "
                      f"Energetic barriers: {hott_summary['energetic_barriers']}")
                print(f"  Mean proof confidence: {hott_summary['mean_confidence']:.3f}")
            except Exception as e:
                import traceback
                print(f"[Track 6] HoTT Sidecar failed: {e}")
                traceback.print_exc()

        # Finalize waterfall checkpoint manifest
        if waterfall_ckpt is not None:
            try:
                waterfall_ckpt.save_manifest()
                out["waterfall_checkpoint_dir"] = str(waterfall_ckpt.checkpoint_dir)
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to save manifest: {e}")

        return out


def run_multi_observer_experiment(
    corpus_name: str,
    months_data: Dict[str, List[Dict]],
    output_dir: str,
    seeds: List[int] = [42, 43, 44, 45, 46],
    kernels: List[str] = ["rbf", "laplacian", "rq", "imq"],
    device: str = "cuda",
    record_attention: bool = True,
    config: Optional[Dict] = None,
):
    """
    Run a full multi-observer experiment suite.

    For each kernel and each observer seed:
      - initialize pipeline
      - run `process_month()` for each month
      - save outputs in a structured directory

    This function is intentionally "outer-loop glue"  it doesnt decide your science;
    it ensures outputs are organized and reproducible.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for kernel in kernels:
        print(f"\n=== Running kernel: {kernel} ===")
        all_results[kernel] = {}

        for seed in seeds:
            print(f"  -> Observer seed: {seed}")
            run_dir = os.path.join(output_dir, corpus_name, kernel, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            # Allow config dict to override pipeline knobs WITHOUT adding CLI sprawl.
            cfg = config or {}
            import inspect as _inspect
            sig = _inspect.signature(initialize_full_pipeline)
            filtered_cfg = {k: v for k, v in cfg.items() if k in sig.parameters}
            components = initialize_full_pipeline(
                random_seed=seed,
                device=device,
                **filtered_cfg,
            )

            # Set kernel type (RKS)
            if components["rks_map"] is not None:
                components["rks_map"].rebuild(kernel_type=kernel)

            provenance_dir = os.path.join(run_dir, "provenance")
            pipeline = BeliefTransformerPipeline(
                components=components,
                random_seed=seed,
                enable_provenance=True,
                provenance_dir=provenance_dir,
            )

            run_results = {}
            for month_name, articles in months_data.items():
                month_out = pipeline.process_month(articles, month_name=month_name)

                # Persist artifacts
                month_path = os.path.join(run_dir, f"{month_name}.npz")
                np.savez_compressed(
                    month_path,
                    features=month_out["features"],
                    **({"features_cli": month_out["features_cli"]} if "features_cli" in month_out else {}),
                )

                # Metadata sidecar (includes timestamps + bt_uid)
                meta_path = os.path.join(run_dir, f"{month_name}_metadata.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "month": month_name,
                            "corpus": corpus_name,
                            "kernel": kernel,
                            "seed": seed,
                            "diagnostics": month_out.get("diagnostics", {}),
                            "article_metadata": month_out.get("article_metadata", []),
                            "timeline": month_out.get("timeline", []),
                            "bt_uid_list": month_out.get("bt_uid_list", []),
                            "provenance": month_out.get("provenance", []),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                run_results[month_name] = {
                    "features_path": month_path,
                    "metadata_path": meta_path,
                    "n_articles": len(articles),
                }

            # Save attention if requested
            if record_attention and components.get("recorder") is not None:
                attn_out_path = os.path.join(run_dir, "attention_records.json")
                with open(attn_out_path, "w", encoding="utf-8") as f:
                    json.dump(components["recorder"].to_dict(), f, indent=2)

            all_results[kernel][seed] = run_results

    # Save summary index
    index_path = os.path.join(output_dir, corpus_name, "index.json")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ============================================================================
# COMPATIBILITY WRAPPER FOR run_experiments.py
# ============================================================================

def run_multi_observer_experiment_simple(
    articles: list,
    seeds: list,
    use_contrastive: bool = True,
    use_pca_removal: bool = False,
    use_cls_tokens: bool = False,
    shared_pca: bool = False,
    kernel_type: str = 'rbf',
    kernel_types = None,
    kernel_params = None,
    use_gru: bool = True,
    use_multi_framing_rks: bool = True,
    device: str = 'cuda',
    track_variance: bool = True,
    output_dir = Path('outputs'),
    rks_sigma = None,
    corpus_name: str = 'unknown',
    nli_cache_path: str = None,
    **kwargs
):
    """
    Compatibility wrapper for run_experiments.py
    
    Now saves complete metadata for downstream analysis:
    - bt_uid_list: stable article IDs for cross-observer alignment
    - article_metadata: per-article provenance info
    - meta: experiment configuration (kernel, channel, seed, corpus)
    
    Optimization: If nli_cache_path is provided, NLI embeddings are cached to disk
    and reused across different kernel types (4x speedup for full matrix).
    """
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine channel from config
    channel = 'cls' if use_cls_tokens else 'logits'
    
    # Load or create shared NLI cache for all seeds (since NLI doesn't depend on seed)
    shared_nli_cache = None
    if nli_cache_path:
        nli_cache_file = Path(nli_cache_path)
        if nli_cache_file.exists():
            try:
                cached_data = torch.load(nli_cache_file, map_location='cpu', weights_only=False)
                if cached_data.get('n_articles') == len(articles) and cached_data.get('channel') == channel:
                    shared_nli_cache = cached_data['nli_pairs']
                    print(f"  [CACHE HIT] Loaded NLI embeddings from {nli_cache_file}")
                else:
                    print(f"  [CACHE MISS] Cache mismatch (articles: {cached_data.get('n_articles')} vs {len(articles)}, channel: {cached_data.get('channel')} vs {channel})")
            except Exception as e:
                print(f"  [CACHE ERROR] Could not load {nli_cache_file}: {e}")
    
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Observer {seed} | kernel={kernel_type} | channel={channel}")
        print(f"{'='*70}")
        
        components = initialize_full_pipeline(
            random_seed=seed,
            device=device,
            use_contrastive=use_contrastive,
            use_cls_tokens=use_cls_tokens,
            pca_remove=use_pca_removal,
            global_pca_component=shared_pca,
            use_gru=use_gru,
            use_rks=use_multi_framing_rks,
            # Increase dimensionality by default (can be overridden via kwargs)
            output_dim=int(kwargs.get("output_dim", 2048)),
            rks_output_dim=kwargs.get("rks_output_dim", None),
            geometry_mode=kwargs.get("geometry_mode", "rks"),
            compare_logits_vs_cli=bool(kwargs.get("compare_logits_vs_cli", False)),
            adult_kernel=kwargs.get("adult_kernel", "rbf"),
            adult_gamma=float(kwargs.get("adult_gamma", 1.0)),
            adult_sigma=kwargs.get("adult_sigma", None),
            adult_center=bool(kwargs.get("adult_center", True)),
            adult_nystrom_m=int(kwargs.get("adult_nystrom_m", 256)),
            embedding_dim=24 if not use_cls_tokens else 8192,
        )
        
        pipeline = BeliefTransformerPipeline(
            components=components,
            random_seed=seed,
        )
        
        # Inject shared NLI cache if available (avoids re-running DeBERTa)
        if shared_nli_cache is not None:
            pipeline._nli_cache = shared_nli_cache
        
        result = pipeline.process_month(articles, month_name="batch")
        
        # Save NLI cache after first extraction (for reuse across kernels)
        if shared_nli_cache is None and nli_cache_path and pipeline._nli_cache is not None:
            nli_cache_file = Path(nli_cache_path)
            nli_cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save({
                    'nli_pairs': pipeline._nli_cache,
                    'n_articles': len(articles),
                    'channel': channel,
                }, nli_cache_file)
                shared_nli_cache = pipeline._nli_cache  # Use for subsequent seeds
                print(f"  [CACHE SAVED] NLI embeddings to {nli_cache_file}")
            except Exception as e:
                print(f"  [CACHE WARN] Could not save cache: {e}")
        
        # Build comprehensive output artifact with FULL INFORMATION PRESERVATION
        
        # Compute variance tracking for this observer
        variance_tracker = VarianceTracker()
        if result.get('embeddings') is not None:
            variance_tracker.record_stage('final_embeddings', result['embeddings'])
        if result.get('features') is not None:
            variance_tracker.record_stage('final_features', result['features'])
        if result.get('embeddings_cli') is not None:
            variance_tracker.record_stage('logits_embeddings', result['embeddings_cli'])
        
        # Get git info for reproducibility
        git_hash = get_git_hash()
        git_dirty = get_git_dirty()
        timestamp = datetime.datetime.now().isoformat()
        
        output_artifact = {
            # Primary features (backward compatible)
            'embeddings': result['embeddings'],
            'features': result.get('features'),
            'embeddings_cli': result.get('embeddings_cli'),
            'features_cli': result.get('features_cli'),
            
            # Legacy fields (backward compatible)
            'seed': seed,
            'n_articles': len(articles),
            
            # Stable IDs for cross-observer alignment (CRITICAL)
            'ids': result.get('bt_uid_list', []),
            'bt_uid_list': result.get('bt_uid_list', []),
            
            # Per-article metadata for provenance
            'article_metadata': result.get('article_metadata', []),
            
            # Comprehensive experiment metadata
            'meta': {
                'kernel': kernel_type,
                'channel': channel,
                'seed': seed,
                'corpus': corpus_name,
                'use_contrastive': use_contrastive,
                'use_pca_removal': use_pca_removal,
                'use_gru': use_gru,
                'shared_pca': shared_pca,
                'kernel_params': kernel_params or {},
                # NEW: Reproducibility info
                'git_hash': git_hash,
                'git_dirty': git_dirty,
                'timestamp': timestamp,
                'n_articles': len(articles),
            },
            
            # NEW: Stage-by-stage variance tracking
            'variance_tracking': variance_tracker.to_dict(),
            
            # NEW: Article provenance summary
            'provenance': {
                'canonical_ids': result.get('bt_uid_list', []),
                'titles': [a.get('title', '')[:100] for a in articles],
                'sources': [a.get('source', a.get('publisher', 'unknown')) for a in articles],
                'urls': [a.get('url', '') for a in articles],
            },
        }
        
        # Save directly to output_dir (caller already creates structured path)
        # NOTE: Do NOT use build_structured_output_path here - that causes double-nesting
        # when run_full_experiment_suite already passes kernel/channel/corpus path
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"observer_{seed}.pt"
        torch.save(output_artifact, output_file)
        
        print(f"[OK] Saved: {output_file}")
        print(f"  -> {len(result.get('bt_uid_list', []))} articles with stable IDs")
        results[seed] = result
    
    return results


_original_run_multi_observer = run_multi_observer_experiment

def run_multi_observer_experiment(*args, articles=None, **kwargs):
    if articles is not None:
        return run_multi_observer_experiment_simple(articles=articles, **kwargs)
    else:
        return _original_run_multi_observer(*args, **kwargs)



