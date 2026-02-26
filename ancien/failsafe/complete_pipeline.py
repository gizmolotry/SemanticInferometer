"""
complete_pipeline.py â€” end-to-end orchestration for the Belief Transformer / bias-geometry experiments.

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
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import datetime
from email.utils import parsedate_to_datetime

from .nli_extraction import NLIExtractor
from .temporal_gru import TemporalGRU
from .rks_feature_map import RKSFeatureMap
from .cross_article_attention import CrossArticleAttention
from .pca_removal import remove_top_pca_component
from .attention_recorder import AttentionRecorder
from .provenance_tracker import ProvenanceTracker, ProvenanceEntry, PipelineProvenanceEntry

# Optional provenance sidecar (NOT injected into geometry; prevents 'metadata cheating')
try:
    from .provenance import build_metadata_indices, encode_article_metadata  # type: ignore
    _HAVE_PROVENANCE_SIDECAR = True
except Exception:
    build_metadata_indices = None  # type: ignore
    encode_article_metadata = None  # type: ignore
    _HAVE_PROVENANCE_SIDECAR = False


# Canonical ID system (thesis-grade reproducibility)
try:
    from .canonical_ids import (
        assign_canonical_uids,
        canonical_sort,
        create_manifest,
        compute_corpus_hash,
        verify_array_alignment,
    )
    _HAS_CANONICAL_IDS = True
except Exception:
    # Fall back to legacy stable_article_uid logic if canonical_ids.py isn't present
    _HAS_CANONICAL_IDS = False


# --------------------------------------------------------------------------------------
# Timestamp + stable ID utilities
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


class VarianceTracker:
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
# Kernel geometry helpers ("adult mode"): exact kernel PCA + Nyström approximation
# --------------------------------------------------------------------------------------

def _pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
    """Return NxN matrix of squared Euclidean distances."""
    # torch.cdist returns Euclidean distances; square it
    return torch.cdist(X, X, p=2) ** 2


def _pairwise_l1_dists(X: torch.Tensor) -> torch.Tensor:
    """Return NxN matrix of L1 distances."""
    # Efficient L1 cdist isn't built-in; use broadcast for moderate N.
    # For N<=2000 this is usually fine; for larger, prefer chunking.
    diff = X.unsqueeze(1) - X.unsqueeze(0)  # [N,N,D]
    return diff.abs().sum(dim=-1)


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
    Nyström approximation for kernel PCA.

    Strategy:
      - sample m landmark indices (uniform)
      - compute C = K(X, landmarks) [N,m]
      - compute W = K(landmarks, landmarks) [m,m]
      - approximate K ≈ C W^{-1} C^T (optionally centered approximately by centering the implicit K)

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

    # Nyström feature map: Phi = C * evecs_k * diag(1/sqrt(evals_k))
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
    geometry_mode: str = "rks",  # "rks" | "kernel_pca" | "nystrom"
    adult_kernel: str = "rbf",
    adult_gamma: float = 1.0,
    adult_sigma: Optional[float] = None,
    adult_center: bool = True,
    adult_nystrom_m: int = 256,
    compare_logits_vs_cli: bool = False,
    rks_output_dim: Optional[int] = None,
):
    """
    Initialize the full belief transformer pipeline components.

    This is the "module factory" â€” it builds the objects that `BeliefTransformerPipeline`
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

    # Auto-adjust for CLS mode
    if use_cls_tokens:
        embedding_dim = 8192  # 8 pairs x 1024D per pair
        pca_remove = False  # PCA applied to CLS tokens during NLI extraction
        print(f"CLS+Logits mode enabled: embedding_dim auto-set to {embedding_dim}")
        print(f"  PCA will be applied to CLS tokens during NLI extraction (not in pipeline)")

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

    # CLS+Logits stacking regime (only if NLIExtractor version supports it)
    if use_cls_tokens:
        nli_kwargs.update({
            "use_cls_tokens": True,
            "projection_dim": 256,
            "apply_pca_to_cls": True,
            "normalize_before_projection": True,
        })

    sig = inspect.signature(NLIExtractor.__init__)
    filtered = {k: v for k, v in nli_kwargs.items() if k in sig.parameters}
    missing = [k for k in nli_kwargs.keys() if k not in filtered]
    if missing:
        # Don't hard fail; just document the downgrade.
        if use_cls_tokens and any(k in missing for k in ("use_cls_tokens", "projection_dim", "apply_pca_to_cls")):
            print("WARNING: NLIExtractor.__init__ does not accept CLS+Logits args; falling back to legacy extraction.")
        if compare_logits_vs_cli and "extract_cli" in missing:
            print("WARNING: NLIExtractor.__init__ does not accept extract_cli; logits/CLI comparison disabled.")

    nli_extractor = NLIExtractor(**filtered)


    gru_model = None
    if use_gru:
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
        rks_map = RKSFeatureMap(
            input_dim=output_dim if use_gru else embedding_dim,
            output_dim=final_dim,
            kernel_type="rbf",
            gamma=1.0,
            random_seed=random_seed,
            device=device,
        )


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
    }


class BeliefTransformerPipeline:
    """
    Main pipeline class for multi-step bias geometry processing.

    This class is instantiated per "observer" seed â€” meaning each run can have its own
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

        # Geometry mode: "rks" (Random Kitchen Sinks) or "kernel_pca"/"nystrom" (kernel trick / spectral)
        self.geometry_mode = (components.get("geometry_mode", "rks") or "rks").lower().strip()
        self.adult_kernel = components.get("adult_kernel", "rbf")
        self.adult_gamma = float(components.get("adult_gamma", 1.0))
        self.adult_sigma = components.get("adult_sigma", None)
        self.adult_center = bool(components.get("adult_center", True))
        self.adult_nystrom_m = int(components.get("adult_nystrom_m", 256))

        # Nominal representation dim (may be capped at runtime for exact kernel PCA).
        self.final_dim = int(components.get("final_dim", 0) or 0)

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

    def process_month(self, articles: List[Dict], month_name: str = "unknown") -> Dict:
        """
        Process a month of articles through the full pipeline.

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
        # Canonical IDs + timestamp normalization (thesis-grade joins)
        # ------------------------------------------------------------------
        # We never trust list index. We generate canonical `bt_uid` values and put the
        # corpus into a canonical order so embeddings/maps are joinable across runs.
        #
        # NOTE:
        # - Canonical order (lexicographic by bt_uid) is the "index spine" for all arrays.
        # - TemporalGRU still operates in timestamp order, but outputs are restored to
        #   canonical order before we save anything.
        # - We preserve `original_index` so you can trace back to input order when needed.
        if not articles:
            articles = []

        # Work on shallow copies so the caller's list isn't mutated unexpectedly.
        articles_work: List[Dict] = []
        for i, a in enumerate(articles):
            if isinstance(a, dict):
                d = dict(a)
            else:
                # extremely defensive: tolerate non-dict rows
                d = {"content": str(a)}
            d["_bt_original_index"] = i
            articles_work.append(d)

        canonical_stats = {}
        if _HAS_CANONICAL_IDS:
            articles_work, canonical_stats = assign_canonical_uids(articles_work)
            canonical_articles = canonical_sort(articles_work)
        else:
            # Legacy: compute bt_uid via stable_article_uid and preserve original order.
            _uid_collision_counter = {}
            for i, art in enumerate(articles_work):
                art["bt_uid"] = stable_article_uid(art, i, _uid_collision_counter)
            canonical_articles = list(articles_work)

        # Canonical UID list + index maps
        bt_uids = [a.get("bt_uid") for a in canonical_articles]
        uid_to_canonical_index = {uid: i for i, uid in enumerate(bt_uids)}

        # Timestamp parsing aligned to canonical order
        ts_source_keys: List[Optional[str]] = []
        ts_raw_values: List[Any] = []
        ts_epoch_s: List[Optional[int]] = []
        ts_iso_utc: List[Optional[str]] = []

        for art in canonical_articles:
            src_key, raw_ts, _dt, epoch_s, iso_utc = extract_article_timestamp(art)
            ts_source_keys.append(src_key)
            ts_raw_values.append(raw_ts)
            ts_epoch_s.append(epoch_s)
            ts_iso_utc.append(iso_utc)

        n_with_epoch = sum(1 for x in ts_epoch_s if x is not None)
        timestamp_coverage = {
            "total": len(canonical_articles),
            "parseable": n_with_epoch,
            "parseable_fraction": float(n_with_epoch / max(1, len(canonical_articles))),
            "fields_seen": sorted({k for k in ts_source_keys if k}),
        }

        # A single join table: maps canonical index <-> original index, and carries normalized timestamps.
        timeline: List[Dict[str, Any]] = []
        for canon_i, art in enumerate(canonical_articles):
            orig_i = int(art.get("_bt_original_index", canon_i))
            timeline.append({
                "bt_uid": art.get("bt_uid"),
                "canonical_index": canon_i,
                "original_index": orig_i,
                "sorted_position": canon_i,  # overwritten if GRU temporal sort is applied
                "timestamp_field": ts_source_keys[canon_i],
                "timestamp_raw": ts_raw_values[canon_i],
                "timestamp_epoch_s": ts_epoch_s[canon_i],
                "timestamp_iso_utc": ts_iso_utc[canon_i],
            })

        # Canonical corpus hash (helps detect silent data drift)
        canonical_hash = None
        if _HAS_CANONICAL_IDS:
            try:
                canonical_hash = compute_corpus_hash(canonical_articles)
            except Exception:
                canonical_hash = None

        # Store article metadata for provenance
        provenance_metadata = None
        if self.enable_provenance and self.provenance_tracker:
            provenance_metadata = {
                "month": month_name,
                "n_articles": len(canonical_articles),
                "canonical_hash": canonical_hash,
                "canonical_stats": canonical_stats,
                "articles": {
                    canon_i: {
                        "canonical_index": canon_i,
                        "original_index": int(art.get("_bt_original_index", canon_i)),
                        "bt_uid": art.get("bt_uid"),
                        "id": art.get("id", art.get("url", f"article_{canon_i}")),
                        "source": art.get("source", "unknown"),
                        "url": art.get("url", ""),
                        "timestamp_field": ts_source_keys[canon_i],
                        "timestamp_raw": ts_raw_values[canon_i],
                        "timestamp_epoch_s": ts_epoch_s[canon_i],
                        "timestamp_iso_utc": ts_iso_utc[canon_i],
                        "title": art.get("title", ""),
                        "content_preview": art.get("content", "")[:200] if art.get("content") else "",
                    }
                    for canon_i, art in enumerate(canonical_articles)
                },
            }

        # Use canonical order for all downstream computation + saved arrays
        articles = canonical_articles
        
        # ------------------------------------------------------------------
        # Optional provenance sidecar (analysis only)
        # ------------------------------------------------------------------
        # We compute provenance embeddings purely as a sidecar artifact for later
        # correlation/cheat-detection diagnostics. We DO NOT fuse them into the
        # semantic geometry features (logits/CLS/etc.).
        provenance_sidecar = None
        if _HAVE_PROVENANCE_SIDECAR and build_metadata_indices is not None and encode_article_metadata is not None:
            try:
                meta_idx = build_metadata_indices(articles)
                # Keep this small to avoid bloating artifacts; adjust if you explicitly want more.
                prov_dim = 32
                from .provenance import ProvenanceEncoder  # type: ignore
                enc = ProvenanceEncoder(
                    feature_dim=prov_dim,
                    num_sources=int(meta_idx.get('num_sources', 1)),
                    num_authors=int(meta_idx.get('num_authors', 0)),
                    min_author_articles=int(meta_idx.get('min_author_articles', 10)),
                )
                with torch.no_grad():
                    prov_emb = encode_article_metadata(articles, meta_idx, enc, date_range=None)
                provenance_sidecar = {
                    'dim': int(prov_dim),
                    'embeddings': prov_emb.detach().cpu().numpy().astype('float32'),
                    'indices': meta_idx,
                }
                diagnostics.setdefault('provenance_sidecar', {})['enabled'] = True
                diagnostics['provenance_sidecar']['dim'] = int(prov_dim)
                diagnostics['provenance_sidecar']['num_sources'] = int(meta_idx.get('num_sources', 0))
                diagnostics['provenance_sidecar']['num_authors'] = int(meta_idx.get('num_authors', 0))
            except Exception as e:
                diagnostics.setdefault('provenance_sidecar', {})['error'] = str(e)


        # Diagnostic tracking

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
        diagnostics["uid_collision_bases"] = {
            "n_articles": int(canonical_stats.get("n_articles", len(articles))),
            "n_unique_base_uids": int(canonical_stats.get("n_unique_base_uids", len(set(bt_uids)))),
            "n_collisions": int(canonical_stats.get("n_collisions", 0)),
        }

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
                canon_i = entry.get("canonical_index", entry.get("original_index"))
                entry["sorted_position"] = inverse_indices[int(canon_i)]

            diagnostics["steps"].append("temporal_sort_applied")
            diagnostics["temporal_sort_indices"] = sorted_indices

        # ------------------------------------------------------------------
        # Shared downstream runner (GRU -> geometry -> attention -> (optional) PCA)
        # ------------------------------------------------------------------
        def _run_channel(channel_name: str, X_in: torch.Tensor) -> torch.Tensor:
            ch_diag = diagnostics["channels"].setdefault(channel_name, {})
            t0 = time.time()

            # Step 2: Optional temporal GRU
            X = X_in
            if self.gru_model is not None:
                t_gru = time.time()
                # Reorder for GRU
                sorted_tensor = X[sorted_indices]
                if sorted_tensor.dim() > 2:
                    sorted_tensor = sorted_tensor.squeeze()

                temporal_out = self.gru_model(sorted_tensor.unsqueeze(0))  # (1, N, D)
                temporal_out = temporal_out.squeeze(0)  # (N, D)

                # Restore original order
                inv = torch.tensor(inverse_indices, dtype=torch.long, device=temporal_out.device)
                X = temporal_out[inv]

                ch_diag["timing_temporal_gru"] = time.time() - t_gru
                ch_diag["temporal_variance"] = float(X.var().item())

            # Step 3: Geometry stage
            t_geom = time.time()
            Z = X
            if self.geometry_mode == "rks" and self.rks_map is not None:
                Z = self.rks_map.transform(X)
                diagnostics["steps"].append(f"rks_feature_map:{channel_name}")
            elif self.geometry_mode == "kernel_pca":
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
            elif self.geometry_mode == "nystrom":
                # Nyström approximation (sub-quadratic in N for large N)
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
                if A.dim() == 2:
                    cur_d = int(A.shape[1])
                    # Try to detect mismatch without relying on internal attribute names.
                    need_rebuild = False
                    for attr in ("feature_dim", "d_model", "dim"):
                        if hasattr(self.attention_model, attr):
                            if int(getattr(self.attention_model, attr)) != cur_d:
                                need_rebuild = True
                            break
                    if need_rebuild:
                        self.attention_model = CrossArticleAttention(
                            feature_dim=cur_d,
                            num_heads=8,
                        )

                attn_out, attn_weights = self.attention_model(A)
                self.recorder.record(f"{month_name}:{channel_name}", attn_weights)

                A = attn_out
                if self.normalize_features:
                    A = F.normalize(A, dim=1)

                ch_diag["timing_attention"] = time.time() - t_attn
                ch_diag["attention_variance"] = float(A.var().item())

            # Step 5: Optional PCA removal
            # IMPORTANT RULE:
            # - If compare_logits_vs_cli=True, we skip pipeline PCA entirely so the comparison is fair.
            # - If CLS stacking is enabled, PCA is handled inside the extractor (CLS only), not here.
            F_out = A
            if self.compare_logits_vs_cli:
                diagnostics["steps"].append(f"pca_skipped_compare_mode:{channel_name}")
            elif self.use_cls_tokens:
                diagnostics["steps"].append(f"pca_skipped_cls_mode:{channel_name}")
            elif self.pca_remove:
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
            else:
                ch_diag["final_variance"] = float(F_out.var().item())

            ch_diag["timing_total_channel"] = time.time() - t0
            return F_out

        # Run all channels
        out_by_channel: Dict[str, torch.Tensor] = {}
        for ch, Xch in base_by_channel.items():
            out_by_channel[ch] = _run_channel(ch, Xch)

        # Choose the canonical "features" output for backward compatibility.
        if self.compare_logits_vs_cli:
            final_features = out_by_channel["logits"]
        else:
            final_features = out_by_channel["main"]

        diagnostics["variance"]["final_variance"] = float(final_features.var().item())
        diagnostics["timing"]["total"] = time.time() - start_total

        # Expose comparison outputs (if present)
        features_cli = out_by_channel.get("cli", None)

        # Provenance log (optional)
        provenance_entries = []
        if self.enable_provenance and self.provenance_tracker is not None:
            # Record the "canonical" channel as the primary artifact.
            try:
                prov_entry = PipelineProvenanceEntry(
                    month=month_name,
                    seed=int(self.random_seed),
                    mode=f"{self.geometry_mode}|cls={int(self.use_cls_tokens)}|compare={int(self.compare_logits_vs_cli)}",
                    embedding_type=("logits" if self.compare_logits_vs_cli else "main"),
                    n_articles=int(final_features.shape[0]),
                    dim=int(final_features.shape[1]),
                    pipeline_steps=diagnostics.get("steps", []),
                    timings=diagnostics.get("timing", diagnostics.get("timings", {})),
                    variance=diagnostics.get("variance", {}),
                    metadata=provenance_metadata,
                    timestamp=float(time.time()),
                )
                self.provenance_tracker.add_entry(prov_entry)
                # Optional: if we computed a secondary channel, record it too.
                if self.compare_logits_vs_cli and features_cli is not None:
                    prov_entry_cli = PipelineProvenanceEntry(
                        month=month_name,
                        seed=int(self.random_seed),
                        mode=f"{self.geometry_mode}|cls={int(self.use_cls_tokens)}|compare={int(self.compare_logits_vs_cli)}",
                        embedding_type="cli",
                        n_articles=int(features_cli.shape[0]),
                        dim=int(features_cli.shape[1]),
                        pipeline_steps=diagnostics.get("steps", []),
                        timings=diagnostics.get("timing", diagnostics.get("timings", {})),
                        variance=diagnostics.get("variance", {}),
                        metadata=provenance_metadata,
                        timestamp=float(time.time()),
                    )
                    self.provenance_tracker.add_entry(prov_entry_cli)
                # Stop here (we don't want any old half-line calls)
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
            "provenance_sidecar": provenance_sidecar,
        }

        if features_cli is not None:
            out["features_cli"] = features_cli.detach().cpu().numpy()
            out["embeddings_cli"] = features_cli.detach().cpu().numpy()


        # ------------------------------------------------------------------
        # Canonical integrity: verify array lengths + write manifest (if enabled)
        # ------------------------------------------------------------------
        if _HAS_CANONICAL_IDS:
            try:
                arrays_to_check = {"embeddings": out.get("embeddings")}
                if out.get("features_cli") is not None:
                    arrays_to_check["features_cli"] = out.get("features_cli")
                verify_array_alignment(arrays=arrays_to_check, uids=bt_uids)
            except Exception as e:
                diagnostics.setdefault("canonical_ids", {})
                diagnostics["canonical_ids"]["alignment_error"] = str(e)

            # Persist a per-month manifest next to provenance (if a directory is known)
            try:
                if self.provenance_dir is not None:
                    run_root = Path(self.provenance_dir).parent
                    manifest_path = run_root / f"{month_name}_manifest.json"
                    manifest = create_manifest(canonical_articles, manifest_path)
                    diagnostics.setdefault("canonical_ids", {})
                    diagnostics["canonical_ids"]["manifest_path"] = str(manifest_path)
                    diagnostics["canonical_ids"]["canonical_hash"] = manifest.get("canonical_hash")
            except Exception as e:
                diagnostics.setdefault("canonical_ids", {})
                diagnostics["canonical_ids"]["manifest_error"] = str(e)

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

    This function is intentionally "outer-loop glue" â€” it doesnâ€™t decide your science;
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
                components["rks_map"].kernel_type = kernel

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
    **kwargs
):
    """Compatibility wrapper for run_experiments.py"""
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Observer {seed}")
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
        
        result = pipeline.process_month(articles, month_name="batch")
        
        output_file = output_dir / f"observer_{seed}.pt"
        torch.save({
            'embeddings': result['embeddings'],
            'features': result.get('features'),
            'embeddings_cli': result.get('embeddings_cli'),
            'features_cli': result.get('features_cli'),
            'seed': seed,
            'n_articles': len(articles),
        }, output_file)
        
        print(f"âœ… Saved: {output_file}")
        results[seed] = result
    
    return results


_original_run_multi_observer = run_multi_observer_experiment

def run_multi_observer_experiment(*args, articles=None, **kwargs):
    if articles is not None:
        return run_multi_observer_experiment_simple(articles=articles, **kwargs)
    else:
        return _original_run_multi_observer(*args, **kwargs)