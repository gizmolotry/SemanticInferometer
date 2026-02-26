"""
core/nli_extraction.py

Contrastive multi-framing NLI extraction.

This file supports two regimes:

1) Legacy (24D-ish) contrastive triplets
   - n_pairs (typically 8)
   - per pair: [score_A, score_B, contrast] (3D)
   - total: n_pairs * 3 (typically 24D)

2) CLS+Logits stacking (typically 8192D)
   - per pair: CLS_delta (768D) + projected logits (projection_dim, default 256) => 1024D
   - total: n_pairs * (768 + projection_dim) (typically 8 * 1024 = 8192)

Design rules (matching complete_pipeline.py expectations):
- Paragraph awareness is supported in BOTH regimes and is forced ON by the NLIExtractor wrapper.
- PCA removal is applied to CLS features ONLY (when enabled), NEVER to logits.
- Projection layers are initialized with a fixed seed so they are NOT observer-seed-dependent.
- The NLIExtractor wrapper always returns one embedding per input article (no silent dropping),
  so downstream joins remain index-stable.

Notes:
- We use AutoModelForSequenceClassification for MNLI. To obtain "CLS" vectors, we request
  output_hidden_states=True and take the final layer hidden state at position 0, then project
  deterministically to 768D if needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _resolve_config_path(config_path: str) -> Path:
    p = Path(config_path)
    if not p.is_absolute():
        # Expect repo layout: core/nli_extraction.py -> repo_root/core/...
        repo_root = Path(__file__).resolve().parents[1]
        p = repo_root / config_path
    return p


def _split_into_paragraphs(text: str, max_paragraphs: int) -> List[str]:
    """
    Simple paragraph splitter:
      - split on blank lines
      - strip whitespace
      - return up to max_paragraphs non-empty paragraphs
    """
    if not isinstance(text, str):
        return []
    t = text.strip()
    if not t:
        return []
    # Normalize newlines and split on blank lines (>=1 empty line)
    chunks = re.split(r"\n\s*\n+", t)
    paras = [c.strip() for c in chunks if c and c.strip()]
    return paras[:max_paragraphs] if max_paragraphs and max_paragraphs > 0 else paras


def _normalize_weights(weights: List[float], n: int) -> List[float]:
    if n <= 0:
        return []
    w = weights[:n] if weights else []
    if len(w) < n:
        # extend with last weight if provided, else uniform
        if w:
            w = w + [w[-1]] * (n - len(w))
        else:
            w = [1.0] * n
    s = float(sum(w))
    if s <= 0:
        return [1.0 / n] * n
    return [float(x) / s for x in w]


def _init_fixed_linear(in_dim: int, out_dim: int, seed: int = 1337) -> nn.Linear:
    """
    Create a deterministic (fixed-seed) Linear layer with no bias.

    Why: complete_pipeline.py sets torch.manual_seed(observer_seed) per observer.
    If we used vanilla nn.Linear init, the CLS/logits projections would drift with
    observer_seed (bad: it contaminates what should be "content features" with
    observer randomness).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    layer = nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        w = torch.randn((out_dim, in_dim), generator=gen) / (in_dim ** 0.5)
        layer.weight.copy_(w)
    for p in layer.parameters():
        p.requires_grad = False
    return layer


def remove_top_pca_component_batch(X: torch.Tensor) -> torch.Tensor:
    """
    Remove the top principal component from X (2D: N x D), returning X' with same shape.
    We center, compute first PC via SVD, subtract projection, then re-add mean.

    Kept local to avoid import fragility; complete_pipeline also has pca_removal.py.
    """
    if X.dim() != 2:
        raise ValueError(f"Expected 2D tensor for PCA removal, got {tuple(X.shape)}")
    if X.shape[0] < 2:
        return X

    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu

    try:
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        v = Vh[0]
    except Exception:
        _, _, V = torch.pca_lowrank(Xc, q=1, center=False)
        v = V[:, 0]

    proj = (Xc @ v.unsqueeze(-1)) * v.unsqueeze(0)
    Xc2 = Xc - proj
    return Xc2 + mu


# --------------------------------------------------------------------------------------
# Legacy: contrastive triplets (24D-ish)
# --------------------------------------------------------------------------------------

class MultiFramingNLIExtractorLogits(nn.Module):
    """
    Contrastive NLI extractor (legacy 24D-ish).

    For each hypothesis pair (A,B), compute:
      - score_A = P(entail|A) - P(contradict|A)
      - score_B = P(entail|B) - P(contradict|B)
      - contrast = score_A - score_B

    Output is concatenation of [score_A, score_B, contrast] for all pairs.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        queries_config: str = "config/framing_queries.yaml",
        device: str = "cuda",
        max_length: int = 512,
        paragraph_aware: bool = False,
        paragraph_weights: Optional[List[float]] = None,
        cli_mode: str = "contrastive",
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.paragraph_aware = bool(paragraph_aware)
        self.paragraph_weights = paragraph_weights or [0.5, 0.3, 0.2]
        self.cli_mode = cli_mode
        self.extract_cli = False  # toggled by wrapper when needed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nli_model.to(self.device)
        self.nli_model.eval()
        for p in self.nli_model.parameters():
            p.requires_grad = False

        self.hypothesis_pairs = self._load_contrastive_pairs(queries_config)
        self.n_pairs = len(self.hypothesis_pairs)
        self.output_dim = self.n_pairs * 3

    def _load_contrastive_pairs(self, config_path: str) -> List[Tuple[str, str]]:
        p = _resolve_config_path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        pairs: List[Tuple[str, str]] = []
        queries = config.get("queries", {})
        for pair_name, pair_data in queries.items():
            if "A" not in pair_data or "B" not in pair_data:
                raise ValueError(
                    f"Pair '{pair_name}' missing 'A' or 'B'. Found keys: {list(pair_data.keys())}"
                )
            pairs.append((pair_data["A"], pair_data["B"]))
        if not pairs:
            raise ValueError(f"No queries loaded from {p}")
        return pairs

    @torch.no_grad()
    def _probs_for_hypothesis(self, premise: str, hypothesis: str) -> torch.Tensor:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        outputs = self.nli_model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu()

    @torch.no_grad()
    def extract_contrastive_triplet(self, article_text: str, hyp_a: str, hyp_b: str) -> torch.Tensor:
        probs_a = self._probs_for_hypothesis(article_text, hyp_a)
        probs_b = self._probs_for_hypothesis(article_text, hyp_b)

        c_a, n_a, e_a = probs_a.tolist()
        c_b, n_b, e_b = probs_b.tolist()

        score_a = e_a - c_a
        score_b = e_b - c_b
        contrast = score_a - score_b

        return torch.tensor([score_a, score_b, contrast], dtype=torch.float32)

    @torch.no_grad()
    def extract_contrastive_dual(self, article_text: str, hyp_a: str, hyp_b: str) -> Tuple[torch.Tensor, torch.Tensor]:
        probs_a = self._probs_for_hypothesis(article_text, hyp_a)
        probs_b = self._probs_for_hypothesis(article_text, hyp_b)

        c_a, n_a, e_a = probs_a.tolist()
        c_b, n_b, e_b = probs_b.tolist()

        logits_like = torch.tensor([c_a, n_a, e_a, c_b, n_b, e_b], dtype=torch.float32)

        score_a = e_a - c_a
        score_b = e_b - c_b
        contrast = score_a - score_b

        if self.cli_mode == "margin_uncertainty":
            margin = contrast
            uncertainty = 1.0 - ((n_a + n_b) / 2.0)
            cli = torch.tensor([margin, uncertainty], dtype=torch.float32)
        else:
            cli = torch.tensor([score_a, score_b, contrast], dtype=torch.float32)

        return logits_like, cli

    @torch.no_grad()
    def extract_multi_framing(self, article_text: str) -> torch.Tensor:
        if self.paragraph_aware:
            return self.extract_multi_framing_paragraph_aware(article_text)

        feats = [self.extract_contrastive_triplet(article_text, a, b) for (a, b) in self.hypothesis_pairs]
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def extract_multi_framing_dual(self, article_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.paragraph_aware:
            return self.extract_multi_framing_dual_paragraph_aware(article_text)

        logits_list, cli_list = [], []
        for hyp_a, hyp_b in self.hypothesis_pairs:
            l, c = self.extract_contrastive_dual(article_text, hyp_a, hyp_b)
            logits_list.append(l)
            cli_list.append(c)
        return torch.cat(logits_list, dim=-1), torch.cat(cli_list, dim=-1)

    @torch.no_grad()
    def extract_multi_framing_paragraph_aware(self, article_text: str) -> torch.Tensor:
        paras = _split_into_paragraphs(article_text, max_paragraphs=len(self.paragraph_weights))
        if not paras:
            feats = [self.extract_contrastive_triplet(article_text, a, b) for (a, b) in self.hypothesis_pairs]
            return torch.cat(feats, dim=-1)

        w = _normalize_weights(self.paragraph_weights, len(paras))
        weighted = None
        for i, para in enumerate(paras):
            f = [self.extract_contrastive_triplet(para, a, b) for (a, b) in self.hypothesis_pairs]
            f = torch.cat(f, dim=-1)
            weighted = f * w[i] if weighted is None else weighted + (f * w[i])
        return weighted

    @torch.no_grad()
    def extract_multi_framing_dual_paragraph_aware(self, article_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        paras = _split_into_paragraphs(article_text, max_paragraphs=len(self.paragraph_weights))
        if not paras:
            return self.extract_multi_framing_dual(article_text)

        w = _normalize_weights(self.paragraph_weights, len(paras))
        logits_acc, cli_acc = None, None
        for i, para in enumerate(paras):
            logits_list, cli_list = [], []
            for hyp_a, hyp_b in self.hypothesis_pairs:
                l, c = self.extract_contrastive_dual(para, hyp_a, hyp_b)
                logits_list.append(l)
                cli_list.append(c)
            lcat = torch.cat(logits_list, dim=-1)
            ccat = torch.cat(cli_list, dim=-1)

            logits_acc = lcat * w[i] if logits_acc is None else logits_acc + (lcat * w[i])
            cli_acc = ccat * w[i] if cli_acc is None else cli_acc + (ccat * w[i])
        return logits_acc, cli_acc


# Backwards aliases (older code may import these names)
MultiFramingNLIExtractor = MultiFramingNLIExtractorLogits


# --------------------------------------------------------------------------------------
# New: CLS+Logits stacking (8192D-ish)
# --------------------------------------------------------------------------------------

class MultiFramingNLIExtractorCLSLogits(nn.Module):
    """
    CLS+Logits extractor.

    For each hypothesis pair (A,B), compute:
      - cls_delta: projected (CLS(A) - CLS(B)) -> 768D
      - logits_proj: projected concat([logits(A), logits(B)]) -> projection_dim (default 256)
      - per pair: concat([cls_delta, logits_proj]) -> (768+P)

    Total per article: n_pairs * (768+P)  (typically 8*(768+256)=8192).

    IMPORTANT: PCA removal (if enabled) should be applied to CLS features ONLY,
    and is done by the NLIExtractor wrapper to keep logic centralized.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        queries_config: str = "config/framing_queries.yaml",
        device: str = "cuda",
        max_length: int = 512,
        paragraph_aware: bool = False,
        paragraph_weights: Optional[List[float]] = None,
        projection_dim: int = 256,
        cls_target_dim: int = 768,
        normalize_before_projection: bool = True,
        fixed_projection_seed: int = 1337,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.paragraph_aware = bool(paragraph_aware)
        self.paragraph_weights = paragraph_weights or [0.5, 0.3, 0.2]

        self.projection_dim = int(projection_dim)
        self.cls_target_dim = int(cls_target_dim)
        self.normalize_before_projection = bool(normalize_before_projection)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nli_model.to(self.device)
        self.nli_model.eval()
        for p in self.nli_model.parameters():
            p.requires_grad = False

        hidden_size = getattr(self.nli_model.config, "hidden_size", None) or self.cls_target_dim
        self._raw_hidden_size = int(hidden_size)

        # Deterministic projections (not tied to observer seeds)
        self.logits_proj = _init_fixed_linear(6, self.projection_dim, seed=fixed_projection_seed + 1).to(self.device)
        if self._raw_hidden_size == self.cls_target_dim:
            self.cls_proj = None
        else:
            self.cls_proj = _init_fixed_linear(self._raw_hidden_size, self.cls_target_dim, seed=fixed_projection_seed + 2).to(self.device)

        self.hypothesis_pairs = self._load_contrastive_pairs(queries_config)
        self.n_pairs = len(self.hypothesis_pairs)

        self.per_pair_dim = self.cls_target_dim + self.projection_dim
        self.output_dim = self.n_pairs * self.per_pair_dim

    def _load_contrastive_pairs(self, config_path: str) -> List[Tuple[str, str]]:
        p = _resolve_config_path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        pairs: List[Tuple[str, str]] = []
        queries = config.get("queries", {})
        for pair_name, pair_data in queries.items():
            if "A" not in pair_data or "B" not in pair_data:
                raise ValueError(
                    f"Pair '{pair_name}' missing 'A' or 'B'. Found keys: {list(pair_data.keys())}"
                )
            pairs.append((pair_data["A"], pair_data["B"]))
        if not pairs:
            raise ValueError(f"No queries loaded from {p}")
        return pairs

    @torch.no_grad()
    def _encode(self, premise: str, hypothesis: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        outputs = self.nli_model(**inputs, output_hidden_states=True, return_dict=True)

        logits = outputs.logits.squeeze(0).detach().float()  # (3,)
        hs = outputs.hidden_states[-1]
        cls = hs[:, 0, :].squeeze(0).detach().float()  # (hidden,)

        # Defensive: if hidden size surprises, re-init projection deterministically
        if cls.numel() != self._raw_hidden_size:
            self._raw_hidden_size = int(cls.numel())
            if self._raw_hidden_size != self.cls_target_dim:
                self.cls_proj = _init_fixed_linear(self._raw_hidden_size, self.cls_target_dim, seed=1337 + 2).to(self.device)
            else:
                self.cls_proj = None

        return logits, cls

    @torch.no_grad()
    def extract_pair_parts(self, article_text: str, hyp_a: str, hyp_b: str) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_a, cls_a = self._encode(article_text, hyp_a)
        logits_b, cls_b = self._encode(article_text, hyp_b)

        if self.normalize_before_projection:
            cls_a = F.normalize(cls_a.unsqueeze(0), dim=-1).squeeze(0)
            cls_b = F.normalize(cls_b.unsqueeze(0), dim=-1).squeeze(0)
            logits_a = logits_a / (logits_a.norm() + 1e-12)
            logits_b = logits_b / (logits_b.norm() + 1e-12)

        cls_delta = cls_a - cls_b
        if self.cls_proj is not None:
            cls_delta = self.cls_proj(cls_delta)
        cls_delta = cls_delta.view(-1)

        logits6 = torch.cat([logits_a, logits_b], dim=-1).view(-1)  # (6,)
        logits_proj = self.logits_proj(logits6).view(-1)  # (P,)

        return cls_delta.detach().cpu(), logits_proj.detach().cpu()

    @torch.no_grad()
    def extract_article_parts(self, article_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          cls_mat: [n_pairs, 768]
          log_mat: [n_pairs, P]
        """
        if not self.paragraph_aware:
            cls_list, log_list = [], []
            for a, b in self.hypothesis_pairs:
                cls_d, lp = self.extract_pair_parts(article_text, a, b)
                cls_list.append(cls_d)
                log_list.append(lp)
            return torch.stack(cls_list, dim=0), torch.stack(log_list, dim=0)

        paras = _split_into_paragraphs(article_text, max_paragraphs=len(self.paragraph_weights))
        if not paras:
            return self.extract_article_parts(article_text)

        w = _normalize_weights(self.paragraph_weights, len(paras))

        cls_acc, log_acc = None, None
        for i, para in enumerate(paras):
            cls_list, log_list = [], []
            for a, b in self.hypothesis_pairs:
                cls_d, lp = self.extract_pair_parts(para, a, b)
                cls_list.append(cls_d)
                log_list.append(lp)
            cls_mat = torch.stack(cls_list, dim=0)
            log_mat = torch.stack(log_list, dim=0)

            cls_acc = cls_mat * w[i] if cls_acc is None else cls_acc + (cls_mat * w[i])
            log_acc = log_mat * w[i] if log_acc is None else log_acc + (log_mat * w[i])

        return cls_acc, log_acc


# --------------------------------------------------------------------------------------
# Wrapper used by core/complete_pipeline.py
# --------------------------------------------------------------------------------------

class NLIExtractor:
    """
    Wrapper for complete_pipeline.py.

    complete_pipeline.initialize_full_pipeline() may pass:
      use_cls_tokens, projection_dim, apply_pca_to_cls, normalize_before_projection, paragraph_aware, paragraph_weights.

    We accept those args here to prevent keyword crashes and to route to the right extractor.

    IMPORTANT: Paragraph-awareness is forced ON here (regardless of caller arg),
    because it’s a core assumption of your current pipeline.
    """

    def __init__(
        self,
        device: str = "cuda",
        queries_config: str = "config/framing_queries.yaml",
        extract_cli: bool = False,
        cli_mode: str = "contrastive",
        paragraph_aware: bool = True,
        paragraph_weights: Optional[List[float]] = None,
        use_cls_tokens: bool = False,
        projection_dim: int = 256,
        apply_pca_to_cls: bool = True,
        normalize_before_projection: bool = True,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        max_length: int = 512,
        **_ignored: Any,
    ):
        self.device = device
        self.extract_cli = bool(extract_cli)
        self.cli_mode = cli_mode

        # Force paragraph-aware always on (argument accepted for legacy callers, ignored)
        self.paragraph_aware = True
        self.paragraph_weights = paragraph_weights or [0.5, 0.3, 0.2]

        self.use_cls_tokens = bool(use_cls_tokens)
        self.projection_dim = int(projection_dim)
        self.apply_pca_to_cls = bool(apply_pca_to_cls)
        self.normalize_before_projection = bool(normalize_before_projection)

        if self.use_cls_tokens:
            self.extractor = MultiFramingNLIExtractorCLSLogits(
                model_name=model_name,
                queries_config=queries_config,
                device=device,
                max_length=max_length,
                paragraph_aware=self.paragraph_aware,
                paragraph_weights=self.paragraph_weights,
                projection_dim=self.projection_dim,
                normalize_before_projection=self.normalize_before_projection,
            )
        else:
            self.extractor = MultiFramingNLIExtractorLogits(
                model_name=model_name,
                queries_config=queries_config,
                device=device,
                max_length=max_length,
                paragraph_aware=self.paragraph_aware,
                paragraph_weights=self.paragraph_weights,
                cli_mode=self.cli_mode,
            )

    def _get_text(self, article: Union[str, Dict[str, str]]) -> str:
        if isinstance(article, dict):
            return (article.get("content") or article.get("text") or "").strip()
        return str(article).strip()

    def _zero_embedding(self) -> torch.Tensor:
        dim = self.extractor.output_dim
        return torch.zeros((dim,), dtype=torch.float32)

    def extract_nli_pairs(self, articles: List[Union[str, Dict[str, str]]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if self.use_cls_tokens:
            cls_batches, log_batches, errors = [], [], []

            for art in articles:
                text = self._get_text(art)
                if not text or len(text) < 5:
                    cls_batches.append(torch.zeros((self.extractor.n_pairs, self.extractor.cls_target_dim), dtype=torch.float32))
                    log_batches.append(torch.zeros((self.extractor.n_pairs, self.extractor.projection_dim), dtype=torch.float32))
                    errors.append("too_short")
                    continue
                try:
                    cls_mat, log_mat = self.extractor.extract_article_parts(text)
                    cls_batches.append(cls_mat.float())
                    log_batches.append(log_mat.float())
                    errors.append(None)
                except Exception as e:
                    cls_batches.append(torch.zeros((self.extractor.n_pairs, self.extractor.cls_target_dim), dtype=torch.float32))
                    log_batches.append(torch.zeros((self.extractor.n_pairs, self.extractor.projection_dim), dtype=torch.float32))
                    errors.append(str(e))

            cls_batch = torch.stack(cls_batches, dim=0)  # (N, pairs, 768)
            log_batch = torch.stack(log_batches, dim=0)  # (N, pairs, P)

            # PCA removal: CLS only (never logits)
            if self.apply_pca_to_cls and cls_batch.shape[0] >= 2:
                X = cls_batch.reshape(-1, cls_batch.shape[-1])  # (N*pairs, 768)
                X2 = remove_top_pca_component_batch(X)
                cls_batch = X2.reshape_as(cls_batch)

            for i in range(cls_batch.shape[0]):
                cls_flat = cls_batch[i].reshape(-1)
                log_flat = log_batch[i].reshape(-1)
                emb = torch.cat([cls_flat, log_flat], dim=-1).to(torch.float32)

                payload: Dict[str, Any] = {
                    "embedding": emb,
                    "embedding_cls": cls_flat.to(torch.float32),
                    "embedding_logits": log_flat.to(torch.float32),  # projected logits only
                }
                if errors[i] is not None:
                    payload["error"] = errors[i]
                results.append(payload)

            return results

        # Legacy regime
        for art in articles:
            text = self._get_text(art)
            if not text or len(text) < 5:
                results.append({"embedding": self._zero_embedding(), "error": "too_short"})
                continue

            try:
                if self.extract_cli:
                    logits_like, cli = self.extractor.extract_multi_framing_dual(text)
                    results.append({
                        "embedding": cli.to(torch.float32),
                        "embedding_cli": cli.to(torch.float32),
                        "embedding_logits": logits_like.to(torch.float32),
                    })
                else:
                    emb = self.extractor.extract_multi_framing(text)
                    results.append({"embedding": emb.to(torch.float32)})
            except Exception as e:
                results.append({"embedding": self._zero_embedding(), "error": str(e)})

        return results


# --------------------------------------------------------------------------------------
# Optional helper: run BOTH regimes (24D-ish + 8192D-ish)
# --------------------------------------------------------------------------------------

class DualNLIExtractor:
    """
    Convenience wrapper to always extract BOTH embeddings per article:

      - legacy: contrastive triplets (n_pairs*3; typically 24D)
      - cls: CLS+projected logits (n_pairs*(768+P); typically 8192D)

    This is not required by complete_pipeline.py, but it is useful for
    orchestrators / runners that want to save both in a single pass.
    """

    def __init__(
        self,
        device: str = "cuda",
        queries_config: str = "config/framing_queries.yaml",
        cli_mode: str = "contrastive",
        projection_dim: int = 256,
        apply_pca_to_cls: bool = True,
        normalize_before_projection: bool = True,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        max_length: int = 512,
        paragraph_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ):
        self.legacy = NLIExtractor(
            device=device,
            queries_config=queries_config,
            extract_cli=False,
            cli_mode=cli_mode,
            paragraph_aware=True,
            paragraph_weights=paragraph_weights,
            use_cls_tokens=False,
            model_name=model_name,
            max_length=max_length,
            **kwargs,
        )
        self.cls = NLIExtractor(
            device=device,
            queries_config=queries_config,
            extract_cli=False,
            paragraph_aware=True,
            paragraph_weights=paragraph_weights,
            use_cls_tokens=True,
            projection_dim=projection_dim,
            apply_pca_to_cls=apply_pca_to_cls,
            normalize_before_projection=normalize_before_projection,
            model_name=model_name,
            max_length=max_length,
            **kwargs,
        )

    def extract_both(self, articles: List[Union[str, Dict[str, str]]]) -> List[Dict[str, Any]]:
        legacy_out = self.legacy.extract_nli_pairs(articles)
        cls_out = self.cls.extract_nli_pairs(articles)

        results: List[Dict[str, Any]] = []
        for lo, co in zip(legacy_out, cls_out):
            results.append({
                "legacy": lo.get("embedding"),
                "cls": co.get("embedding"),
                "legacy_meta": {k: v for k, v in lo.items() if k != "embedding"},
                "cls_meta": {k: v for k, v in co.items() if k != "embedding"},
            })
        return results


if __name__ == "__main__":
    # Smoke tests (shapes)
    articles = [
        {"content": "Israel says it is responding to attacks. Hamas says it is resisting occupation."},
        {"content": "A ceasefire was proposed and both sides accused the other of violations."},
    ]

    print("Legacy wrapper...")
    nli = NLIExtractor(device="cpu", use_cls_tokens=False, paragraph_aware=True)
    out = nli.extract_nli_pairs(articles)
    print("  n =", len(out), "dim =", out[0]["embedding"].numel())

    print("CLS+logits wrapper...")
    nli2 = NLIExtractor(device="cpu", use_cls_tokens=True, projection_dim=256, apply_pca_to_cls=True)
    out2 = nli2.extract_nli_pairs(articles)
    print("  n =", len(out2), "dim =", out2[0]["embedding"].numel(), "(expected ~8192 if 8 pairs)")
