"""
unified_extraction.py - Single Inference Pass for Complete Belief Transformer Output

This module implements the pipeline pivot: one DeBERTa inference produces ALL outputs:
1. Raw logits (24D) - verdict coordinates, NO kernel expansion
2. Individual bot CLS embeddings (8 x 768D) - for Dirichlet fusion
3. Optional attention head outputs - diagnostic recording

Design principles from the pivot document:
- Logits are judgment coordinates, not a semantic manifold
- CLS embeddings carry the semantic geometry
- Observers are Dirichlet mixtures, not random projections
- Heads are logged for forensics, not used in pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yaml


@dataclass
class UnifiedExtractionConfig:
    """Configuration for unified extraction."""
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    queries_config: str = "config/framing_queries.yaml"
    max_length: int = 512
    device: str = "cuda"
    
    # Output controls
    extract_logits: bool = True          # Raw 24D logits (3 classes x 8 framings)
    extract_cls: bool = True             # Per-bot CLS embeddings
    record_heads: bool = False           # Record attention head outputs
    
    # Paragraph awareness
    paragraph_aware: bool = True
    paragraph_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # Dirichlet fusion config (applied after extraction)
    dirichlet_alpha: float = 1.0
    dirichlet_n_observers: int = 50
    dirichlet_rks_dim: int = 2048
    dirichlet_seed: int = 42


class UnifiedNLIExtractor(nn.Module):
    """
    Single-pass extractor that produces all belief transformer outputs.
    
    For each article and each of 8 hypothesis pairs (bots), extracts:
    - Raw logits: [3] per bot -> [8, 3] per article -> flattened to [24]
    - CLS embedding: [hidden_dim] per bot -> [8, hidden_dim] per article
    - (Optional) Attention patterns from each head
    
    This replaces the fragmented extraction logic with one clean pass.
    """
    
    def __init__(self, config: UnifiedExtractionConfig = None):
        super().__init__()
        
        if config is None:
            config = UnifiedExtractionConfig()
        self.config = config
        
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Load model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            output_attentions=config.record_heads,
            output_hidden_states=True,
        )
        self.nli_model.to(self.device)
        self.nli_model.eval()
        
        for p in self.nli_model.parameters():
            p.requires_grad = False
        
        self.hidden_size = self.nli_model.config.hidden_size
        
        # Load hypothesis pairs
        self.hypothesis_pairs = self._load_hypothesis_pairs(config.queries_config)
        self.n_bots = len(self.hypothesis_pairs)
        
        print(f"[UnifiedExtractor] Loaded {self.n_bots} hypothesis pairs (bots)")
        print(f"[UnifiedExtractor] Hidden size: {self.hidden_size}")
        print(f"[UnifiedExtractor] Device: {self.device}")
    
    def _load_hypothesis_pairs(self, config_path: str) -> List[Tuple[str, str]]:
        """Load contrastive hypothesis pairs from config."""
        p = Path(config_path)
        if not p.is_absolute():
            # Try relative to repo root
            repo_root = Path(__file__).resolve().parents[1]
            p = repo_root / config_path
        
        if not p.exists():
            raise FileNotFoundError(f"Queries config not found: {p}")
        
        with open(p, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        pairs = []
        for pair_name, pair_data in config.get('queries', {}).items():
            if 'A' in pair_data and 'B' in pair_data:
                pairs.append((pair_data['A'], pair_data['B']))
        
        if not pairs:
            raise ValueError(f"No valid hypothesis pairs in {p}")
        
        return pairs
    
    @torch.no_grad()
    def _encode_single(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single premise-hypothesis pair.
        
        Returns dict with:
        - 'logits': [3] raw NLI logits (contradiction, neutral, entailment)
        - 'cls': [hidden_dim] CLS token embedding
        - 'attentions': list of attention tensors (if record_heads=True)
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=self.config.max_length,
            padding=True,
        ).to(self.device)
        
        outputs = self.nli_model(
            **inputs,
            output_hidden_states=True,
            output_attentions=self.config.record_heads,
            return_dict=True,
        )
        
        result = {
            'logits': outputs.logits.squeeze(0).cpu(),  # [3]
            'cls': outputs.hidden_states[-1][:, 0, :].squeeze(0).cpu(),  # [hidden]
        }
        
        if self.config.record_heads and outputs.attentions is not None:
            # Each attention is [batch, heads, seq, seq]
            # We want [n_layers, n_heads, seq, seq]
            result['attentions'] = [a.squeeze(0).cpu() for a in outputs.attentions]
        
        return result
    
    @torch.no_grad()
    def extract_article(
        self, 
        article_text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all representations for a single article.
        
        Returns:
        - 'logits_raw': [n_bots, 3] raw logits per bot
        - 'logits_flat': [n_bots * 3] = [24] flattened logits
        - 'cls_per_bot': [n_bots, hidden_dim] CLS per bot (for Dirichlet)
        - 'cls_stacked': [n_bots * hidden_dim] stacked CLS (legacy compat)
        - 'head_attentions': dict of attention patterns (if record_heads)
        """
        # Handle paragraph-aware extraction
        if self.config.paragraph_aware:
            return self._extract_paragraph_aware(article_text)
        
        logits_list = []
        cls_list = []
        attentions_by_bot = []
        
        for bot_idx, (hyp_a, hyp_b) in enumerate(self.hypothesis_pairs):
            # Encode with hypothesis A
            enc_a = self._encode_single(article_text, hyp_a)
            # Encode with hypothesis B
            enc_b = self._encode_single(article_text, hyp_b)
            
            # Raw logits: concat A and B -> [6] or use contrast
            # Following original design: store [logits_a - logits_b] as the "judgment"
            # But for raw mode, we want the actual softmax inputs
            # Let's store both A and B logits
            logits_combined = torch.stack([enc_a['logits'], enc_b['logits']], dim=0)  # [2, 3]
            
            # For backward compat, compute contrast triplet
            probs_a = F.softmax(enc_a['logits'], dim=-1)
            probs_b = F.softmax(enc_b['logits'], dim=-1)
            score_a = probs_a[2] - probs_a[0]  # entail - contradict
            score_b = probs_b[2] - probs_b[0]
            contrast = score_a - score_b
            logits_triplet = torch.tensor([score_a, score_b, contrast])
            
            logits_list.append(logits_triplet)
            
            # CLS: use delta (A - B) as the bot's semantic signal
            cls_delta = enc_a['cls'] - enc_b['cls']
            cls_list.append(cls_delta)
            
            # Attention heads (if recording)
            if self.config.record_heads:
                attentions_by_bot.append({
                    'bot_idx': bot_idx,
                    'hyp_a': hyp_a[:50],  # Truncate for storage
                    'hyp_b': hyp_b[:50],
                    'attentions_a': enc_a.get('attentions'),
                    'attentions_b': enc_b.get('attentions'),
                })
        
        # Stack results
        logits_raw = torch.stack(logits_list, dim=0)  # [n_bots, 3]
        cls_per_bot = torch.stack(cls_list, dim=0)    # [n_bots, hidden]
        
        result = {
            'logits_raw': logits_raw,
            'logits_flat': logits_raw.reshape(-1),  # [24]
            'cls_per_bot': cls_per_bot,
            'cls_stacked': cls_per_bot.reshape(-1),  # [n_bots * hidden]
        }
        
        if self.config.record_heads:
            result['head_attentions'] = attentions_by_bot
        
        return result
    
    @torch.no_grad()
    def _extract_paragraph_aware(
        self, 
        article_text: str
    ) -> Dict[str, torch.Tensor]:
        """Paragraph-weighted extraction."""
        import re
        
        # Split into paragraphs
        text = article_text.strip()
        chunks = re.split(r'\n\s*\n+', text)
        paras = [c.strip() for c in chunks if c and c.strip()]
        
        max_paras = len(self.config.paragraph_weights)
        paras = paras[:max_paras] if paras else [text]
        
        # Normalize weights
        weights = self.config.paragraph_weights[:len(paras)]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights] if w_sum > 0 else [1.0 / len(paras)] * len(paras)
        
        # Accumulate weighted results
        logits_acc = None
        cls_acc = None
        
        for para, weight in zip(paras, weights):
            # Extract for this paragraph (non-paragraph-aware mode)
            self.config.paragraph_aware = False
            para_result = self.extract_article(para)
            self.config.paragraph_aware = True
            
            if logits_acc is None:
                logits_acc = para_result['logits_raw'] * weight
                cls_acc = para_result['cls_per_bot'] * weight
            else:
                logits_acc += para_result['logits_raw'] * weight
                cls_acc += para_result['cls_per_bot'] * weight
        
        return {
            'logits_raw': logits_acc,
            'logits_flat': logits_acc.reshape(-1),
            'cls_per_bot': cls_acc,
            'cls_stacked': cls_acc.reshape(-1),
        }
    
    def extract_batch(
        self, 
        articles: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all articles in batch.
        
        Returns:
        - 'logits_raw': [N, n_bots, 3]
        - 'logits_flat': [N, 24]
        - 'cls_per_bot': [N, n_bots, hidden]
        - 'cls_stacked': [N, n_bots * hidden]
        - 'head_attentions': list of per-article attention dicts (if record_heads)
        """
        all_logits = []
        all_cls = []
        all_attentions = []
        
        n = len(articles)
        for i, article in enumerate(articles):
            if show_progress and (i % 50 == 0 or i == n - 1):
                print(f"  Extracting: {i+1}/{n}")
            
            text = article.get('text', article.get('content', ''))
            if not text:
                # Empty article - use zeros
                all_logits.append(torch.zeros(self.n_bots, 3))
                all_cls.append(torch.zeros(self.n_bots, self.hidden_size))
                continue
            
            result = self.extract_article(text)
            all_logits.append(result['logits_raw'])
            all_cls.append(result['cls_per_bot'])
            
            if self.config.record_heads and 'head_attentions' in result:
                all_attentions.append(result['head_attentions'])
        
        # Stack
        logits_raw = torch.stack(all_logits, dim=0)  # [N, n_bots, 3]
        cls_per_bot = torch.stack(all_cls, dim=0)    # [N, n_bots, hidden]
        
        output = {
            'logits_raw': logits_raw,
            'logits_flat': logits_raw.reshape(n, -1),
            'cls_per_bot': cls_per_bot,
            'cls_stacked': cls_per_bot.reshape(n, -1),
            'n_articles': n,
            'n_bots': self.n_bots,
            'hidden_size': self.hidden_size,
        }
        
        if self.config.record_heads and all_attentions:
            output['head_attentions'] = all_attentions
        
        return output


class DirichletFusion(nn.Module):
    """
    Dirichlet observer fusion for CLS embeddings.
    
    Takes [N, n_bots, hidden] CLS embeddings and produces observer-weighted
    fused representations using shared RKS projection.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        n_bots: int = 8,
        rks_dim: int = 2048,
        n_observers: int = 50,
        alpha: float = 1.0,
        kernel_type: str = 'rbf',
        seed: int = 42,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_bots = n_bots
        self.rks_dim = rks_dim
        self.n_observers = n_observers
        self.alpha = alpha
        self.seed = seed
        
        # Shared RKS basis (fixed across all bots/articles)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer(
            'omega',
            torch.randn(hidden_dim, rks_dim, generator=gen)
        )
        self.register_buffer(
            'b',
            torch.rand(rks_dim, generator=gen) * 2 * np.pi
        )
        
        self._sigma = None
    
    def _estimate_sigma(self, X: torch.Tensor) -> float:
        """Estimate bandwidth from data."""
        with torch.no_grad():
            flat = X.reshape(-1, self.hidden_dim)
            n = min(500, flat.shape[0])
            idx = torch.randperm(flat.shape[0])[:n]
            X_sub = flat[idx]
            
            dists = torch.cdist(X_sub, X_sub)
            mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            
            if dists[mask].numel() > 0:
                sigma = float(torch.median(dists[mask]).item())
            else:
                sigma = 1.0
            
            return max(sigma, 1e-6)
    
    def _rks_project(self, X: torch.Tensor) -> torch.Tensor:
        """Project to RKHS using RKS."""
        if self._sigma is None:
            self._sigma = self._estimate_sigma(X)
            print(f"  [Dirichlet] Estimated sigma: {self._sigma:.4f}")
        
        omega_scaled = self.omega / self._sigma
        proj = X @ omega_scaled + self.b
        scale = np.sqrt(2.0 / self.rks_dim)
        return scale * torch.cos(proj)
    
    def forward(
        self,
        cls_per_bot: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse bot CLS embeddings via Dirichlet-weighted mixtures.
        
        Args:
            cls_per_bot: [N, n_bots, hidden_dim]
            
        Returns:
            - 'fused': [N, rks_dim] mean fused representation
            - 'fused_std': [N, rks_dim] std across observers
            - 'observer_samples': [N, n_observers, rks_dim] all samples
        """
        N, B, H = cls_per_bot.shape
        
        # Project all bots to RKHS
        flat = cls_per_bot.reshape(N * B, H)
        phi_flat = self._rks_project(flat)
        phi = phi_flat.reshape(N, B, -1)  # [N, B, rks_dim]
        
        D = phi.shape[-1]
        
        # Sample Dirichlet weights
        alpha_vec = torch.full((B,), self.alpha)
        dirichlet = torch.distributions.Dirichlet(alpha_vec)
        weights = dirichlet.sample((self.n_observers,)).to(phi.device)  # [K, B]
        
        # Compute fused representations
        # phi: [N, B, D], weights: [K, B] -> [N, K, D]
        observer_samples = torch.einsum('nbd,kb->nkd', phi, weights)
        
        fused_mean = observer_samples.mean(dim=1)
        fused_std = observer_samples.std(dim=1)
        
        return {
            'fused': fused_mean,
            'fused_std': fused_std,
            'observer_samples': observer_samples,
            'bot_rkhs': phi,
        }


def run_unified_pipeline(
    articles: List[Dict],
    config: UnifiedExtractionConfig = None,
    output_dir: Path = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete unified pipeline.
    
    Single inference pass produces:
    1. Raw logits (24D) - saved as-is
    2. Dirichlet-fused CLS - observer mixtures
    3. Head attentions (if enabled) - diagnostic sidecar
    
    Returns dict with all outputs, also saves to output_dir if provided.
    """
    if config is None:
        config = UnifiedExtractionConfig()
    
    print(f"\n{'='*70}")
    print("UNIFIED BELIEF TRANSFORMER PIPELINE")
    print(f"{'='*70}")
    print(f"  Articles: {len(articles)}")
    print(f"  Extract logits: {config.extract_logits}")
    print(f"  Extract CLS: {config.extract_cls}")
    print(f"  Record heads: {config.record_heads}")
    
    # Initialize extractor
    print("\n[1/3] Initializing extractor...")
    extractor = UnifiedNLIExtractor(config)
    
    # Extract all
    print("\n[2/3] Extracting representations...")
    extraction = extractor.extract_batch(articles)
    
    # Build output
    output = {
        'n_articles': extraction['n_articles'],
        'n_bots': extraction['n_bots'],
        'config': {
            'model_name': config.model_name,
            'max_length': config.max_length,
            'paragraph_aware': config.paragraph_aware,
            'record_heads': config.record_heads,
        },
    }
    
    # 1. Raw logits (no kernel, no projection)
    if config.extract_logits:
        output['logits_raw'] = extraction['logits_flat']  # [N, 24]
        output['logits_per_bot'] = extraction['logits_raw']  # [N, 8, 3]
        print(f"  Logits shape: {output['logits_raw'].shape}")
    
    # 2. CLS with Dirichlet fusion
    if config.extract_cls:
        print("\n[3/3] Running Dirichlet fusion...")
        fusion = DirichletFusion(
            hidden_dim=extraction['hidden_size'],
            n_bots=extraction['n_bots'],
            rks_dim=config.dirichlet_rks_dim,
            n_observers=config.dirichlet_n_observers,
            alpha=config.dirichlet_alpha,
            seed=config.dirichlet_seed,
        )
        
        cls_per_bot = extraction['cls_per_bot']
        fusion_result = fusion(cls_per_bot)
        
        output['cls_per_bot'] = cls_per_bot  # [N, 8, hidden]
        output['cls_fused'] = fusion_result['fused']  # [N, rks_dim]
        output['cls_fused_std'] = fusion_result['fused_std']
        output['cls_observer_samples'] = fusion_result['observer_samples']
        
        print(f"  CLS per bot shape: {cls_per_bot.shape}")
        print(f"  Fused CLS shape: {output['cls_fused'].shape}")
    
    # 3. Head attentions (diagnostic sidecar)
    if config.record_heads and 'head_attentions' in extraction:
        output['head_attentions'] = extraction['head_attentions']
        print(f"  Recorded attention heads for {len(extraction['head_attentions'])} articles")
    
    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main artifact
        artifact = {
            'embeddings': output.get('cls_fused', output.get('logits_raw')),
            'features': output.get('cls_fused', output.get('logits_raw')),
            'logits_raw': output.get('logits_raw'),
            'cls_fused': output.get('cls_fused'),
            'cls_fused_std': output.get('cls_fused_std'),
            'cls_per_bot': output.get('cls_per_bot'),
            'seed': seed,
            'n_articles': output['n_articles'],
            'meta': output['config'],
        }
        
        torch.save(artifact, output_dir / f"observer_{seed}.pt")
        print(f"\n[OK] Saved: {output_dir / f'observer_{seed}.pt'}")
        
        # Head attentions as separate file (can be large)
        if 'head_attentions' in output:
            torch.save(
                {'head_attentions': output['head_attentions']},
                output_dir / f"head_attentions_{seed}.pt"
            )
            print(f"[OK] Saved: {output_dir / f'head_attentions_{seed}.pt'}")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    
    return output
