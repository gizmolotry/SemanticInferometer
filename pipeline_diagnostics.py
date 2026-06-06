"""
Pipeline Diagnostics - Observer Collapse and Normalization Analysis

Tracks representations across pipeline stages to detect:
1. Observer collapse (all observers producing similar outputs)
2. Normalization issues (sigma too small/large, embedding scale drift)
3. Bot divergence (do the 8 bots actually differ?)
4. Curvature anomalies (flat vs structured semantic space)

Usage:
    from pipeline_diagnostics import PipelineDiagnostics
    
    diag = PipelineDiagnostics()
    diag.record_stage('raw_cls', cls_per_bot)
    diag.record_stage('rkhs_projected', phi)
    diag.record_stage('fused', fused_mean)
    
    report = diag.generate_report()
    diag.save_report('diagnostics.json')
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import numpy as np


@dataclass
class StageStats:
    """Statistics for a single pipeline stage."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    
    # Basic stats
    mean: float
    std: float
    min: float
    max: float
    
    # Norm stats
    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float
    
    # NaN/Inf check
    has_nan: bool
    has_inf: bool
    nan_count: int
    inf_count: int
    
    # Distribution stats
    percentiles: Dict[str, float]  # 1, 5, 25, 50, 75, 95, 99
    
    # Computed metrics (optional)
    effective_rank: Optional[float] = None
    condition_number: Optional[float] = None


@dataclass 
class BotDivergenceStats:
    """Statistics measuring how different the 8 bots are."""
    stage: str
    n_bots: int
    
    # Pairwise cosine similarities between bot means
    mean_pairwise_cosine: float
    min_pairwise_cosine: float
    max_pairwise_cosine: float
    
    # Variance across bots (per article, then averaged)
    mean_inter_bot_variance: float
    
    # Are bots distinguishable?
    bot_means_distinct: bool  # True if min_pairwise_cosine < 0.99
    
    # Per-bot norms
    bot_norm_means: List[float]
    bot_norm_stds: List[float]


@dataclass
class ObserverCollapseStats:
    """Statistics measuring observer collapse (all observers same)."""
    stage: str
    n_observers: int
    
    # Variance across observers (per article)
    mean_inter_observer_variance: float
    min_inter_observer_variance: float
    max_inter_observer_variance: float
    
    # Pairwise distances between observer outputs
    mean_pairwise_distance: float
    
    # Collapse detection
    collapsed: bool  # True if variance < threshold
    collapse_ratio: float  # How collapsed (0 = diverse, 1 = identical)


@dataclass
class SigmaAnalysis:
    """Analysis of RKS bandwidth (sigma) estimation."""
    estimated_sigma: float
    
    # Distance distribution that sigma was estimated from
    dist_min: float
    dist_median: float
    dist_mean: float
    dist_max: float
    dist_std: float
    
    # Sigma quality assessment
    sigma_quality: str  # 'good', 'too_small', 'too_large'
    recommended_sigma: Optional[float] = None


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a pipeline run."""
    timestamp: str
    pipeline_id: str
    
    # Per-stage stats
    stages: Dict[str, StageStats]
    
    # Cross-stage analysis
    bot_divergence: Dict[str, BotDivergenceStats]
    observer_collapse: Optional[ObserverCollapseStats]
    sigma_analysis: Optional[SigmaAnalysis]
    
    # Alerts
    alerts: List[str]
    
    # Summary
    health_score: float  # 0-1, higher is better
    

class PipelineDiagnostics:
    """
    Diagnostic recorder for Belief Transformer pipeline.
    
    Records tensors at various stages and computes health metrics.
    """
    
    def __init__(self, pipeline_id: Optional[str] = None):
        self.pipeline_id = pipeline_id or self._generate_id()
        self.stages: Dict[str, torch.Tensor] = {}
        self.stage_stats: Dict[str, StageStats] = {}
        self.metadata: Dict[str, Any] = {}
        self.alerts: List[str] = []
        
        self._sigma_distances: Optional[torch.Tensor] = None
        self._estimated_sigma: Optional[float] = None
    
    def _generate_id(self) -> str:
        """Generate unique pipeline ID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(str(id(self)).encode()).hexdigest()[:6]
        return f"diag_{ts}_{h}"
    
    def record_stage(
        self, 
        name: str, 
        tensor: torch.Tensor,
        subsample: int = 5000,
        compute_rank: bool = False,
    ):
        """
        Record a tensor at a pipeline stage.
        
        Args:
            name: Stage name (e.g., 'raw_cls', 'rkhs_projected', 'fused')
            tensor: The tensor to record
            subsample: Max samples for expensive computations
            compute_rank: Whether to compute effective rank (expensive)
        """
        # Store subsample for later analysis
        if tensor.numel() > subsample * tensor.shape[-1]:
            idx = torch.randperm(tensor.shape[0])[:subsample]
            stored = tensor[idx].detach().cpu()
        else:
            stored = tensor.detach().cpu()
        
        self.stages[name] = stored
        
        # Compute stats
        stats = self._compute_stage_stats(name, tensor, compute_rank)
        self.stage_stats[name] = stats
        
        # Check for issues
        self._check_stage_health(name, stats)
    
    def record_sigma_estimation(
        self,
        distances: torch.Tensor,
        estimated_sigma: float,
    ):
        """Record the distance distribution used for sigma estimation."""
        self._sigma_distances = distances.detach().cpu()
        self._estimated_sigma = estimated_sigma
    
    def record_metadata(self, key: str, value: Any):
        """Record arbitrary metadata."""
        self.metadata[key] = value
    
    def _compute_stage_stats(
        self, 
        name: str, 
        tensor: torch.Tensor,
        compute_rank: bool = False,
    ) -> StageStats:
        """Compute comprehensive statistics for a tensor."""
        with torch.no_grad():
            flat = tensor.reshape(-1).float()
            
            # Basic stats
            mean = float(flat.mean().item())
            std = float(flat.std().item())
            min_val = float(flat.min().item())
            max_val = float(flat.max().item())
            
            # NaN/Inf
            has_nan = bool(torch.isnan(flat).any().item())
            has_inf = bool(torch.isinf(flat).any().item())
            nan_count = int(torch.isnan(flat).sum().item())
            inf_count = int(torch.isinf(flat).sum().item())
            
            # Norms (along last dim)
            if tensor.dim() >= 2:
                norms = tensor.reshape(-1, tensor.shape[-1]).float().norm(dim=-1)
            else:
                norms = tensor.float().abs()
            
            norm_mean = float(norms.mean().item())
            norm_std = float(norms.std().item())
            norm_min = float(norms.min().item())
            norm_max = float(norms.max().item())
            
            # Percentiles
            percentiles = {}
            for p in [1, 5, 25, 50, 75, 95, 99]:
                percentiles[f'p{p}'] = float(torch.quantile(flat, p/100).item())
            
            # Effective rank (expensive)
            effective_rank = None
            condition_number = None
            if compute_rank and tensor.dim() >= 2:
                try:
                    # Subsample for efficiency
                    sub = tensor.reshape(-1, tensor.shape[-1])[:1000].float()
                    _, S, _ = torch.linalg.svd(sub, full_matrices=False)
                    S = S.clamp(min=1e-10)
                    
                    # Participation ratio as effective rank
                    total = S.sum()
                    effective_rank = float(((total ** 2) / (S ** 2).sum()).item())
                    
                    # Condition number
                    condition_number = float((S[0] / S[-1]).item())
                except Exception:
                    pass
            
            return StageStats(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                mean=mean,
                std=std,
                min=min_val,
                max=max_val,
                norm_mean=norm_mean,
                norm_std=norm_std,
                norm_min=norm_min,
                norm_max=norm_max,
                has_nan=has_nan,
                has_inf=has_inf,
                nan_count=nan_count,
                inf_count=inf_count,
                percentiles=percentiles,
                effective_rank=effective_rank,
                condition_number=condition_number,
            )
    
    def _check_stage_health(self, name: str, stats: StageStats):
        """Check for common issues and add alerts."""
        if stats.has_nan:
            self.alerts.append(f"[{name}] Contains {stats.nan_count} NaN values!")
        
        if stats.has_inf:
            self.alerts.append(f"[{name}] Contains {stats.inf_count} Inf values!")
        
        if stats.std < 1e-6:
            self.alerts.append(f"[{name}] Near-zero std ({stats.std:.2e}) - possible collapse")
        
        if stats.norm_std / (stats.norm_mean + 1e-10) < 0.01:
            self.alerts.append(f"[{name}] Very uniform norms - possible over-normalization")
        
        if stats.condition_number and stats.condition_number > 1e6:
            self.alerts.append(f"[{name}] High condition number ({stats.condition_number:.2e})")
    
    def analyze_bot_divergence(self, stage_name: str = 'raw_cls') -> Optional[BotDivergenceStats]:
        """
        Analyze how different the 8 bots are.
        
        Expects tensor shape [N, B, H] where B=8 bots.
        """
        if stage_name not in self.stages:
            return None
        
        tensor = self.stages[stage_name]
        if tensor.dim() != 3:
            return None
        
        N, B, H = tensor.shape
        
        with torch.no_grad():
            # Per-bot means: [B, H]
            bot_means = tensor.float().mean(dim=0)
            
            # Pairwise cosine similarities
            bot_norms = bot_means.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            bot_normalized = bot_means / bot_norms
            cosine_sim = bot_normalized @ bot_normalized.T  # [B, B]
            
            # Get off-diagonal elements
            mask = ~torch.eye(B, dtype=torch.bool)
            pairwise_cosines = cosine_sim[mask]
            
            mean_cos = float(pairwise_cosines.mean().item())
            min_cos = float(pairwise_cosines.min().item())
            max_cos = float(pairwise_cosines.max().item())
            
            # Inter-bot variance per article
            inter_bot_var = tensor.float().var(dim=1).mean()  # [N, H] -> scalar
            
            # Per-bot norms
            bot_norms_per_article = tensor.float().norm(dim=-1)  # [N, B]
            bot_norm_means = bot_norms_per_article.mean(dim=0).tolist()
            bot_norm_stds = bot_norms_per_article.std(dim=0).tolist()
            
            distinct = min_cos < 0.99
            
            stats = BotDivergenceStats(
                stage=stage_name,
                n_bots=B,
                mean_pairwise_cosine=mean_cos,
                min_pairwise_cosine=min_cos,
                max_pairwise_cosine=max_cos,
                mean_inter_bot_variance=float(inter_bot_var.item()),
                bot_means_distinct=distinct,
                bot_norm_means=bot_norm_means,
                bot_norm_stds=bot_norm_stds,
            )
            
            if not distinct:
                self.alerts.append(
                    f"[{stage_name}] Bot collapse detected! "
                    f"Min pairwise cosine = {min_cos:.4f}"
                )
            
            return stats
    
    def analyze_observer_collapse(
        self, 
        observer_samples: torch.Tensor,
        stage_name: str = 'observer_samples',
    ) -> ObserverCollapseStats:
        """
        Analyze observer collapse from Dirichlet samples.
        
        Args:
            observer_samples: [N, K, D] where K is number of observers
        """
        N, K, D = observer_samples.shape
        
        with torch.no_grad():
            samples = observer_samples.float()
            
            # Inter-observer variance per article
            inter_obs_var = samples.var(dim=1)  # [N, D]
            mean_var_per_article = inter_obs_var.mean(dim=-1)  # [N]
            
            mean_var = float(mean_var_per_article.mean().item())
            min_var = float(mean_var_per_article.min().item())
            max_var = float(mean_var_per_article.max().item())
            
            # Pairwise distances between observers (subsample)
            sub_n = min(100, N)
            sub_k = min(10, K)
            sub = samples[:sub_n, :sub_k]  # [sub_n, sub_k, D]
            
            # Flatten to [sub_n * sub_k, D] and compute distances
            flat = sub.reshape(-1, D)
            dists = torch.cdist(flat, flat)
            mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            mean_dist = float(dists[mask].mean().item()) if dists[mask].numel() > 0 else 0.0
            
            # Collapse detection
            # If variance is very low relative to the mean embedding norm
            mean_norm = samples.norm(dim=-1).mean()
            collapse_ratio = 1.0 - (mean_var / (mean_norm ** 2 + 1e-10)).clamp(0, 1).item()
            collapsed = collapse_ratio > 0.99
            
            stats = ObserverCollapseStats(
                stage=stage_name,
                n_observers=K,
                mean_inter_observer_variance=mean_var,
                min_inter_observer_variance=min_var,
                max_inter_observer_variance=max_var,
                mean_pairwise_distance=mean_dist,
                collapsed=collapsed,
                collapse_ratio=collapse_ratio,
            )
            
            if collapsed:
                self.alerts.append(
                    f"[{stage_name}] Observer collapse detected! "
                    f"Collapse ratio = {collapse_ratio:.4f}"
                )
            
            return stats
    
    def analyze_sigma(self) -> Optional[SigmaAnalysis]:
        """Analyze sigma estimation quality."""
        if self._sigma_distances is None or self._estimated_sigma is None:
            return None
        
        dists = self._sigma_distances.float()
        sigma = self._estimated_sigma
        
        dist_min = float(dists.min().item())
        dist_median = float(torch.median(dists).item())
        dist_mean = float(dists.mean().item())
        dist_max = float(dists.max().item())
        dist_std = float(dists.std().item())
        
        # Quality assessment
        # Good sigma should be close to median distance
        if sigma < dist_median * 0.1:
            quality = 'too_small'
            recommended = dist_median
        elif sigma > dist_median * 10:
            quality = 'too_large'
            recommended = dist_median
        else:
            quality = 'good'
            recommended = None
        
        analysis = SigmaAnalysis(
            estimated_sigma=sigma,
            dist_min=dist_min,
            dist_median=dist_median,
            dist_mean=dist_mean,
            dist_max=dist_max,
            dist_std=dist_std,
            sigma_quality=quality,
            recommended_sigma=recommended,
        )
        
        if quality != 'good':
            self.alerts.append(
                f"[sigma] Quality: {quality}. "
                f"Estimated: {sigma:.4f}, Recommended: {recommended:.4f}"
            )
        
        return analysis
    
    def compute_health_score(self) -> float:
        """
        Compute overall pipeline health score (0-1).
        
        Factors:
        - No NaN/Inf: +0.3
        - Reasonable stds: +0.2
        - Bots divergent: +0.2
        - Observers not collapsed: +0.2
        - Good sigma: +0.1
        """
        score = 0.0
        
        # NaN/Inf check
        has_bad_values = any(s.has_nan or s.has_inf for s in self.stage_stats.values())
        if not has_bad_values:
            score += 0.3
        
        # Std check
        has_low_std = any(s.std < 1e-6 for s in self.stage_stats.values())
        if not has_low_std:
            score += 0.2
        
        # Bot divergence (check if recorded)
        bot_stats = self.analyze_bot_divergence('raw_cls')
        if bot_stats is None:
            bot_stats = self.analyze_bot_divergence('rkhs_projected')
        if bot_stats and bot_stats.bot_means_distinct:
            score += 0.2
        elif bot_stats is None:
            score += 0.1  # Can't check, give partial credit
        
        # Observer collapse (would need observer_samples)
        # Skip for now, give partial credit
        score += 0.1
        
        # Sigma
        sigma_analysis = self.analyze_sigma()
        if sigma_analysis and sigma_analysis.sigma_quality == 'good':
            score += 0.1
        elif sigma_analysis is None:
            score += 0.05
        
        return min(score, 1.0)
    
    def generate_report(self) -> DiagnosticReport:
        """Generate complete diagnostic report."""
        # Compute all analyses
        bot_divergence = {}
        for stage_name in ['raw_cls', 'rkhs_projected']:
            if stage_name in self.stages:
                stats = self.analyze_bot_divergence(stage_name)
                if stats:
                    bot_divergence[stage_name] = stats
        
        sigma_analysis = self.analyze_sigma()
        health_score = self.compute_health_score()
        
        return DiagnosticReport(
            timestamp=datetime.now().isoformat(),
            pipeline_id=self.pipeline_id,
            stages=self.stage_stats,
            bot_divergence=bot_divergence,
            observer_collapse=None,  # Set externally if needed
            sigma_analysis=sigma_analysis,
            alerts=self.alerts,
            health_score=health_score,
        )
    
    def save_report(self, path: str):
        """Save report to JSON."""
        report = self.generate_report()
        
        # Convert dataclasses to dicts
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_dict(v) for v in obj]
            else:
                return obj
        
        with open(path, 'w') as f:
            json.dump(to_dict(report), f, indent=2)
        
        print(f"[Diagnostics] Report saved to {path}")
    
    def print_summary(self):
        """Print a quick summary to console."""
        print("\n" + "=" * 60)
        print("PIPELINE DIAGNOSTICS SUMMARY")
        print("=" * 60)
        
        print(f"\nPipeline ID: {self.pipeline_id}")
        print(f"Stages recorded: {list(self.stages.keys())}")
        
        # Stage summaries
        print("\n--- Stage Statistics ---")
        for name, stats in self.stage_stats.items():
            print(f"\n{name}:")
            print(f"  Shape: {stats.shape}")
            print(f"  Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")
            print(f"  Norm: {stats.norm_mean:.4f} ± {stats.norm_std:.4f}")
            if stats.has_nan or stats.has_inf:
                print(f"  ⚠️  NaN: {stats.nan_count}, Inf: {stats.inf_count}")
        
        # Bot divergence
        print("\n--- Bot Divergence ---")
        for stage_name in ['raw_cls', 'rkhs_projected']:
            if stage_name in self.stages:
                stats = self.analyze_bot_divergence(stage_name)
                if stats:
                    status = "✓ DIVERSE" if stats.bot_means_distinct else "✗ COLLAPSED"
                    print(f"{stage_name}: {status}")
                    print(f"  Pairwise cosine: {stats.min_pairwise_cosine:.4f} - {stats.max_pairwise_cosine:.4f}")
        
        # Sigma
        sigma = self.analyze_sigma()
        if sigma:
            print("\n--- Sigma Analysis ---")
            print(f"  Estimated: {sigma.estimated_sigma:.4f}")
            print(f"  Distance median: {sigma.dist_median:.4f}")
            print(f"  Quality: {sigma.sigma_quality}")
        
        # Alerts
        if self.alerts:
            print("\n--- ALERTS ---")
            for alert in self.alerts:
                print(f"  ⚠️  {alert}")
        
        # Health score
        score = self.compute_health_score()
        print(f"\n--- Health Score: {score:.2f}/1.00 ---")
        print("=" * 60 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_diagnose(
    cls_per_bot: torch.Tensor,
    fused: Optional[torch.Tensor] = None,
    bot_rkhs: Optional[torch.Tensor] = None,
    sigma: Optional[float] = None,
    sigma_distances: Optional[torch.Tensor] = None,
) -> PipelineDiagnostics:
    """
    Quick diagnostic check for common pipeline outputs.
    
    Args:
        cls_per_bot: [N, 8, H] raw CLS embeddings
        fused: [N, D] fused output (optional)
        bot_rkhs: [N, 8, D] RKHS-projected bots (optional)
        sigma: Estimated bandwidth (optional)
        sigma_distances: Distance distribution used for sigma (optional)
    
    Returns:
        PipelineDiagnostics with analysis
    """
    diag = PipelineDiagnostics()
    
    diag.record_stage('raw_cls', cls_per_bot)
    
    if bot_rkhs is not None:
        diag.record_stage('rkhs_projected', bot_rkhs)
    
    if fused is not None:
        diag.record_stage('fused', fused)
    
    if sigma is not None and sigma_distances is not None:
        diag.record_sigma_estimation(sigma_distances, sigma)
    
    diag.print_summary()
    
    return diag


def check_for_collapse(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Quick check if a tensor shows signs of collapse.
    
    Returns True if collapsed (bad), False if diverse (good).
    """
    with torch.no_grad():
        std = tensor.float().std().item()
        if std < 1e-6:
            print(f"[COLLAPSE] {name} has near-zero std: {std:.2e}")
            return True
        
        # Check if all rows are similar
        if tensor.dim() >= 2:
            norms = tensor.reshape(-1, tensor.shape[-1]).float().norm(dim=-1)
            norm_cv = norms.std() / (norms.mean() + 1e-10)
            if norm_cv < 0.01:
                print(f"[COLLAPSE] {name} has uniform norms (CV={norm_cv:.4f})")
                return True
        
        return False
