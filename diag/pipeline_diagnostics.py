"""
Pipeline Diagnostics - Standalone Sidecar for Observer Analysis

Run this AFTER inference to analyze saved outputs for:
1. Observer collapse (all observers producing similar outputs)
2. Normalization issues (sigma, embedding scale)
3. Bot divergence (do the 8 bots actually differ?)
4. Curvature anomalies (flat vs structured semantic space)

LOCATION: D:\\belief-transformer\\V3\\diag\\pipeline_diagnostics.py

Usage:
    cd D:\\belief-transformer\\V3\\diag
    python pipeline_diagnostics.py --input ../outputs/observer_42.pt
    python pipeline_diagnostics.py --input ../outputs --all  # Analyze all .pt files
    python pipeline_diagnostics.py --input ../outputs/observer_42.pt --save-report
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StageStats:
    """Statistics for a tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    
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


@dataclass 
class BotDivergenceStats:
    """Statistics measuring how different the 8 bots are."""
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
class CurvatureStats:
    """Statistics from curvature metrics."""
    n_articles: int
    
    # Participation ratio (effective dimensionality)
    pr_mean: float
    pr_std: float
    pr_min: float
    pr_max: float
    
    # Effective rank at 90%
    er90_mean: float
    er90_std: float
    
    # Lambda ratio (dominance)
    lr_mean: float
    lr_median: float


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a saved artifact."""
    timestamp: str
    input_file: str
    
    # Basic info
    n_articles: int
    embedding_dim: int
    
    # Stats
    fused_stats: Optional[StageStats]
    fused_std_stats: Optional[StageStats]
    bot_rkhs_stats: Optional[StageStats]
    
    # Analysis
    bot_divergence: Optional[BotDivergenceStats]
    curvature_stats: Optional[CurvatureStats]
    
    # Provenance
    provenance: Dict[str, Any]
    
    # Alerts
    alerts: List[str]
    
    # Summary
    health_score: float  # 0-1, higher is better


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_tensor_stats(tensor: torch.Tensor, name: str) -> StageStats:
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
        
        return StageStats(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
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
        )


def analyze_bot_divergence(bot_rkhs: torch.Tensor) -> BotDivergenceStats:
    """
    Analyze how different the 8 bots are.
    
    Args:
        bot_rkhs: [N, B, D] where B=8 bots
    """
    N, B, D = bot_rkhs.shape
    
    with torch.no_grad():
        tensor = bot_rkhs.float()
        
        # Per-bot means: [B, D]
        bot_means = tensor.mean(dim=0)
        
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
        inter_bot_var = tensor.var(dim=1).mean()  # [N, D] -> scalar
        
        # Per-bot norms
        bot_norms_per_article = tensor.norm(dim=-1)  # [N, B]
        bot_norm_means = bot_norms_per_article.mean(dim=0).tolist()
        bot_norm_stds = bot_norms_per_article.std(dim=0).tolist()
        
        distinct = min_cos < 0.99
        
        return BotDivergenceStats(
            n_bots=B,
            mean_pairwise_cosine=mean_cos,
            min_pairwise_cosine=min_cos,
            max_pairwise_cosine=max_cos,
            mean_inter_bot_variance=float(inter_bot_var.item()),
            bot_means_distinct=distinct,
            bot_norm_means=bot_norm_means,
            bot_norm_stds=bot_norm_stds,
        )


def analyze_curvature(curvature: Dict[str, torch.Tensor]) -> CurvatureStats:
    """Analyze curvature metrics."""
    pr = curvature.get('participation_ratio')
    er = curvature.get('effective_rank_90')
    lr = curvature.get('lambda1_lambda2')
    
    if pr is None:
        return None
    
    pr = pr.float()
    n = pr.shape[0]
    
    stats = CurvatureStats(
        n_articles=n,
        pr_mean=float(pr.mean().item()),
        pr_std=float(pr.std().item()),
        pr_min=float(pr.min().item()),
        pr_max=float(pr.max().item()),
        er90_mean=float(er.float().mean().item()) if er is not None else 0,
        er90_std=float(er.float().std().item()) if er is not None else 0,
        lr_mean=float(lr.mean().item()) if lr is not None else 0,
        lr_median=float(lr.median().item()) if lr is not None else 0,
    )
    
    return stats


def compute_health_score(
    fused_stats: StageStats,
    bot_divergence: Optional[BotDivergenceStats],
    alerts: List[str],
) -> float:
    """
    Compute overall health score (0-1).
    
    Factors:
    - No NaN/Inf: +0.3
    - Reasonable std: +0.2
    - Bots divergent: +0.3
    - No critical alerts: +0.2
    """
    score = 0.0
    
    # NaN/Inf check
    if not fused_stats.has_nan and not fused_stats.has_inf:
        score += 0.3
    
    # Std check
    if fused_stats.std > 1e-6:
        score += 0.2
    
    # Bot divergence
    if bot_divergence and bot_divergence.bot_means_distinct:
        score += 0.3
    elif bot_divergence is None:
        score += 0.15  # Can't check
    
    # Alerts
    critical_alerts = [a for a in alerts if 'CRITICAL' in a or 'collapse' in a.lower()]
    if len(critical_alerts) == 0:
        score += 0.2
    
    return min(score, 1.0)


# =============================================================================
# MAIN DIAGNOSTIC FUNCTION
# =============================================================================

def analyze_artifact(filepath: str) -> DiagnosticReport:
    """
    Analyze a saved observer artifact (.pt file).
    
    Args:
        filepath: Path to observer_XX.pt file
        
    Returns:
        DiagnosticReport with full analysis
    """
    filepath = Path(filepath)
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filepath.name}")
    print(f"{'='*60}")
    
    # Load artifact
    data = torch.load(filepath, map_location='cpu')
    
    alerts = []
    
    # Extract tensors
    fused = data.get('fused') or data.get('embeddings') or data.get('features')
    fused_std = data.get('fused_std')
    bot_rkhs = data.get('bot_rkhs') or data.get('rkhs_views')
    curvature = data.get('curvature')
    provenance = data.get('meta', {}).get('provenance', {})
    
    if fused is None:
        print("ERROR: No fused embeddings found in artifact")
        return None
    
    n_articles = fused.shape[0]
    embedding_dim = fused.shape[-1]
    
    print(f"\nBasic Info:")
    print(f"  Articles: {n_articles}")
    print(f"  Embedding dim: {embedding_dim}")
    
    # Analyze fused
    print(f"\n--- Fused Embeddings ---")
    fused_stats = compute_tensor_stats(fused, 'fused')
    print(f"  Shape: {fused_stats.shape}")
    print(f"  Mean: {fused_stats.mean:.6f}")
    print(f"  Std: {fused_stats.std:.6f}")
    print(f"  Norm: {fused_stats.norm_mean:.4f} ± {fused_stats.norm_std:.4f}")
    
    if fused_stats.has_nan:
        alerts.append(f"CRITICAL: fused has {fused_stats.nan_count} NaN values!")
        print(f"  ⚠️  NaN count: {fused_stats.nan_count}")
    if fused_stats.has_inf:
        alerts.append(f"CRITICAL: fused has {fused_stats.inf_count} Inf values!")
        print(f"  ⚠️  Inf count: {fused_stats.inf_count}")
    if fused_stats.std < 1e-6:
        alerts.append("WARNING: fused has near-zero std - possible collapse")
        print(f"  ⚠️  Near-zero std!")
    
    # Analyze fused_std
    fused_std_stats = None
    if fused_std is not None:
        print(f"\n--- Fused Std (Observer Variance) ---")
        fused_std_stats = compute_tensor_stats(fused_std, 'fused_std')
        print(f"  Mean observer std: {fused_std_stats.mean:.6f}")
        print(f"  Std of observer std: {fused_std_stats.std:.6f}")
        
        if fused_std_stats.mean < 1e-6:
            alerts.append("WARNING: Very low observer variance - possible observer collapse")
            print(f"  ⚠️  Low observer variance!")
    
    # Analyze bot_rkhs
    bot_rkhs_stats = None
    bot_divergence = None
    if bot_rkhs is not None and bot_rkhs.dim() == 3:
        print(f"\n--- Bot RKHS (V-Observers) ---")
        bot_rkhs_stats = compute_tensor_stats(bot_rkhs, 'bot_rkhs')
        print(f"  Shape: {bot_rkhs_stats.shape}")
        print(f"  Norm: {bot_rkhs_stats.norm_mean:.4f} ± {bot_rkhs_stats.norm_std:.4f}")
        
        bot_divergence = analyze_bot_divergence(bot_rkhs)
        status = "✓ DIVERSE" if bot_divergence.bot_means_distinct else "✗ COLLAPSED"
        print(f"\n  Bot Divergence: {status}")
        print(f"  Pairwise cosine: {bot_divergence.min_pairwise_cosine:.4f} - {bot_divergence.max_pairwise_cosine:.4f}")
        print(f"  Inter-bot variance: {bot_divergence.mean_inter_bot_variance:.6f}")
        
        if not bot_divergence.bot_means_distinct:
            alerts.append(f"WARNING: Bot collapse detected (min cosine = {bot_divergence.min_pairwise_cosine:.4f})")
    
    # Analyze curvature
    curvature_stats = None
    if curvature is not None:
        print(f"\n--- Curvature Metrics ---")
        curvature_stats = analyze_curvature(curvature)
        if curvature_stats:
            print(f"  Participation ratio: {curvature_stats.pr_mean:.2f} ± {curvature_stats.pr_std:.2f}")
            print(f"  Effective rank (90%): {curvature_stats.er90_mean:.1f} ± {curvature_stats.er90_std:.1f}")
            print(f"  Lambda ratio (median): {curvature_stats.lr_median:.2f}")
            
            if curvature_stats.pr_mean < 2:
                alerts.append("WARNING: Low participation ratio - semantic space may be 1D")
    
    # Provenance
    if provenance:
        print(f"\n--- Provenance ---")
        for k, v in provenance.items():
            if not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")
    
    # Health score
    health_score = compute_health_score(fused_stats, bot_divergence, alerts)
    
    # Alerts
    if alerts:
        print(f"\n--- ALERTS ---")
        for alert in alerts:
            print(f"  ⚠️  {alert}")
    
    print(f"\n{'='*60}")
    print(f"HEALTH SCORE: {health_score:.2f}/1.00")
    print(f"{'='*60}\n")
    
    return DiagnosticReport(
        timestamp=datetime.now().isoformat(),
        input_file=str(filepath),
        n_articles=n_articles,
        embedding_dim=embedding_dim,
        fused_stats=fused_stats,
        fused_std_stats=fused_std_stats,
        bot_rkhs_stats=bot_rkhs_stats,
        bot_divergence=bot_divergence,
        curvature_stats=curvature_stats,
        provenance=provenance,
        alerts=alerts,
        health_score=health_score,
    )


def save_report(report: DiagnosticReport, output_path: str):
    """Save report to JSON."""
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        else:
            return obj
    
    with open(output_path, 'w') as f:
        json.dump(to_dict(report), f, indent=2)
    
    print(f"Report saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Belief Transformer pipeline outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_diagnostics.py --input ../outputs/observer_42.pt
  python pipeline_diagnostics.py --input ../outputs --all
  python pipeline_diagnostics.py --input ../outputs/observer_42.pt --save-report
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to .pt file or directory containing .pt files'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze all .pt files in directory'
    )
    parser.add_argument(
        '--save-report', '-s',
        action='store_true',
        help='Save JSON report alongside each analyzed file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Directory for reports (default: same as input)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        if args.all:
            files = list(input_path.glob('*.pt'))
        else:
            files = list(input_path.glob('observer_*.pt'))
        
        if not files:
            print(f"No .pt files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(files)} files to analyze")
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)
    
    # Analyze each file
    reports = []
    for f in sorted(files):
        try:
            report = analyze_artifact(str(f))
            if report:
                reports.append(report)
                
                if args.save_report:
                    out_dir = Path(args.output_dir) if args.output_dir else f.parent
                    out_path = out_dir / f"{f.stem}_diagnostics.json"
                    save_report(report, str(out_path))
        except Exception as e:
            print(f"Error analyzing {f}: {e}")
    
    # Summary
    if len(reports) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Files analyzed: {len(reports)}")
        
        scores = [r.health_score for r in reports]
        print(f"Health scores: {min(scores):.2f} - {max(scores):.2f} (mean: {np.mean(scores):.2f})")
        
        all_alerts = []
        for r in reports:
            all_alerts.extend(r.alerts)
        
        if all_alerts:
            print(f"\nTotal alerts: {len(all_alerts)}")
            # Count unique alerts
            from collections import Counter
            alert_counts = Counter(all_alerts)
            for alert, count in alert_counts.most_common(5):
                print(f"  [{count}x] {alert}")


if __name__ == '__main__':
    main()