"""
Enhanced Cross-Article Attention with Observer Diversity
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossArticleAttention(nn.Module):
    """Cross-article attention with architectural diversity parameters."""
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.top_k = top_k
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        print(f"CrossArticleAttention initialized:")
        print(f"  - Heads: {num_heads}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Top-k: {top_k if top_k else 'None (dense)'}")
    
    def forward(self, article_tokens: torch.Tensor) -> torch.Tensor:
        N = article_tokens.shape[0]
        tokens = article_tokens.unsqueeze(0)
        
        # Temperature scaling
        scaled_tokens = tokens / self.temperature
        
        # Attention
        attn_output, attn_weights = self.attention(
            scaled_tokens, tokens, tokens,
            need_weights=True,
            average_attn_weights=True
        )
        
        attn_matrix = attn_weights.squeeze(0)
        
        # Sparsity
        if self.top_k is not None and self.top_k < N:
            attn_matrix = self._apply_topk_sparsity(attn_matrix, self.top_k)
        
        return attn_matrix
    
    def _apply_topk_sparsity(self, attn_matrix: torch.Tensor, k: int) -> torch.Tensor:
        N = attn_matrix.shape[0]
        top_k_values, top_k_indices = torch.topk(attn_matrix, k=k, dim=-1)
        sparse_matrix = torch.zeros_like(attn_matrix)
        for i in range(N):
            sparse_matrix[i, top_k_indices[i]] = top_k_values[i]
        row_sums = sparse_matrix.sum(dim=-1, keepdim=True)
        sparse_matrix = sparse_matrix / (row_sums + 1e-10)
        return sparse_matrix


def create_observer_configs(mode: str = 'diverse') -> dict:
    """Create observer configurations."""
    
    if mode == 'baseline':
        return {
            42: {'num_heads': 8, 'temperature': 1.0, 'top_k': None, 'description': 'Baseline'},
            43: {'num_heads': 8, 'temperature': 1.0, 'top_k': None, 'description': 'Baseline'},
            44: {'num_heads': 8, 'temperature': 1.0, 'top_k': None, 'description': 'Baseline'},
            45: {'num_heads': 8, 'temperature': 1.0, 'top_k': None, 'description': 'Baseline'},
            46: {'num_heads': 8, 'temperature': 1.0, 'top_k': None, 'description': 'Baseline'},
        }
    
    elif mode == 'diverse':
        return {
            1: {'num_heads': 8, 'temperature': 0.5, 'top_k': 200, 'description': 'Sharp, moderately sparse'},
            1000: {'num_heads': 16, 'temperature': 1.0, 'top_k': 500, 'description': 'Balanced, medium density'},
            100000: {'num_heads': 32, 'temperature': 2.0, 'top_k': 1000, 'description': 'Soft, dense'},
            1000000: {'num_heads': 64, 'temperature': 0.1, 'top_k': 100, 'description': 'Very sharp, very sparse'},
            10000000: {'num_heads': 128, 'temperature': 5.0, 'top_k': 2000, 'description': 'Very soft, very dense'}
        }
    
    elif mode == 'extreme':
        return {
            1: {'num_heads': 4, 'temperature': 0.01, 'top_k': 10, 'description': 'Extremely sharp (top-10)'},
            1000000: {'num_heads': 8, 'temperature': 0.5, 'top_k': 100, 'description': 'Sharp, sparse'},
            100000000: {'num_heads': 32, 'temperature': 2.0, 'top_k': 1000, 'description': 'Soft, medium'},
            10000000000: {'num_heads': 128, 'temperature': 10.0, 'top_k': None, 'description': 'Extremely soft, dense'}
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
