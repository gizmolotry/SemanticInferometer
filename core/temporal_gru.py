"""
Temporal GRU Memory - FIXED VERSION

Processes articles sequentially by timestamp, carrying discourse context
across the temporal sequence.

KEY FIX:
-------
- OLD: Processed each article independently with fake sequence (unsqueeze(1))
- NEW: Processes articles as TRUE temporal sequence

The GRU hidden state accumulates information from:
  Article_1 (earliest) → Article_2 → ... → Article_N (latest)

Each article's representation is influenced by all previous articles in
the temporal sequence.

Design:
------
- FROZEN after initialization (arbitrary but stable memory dynamics)
- Stateful (hidden state persists across batches if not reset)
- Residual connection (adds to features rather than replacing)
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalGRU(nn.Module):
    """
    GRU-based temporal memory for discourse context.
    
    Processes articles sequentially by timestamp, where each article's
    representation depends on previous articles.
    
    Parameters
    ----------
    feature_dim : int
        Dimensionality of article features (e.g., 24 for contrastive NLI)
    hidden_dim : int
        GRU hidden state dimension (default: same as feature_dim)
    seed : int, optional
        Random seed for frozen initialization
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # GRU cell - processes sequence
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=self.hidden_dim,
            batch_first=False  # [seq_len, batch, features]
        )
        
        # Project back to feature space (if hidden_dim != feature_dim)
        if self.hidden_dim != feature_dim:
            self.proj = nn.Linear(self.hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()
        
        # Hidden state (persistent across batches unless reset)
        self.hidden_state = None
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        print(f"\nTemporalGRU initialized:")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - Seed: {seed if seed is not None else 'random'}")
        print(f"  - Status: FROZEN (random initialization)")
    
    def forward(
        self,
        article_features: torch.Tensor,
        reset_hidden: bool = False
    ) -> torch.Tensor:
        """
        Process articles as temporal sequence.
        
        Parameters
        ----------
        article_features : torch.Tensor
            Article features, shape [N_articles, feature_dim]
            MUST be sorted by timestamp!
        reset_hidden : bool
            If True, reset hidden state before processing
        
        Returns
        -------
        torch.Tensor
            Temporally-augmented features, shape [N_articles, feature_dim]
            
        Notes
        -----
        - Output[0] depends only on input[0]
        - Output[1] depends on input[0] and input[1]
        - Output[i] depends on input[0:i+1]
        """
        # Reset if requested
        if reset_hidden:
            self.hidden_state = None
        
        # Reshape for GRU: [seq_len, batch_size, features]
        # We treat articles as sequence, batch size = 1
        x = article_features.unsqueeze(1)  # [N_articles, 1, feature_dim]
        
        # Process as sequence with previous hidden state
        # GRU returns: (output, hidden)
        # - output: [N_articles, 1, hidden_dim]
        # - hidden: [1, 1, hidden_dim] (final hidden state)
        output, self.hidden_state = self.gru(x, self.hidden_state)
        
        # Project back to feature space
        memory_features = self.proj(output.squeeze(1))  # [N_articles, feature_dim]
        
        # Residual connection
        return article_features + memory_features
    
    def reset_hidden(self):
        """Reset hidden state (call before new temporal batch)"""
        self.hidden_state = None
    
    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Get current hidden state (for inspection)"""
        return self.hidden_state


def process_temporal_batch(
    articles: list,
    article_features: torch.Tensor,
    gru: TemporalGRU
) -> torch.Tensor:
    """
    Helper function to process a temporal batch with proper sorting.
    
    Parameters
    ----------
    articles : list of dict
        Articles with 'published_at' or 'date' field
    article_features : torch.Tensor
        Features corresponding to articles, shape [N, feature_dim]
    gru : TemporalGRU
        Temporal GRU module
    
    Returns
    -------
    torch.Tensor
        Temporally-augmented features in ORIGINAL order
    """
    # Sort by timestamp
    sorted_indices = sorted(
        range(len(articles)),
        key=lambda i: articles[i].get('published_at', articles[i].get('date', ''))
    )
    
    # Reorder features
    sorted_features = article_features[sorted_indices]
    
    # Process sequentially
    temporal_features = gru(sorted_features)
    
    # Restore original order
    inverse_indices = torch.tensor([sorted_indices.index(i) for i in range(len(articles))])
    restored_features = temporal_features[inverse_indices]
    
    return restored_features


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Fixed Temporal GRU")
    print("="*70)
    
    # Create GRU
    torch.manual_seed(42)
    gru = TemporalGRU(feature_dim=24, hidden_dim=24, seed=42)
    
    print("\n--- Test 1: Sequential Processing ---")
    
    # Simulate 3 sequential articles
    article_1 = torch.randn(1, 24)
    article_2 = torch.randn(1, 24)
    article_3 = torch.randn(1, 24)
    
    # Process sequentially
    out_1 = gru(article_1, reset_hidden=True)
    out_2 = gru(article_2, reset_hidden=False)  # Depends on article_1
    out_3 = gru(article_3, reset_hidden=False)  # Depends on article_1 & article_2
    
    print(f"\nArticle 1 output change: {(out_1 - article_1).abs().mean():.4f}")
    print(f"Article 2 output change: {(out_2 - article_2).abs().mean():.4f}")
    print(f"Article 3 output change: {(out_3 - article_3).abs().mean():.4f}")
    print(f"Hidden state shape: {gru.hidden_state.shape}")
    
    print("\n--- Test 2: Batch Processing ---")
    
    # Process all 3 at once (as sequence)
    articles_batch = torch.cat([article_1, article_2, article_3], dim=0)  # [3, 24]
    
    gru.reset_hidden()
    batch_output = gru(articles_batch)  # [3, 24]
    
    print(f"Batch input shape: {articles_batch.shape}")
    print(f"Batch output shape: {batch_output.shape}")
    
    # Check that batch processing gives same results
    print("\nComparing sequential vs batch:")
    print(f"  Article 1 diff: {(batch_output[0] - out_1[0]).abs().mean():.6f}")
    print(f"  Article 2 diff: {(batch_output[1] - out_2[0]).abs().mean():.6f}")
    print(f"  Article 3 diff: {(batch_output[2] - out_3[0]).abs().mean():.6f}")
    
    print("\n--- Test 3: Temporal Dependency ---")
    
    # Show that article 3's output depends on article 1
    gru.reset_hidden()
    
    # Process article 1 alone, then article 3
    _ = gru(article_1, reset_hidden=True)
    out_3_with_context = gru(article_3)
    
    # Process article 3 alone (no context)
    out_3_no_context = gru(article_3, reset_hidden=True)
    
    context_effect = (out_3_with_context - out_3_no_context).abs().mean()
    print(f"\nContext effect on article 3: {context_effect:.4f}")
    print(f"(Should be >0, showing temporal dependency)")
    
    print("\n--- Test 4: Month-to-Month Memory ---")
    
    # Simulate processing 2 months of articles
    for month in [1, 2]:
        print(f"\nMonth {month}:")
        articles = torch.randn(50, 24)
        output = gru(articles, reset_hidden=False)  # Don't reset between months!
        
        change = (output - articles).abs().mean()
        print(f"  Mean feature change: {change:.4f}")
        print(f"  Hidden state norm: {gru.hidden_state.norm():.4f}")
    
    print("\n" + "="*70)
    print("✓ Temporal GRU tests complete!")
    print("="*70)
    print("\nKey takeaway:")
    print("  - Articles are now processed as TRUE temporal sequence")
    print("  - Each article's representation depends on previous articles")
    print("  - Hidden state carries temporal context forward")
