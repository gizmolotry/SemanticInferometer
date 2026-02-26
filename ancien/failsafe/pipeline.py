"""Single-Month Processing Pipeline

Orchestrates the complete pipeline for processing one month of articles:
1. Multi-framing NLI extraction
2. Provenance encoding
3. Article token construction
4. Cross-article attention computation

This is the core pipeline that will be called sequentially for each month.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from .nli_extraction import MultiFramingNLIExtractor
from .provenance import ProvenanceEncoder, build_metadata_indices, encode_article_metadata


class CrossArticleAttention(nn.Module):
    """Compute attention weights between articles.
    
    This is a FROZEN, randomly-initialized attention mechanism that acts as
    an "arbitrary observer." Different random seeds create different observers
    with different ways of perceiving discourse structure.
    
    Parameters
    ----------
    feature_dim : int
        Dimensionality of article token features
    num_heads : int
        Number of attention heads
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        print(f"Cross-article attention initialized:")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Num heads: {num_heads}")
        print(f"  - Status: FROZEN (random initialization)")
    
    def forward(self, article_tokens: torch.Tensor) -> torch.Tensor:
        """Compute attention matrix between articles.
        
        Parameters
        ----------
        article_tokens : torch.Tensor
            Article representations, shape (n_articles, feature_dim)
        
        Returns
        -------
        torch.Tensor
            Attention matrix, shape (n_articles, n_articles)
            Entry [i,j] = how much article i attends to article j
        """
        # Add batch dimension
        x = article_tokens.unsqueeze(0)  # (1, n_articles, feature_dim)
        
        # Self-attention over articles
        _, attn_weights = self.attention(
            x, x, x,
            need_weights=True,
            average_attn_weights=True  # Average over heads for visualization
        )
        
        # Remove batch dimension
        return attn_weights.squeeze(0)  # (n_articles, n_articles)


class MonthlyPipeline:
    """Process one month of articles through the complete pipeline.
    
    Parameters
    ----------
    nli_extractor : MultiFramingNLIExtractor
        Pre-initialized NLI extractor
    provenance_encoder : ProvenanceEncoder
        Pre-initialized provenance encoder
    attention_module : CrossArticleAttention
        Pre-initialized attention mechanism
    metadata_indices : Dict
        Source/author ID mappings
    date_range : tuple, optional
        (min_timestamp, max_timestamp) for timestamp normalization
    """
    
    def __init__(
        self,
        nli_extractor: MultiFramingNLIExtractor,
        provenance_encoder: ProvenanceEncoder,
        attention_module: CrossArticleAttention,
        metadata_indices: Dict,
        date_range: Optional[Tuple[float, float]] = None
    ):
        self.nli_extractor = nli_extractor
        self.provenance_encoder = provenance_encoder
        self.attention_module = attention_module
        self.metadata_indices = metadata_indices
        self.date_range = date_range
        
        # Ensure all components are in eval mode and frozen
        self.nli_extractor.eval()
        self.provenance_encoder.eval()
        self.attention_module.eval()
        
        for module in [self.provenance_encoder, self.attention_module]:
            for param in module.parameters():
                param.requires_grad = False
    
    @torch.no_grad()
    def process_month(
        self,
        articles: List[Dict],
        month_id: int,
        show_progress: bool = True
    ) -> Dict:
        """Process one month of articles.
        
        Parameters
        ----------
        articles : List[Dict]
            Article dictionaries with fields:
            - 'text': str (content)
            - 'source': str (publisher)
            - 'timestamp': float (unix timestamp)
            - 'author': str (optional)
            - 'url': str (optional, for tracking)
        month_id : int
            Month identifier (1-12)
        show_progress : bool
            Whether to show progress bars
        
        Returns
        -------
        Dict
            Results containing:
            - attention_matrix: (n_articles, n_articles) attention weights
            - article_tokens: (n_articles, feature_dim) final representations
            - metadata: List[Dict] article metadata
            - month_id: int
        """
        n_articles = len(articles)
        
        print(f"\n{'='*60}")
        print(f"Processing Month {month_id}: {n_articles} articles")
        print(f"{'='*60}")
        
        # Step 1: Multi-framing NLI extraction
        print("\n[1/4] Extracting multi-framing features...")
        multi_framing_features = self.nli_extractor.extract_batch(
            articles,
            show_progress=show_progress
        )
        print(f"   ✅ Shape: {multi_framing_features.shape}")
        
        # Step 2: Provenance encoding
        print("\n[2/4] Encoding provenance metadata...")
        metadata_tensors = encode_article_metadata(
            articles,
            self.metadata_indices,
            date_range=self.date_range
        )
        
        provenance_embeddings = self.provenance_encoder(
            metadata_tensors['source_ids'],
            metadata_tensors['timestamps'],
            metadata_tensors.get('author_ids')
        )
        print(f"   ✅ Shape: {provenance_embeddings.shape}")
        
        # Step 3: Construct article tokens (additive fusion)
        print("\n[3/4] Constructing article tokens...")
        article_tokens = multi_framing_features + provenance_embeddings
        print(f"   ✅ Shape: {article_tokens.shape}")
        print(f"   Token norm (mean): {article_tokens.norm(dim=-1).mean():.2f}")
        
        # Step 4: Cross-article attention
        print("\n[4/4] Computing cross-article attention...")
        attention_matrix = self.attention_module(article_tokens)
        print(f"   ✅ Shape: {attention_matrix.shape}")
        print(f"   Attention sparsity: {(attention_matrix < 0.01).float().mean():.2%}")
        print(f"   Max attention: {attention_matrix.max():.4f}")
        
        # Collect metadata
        metadata = []
        for article in articles:
            metadata.append({
                'source': article['source'],
                'timestamp': article['timestamp'],
                'author': article.get('author', 'unknown'),
                'url': article.get('url', ''),
                'text_preview': article.get('content', article.get('text', ''))[:200] + '...'
            })
        
        results = {
            'attention_matrix': attention_matrix.cpu(),
            'article_tokens': article_tokens.cpu(),
            'metadata': metadata,
            'month_id': month_id,
            'n_articles': n_articles
        }
        
        print(f"\n✅ Month {month_id} processing complete!\n")
        
        return results


def initialize_pipeline(
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli',
    queries_config: str = 'config/framing_queries.yaml',
    device: str = 'cuda',
    num_attention_heads: int = 8,
    random_seed: Optional[int] = None
) -> Tuple[MultiFramingNLIExtractor, ProvenanceEncoder, CrossArticleAttention]:
    """Initialize all pipeline components.
    
    Parameters
    ----------
    nli_model_name : str
        HuggingFace model for NLI extraction
    queries_config : str
        Path to framing queries YAML
    device : str
        Device for computation
    num_attention_heads : int
        Number of attention heads in cross-article attention
    random_seed : int, optional
        Random seed for reproducible "observers"
        Different seeds = different perspectives
    
    Returns
    -------
    Tuple
        (nli_extractor, provenance_encoder, attention_module)
        Note: Provenance encoder not fully initialized (needs article corpus)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        print(f"\n🌱 Random seed set to {random_seed}")
        print("   This creates a stable, repeatable 'observer' perspective")
    
    print("\n" + "="*60)
    print("Initializing Belief Transformer Pipeline")
    print("="*60)
    
    # Initialize NLI extractor (frozen pre-trained model)
    print("\n[1/3] Initializing NLI extractor...")
    nli_extractor = MultiFramingNLIExtractor(
        model_name=nli_model_name,
        queries_config=queries_config,
        device=device
    )
    
    feature_dim = nli_extractor.n_framings * nli_extractor.embedding_dim
    
    # Provenance encoder will be built after seeing corpus
    # (need to know num_sources, num_authors)
    print("\n[2/3] Provenance encoder (will be built from corpus)")
    print(f"   Feature dim: {feature_dim}")
    
    # Initialize attention module (frozen random init)
    print("\n[3/3] Initializing cross-article attention...")
    attention_module = CrossArticleAttention(
        feature_dim=feature_dim,
        num_heads=num_attention_heads
    )
    
    # Freeze attention
    for param in attention_module.parameters():
        param.requires_grad = False
    
    print("\n" + "="*60)
    print("Pipeline Initialization Complete")
    print("="*60)
    
    return nli_extractor, None, attention_module  # Provenance built later


if __name__ == "__main__":
    # Test pipeline on dummy data
    print("="*60)
    print("Monthly Pipeline - Test Run")
    print("="*60)
    
    # Create dummy articles
    dummy_articles = []
    sources = ['NYT', 'Fox', 'BBC', 'Al Jazeera']
    
    for i in range(20):  # Small test set
        dummy_articles.append({
            'text': f"""
            This is test article {i}. Israeli forces conducted operations in Gaza
            while Hamas launched rockets. The conflict continues with civilian
            casualties reported on both sides. International community calls for
            ceasefire.
            """,
            'source': sources[i % len(sources)],
            'timestamp': 1704067200 + i * 86400,  # Jan 2024, daily increment
            'author': f'Author {i % 5}',
            'url': f'http://example.com/article{i}'
        })
    
    # Initialize components (use small model for testing)
    print("\nInitializing pipeline...")
    nli_extractor, _, attention = initialize_pipeline(
        nli_model_name='microsoft/deberta-v3-base',  # Smaller for testing
        device='cpu',
        random_seed=42
    )
    
    # Build metadata indices
    from provenance import build_metadata_indices
    indices = build_metadata_indices(dummy_articles)
    
    # Build provenance encoder
    provenance_encoder = ProvenanceEncoder(
        feature_dim=nli_extractor.n_framings * nli_extractor.embedding_dim,
        num_sources=indices['num_sources'],
        num_authors=indices.get('num_authors')
    )
    
    # Freeze provenance
    for param in provenance_encoder.parameters():
        param.requires_grad = False
    
    # Create pipeline
    pipeline = MonthlyPipeline(
        nli_extractor=nli_extractor,
        provenance_encoder=provenance_encoder,
        attention_module=attention,
        metadata_indices=indices
    )
    
    # Process test month
    results = pipeline.process_month(dummy_articles, month_id=1)
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Attention matrix shape: {results['attention_matrix'].shape}")
    print(f"Article tokens shape: {results['article_tokens'].shape}")
    print(f"Metadata entries: {len(results['metadata'])}")
    print(f"\n✅ Pipeline test successful!")
