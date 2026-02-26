"""
Provenance Encoding - FIXED VERSION

Fixes:
1. Robust timestamp conversion (handles strings)
2. Device placement (tensors moved to encoder's device before forward pass)

DEVICE FIX:
-----------
The encoder might be on CUDA, but we were creating tensors on CPU.
Now we detect the encoder's device and move all tensors there before calling forward().
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


def _safe_float_convert(value):
    """
    Safely convert a value to float, handling strings, None, and edge cases.
    
    This is a defensive helper that ensures timestamps are always numeric,
    even if the normalization in load_articles() somehow failed.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # Try direct float conversion
            return float(value)
        except (ValueError, TypeError):
            # If that fails, return 0.0
            return 0.0
    # For any other type, return 0.0
    return 0.0


class ProvenanceEncoder(nn.Module):
    """Encode article metadata into embedding space."""
    
    def __init__(
        self,
        feature_dim: int,
        num_sources: int,
        num_authors: Optional[int] = None,
        min_author_articles: int = 10
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.min_author_articles = min_author_articles
        
        # Allocate dimension budget
        if num_authors is not None:
            source_dim = feature_dim // 3
            author_dim = feature_dim // 3
            timestamp_dim = feature_dim - source_dim - author_dim
        else:
            source_dim = feature_dim // 2
            timestamp_dim = feature_dim - source_dim
            author_dim = 0
        
        self.source_embed = nn.Embedding(num_sources, source_dim)
        
        if num_authors is not None:
            self.author_embed = nn.Embedding(num_authors + 1, author_dim)
            self.unknown_author_id = num_authors
        else:
            self.author_embed = None
        
        self.timestamp_proj = nn.Linear(1, timestamp_dim)
        
        print(f"Provenance encoder initialized:")
        print(f"  - Source embedding: {num_sources} sources -> {source_dim}D")
        if num_authors is not None:
            print(f"  - Author embedding: {num_authors} authors -> {author_dim}D")
        print(f"  - Timestamp projection: 1D -> {timestamp_dim}D")
        print(f"  - Total output: {feature_dim}D")
    
    def forward(
        self,
        source_ids: torch.Tensor,
        timestamps: torch.Tensor,
        author_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode provenance metadata."""
        source_emb = self.source_embed(source_ids)
        timestamp_emb = self.timestamp_proj(timestamps.unsqueeze(-1))
        
        if author_ids is not None and self.author_embed is not None:
            author_emb = self.author_embed(author_ids)
            provenance = torch.cat([source_emb, author_emb, timestamp_emb], dim=-1)
        else:
            provenance = torch.cat([source_emb, timestamp_emb], dim=-1)
        
        return provenance


def build_metadata_indices(
    articles: List[Dict],
    min_author_articles: int = 10
) -> Dict:
    """Build source and author ID mappings from article metadata."""
    from collections import Counter
    
    sources = sorted(set(a['source'] for a in articles))
    source_to_id = {source: i for i, source in enumerate(sources)}
    id_to_source = {i: source for source, i in source_to_id.items()}
    
    result = {
        'source_to_id': source_to_id,
        'id_to_source': id_to_source,
        'num_sources': len(sources)
    }
    
    if 'author' in articles[0]:
        author_counts = Counter(a['author'] for a in articles if a.get('author'))
        
        frequent_authors = sorted([
            author for author, count in author_counts.items()
            if count >= min_author_articles
        ])
        
        author_to_id = {author: i for i, author in enumerate(frequent_authors)}
        id_to_author = {i: author for author, i in author_to_id.items()}
        
        result.update({
            'author_to_id': author_to_id,
            'id_to_author': id_to_author,
            'num_authors': len(frequent_authors),
            'author_counts': author_counts
        })
        
        print(f"Built author index: {len(frequent_authors)} frequent authors")
        print(f"  (threshold: >={min_author_articles} articles)")
        print(f"  Rare authors will map to 'unknown_author' embedding")
    
    print(f"Built source index: {len(sources)} sources")
    
    return result


def encode_article_metadata(
    articles: List[Dict],
    metadata_indices: Dict,
    encoder: ProvenanceEncoder,
    date_range: Optional[tuple] = None
) -> torch.Tensor:
    """
    Convert article metadata to provenance embeddings.
    
    CRITICAL FIXES:
    1. Uses _safe_float_convert() to ensure ALL timestamps are float64
    2. Moves all tensors to encoder's device before forward pass
    
    Parameters
    ----------
    articles : List[Dict]
        Articles with 'source', 'timestamp', and optionally 'author'
    metadata_indices : Dict
        Output from build_metadata_indices()
    encoder : ProvenanceEncoder
        The provenance encoder module (may be on CUDA or CPU)
    date_range : tuple, optional
        (min_timestamp, max_timestamp) for normalization
    
    Returns
    -------
    torch.Tensor
        Provenance embeddings, shape (n_articles, feature_dim)
        On the same device as the encoder
    """
    source_to_id = metadata_indices['source_to_id']
    author_to_id = metadata_indices.get('author_to_id')
    unknown_author_id = metadata_indices.get('num_authors')
    
    # Extract source IDs (CPU)
    source_ids = torch.LongTensor([
        source_to_id[a['source']] for a in articles
    ])
    
    # CRITICAL FIX: Force conversion to float even if timestamps are strings
    timestamps = np.array([
        _safe_float_convert(a.get('timestamp'))
        for a in articles
    ], dtype=np.float64)
    
    if date_range is None:
        t_min, t_max = timestamps.min(), timestamps.max()
    else:
        t_min, t_max = date_range
    
    timestamps_normalized = (timestamps - t_min) / (t_max - t_min + 1e-8)
    timestamps_tensor = torch.FloatTensor(timestamps_normalized)
    
    # Extract author IDs if present (CPU)
    author_ids = None
    if author_to_id is not None and articles and 'author' in articles[0]:
        author_ids_list = []
        for a in articles:
            author = a.get('author')
            if author and author in author_to_id:
                author_ids_list.append(author_to_id[author])
            else:
                author_ids_list.append(unknown_author_id)
        
        author_ids = torch.LongTensor(author_ids_list)
    
    # DEVICE FIX: Move all tensors to encoder's device before calling
    # The encoder might be on CUDA while these tensors are on CPU
    device = next(encoder.parameters()).device
    source_ids = source_ids.to(device)
    timestamps_tensor = timestamps_tensor.to(device)
    if author_ids is not None:
        author_ids = author_ids.to(device)
    
    # Encode
    provenance_embeddings = encoder(source_ids, timestamps_tensor, author_ids)
    
    return provenance_embeddings