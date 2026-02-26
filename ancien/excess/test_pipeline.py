"""
Complete Pipeline - FIXED FOR YOUR DATA FORMAT

Your articles use:
- 'content' not 'text'
- 'publisher' not 'source'  
- 'published_at' not 'timestamp'
"""

import sys
sys.path.append('.')

from pipeline_enhanced import CrossArticleAttention, create_observer_configs
from core.nli_extraction import MultiFramingNLIExtractor
from core.provenance import ProvenanceEncoder, build_metadata_indices, encode_article_metadata
from core.temporal_gru import TemporalGRU
from core.framing_rope import FramingRoPE

import torch
import os
import hashlib
from typing import Dict, List
from datetime import datetime


def convert_date_to_timestamp(date_string):
    """Convert ISO date string to Unix timestamp."""
    try:
        if isinstance(date_string, (int, float)):
            return date_string
        dt = datetime.fromisoformat(str(date_string).replace('Z', '+00:00'))
        return int(dt.timestamp())
    except:
        return 0  # Default if conversion fails


def initialize_diverse_pipeline(
    observer_seed: int,
    observer_config: Dict,
    nli_model_name: str = 'microsoft/deberta-v3-large',
    device: str = 'cuda',
    use_gru: bool = True,
    use_framing_rope: bool = True,
    queries_config: str = 'config/framing_queries.yaml'
) -> Dict:
    print("\n" + "="*70)
    print(f"INITIALIZING OBSERVER {observer_seed}")
    print(f"Config: {observer_config['description']}")
    print("="*70 + "\n")
    
    torch.manual_seed(observer_seed)
    
    # NLI Extractor
    print("### Reader Layer: NLI Extractor ###")
    nli_extractor = MultiFramingNLIExtractor(
        model_name=nli_model_name,
        queries_config=queries_config,
        device=device
    )
    
    n_framings = nli_extractor.n_framings
    embedding_dim = nli_extractor.embedding_dim
    feature_dim = n_framings * embedding_dim
    
    # Optional components
    framing_rope = None
    if use_framing_rope:
        framing_rope = FramingRoPE(feature_dim=feature_dim, n_framings=n_framings, max_angle=0.1)
        for param in framing_rope.parameters():
            param.requires_grad = False
    
    temporal_gru = None
    if use_gru:
        temporal_gru = TemporalGRU(feature_dim=feature_dim, hidden_dim=512)
        for param in temporal_gru.parameters():
            param.requires_grad = False
    
    # Observer
    attention = CrossArticleAttention(
        feature_dim=feature_dim,
        num_heads=observer_config['num_heads'],
        temperature=observer_config['temperature'],
        top_k=observer_config['top_k']
    )
    for param in attention.parameters():
        param.requires_grad = False
    
    return {
        'nli_extractor': nli_extractor,
        'framing_rope': framing_rope,
        'temporal_gru': temporal_gru,
        'attention': attention,
        'feature_dim': feature_dim,
        'config': {
            'seed': observer_seed,
            'observer_config': observer_config,
            'use_gru': use_gru,
            'use_framing_rope': use_framing_rope
        }
    }


def process_articles_with_observer(
    articles: List[Dict],
    components: Dict,
    metadata_indices: Dict,
    observer_seed: int
) -> Dict:
    nli = components['nli_extractor']
    rope = components['framing_rope']
    gru = components['temporal_gru']
    attention = components['attention']
    feature_dim = components['feature_dim']
    
    # Provenance
    torch.manual_seed(observer_seed)
    provenance = ProvenanceEncoder(feature_dim=feature_dim, num_sources=metadata_indices['num_sources'])
    for param in provenance.parameters():
        param.requires_grad = False
    
    # Cache - MODIFIED to distinguish shuffle from real
    article_hash = hashlib.md5(
        str([a.get('url', str(i)) for i, a in enumerate(articles)]).encode()
    ).hexdigest()[:8]
    
    # Add corpus type to cache key
    corpus_type = 'shuffle' if articles[0].get('control_type') == 'length_matched_shuffle' else 'real'
    cache_file = f'outputs/nli_cache_{corpus_type}_{article_hash}.pt'
    os.makedirs('outputs', exist_ok=True)
    
    # NLI extraction
    print("  [1/6] Multi-framing extraction...")
    if os.path.exists(cache_file):
        print(f"    Loading cached from {cache_file}")
        multi_framing_features = torch.load(cache_file)
    else:
        texts = [a.get('content', '') for a in articles]  # FIXED: use 'content'
        multi_framing_features = nli.extract_batch(texts)
        torch.save(multi_framing_features, cache_file)
        print(f"    Cached to {cache_file}")
    
    # Provenance
    print("  [2/6] Provenance encoding...")
    prov_features = encode_article_metadata(articles, metadata_indices, provenance)
    
    # Combine
    article_tokens = multi_framing_features + prov_features
    print(f"  [3/6] Combined: {article_tokens.shape}")
    
    # Optional layers
    if rope:
        print("  [4/6] Framing RoPE...")
        article_tokens = rope(article_tokens)
    else:
        print("  [4/6] Framing RoPE: SKIPPED")
    
    if gru:
        print("  [5/6] Temporal GRU...")
        article_tokens = gru(article_tokens)
    else:
        print("  [5/6] Temporal GRU: SKIPPED")
    
    # Attention
    print("  [6/6] Computing attention...")
    attention_matrix = attention(article_tokens)
    
    # Metadata - FIXED for your data format
    metadata = []
    for i, article in enumerate(articles):
        timestamp = article.get('timestamp', 0)
        if isinstance(timestamp, str):
            timestamp = convert_date_to_timestamp(timestamp)
        
        metadata.append({
            'index': i,
            'source': article.get('publisher', article.get('source', 'unknown')),  # FIXED
            'timestamp': timestamp,  # FIXED - numeric
            'url': article.get('url', ''),
            'text_preview': article.get('content', '')[:200]  # FIXED
        })
    
    return {
        'attention_matrix': attention_matrix,
        'article_tokens': article_tokens,
        'metadata': metadata,
        'observer_seed': observer_seed,
        'observer_config': components['config']['observer_config'],
        'corpus_type': 'real',
        'corpus_name': 'gaza_israel'
    }


def run_diverse_observer_experiment(
    articles: List[Dict],
    observer_mode: str = 'diverse',
    use_gru: bool = True,
    use_framing_rope: bool = True,
    device: str = 'cuda',
    nli_model_name: str = 'microsoft/deberta-v3-large'
) -> Dict[int, Dict]:
    print("="*70)
    print("DIVERSE OBSERVER EXPERIMENT")
    print(f"Mode: {observer_mode}, Articles: {len(articles)}")
    print("="*70)
    
    # Normalize keys for provenance compatibility
    # Your data uses 'publisher' and 'published_at', but provenance expects 'source' and 'timestamp'
    for article in articles:
        if 'publisher' in article and 'source' not in article:
            article['source'] = article['publisher']
        if 'published_at' in article and 'timestamp' not in article:
            # Convert ISO date string to numeric timestamp
            article['timestamp'] = convert_date_to_timestamp(article['published_at'])
        elif 'timestamp' in article and isinstance(article['timestamp'], str):
            # Convert if timestamp is still a string
            article['timestamp'] = convert_date_to_timestamp(article['timestamp'])
    
    observer_configs = create_observer_configs(mode=observer_mode)
    metadata_indices = build_metadata_indices(articles)
    
    results = {}
    for seed, observer_config in observer_configs.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING OBSERVER {seed}")
        print(f"{'='*70}")
        
        components = initialize_diverse_pipeline(
            observer_seed=seed,
            observer_config=observer_config,
            use_gru=use_gru,
            use_framing_rope=use_framing_rope,
            device=device,
            nli_model_name=nli_model_name
        )
        
        result = process_articles_with_observer(
            articles=articles,
            components=components,
            metadata_indices=metadata_indices,
            observer_seed=seed
        )
        
        results[seed] = result
        
        output_file = f'outputs/diverse_observer_{seed}.pt'
        torch.save(result, output_file)
        print(f"  Saved to {output_file}")
    
    return results