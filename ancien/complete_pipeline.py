"""
LOGITS-BASED Complete Belief Transformer Pipeline

Changes from original:
1. Uses MultiFramingNLIExtractorLogits (24D logits instead of 12288D embeddings)
2. Uses improved framing queries (config/framing_queries.yaml)
3. Feature dimension: 8 framings × 3 logits = 24D
4. Added NLI feature caching
5. Fixed provenance encoding argument order
6. Added metadata tracking with titles
"""

import torch
import os
import hashlib
from typing import Dict, List, Optional
from .nli_extraction_logits import MultiFramingNLIExtractorLogits
from .provenance import ProvenanceEncoder, build_metadata_indices, encode_article_metadata
from .pipeline import CrossArticleAttention
from .temporal_gru import TemporalGRU
from .framing_rope import FramingRoPE


def initialize_full_pipeline(
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli',
    device: str = 'cuda',
    random_seed: int = 42,
    use_gru: bool = True,
    use_framing_rope: bool = True,
    queries_config: str = 'config/framing_queries.yaml'
) -> Dict:
    """
    Initialize the complete pipeline with all components.
    
    Parameters
    ----------
    nli_model_name : str
        HuggingFace model name (should be MNLI-finetuned)
    device : str
        'cuda' or 'cpu'
    random_seed : int
        Observer seed (creates specific perspective)
    use_gru : bool
        Include temporal memory
    use_framing_rope : bool
        Include framing position encoding
    queries_config : str
        Path to framing queries YAML
    
    Returns
    -------
    dict
        All initialized components
    """
    print("\n" + "="*70)
    print(f"INITIALIZING COMPLETE BELIEF TRANSFORMER (LOGITS-BASED)")
    print(f"Observer Seed: {random_seed}")
    print(f"GRU: {'ENABLED' if use_gru else 'DISABLED'}")
    print(f"Framing RoPE: {'ENABLED' if use_framing_rope else 'DISABLED'}")
    print("="*70 + "\n")
    
    # Set seed for reproducibility
    torch.manual_seed(random_seed)
    
    # 1. NLI Extractor (frozen pre-trained, using LOGITS)
    print("### Component 1: NLI Extractor (Logits-Based) ###")
    nli_extractor = MultiFramingNLIExtractorLogits(
        model_name=nli_model_name,
        queries_config=queries_config,
        device=device
    )
    
    # Calculate feature dimensions
    # CRITICAL: Logits are 3D per framing, not 1536D embeddings
    n_framings = nli_extractor.n_framings
    logits_dim = 3  # [contradiction, neutral, entailment]
    feature_dim = n_framings * logits_dim  # e.g., 8 × 3 = 24
    
    print(f"\nFeature dimensions:")
    print(f"  Framings: {n_framings}")
    print(f"  Logits per framing: {logits_dim}")
    print(f"  Total: {feature_dim}")
    
    # 2. Framing RoPE (optional)
    framing_rope = None
    if use_framing_rope:
        print("\n### Component 2: Framing RoPE ###")
        framing_rope = FramingRoPE(
            feature_dim=feature_dim,
            n_framings=n_framings,
            max_angle=0.1
        )
        for param in framing_rope.parameters():
            param.requires_grad = False
    
    # 3. Temporal GRU (optional)
    temporal_gru = None
    if use_gru:
        print("\n### Component 3: Temporal GRU ###")
        temporal_gru = TemporalGRU(
            feature_dim=feature_dim,
            hidden_dim=512
        )
        for param in temporal_gru.parameters():
            param.requires_grad = False
    
    # 4. Cross-Article Attention (the observer)
    print("\n### Component 4: Cross-Article Attention ###")
    attention = CrossArticleAttention(
        feature_dim=feature_dim,
        num_heads=8
    )
    for param in attention.parameters():
        param.requires_grad = False
    
    print("\n" + "="*70)
    print("✓ PIPELINE INITIALIZATION COMPLETE")
    print("="*70 + "\n")
    
    return {
        'nli_extractor': nli_extractor,
        'framing_rope': framing_rope,
        'temporal_gru': temporal_gru,
        'attention': attention,
        'feature_dim': feature_dim,
        'config': {
            'seed': random_seed,
            'use_gru': use_gru,
            'use_framing_rope': use_framing_rope,
            'nli_model': nli_model_name
        }
    }


class CompletePipeline:
    """
    Full pipeline orchestrator with all components.
    """
    
    def __init__(
        self,
        components: Dict,
        metadata_indices: Dict,
        provenance_encoder: Optional[ProvenanceEncoder] = None
    ):
        self.nli = components['nli_extractor']
        self.rope = components['framing_rope']
        self.gru = components['temporal_gru']
        self.attention = components['attention']
        self.feature_dim = components['feature_dim']
        self.config = components['config']
        
        self.metadata_indices = metadata_indices
        
        # Create provenance encoder if not provided
        if provenance_encoder is None:
            torch.manual_seed(self.config['seed'])  # Same seed for consistency
            self.provenance = ProvenanceEncoder(
                feature_dim=self.feature_dim,
                num_sources=metadata_indices['num_sources']
            )
            for param in self.provenance.parameters():
                param.requires_grad = False
        else:
            self.provenance = provenance_encoder
    
    def process_month(
        self,
        articles: List[Dict],
        month_id: int
    ) -> Dict:
        """
        Process one month through the complete pipeline.
        
        Parameters
        ----------
        articles : list of dict
            Articles with 'content', 'source', 'timestamp', 'title', etc.
        month_id : int
            Month identifier (1-12)
        
        Returns
        -------
        dict
            - attention_matrix: (N, N) tensor
            - article_tokens: (N, feature_dim) tensor
            - metadata: list of dicts
        """
        print(f"\n### Processing Month {month_id} ###")
        print(f"Articles: {len(articles)}")
        
        # Generate cache key from article URLs (same corpus = same features)
        article_hash = hashlib.md5(
            str([a.get('url', str(i)) for i, a in enumerate(articles)]).encode()
        ).hexdigest()[:8]
        cache_file = f'outputs/nli_cache_{article_hash}.pt'
        
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # 1. NLI extraction (WITH CACHING)
        print("  [1/6] Multi-framing extraction (logits-based)...")
        if os.path.exists(cache_file):
            print(f"    ✓ Loading cached features from {cache_file}")
            multi_framing_features = torch.load(cache_file, weights_only=False)
        else:
            # Extract articles - handle both 'content' and 'text' keys
            texts = [a.get('content', a.get('text', '')) for a in articles]
            multi_framing_features = self.nli.extract_batch(texts, show_progress=True)
            # Cache it
            torch.save(multi_framing_features, cache_file)
            print(f"    ✓ Cached features to {cache_file}")
        
        print(f"    Shape: {multi_framing_features.shape}")
        
        # 2. Provenance encoding
        print("  [2/6] Provenance encoding...")
        prov_features = encode_article_metadata(
            articles,
            self.metadata_indices,
            self.provenance
        )
        print(f"    Shape: {prov_features.shape}")
        
        # 3. Combine (additive fusion)
        article_tokens = multi_framing_features + prov_features
        print(f"  [3/6] Combined features: {article_tokens.shape}")
        
        # 4. Optional: Framing RoPE
        if self.rope is not None:
            print("  [4/6] Applying Framing RoPE...")
            article_tokens = self.rope(article_tokens)
        else:
            print("  [4/6] Framing RoPE: SKIPPED")
        
        # 5. Optional: Temporal GRU
        if self.gru is not None:
            print("  [5/6] Processing with Temporal GRU...")
            article_tokens = self.gru(article_tokens)
        else:
            print("  [5/6] Temporal GRU: SKIPPED")
        
        # 6. Cross-article attention
        print("  [6/6] Computing attention...")
        attention_matrix = self.attention(article_tokens)
        
        # Prepare metadata with title
        metadata = []
        for i, article in enumerate(articles):
            metadata.append({
                'index': i,
                'title': article.get('title', '')[:200],
                'source': article.get('source', article.get('publisher', 'unknown')),
                'timestamp': article.get('timestamp', article.get('published_at', 0)),
                'url': article.get('url', ''),
                'text_preview': article.get('content', article.get('text', ''))[:200]
            })
        
        print(f"  ✓ Month {month_id} complete\n")
        
        return {
            'attention_matrix': attention_matrix,
            'article_tokens': article_tokens,
            'metadata': metadata,
            'month_id': month_id,
            'observer_seed': self.config['seed']
        }
    
    def reset_temporal_memory(self):
        """Reset GRU hidden state (call before new 12-month sequence)"""
        if self.gru is not None:
            self.gru.reset()


def run_multi_observer_experiment(
    articles: List[Dict],
    seeds: List[int] = [42, 43, 44, 45, 46],
    use_gru: bool = True,
    use_framing_rope: bool = True,
    device: str = 'cuda',
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli'
) -> Dict[int, Dict]:
    """
    Run the same corpus through multiple observers.
    
    This is THE key experiment.
    """
    print("\n" + "="*70)
    print("MULTI-OBSERVER EXPERIMENT (LOGITS-BASED)")
    print(f"Observers: {seeds}")
    print(f"Articles: {len(articles)}")
    print(f"Device: {device}")
    print("="*70)
    
    # Build metadata indices once
    metadata_indices = build_metadata_indices(articles)
    
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"OBSERVER {seed}")
        print(f"{'='*70}")
        
        # Initialize with this seed
        components = initialize_full_pipeline(
            random_seed=seed,
            use_gru=use_gru,
            use_framing_rope=use_framing_rope,
            device=device,
            nli_model_name=nli_model_name
        )
        
        # Create pipeline
        pipeline = CompletePipeline(
            components=components,
            metadata_indices=metadata_indices
        )
        
        # Process
        result = pipeline.process_month(articles, month_id=1)
        results[seed] = result
        
        # Save
        os.makedirs('outputs', exist_ok=True)
        torch.save(result, f'outputs/observer_{seed}.pt')
        print(f"  → Saved to outputs/observer_{seed}.pt")
    
    print("\n" + "="*70)
    print("✓ MULTI-OBSERVER EXPERIMENT COMPLETE")
    print("="*70)
    
    return results
