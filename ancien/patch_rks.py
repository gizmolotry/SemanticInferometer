"""
Automatically patch complete_pipeline.py to add RKS support.

This script:
1. Adds RKS import
2. Updates initialize_full_pipeline signature
3. Modifies CompletePipeline.__init__ to include RKS
4. Updates process_month to apply RKS
5. Updates run_multi_observer_experiment signature
"""

import re

def patch_complete_pipeline(filepath: str):
    """Patch complete_pipeline.py with RKS support."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Patching complete_pipeline.py with RKS support...")
    print("="*70)
    
    # 1. Add RKS import after other imports
    if 'from .rks_expansion import RandomKitchenSinksExpander' not in content:
        print("✓ Adding RKS import...")
        import_block = "from .framing_rope import FramingRoPE"
        content = content.replace(
            import_block,
            import_block + "\nfrom .rks_expansion import RandomKitchenSinksExpander"
        )
    
    # 2. Update initialize_full_pipeline signature
    if 'use_rks: bool = True' not in content:
        print("✓ Adding RKS parameters to initialize_full_pipeline...")
        old_sig = "    queries_config: str = 'config/framing_queries.yaml'\n) -> Dict:"
        new_sig = """    queries_config: str = 'config/framing_queries.yaml',
    use_rks: bool = True,
    rks_dim: int = 512,
    rks_sigma: float = 1.0
) -> Dict:"""
        content = content.replace(old_sig, new_sig)
    
    # 3. Update feature dimension calculation in initialize_full_pipeline
    if 'Base feature dimensions (pre-RKS' not in content:
        print("✓ Updating feature dimension logic...")
        old_feature_calc = """    # Calculate feature dimensions
    # CRITICAL: Logits are 3D per framing, not 1536D embeddings
    n_framings = nli_extractor.n_framings
    logits_dim = 3  # [contradiction, neutral, entailment]
    feature_dim = n_framings * logits_dim  # e.g., 8 × 3 = 24
    
    print(f"\\nFeature dimensions:")
    print(f"  Framings: {n_framings}")
    print(f"  Logits per framing: {logits_dim}")
    print(f"  Total: {feature_dim}")"""
        
        new_feature_calc = """    # Base feature dimensions (pre-RKS, pre-provenance)
    n_framings = nli_extractor.n_framings
    logits_dim = 3  # [contradiction, neutral, entailment]
    base_feature_dim = n_framings * logits_dim  # e.g., 8 × 3 = 24
    
    # Observer feature dim (after RKS expansion)
    observer_feature_dim = rks_dim if use_rks else base_feature_dim
    
    print(f"\\nFeature dimensions:")
    print(f"  Framings: {n_framings}")
    print(f"  Logits per framing: {logits_dim}")
    print(f"  Base feature dim: {base_feature_dim}")
    if use_rks:
        print(f"  RKS-expanded dim: {observer_feature_dim}")
    else:
        print(f"  RKS: DISABLED (observer dim = base dim)")"""
        
        content = content.replace(old_feature_calc, new_feature_calc)
    
    # 4. Update FramingRoPE to use base_feature_dim
    content = content.replace(
        "        framing_rope = FramingRoPE(\n            feature_dim=feature_dim,",
        "        framing_rope = FramingRoPE(\n            feature_dim=base_feature_dim,"
    )
    
    # 5. Update TemporalGRU and CrossArticleAttention to use observer_feature_dim
    content = content.replace(
        "        temporal_gru = TemporalGRU(\n            feature_dim=feature_dim,",
        "        temporal_gru = TemporalGRU(\n            feature_dim=observer_feature_dim,"
    )
    content = content.replace(
        "    attention = CrossArticleAttention(\n        feature_dim=feature_dim,",
        "    attention = CrossArticleAttention(\n        feature_dim=observer_feature_dim,"
    )
    
    # 6. Update return dict in initialize_full_pipeline
    if "'base_feature_dim'" not in content:
        print("✓ Updating return dict with RKS config...")
        old_return = """    return {
        'nli_extractor': nli_extractor,
        'framing_rope': framing_rope,
        'temporal_gru': temporal_gru,
        'attention': attention,
        'feature_dim': feature_dim,"""
        
        new_return = """    return {
        'nli_extractor': nli_extractor,
        'framing_rope': framing_rope,
        'temporal_gru': temporal_gru,
        'attention': attention,
        'feature_dim': observer_feature_dim,      # dim seen by GRU/attention
        'base_feature_dim': base_feature_dim,     # 24D logits space
        'use_rks': use_rks,
        'rks_dim': rks_dim,
        'rks_sigma': rks_sigma,"""
        
        content = content.replace(old_return, new_return)
    
    # 7. Update run_multi_observer_experiment signature
    if 'use_rks: bool = True,' not in content:
        print("✓ Adding RKS parameters to run_multi_observer_experiment...")
        old_exp_sig = """    device: str = 'cuda',
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli'
) -> Dict[int, Dict]:"""
        
        new_exp_sig = """    device: str = 'cuda',
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli',
    use_rks: bool = True,
    rks_dim: int = 512,
    rks_sigma: float = 1.0
) -> Dict[int, Dict]:"""
        
        content = content.replace(old_exp_sig, new_exp_sig)
    
    # 8. Update initialize_full_pipeline call in run_multi_observer_experiment
    if 'use_rks=use_rks' not in content:
        print("✓ Passing RKS params to initialize_full_pipeline...")
        old_init_call = """        components = initialize_full_pipeline(
            random_seed=seed,
            use_gru=use_gru,
            use_framing_rope=use_framing_rope,
            device=device,
            nli_model_name=nli_model_name
        )"""
        
        new_init_call = """        components = initialize_full_pipeline(
            random_seed=seed,
            use_gru=use_gru,
            use_framing_rope=use_framing_rope,
            device=device,
            nli_model_name=nli_model_name,
            use_rks=use_rks,
            rks_dim=rks_dim,
            rks_sigma=rks_sigma
        )"""
        
        content = content.replace(old_init_call, new_init_call)
    
    # Write patched file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("="*70)
    print("✓✓✓ complete_pipeline.py patched successfully!")
    print("\nNext step: Patch CompletePipeline class")


def patch_complete_pipeline_class(filepath: str):
    """Patch CompletePipeline class to integrate RKS."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\nPatching CompletePipeline class...")
    print("="*70)
    
    # 1. Update __init__ to add RKS attributes
    if 'self.base_feature_dim' not in content:
        print("✓ Adding RKS attributes to __init__...")
        old_init = """        self.nli = components['nli_extractor']
        self.rope = components['framing_rope']
        self.gru = components['temporal_gru']
        self.attention = components['attention']
        self.feature_dim = components['feature_dim']
        self.config = components['config']"""
        
        new_init = """        self.nli = components['nli_extractor']
        self.rope = components['framing_rope']
        self.gru = components['temporal_gru']
        self.attention = components['attention']
        
        # Dim seen by GRU/attention
        self.feature_dim = components['feature_dim']
        # Dim of NLI+provenance space (before RKS)
        self.base_feature_dim = components.get('base_feature_dim', self.feature_dim)
        
        self.use_rks = components.get('use_rks', False)
        self.rks_dim = components.get('rks_dim', self.feature_dim)
        self.rks_sigma = components.get('rks_sigma', 1.0)
        
        self.config = components['config']"""
        
        content = content.replace(old_init, new_init)
    
    # 2. Add RKS expander creation
    if 'self.rks = RandomKitchenSinksExpander' not in content:
        print("✓ Adding RKS expander creation...")
        old_metadata = """        self.metadata_indices = metadata_indices"""
        
        new_metadata = """        self.metadata_indices = metadata_indices
        
        # Create RKS expander if requested
        if self.use_rks:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\\n### Component: Random Kitchen Sinks ###")
            print(f"  Input dim: {self.base_feature_dim}")
            print(f"  Output dim: {self.rks_dim}")
            print(f"  Sigma: {self.rks_sigma}")
            self.rks = RandomKitchenSinksExpander(
                input_dim=self.base_feature_dim,
                output_dim=self.rks_dim,
                sigma=self.rks_sigma,
                seed=self.config['seed'],
                device=device,
            )
        else:
            self.rks = None"""
        
        content = content.replace(old_metadata, new_metadata)
    
    # 3. Update provenance encoder to use base_feature_dim
    content = content.replace(
        "            self.provenance = ProvenanceEncoder(\n                feature_dim=self.feature_dim,",
        "            self.provenance = ProvenanceEncoder(\n                feature_dim=self.base_feature_dim,"
    )
    
    # 4. Update process_month to apply RKS
    if '[5/7] Applying Random Kitchen Sinks' not in content:
        print("✓ Adding RKS application in process_month...")
        old_process = """        # 3. Combine (additive fusion)
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
        print("  [6/6] Computing attention...")"""
        
        new_process = """        # 3. Combine (additive fusion)
        article_tokens = multi_framing_features + prov_features
        print(f"  [3/7] Combined features (base space): {article_tokens.shape}")
        
        # 4. Optional: Framing RoPE
        if self.rope is not None:
            print("  [4/7] Applying Framing RoPE...")
            article_tokens = self.rope(article_tokens)
        else:
            print("  [4/7] Framing RoPE: SKIPPED")
        
        # 5. Optional: Random Kitchen Sinks expansion
        if self.rks is not None:
            print("  [5/7] Applying Random Kitchen Sinks expansion...")
            article_tokens = self.rks(article_tokens)
            print(f"        RKS features: {article_tokens.shape}")
        else:
            print("  [5/7] RKS expansion: SKIPPED (using base features)")
        
        # 6. Optional: Temporal GRU
        if self.gru is not None:
            print("  [6/7] Processing with Temporal GRU...")
            article_tokens = self.gru(article_tokens)
        else:
            print("  [6/7] Temporal GRU: SKIPPED")
        
        # 7. Cross-article attention
        print("  [7/7] Computing attention...")"""
        
        content = content.replace(old_process, new_process)
    
    # Write patched file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("="*70)
    print("✓✓✓ CompletePipeline class patched successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python patch_rks.py <path_to_complete_pipeline.py>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print("\n" + "="*70)
    print("PATCHING COMPLETE_PIPELINE.PY WITH RKS SUPPORT")
    print("="*70 + "\n")
    
    # First patch the module-level functions
    patch_complete_pipeline(filepath)
    
    # Then patch the CompletePipeline class
    patch_complete_pipeline_class(filepath)
    
    print("\n" + "="*70)
    print("ALL PATCHES APPLIED SUCCESSFULLY!")
    print("="*70)
    print("\nYou can now:")
    print("1. Copy rks_expansion_CORE.py to core/rks_expansion.py")
    print("2. Run experiments with RKS: python run_experiments.py")