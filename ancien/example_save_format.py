"""
Minimal Example: Save Observer Data for Visualization
======================================================

This shows exactly what format your pipeline needs to output
for the visualization to work.
"""

import torch
import numpy as np


def example_save_observer_data():
    """
    Minimal example of saving observer data from your pipeline.
    Adapt this to your actual code.
    """
    
    # === FROM YOUR PIPELINE ===
    # You already have these from your observer generation:
    
    # 1. Cross-article attention matrix
    # Shape: [n_articles, n_articles]
    # Each row i shows how article i attends to all other articles
    cross_article_attention = np.random.rand(50, 50)  # Your actual attention matrix
    cross_article_attention = cross_article_attention / cross_article_attention.sum(axis=1, keepdims=True)
    
    # 2. Article embeddings from DeBERTa
    # Shape: [n_articles, 768] for DeBERTa-base
    article_embeddings = np.random.randn(50, 768)  # Your actual embeddings
    
    # 3. Provenance token embeddings (optional but recommended)
    # These are learned embeddings for publisher/section/country
    publisher_embeddings = np.random.randn(10, 64)  # n_publishers x embed_dim
    section_embeddings = np.random.randn(5, 64)     # n_sections x embed_dim
    country_embeddings = np.random.randn(3, 64)     # n_countries x embed_dim
    
    # 4. Observer configuration
    observer_seed = 1000
    observer_temperature = 1.0
    observer_sparsity = 200
    observer_num_heads = 16
    
    # === SAVE IN THIS FORMAT ===
    observer_data = {
        # Required fields
        'random_seed': observer_seed,           # Or just 'seed'
        'attention_matrix': torch.from_numpy(cross_article_attention).float(),
        
        # Optional but highly recommended
        'temperature': observer_temperature,
        'sparsity': observer_sparsity,
        'num_heads': observer_num_heads,
        
        # Optional: embeddings
        'embeddings': torch.from_numpy(article_embeddings).float(),
        # Or: 'article_embeddings': torch.from_numpy(article_embeddings).float(),
        
        # Optional: provenance tokens
        'provenance_tokens': {
            'publisher': torch.from_numpy(publisher_embeddings).float(),
            'section': torch.from_numpy(section_embeddings).float(),
            'country': torch.from_numpy(country_embeddings).float()
        }
    }
    
    # Save to file
    output_path = f'results/observers/observer_{observer_seed}.pt'
    torch.save(observer_data, output_path)
    
    print(f"✓ Saved observer data to {output_path}")
    print(f"  Attention matrix shape: {cross_article_attention.shape}")
    print(f"  Embeddings shape: {article_embeddings.shape}")


def example_integration_with_your_pipeline():
    """
    Example showing how to integrate with your existing pipeline.
    This is pseudocode - adapt to your actual implementation.
    """
    
    # Your existing pipeline code:
    """
    for observer_config in observer_configs:
        seed, temp, sparsity, heads = observer_config
        
        # Your observer generation
        observer = create_observer(seed, temp, sparsity, heads)
        
        # Process articles
        for article in articles:
            features = deberta.encode(article.text)
            # ... your processing ...
        
        # Generate cross-article attention
        attention_matrix = compute_cross_article_attention(observer, articles)
        
        # ===== ADD THIS BLOCK =====
        # Save for visualization
        observer_data = {
            'random_seed': seed,
            'temperature': temp,
            'sparsity': sparsity,
            'num_heads': heads,
            'attention_matrix': attention_matrix,  # torch.Tensor [n_articles, n_articles]
            'embeddings': article_embeddings,      # torch.Tensor [n_articles, 768]
            'provenance_tokens': provenance_tokens # dict of torch.Tensors
        }
        torch.save(observer_data, f'results/observers/observer_{seed}.pt')
        # ===== END ADDITION =====
    """
    pass


def example_loading_for_verification():
    """
    Example showing how to verify your saved data
    """
    
    # Load saved data
    data = torch.load('results/observers/observer_1000.pt')
    
    # Check what's in there
    print("Keys in saved data:", list(data.keys()))
    
    # Check shapes
    print(f"Attention matrix shape: {data['attention_matrix'].shape}")
    if 'embeddings' in data:
        print(f"Embeddings shape: {data['embeddings'].shape}")
    if 'provenance_tokens' in data:
        print(f"Provenance tokens: {list(data['provenance_tokens'].keys())}")
    
    # Verify it's valid
    attn = data['attention_matrix']
    print(f"Attention matrix:")
    print(f"  Min: {attn.min():.4f}")
    print(f"  Max: {attn.max():.4f}")
    print(f"  Row sums (should be ~1.0): {attn.sum(dim=1)[:5]}")


def minimal_pipeline_example():
    """
    Complete minimal example from scratch
    """
    
    # Simulate your pipeline
    n_articles = 50
    n_observers = 5
    
    observer_configs = [
        (1, 0.5, 100, 8),
        (1000, 1.0, 200, 16),
        (100000, 2.0, 500, 32),
        (1000000, 3.0, 1000, 64),
        (10000000, 5.0, 2000, 128)
    ]
    
    for seed, temp, sparsity, heads in observer_configs:
        # Simulate observer generation
        np.random.seed(seed)
        
        # 1. Generate random attention (your actual code will use DeBERTa)
        attention = np.random.rand(n_articles, n_articles)
        attention = attention / temp  # Temperature scaling
        attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize
        
        # 2. Generate embeddings (your actual code uses DeBERTa)
        embeddings = np.random.randn(n_articles, 768)
        
        # 3. Save
        observer_data = {
            'random_seed': seed,
            'temperature': temp,
            'sparsity': sparsity,
            'num_heads': heads,
            'attention_matrix': torch.from_numpy(attention).float(),
            'embeddings': torch.from_numpy(embeddings).float()
        }
        
        torch.save(observer_data, f'results/observers/observer_{seed}.pt')
        print(f"✓ Saved observer {seed}")


if __name__ == '__main__':
    # Run minimal example
    print("=" * 60)
    print("MINIMAL EXAMPLE: SAVING OBSERVER DATA")
    print("=" * 60)
    
    example_save_observer_data()
    
    print("\n" + "=" * 60)
    print("INTEGRATION PSEUDOCODE")
    print("=" * 60)
    print(example_integration_with_your_pipeline.__doc__)
    
    print("\n" + "=" * 60)
    print("KEY POINTS")
    print("=" * 60)
    print("""
1. REQUIRED FIELDS:
   - 'random_seed' or 'seed': int
   - 'attention_matrix': torch.Tensor of shape [n_articles, n_articles]
   
2. RECOMMENDED FIELDS:
   - 'temperature': float
   - 'sparsity': int  
   - 'num_heads': int
   
3. OPTIONAL FIELDS:
   - 'embeddings' or 'article_embeddings': torch.Tensor [n_articles, embed_dim]
   - 'provenance_tokens': dict of torch.Tensors
   
4. FILE NAMING:
   - Use pattern: observer_{seed}.pt
   - Put all in same directory
   - Visualization will find them automatically
   
5. ATTENTION MATRIX REQUIREMENTS:
   - Shape: [n_articles, n_articles]
   - Values: attention weights (typically sum to 1 per row)
   - Row i = how article i attends to all other articles
   
6. WHAT THE VISUALIZATION DOES:
   - Loads all observer_*.pt files from directory
   - Computes variance across observers
   - Shows where observers disagree (high variance)
   - Proves observer-dependence for your thesis
""")
