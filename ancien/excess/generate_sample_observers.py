"""
Generate sample observer data for testing the interactive visualization

Creates synthetic observer results with varying degrees of diversity
to demonstrate the visualization capabilities.
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def generate_sample_attention_matrix(n_articles: int, seed: int, 
                                    temperature: float, sparsity: int) -> np.ndarray:
    """Generate a sample attention matrix with specified characteristics"""
    np.random.seed(seed)
    
    # Start with random attention
    attn = np.random.randn(n_articles, n_articles)
    
    # Apply temperature scaling
    attn = attn / temperature
    
    # Softmax over columns (each article attends to others)
    exp_attn = np.exp(attn - np.max(attn, axis=1, keepdims=True))
    attn = exp_attn / exp_attn.sum(axis=1, keepdims=True)
    
    # Apply sparsity (keep only top-k)
    for i in range(n_articles):
        top_k_indices = np.argsort(attn[i])[-sparsity:]
        mask = np.zeros(n_articles, dtype=bool)
        mask[top_k_indices] = True
        attn[i, ~mask] = 0
        # Renormalize
        attn[i] = attn[i] / attn[i].sum()
    
    return attn


def generate_provenance_tokens(n_publishers: int = 10, 
                               n_sections: int = 5,
                               n_countries: int = 3,
                               embed_dim: int = 64) -> dict:
    """Generate sample provenance token embeddings"""
    np.random.seed(42)
    
    return {
        'publisher': np.random.randn(n_publishers, embed_dim),
        'section': np.random.randn(n_sections, embed_dim),
        'country': np.random.randn(n_countries, embed_dim)
    }


def generate_observer_data(seed: int, temperature: float, sparsity: int, 
                          num_heads: int, n_articles: int, embed_dim: int) -> dict:
    """Generate complete observer data package"""
    
    # Generate attention matrix
    attn_matrix = generate_sample_attention_matrix(n_articles, seed, temperature, sparsity)
    
    # Generate article embeddings
    np.random.seed(seed)
    embeddings = np.random.randn(n_articles, embed_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Generate provenance tokens
    prov_tokens = generate_provenance_tokens(embed_dim=embed_dim)
    
    return {
        'random_seed': seed,
        'seed': seed,
        'temperature': temperature,
        'sparsity': sparsity,
        'num_heads': num_heads,
        'attention_matrix': torch.from_numpy(attn_matrix).float(),
        'embeddings': torch.from_numpy(embeddings).float(),
        'article_embeddings': torch.from_numpy(embeddings).float(),
        'provenance_tokens': {k: torch.from_numpy(v).float() 
                             for k, v in prov_tokens.items()}
    }


def generate_diverse_observers(output_dir: Path, n_articles: int = 50):
    """Generate a set of diverse observers to test visualization"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration matching your diverse observer setup
    observer_configs = [
        # Extreme seed spacing
        {'seed': 1, 'temperature': 0.5, 'sparsity': 100, 'num_heads': 8},
        {'seed': 1000, 'temperature': 1.0, 'sparsity': 200, 'num_heads': 16},
        {'seed': 100000, 'temperature': 2.0, 'sparsity': 500, 'num_heads': 32},
        {'seed': 1000000, 'temperature': 3.0, 'sparsity': 1000, 'num_heads': 64},
        {'seed': 10000000, 'temperature': 5.0, 'sparsity': 2000, 'num_heads': 128},
    ]
    
    print(f"Generating {len(observer_configs)} diverse observers...")
    print(f"Articles: {n_articles}")
    print(f"Output: {output_dir}\n")
    
    for config in observer_configs:
        observer_data = generate_observer_data(
            seed=config['seed'],
            temperature=config['temperature'],
            sparsity=config['sparsity'],
            num_heads=config['num_heads'],
            n_articles=n_articles,
            embed_dim=768  # DeBERTa base dimension
        )
        
        filename = f"diverse_observer_{config['seed']}.pt"
        filepath = output_dir / filename
        
        torch.save(observer_data, filepath)
        
        print(f"✓ Saved {filename}")
        print(f"    Seed: {config['seed']}")
        print(f"    Temperature: {config['temperature']}")
        print(f"    Sparsity: {config['sparsity']}")
        print(f"    Num Heads: {config['num_heads']}")
        print()


def generate_collapsed_observers(output_dir: Path, n_articles: int = 50):
    """Generate a set of collapsed observers (similar patterns) for comparison"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration showing collapse (similar seeds, same architecture)
    observer_configs = [
        {'seed': 42, 'temperature': 1.0, 'sparsity': 100, 'num_heads': 8},
        {'seed': 43, 'temperature': 1.0, 'sparsity': 100, 'num_heads': 8},
        {'seed': 44, 'temperature': 1.0, 'sparsity': 100, 'num_heads': 8},
        {'seed': 45, 'temperature': 1.0, 'sparsity': 100, 'num_heads': 8},
        {'seed': 46, 'temperature': 1.0, 'sparsity': 100, 'num_heads': 8},
    ]
    
    print(f"Generating {len(observer_configs)} collapsed observers...")
    print(f"Articles: {n_articles}")
    print(f"Output: {output_dir}\n")
    
    for config in observer_configs:
        observer_data = generate_observer_data(
            seed=config['seed'],
            temperature=config['temperature'],
            sparsity=config['sparsity'],
            num_heads=config['num_heads'],
            n_articles=n_articles,
            embed_dim=768
        )
        
        filename = f"collapsed_observer_{config['seed']}.pt"
        filepath = output_dir / filename
        
        torch.save(observer_data, filepath)
        
        print(f"✓ Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample observer data')
    parser.add_argument('--output_dir', type=str, default='sample_observers/',
                       help='Output directory for observer files')
    parser.add_argument('--n_articles', type=int, default=50,
                       help='Number of articles to simulate')
    parser.add_argument('--mode', type=str, choices=['diverse', 'collapsed', 'both'],
                       default='diverse',
                       help='Generate diverse or collapsed observers')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.mode in ['diverse', 'both']:
        print("=" * 60)
        print("GENERATING DIVERSE OBSERVERS")
        print("=" * 60)
        diverse_dir = output_dir / 'diverse'
        generate_diverse_observers(diverse_dir, args.n_articles)
    
    if args.mode in ['collapsed', 'both']:
        print("\n" + "=" * 60)
        print("GENERATING COLLAPSED OBSERVERS")
        print("=" * 60)
        collapsed_dir = output_dir / 'collapsed'
        generate_collapsed_observers(collapsed_dir, args.n_articles)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nNext step: Run visualization")
    print(f"  python interactive_belief_map.py --data_dir {output_dir / args.mode}")


if __name__ == '__main__':
    main()
