"""
Control Experiment: Raw Attention Matrix Test

Tests whether observer variance comes from:
1. Semantic structure in DeBERTa features (GOOD - what we want)
2. Just the random attention mechanism itself (BAD - artifact)

Approach:
- Generate random/uniform feature matrices (no semantic content)
- Run same observer architectures on these
- Compare variance to real corpus

If Real variance > Control variance → semantic signal
If Real variance ≈ Control variance → just architectural artifact
"""

import torch
import numpy as np
from pathlib import Path
import json
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import ttest_ind
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not installed, using numpy fallbacks")
    HAS_SCIPY = False
    
    def pdist(X, metric='euclidean'):
        """Simple fallback for pdist"""
        n = len(X)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                if metric == 'cosine':
                    dist = 1 - np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                else:
                    dist = np.linalg.norm(X[i] - X[j])
                distances.append(dist)
        return np.array(distances)
    
    def ttest_ind(a, b):
        """Simple fallback for t-test"""
        from math import sqrt
        n1, n2 = len(a), len(b)
        mean1, mean2 = np.mean(a), np.mean(b)
        var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_se = sqrt(var1/n1 + var2/n2)
        t_stat = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0
        # Simple p-value approximation
        p_value = 0.05 if abs(t_stat) > 2 else 0.5
        return t_stat, p_value


class SimpleAttention(nn.Module):
    """Simple attention mechanism for control - no custom dependencies"""
    def __init__(self, feature_dim, num_heads=8, temperature=1.0, top_k=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.temperature = temperature
        self.top_k = top_k
        
    def forward(self, x):
        # x shape: (batch, n_articles, feature_dim) or (n_articles, feature_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim
        
        # Scale by temperature
        x_scaled = x / self.temperature
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            x_scaled, x_scaled, x_scaled,
            need_weights=True,
            average_attn_weights=True  # Average over heads
        )
        
        # attn_weights: (batch, n_articles, n_articles)
        attn_matrix = attn_weights.squeeze(0)  # Remove batch dim
        
        # Apply sparsity if needed
        if self.top_k is not None and self.top_k < attn_matrix.shape[0]:
            # Keep only top-k per row
            topk_values, topk_indices = torch.topk(attn_matrix, k=self.top_k, dim=-1)
            sparse_matrix = torch.zeros_like(attn_matrix)
            sparse_matrix.scatter_(-1, topk_indices, topk_values)
            # Re-normalize
            attn_matrix = sparse_matrix / (sparse_matrix.sum(dim=-1, keepdim=True) + 1e-8)
        
        return attn_matrix


def generate_control_matrices(n_articles=4054, feature_dim=8192, control_type='random'):
    """
    Generate control feature matrices with NO semantic structure
    
    Args:
        n_articles: Number of "fake articles" 
        feature_dim: Dimension matching DeBERTa features
        control_type: Type of control
            'random': Random Gaussian noise
            'uniform': Uniform random
            'constant': All identical vectors
            'identity': Diagonal structure
    
    Returns:
        torch.Tensor: (n_articles, feature_dim)
    """
    print(f"\nGenerating {control_type} control matrix: ({n_articles} x {feature_dim})")
    
    if control_type == 'random':
        # Random Gaussian (mean=0, std=1)
        features = torch.randn(n_articles, feature_dim)
        
    elif control_type == 'uniform':
        # Uniform random [0, 1]
        features = torch.rand(n_articles, feature_dim)
        
    elif control_type == 'constant':
        # All articles identical
        base_vector = torch.randn(feature_dim)
        features = base_vector.unsqueeze(0).repeat(n_articles, 1)
        
    elif control_type == 'identity':
        # One-hot encoding (extreme orthogonality)
        # Pad/truncate to feature_dim
        features = torch.zeros(n_articles, feature_dim)
        for i in range(min(n_articles, feature_dim)):
            features[i, i] = 1.0
            
    else:
        raise ValueError(f"Unknown control_type: {control_type}")
    
    # Normalize (like DeBERTa features would be)
    features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    print(f"  Pairwise cosine distances (first 100):")
    
    # Check diversity
    sample_features = features[:100].numpy()
    distances = pdist(sample_features, metric='cosine')
    print(f"    Mean: {distances.mean():.4f}, Std: {distances.std():.4f}")
    
    return features


def run_observer_on_control(features, observer_config, device='cuda'):
    """
    Run single observer on control features
    
    Args:
        features: (n_articles, feature_dim) tensor
        observer_config: dict with {seed, num_heads, temperature, top_k}
        
    Returns:
        attention_matrix: (n_articles, n_articles)
    """
    seed = observer_config['seed']
    num_heads = observer_config.get('num_heads', 8)
    temperature = observer_config.get('temperature', 1.0)
    top_k = observer_config.get('top_k', None)
    
    print(f"\n  Observer {seed}: heads={num_heads}, temp={temperature}, top_k={top_k}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create attention mechanism
    attention = SimpleAttention(
        feature_dim=features.shape[1],
        num_heads=num_heads,
        temperature=temperature,
        top_k=top_k
    ).to(device)
    
    # Freeze
    for param in attention.parameters():
        param.requires_grad = False
    
    # Run attention
    features = features.to(device)
    with torch.no_grad():
        attn_matrix = attention(features)
        attn_matrix = attn_matrix.cpu()
    
    # Statistics
    attn_copy = attn_matrix.clone()
    attn_copy.fill_diagonal_(0)
    print(f"    Attention stats:")
    print(f"      Mean: {attn_matrix.mean():.6f}")
    print(f"      Std: {attn_matrix.std():.6f}")
    print(f"      Off-diagonal mean: {attn_copy.mean():.6f}")
    
    return attn_matrix.numpy()


def compute_attention_variance(attention_matrices):
    """
    Compute variance across observer attention matrices
    
    Args:
        attention_matrices: dict {seed: (n, n) array}
    
    Returns:
        variance_metrics: dict with various variance measures
    """
    # Stack all matrices
    seeds = sorted(attention_matrices.keys())
    matrices = np.stack([attention_matrices[s] for s in seeds])  # (n_observers, n_articles, n_articles)
    
    # Variance per article pair (across observers)
    pointwise_variance = matrices.var(axis=0)  # (n_articles, n_articles)
    
    # Mean variance (excluding diagonal)
    n = pointwise_variance.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_variance = pointwise_variance[mask].mean()
    
    # Pairwise observer differences
    pairwise_diffs = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            diff = np.abs(matrices[i] - matrices[j])[mask].mean()
            pairwise_diffs.append(diff)
    
    return {
        'mean_pointwise_variance': mean_variance,
        'mean_pairwise_difference': np.mean(pairwise_diffs),
        'std_pairwise_difference': np.std(pairwise_diffs),
        'pairwise_differences': pairwise_diffs
    }


def run_control_experiment(control_types=['random', 'uniform', 'constant'], 
                          n_articles=4054,
                          feature_dim=8192,
                          observer_configs=None,
                          device='cuda'):
    """
    Run full control experiment across multiple control types
    """
    if observer_configs is None:
        # Use same diverse observer configs as real experiment
        observer_configs = [
            {'seed': 1, 'num_heads': 8, 'temperature': 0.5, 'top_k': 200},
            {'seed': 1000, 'num_heads': 16, 'temperature': 1.0, 'top_k': 500},
            {'seed': 100000, 'num_heads': 32, 'temperature': 2.0, 'top_k': 1000},
            {'seed': 1000000, 'num_heads': 64, 'temperature': 0.1, 'top_k': 100},
            {'seed': 10000000, 'num_heads': 128, 'temperature': 5.0, 'top_k': 2000},
        ]
    
    results = {}
    
    for control_type in control_types:
        print(f"\n{'='*70}")
        print(f"CONTROL TYPE: {control_type.upper()}")
        print(f"{'='*70}")
        
        # Generate control features
        features = generate_control_matrices(
            n_articles=n_articles,
            feature_dim=feature_dim,
            control_type=control_type
        )
        
        # Run each observer
        attention_matrices = {}
        for config in observer_configs:
            attn = run_observer_on_control(features, config, device=device)
            attention_matrices[config['seed']] = attn
        
        # Compute variance
        variance_metrics = compute_attention_variance(attention_matrices)
        
        results[control_type] = {
            'variance_metrics': variance_metrics,
            'attention_matrices': attention_matrices  # Save for later analysis
        }
        
        print(f"\n  RESULTS for {control_type}:")
        print(f"    Mean pointwise variance: {variance_metrics['mean_pointwise_variance']:.6f}")
        print(f"    Mean pairwise difference: {variance_metrics['mean_pairwise_difference']:.6f}")
    
    return results


def compare_real_vs_control(real_results_dir='outputs', control_results=None):
    """
    Load real corpus results and compare to control
    
    Args:
        real_results_dir: Directory with real corpus observer outputs
        control_results: Results from run_control_experiment()
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: REAL vs CONTROL")
    print(f"{'='*70}")
    
    # Load real corpus attention matrices
    real_attention = {}
    observer_files = list(Path(real_results_dir).glob('diverse_observer_*.pt'))
    
    if len(observer_files) == 0:
        print(f"\nWARNING: No real corpus results found in {real_results_dir}")
        print("Run real corpus experiment first:")
        print("  python run_diverse_experiments.py --mode diverse --corpus real")
        return
    
    for obs_file in observer_files:
        data = torch.load(obs_file)
        
        # Try different possible key names for seed
        if 'random_seed' in data:
            seed = data['random_seed']
        elif 'seed' in data:
            seed = data['seed']
        elif 'observer_seed' in data:
            seed = data['observer_seed']
        else:
            # Extract from filename: diverse_observer_42.pt -> 42
            seed = int(obs_file.stem.split('_')[-1])
            print(f"  Warning: No seed key found, extracted {seed} from filename")
        
        # Try different possible key names for attention matrix
        if 'attention_matrix' in data:
            attn_matrix = data['attention_matrix']
        elif 'attention' in data:
            attn_matrix = data['attention']
        elif 'attn_matrix' in data:
            attn_matrix = data['attn_matrix']
        else:
            print(f"  ERROR: Can't find attention matrix in {obs_file.name}")
            print(f"  Available keys: {list(data.keys())}")
            continue
        
        # Convert to numpy if needed
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        
        real_attention[seed] = attn_matrix
    
    print(f"Loaded {len(real_attention)} real corpus observers")
    
    # Compute real variance
    real_variance = compute_attention_variance(real_attention)
    
    print(f"\nREAL CORPUS:")
    print(f"  Mean pointwise variance: {real_variance['mean_pointwise_variance']:.6f}")
    print(f"  Mean pairwise difference: {real_variance['mean_pairwise_difference']:.6f}")
    
    # Compare to each control
    if control_results:
        print(f"\nCONTROL COMPARISONS:")
        for control_type, control_data in control_results.items():
            ctrl_variance = control_data['variance_metrics']
            
            real_var = real_variance['mean_pairwise_difference']
            ctrl_var = ctrl_variance['mean_pairwise_difference']
            ratio = real_var / ctrl_var if ctrl_var > 0 else float('inf')
            
            print(f"\n  {control_type.upper()}:")
            print(f"    Control variance: {ctrl_var:.6f}")
            print(f"    Real / Control ratio: {ratio:.2f}x")
            
            # Statistical test
            real_diffs = real_variance['pairwise_differences']
            ctrl_diffs = ctrl_variance['pairwise_differences']
            t_stat, p_value = ttest_ind(real_diffs, ctrl_diffs)
            
            print(f"    t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"    ✓ SIGNIFICANT difference (p < 0.05)")
            else:
                print(f"    ✗ No significant difference (p >= 0.05)")
    
    # Save comparison
    comparison_output = {
        'real': real_variance,
        'controls': {ct: cr['variance_metrics'] for ct, cr in control_results.items()} if control_results else {}
    }
    
    output_path = Path(real_results_dir) / 'real_vs_control_attention.json'
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(x) for x in obj]
            return obj
        
        json.dump(convert_numpy(comparison_output), f, indent=2)
    
    print(f"\nSaved comparison to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-types', nargs='+', 
                       default=['random', 'uniform', 'constant'],
                       help='Types of control matrices to test')
    parser.add_argument('--n-articles', type=int, default=4054,
                       help='Number of articles (match real corpus)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only compare existing results, skip control generation')
    
    args = parser.parse_args()
    
    if not args.compare_only:
        # Run control experiments
        print(f"\n{'#'*70}")
        print(f"# RAW ATTENTION CONTROL EXPERIMENT")
        print(f"# Testing if observer variance is semantic or architectural")
        print(f"#{'#'*70}\n")
        
        control_results = run_control_experiment(
            control_types=args.control_types,
            n_articles=args.n_articles,
            device=args.device
        )
        
        # Save control results
        output_path = Path('outputs') / 'control_attention_results.pt'
        torch.save(control_results, output_path)
        print(f"\nSaved control results to: {output_path}")
    else:
        # Load existing control results
        control_path = Path('outputs') / 'control_attention_results.pt'
        if control_path.exists():
            control_results = torch.load(control_path)
            print(f"Loaded control results from: {control_path}")
        else:
            print(f"ERROR: No control results found at {control_path}")
            print("Run without --compare-only first")
            sys.exit(1)
    
    # Compare to real corpus
    compare_real_vs_control(control_results=control_results)