"""
Observer-Dependent Geometry Visualization
==========================================

Shows how different observers create different geometric arrangements of articles.
This is the INTERPRETABLE visualization you need for your thesis.

Each observer creates a different "geometry" - articles are positioned differently
in semantic space depending on the observer's perspective.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed, using PCA instead")
import pandas as pd
import argparse


def load_observer_data(filepath: Path) -> dict:
    """Load a single observer file"""
    data = torch.load(filepath, map_location='cpu')
    
    # Get seed
    seed = data.get('random_seed', data.get('seed', int(filepath.stem.split('_')[-1])))
    
    # Get attention matrix
    attn = data.get('attention_matrix', data.get('attention', data.get('attn_matrix')))
    if torch.is_tensor(attn):
        attn = attn.numpy()
    
    # Get embeddings
    embeddings = data.get('embeddings', data.get('article_embeddings', None))
    if embeddings is not None and torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    
    return {
        'seed': seed,
        'temperature': data.get('temperature', 1.0),
        'attention': attn,
        'embeddings': embeddings
    }


def compute_observer_geometry(attention_matrix: np.ndarray, 
                              embeddings: np.ndarray = None,
                              method: str = 'attention') -> np.ndarray:
    """
    Compute 2D positions for articles based on observer's perspective.
    
    Methods:
    - 'attention': Use attention matrix as distance matrix
    - 'embeddings': Use embeddings directly (if available)
    """
    
    if method == 'embeddings' and embeddings is not None:
        # Use embeddings directly
        X = embeddings
    else:
        # Use attention matrix
        # Convert attention to distance: high attention = close, low attention = far
        # Distance = 1 - attention (normalized)
        distances = 1 - attention_matrix
        X = distances
    
    # Reduce to 2D
    if HAS_UMAP and X.shape[0] > 15:  # UMAP needs enough points
        try:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, X.shape[0]-1))
            X_2d = reducer.fit_transform(X)
            method_used = "UMAP"
        except Exception as e:
            print(f"  UMAP failed: {e}, using PCA")
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
            method_used = "PCA"
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        method_used = "PCA"
    
    return X_2d, method_used


def create_observer_comparison_map(observer_files: List[Path], 
                                   output_path: Path,
                                   max_observers: int = 5):
    """
    Create the interpretable visualization showing how different observers
    arrange articles in geometric space.
    """
    
    print("\n" + "="*70)
    print("CREATING OBSERVER-DEPENDENT GEOMETRY VISUALIZATION")
    print("="*70)
    
    # Load all observers
    observers = []
    for obs_file in observer_files[:max_observers]:
        print(f"\nLoading {obs_file.name}...")
        obs_data = load_observer_data(obs_file)
        observers.append(obs_data)
        print(f"  Seed: {obs_data['seed']}")
        print(f"  Temperature: {obs_data['temperature']}")
        print(f"  Articles: {obs_data['attention'].shape[0]}")
    
    n_observers = len(observers)
    n_articles = observers[0]['attention'].shape[0]
    
    print(f"\nProcessing {n_observers} observers with {n_articles} articles")
    
    # Compute 2D positions for each observer
    print("\nComputing geometric projections...")
    all_positions = []
    methods_used = []
    
    for i, obs in enumerate(observers):
        print(f"  Observer {i+1}/{n_observers} (seed {obs['seed']})...")
        positions, method = compute_observer_geometry(
            obs['attention'], 
            obs['embeddings'],
            method='embeddings' if obs['embeddings'] is not None else 'attention'
        )
        all_positions.append(positions)
        methods_used.append(method)
    
    # Create interactive visualization
    print("\nCreating interactive plots...")
    
    # Calculate grid layout
    n_cols = min(3, n_observers)
    n_rows = (n_observers + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Observer Seed {obs['seed']}<br>T={obs['temperature']:.1f}" 
                       for obs in observers],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # Color articles consistently across observers
    # Use a simple index-based coloring for now
    colors = np.arange(n_articles)
    
    # Add each observer's geometry
    for idx, (obs, positions) in enumerate(zip(observers, all_positions)):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        fig.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale='Viridis',
                    showscale=(idx == 0),  # Show colorbar only for first plot
                    colorbar=dict(title="Article<br>Index")
                ),
                text=[f"Article {i}" for i in range(n_articles)],
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Dimension 1", row=row, col=col)
        fig.update_yaxes(title_text="Dimension 2", row=row, col=col)
    
    fig.update_layout(
        title_text=f"Observer-Dependent Geometry: {n_observers} Different Perspectives<br>" +
                   "<sub>Each panel shows how one observer arranges articles in semantic space</sub>",
        height=400 * n_rows,
        width=400 * n_cols,
        showlegend=False
    )
    
    # Compute observer variance statistics
    print("\nComputing variance statistics...")
    
    # Stack all attention matrices
    attn_stack = np.stack([obs['attention'] for obs in observers])
    variance_matrix = np.var(attn_stack, axis=0)
    
    # Remove diagonal
    mask = ~np.eye(n_articles, dtype=bool)
    variance_values = variance_matrix[mask]
    
    mean_variance = variance_values.mean()
    max_variance = variance_values.max()
    
    # Compute pairwise observer distances
    attn_flat = attn_stack.reshape(n_observers, -1)
    observer_distances = squareform(pdist(attn_flat, metric='euclidean'))
    mean_observer_dist = observer_distances[np.triu_indices_from(observer_distances, k=1)].mean()
    
    print(f"\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Mean attention variance: {mean_variance:.6f}")
    print(f"Max attention variance: {max_variance:.6f}")
    print(f"Mean observer distance: {mean_observer_dist:.4f}")
    
    # Interpretation
    print(f"\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if mean_variance < 0.0001:
        print("⚠️  OBSERVER COLLAPSE DETECTED")
        print("   Mean variance < 0.0001 indicates observers are nearly identical")
        print("   → Need extreme seed spacing and architectural diversity")
    elif mean_variance < 0.001:
        print("⚠️  LOW OBSERVER DIVERSITY")
        print("   Mean variance < 0.001 indicates limited observer-dependence")
        print("   → Consider increasing architectural diversity")
    else:
        print("✓  OBSERVER DIVERSITY ACHIEVED")
        print(f"   Mean variance = {mean_variance:.6f} shows measurable observer-dependence")
        print("   → Different observers see different geometric structures")
    
    if mean_observer_dist < 0.01:
        print("\n⚠️  Observers are extremely similar (distance < 0.01)")
    elif mean_observer_dist < 0.1:
        print(f"\n⚠️  Observers show limited diversity (distance = {mean_observer_dist:.4f})")
    else:
        print(f"\n✓  Observers show good diversity (distance = {mean_observer_dist:.4f})")
    
    # Save HTML
    print(f"\n" + "="*70)
    print("SAVING")
    print("="*70)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Observer-Dependent Geometry</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .interpretation {{
            background-color: {'#fff3cd' if mean_variance < 0.001 else '#d4edda'};
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid {'#ffc107' if mean_variance < 0.001 else '#28a745'};
        }}
        .plot-container {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Observer-Dependent Geometry Visualization</h1>
        <p>Demonstrating that "there is no platonic center" in rhetorical measurement</p>
    </div>
    
    <div class="stats">
        <h2>Summary Statistics</h2>
        <ul>
            <li><strong>Number of Observers:</strong> {n_observers}</li>
            <li><strong>Number of Articles:</strong> {n_articles}</li>
            <li><strong>Mean Attention Variance:</strong> {mean_variance:.6f}</li>
            <li><strong>Max Attention Variance:</strong> {max_variance:.6f}</li>
            <li><strong>Mean Observer Distance:</strong> {mean_observer_dist:.4f}</li>
            <li><strong>Projection Method:</strong> {methods_used[0]}</li>
        </ul>
    </div>
    
    <div class="interpretation">
        <h2>Interpretation</h2>
        <p><strong>What this shows:</strong> Each panel represents one observer's "view" of how articles relate to each other in semantic space.</p>
        <p><strong>Observer-dependence:</strong> If articles appear in different positions across panels, this demonstrates observer-dependent geometry.</p>
        <p><strong>Your thesis:</strong> Different observers create measurably different geometric structures, proving "there is no platonic center" in rhetorical measurement.</p>
        
        <h3>Current Status:</h3>
        <p>{'⚠️ <strong>Observer collapse detected.</strong> Variance is extremely low, indicating observers are nearly identical. Need to implement extreme seed spacing and architectural diversity.' if mean_variance < 0.0001 else '✓ <strong>Observer diversity achieved.</strong> Variance of ' + f'{mean_variance:.6f}' + ' demonstrates measurable observer-dependent geometry.'}</p>
    </div>
    
    <div class="plot-container" id="mainPlot"></div>
    
    <script>
        var plotData = {fig.to_json()};
        Plotly.newPlot('mainPlot', plotData.data, plotData.layout);
    </script>
</body>
</html>
"""
    
    output_path.write_text(html_content, encoding='utf-8')
    print(f"✓ Saved to {output_path}")
    print(f"\nOpen in browser: file://{output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Create observer-dependent geometry visualization'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing observer .pt files')
    parser.add_argument('--pattern', type=str, default='*observer*.pt',
                       help='Glob pattern for observer files')
    parser.add_argument('--output', type=str, default='observer_geometry.html',
                       help='Output HTML file')
    parser.add_argument('--max_observers', type=int, default=5,
                       help='Maximum number of observers to visualize')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    observer_files = sorted(list(data_dir.glob(args.pattern)))
    
    if not observer_files:
        print(f"ERROR: No files found matching {args.pattern} in {data_dir}")
        # Show what's available
        all_pt = list(data_dir.glob('*.pt'))
        if all_pt:
            print(f"\nFound {len(all_pt)} .pt files:")
            for f in all_pt[:10]:
                print(f"  - {f.name}")
        return
    
    print(f"Found {len(observer_files)} observer files")
    
    create_observer_comparison_map(
        observer_files=observer_files,
        output_path=Path(args.output),
        max_observers=args.max_observers
    )


if __name__ == '__main__':
    main()
