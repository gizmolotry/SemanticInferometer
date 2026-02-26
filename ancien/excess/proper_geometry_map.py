"""
PROPER Observer-Dependent Geometry Map
=======================================

This is what you actually want:
- Articles positioned in 2D space by observer
- Colored by SOURCE (NYT, Fox, etc.)
- Clickable with full metadata
- Provenance visible and interpretable

Requires: Observer files WITH article_metadata
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from typing import Dict, List
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
import pandas as pd
import argparse


def load_observer_with_metadata(filepath: Path) -> dict:
    """Load observer file and check for metadata"""
    data = torch.load(filepath, map_location='cpu')
    
    seed = data.get('random_seed', data.get('seed', int(filepath.stem.split('_')[-1])))
    
    attn = data.get('attention_matrix', data.get('attention'))
    if torch.is_tensor(attn):
        attn = attn.numpy()
    
    embeddings = data.get('embeddings', data.get('article_embeddings'))
    if embeddings is not None and torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    
    # in proper_geometry_map.py, inside load_observer_with_metadata(...)
    metadata = data.get('article_metadata', data.get('metadata', None))

    
    return {
        'seed': seed,
        'temperature': data.get('temperature', 1.0),
        'attention': attn,
        'embeddings': embeddings,
        'metadata': metadata,
        'has_metadata': metadata is not None
    }


def create_proper_interactive_map(observer_files: List[Path],
                                  output_path: Path,
                                  max_observers: int = 5,
                                  max_articles: int = 500):
    """
    Create PROPER interactive map with provenance, source colors, and metadata
    """
    
    print("\n" + "="*70)
    print("PROPER OBSERVER-DEPENDENT GEOMETRY MAP")
    print("="*70)
    
    # Load observers
    observers = []
    has_metadata = True
    
    for obs_file in observer_files[:max_observers]:
        print(f"\nLoading {obs_file.name}...")
        obs = load_observer_with_metadata(obs_file)
        observers.append(obs)
        print(f"  Seed: {obs['seed']}, Articles: {obs['attention'].shape[0]}")
        
        if not obs['has_metadata']:
            print("  ⚠️  NO METADATA FOUND")
            has_metadata = False
    
    n_observers = len(observers)
    n_articles = observers[0]['attention'].shape[0]
    
    # Sample if too many articles
    if n_articles > max_articles:
        print(f"\n⚠️  Too many articles ({n_articles}), sampling {max_articles} for visualization")
        article_indices = np.random.choice(n_articles, max_articles, replace=False)
        article_indices.sort()
    else:
        article_indices = np.arange(n_articles)
    
    # Compute 2D positions
    print("\nComputing UMAP projections...")
    all_positions = []
    
    for i, obs in enumerate(observers):
        print(f"  Observer {i+1}/{n_observers} (seed {obs['seed']})...")
        
        # Use embeddings if available
        if obs['embeddings'] is not None:
            X = obs['embeddings'][article_indices]
        else:
            X = obs['attention'][article_indices][:, article_indices]
        
        # Project to 2D
        if HAS_UMAP and len(article_indices) > 15:
            try:
                reducer = UMAP(n_components=2, random_state=42, 
                             n_neighbors=min(15, len(article_indices)-1))
                X_2d = reducer.fit_transform(X)
            except:
                reducer = PCA(n_components=2)
                X_2d = reducer.fit_transform(X)
        else:
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
        
        all_positions.append(X_2d)
    
    # Prepare visualization data
    print("\nPreparing visualization...")
    
    if has_metadata and observers[0]['metadata'] is not None:
        # Extract metadata for sampled articles
        metadata = observers[0]['metadata']
        
        # Get unique sources for color mapping
        sources = [metadata[i].get('source', 'Unknown') for i in article_indices]
        unique_sources = sorted(list(set(sources)))
        
        print(f"\nFound {len(unique_sources)} unique sources:")
        for src in unique_sources[:10]:
            count = sources.count(src)
            print(f"  - {src}: {count} articles")
        if len(unique_sources) > 10:
            print(f"  ... and {len(unique_sources) - 10} more")
        
        # Create color mapping
        color_map = {src: i for i, src in enumerate(unique_sources)}
        colors = [color_map[src] for src in sources]
        
        # Hover text with metadata
        hover_texts = []
        for i in article_indices:
            meta = metadata[i]
            text = f"<b>{meta.get('source', 'Unknown')}</b><br>"
            text += f"{meta.get('title', 'No title')[:80]}...<br>"
            text += f"Date: {meta.get('date', 'Unknown')}<br>"
            text += f"Index: {i}"
            hover_texts.append(text)
    else:
        # No metadata - use generic coloring
        print("\n⚠️  NO METADATA - Using generic visualization")
        print("   Articles will be colored by index, not source")
        print("   To fix: Add 'article_metadata' to your observer files")
        
        unique_sources = ['Unknown']
        colors = list(range(len(article_indices)))
        hover_texts = [f"Article {i}" for i in article_indices]
    
    # Create subplots
    n_cols = min(3, n_observers)
    n_rows = (n_observers + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Observer Seed {obs['seed']}" for obs in observers],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # Add traces
    for idx, (obs, positions) in enumerate(zip(observers, all_positions)):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        fig.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors,
                    colorscale='Viridis' if not has_metadata else 'Portland',
                    showscale=(idx == 0),
                    colorbar=dict(
                        title="Source" if has_metadata else "Article",
                        tickvals=list(range(len(unique_sources)))[:10],
                        ticktext=unique_sources[:10] if has_metadata else []
                    ) if has_metadata else dict(title="Article Index"),
                    line=dict(width=0.5, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="UMAP 1", row=row, col=col, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="UMAP 2", row=row, col=col, showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.update_layout(
        title_text=f"Observer-Dependent Geometry: {n_observers} Perspectives on {len(article_indices)} Articles<br>" +
                   "<sub>Each observer creates a different geometric arrangement</sub>",
        height=450 * n_rows,
        width=450 * n_cols,
        plot_bgcolor='white',
        showlegend=False
    )
    
    # Compute statistics
    attn_stack = np.stack([obs['attention'] for obs in observers])
    variance_matrix = np.var(attn_stack, axis=0)
    mask = ~np.eye(n_articles, dtype=bool)
    mean_variance = variance_matrix[mask].mean()
    
    attn_flat = attn_stack.reshape(n_observers, -1)
    observer_distances = squareform(pdist(attn_flat, metric='euclidean'))
    mean_observer_dist = observer_distances[np.triu_indices_from(observer_distances, k=1)].mean()
    
    # Create HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Observer-Dependent Geometry</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 25px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .interpretation {{
            background: {'#fff3cd' if mean_variance < 0.001 else '#d4edda'};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid {'#ffc107' if mean_variance < 0.001 else '#28a745'};
        }}
        .warning {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #ffc107;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Observer-Dependent Geometry</h1>
            <p style="font-size: 18px; margin: 10px 0 0 0;">
                Proving "There Is No Platonic Center" in Rhetorical Measurement
            </p>
        </div>
        
        {'<div class="warning"><strong>⚠️ NO METADATA:</strong> Articles not labeled by source. Add article_metadata to observer files for proper visualization.</div>' if not has_metadata else ''}
        
        <div class="stats">
            <div class="stat-card">
                <h3>Observers</h3>
                <div class="value">{n_observers}</div>
            </div>
            <div class="stat-card">
                <h3>Articles</h3>
                <div class="value">{len(article_indices)}</div>
            </div>
            <div class="stat-card">
                <h3>Mean Variance</h3>
                <div class="value">{mean_variance:.6f}</div>
            </div>
            <div class="stat-card">
                <h3>Observer Distance</h3>
                <div class="value">{mean_observer_dist:.2f}</div>
            </div>
            {'<div class="stat-card"><h3>Sources</h3><div class="value">' + str(len(unique_sources)) + '</div></div>' if has_metadata else ''}
        </div>
        
        <div class="interpretation">
            <h2>🎯 What This Shows</h2>
            <p><strong>Observer-Dependent Geometry:</strong> Each panel shows how ONE observer arranges articles in semantic space.</p>
            <p><strong>Your Thesis:</strong> Different observers create measurably different geometric structures. There is no single "correct" arrangement.</p>
            
            <h3>Current Status:</h3>
            {'<p>⚠️ <strong>Observer Collapse Detected:</strong> Mean variance = ' + f'{mean_variance:.6f}' + ' (< 0.001) indicates observers see nearly identical patterns. Need extreme seed spacing and architectural diversity.</p>' if mean_variance < 0.001 else '<p>✅ <strong>Observer Diversity Achieved:</strong> Mean variance = ' + f'{mean_variance:.6f}' + ' shows measurable observer-dependent geometry.</p>'}
            
            <p><strong>Observer Distance:</strong> {mean_observer_dist:.2f} - {'Low (observers too similar)' if mean_observer_dist < 0.1 else 'Good (observers show diversity)'}</p>
        </div>
        
        <div id="mainPlot"></div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3>📖 How to Read This Map</h3>
            <ul>
                <li><strong>Each dot</strong> = one article</li>
                <li><strong>Position</strong> = where this observer places it in semantic space</li>
                {'<li><strong>Color</strong> = source/publisher (e.g., NYT vs Fox News)</li>' if has_metadata else '<li><strong>Color</strong> = article index (add metadata for source colors)</li>'}
                <li><strong>Hover</strong> = see article details</li>
                <li><strong>Compare panels</strong> = see how different observers create different geometries</li>
            </ul>
            
            <h3>🎓 For Your Thesis</h3>
            <p>If articles move between panels: <strong>Observer-dependent geometry demonstrated.</strong></p>
            <p>If articles stay in same positions: <strong>Observer collapse detected - fix needed.</strong></p>
        </div>
    </div>
    
    <script>
        var plotData = {fig.to_json()};
        Plotly.newPlot('mainPlot', plotData.data, plotData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    output_path.write_text(html, encoding='utf-8')
    
    print(f"\n" + "="*70)
    print("✅ SAVED")
    print("="*70)
    print(f"File: {output_path.absolute()}")
    print(f"\nOpen in browser: file://{output_path.absolute()}")
    
    if not has_metadata:
        print("\n" + "="*70)
        print("⚠️  TO GET PROPER VISUALIZATION:")
        print("="*70)
        print("Add this to your observer saving code:")
        print("""
article_metadata = [
    {
        'title': article['title'],
        'source': article['source'],  # NYT, Fox, etc.
        'date': article['date'],
        'url': article.get('url', '')
    }
    for article in articles
]
observer_data['article_metadata'] = article_metadata
        """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pattern', type=str, default='*observer*.pt')
    parser.add_argument('--output', type=str, default='proper_geometry_map.html')
    parser.add_argument('--max_observers', type=int, default=5)
    parser.add_argument('--max_articles', type=int, default=500,
                       help='Subsample to this many articles for visualization speed')
    
    args = parser.parse_args()
    
    observer_files = sorted(list(Path(args.data_dir).glob(args.pattern)))
    
    if not observer_files:
        print(f"ERROR: No files found matching {args.pattern}")
        return
    
    create_proper_interactive_map(
        observer_files=observer_files,
        output_path=Path(args.output),
        max_observers=args.max_observers,
        max_articles=args.max_articles
    )


if __name__ == '__main__':
    main()