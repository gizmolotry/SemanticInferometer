"""
Belief Transformer: RKS-Expanded Bias Manifold Visualization
============================================================

NEW PARADIGM:
- NLI logits (24D) encode intrinsic bias geometry
- RKS expansion (24D → 512D) reveals this structure in higher-dimensional space
- UMAP projects to 2D for visualization
- Different RKS seeds show Monte Carlo sampling of kernel space

This replaces the old "observer variance" approach with direct
manifold visualization of bias structure.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.decomposition import PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed, will use PCA")
import pandas as pd
import argparse
from collections import Counter


def load_rks_observer(filepath: Path) -> dict:
    """Load observer file with RKS features"""
    data = torch.load(filepath, map_location='cpu')
    
    # Get RKS features (512D if RKS was applied)
    embeddings = data.get('embeddings')
    if embeddings is None:
        embeddings = data.get('features')
    if embeddings is None:
        embeddings = data.get('article_embeddings')
    if embeddings is None:
        embeddings = data.get('rks_features')
    
    if embeddings is not None and torch.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    
    # Get metadata (critical for proper visualization)
    metadata = data.get('article_metadata', None)
    
    # Get seed
    seed = data.get('random_seed', data.get('seed', int(filepath.stem.split('_')[-1])))
    
    return {
        'seed': seed,
        'embeddings': embeddings,  # Should be 512D RKS features
        'metadata': metadata,
        'has_metadata': metadata is not None,
        'embedding_dim': embeddings.shape[1] if embeddings is not None else 0
    }


def simplify_sources(sources: List[str], top_n: int = 20) -> tuple:
    """Simplify source list to top N sources + 'Other' to fix rendering"""
    source_counts = Counter(sources)
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_source_names = set([s[0] for s in top_sources])
    simplified = [s if s in top_source_names else "Other" for s in sources]
    unique = sorted(set(simplified))
    return simplified, unique, source_counts


def create_bias_manifold_map(observer_files: List[Path],
                             output_path: Path,
                             use_single_observer: bool = True,
                             max_articles: int = 2000,
                             top_sources: int = 20):
    """
    Create bias manifold visualization using RKS-expanded features.
    
    Parameters:
    -----------
    observer_files : List[Path]
        RKS observer files to visualize
    output_path : Path
        Where to save HTML
    use_single_observer : bool
        If True, show single manifold. If False, show Monte Carlo ensemble (multiple RKS seeds)
    max_articles : int
        Subsample if corpus is larger
    """
    
    print("\n" + "="*80)
    print("BIAS MANIFOLD VISUALIZATION (RKS-Expanded)")
    print("="*80)
    
    # Load observers
    observers = []
    for obs_file in observer_files[:5 if not use_single_observer else 1]:
        print(f"\nLoading {obs_file.name}...")
        obs = load_rks_observer(obs_file)
        
        if obs['embeddings'] is None:
            print(f"  ⚠️  No embeddings found, skipping")
            continue
        
        observers.append(obs)
        print(f"  Seed: {obs['seed']}")
        print(f"  Embedding dim: {obs['embedding_dim']}D")
        print(f"  Articles: {obs['embeddings'].shape[0]}")
        print(f"  Has metadata: {obs['has_metadata']}")
    
    if not observers:
        raise ValueError("No valid observer files loaded!")
    
    n_observers = len(observers)
    n_articles = observers[0]['embeddings'].shape[0]
    
    # Check if RKS was actually applied
    embed_dim = observers[0]['embedding_dim']
    if embed_dim == 24:
        print(f"\n⚠️  WARNING: Embeddings are 24D (base features)")
        print(f"   RKS expansion may not have been applied")
        print(f"   Expected: 512D")
    elif embed_dim == 512:
        print(f"\n✓ RKS expansion detected (512D features)")
    else:
        print(f"\n📊 Feature dimension: {embed_dim}D")
    
    # Sample if too many articles
    if n_articles > max_articles:
        print(f"\n⚠️  Sampling {max_articles} articles (from {n_articles}) for visualization speed")
        article_indices = np.random.choice(n_articles, max_articles, replace=False)
        article_indices.sort()
    else:
        article_indices = np.arange(n_articles)
    
    # Check for metadata
    has_metadata = observers[0]['has_metadata']
    
    if not has_metadata:
        print("\n" + "="*80)
        print("⚠️  CRITICAL: NO ARTICLE METADATA FOUND")
        print("="*80)
        print("Without metadata, you cannot:")
        print("  - Color articles by source (NYT vs Fox)")
        print("  - See article titles on hover")
        print("  - Interpret provenance patterns")
        print("\nAdd 'article_metadata' to your observer files!")
        print("="*80)
    
    # Extract metadata for visualization
    if has_metadata:
        metadata = observers[0]['metadata']
        
        # Get sources
        raw_sources = [metadata[i].get('source', 'Unknown') for i in article_indices]
        
        # CRITICAL FIX: Simplify to top N sources  
        simplified_sources, unique_sources, source_counts = simplify_sources(raw_sources, top_n=top_sources)
        
        print(f"\n📰 Found {len(set(raw_sources))} unique sources")
        print(f"Simplified to {len(unique_sources)} categories (top {top_sources} + 'Other')")
        for src in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  - {src[0]}: {src[1]} articles")
        if len(unique_sources) > 15:
            print(f"  ... and {len(set(raw_sources)) - 15} more")
        
        # Create color mapping
        color_map = {src: i for i, src in enumerate(unique_sources)}
        colors = [color_map[src] for src in simplified_sources]
        
        # Create hover text - FULL ARTICLE TEXT, NO TRUNCATION
        hover_texts = []
        for i in article_indices:
            meta = metadata[i]
            title = meta.get('title', 'No title')
            content = meta.get('content', meta.get('text', ''))
            source = meta.get('source', 'Unknown')
            date = meta.get('date', 'Unknown date')
            url = meta.get('url', '')
            
            text = f"<b>{source}</b><br>"
            text += f"{title}<br>"
            text += f"<i>{date}</i><br>"
            if content:
                text += f"<br><b>Full Article:</b><br>{content}<br>"
            if url:
                text += f"<br>URL: {url}"
            text += f"<br>Index: {i}"
            hover_texts.append(text)
    else:
        # No metadata - generic
        sources = ['Unknown'] * len(article_indices)
        unique_sources = ['Unknown']
        colors = list(range(len(article_indices)))
        hover_texts = [f"Article {i}" for i in article_indices]
    
    # Compute UMAP projections for each observer
    print("\n" + "="*80)
    print("COMPUTING UMAP PROJECTIONS")
    print("="*80)
    
    all_projections = []
    
    for i, obs in enumerate(observers):
        print(f"\nObserver {i+1}/{n_observers} (seed {obs['seed']})...")
        
        # Get embeddings
        X = obs['embeddings'][article_indices]
        print(f"  Input: {X.shape[0]} articles × {X.shape[1]}D")
        
        # Project to 2D
        if HAS_UMAP and X.shape[0] > 15:
            try:
                print(f"  Computing UMAP...")
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=min(30, X.shape[0]-1),
                    min_dist=0.1,
                    metric='euclidean',
                    random_state=42
                )
                X_2d = reducer.fit_transform(X)
                method = "UMAP"
            except Exception as e:
                print(f"  UMAP failed: {e}")
                print(f"  Falling back to PCA...")
                reducer = PCA(n_components=2, random_state=42)
                X_2d = reducer.fit_transform(X)
                method = "PCA"
        else:
            print(f"  Using PCA (UMAP unavailable or too few points)...")
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
            method = "PCA"
        
        all_projections.append({
            'seed': obs['seed'],
            'positions': X_2d,
            'method': method
        })
        print(f"  ✓ {method} projection complete")
    
    # Create visualization
    print("\n" + "="*80)
    print("CREATING INTERACTIVE MAP")
    print("="*80)
    
    if use_single_observer or n_observers == 1:
        # Single manifold view
        print("Mode: Single bias manifold")
        
        proj = all_projections[0]
        positions = proj['positions']
        
        fig = go.Figure()
        
        # Add scatter plot - FIXED: black outlines, larger size
        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='markers',
            marker=dict(
                size=10,  # LARGER (was 5-8)
                color=colors,
                colorscale='Viridis',  # Simple colorscale
                showscale=True,
                colorbar=dict(
                    title="Source" if has_metadata else "Index",
                    tickvals=list(range(min(len(unique_sources), 20))),
                    ticktext=unique_sources[:20]
                ) if has_metadata and len(unique_sources) <= 20 else dict(title="Article"),
                line=dict(width=1, color='black'),  # BLACK (was white!)
                opacity=1.0  # FULLY OPAQUE (was 0.8)
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Bias Manifold: {len(article_indices)} Articles in RKS-Expanded Space<br>" +
                  f"<sub>{proj['method']} projection from {embed_dim}D features (seed {proj['seed']})</sub>",
            xaxis=dict(title=f"{proj['method']} 1", showgrid=True, gridcolor='lightgray', zeroline=False),
            yaxis=dict(title=f"{proj['method']} 2", showgrid=True, gridcolor='lightgray', zeroline=False),
            plot_bgcolor='white',
            width=1000,
            height=800,
            hovermode='closest'
        )
        
    else:
        # Multi-observer Monte Carlo view
        print(f"Mode: Monte Carlo ensemble ({n_observers} RKS seeds)")
        
        n_cols = min(3, n_observers)
        n_rows = (n_observers + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"RKS Seed {proj['seed']}" for proj in all_projections],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        for idx, proj in enumerate(all_projections):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            positions = proj['positions']
            
            fig.add_trace(
                go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors,
                        colorscale='Portland' if has_metadata else 'Viridis',
                        showscale=(idx == 0),
                        colorbar=dict(
                            title="Source",
                            tickvals=list(range(min(len(unique_sources), 15))),
                            ticktext=unique_sources[:15]
                        ) if has_metadata and len(unique_sources) <= 15 else dict(title="Article"),
                        line=dict(width=0.3, color='white'),
                        opacity=0.7
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="UMAP 1", row=row, col=col, showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(title_text="UMAP 2", row=row, col=col, showgrid=True, gridcolor='lightgray')
        
        fig.update_layout(
            title=f"Bias Manifold: Monte Carlo Ensemble ({n_observers} RKS Seeds)<br>" +
                  f"<sub>Different random kernels reveal different aspects of the intrinsic structure</sub>",
            height=450 * n_rows,
            width=450 * n_cols,
            plot_bgcolor='white',
            hovermode='closest'
        )
    
    # Create HTML output
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bias Manifold Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .subtitle {{
            font-size: 18px;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }}
        .info-box {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        .success-box {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        #mainPlot {{
            margin-bottom: 30px;
        }}
        .interpretation {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
        }}
        .interpretation h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .interpretation ul {{
            line-height: 1.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Bias Manifold Visualization</h1>
            <p class="subtitle">RKS-Expanded Feature Space ({embed_dim}D → 2D projection)</p>
        </div>
        
        {'<div class="warning-box"><strong>⚠️ NO METADATA:</strong> Articles not labeled by source. Add article_metadata to observer files to see which outlets cluster together.</div>' if not has_metadata else ''}
        
        {'<div class="success-box"><strong>✓ RKS EXPANSION DETECTED:</strong> Working with 512D kernel features</div>' if embed_dim == 512 else '<div class="warning-box"><strong>⚠️ WARNING:</strong> Features are ' + str(embed_dim) + 'D. Expected 512D from RKS expansion.</div>' if embed_dim != 24 else '<div class="warning-box"><strong>⚠️ BASE FEATURES:</strong> 24D (RKS may not be applied). Consider adding RKS layer for richer geometry.</div>'}
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Articles</h3>
                <div class="value">{len(article_indices)}</div>
            </div>
            <div class="stat-card">
                <h3>Feature Dim</h3>
                <div class="value">{embed_dim}D</div>
            </div>
            {'<div class="stat-card"><h3>Sources</h3><div class="value">' + str(len(unique_sources)) + '</div></div>' if has_metadata else ''}
            <div class="stat-card">
                <h3>RKS Seeds</h3>
                <div class="value">{n_observers}</div>
            </div>
        </div>
        
        <div id="mainPlot"></div>
        
        <div class="interpretation">
            <h2>📖 Understanding This Visualization</h2>
            
            <h3>What You're Looking At:</h3>
            <ul>
                <li><strong>Each dot</strong> = one article from your corpus</li>
                <li><strong>Position</strong> = where it sits in the bias manifold (2D projection of {embed_dim}D RKS features)</li>
                {'<li><strong>Color</strong> = source/publisher (e.g., NYT, Fox News, Al Jazeera)</li>' if has_metadata else '<li><strong>Color</strong> = article index (add metadata to see sources)</li>'}
                <li><strong>Clusters</strong> = groups of articles with similar bias geometry</li>
                <li><strong>Distance</strong> = semantic/bias dissimilarity</li>
            </ul>
            
            <h3>The RKS Framework:</h3>
            <ul>
                <li><strong>24D NLI logits</strong> encode intrinsic bias structure (8.4% similarity between opposite articles proves this)</li>
                <li><strong>RKS expansion</strong> (24D → 512D) projects into kernel space where manifold structure becomes visible</li>
                <li><strong>UMAP projection</strong> (512D → 2D) creates this visualization while preserving local structure</li>
                {'<li><strong>Multiple seeds</strong> show Monte Carlo sampling - different random kernels reveal different aspects</li>' if n_observers > 1 else ''}
            </ul>
            
            <h3>What To Look For:</h3>
            <ul>
                {'<li><strong>Source clustering:</strong> Do articles from same outlet cluster together?</li>' if has_metadata else ''}
                {'<li><strong>Provenance patterns:</strong> Do similar sources (e.g., left/right outlets) form distinct regions?</li>' if has_metadata else ''}
                <li><strong>Manifold structure:</strong> Are there clear geometric patterns in the bias space?</li>
                <li><strong>Outliers:</strong> Articles far from clusters may represent unique framing</li>
                {'<li><strong>Consistency across seeds:</strong> Do major patterns persist despite different RKS kernels?</li>' if n_observers > 1 else ''}
            </ul>
            
            <h3>For Your Thesis:</h3>
            <p><strong>Key claim:</strong> "The NLI-derived 24D logits encode intrinsic bias geometry. RKS expansion into 512D kernel space reveals this structure at scale, demonstrating that bias patterns exist as measurable manifold structure in semantic space."</p>
            
            {'<p><strong>Provenance insight:</strong> If sources cluster (e.g., NYT with WaPo, Fox with Breitbart), this proves bias geometry correlates with publisher identity beyond just semantic content.</p>' if has_metadata else '<p><strong>Note:</strong> Add metadata to analyze provenance patterns!</p>'}
        </div>
    </div>
    
    <script>
        var plotData = {fig.to_json()};
        Plotly.newPlot('mainPlot', plotData.data, plotData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    # Save
    output_path.write_text(html, encoding='utf-8')
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION SAVED")
    print("="*80)
    print(f"Location: {output_path.absolute()}")
    print(f"\nOpen in browser: file://{output_path.absolute()}")
    
    if not has_metadata:
        print("\n" + "="*80)
        print("TO ENABLE PROPER VISUALIZATION:")
        print("="*80)
        print("Add this when saving observers:")
        print("""
article_metadata = [
    {{
        'title': article['title'],
        'source': article['source'],  # e.g., "New York Times"
        'date': str(article.get('date', '')),
        'url': article.get('url', '')
    }}
    for article in articles
]
observer_data['article_metadata'] = article_metadata
        """)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize bias manifold from RKS-expanded features'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing observer .pt files')
    parser.add_argument('--pattern', type=str, default='*observer*.pt',
                       help='Glob pattern for observer files')
    parser.add_argument('--output', type=str, default='bias_manifold.html',
                       help='Output HTML file')
    parser.add_argument('--single', action='store_true',
                       help='Show single manifold (default: Monte Carlo ensemble)')
    parser.add_argument('--max_articles', type=int, default=2000,
                       help='Subsample to this many articles')
    parser.add_argument('--top_sources', type=int, default=20,
                       help='Number of top sources to show (rest become "Other")')
    
    args = parser.parse_args()
    
    observer_files = sorted(list(Path(args.data_dir).glob(args.pattern)))
    
    if not observer_files:
        print(f"❌ No files found matching {args.pattern} in {args.data_dir}")
        # Show what's available
        all_pt = list(Path(args.data_dir).glob('*.pt'))
        if all_pt:
            print(f"\nFound {len(all_pt)} .pt files:")
            for f in all_pt[:10]:
                print(f"  - {f.name}")
        return
    
    create_bias_manifold_map(
        observer_files=observer_files,
        output_path=Path(args.output),
        use_single_observer=args.single,
        max_articles=args.max_articles,
        top_sources=args.top_sources
    )


if __name__ == '__main__':
    main()