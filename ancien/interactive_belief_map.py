"""
Interactive Belief Transformer Visualization
============================================

Visualizes:
1. Provenance token embeddings (publisher/section/country)
2. Observer attention patterns with variance
3. Cross-article geometric structure
4. Observer diversity and collapse detection

Usage:
    python interactive_belief_map.py --data_dir results/ --output belief_map.html
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
try:
    from umap import UMAP
except ImportError:
    UMAP = None  # Will handle gracefully if not installed
import pandas as pd
import argparse


@dataclass
class ObserverData:
    """Container for observer-specific data"""
    seed: int
    temperature: float
    sparsity: int
    num_heads: int
    attention_matrix: np.ndarray
    article_embeddings: np.ndarray
    provenance_tokens: Dict[str, np.ndarray]


class BeliefTransformerViz:
    """Interactive visualization for Belief Transformer interpretability"""
    
    def __init__(self, data_dir: Path, pattern: str = '*observer*.pt'):
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self.observers: List[ObserverData] = []
        self.article_metadata = []
        
    def load_observer_data(self, observer_file: Path) -> ObserverData:
        """Load a single observer's results"""
        data = torch.load(observer_file, map_location='cpu')
        
        # Handle different possible key names
        seed = data.get('random_seed', data.get('seed', 
                       int(observer_file.stem.split('_')[-1])))
        
        # Extract attention matrix
        attn = data.get('attention_matrix', data.get('attention', data.get('attn_matrix')))
        if torch.is_tensor(attn):
            attn = attn.numpy()
            
        # Get embeddings if available
        embeddings = data.get('embeddings', data.get('article_embeddings', None))
        if embeddings is not None and torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()
            
        # Get provenance tokens
        prov = data.get('provenance_tokens', {})
        if isinstance(prov, dict):
            prov = {k: v.numpy() if torch.is_tensor(v) else v 
                   for k, v in prov.items()}
        
        return ObserverData(
            seed=seed,
            temperature=data.get('temperature', 1.0),
            sparsity=data.get('sparsity', 100),
            num_heads=data.get('num_heads', 8),
            attention_matrix=attn,
            article_embeddings=embeddings,
            provenance_tokens=prov
        )
    
    def load_all_observers(self):
        """Load all observer files from data directory"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        observer_files = list(self.data_dir.glob(self.pattern))
        print(f"Found {len(observer_files)} observer files matching pattern '{self.pattern}'")
        
        if len(observer_files) == 0:
            # Try to be helpful
            all_pt_files = list(self.data_dir.glob('*.pt'))
            if all_pt_files:
                print(f"\nFound {len(all_pt_files)} .pt files in directory:")
                for f in all_pt_files[:10]:  # Show first 10
                    print(f"  - {f.name}")
                if len(all_pt_files) > 10:
                    print(f"  ... and {len(all_pt_files) - 10} more")
                print("\nTry using --pattern to filter specific files")
            raise ValueError(f"No observer files found matching pattern '{self.pattern}' in {self.data_dir}")
        
        for obs_file in observer_files:
            try:
                obs_data = self.load_observer_data(obs_file)
                self.observers.append(obs_data)
                print(f"  Loaded observer seed={obs_data.seed}")
            except Exception as e:
                print(f"  Error loading {obs_file.name}: {e}")
        
        if not self.observers:
            raise ValueError("No observer data loaded!")
    
    def compute_attention_variance(self) -> np.ndarray:
        """Compute variance in attention across observers"""
        n_observers = len(self.observers)
        n_articles = self.observers[0].attention_matrix.shape[0]
        
        # Stack all attention matrices: [n_observers, n_articles, n_articles]
        attn_stack = np.stack([obs.attention_matrix for obs in self.observers])
        
        # Variance across observers for each pair of articles
        variance_matrix = np.var(attn_stack, axis=0)
        
        return variance_matrix
    
    def compute_observer_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distances between observers"""
        n_observers = len(self.observers)
        
        # Flatten each attention matrix
        attn_flat = np.array([obs.attention_matrix.flatten() 
                             for obs in self.observers])
        
        # Compute pairwise distances
        distances = squareform(pdist(attn_flat, metric='euclidean'))
        
        return distances
    
    def plot_observer_heatmap(self) -> go.Figure:
        """Create heatmap showing observer diversity"""
        dist_matrix = self.compute_observer_distance_matrix()
        
        labels = [f"Seed {obs.seed}<br>T={obs.temperature:.1f}<br>H={obs.num_heads}"
                 for obs in self.observers]
        
        fig = go.Figure(data=go.Heatmap(
            z=dist_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            text=np.round(dist_matrix, 4),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Observer Distance Matrix<br><sub>Higher values = more diverse perspectives</sub>",
            xaxis_title="Observer",
            yaxis_title="Observer",
            height=600,
            width=800
        )
        
        return fig
    
    def plot_attention_variance_heatmap(self) -> go.Figure:
        """Create heatmap of attention variance across observers"""
        variance = self.compute_attention_variance()
        
        # Get article labels if available
        if self.article_metadata:
            labels = [f"Art {i}: {meta.get('title', '')[:30]}" 
                     for i, meta in enumerate(self.article_metadata)]
        else:
            labels = [f"Article {i}" for i in range(variance.shape[0])]
        
        fig = go.Figure(data=go.Heatmap(
            z=variance,
            x=labels,
            y=labels,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Variance")
        ))
        
        fig.update_layout(
            title="Cross-Article Attention Variance<br><sub>Variance across all observers</sub>",
            xaxis_title="Article",
            yaxis_title="Article",
            height=800,
            width=800
        )
        
        return fig
    
    def plot_provenance_embedding_space(self) -> go.Figure:
        """Visualize provenance token embeddings in 2D"""
        if not self.observers[0].provenance_tokens:
            return None
        
        prov_tokens = self.observers[0].provenance_tokens  # Use first observer
        
        # Collect all provenance embeddings
        all_embeddings = []
        all_labels = []
        all_types = []
        
        for prov_type, embeddings in prov_tokens.items():
            if embeddings is None or len(embeddings) == 0:
                continue
            
            # Reduce dimensionality
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            all_embeddings.append(embeddings)
            all_types.extend([prov_type] * len(embeddings))
            all_labels.extend([f"{prov_type}_{i}" for i in range(len(embeddings))])
        
        if not all_embeddings:
            return None
        
        # Concatenate and reduce to 2D
        X = np.vstack(all_embeddings)
        
        if X.shape[1] > 2:
            # Use UMAP if available, otherwise fall back to PCA
            if UMAP is not None:
                try:
                    reducer = UMAP(n_components=2, random_state=42)
                    X_2d = reducer.fit_transform(X)
                    method = "UMAP"
                except Exception as e:
                    print(f"  UMAP failed, using PCA: {e}")
                    reducer = PCA(n_components=2, random_state=42)
                    X_2d = reducer.fit_transform(X)
                    method = "PCA"
            else:
                reducer = PCA(n_components=2, random_state=42)
                X_2d = reducer.fit_transform(X)
                method = "PCA"
        else:
            X_2d = X
            method = "Raw"
        
        # Create scatter plot
        df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'type': all_types,
            'label': all_labels
        })
        
        fig = px.scatter(df, x='x', y='y', color='type', hover_data=['label'],
                        title=f"Provenance Token Embedding Space ({method})")
        
        fig.update_layout(height=600, width=800)
        
        return fig
    
    def plot_observer_attention_comparison(self, article_idx: int = 0) -> go.Figure:
        """Compare attention patterns for a specific article across observers"""
        n_observers = len(self.observers)
        n_articles = self.observers[0].attention_matrix.shape[0]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=(n_observers + 1) // 2,
            subplot_titles=[f"Seed {obs.seed} (T={obs.temperature:.1f})" 
                           for obs in self.observers]
        )
        
        for idx, obs in enumerate(self.observers):
            row = (idx // ((n_observers + 1) // 2)) + 1
            col = (idx % ((n_observers + 1) // 2)) + 1
            
            attention_weights = obs.attention_matrix[article_idx, :]
            
            fig.add_trace(
                go.Bar(x=list(range(n_articles)), y=attention_weights,
                      name=f"Seed {obs.seed}",
                      showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Attention Patterns for Article {article_idx} Across Observers",
            height=600,
            width=1200
        )
        
        return fig
    
    def plot_variance_distribution(self) -> go.Figure:
        """Plot distribution of attention variance"""
        variance = self.compute_attention_variance()
        
        # Flatten and remove diagonal
        mask = ~np.eye(variance.shape[0], dtype=bool)
        variance_values = variance[mask]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=variance_values,
            nbinsx=50,
            name="Variance Distribution"
        ))
        
        # Add vertical line for mean
        mean_var = np.mean(variance_values)
        fig.add_vline(x=mean_var, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_var:.6f}")
        
        fig.update_layout(
            title="Distribution of Cross-Article Attention Variance",
            xaxis_title="Variance",
            yaxis_title="Frequency",
            height=500,
            width=800
        )
        
        return fig
    
    def plot_observer_parameters(self) -> go.Figure:
        """Visualize observer parameter diversity"""
        df = pd.DataFrame([
            {
                'seed': obs.seed,
                'temperature': obs.temperature,
                'sparsity': obs.sparsity,
                'num_heads': obs.num_heads
            }
            for obs in self.observers
        ])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Temperature', 'Sparsity', 'Num Heads', 'Seed Distribution']
        )
        
        # Temperature distribution
        fig.add_trace(go.Bar(x=df.index, y=df['temperature'], name='Temperature'),
                     row=1, col=1)
        
        # Sparsity distribution
        fig.add_trace(go.Bar(x=df.index, y=df['sparsity'], name='Sparsity'),
                     row=1, col=2)
        
        # Num heads distribution
        fig.add_trace(go.Bar(x=df.index, y=df['num_heads'], name='Num Heads'),
                     row=2, col=1)
        
        # Seed scatter
        fig.add_trace(go.Scatter(x=df.index, y=df['seed'], mode='markers',
                                marker=dict(size=10), name='Seed'),
                     row=2, col=2)
        
        fig.update_layout(
            title="Observer Parameter Diversity",
            height=700,
            width=1000,
            showlegend=False
        )
        
        return fig
    
    def create_dashboard(self, output_path: Path):
        """Create complete interactive dashboard"""
        print("\n=== Creating Belief Transformer Dashboard ===\n")
        
        # Load data
        print("Loading observer data...")
        self.load_all_observers()
        
        print(f"Loaded {len(self.observers)} observers")
        print(f"Attention matrix shape: {self.observers[0].attention_matrix.shape}")
        
        # Compute summary statistics
        variance_matrix = self.compute_attention_variance()
        mean_variance = np.mean(variance_matrix)
        max_variance = np.max(variance_matrix)
        
        dist_matrix = self.compute_observer_distance_matrix()
        mean_dist = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        
        print(f"\nSummary Statistics:")
        print(f"  Mean attention variance: {mean_variance:.6f}")
        print(f"  Max attention variance: {max_variance:.6f}")
        print(f"  Mean observer distance: {mean_dist:.4f}")
        
        # Create all plots
        print("\nGenerating visualizations...")
        
        plots = []
        
        # 1. Observer diversity
        print("  1. Observer distance matrix...")
        fig1 = self.plot_observer_heatmap()
        plots.append(fig1)
        
        # 2. Attention variance
        print("  2. Attention variance heatmap...")
        fig2 = self.plot_attention_variance_heatmap()
        plots.append(fig2)
        
        # 3. Variance distribution
        print("  3. Variance distribution...")
        fig3 = self.plot_variance_distribution()
        plots.append(fig3)
        
        # 4. Observer parameters
        print("  4. Observer parameters...")
        fig4 = self.plot_observer_parameters()
        plots.append(fig4)
        
        # 5. Attention comparison (first article)
        print("  5. Attention pattern comparison...")
        fig5 = self.plot_observer_attention_comparison(article_idx=0)
        plots.append(fig5)
        
        # 6. Provenance space (if available)
        print("  6. Provenance embedding space...")
        fig6 = self.plot_provenance_embedding_space()
        if fig6 is not None:
            plots.append(fig6)
        
        # Combine into HTML
        print(f"\nSaving dashboard to {output_path}...")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Belief Transformer Dashboard</title>
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
        .plot-container {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Belief Transformer Interpretability Dashboard</h1>
        <p>Observer-dependent geometry in news bias detection</p>
    </div>
    
    <div class="stats">
        <h2>Summary Statistics</h2>
        <ul>
            <li><strong>Number of Observers:</strong> {len(self.observers)}</li>
            <li><strong>Number of Articles:</strong> {self.observers[0].attention_matrix.shape[0]}</li>
            <li><strong>Mean Attention Variance:</strong> {mean_variance:.6f}</li>
            <li><strong>Max Attention Variance:</strong> {max_variance:.6f}</li>
            <li><strong>Mean Observer Distance:</strong> {mean_dist:.4f}</li>
        </ul>
        <p><em>Higher variance = observers see different structures (good for observer-dependence thesis)</em></p>
        <p><em>Lower variance = observers collapsed to similar patterns (indicates problem)</em></p>
    </div>
"""
        
        for i, fig in enumerate(plots):
            html_content += f'<div class="plot-container" id="plot{i}"></div>\n'
        
        html_content += """
    <script>
"""
        
        for i, fig in enumerate(plots):
            json_data = fig.to_json()
            html_content += f"""
        var plot{i} = {json_data};
        Plotly.newPlot('plot{i}', plot{i}.data, plot{i}.layout);
"""
        
        html_content += """
    </script>
</body>
</html>
"""
        
        output_path.write_text(html_content)
        print(f"✓ Dashboard saved to {output_path}")
        print(f"\nOpen in browser: file://{output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Create interactive Belief Transformer visualization')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing observer .pt files')
    parser.add_argument('--output', type=str, default='belief_transformer_dashboard.html',
                       help='Output HTML file path')
    parser.add_argument('--pattern', type=str, default='*observer*.pt',
                       help='Glob pattern for observer files (e.g., "diverse_observer*.pt")')
    
    args = parser.parse_args()
    
    viz = BeliefTransformerViz(data_dir=args.data_dir, pattern=args.pattern)
    viz.create_dashboard(output_path=Path(args.output))


if __name__ == '__main__':
    main()