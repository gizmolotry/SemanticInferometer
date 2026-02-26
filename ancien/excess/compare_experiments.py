"""
Compare Multiple Experimental Runs
===================================

Utility for comparing:
- Different observer configurations (collapsed vs diverse)
- Real vs control corpus
- Before vs after architectural changes

Generates comparative statistics and side-by-side visualizations
"""

import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd
from dataclasses import dataclass
import argparse


@dataclass
class ExperimentStats:
    """Statistics for a single experiment run"""
    name: str
    n_observers: int
    n_articles: int
    mean_variance: float
    std_variance: float
    max_variance: float
    mean_observer_distance: float
    variance_percentiles: Dict[int, float]
    observer_configs: List[Dict]
    

class ExperimentComparator:
    """Compare multiple experimental runs"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentStats] = {}
    
    def load_experiment(self, name: str, data_dir: Path) -> ExperimentStats:
        """Load statistics from an experiment directory"""
        data_dir = Path(data_dir)
        observer_files = list(data_dir.glob('*observer*.pt'))
        
        if not observer_files:
            raise ValueError(f"No observer files found in {data_dir}")
        
        print(f"\nLoading {name}...")
        print(f"  Found {len(observer_files)} observers")
        
        # Load all observers
        attention_matrices = []
        observer_configs = []
        
        for obs_file in observer_files:
            data = torch.load(obs_file, map_location='cpu')
            
            # Get seed
            seed = data.get('random_seed', data.get('seed', 
                           int(obs_file.stem.split('_')[-1])))
            
            # Get attention
            attn = data.get('attention_matrix', data.get('attention', data.get('attn_matrix')))
            if torch.is_tensor(attn):
                attn = attn.numpy()
            
            attention_matrices.append(attn)
            
            # Get config
            config = {
                'seed': seed,
                'temperature': data.get('temperature', 1.0),
                'sparsity': data.get('sparsity', 100),
                'num_heads': data.get('num_heads', 8)
            }
            observer_configs.append(config)
        
        # Stack attention matrices
        attn_stack = np.stack(attention_matrices)  # [n_observers, n_articles, n_articles]
        
        # Compute variance
        variance = np.var(attn_stack, axis=0)
        
        # Remove diagonal for statistics
        mask = ~np.eye(variance.shape[0], dtype=bool)
        variance_values = variance[mask]
        
        # Compute observer distances
        from scipy.spatial.distance import pdist, squareform
        attn_flat = attn_stack.reshape(len(attention_matrices), -1)
        distances = squareform(pdist(attn_flat, metric='euclidean'))
        triu_indices = np.triu_indices_from(distances, k=1)
        mean_distance = distances[triu_indices].mean()
        
        # Compute percentiles
        percentiles = {
            25: np.percentile(variance_values, 25),
            50: np.percentile(variance_values, 50),
            75: np.percentile(variance_values, 75),
            95: np.percentile(variance_values, 95)
        }
        
        stats = ExperimentStats(
            name=name,
            n_observers=len(attention_matrices),
            n_articles=attention_matrices[0].shape[0],
            mean_variance=variance_values.mean(),
            std_variance=variance_values.std(),
            max_variance=variance_values.max(),
            mean_observer_distance=mean_distance,
            variance_percentiles=percentiles,
            observer_configs=observer_configs
        )
        
        print(f"  Mean variance: {stats.mean_variance:.6f}")
        print(f"  Mean observer distance: {stats.mean_observer_distance:.4f}")
        
        self.experiments[name] = stats
        return stats
    
    def compare_two_experiments(self, name1: str, name2: str, 
                                data_dir1: Path, data_dir2: Path) -> Dict:
        """Statistical comparison between two experiments"""
        
        # Load both experiments
        if name1 not in self.experiments:
            self.load_experiment(name1, data_dir1)
        if name2 not in self.experiments:
            self.load_experiment(name2, data_dir2)
        
        exp1 = self.experiments[name1]
        exp2 = self.experiments[name2]
        
        # Load full variance distributions for statistical testing
        data_dir1 = Path(data_dir1)
        data_dir2 = Path(data_dir2)
        
        # Load variance values
        def get_variance_values(data_dir):
            observer_files = list(data_dir.glob('*observer*.pt'))
            attention_matrices = []
            for f in observer_files:
                data = torch.load(f, map_location='cpu')
                attn = data.get('attention_matrix', data.get('attention', data.get('attn_matrix')))
                if torch.is_tensor(attn):
                    attn = attn.numpy()
                attention_matrices.append(attn)
            
            attn_stack = np.stack(attention_matrices)
            variance = np.var(attn_stack, axis=0)
            mask = ~np.eye(variance.shape[0], dtype=bool)
            return variance[mask]
        
        var1 = get_variance_values(data_dir1)
        var2 = get_variance_values(data_dir2)
        
        # Statistical tests
        ttest_stat, ttest_pval = ttest_ind(var1, var2)
        mw_stat, mw_pval = mannwhitneyu(var1, var2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((var1.std()**2 + var2.std()**2) / 2)
        cohens_d = (var1.mean() - var2.mean()) / pooled_std
        
        comparison = {
            'experiment_1': name1,
            'experiment_2': name2,
            'mean_variance_1': exp1.mean_variance,
            'mean_variance_2': exp2.mean_variance,
            'variance_ratio': exp1.mean_variance / exp2.mean_variance if exp2.mean_variance > 0 else float('inf'),
            'mean_distance_1': exp1.mean_observer_distance,
            'mean_distance_2': exp2.mean_observer_distance,
            'distance_ratio': exp1.mean_observer_distance / exp2.mean_observer_distance if exp2.mean_observer_distance > 0 else float('inf'),
            'ttest_statistic': ttest_stat,
            'ttest_pvalue': ttest_pval,
            'mannwhitney_statistic': mw_stat,
            'mannwhitney_pvalue': mw_pval,
            'cohens_d': cohens_d
        }
        
        return comparison
    
    def generate_comparison_report(self, comparisons: List[Dict], output_path: Path):
        """Generate markdown report comparing experiments"""
        
        report = f"""# Belief Transformer Experiment Comparison Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Experiment | N Observers | N Articles | Mean Variance | Std Variance | Mean Distance |
|------------|-------------|------------|---------------|--------------|---------------|
"""
        
        for name, stats in self.experiments.items():
            report += f"| {name} | {stats.n_observers} | {stats.n_articles} | "
            report += f"{stats.mean_variance:.6f} | {stats.std_variance:.6f} | "
            report += f"{stats.mean_observer_distance:.4f} |\n"
        
        report += "\n## Pairwise Comparisons\n\n"
        
        for comp in comparisons:
            report += f"""### {comp['experiment_1']} vs {comp['experiment_2']}

**Variance Comparison:**
- Mean variance (1): {comp['mean_variance_1']:.6f}
- Mean variance (2): {comp['mean_variance_2']:.6f}
- Ratio (1/2): {comp['variance_ratio']:.2f}x

**Observer Distance Comparison:**
- Mean distance (1): {comp['mean_distance_1']:.4f}
- Mean distance (2): {comp['mean_distance_2']:.4f}
- Ratio (1/2): {comp['distance_ratio']:.2f}x

**Statistical Tests:**
- T-test: t = {comp['ttest_statistic']:.4f}, p = {comp['ttest_pvalue']:.4e}
- Mann-Whitney U: U = {comp['mannwhitney_statistic']:.1f}, p = {comp['mannwhitney_pvalue']:.4e}
- Cohen's d (effect size): {comp['cohens_d']:.4f}

**Interpretation:**
"""
            # Add interpretation
            if comp['ttest_pvalue'] < 0.001:
                report += "- **Highly significant difference** (p < 0.001)\n"
            elif comp['ttest_pvalue'] < 0.05:
                report += "- **Significant difference** (p < 0.05)\n"
            else:
                report += "- No significant difference (p >= 0.05)\n"
            
            if abs(comp['cohens_d']) > 0.8:
                report += "- **Large effect size** (|d| > 0.8)\n"
            elif abs(comp['cohens_d']) > 0.5:
                report += "- **Medium effect size** (0.5 < |d| < 0.8)\n"
            elif abs(comp['cohens_d']) > 0.2:
                report += "- **Small effect size** (0.2 < |d| < 0.5)\n"
            else:
                report += "- Negligible effect size (|d| < 0.2)\n"
            
            report += "\n---\n\n"
        
        # Observer configuration comparison
        report += "## Observer Configuration Details\n\n"
        
        for name, stats in self.experiments.items():
            report += f"### {name}\n\n"
            report += "| Seed | Temperature | Sparsity | Num Heads |\n"
            report += "|------|-------------|----------|----------|\n"
            
            for config in stats.observer_configs:
                report += f"| {config['seed']} | {config['temperature']:.2f} | "
                report += f"{config['sparsity']} | {config['num_heads']} |\n"
            
            report += "\n"
        
        # Write report
        output_path.write_text(report)
        print(f"\n✓ Comparison report saved to {output_path}")
    
    def export_for_thesis(self, output_dir: Path):
        """Export statistics in formats suitable for thesis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        json_data = {}
        for name, stats in self.experiments.items():
            json_data[name] = {
                'n_observers': stats.n_observers,
                'n_articles': stats.n_articles,
                'mean_variance': float(stats.mean_variance),
                'std_variance': float(stats.std_variance),
                'max_variance': float(stats.max_variance),
                'mean_observer_distance': float(stats.mean_observer_distance),
                'percentiles': {str(k): float(v) for k, v in stats.variance_percentiles.items()}
            }
        
        json_path = output_dir / 'experiment_statistics.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ JSON stats saved to {json_path}")
        
        # Export as CSV for tables
        rows = []
        for name, stats in self.experiments.items():
            rows.append({
                'experiment': name,
                'n_observers': stats.n_observers,
                'mean_variance': stats.mean_variance,
                'std_variance': stats.std_variance,
                'mean_distance': stats.mean_observer_distance,
                'p25': stats.variance_percentiles[25],
                'p50': stats.variance_percentiles[50],
                'p75': stats.variance_percentiles[75],
                'p95': stats.variance_percentiles[95]
            })
        
        df = pd.DataFrame(rows)
        csv_path = output_dir / 'experiment_statistics.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"✓ CSV table saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare experimental runs')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Experiment names (e.g., "real" "control")')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                       help='Data directories for each experiment')
    parser.add_argument('--output_dir', type=str, default='comparison_results/',
                       help='Output directory for comparison reports')
    
    args = parser.parse_args()
    
    if len(args.experiments) != len(args.data_dirs):
        raise ValueError("Number of experiment names must match number of data directories")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BELIEF TRANSFORMER EXPERIMENT COMPARISON")
    print("=" * 70)
    
    # Load all experiments
    comparator = ExperimentComparator()
    
    for name, data_dir in zip(args.experiments, args.data_dirs):
        comparator.load_experiment(name, Path(data_dir))
    
    # Generate pairwise comparisons
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS")
    print("=" * 70)
    
    comparisons = []
    for i in range(len(args.experiments)):
        for j in range(i + 1, len(args.experiments)):
            name1, name2 = args.experiments[i], args.experiments[j]
            dir1, dir2 = args.data_dirs[i], args.data_dirs[j]
            
            comp = comparator.compare_two_experiments(name1, name2, dir1, dir2)
            comparisons.append(comp)
            
            print(f"\n{name1} vs {name2}:")
            print(f"  Variance ratio: {comp['variance_ratio']:.2f}x")
            print(f"  Distance ratio: {comp['distance_ratio']:.2f}x")
            print(f"  p-value: {comp['ttest_pvalue']:.4e}")
            print(f"  Cohen's d: {comp['cohens_d']:.4f}")
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORTS")
    print("=" * 70)
    
    report_path = output_dir / 'comparison_report.md'
    comparator.generate_comparison_report(comparisons, report_path)
    
    # Export for thesis
    comparator.export_for_thesis(output_dir)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
