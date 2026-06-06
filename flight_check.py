"""
Flight Check - Inspect .pt experiment files

Quickly shows:
- File size
- Creation date
- Pipeline configuration (paragraph-aware, multi-kernel, etc.)
- Feature dimensions
- Number of articles
- Architecture fingerprint

Usage:
    python flight_check.py                          # Check all .pt files in outputs/
    python flight_check.py --file outputs/seed_42.pt  # Check specific file
    python flight_check.py --pattern "control_*"    # Check matching pattern
"""

import torch
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any


def get_file_info(filepath: Path) -> Dict[str, Any]:
    """Get basic file information."""
    stat = filepath.stat()
    return {
        'name': filepath.name,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M'),
        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
    }


def inspect_pt_file(filepath: Path) -> Dict[str, Any]:
    """Deep inspection of .pt file contents."""
    try:
        data = torch.load(filepath, map_location='cpu')
        
        info = get_file_info(filepath)
        
        # Extract key information
        if isinstance(data, dict):
            # Check for features/embeddings
            if 'features' in data:
                features = data['features']
                info['feature_shape'] = list(features.shape)
                info['n_articles'] = features.shape[0]
                info['feature_dim'] = features.shape[1]
            elif 'embeddings' in data:
                embeddings = data['embeddings']
                info['feature_shape'] = list(embeddings.shape)
                info['n_articles'] = embeddings.shape[0]
                info['feature_dim'] = embeddings.shape[1]
            elif 'article_tokens' in data:
                tokens = data['article_tokens']
                info['feature_shape'] = list(tokens.shape)
                info['n_articles'] = tokens.shape[0]
                info['feature_dim'] = tokens.shape[1]
            
            # Check for metadata
            if 'metadata' in data:
                metadata = data['metadata']
                if isinstance(metadata, list):
                    info['has_metadata'] = True
                    info['metadata_count'] = len(metadata)
            
            # Check for diagnostics (architecture fingerprint)
            if 'diagnostics' in data:
                diag = data['diagnostics']
                info['diagnostics'] = {
                    'nli_cache_hit': diag.get('nli_cache_hit', False),
                    'pca_var_removed': diag.get('pca_var_removed', 0.0),
                    'temporal_sorted': diag.get('temporal_sorted', False),
                }
            
            # Check for attention matrix
            if 'attention_matrix' in data:
                attn = data['attention_matrix']
                info['attention_shape'] = list(attn.shape)
        
        else:
            # It's a tensor directly
            info['feature_shape'] = list(data.shape)
            info['n_articles'] = data.shape[0] if len(data.shape) > 0 else 0
            info['feature_dim'] = data.shape[1] if len(data.shape) > 1 else 0
        
        # Architecture fingerprint based on size
        feature_dim = info.get('feature_dim', 0)
        if feature_dim == 512:
            info['pipeline_type'] = 'Multi-kernel RKS (NEW)'
            info['paragraph_aware'] = 'LIKELY YES (check size)'
        elif feature_dim == 256:
            info['pipeline_type'] = 'Single kernel RKS (OLD)'
            info['paragraph_aware'] = 'NO'
        elif feature_dim == 24:
            info['pipeline_type'] = 'NLI only (no RKS)'
            info['paragraph_aware'] = 'UNKNOWN'
        else:
            info['pipeline_type'] = f'Unknown ({feature_dim}D)'
            info['paragraph_aware'] = 'UNKNOWN'
        
        # Check if it's likely NEW or OLD based on file size
        size_mb = info['size_mb']
        n_articles = info.get('n_articles', 0)
        
        if n_articles > 0:
            mb_per_article = size_mb / n_articles
            if mb_per_article > 0.010:  # >10KB per article
                info['pipeline_generation'] = 'NEW (paragraph-aware + multi-kernel)'
            else:
                info['pipeline_generation'] = 'OLD (no paragraph-aware)'
        
        return info
        
    except Exception as e:
        info = get_file_info(filepath)
        info['error'] = str(e)
        return info


def print_file_report(filepath: Path):
    """Print detailed report for a single file."""
    print(f"\n{'='*80}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*80}")
    
    info = inspect_pt_file(filepath)
    
    # Basic info
    print(f"\n📁 FILE INFO:")
    print(f"   Path:     {filepath}")
    print(f"   Size:     {info['size_mb']:.2f} MB")
    print(f"   Created:  {info['created']}")
    print(f"   Modified: {info['modified']}")
    
    if 'error' in info:
        print(f"\n❌ ERROR: {info['error']}")
        return
    
    # Features
    if 'feature_shape' in info:
        print(f"\n📊 DATA:")
        print(f"   Articles:     {info.get('n_articles', 'N/A')}")
        print(f"   Feature dim:  {info.get('feature_dim', 'N/A')}")
        print(f"   Shape:        {info.get('feature_shape', 'N/A')}")
        
        if 'attention_shape' in info:
            print(f"   Attention:    {info['attention_shape']}")
    
    # Architecture
    print(f"\n🏗️  ARCHITECTURE:")
    print(f"   Pipeline:     {info.get('pipeline_type', 'Unknown')}")
    print(f"   Generation:   {info.get('pipeline_generation', 'Unknown')}")
    print(f"   Paragraph-aware: {info.get('paragraph_aware', 'Unknown')}")
    
    # Diagnostics
    if 'diagnostics' in info:
        print(f"\n🔧 DIAGNOSTICS:")
        diag = info['diagnostics']
        print(f"   NLI cached:   {diag.get('nli_cache_hit', False)}")
        print(f"   PCA removed:  {diag.get('pca_var_removed', 0.0)*100:.1f}%")
        print(f"   Temporal:     {diag.get('temporal_sorted', False)}")
    
    # Metadata
    if 'has_metadata' in info:
        print(f"\n📝 METADATA:")
        print(f"   Count:        {info.get('metadata_count', 0)}")


def print_comparison_table(files: list):
    """Print comparison table for multiple files."""
    print(f"\n{'='*120}")
    print("FLIGHT CHECK - BATCH COMPARISON")
    print(f"{'='*120}")
    
    # Header
    print(f"\n{'File':<50} {'Size (MB)':<12} {'Articles':<10} {'Dims':<8} {'Pipeline':<25} {'Date':<16}")
    print("-" * 120)
    
    # Process each file
    infos = []
    for filepath in files:
        info = inspect_pt_file(filepath)
        infos.append((filepath, info))
    
    # Sort by creation date
    infos.sort(key=lambda x: x[1].get('created', ''))
    
    # Print rows
    for filepath, info in infos:
        name = filepath.name[:48]
        size = f"{info['size_mb']:.1f}"
        articles = str(info.get('n_articles', 'N/A'))
        dims = str(info.get('feature_dim', 'N/A'))
        
        # Color code pipeline type
        pipeline = info.get('pipeline_generation', 'Unknown')
        if 'NEW' in pipeline:
            pipeline_display = '✅ ' + pipeline[:22]
        elif 'OLD' in pipeline:
            pipeline_display = '⚠️  ' + pipeline[:22]
        else:
            pipeline_display = '❓ ' + pipeline[:22]
        
        date = info.get('created', 'N/A')
        
        print(f"{name:<50} {size:<12} {articles:<10} {dims:<8} {pipeline_display:<25} {date:<16}")
    
    # Summary
    print(f"\n{'='*120}")
    print("SUMMARY:")
    
    new_count = sum(1 for _, info in infos if 'NEW' in info.get('pipeline_generation', ''))
    old_count = sum(1 for _, info in infos if 'OLD' in info.get('pipeline_generation', ''))
    
    print(f"   ✅ NEW pipeline (paragraph-aware + multi-kernel): {new_count} files")
    print(f"   ⚠️  OLD pipeline (no fixes): {old_count} files")
    print(f"   ❓ Unknown: {len(infos) - new_count - old_count} files")
    
    if old_count > 0:
        print(f"\n⚠️  WARNING: You have {old_count} OLD pipeline files!")
        print(f"   These should be regenerated with the new pipeline.")
        print(f"   OLD files are from before paragraph-aware + multi-kernel fixes.")


def main():
    parser = argparse.ArgumentParser(
        description='Flight check - Inspect .pt experiment files'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Specific file to inspect'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        help='Pattern to match (e.g., "control_*", "seed_*")'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        default='outputs',
        help='Directory to search (default: outputs)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed report for each file'
    )
    
    args = parser.parse_args()
    
    # Find files
    if args.file:
        files = [Path(args.file)]
    else:
        search_dir = Path(args.dir)
        pattern = args.pattern or '*.pt'
        files = sorted(search_dir.glob(pattern))
    
    if not files:
        print(f"❌ No files found matching pattern: {args.pattern or '*.pt'}")
        return
    
    # Show results
    if args.detailed or len(files) == 1:
        for filepath in files:
            print_file_report(filepath)
    else:
        print_comparison_table(files)
    
    print(f"\n{'='*120}")


if __name__ == '__main__':
    main()
