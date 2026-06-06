"""
Master Diagnostic Script

Runs complete diagnostic workflow:
1. Check if observer files exist
2. Run comprehensive diagnostics
3. Run component isolation tests (if needed)
4. Generate final verdict

Usage:
    python master_diagnostics.py
    python master_diagnostics.py --with-isolation  # Also test components
"""

import sys
from pathlib import Path
import argparse

# Check Python version
if sys.version_info < (3, 7):
    print("Error: Python 3.7+ required")
    sys.exit(1)

# Check dependencies
try:
    import torch
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.stats import kendalltau
    from sklearn.cluster import KMeans
    print("✓ All dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install torch numpy pandas scipy scikit-learn")
    sys.exit(1)


def check_observer_files(pattern='outputs/real_observer_*.pt'):
    """Check if observer files exist."""
    files = list(Path('.').glob(pattern))
    return files


def run_comprehensive_diagnostics():
    """Run comprehensive diagnostics on existing observers."""
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE DIAGNOSTICS")
    print("="*70)
    
    try:
        from comprehensive_diagnostics import run_comprehensive_diagnostics
        results = run_comprehensive_diagnostics(
            observer_pattern='outputs/real_observer_*.pt',
            output_dir='outputs/diagnostics'
        )
        return results
    except Exception as e:
        print(f"✗ Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_component_tests():
    """Run component isolation tests."""
    print("\n" + "="*70)
    print("RUNNING COMPONENT ISOLATION TESTS")
    print("="*70)
    
    try:
        from component_isolation_tests import run_isolation_tests, load_articles
        
        print("Loading articles...")
        articles = load_articles('data/scraped_articles.jsonl')
        
        # Use small sample for quick testing
        articles = articles[:500]
        print(f"Using {len(articles)} articles for quick test")
        
        results = run_isolation_tests(
            articles=articles,
            seeds=[42, 43, 44],
            device='cuda',
            output_dir='outputs/isolation_tests'
        )
        return results
    except Exception as e:
        print(f"✗ Error running component tests: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_final_verdict(diagnostic_results, isolation_results=None):
    """Generate final verdict and recommendations."""
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if diagnostic_results is None:
        print("\n✗ Cannot generate verdict - diagnostics failed")
        return
    
    summary = diagnostic_results['summary']
    
    print(f"\n{summary['verdict']}")
    print(f"\n{summary['interpretation']}")
    
    # Key metrics
    print("\n" + "="*70)
    print("KEY METRICS")
    print("="*70)
    
    attn_var = diagnostic_results['attention_variance']['mean_variance']
    proc_dist = diagnostic_results['procrustes']['mean_distance']
    nn_disagree = diagnostic_results['nn_disagreement']['mean_disagreement']
    
    print(f"\n1. Attention Variance:      {attn_var:.6f}")
    print(f"   Target: > 0.001 for diversity")
    print(f"   Status: {'✓ PASS' if attn_var > 0.001 else '✗ FAIL'}")
    
    print(f"\n2. Procrustes Distance:     {proc_dist:.4f}")
    print(f"   Target: > 0.1 for different geometries")
    print(f"   Status: {'✓ PASS' if proc_dist > 0.1 else '✗ FAIL'}")
    
    print(f"\n3. NN Disagreement:         {nn_disagree:.4f}")
    print(f"   Target: > 0.2 for structural differences")
    print(f"   Status: {'✓ PASS' if nn_disagree > 0.2 else '✗ FAIL'}")
    
    # Count passes
    passes = sum([
        attn_var > 0.001,
        proc_dist > 0.1,
        nn_disagree > 0.2
    ])
    
    # Thesis viability
    print("\n" + "="*70)
    print("THESIS VIABILITY")
    print("="*70)
    
    if passes >= 2:
        print("\n✓✓ THESIS VIABLE")
        print("\nMultiple metrics show observer-dependent geometry.")
        print("You have evidence that different observers create")
        print("measurably different semantic structures.")
        
        print("\nStrength of evidence:")
        if passes == 3:
            print("  STRONG: All three metrics show diversity")
        else:
            print("  MODERATE: Two of three metrics show diversity")
        
    elif passes == 1:
        print("\n⚠ THESIS WEAK")
        print("\nOnly one metric shows observer diversity.")
        print("Evidence is marginal - consider:")
        print("  1. Increasing observer diversity (tune σ, increase D)")
        print("  2. Using more polarized corpus")
        print("  3. Reframing thesis to focus on metric that works")
        
    else:
        print("\n✗ THESIS NOT SUPPORTED")
        print("\nNo metrics show observer diversity.")
        print("Observers produce nearly identical results.")
        
        if summary['failure_mode']:
            print(f"\nLikely cause: {summary['failure_mode']}")
        
        print("\nRecommended actions:")
        print("  1. Check RKS σ auto-estimation")
        print("  2. Run component isolation tests")
        print("  3. Test on synthetic corpus with known structure")
        print("  4. Consider pivot: maybe this domain HAS platonic center")
    
    # Component isolation analysis
    if isolation_results:
        print("\n" + "="*70)
        print("COMPONENT ANALYSIS")
        print("="*70)
        
        rks_var = isolation_results['rks_only']['mean']
        full_var = isolation_results['full']['mean']
        
        print(f"\nRKS alone:      {rks_var:.6f}")
        print(f"Full pipeline:  {full_var:.6f}")
        
        if rks_var < 0.0001:
            print("\n✗ RKS is not creating variance")
            print("  Check: σ auto-estimation and RKS correlation")
        elif full_var < rks_var:
            print("\n⚠ Pipeline is REDUCING variance")
            print("  Problem: GRU or RoPE might be collapsing features")
        else:
            print("\n✓ Components are working as expected")
    
    # Recommendations
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if passes >= 2:
        print("\n1. ✓ Proceed with analysis")
        print("2. Identify which articles drive divergence")
        print("3. Analyze semantic patterns in high-variance articles")
        print("4. Generate visualizations for thesis")
        print("5. Write up results")
        
    elif passes == 1:
        print("\n1. Investigate why two metrics fail")
        print("2. Try different σ values manually: [0.5, 2.0, 5.0, 10.0]")
        print("3. Increase RKS dimensions: 512 → 1024")
        print("4. Test on different corpus (more polarized sources)")
        
    else:
        print("\n1. Run component isolation tests:")
        print("   python component_isolation_tests.py")
        print("2. Check σ auto-estimation in logs")
        print("3. Test on synthetic corpus:")
        print("   python test_synthetic_corpus.py")
        print("4. Consider thesis pivot")
    
    # Files generated
    print("\n" + "="*70)
    print("OUTPUTS")
    print("="*70)
    
    print("\nGenerated files:")
    print("  outputs/diagnostics/diagnostic_results.json")
    print("  outputs/diagnostics/diagnostic_report.html  ← OPEN THIS")
    
    if isolation_results:
        print("  outputs/isolation_tests/isolation_summary.json")
    
    print("\nView interactive report:")
    print("  Open: outputs/diagnostics/diagnostic_report.html")


def main():
    parser = argparse.ArgumentParser(
        description='Master diagnostic script for observer variance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_diagnostics.py
  python master_diagnostics.py --with-isolation
  python master_diagnostics.py --skip-check
        """
    )
    parser.add_argument('--with-isolation', action='store_true',
                       help='Also run component isolation tests (slow)')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip checking for observer files')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MASTER DIAGNOSTIC WORKFLOW")
    print("="*70)
    
    # Step 1: Check for observer files
    if not args.skip_check:
        print("\n[1/4] Checking for observer files...")
        files = check_observer_files('outputs/real_observer_*.pt')
        
        if not files:
            print("✗ No observer files found!")
            print("\nYou need to run the experiment first:")
            print("  python run_experiments.py --mode real")
            sys.exit(1)
        
        print(f"✓ Found {len(files)} observer files")
    else:
        print("\n[1/4] Skipping file check...")
    
    # Step 2: Run comprehensive diagnostics
    print("\n[2/4] Running comprehensive diagnostics...")
    diagnostic_results = run_comprehensive_diagnostics()
    
    # Step 3: Run component isolation (optional)
    isolation_results = None
    if args.with_isolation:
        print("\n[3/4] Running component isolation tests...")
        isolation_results = run_component_tests()
    else:
        print("\n[3/4] Skipping component isolation tests (use --with-isolation to enable)")
    
    # Step 4: Generate verdict
    print("\n[4/4] Generating final verdict...")
    generate_final_verdict(diagnostic_results, isolation_results)
    
    print("\n" + "="*70)
    print("✓ DIAGNOSTIC WORKFLOW COMPLETE")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
