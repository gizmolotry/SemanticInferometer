"""
Quick runner for testing semantic vs architectural variance

This tests the KEY question:
  "Is observer variance coming from semantic structure or just random attention mechanics?"

Usage:
  python run_control_test.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Command failed with code {result.returncode}")
        return False
    else:
        print(f"\n✓ SUCCESS")
        return True

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              CONTROL EXPERIMENT: RAW ATTENTION TEST           ║
║                                                                ║
║  Tests if observer variance is:                               ║
║    A) Semantic (captures structure in text) ✓ GOOD            ║
║    B) Architectural artifact (just random noise) ✗ BAD        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    # Check if real corpus results exist
    real_results = list(Path('outputs').glob('diverse_observer_*.pt'))
    
    if len(real_results) == 0:
        print("\n⚠️  No real corpus results found!")
        print("\nYou need to run the real corpus experiment first:")
        print("  python run_diverse_experiments.py --mode diverse --corpus real")
        
        response = input("\nDo you want me to run it now? [y/N]: ")
        if response.lower() == 'y':
            success = run_command(
                ['python', 'run_diverse_experiments.py', '--mode', 'diverse', '--corpus', 'real'],
                "Real corpus with diverse observers"
            )
            if not success:
                print("\n❌ Failed to run real corpus. Fix errors and try again.")
                sys.exit(1)
        else:
            print("\nOkay, run it yourself first, then come back.")
            sys.exit(0)
    else:
        print(f"✓ Found {len(real_results)} real corpus observer results")
    
    # Run control experiment
    print("\n\nNow running control experiment (random attention matrices)...")
    success = run_command(
        ['python', 'control_raw_attention.py'],
        "Control experiment on random matrices"
    )
    
    if not success:
        print("\n❌ Control experiment failed")
        sys.exit(1)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print("\nResults saved to:")
    print("  outputs/control_attention_results.pt")
    print("  outputs/real_vs_control_attention.json")
    print("\nCheck the output above for:")
    print("  • Real variance vs Control variance")
    print("  • Statistical significance (p-value)")
    print("\nWhat you want to see:")
    print("  ✓ Real > Control (semantic structure creates variance)")
    print("  ✓ p < 0.05 (statistically significant difference)")
    print("\nWhat would be bad:")
    print("  ✗ Real ≈ Control (variance is just architectural artifact)")
    print("  ✗ p > 0.05 (no significant difference)")

if __name__ == '__main__':
    main()
