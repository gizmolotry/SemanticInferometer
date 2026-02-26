"""
SETUP LOGITS-BASED NLI

Simple script to set up the new logits-based extraction.
Just copies files to the right places.
"""

import os
import shutil

print("="*70)
print("SETUP: LOGITS-BASED NLI EXTRACTION")
print("="*70)

# Check we're in V3
if not os.path.exists('core') or not os.path.exists('config'):
    print("\n✗ ERROR: Run this from D:\\belief-transformer\\V3")
    exit(1)

print("\nCurrent directory:", os.getcwd())

# Step 1: Copy improved queries to config/
print("\n[1/3] Setting up improved framing queries...")
src = 'framing_queries_improved.yaml'
dst = 'config/framing_queries_improved.yaml'

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"  ✓ Copied {src} → {dst}")
elif os.path.exists(f'outputs/{src}'):
    shutil.copy(f'outputs/{src}', dst)
    print(f"  ✓ Copied outputs/{src} → {dst}")
else:
    print(f"  ✗ ERROR: {src} not found!")
    print("  Download it from Claude and place in V3/ directory")
    exit(1)

# Step 2: Copy nli_extraction_logits.py to core/
print("\n[2/3] Setting up logits-based NLI extractor...")
src = 'nli_extraction_logits.py'
dst = 'core/nli_extraction_logits.py'

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"  ✓ Copied {src} → {dst}")
elif os.path.exists(f'outputs/{src}'):
    shutil.copy(f'outputs/{src}', dst)
    print(f"  ✓ Copied outputs/{src} → {dst}")
else:
    print(f"  ✗ ERROR: {src} not found!")
    print("  Download it from Claude and place in V3/ directory")
    exit(1)

# Step 3: Copy test script
print("\n[3/3] Setting up test script...")
src = 'test_logits_improved.py'

if os.path.exists(src):
    print(f"  ✓ {src} already in place")
elif os.path.exists(f'outputs/{src}'):
    shutil.copy(f'outputs/{src}', '.')
    print(f"  ✓ Copied outputs/{src} → .")
else:
    print(f"  ✗ ERROR: {src} not found!")
    exit(1)

print("\n" + "="*70)
print("✓ SETUP COMPLETE")
print("="*70)

print("\nFiles in place:")
print("  ✓ config/framing_queries_improved.yaml")
print("  ✓ core/nli_extraction_logits.py")
print("  ✓ test_logits_improved.py")

print("\nNext steps:")
print("  1. cd D:\\belief-transformer\\V3")
print("  2. python test_logits_improved.py")
print()
print("If test succeeds:")
print("  3. Replace core/nli_extraction.py with nli_extraction_logits.py")
print("  4. Replace config/framing_queries.yaml with framing_queries_improved.yaml")
print("  5. Delete outputs\\nli_cache_*.pt")
print("  6. Re-run experiments")

print("\n" + "="*70)
