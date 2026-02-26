"""
Auto-fix complete_pipeline_diverse_FIXED.py for logits-based extraction
"""

import sys

print("="*70)
print("PATCHING complete_pipeline_diverse_FIXED.py")
print("="*70)

try:
    with open('complete_pipeline_diverse_FIXED.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
except FileNotFoundError:
    print("\n✗ ERROR: complete_pipeline_diverse_FIXED.py not found!")
    print("Are you in D:\\belief-transformer\\V3?")
    sys.exit(1)

fixed_lines = []
changes_made = 0

for i, line in enumerate(lines, 1):
    # Fix 1: Replace embedding_dim assignment
    if 'embedding_dim = nli_extractor.embedding_dim' in line:
        indent = len(line) - len(line.lstrip())
        fixed_lines.append(' ' * indent + 'logits_dim = 3  # [contradiction, neutral, entailment]\n')
        print(f"  Line {i}: Changed embedding_dim assignment")
        changes_made += 1
    
    # Fix 2: Replace feature_dim calculation
    elif 'feature_dim = n_framings * embedding_dim' in line:
        indent = len(line) - len(line.lstrip())
        fixed_lines.append(' ' * indent + 'feature_dim = n_framings * logits_dim  # e.g., 8 × 3 = 24\n')
        print(f"  Line {i}: Changed feature_dim calculation")
        changes_made += 1
    
    # Fix 3: Replace print statement
    elif 'Embedding dim:' in line:
        fixed_lines.append(line.replace('Embedding dim:', 'Logits dim:').replace('embedding_dim', 'logits_dim'))
        print(f"  Line {i}: Changed print statement")
        changes_made += 1
    
    # Keep everything else
    else:
        fixed_lines.append(line)

if changes_made == 0:
    print("\n⚠ No changes made - file might already be fixed!")
else:
    # Write back
    with open('complete_pipeline_diverse_FIXED.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"\n✓ Made {changes_made} changes")
    print("✓ File patched successfully!")

print("\n" + "="*70)
print("Now run:")
print("  python run_diverse_experiments.py --corpus real --mode diverse")
print("="*70)
