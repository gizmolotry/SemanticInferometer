"""
Apply Critical Fixes to Belief Transformer
1. Change model to MNLI version
2. Add proper metadata tracking
"""

import os
import re

def fix_model_name(filepath):
    """Change deberta-v3-large to deberta-v2-xlarge-mnli"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the model name
    original = content
    content = content.replace(
        'microsoft/deberta-v3-large',
        'microsoft/deberta-v2-xlarge-mnli'
    )
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed model name in: {filepath}")
        return True
    else:
        print(f"  No changes needed in: {filepath}")
        return False

def fix_metadata(filepath):
    """Add title to metadata tracking"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the metadata building section
    original = content
    
    # Pattern to find metadata append block
    pattern = r"(metadata\.append\(\{\s*'index': i,\s*)'source':"
    replacement = r"\1'title': article.get('title', '')[:200],\n            'source':"
    
    content = re.sub(pattern, replacement, content)
    
    # Also fix text_preview to use 'content'
    content = content.replace(
        "'text_preview': article['text'][:200]",
        "'text_preview': article.get('content', article.get('text', ''))[:200]"
    )
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed metadata in: {filepath}")
        return True
    else:
        print(f"  No metadata changes needed in: {filepath}")
        return False

print("="*70)
print("APPLYING CRITICAL FIXES")
print("="*70)

# Fix 1: Model names
print("\n### FIX 1: Changing to MNLI model ###\n")
files_to_fix_model = [
    'core/nli_extraction.py',
    'core/complete_pipeline.py',
    'core/pipeline.py'
]

model_fixed = 0
for fpath in files_to_fix_model:
    if os.path.exists(fpath):
        if fix_model_name(fpath):
            model_fixed += 1
    else:
        print(f"✗ File not found: {fpath}")

# Fix 2: Metadata
print("\n### FIX 2: Adding title to metadata ###\n")
files_to_fix_metadata = [
    'core/complete_pipeline.py',
    'core/pipeline.py'
]

metadata_fixed = 0
for fpath in files_to_fix_metadata:
    if os.path.exists(fpath):
        if fix_metadata(fpath):
            metadata_fixed += 1
    else:
        print(f"✗ File not found: {fpath}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Model name fixes: {model_fixed}/3")
print(f"Metadata fixes: {metadata_fixed}/2")
print()

if model_fixed > 0 or metadata_fixed > 0:
    print("✓ FIXES APPLIED!")
    print()
    print("CRITICAL: Delete NLI cache:")
    print("  del outputs\\nli_cache_*.pt")
    print()
    print("Then test:")
    print("  python test_hypothesis_sensitivity.py")
else:
    print("⚠ No changes made (files may already be fixed)")

print("="*70)
