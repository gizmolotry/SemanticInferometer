"""
DEPLOY LOGITS-BASED SYSTEM

Final deployment script that:
1. Backs up current files
2. Deploys logits-based complete_pipeline
3. Deletes old cache
4. Verifies setup
"""

import os
import shutil
from pathlib import Path

print("="*70)
print("DEPLOYING LOGITS-BASED BELIEF TRANSFORMER")
print("="*70)

# Verify we're in the right place
if not os.path.exists('core') or not os.path.exists('config'):
    print("\n✗ ERROR: Run this from D:\\belief-transformer\\V3")
    exit(1)

print("\nCurrent directory:", os.getcwd())

# Check prerequisites
print("\n[1/5] Checking prerequisites...")
required_files = {
    'core/nli_extraction_logits.py': 'Logits-based NLI extractor',
    'config/framing_queries.yaml': 'Framing queries',
    'complete_pipeline_LOGITS.py': 'Updated complete pipeline'
}

missing = []
for fpath, desc in required_files.items():
    if not os.path.exists(fpath):
        missing.append(f"{fpath} ({desc})")
        
if missing:
    print("\n✗ ERROR: Missing required files:")
    for m in missing:
        print(f"  - {m}")
    print("\nMake sure:")
    print("  1. nli_extraction_logits.py is in core/")
    print("  2. framing_queries.yaml is in config/")
    print("  3. complete_pipeline_LOGITS.py is in V3/")
    exit(1)

print("  ✓ All required files present")

# Backup current files
print("\n[2/5] Creating backups...")
backup_dir = Path('core_backup_deployment')
backup_dir.mkdir(exist_ok=True)

files_to_backup = [
    'core/complete_pipeline.py',
    'core/nli_extraction.py',
    'config/framing_queries.yaml'
]

for fpath in files_to_backup:
    if os.path.exists(fpath):
        backup_path = backup_dir / Path(fpath).name
        shutil.copy(fpath, backup_path)
        print(f"  ✓ Backed up {fpath}")

# Deploy new complete_pipeline
print("\n[3/5] Deploying logits-based complete_pipeline...")
shutil.copy('complete_pipeline_LOGITS.py', 'core/complete_pipeline.py')
print("  ✓ Deployed core/complete_pipeline.py")

# Verify nli_extraction_logits is in place
print("\n[4/5] Verifying NLI extractor...")
if os.path.exists('core/nli_extraction_logits.py'):
    print("  ✓ core/nli_extraction_logits.py present")
else:
    print("  ✗ ERROR: core/nli_extraction_logits.py missing!")
    exit(1)

# Delete old cache
print("\n[5/5] Cleaning old cache...")
cache_files = list(Path('outputs').glob('nli_cache_*.pt'))
if cache_files:
    for cache in cache_files:
        cache.unlink()
        print(f"  ✓ Deleted {cache}")
else:
    print("  ✓ No old cache to delete")

# Delete old observer files
print("\nCleaning old observer files...")
observer_patterns = [
    'observer_*.pt',
    'diverse_observer_*.pt',
    'shuffle_observer_*.pt',
    'control_attention_results.pt'
]

deleted_count = 0
for pattern in observer_patterns:
    for fpath in Path('outputs').glob(pattern):
        fpath.unlink()
        deleted_count += 1

if deleted_count > 0:
    print(f"  ✓ Deleted {deleted_count} old observer files")
else:
    print("  ✓ No old observer files to delete")

# Summary
print("\n" + "="*70)
print("✓ DEPLOYMENT COMPLETE")
print("="*70)

print("\nDeployed files:")
print("  ✓ core/complete_pipeline.py (LOGITS-BASED)")
print("  ✓ core/nli_extraction_logits.py")
print("  ✓ config/framing_queries.yaml")

print("\nBackups saved to:")
print(f"  {backup_dir}/")

print("\nCleaned:")
print("  ✓ All NLI cache deleted")
print(f"  ✓ {deleted_count} old observer files deleted")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("\n1. Test the deployment:")
print("   cd D:\\belief-transformer\\V3")
print("   python test_logits_improved.py")
print()
print("2. If test succeeds, run full experiment:")
print("   python run_diverse_experiments.py --corpus real --mode diverse")
print()
print("3. Expected results:")
print("   - Feature dimension: 24 (not 12288)")
print("   - Opposite articles <50% similar")
print("   - Real variance 2-5x higher than controls")

print("\n" + "="*70)
print("ROLLBACK (if needed):")
print("="*70)
print("\nTo restore old system:")
print(f"  copy {backup_dir}\\complete_pipeline.py core\\")
print(f"  copy {backup_dir}\\nli_extraction.py core\\")
print(f"  copy {backup_dir}\\framing_queries.yaml config\\")

print("\n" + "="*70)
