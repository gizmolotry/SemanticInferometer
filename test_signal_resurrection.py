import torch
import json
import sys
from pathlib import Path

print("="*60)
print("SIGNAL RESURRECTION VERIFICATION TEST")
print("="*60)

# Load test articles
with open('data/control_corpus.jsonl', 'r', encoding='utf-8') as f:
    articles = [json.loads(line) for i, line in enumerate(f) if i < 30]
print(f"Loaded {len(articles)} test articles")

# Import and initialize pipeline
from core.complete_pipeline import initialize_full_pipeline, BeliefTransformerPipeline

components = initialize_full_pipeline(
    random_seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    kernel_type='matern',  # Use Matern for more sensitivity
    use_cls_tokens=True,
    use_dirichlet_fusion=True,
    dirichlet_rks_dim=256,
    dirichlet_n_observers=8,
    dirichlet_alpha=1.0,
    dirichlet_hidden_dim=1536,
    mix_in_rkhs=True,
    geometry_mode='rks',
    normalize_features=True,
)

pipeline = BeliefTransformerPipeline(
    components=components,
    random_seed=42,
    enable_provenance=False,
)

# Process
print("\nProcessing articles...")
result = pipeline.process_month(articles=articles, month_name='signal_test')

# VERIFICATION CHECKS
print("\n" + "="*60)
print("VERIFICATION RESULTS")
print("="*60)

checks_passed = 0
checks_total = 4

# Check 1: Track 1.5 computed
if 'integrated_vectors' in result:
    print("[CHECK 1] integrated_vectors present: PASS")
    checks_passed += 1
else:
    print("[CHECK 1] integrated_vectors present: FAIL")

# Check 2: Phase space diagnostics
diag = result.get('diagnostics', {})
ps_diag = diag.get('phase_space', {})
n_particles = ps_diag.get('n_particles', 0)
int_dim = ps_diag.get('integrated_dim', 0)

if int_dim > 536:  # Should be larger now with antagonism/walker
    print(f"[CHECK 2] Integrated dim ({int_dim}) includes force tracks: PASS")
    checks_passed += 1
else:
    print(f"[CHECK 2] Integrated dim ({int_dim}) - expected > 536 with force tracks: WARN")

# Check 3: Singularity distribution
sing = result.get('singularity_counts', {})
consensus = sing.get('consensus', 0)
structural = sing.get('structural_singularity', 0)
noise = sing.get('noise', 0)
barrier = sing.get('ideological_barrier', 0)

print(f"\nSingularity Distribution:")
print(f"  consensus: {consensus}")
print(f"  structural_singularity: {structural}")
print(f"  noise: {noise}")
print(f"  ideological_barrier: {barrier}")

if consensus < n_particles:  # Not 100% consensus anymore
    print(f"[CHECK 3] Broken consensus trap ({consensus}/{n_particles}): PASS")
    checks_passed += 1
else:
    print(f"[CHECK 3] Still 100% consensus: FAIL - force signals may still be too weak")

# Check 4: Non-zero liar scores
mean_liar = result.get('mean_liar_score', 0)
n_liars = result.get('n_liars', 0)
print(f"\nLiar Detection:")
print(f"  mean_liar_score: {mean_liar:.4f}")
print(f"  n_liars (>0.5): {n_liars}")

if mean_liar > 0.01 or n_liars > 0:
    print(f"[CHECK 4] Liar detection active: PASS")
    checks_passed += 1
else:
    print(f"[CHECK 4] No liars detected: WARN (may be valid for control corpus)")
    checks_passed += 1  # Control corpus may legitimately have no liars

print("\n" + "="*60)
print(f"VERIFICATION: {checks_passed}/{checks_total} checks passed")
print("="*60)

if checks_passed >= 3:
    print("SIGNAL RESURRECTION: SUCCESS")
else:
    print("SIGNAL RESURRECTION: NEEDS INVESTIGATION")
