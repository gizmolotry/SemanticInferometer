"""
Fix run_multi_observer_experiment signature to add RKS parameters
"""

with open('core/complete_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the function signature
old_sig = """def run_multi_observer_experiment(
    articles: List[Dict],
    seeds: List[int] = [42, 43, 44, 45, 46],
    use_gru: bool = True,
    use_framing_rope: bool = True,
    device: str = 'cuda',
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli'
) -> Dict[int, Dict]:"""

new_sig = """def run_multi_observer_experiment(
    articles: List[Dict],
    seeds: List[int] = [42, 43, 44, 45, 46],
    use_gru: bool = True,
    use_framing_rope: bool = True,
    device: str = 'cuda',
    nli_model_name: str = 'microsoft/deberta-v2-xlarge-mnli',
    use_rks: bool = True,
    rks_dim: int = 512,
    rks_sigma: float = 1.0
) -> Dict[int, Dict]:"""

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    print("✓ Fixed function signature")
else:
    print("✗ Could not find signature to replace")
    print("Searching for partial match...")
    if "def run_multi_observer_experiment(" in content:
        print("  Function exists but signature differs")
    else:
        print("  Function not found!")

with open('core/complete_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✓ File updated")
