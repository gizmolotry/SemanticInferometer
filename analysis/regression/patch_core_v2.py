import os
import re

# Fix complete_pipeline.py to pass kernel_type/params correctly
core_path = os.path.join("..", "core", "complete_pipeline.py")
with open(core_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update initialize_full_pipeline to use kernel_type for RKSFeatureMap (already patched, but let's be sure)
content = re.sub(r'rks_map = RKSFeatureMap\(.*?kernel_type="rbf",', 
                 lambda m: m.group(0).replace('kernel_type="rbf"', 'kernel_type=kernel_type'), 
                 content, flags=re.DOTALL)

# 2. Update run_multi_observer_experiment_simple to pass kernel_type and params
# Search for initialize_full_pipeline call inside run_multi_observer_experiment_simple
old_call = """        components = initialize_full_pipeline(
            random_seed=seed,
            device=device,
            use_contrastive=use_contrastive,
            use_cls_tokens=use_cls_tokens,
            pca_remove=use_pca_removal,
            global_pca_component=shared_pca,
            use_gru=use_gru,
            use_rks=use_multi_framing_rks,"""

new_call = """        components = initialize_full_pipeline(
            random_seed=seed,
            device=device,
            use_contrastive=use_contrastive,
            use_cls_tokens=use_cls_tokens,
            pca_remove=use_pca_removal,
            global_pca_component=shared_pca,
            use_gru=use_gru,
            use_rks=use_multi_framing_rks,
            kernel_type=kernel_type,
            kernel_bandwidth=rks_sigma,"""

if old_call in content:
    content = content.replace(old_call, new_call)
    print("[OK] Fixed initialize_full_pipeline call in run_multi_observer_experiment_simple")
else:
    print("[WARN] Could not find initialize_full_pipeline call for replacement")

with open(core_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] Core pipeline patched.")
