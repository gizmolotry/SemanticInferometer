import os
import re

# Fix hardcoded 'rbf' in complete_pipeline.py
core_path = os.path.join("..", "core", "complete_pipeline.py")
with open(core_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace hardcoded 'rbf' in RKSFeatureMap initialization
old_rks = """        rks_map = RKSFeatureMap(
            input_dim=output_dim if gru_model is not None else embedding_dim,
            output_dim=final_dim,
            kernel_type="rbf","""

new_rks = """        rks_map = RKSFeatureMap(
            input_dim=output_dim if gru_model is not None else embedding_dim,
            output_dim=final_dim,
            kernel_type=kernel_type,"""

if old_rks in content:
    content = content.replace(old_rks, new_rks)
    print("[OK] Fixed RKSFeatureMap kernel_type in complete_pipeline.py")
else:
    # Fallback regex
    content = re.sub(r'rks_map = RKSFeatureMap\(.*?kernel_type="rbf",', 
                     lambda m: m.group(0).replace('kernel_type="rbf"', 'kernel_type=kernel_type'), 
                     content, flags=re.DOTALL)
    print("[INFO] Applied regex fix for RKSFeatureMap")

with open(core_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] Core pipeline patched.")
