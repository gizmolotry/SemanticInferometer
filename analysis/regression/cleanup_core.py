import os

# Cleanup complete_pipeline.py from repeated arguments
core_path = os.path.join("..", "core", "complete_pipeline.py")
with open(core_path, "r", encoding="utf-8") as f:
    content = f.read()

# Repeated block to remove
bad_block = """            kernel_type=kernel_type,
            kernel_bandwidth=rks_sigma,
            kernel_type=kernel_type,
            kernel_bandwidth=rks_sigma,"""

good_block = """            kernel_type=kernel_type,
            kernel_bandwidth=rks_sigma,"""

if bad_block in content:
    content = content.replace(bad_block, good_block)
    print("[OK] Removed duplicate block.")
else:
    # Try with different indentations
    import re
    content = re.sub(r'(kernel_type=kernel_type,\s+kernel_bandwidth=rks_sigma,\s+)(kernel_type=kernel_type,\s+kernel_bandwidth=rks_sigma,)', 
                     r'\1', content, flags=re.MULTILINE)
    print("[INFO] Applied regex cleanup.")

with open(core_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] Duplicates removed from core pipeline.")
