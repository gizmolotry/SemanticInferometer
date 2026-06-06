import os
import re
from pathlib import Path

# Fix run_experiments.py kernel override and pipeline_config
runner_path = os.path.join("..", "run_experiments.py")
with open(runner_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Ensure kernel_type override is applied to pipeline_config correctly
# Find the line: if getattr(args, "kernel_type", None):
# and ensure it updates both mode_config and pipeline_config if defined.

# More robustly: let's find the main() part where it defines the pipeline_config 
# for standard experiments and ensure it uses the overridden mode_config.

# Actually, the simplest fix is to find the part where it sets the override
# and add pipeline_config['kernel_type'] = args.kernel_type there.

old_override = """    if getattr(args, "kernel_type", None):
        mode_config["kernel_type"] = args.kernel_type
        mode_config.pop("kernel_types", None)
        mode_config.pop("kernel_types_per_framing", None)"""

new_override = """    if getattr(args, "kernel_type", None):
        mode_config["kernel_type"] = args.kernel_type
        mode_config.pop("kernel_types", None)
        mode_config.pop("kernel_types_per_framing", None)
        if 'pipeline_config' in locals():
            pipeline_config['kernel_type'] = args.kernel_type
        print(f"Kernel override: {args.kernel_type}")"""

if old_override in content:
    content = content.replace(old_override, new_override)
elif 'if getattr(args, "kernel_type", None):' in content:
    # Fallback if whitespace differs
    content = re.sub(r'if getattr\(args, "kernel_type", None\):\s+mode_config\["kernel_type"\] = args\.kernel_type\s+mode_config\.pop\("kernel_types", None\)\s+mode_config\.pop\("kernel_types_per_framing", None\)', 
                     new_override, content)

# 2. Fix the batch size logic to be absolutely sure it's 512
content = re.sub(r"batch_size\s*=\s*\d+", "batch_size = 512", content)

with open(runner_path, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] Kernel override and batch size patched.")
