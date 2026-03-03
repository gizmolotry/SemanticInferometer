import subprocess
import os
from datetime import datetime
from pathlib import Path

# Config
EXP_DIR = f"outputs/experiments/runs/matern_verified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(EXP_DIR, exist_ok=True)

REAL_CORPUS = "..\\sythgen\\high_quality_articles.jsonl"
SHUFFLED_CORPUS = "outputs\\synthetic_controls\\synthetic_shuffled.jsonl"
RANDOM_CORPUS = "outputs\\synthetic_controls\\synthetic_random.jsonl"
CONSTANT_CORPUS = "outputs\\synthetic_controls\\synthetic_constant.jsonl"

SEEDS = ["1", "10", "100", "42"]
LIMIT = "50"
KERNEL = "matern"

CONDITIONS = [
    ("real", REAL_CORPUS),
    ("control_shuffled", SHUFFLED_CORPUS),
    ("control_random", RANDOM_CORPUS),
    ("control_constant", CONSTANT_CORPUS),
]

print(f"[LAUNCHER] Output directory: {EXP_DIR}")

# Process each condition
for cond_name, cond_corpus in CONDITIONS:
    out_path = Path(EXP_DIR) / "matern" / "cls" / cond_name
    os.makedirs(out_path, exist_ok=True)
    
    # We explicitly do NOT use a shared NLI cache here to ensure 
    # Real data is actually compared against Control noise.
    cmd = [
        "python", "..\\run_experiments.py",
        "--corpus", cond_corpus,
        "--kernel-type", KERNEL,
        "--seeds", *SEEDS,
        "--mode", "cls_logits",
        "--limit", LIMIT,
        "--output-root", str(out_path)
    ]
    
    print(f"\n[LAUNCHER] --- STARTING CONDITION: {cond_name} ---")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command and stream output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] Condition {cond_name} failed with code {process.returncode}")
    else:
        print(f"\n[OK] Condition {cond_name} complete.")

# Final verification
verify_cmd = ["python", "verification\\verify_run.py", EXP_DIR]
print(f"\n[LAUNCHER] --- RUNNING FINAL VERIFICATION ---")
print(f"Command: {' '.join(verify_cmd)}")
subprocess.run(verify_cmd)

print("\n[LAUNCHER] FULL SUITE COMPLETE.")
