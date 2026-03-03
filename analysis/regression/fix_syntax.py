import os

# Fix run_experiments.py syntax error
runner_path = os.path.join("..", "run_experiments.py")
with open(runner_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # If the line is just a comma and the previous line ended with default='real',
    if line.strip() == "," and i > 0 and "default='real'," in lines[i-1]:
        print(f"Fixing extra comma at line {i+1}")
        continue # Skip this line
    new_lines.append(line)

with open(runner_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("[OK] Syntax error fixed.")
