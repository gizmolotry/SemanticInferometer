import re
from pathlib import Path

def patch_runner_ablations(file_path: Path):
    content = file_path.read_text(encoding='utf-8')
    
    # Let's do a simpler patch: append verification summary to the end of the script instructions.
    if 'View verification report' not in content:
        old_str = 'print(f"5. View alpha sweep: cat alpha_sweep_summary.json")'
        new_str = 'print(f"5. View alpha sweep: cat alpha_sweep_summary.json")\n    if getattr(args, \'verify\', False):\n        print(f"11. View verification report: cat {exp_dir / \'verification_report.json\'}")'
        content = content.replace(old_str, new_str)
        print("Updated final instructions.")

    file_path.write_text(content, encoding='utf-8')

if __name__ == "__main__":
    patch_runner_ablations(Path("../run_full_experiment_suite.py"))
