import re
from pathlib import Path

def patch_run_experiments(file_path: Path):
    content = file_path.read_text(encoding='utf-8')
    
    # 1. Add flags to argparse
    if '--no-crn' not in content:
        arg_pattern = r'(parser\.add_argument\(\s*"--locked-weights-path",.*?\))'
        arg_replacement = r'\1\n\n    parser.add_argument(\n        "--no-crn",\n        action="store_true",\n        help="Disable CRN for Dirichlet fusion"\n    )\n\n    parser.add_argument(\n        "--alpha-collapse",\n        action="store_true",\n        help="Force alpha=1e6 for collapse ablation"\n    )'
        content = re.sub(arg_pattern, arg_replacement, content, flags=re.DOTALL)
        print("Added --no-crn and --alpha-collapse flags.")

    # 2. Modify run_dirichlet_fusion_experiment to respect flags
    if '[ABLATION]' not in content:
        # Use simple string replacement for reliability
        old_lines = "        alpha=mode_config.get('dirichlet_alpha', 1.0),"
        new_lines = "        alpha=alpha_val,"
        
        content = content.replace(old_lines, new_lines)
        
        inject_marker = "    # Now configure fusion with actual hidden dim"
        inject_code = '''
    # Handle ablations
    alpha_val = mode_config.get('dirichlet_alpha', 1.0)
    if getattr(args, 'alpha_collapse', False):
        print("[ABLATION] FORCING ALPHA COLLAPSE (alpha=1e6)")
        alpha_val = 1e6
        
    crn_enabled = True
    if getattr(args, 'no_crn', False):
        print("[ABLATION] DISABLING CRN")
        crn_enabled = False
'''
        content = content.replace(inject_marker, inject_code + "\n" + inject_marker)
        
        # Also need to update the crn_enabled parameter in config
        content = content.replace("crn_enabled=True,", "crn_enabled=crn_enabled,")
        print("Injected ablation logic into run_dirichlet_fusion_experiment.")

    file_path.write_text(content, encoding='utf-8')

if __name__ == "__main__":
    patch_run_experiments(Path("../run_experiments.py"))
