import re
from pathlib import Path

def patch_runner(file_path: Path):
    content = file_path.read_text(encoding='utf-8')
    
    # 1. Add --verify to argparse
    if '--verify' not in content:
        arg_pattern = r'(parser\.add_argument\(\s*"--dirichlet-basis-seed",.*?\))'
        arg_replacement = r'\1\n\n    parser.add_argument(\n        "--verify",\n        action="store_true",\n        help="Run verification harness after experiment suite"\n    )'
        content = re.sub(arg_pattern, arg_replacement, content, flags=re.DOTALL)
        print("Added --verify flag.")

    # 2. Add verification hook before summary
    if 'RUNNING VERIFICATION HARNESS' not in content:
        # Match the start of the summary section
        hook_pattern = r'(# Summary\s+print\(f"\\n\{\'=\'\*80\}"\)\s+print\("SUITE COMPLETE"\))'
        hook_code = r'''
    # =========================================================================
    # VERIFICATION HARNESS
    # =========================================================================
    if getattr(args, 'verify', False):
        print("\n" + "="*80)
        print("RUNNING VERIFICATION HARNESS")
        print("="*80)
        try:
            import sys
            import os
            # Ensure analysis is in path
            analysis_path = Path("analysis").resolve()
            if str(analysis_path) not in sys.path:
                sys.path.append(str(analysis_path))
            
            from verification.verify_run import verify_layer, write_report
            
            reports = []
            # Verify each kernel/channel layer
            for channel in args.channels:
                if channel == "gradient": continue
                for kernel in args.kernels:
                    layer_dir = exp_dir / kernel / channel
                    if layer_dir.exists():
                        print(f"Verifying Layer: {kernel}/{channel}")
                        report = verify_layer(layer_dir, exp_dir)
                        reports.append(report)
            
            if reports:
                write_report(reports, exp_dir)
                print(f"[VERIFY] Verification report saved to: {exp_dir / 'verification_report.json'}")
        except Exception as e:
            print(f"[VERIFY] Error during verification: {e}")
            import traceback
            traceback.print_exc()

    '''
        content = re.sub(hook_pattern, hook_code + r'\1', content, flags=re.DOTALL)
        print("Added verification hook.")

    file_path.write_text(content, encoding='utf-8')

if __name__ == "__main__":
    patch_runner(Path("../run_full_experiment_suite.py"))
