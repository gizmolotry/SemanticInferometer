import re
from pathlib import Path

def patch_runner_final(file_path: Path):
    content = file_path.read_text(encoding='utf-8')
    
    # 1. Update run_single_corpus signature to accept extra_flags
    # We use a very specific string to avoid matching other parts
    old_sig = 'nli_cache_path: str = None\n)'
    if old_sig in content:
        content = content.replace(old_sig, 'nli_cache_path: str = None,\n    extra_flags: List[str] = None\n)')
        print("Updated run_single_corpus signature.")

    # Add logic to append extra_flags to cmd
    if 'if extra_flags:' not in content:
        # Match the end of the cmd list construction
        cmd_end = 'if nli_cache_path:'
        content = content.replace(cmd_end, 'if extra_flags:\n        cmd.extend(extra_flags)\n\n    if nli_cache_path:')
        print("Updated run_single_corpus command construction.")

    # 2. Update verification block to run ablations
    if 'RUNNING AUTOMATED ABLATIONS' not in content:
        ablation_logic = r'''
            # -----------------------------
            # RUNNING AUTOMATED ABLATIONS
            # -----------------------------
            # Pick first representative layer for ablation
            rep_channel = [c for c in args.channels if c != 'gradient'][0]
            rep_kernel = args.kernels[0]
            
            print(f"\n[VERIFY] Running Ablation A1: CRN OFF ({rep_kernel}/{rep_channel})")
            for corpus in args.corpora:
                a1_out = exp_dir / "ablation" / "crn_off" / rep_kernel / rep_channel / corpus
                a1_out.mkdir(parents=True, exist_ok=True)
                run_single_corpus(corpus, args.seeds, args.limit, a1_out, 
                                 mode=get_mode_for_channel(rep_channel, args.mode), 
                                 track_variance=False, kernel_type=rep_kernel,
                                 extra_flags=["--no-crn"])
            
            if getattr(args, 'alpha_sweep', False):
                print(f"\n[VERIFY] Running Ablation A2: ALPHA COLLAPSE ({rep_kernel}/{rep_channel})")
                for corpus in args.corpora:
                    a2_out = exp_dir / "ablation" / "alpha_collapse" / rep_kernel / rep_channel / corpus
                    a2_out.mkdir(parents=True, exist_ok=True)
                    run_single_corpus(corpus, args.seeds, args.limit, a2_out, 
                                     mode=get_mode_for_channel(rep_channel, args.mode), 
                                     track_variance=False, kernel_type=rep_kernel,
                                     extra_flags=["--alpha-collapse"])
'''
        # Inject after reports = []
        # We search for the specific indentation inside the verify block
        content = content.replace('reports = []', 'reports = []' + ablation_logic)
        print("Injected automated ablation logic.")

    file_path.write_text(content, encoding='utf-8')

if __name__ == "__main__":
    patch_runner_final(Path("../run_full_experiment_suite.py"))
