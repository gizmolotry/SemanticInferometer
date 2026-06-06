from pathlib import Path
import re

target = Path("../run_experiments.py")
content = target.read_text(encoding="utf-8")

# 1. Add missing flags to argparse
if '--no-crn' not in content:
    old_arg = r'''    parser.add_argument(
        '--locked-weights-path',
        type=str,
        default=None,
        help='Path to locked Dirichlet weights file for CRN reproducibility'
    )'''
    
    new_arg = old_arg + r'''

    parser.add_argument(
        "--no-crn",
        action="store_true",
        help="Disable CRN for Dirichlet fusion"
    )

    parser.add_argument(
        "--alpha-collapse",
        action="store_true",
        help="Force alpha=1e6 for collapse ablation"
    )'''
    content = content.replace(old_arg, new_arg)

# 2. Fix DirichletFusionConfig instantiation to use crn_enabled
# First find where the config is created
config_block_old = r'''    config = DirichletFusionConfig(
        n_bots=8,
        hidden_dim=actual_hidden,
        rks_dim=mode_config.get('dirichlet_rks_dim', 2048),
        n_observers=mode_config.get('dirichlet_n_observers', 50),
        alpha=alpha_val,
        kernel_type=mode_config.get('kernel_type', 'rbf'),
        basis_seed=mode_config.get('dirichlet_basis_seed', 42),
    )'''

# We need to add crn_enabled=crn_enabled to it
if 'crn_enabled=crn_enabled' not in content:
    config_block_new = r'''    config = DirichletFusionConfig(
        n_bots=8,
        hidden_dim=actual_hidden,
        rks_dim=mode_config.get('dirichlet_rks_dim', 2048),
        n_observers=mode_config.get('dirichlet_n_observers', 50),
        alpha=alpha_val,
        kernel_type=mode_config.get('kernel_type', 'rbf'),
        basis_seed=mode_config.get('dirichlet_basis_seed', 42),
        crn_enabled=crn_enabled,
    )'''
    content = content.replace(config_block_old, config_block_new)

target.write_text(content, encoding="utf-8")
print("Fixed run_experiments.py flags and config usage")
