#!/usr/bin/env python
"""
run_full_experiment.py - Complete Multi-Cycle Pierre Experiment
================================================================

Orchestrates:
1. Cycle 0: Baseline with base model
2. Train DPO on Cycle 0 data
3. Cycle 1: Run with trained model
4. Repeat for N cycles

This is the complete Pierre documentary experiment.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add project to path
sys.path.append(str(Path(__file__).parent))

from flywheel_COMPLETE import CompleteFlywheel
from reward_model import NLIVectorRewardSystem
from backends import BackendGGUF, BackendHFLocal

# Compatibility alias
DebertaHeadSystem = NLIVectorRewardSystem

# Conditional import for training (avoid peft/transformers dependency when not needed)
train_cycle_dpo = None
load_cycle_model = None


def load_seeds(dataset_file: Path, num_seeds: int = 50):
    """Load seed prompts from dataset."""
    import random
    
    with open(dataset_file) as f:
        data = json.load(f)
    
    # Extract contexts as seeds
    if isinstance(data, list):
        seeds = [item['Context'] for item in data if 'Context' in item]
    else:
        seeds = []
    
    # Sample if too many
    if len(seeds) > num_seeds:
        seeds = random.sample(seeds, num_seeds)
    
    return seeds


def analyze_cycle_drift(cycle_data: list, output_dir: Path):
    """
    Analyze drift across cycles.
    
    Track:
    - Engagement trajectory
    - Emotional state progression
    - Topic drift
    - Satisfaction vs wellbeing
    
    Generate visualizations for documentary.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Warning] matplotlib not installed, skipping visualization")
        return
    
    print(f"\n{'='*70}")
    print(f"CROSS-CYCLE DRIFT ANALYSIS")
    print(f"{'='*70}\n")
    
    # Extract metrics per cycle
    cycle_metrics = []
    
    for cycle in cycle_data:
        cycle_num = cycle['cycle']
        convos = cycle['conversations']
        
        # Aggregate scores across all conversations
        all_engage = []
        all_empathy = []
        all_satisfaction = []
        emotional_states = []
        
        for convo in convos:
            for turn in convo.get('turn_data', []):
                scores = turn.get('assistant_scores', {})
                all_engage.append(scores.get('engage_prob', 0))
                all_empathy.append(scores.get('empathy', 0))
                
                # User state tracking
                user_state = turn.get('user_state', {})
                if user_state.get('satisfaction_history'):
                    all_satisfaction.extend(user_state['satisfaction_history'])
                emotional_states.append(user_state.get('state', 'neutral'))
        
        cycle_metrics.append({
            'cycle': cycle_num,
            'engagement_mean': np.mean(all_engage) if all_engage else 0,
            'empathy_mean': np.mean(all_empathy) if all_empathy else 0,
            'satisfaction_mean': np.mean(all_satisfaction) if all_satisfaction else 0,
            'engagement_std': np.std(all_engage) if all_engage else 0,
            'emotional_states': emotional_states,
        })
    
    # Print table
    print("Cycle | Engagement | Empathy | Satisfaction")
    print("-" * 70)
    for m in cycle_metrics:
        print(f"  {m['cycle']}   |   {m['engagement_mean']:.3f}    |  {m['empathy_mean']:.3f}   |     {m['satisfaction_mean']:.3f}")
    
    # Plot drift
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    cycles = [m['cycle'] for m in cycle_metrics]
    
    # Engagement
    axes[0, 0].plot(cycles, [m['engagement_mean'] for m in cycle_metrics], 'o-', linewidth=2)
    axes[0, 0].set_title('Engagement Over Cycles', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Cycle')
    axes[0, 0].set_ylabel('Engagement Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Empathy
    axes[0, 1].plot(cycles, [m['empathy_mean'] for m in cycle_metrics], 'o-', color='red', linewidth=2)
    axes[0, 1].set_title('Empathy Over Cycles', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Cycle')
    axes[0, 1].set_ylabel('Empathy Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Satisfaction
    axes[1, 0].plot(cycles, [m['satisfaction_mean'] for m in cycle_metrics], 'o-', color='orange', linewidth=2)
    axes[1, 0].set_title('User Satisfaction Over Cycles', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Cycle')
    axes[1, 0].set_ylabel('Satisfaction Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Engagement vs Satisfaction (the key documentary insight)
    axes[1, 1].scatter(
        [m['satisfaction_mean'] for m in cycle_metrics],
        [m['engagement_mean'] for m in cycle_metrics],
        s=100,
        c=cycles,
        cmap='viridis'
    )
    axes[1, 1].set_title('Engagement vs Satisfaction', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Satisfaction Score')
    axes[1, 1].set_ylabel('Engagement Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_file = output_dir / "drift_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nDrift plot saved to: {plot_file}")
    
    # Save metrics
    metrics_file = output_dir / "drift_metrics.json"
    with open(metrics_file, 'w') as f:
        # Remove emotional_states for JSON serialization
        serializable_metrics = []
        for m in cycle_metrics:
            m_copy = m.copy()
            m_copy['emotional_states'] = list(set(m['emotional_states']))  # Unique states
            serializable_metrics.append(m_copy)
        json.dump(serializable_metrics, f, indent=2)
    print(f"Drift metrics saved to: {metrics_file}")
    
    print(f"\n{'='*70}\n")


def run_full_experiment(
    num_cycles: int = 4,
    num_seeds: int = 50,
    config_file: Path = None,
    dataset_file: Path = None,
    output_dir: Path = None,
    use_monte_carlo: bool = False,
    use_dpp: bool = True,
    train_between_cycles: bool = True
):
    """
    Run complete multi-cycle experiment.
    
    Args:
        num_cycles: Number of cycles to run (0, 1, 2, ...)
        num_seeds: Number of seed conversations per cycle
        config_file: Path to config YAML
        dataset_file: Path to dataset JSON
        output_dir: Where to save results
        use_monte_carlo: Enable Monte Carlo trajectory scoring
        use_dpp: Enable DPP diversity sampling
        train_between_cycles: Train DPO between cycles
    """
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/pierre_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PIERRE DOCUMENTARY EXPERIMENT")
    print(f"{'='*70}")
    print(f"Cycles: {num_cycles}")
    print(f"Seeds per cycle: {num_seeds}")
    print(f"Monte Carlo: {use_monte_carlo}")
    print(f"DPP sampling: {use_dpp}")
    print(f"Training: {train_between_cycles}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load training functions only if needed
    if train_between_cycles:
        try:
            from train_dpo_cycle import train_cycle_dpo, load_cycle_model
            globals()['train_cycle_dpo'] = train_cycle_dpo
            globals()['load_cycle_model'] = load_cycle_model
            print("[Training] DPO training imports successful\n")
        except ImportError as e:
            print(f"[Warning] Could not import training functions: {e}")
            print("[Warning] Disabling training between cycles\n")
            train_between_cycles = False
    
    # Load config
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Override experimental settings
    config.setdefault('experimental', {})
    config['experimental']['use_monte_carlo'] = use_monte_carlo
    config['experimental']['use_dpp_sampling'] = use_dpp
    
    # Load seeds
    print("Loading seeds...")
    seeds = load_seeds(dataset_file, num_seeds)
    print(f"  Loaded {len(seeds)} seeds\n")
    
    # Initialize reward model (shared across cycles)
    print("Initializing reward model...")
    reward_model = NLIVectorRewardSystem()
    print("  ✓ Reward model loaded\n")
    
    # Storage for all cycle data
    all_cycle_data = []
    
    # Run cycles
    for cycle_num in range(num_cycles):
        print(f"\n{'#'*70}")
        print(f"# CYCLE {cycle_num}")
        print(f"{'#'*70}\n")
        
        # Load appropriate model
        if cycle_num == 0:
            print("Loading base model (Cycle 0)...")
            assistant_model = BackendGGUF(
                model_path=config['assistant']['gguf_path'],
                n_ctx=config['assistant'].get('n_ctx', 2048),
                gpu_layers=config['assistant'].get('gpu_layers', 32)
            )
        else:
            if train_between_cycles:
                print(f"Loading trained model (Cycle {cycle_num})...")
                # Load model with LoRA adapter from previous cycle
                adapter_path = output_dir / f"cycle_{cycle_num}_adapter"
                assistant_model = BackendGGUF(
                    model_path=config['assistant']['gguf_path'],
                    n_ctx=config['assistant'].get('n_ctx', 2048),
                    gpu_layers=config['assistant'].get('gpu_layers', 32)
                )
                # Note: You'll need to implement adapter loading for GGUF
                # Or switch to HF model for training cycles
            else:
                print("Loading base model (no training)...")
                assistant_model = BackendGGUF(
                    model_path=config['assistant']['gguf_path'],
                    n_ctx=config['assistant'].get('n_ctx', 2048),
                    gpu_layers=config['assistant'].get('gpu_layers', 32)
                )
        
        # Load user simulator
        print("Loading user simulator...")
        usersim_model = BackendHFLocal(
            path=config['usersim']['model_path'],
            n_ctx=config['usersim'].get('n_ctx', 2048),
            dtype=config['usersim'].get('dtype', 'float16'),
            device_map=config['usersim'].get('device_map', 'auto'),
            trust_remote_code=config['usersim'].get('trust_remote_code', True)
        )
        
        # Initialize flywheel
        flywheel = CompleteFlywheel(
            assistant_backend=assistant_model,
            usersim_backend=usersim_model,
            reward_model=reward_model,
            config=config
        )
        
        # Run cycle
        cycle_output_dir = output_dir / f"cycle_{cycle_num}"
        cycle_data = flywheel.run_cycle(
            seeds=seeds,
            cycle_num=cycle_num,
            output_dir=cycle_output_dir
        )
        
        all_cycle_data.append(cycle_data)
        
        # Train for next cycle (if not last cycle)
        if train_between_cycles and cycle_num < num_cycles - 1:
            print(f"\n{'='*70}")
            print(f"TRAINING FOR CYCLE {cycle_num + 1}")
            print(f"{'='*70}\n")
            
            dpo_file = cycle_output_dir / f"cycle{cycle_num}_dpo_pairs.jsonl"
            adapter_output = output_dir / f"cycle_{cycle_num + 1}_adapter"
            
            try:
                train_cycle_dpo(
                    cycle_num=cycle_num,
                    dpo_pairs_file=dpo_file,
                    output_dir=adapter_output,
                    num_epochs=1,
                    beta=0.1
                )
            except Exception as e:
                print(f"Training failed: {e}")
                print("Continuing with base model for next cycle...")
    
    # Analyze drift across cycles
    analyze_cycle_drift(all_cycle_data, output_dir)
    
    # Save full experiment data
    experiment_data = {
        'config': config,
        'num_cycles': num_cycles,
        'num_seeds': num_seeds,
        'cycles': all_cycle_data
    }
    
    experiment_file = output_dir / "full_experiment.json"
    with open(experiment_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return all_cycle_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full Pierre experiment")
    parser.add_argument("--cycles", type=int, default=4, help="Number of cycles")
    parser.add_argument("--seeds", type=int, default=50, help="Seeds per cycle")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config YAML")
    parser.add_argument("--dataset", type=Path, default=Path("D:/AI_Project/Llamav2_synthdata/models/combined_dataset.json"), help="Dataset JSON")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--monte-carlo", action="store_true", help="Enable Monte Carlo")
    parser.add_argument("--no-dpp", action="store_true", help="Disable DPP")
    parser.add_argument("--no-training", action="store_true", help="Skip DPO training")
    
    args = parser.parse_args()
    
    run_full_experiment(
        num_cycles=args.cycles,
        num_seeds=args.seeds,
        config_file=args.config,
        dataset_file=args.dataset,
        output_dir=args.output,
        use_monte_carlo=args.monte_carlo,
        use_dpp=not args.no_dpp,
        train_between_cycles=not args.no_training
    )
