# Track 4 Seven-Branch Seed-420 All-Kernel Sweep - 2026-05-21

Run: `seed=420`, kernels `rbf/matern/imq`, bases `track2/integrated`, corpora `real/control_random/control_shuffled`, all seven observer-state engineering branches.

Key finding: `null_calibrated_hysteresis` and `richer_walker_state` both pass across all three kernels for seed 420. Baseline and gate-only still fail at `matern|seed420|integrated`. `track2_default` remains safe but avoids integrated traversal rather than repairing it.

```json
[
  {
    "action_branch": "track2_default",
    "action_ratio_by_basis": {
      "track2": 5.271682733736
    },
    "action_ratio_by_kernel": {
      "imq": 4.541147017104,
      "matern": 5.906172469092,
      "rbf": 5.281210041443
    },
    "branch_type": "basis_policy",
    "failing_kernel_seed_basis_cells": {},
    "failure_reasons": [],
    "point_estimate": 5.271682733736,
    "safe_for_thesis_claim": true
  },
  {
    "action_branch": "null_calibrated_hysteresis",
    "action_ratio_by_basis": {
      "integrated": 2.441542198899,
      "track2": 5.325829627526
    },
    "action_ratio_by_kernel": {
      "imq": 3.064954414797,
      "matern": 4.176245634331,
      "rbf": 3.143255578074
    },
    "branch_type": "calibration",
    "failing_kernel_seed_basis_cells": {},
    "failure_reasons": [],
    "point_estimate": 3.479414150136,
    "safe_for_thesis_claim": true
  },
  {
    "action_branch": "richer_walker_state",
    "action_ratio_by_basis": {
      "integrated": 2.446899891508,
      "track2": 5.296260165652
    },
    "action_ratio_by_kernel": {
      "imq": 3.06492490044,
      "matern": 4.148139052797,
      "rbf": 3.160002558026
    },
    "branch_type": "state_model",
    "failing_kernel_seed_basis_cells": {},
    "failure_reasons": [],
    "point_estimate": 3.476682325715,
    "safe_for_thesis_claim": true
  },
  {
    "action_branch": "virtual_transition_states",
    "action_ratio_by_basis": {
      "integrated": 2.828797613169,
      "track2": 6.088486920701
    },
    "action_ratio_by_kernel": {
      "imq": 3.66869928872,
      "matern": 4.839175974608,
      "rbf": 3.47893882759
    },
    "branch_type": "graph_densification_proxy",
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 4.010530479077,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "path_ensemble_tpt",
    "action_ratio_by_basis": {
      "integrated": 2.257862885673,
      "track2": 5.863202181431
    },
    "action_ratio_by_kernel": {
      "imq": 3.714473714897,
      "matern": 3.046097711222,
      "rbf": 3.946885751914
    },
    "branch_type": "ensemble_probe",
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 3.547056265456,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "separated_action_channels",
    "action_ratio_by_basis": {
      "integrated": 2.44190243882,
      "track2": 5.327996914711
    },
    "action_ratio_by_kernel": {
      "imq": 3.093329295139,
      "matern": 4.116685653398,
      "rbf": 3.173598066492
    },
    "branch_type": "channel_ablation",
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 3.480369497917,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "baseline_raw_action",
    "action_ratio_by_basis": {
      "integrated": 2.411200772625,
      "track2": 5.271682733736
    },
    "action_ratio_by_kernel": {
      "imq": 3.057897410793,
      "matern": 4.07371703477,
      "rbf": 3.135916143692
    },
    "branch_type": "baseline",
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "kernel_robustness_failed",
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 3.441862677305,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "per_basis_gates",
    "action_ratio_by_basis": {
      "integrated": 2.411200772625,
      "track2": 5.271682733736
    },
    "action_ratio_by_kernel": {
      "imq": 3.057897410793,
      "matern": 4.07371703477,
      "rbf": 3.135916143692
    },
    "branch_type": "gate_policy",
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "kernel_robustness_failed",
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 3.441862677305,
    "safe_for_thesis_claim": false
  }
]
```
