# Track 4 Seven-Branch Observer-State Sweep - 2026-05-21

Stress-cell run: `kernel=matern`, `seed=420`, bases `track2/integrated`, corpora `real/control_random/control_shuffled`, seven engineering branches.

Key finding: `null_calibrated_hysteresis` and `richer_walker_state` repaired the exact integrated-basis shuffled-hysteresis failure in this cell. `track2_default` remains strongest but avoids the integrated basis rather than fixing it.

```json
[
  {
    "action_branch": "track2_default",
    "action_ratio_by_basis": {
      "track2": 5.906172469092
    },
    "branch_type": "basis_policy",
    "failing_basis_seed_cells": {},
    "failing_kernel_seed_basis_cells": {},
    "failure_reasons": [],
    "point_estimate": 5.906172469092,
    "safe_for_thesis_claim": true
  },
  {
    "action_branch": "null_calibrated_hysteresis",
    "action_ratio_by_basis": {
      "integrated": 3.059763972657,
      "track2": 5.942628362851
    },
    "branch_type": "calibration",
    "failing_basis_seed_cells": {},
    "failing_kernel_seed_basis_cells": {},
    "failure_reasons": [],
    "point_estimate": 4.176245634331,
    "safe_for_thesis_claim": true
  },
  {
    "action_branch": "richer_walker_state",
    "action_ratio_by_basis": {
      "integrated": 2.94456023115,
      "track2": 6.045234364419
    },
    "branch_type": "state_model",
    "failing_basis_seed_cells": {},
    "failing_kernel_seed_basis_cells": {},
    "failure_reasons": [],
    "point_estimate": 4.148139052797,
    "safe_for_thesis_claim": true
  },
  {
    "action_branch": "virtual_transition_states",
    "action_ratio_by_basis": {
      "integrated": 3.558688003548,
      "track2": 6.946201581256
    },
    "branch_type": "graph_densification_proxy",
    "failing_basis_seed_cells": {
      "integrated|seed420": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 4.839175974608,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "separated_action_channels",
    "action_ratio_by_basis": {
      "integrated": 2.947912448589,
      "track2": 5.982270221208
    },
    "branch_type": "channel_ablation",
    "failing_basis_seed_cells": {
      "integrated|seed420": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 4.116685653398,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "baseline_raw_action",
    "action_ratio_by_basis": {
      "integrated": 2.925977545078,
      "track2": 5.906172469092
    },
    "branch_type": "baseline",
    "failing_basis_seed_cells": {
      "integrated|seed420": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "shuffled_hysteresis_baseline_not_separated",
      "kernel_robustness_failed",
      "seed_robustness_failed",
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 4.07371703477,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "per_basis_gates",
    "action_ratio_by_basis": {
      "integrated": 2.925977545078,
      "track2": 5.906172469092
    },
    "branch_type": "gate_policy",
    "failing_basis_seed_cells": {
      "integrated|seed420": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "shuffled_hysteresis_baseline_not_separated",
      "kernel_robustness_failed",
      "seed_robustness_failed",
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 4.07371703477,
    "safe_for_thesis_claim": false
  },
  {
    "action_branch": "path_ensemble_tpt",
    "action_ratio_by_basis": {
      "integrated": 2.007727259575,
      "track2": 4.691176454351
    },
    "branch_type": "ensemble_probe",
    "failing_basis_seed_cells": {
      "integrated|seed420": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failing_kernel_seed_basis_cells": {
      "matern|seed420|integrated": [
        "shuffled_hysteresis_baseline_not_separated"
      ]
    },
    "failure_reasons": [
      "basis_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_basis_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
    "point_estimate": 3.046097711222,
    "safe_for_thesis_claim": false
  }
]
```
