# Track 4 Observer-State Robustness Refresh - 2026-05-21

This note records the stricter Track 4 observer-state action gate after adding seed, basis, basis x seed, kernel x basis, and kernel x seed x basis robustness checks.

Interpretation: the `track2` action-basis replay is robust across the 3-seed/3-kernel focused matrix. The `integrated` basis is not robust because `integrated|seed420` fails the shuffled-hysteresis baseline, specifically `matern|seed420|integrated`. Therefore the pooled `summary_all` is intentionally thesis-unsafe for an unqualified multi-basis claim.

```json
{
  "summary_all": {
    "action_ratio_by_basis": {
      "integrated": 2.402412863734,
      "track2": 5.271682733736
    },
    "action_ratio_by_kernel": {
      "imq": 3.102667894013,
      "matern": 4.011597085192,
      "rbf": 3.135388123204
    },
    "action_ratio_by_seed": {
      "42": 3.435355407985,
      "420": 3.441862677305,
      "4200": 3.428393206528
    },
    "claim_failure_reasons": [
      "basis_seed_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
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
    "path": "outputs\\track4_action_graph\\observer_state_matrix_3seed_full_20260521\\summary_all\\track4_observer_state_ablation_summary.json",
    "point_estimate_action_ratio": 3.435198206244,
    "safe_for_thesis_claim": false
  },
  "summary_integrated": {
    "action_ratio_by_basis": {
      "integrated": 2.402412863734
    },
    "action_ratio_by_kernel": {
      "imq": 2.254096204555,
      "matern": 2.832707697147,
      "rbf": 2.110991630529
    },
    "action_ratio_by_seed": {
      "42": 2.40194306187,
      "420": 2.411200772625,
      "4200": 2.394127078061
    },
    "claim_failure_reasons": [
      "seed_robustness_failed",
      "basis_seed_robustness_failed",
      "kernel_seed_basis_robustness_failed"
    ],
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
    "path": "outputs\\track4_action_graph\\observer_state_matrix_3seed_full_20260521\\summary_integrated\\track4_observer_state_ablation_summary.json",
    "point_estimate_action_ratio": 2.402412863734,
    "safe_for_thesis_claim": false
  },
  "summary_track2": {
    "action_ratio_by_basis": {
      "track2": 5.271682733736
    },
    "action_ratio_by_kernel": {
      "imq": 4.541147017104,
      "matern": 5.906172469092,
      "rbf": 5.281210041443
    },
    "action_ratio_by_seed": {
      "42": 5.271682733736,
      "420": 5.271682733736,
      "4200": 5.271682733736
    },
    "claim_failure_reasons": [],
    "failing_basis_seed_cells": {},
    "failing_kernel_seed_basis_cells": {},
    "path": "outputs\\track4_action_graph\\observer_state_matrix_3seed_full_20260521\\summary_track2\\track4_observer_state_ablation_summary.json",
    "point_estimate_action_ratio": 5.271682733736,
    "safe_for_thesis_claim": true
  }
}
```
