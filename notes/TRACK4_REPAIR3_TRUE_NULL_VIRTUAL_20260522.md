# Track 4 Repair3 True Null/Virtual State Sweep - 2026-05-22

## What Changed

- Implemented signed null-calibrated hysteresis/action: `excess_hysteresis = real_hysteresis - shuffled_hysteresis`.
- Exported raw/null/excess/positive-excess/calibrated hysteresis terms in Track 4 NPZ and JSON outputs.
- Promoted `richer_walker_state` into a stateful action model over article node, previous direction, observer transport/hysteresis, and accumulated work bucket.
- Added virtual transition states to the richer repair branch using interpolated embeddings, observer simplex, Track 3 density, and stress/shear fields.
- Added `repair3` branch preset: `baseline_raw_action`, `null_calibrated_hysteresis`, and `richer_walker_state`.

## Full Grid

- Root: `outputs/track4_action_graph/observer_state_repair3_full_20260521`
- Scope: 3 branches x 3 seeds x 3 kernels x 2 bases, with real/control and observer-state ablations.
- Summaries emitted: 324 `track4_action_summary.json` files.
- Branch comparison: `outputs/track4_action_graph/observer_state_repair3_full_20260521/track4_action_branch_comparison.json`
- Thesis evidence refresh: `outputs/thesis_validation/observer_state_action_repair3_20260522`

## Result

| Branch | Thesis-Safe | Point Estimate | Notes |
| --- | --- | ---: | --- |
| `null_calibrated_hysteresis` | yes | 3.487 | Survived all required kernels, seeds, and bases. |
| `richer_walker_state` | no | 3.854 | Stronger point estimate, but failed `matern|seed420|integrated` shuffled-hysteresis robustness. |
| `baseline_raw_action` | no | 3.435 | Failed the same integrated/Matern seed cell and basis-seed robustness. |

## Interpretation

The real repair is currently the null-calibrated action model, not the richer virtual-state model. The richer model is not useless: it creates real latent intermediary nodes and often increases separation, but it does not yet survive the full robustness gate. Treat it as a promising engineering branch, not thesis-safe evidence.

The refreshed thesis evidence marks `track4_observer_state_action_separation` as thesis-safe in exploratory Track 4 scope:

- point estimate: 3.579988283881
- action ratio by basis: `track2=5.084515937415`, `integrated=2.653843624475`
- action ratio by kernel: `imq=3.219761399107`, `matern=4.145057170724`, `rbf=3.321467016368`
- action ratio by seed: `42=3.581470560681`, `420=3.586310828604`, `4200=3.572193819193`

## Verification

- `python -m pytest -q test_track4_action_graph.py test_track4_observer_state_matrix.py test_track4_observer_state_ablation_summary.py test_thesis_claim_matrix.py` passed.
- `python -m pytest -q` passed.
