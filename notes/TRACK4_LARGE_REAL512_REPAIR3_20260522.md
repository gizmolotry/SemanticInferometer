# Track 4 Large Real-512 Repair3 Run - 2026-05-22

## Run

- Pipeline root: `outputs/track4_focused_basis_validation/real_large512_repair_seedgrid_20260522`
- Track 4 replay root: `outputs/track4_action_graph/observer_state_repair3_real_large512_20260522`
- Corpus: `real`
- Article limit: `512`
- Kernels: `rbf`, `matern`, `imq`
- Seeds: `42`, `420`, `4200`
- Pipeline basis: `track2`
- Action replay branches: `baseline_raw_action`, `null_calibrated_hysteresis`, `richer_walker_state`
- Action replay bases: `track2`, `integrated`

## Completion

- 9 large observer artifacts were generated.
- 216 Track 4 action summaries were generated.
- No stderr was emitted by the chained run.
- Targeted Track 4 tests passed after the run.

## Pipeline-Level Track 4 Basis Probe

All 9 large real pipeline cells reported `safe_for_thesis_claim=true` for the Track 4 basis probe.

| Kernel | Seed | Closed Loop Rate | Mean Work Integral | Primary Zones |
| --- | ---: | ---: | ---: | ---: |
| `rbf` | 42 | 1.000 | 21.387 | 4 |
| `rbf` | 420 | 1.000 | 21.909 | 4 |
| `rbf` | 4200 | 1.000 | 21.189 | 4 |
| `matern` | 42 | 1.000 | 5.094 | 4 |
| `matern` | 420 | 0.933 | 8.632 | 4 |
| `matern` | 4200 | 1.000 | 3.882 | 4 |
| `imq` | 42 | 1.000 | 20.637 | 4 |
| `imq` | 420 | 1.000 | 22.530 | 4 |
| `imq` | 4200 | 1.000 | 18.578 | 4 |

## Action Replay Means, Full Variant Only

| Branch | Basis | Mean Action | Mean Raw Hysteresis | Mean Null Hysteresis | Mean Excess Hysteresis | Mean Calibrated Hysteresis | Mean Virtual Nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_raw_action` | `track2` | 27.570 | 6.022 | 0.000 | 6.022 | 6.022 | 0 |
| `baseline_raw_action` | `integrated` | 48.622 | 7.920 | 0.000 | 7.920 | 7.920 | 0 |
| `null_calibrated_hysteresis` | `track2` | 24.036 | 10.266 | 16.376 | -6.110 | 2.212 | 0 |
| `null_calibrated_hysteresis` | `integrated` | 45.081 | 14.227 | 30.871 | -16.644 | 3.517 | 0 |
| `richer_walker_state` | `track2` | 45.571 | 2.394 | 0.000 | 2.394 | 2.394 | 11642 |
| `richer_walker_state` | `integrated` | 58.722 | 3.816 | 0.000 | 3.816 | 3.816 | 11615 |

## Interpretation

This was intentionally real-only, so the branch-comparison thesis gate fails because no control corpora are present. The run is still useful as a scale test:

- The production pipeline can process 512 real articles across 3 kernels and 3 seeds.
- Track 4 basis-probe terrain coverage improved at this scale: all pipeline cells touched all 4 primary zones.
- Null-calibrated hysteresis behaves as a true null action: signed excess is negative when shuffled hysteresis exceeds real hysteresis, and calibrated hysteresis uses only positive excess.
- Richer virtual-state is operational at scale and inserts roughly 11.5k to 11.8k latent transition states per cell.
- The larger run suggests the next thesis-grade test should add matched controls at `limit=512`, not merely repeat real-only.
