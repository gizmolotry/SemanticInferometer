# Track 1.5 Gradient Force Into Track 4 - 2026-05-25

## Purpose

Follow-up to `TRACK15_GRADIENT_PROBE_20260525.md`.

The first probe showed that gradients reduce observer cross-correlation but are
not safe as a wholesale Track 1.5 replacement. This probe asks a narrower
engineering question: if Track 2 content geometry is held fixed, do gradient
features work better as the Track 4 shear / observer-state force field?

## Artifacts

- Script: `scripts/run_track15_gradient_action_probe.py`
- Inputs: `outputs/track15_gradient_probe/real_controls_n16_20260525/track15_gradient_probe_features.npz`
- Output directory: `outputs/track15_gradient_probe/action_graph_n16_20260525`
- Summary: `track15_gradient_action_probe_summary.json`
- Rows: `track15_gradient_action_rows.csv`
- Comparisons: `track15_gradient_action_comparisons.csv`

## Design

All cells use `forward_delta` as the content geometry. Only `metric_stress` and
observer-state simplex are swapped:

- `forward_delta`
- `logit_gradient_delta`
- `anchor_gradient_delta`

This keeps the comparison focused on Track 1.5-as-force rather than allowing
Track 2 geometry to change underneath it.

## Result Snapshot

| Force basis | Real/random action ratio | Real/shuffled action ratio | Stochastic action abs-log | Stochastic hysteresis abs-log |
|---|---:|---:|---:|---:|
| `forward_delta` | 1.750 | 1.627 | 0.523 | 1.775 |
| `logit_gradient_delta` | 1.492 | 1.500 | 0.403 | 0.645 |
| `anchor_gradient_delta` | 1.549 | 1.526 | 0.430 | 2.950 |

## Interpretation

Forward delta remains the best global action separator in this small probe.
Gradients do not beat the production Track 1.5 basis on total action cost.

Anchor gradients are still interesting: they produce the strongest calibrated
hysteresis separation against stochastic controls. That suggests anchor-gradient
sensitivity may be useful as an observer-state / hysteresis term, not as the
entire Track 1.5 or Track 2 basis.

## Engineering Decision

Do not replace Track 1.5 with gradients.

Next viable engineering test:

- Keep `forward_delta` for global shear/action.
- Add an optional `anchor_gradient_hysteresis` component as an ablation branch.
- Test whether it improves observer-state action separation over matched seeds,
  kernels, and bases.

This is a Track 4 repair candidate, not a paper-safe claim yet.
