# Track 1.5 Gradient Probe - 2026-05-25

## Purpose

Test whether replacing or augmenting the current Track 1.5 forward-pass polarity
delta with hidden-state gradients improves observer shear separation.

The experiment compares three feature bases on the same real/control articles:

- `forward_delta`: current A-minus-B mean-pooled observer delta.
- `logit_gradient_delta`: gradient of the NLI entailment logit with respect to
  final hidden states, pooled over tokens, then A-minus-B.
- `anchor_gradient_delta`: gradient of cosine alignment between the article view
  and a pure hypothesis-anchor vector, then A-minus-B.

## Runs

- Smoke: `outputs/track15_gradient_probe/smoke_n1_20260525`
- Main small run: `outputs/track15_gradient_probe/real_controls_n8_20260525`
- Stabilization run: `outputs/track15_gradient_probe/real_controls_n16_20260525`

Main artifacts:

- `track15_gradient_probe_summary.json`
- `track15_gradient_probe_features.npz`
- `metric_cartography.csv`
- `per_article_metrics.csv`

## n=16 Result Snapshot

| Method | Real observer abs cosine | Stochastic abs-log separation | Real/random variance | Real/shuffled variance |
|---|---:|---:|---:|---:|
| `forward_delta` | 0.392 | 0.494 | 0.792 | 0.511 |
| `logit_gradient_delta` | 0.290 | 0.785 | 0.165 | 0.195 |
| `anchor_gradient_delta` | 0.326 | 0.248 | 0.957 | 1.062 |

## Interpretation

The narrow gradient claim is partially supported: gradient bases reduce observer
cross-correlation relative to the current forward delta basis. Logit gradients
are the most orthogonal of the three tested bases.

The replacement claim is not supported. Logit gradients separate from stochastic
controls mostly because shuffled/random controls have much larger gradient
variance than real articles. That may be a useful confusion/noise signal, but it
is not clean evidence of a better semantic manifold basis.

Anchor gradients are more stable against stochastic controls, but they sit close
to variance parity and therefore do not currently provide stronger real/control
separation than the production forward delta basis.

## Engineering Decision

Do not replace Track 1.5 with gradients yet.

Keep gradient sensitivity as an ablation candidate and possible Track 4 force
field input:

- `logit_gradient_delta`: useful for detecting model confusion / control chaos.
- `anchor_gradient_delta`: potentially useful as a smoother framing-force field.
- `forward_delta`: remains the production Track 1.5 basis for now.

Next useful test: run a Track 4 action-graph slice where the shear field is
swapped from `forward_delta` to `anchor_gradient_delta`, while Track 2 content
geometry remains unchanged.
