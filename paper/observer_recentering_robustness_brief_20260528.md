# Observer Recentring Robustness Brief

## Scope

This brief reports the 3x3 robustness and reviewer-baseline packet for observer recentering.

- Synthetic root: `outputs/experiments/runs/experiments_20260506_192553/synthetic`
- Kernels: `rbf`, `matern`, `imq`
- Seeds: `42`, `420`, `4200`
- Labels: `perspective_tag`
- Output packet: `outputs/observer_recenter_robustness_suite/robustness_3x3_baselines_20260528/observer_recenter_robustness_suite.json`
- CSV table: `outputs/observer_recenter_robustness_suite/robustness_3x3_baselines_20260528/observer_recenter_robustness_suite.csv`
- Variant sweep packet: `outputs/observer_recenter_robustness_suite/variant_sweep_20260528/observer_recenter_robustness_suite.json`
- Variant sweep CSV: `outputs/observer_recenter_robustness_suite/variant_sweep_20260528/observer_recenter_robustness_suite.csv`

## Main Result

The single-cell observer-local recomputation result is real, but it is not yet universally robust across kernels and seeds under the strict all-family validation gate.

`local_track_recompute` passes `3/9` cells. All three passing cells are the Matern kernel. RBF and IMQ produce meaningful gains but fail at least one strict validation family in the 28-article 3x3 matrix.

The follow-up variant sweep identifies a stronger local-recompute repair candidate. `local_track_recompute:uniform_weighted_rks` passes `7/9` cells, matching the `cls_mean_pca` pass count while preserving the local recompute ledger and defeating translation-only recentering.

## Baseline Comparison

| Baseline | Cells Passed | Pass Rate | Mean Primary Label Gain | Mean All-Pairs Gain | Mean Silhouette Gain | Mean NMI Gain |
|---|---:|---:|---:|---:|---:|---:|
| `translation_only` | 0/9 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `artifact_view` | 0/9 | 0.000 | n/a | n/a | n/a | n/a |
| `raw_track2_pca` | 6/9 | 0.667 | 0.2563 | 0.2193 | 0.4016 | 0.1653 |
| `cls_mean_pca` | 7/9 | 0.778 | 0.2839 | 0.2984 | 0.1657 | 0.0602 |
| `local_track_recompute` | 3/9 | 0.333 | 0.4174 | 0.2869 | 0.1796 | 0.0880 |
| `local_track_recompute:uniform_weighted_rks` | 7/9 | 0.778 | 0.2895 | 0.3045 | 0.1677 | 0.0671 |
| `local_track_recompute:local_tangent_pca` | 1/9 | 0.111 | 0.4184 | 0.2901 | 0.2024 | 0.0887 |
| `local_track_recompute:cls_mean_pca` | 7/9 | 0.778 | 0.2839 | 0.2984 | 0.1657 | 0.0602 |

Interpretation:

- `local_track_recompute` has the strongest mean primary label gain, but it is less robust under the strict five-family pass gate.
- `local_track_recompute:uniform_weighted_rks` is the strongest current repair candidate: it improves strict pass count to `7/9` and suggests focus-spectral weights were overfitting thin anchors.
- `local_track_recompute:local_tangent_pca` keeps high primary label gain but fails most cells, suggesting bypassing RKS does not solve the robustness problem.
- `raw_track2_pca` and `cls_mean_pca` are serious reviewer baselines, not straw men.
- `translation_only` fails as expected, confirming that simple coordinate centering is not enough.
- `artifact_view` is unsupported in this 3x3 matrix because the cells do not contain pre-rendered observer artifact directories; the earlier artifact-view run remains the direct legacy baseline for the materialized observer-view path.

## Kernel-Level Finding

- Matern: `local_track_recompute` passes `3/3`.
- RBF: `local_track_recompute` fails `3/3` because silhouette passes only `6/8` observer anchors despite all-pairs, NMI/ARI, centroid, and permutation passing.
- IMQ: `local_track_recompute` fails `3/3` because all-pairs/permutation and sometimes NMI/ARI fail for a subset of anchors.
- Uniform-weighted RKS: passes all RBF and Matern cells and fails only IMQ seeds `420` and `4200`, both on NMI/ARI.

## Paper-Safe Claim

Use:

> In a controlled synthetic robustness grid, observer-local recomputation produced the largest mean label-gap gain in its original focus-weighted form, while a uniform-weighted RKS variant matched the strongest PCA baseline at 7/9 passing cells. Translation-only and artifact-view baselines failed, supporting the claim that observer-centered meaning requires recomputed geometry rather than coordinate shifting alone.

Avoid:

> The original focus-weighted observer-local recomputation universally beats all baselines across kernels and seeds.

Corrected publication framing:

> The observer-local recomputation result is strongest as an ablation finding: visual recentering and translation-only controls fail, while recomputation can recover planted perspective structure. However, robustness testing shows that kernel choice and baseline geometry matter, so the publishable claim should emphasize mechanism, ablation, and conditions of success rather than universal superiority.

## Engineering Next Step

The next engineering target is not more prose. It is to validate and possibly promote the uniform-weighted RKS repair. Candidate fixes:

- Rerun `local_track_recompute:uniform_weighted_rks` on a larger synthetic corpus to see whether the two IMQ NMI/ARI failures are small-sample instability.
- Add per-anchor diagnostics for the weak `Liberal Zionist`, `Western Leftist`, and `Arab Pro-Palestine` anchors.
- Test whether uniform weighting should become the production default or remain an ablation branch.
- Run the same robustness grid on a larger synthetic corpus to determine whether the 28-article matrix is too small for strict per-anchor silhouette and NMI gates.
