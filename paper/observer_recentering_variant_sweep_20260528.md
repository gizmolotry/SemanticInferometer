# Observer Recentering Variant Sweep - 2026-05-28

## Artifact

- JSON: `outputs/observer_recenter_robustness_suite/variant_sweep_20260528/observer_recenter_robustness_suite.json`
- CSV: `outputs/observer_recenter_robustness_suite/variant_sweep_20260528/observer_recenter_robustness_suite.csv`
- Grid: `rbf`, `matern`, `imq` x seeds `42`, `420`, `4200`
- Label basis: `perspective_tag`

## Result

| Baseline | Cells Passed | Mean Primary Gain | Mean All-Pairs Gain | Mean Silhouette Gain | Mean NMI Gain |
|---|---:|---:|---:|---:|---:|
| `translation_only` | 0/9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `artifact_view` | 0/9 | n/a | n/a | n/a | n/a |
| `raw_track2_pca` | 6/9 | 0.2563 | 0.2193 | 0.4016 | 0.1653 |
| `cls_mean_pca` | 7/9 | 0.2839 | 0.2984 | 0.1657 | 0.0602 |
| `local_track_recompute` | 3/9 | 0.4174 | 0.2869 | 0.1796 | 0.0880 |
| `local_track_recompute:uniform_weighted_rks` | 7/9 | 0.2895 | 0.3045 | 0.1677 | 0.0671 |
| `local_track_recompute:local_tangent_pca` | 1/9 | 0.4184 | 0.2901 | 0.2024 | 0.0887 |
| `local_track_recompute:cls_mean_pca` | 7/9 | 0.2839 | 0.2984 | 0.1657 | 0.0602 |

## Engineering Interpretation

The original focus-weighted local recompute path is too anchor-sensitive for this small 28-article synthetic grid. It produces the largest mean primary label gain, but it fails strict robustness because some anchors lose silhouette, all-pairs, permutation, or NMI/ARI support.

The strongest repair candidate is `local_track_recompute:uniform_weighted_rks`. It keeps the local recompute contract, preserves shared RKS projection, and raises the strict pass count from `3/9` to `7/9`. This suggests the failure was not "RKS is bad" or "local recompute is fake"; the likely failure was focus-spectral weighting over-amplifying sparse anchor-specific observer directions.

The `local_tangent_pca` variant is an important negative result. It has high mean label gain but passes only `1/9`, so simply bypassing RKS does not stabilize the observer-centered manifold.

## Paper Boundary

Safe claim: observer-centered manifolds require recomputation, not coordinate translation; uniform-weighted local recompute is now a credible repair candidate that matches the strongest simple PCA pass count on this controlled grid.

Unsafe claim: observer-local recompute is universally superior to simple PCA baselines or real-corpus ideological labels are validated without independent labels.
