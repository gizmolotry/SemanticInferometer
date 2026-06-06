# Observer Recentering Robustness Interpretation - 2026-05-28

## Scope

This note interprets the current 3 x 3 observer-recentering robustness packet:

- Run root: `outputs/experiments/runs/experiments_20260506_192553/synthetic`
- Robustness artifact: `outputs/observer_recenter_robustness_suite/robustness_3x3_baselines_20260528/observer_recenter_robustness_suite.json`
- CSV table: `outputs/observer_recenter_robustness_suite/robustness_3x3_baselines_20260528/observer_recenter_robustness_suite.csv`
- Grid: kernels `rbf`, `matern`, `imq` x seeds `42`, `420`, `4200`
- Label basis: `perspective_tag`

## Impact Summary

The result is positive but narrower than a universal observer-recentering claim.

`local_track_recompute` passes the strict five-family gate in the three Matern cells (`3/3` Matern, `3/9` overall) and has the largest mean primary label gain (`0.4174`). That supports observer-local recomputation as a real mechanism on the controlled synthetic packet, especially for the Matern kernel.

The same result fails universal superiority. `raw_track2_pca` passes `6/9` cells and `cls_mean_pca` passes `7/9` cells, so simple PCA baselines are strong reviewer baselines rather than weak controls. Any publishable version must report them directly.

`translation_only` fails all cells (`0/9`) with zero gain, which supports the narrower claim that centering alone is insufficient. The mechanism needs observer-local recomputation of geometry, shear/stress, and density, not just moving coordinates around a selected observer.

## Safe Claims

- Observer-local recomputation is a promising controlled-synthetic mechanism with robust Matern behavior across the tested seeds.
- Local recompute produces the largest mean primary label gain among the tested baselines in this packet.
- Translation-only recentering fails the robustness grid, so the evidence rejects a pure coordinate-shift explanation.
- PCA baselines are strong and must be included as serious comparators in publication language.

## Unsafe Claims

- Observer-local recomputation universally outperforms simpler baselines across kernels and seeds.
- The result is kernel-invariant; current strict-gate support is Matern-specific.
- PCA baselines are merely sanity checks or weak straw controls.
- The 3 x 3 synthetic robustness grid by itself validates real-world ideological labeling.

## Next Engineering Variants Must Prove

Future variants need to beat the current result on robustness, not just on a selected cell. A stronger engineering branch should:

- Preserve the Matern `3/3` pass while improving RBF and IMQ strict-gate pass rates.
- Match or exceed PCA baselines on cell pass count, not only on mean primary label gain.
- Show gains across primary label separation, scale-normalized all-pairs separation, silhouette, and NMI without metric cherry-picking.
- Keep `translation_only`, `artifact_view` where available, `raw_track2_pca`, and `cls_mean_pca` in the comparison set.
- Demonstrate that each added component contributes beyond PCA: local Track 2 geometry, Track 1.5 shear/stress, and Track 3 density should each have ablation evidence.

Publication framing should therefore be: observer-local recomputation has a meaningful, Matern-robust controlled-synthetic signal and defeats translation-only recentering, but it has not yet displaced simple PCA as the universally stronger baseline.
