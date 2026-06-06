# Observer Recentring Result Brief

## Core Finding

Observer recentering is only scientifically meaningful when it recomputes the local observer manifold. The legacy artifact-view baseline can visually center an article, but it does not reconstruct observer-local Track 2 geometry, Track 1.5 shear/stress, or Track 3 density. In the controlled synthetic ideological-label run, artifact-view recentering fails while local Track 2/1.5/3 recomputation passes.

Robustness update: a later 3x3 robustness grid qualifies this single-cell result. Local recomputation passes all Matern cells and has the strongest mean primary label gain, but it does not universally beat simple PCA baselines across `rbf`, `matern`, and `imq`. See `paper/observer_recentering_robustness_brief_20260528.md`.

## Evidence Artifacts

- Passing local recompute: `outputs/observer_recenter_meaning_probe/synthetic_local_track_recompute_multianchor_normalized_20260528/observer_recenter_meaning_probe.json`
- Failing legacy baseline: `outputs/observer_recenter_meaning_probe/synthetic_artifact_view_normalized_20260528/observer_recenter_meaning_probe.json`
- Evaluated run: `outputs/experiments/runs/experiments_20260314_184111/synthetic/rbf_seed42`
- Label basis: `perspective_tag`

## Result Table

| Mode | Description | Anchors | Label Coverage | Primary Gain | Validation Families Passed | Claim Status |
|---|---|---:|---|---:|---:|---|
| `artifact_view` | Legacy pre-rendered artifact/coordinate view | 1 | 1/8 | 0.0015 | 0/5 | Fails |
| `local_track_recompute` | Recompute Track 2/1.5/3 in the selected observer frame | 8 | 8/8 | 0.3458 | 5/5 | Passes |

Primary gain is mean scale-normalized label-gap gain over the translation-only baseline.

## Local Recompute Metrics

- Mean scale-normalized label-gap gain: `0.3458`
- Mean scale-normalized all-pairs separation gain: `0.4186`
- Mean silhouette gain: `0.1718`
- Mean NMI gain: `0.1241`
- Validation families: all-pairs separation, centroid contraction, silhouette, NMI/ARI, permutation within-between
- Pass profile: `8/8` anchors for all five validation families

## Paper Language

Use:

> To test whether observer recentering is more than a visualization effect, we compared a legacy artifact-view baseline against a local recomputation protocol. The artifact-view baseline centers an existing observer artifact but does not recompute the underlying geometry. The local protocol rebuilds Track 2 geometry, Track 1.5 observer shear, and Track 3 density in the selected observer frame. On the controlled synthetic corpus, the artifact-view baseline failed all semantic validation families, while the local recomputation protocol passed all five validation families across all eight planted ideological anchor labels.

Avoid:

> Moving an article to the center of the manifold proves semantic observer relativity.

Corrected claim:

> Observer relativity is supported only when recentering is implemented as observer-local re-estimation of geometry, shear, and density, and when the resulting chart improves label-recoverable structure against translation-only and artifact-view baselines.

## Claim Boundary

- Strong claim: controlled synthetic observer-local recomputation recovers planted ideological structure better than visual/artifact recentering.
- Supported mechanistic claim: observer-conditioned geometry changes are not reducible to translation-only coordinate shifts.
- Unsafe without more evidence: real-world unlabeled articles are ideologically classified correctly.
- Track 4 boundary: traversal remains telemetry/visualization unless separately validated by Track 4 action or terrain gates.
