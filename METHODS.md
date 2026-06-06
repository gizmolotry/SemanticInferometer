# Methods (Canonical Thesis Protocol)

## Scope
This document defines the canonical protocol for thesis claims. Any run outside this protocol is exploratory.

## Research Objective
Evaluate whether framing-sensitive structure is detectable and stable across:
- Real corpus
- Control corpora (`control_constant`, `control_shuffled`, `control_random`)
- Multiple kernel families
- Multiple seeds

## Canonical Configuration
- Runner: `run_full_experiment_suite.py`
- Mode: `enhanced`
- Kernels: `rbf matern imq`
- Seeds: `42 420 4200`
- Channels: `cls`
- Corpora: `real control_constant control_shuffled control_random`
- Article limit per corpus: `120` (fallback defense bundle: `60` if runtime or memory instability appears)

## Canonical Commands

### A) Main corpus matrix
```powershell
python run_full_experiment_suite.py --mode enhanced --seeds 42 420 4200 --kernels rbf matern imq --channels cls --corpora real control_constant control_shuffled control_random --limit 120 --control-metric-basis comprehensive
```

### B) Synthetic validation (ground-truth check)
```powershell
python run_full_experiment_suite.py --synthetic --mode enhanced --seeds 42 420 4200 --kernels rbf matern imq --channels cls --limit 60
```

### C) Observer-centered local recentering validation

Observer-centered manifolds must be tested as local recomputations, not as visual recentering. The trusted protocol is:

```powershell
python scripts\run_observer_recenter_meaning_probe.py --run-dir outputs\experiments\runs\experiments_20260314_184111\synthetic\rbf_seed42 --output-dir outputs\observer_recenter_meaning_probe\synthetic_local_track_recompute_multianchor_normalized_20260528 --min-label-count 2 --label-column perspective_tag --recenter-mode local_track_recompute
```

The legacy comparison baseline is:

```powershell
python scripts\run_observer_recenter_meaning_probe.py --run-dir outputs\experiments\runs\experiments_20260314_184111\synthetic\rbf_seed42 --output-dir outputs\observer_recenter_meaning_probe\synthetic_artifact_view_normalized_20260528 --min-label-count 2 --label-column perspective_tag --recenter-mode artifact_view
```

Methodological rule:
- `artifact_view` is a legacy diagnostic baseline. It may center or display an observer artifact, but it must not be described as recomputing an observer manifold.
- `local_track_recompute` is the thesis-facing observer recentering path. It recomputes local Track 2 geometry, Track 1.5 observer shear/stress, and Track 3 density from the selected observer payload before evaluating semantic structure.
- Distance-based observer/recenter comparisons must use scale-normalized geometry because global and observer-local charts do not share raw coordinate units.

For paper language, describe observer recentering as observer-local re-estimation of geometry, shear, and density. Do not describe it as merely "moving an article to the center."

### D) Observer recentering robustness and reviewer baselines

The robustness packet repeats the observer-recentering test across a 3 x 3 synthetic grid and compares against simple reviewer baselines:

```powershell
python scripts\run_observer_recenter_robustness_suite.py --synthetic-root outputs\experiments\runs\experiments_20260506_192553\synthetic --output-dir outputs\observer_recenter_robustness_suite\robustness_3x3_baselines_20260528 --kernels rbf matern imq --seeds 42 420 4200 --label-column perspective_tag --baselines translation_only artifact_view raw_track2_pca cls_mean_pca local_track_recompute
```

The local-recompute variant sweep uses the same grid and adds controlled variants around the local observer chart:

```powershell
python scripts\run_observer_recenter_robustness_suite.py --synthetic-root outputs\experiments\runs\experiments_20260506_192553\synthetic --output-dir outputs\observer_recenter_robustness_suite\variant_sweep_20260528 --kernels rbf matern imq --seeds 42 420 4200 --label-column perspective_tag --baselines translation_only artifact_view raw_track2_pca cls_mean_pca local_track_recompute local_track_recompute:uniform_weighted_rks local_track_recompute:local_tangent_pca local_track_recompute:cls_mean_pca
```

Baseline definitions:
- `translation_only`: global manifold coordinates translated so each anchor is at the origin; this tests whether centering alone creates apparent semantic gains.
- `artifact_view`: legacy pre-rendered observer artifact directories when present; this tests the older visual recenter path.
- `raw_track2_pca`: PCA projection of saved `features.npy`; this is a simple Track 2 geometry baseline.
- `cls_mean_pca`: PCA projection of the mean `cls_per_bot` observer payload; this tests whether simple embedding geometry already recovers planted labels.
- `local_track_recompute`: observer-local Track 2/1.5/3 recomputation.
- `local_track_recompute:uniform_weighted_rks`: same local recompute path but forces uniform observer weights before the shared RKS projection; this tests whether focus-spectral weighting overfits anchors.
- `local_track_recompute:local_tangent_pca`: same local recompute path but bypasses RKS and projects the weighted local tangent directly.
- `local_track_recompute:cls_mean_pca`: local recompute ledger wrapper using mean observer CLS geometry for the XY chart, while retaining local recompute provenance and terrain diagnostics.

Publication rule:
- Report this robustness grid even when it weakens the central claim. If simple PCA baselines pass more cells than local recomputation, the paper must say so.
- Treat kernel-specific success as conditional evidence, not universal validation.
- Treat `local_track_recompute:uniform_weighted_rks` as the current strongest local-recompute repair candidate, not as the production default, until it is rerun on larger synthetic and real-control packets.

## Recorded Artifacts Per Canonical Run
- Run directory under `outputs/experiments/runs/experiments_YYYYMMDD_HHMMSS`
- `experiment_manifest.json` (or equivalent run metadata file)
- Metric outputs used for thesis tables/figures
- Probe results if enabled
- Any derived figures used in thesis text

## Claim Tiering
- Controlled synthetic validation can support claims about planted ideological-label recovery and observer-local deformation under known labels.
- Real unlabeled corpus runs can support geometry/control/provenance claims, but not label-recovery claims unless source labels, proxy labels, human labels, or another independent validation target is explicitly attached.
- Track 4 should be reported as traversal telemetry or visualization unless its own action/terrain gates pass independently. It should not assign final semantic verdicts for the paper.

## Inclusion Rules
- Include only successful runs with complete required artifacts.
- Exclude runs with missing inputs, empty corpora, or runtime exceptions.
- If a rerun is needed, keep old run for traceability and mark superseded in `RESULTS.md`.

## Reproducibility Rules
- Do not change code between canonical runs in a set.
- Keep seed list fixed.
- Keep kernel list fixed.
- Keep corpus definitions fixed for the full set.
- Record exact command and timestamp for each canonical run ID.
