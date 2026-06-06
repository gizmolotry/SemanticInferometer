# Results (Canonical Evidence Ledger)

This file is the single source of truth for thesis-facing results.

## Reporting Policy
- Report only canonical runs defined in `METHODS.md`.
- Keep exploratory runs in notes, not in final claim tables.
- Every reported number must map to a run ID and artifact path.

## Current Baseline Snapshot (as of 2026-02-26)

### Synthetic run example
- Run ID: `experiments_20260225_225508`
- Artifact: `outputs/experiments/runs/experiments_20260225_225508/experiment_manifest.json`
- Summary:
  - `status`: success
  - `kernel`: rbf
  - `seed`: 42
  - `NMI`: 0.7123805298491227
  - `ARI`: 0.417812262798347

This is a useful baseline but not yet a full canonical matrix (single run key only).

## Canonical Run Registry

_Auto-generated from manifests on 2026-05-05T17:22:06._

| Run ID | Date | Purpose | Command Class | Status | Canonical | Notes |
|---|---|---|---|---|---|---|
| experiments_20260314_162932 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260314_184111 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260314_194600 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260314_195428 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260314_200543 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260314_201423 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260314_202246 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260314_202653 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260314_203523 | 2026-03-14 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260315_121545 | 2026-03-15 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7469; mean_ari=0.4056 |
| experiments_20260315_191129 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_191510 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_191927 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_192345 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_192654 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_193038 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=BeliefTransformerPipeline.process_month.<locals>._run_channel.<locals>._ensure_8 |
| experiments_20260315_193600 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=BeliefTransformerPipeline.process_month.<locals>._run_channel.<locals>._ensure_8 |
| experiments_20260315_193906 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_194211 | 2026-03-15 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim must be divisible by num_heads |
| experiments_20260315_195022 | 2026-03-15 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260315_195247 | 2026-03-15 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260316_165114 | 2026-03-16 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=local variable 'torch' referenced before assignment |
| experiments_20260316_165528 | 2026-03-16 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260318_124750 | 2026-03-18 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=name 'get_git_hash' is not defined |
| experiments_20260318_124908 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.3694; mean_ari=0.1894 |
| experiments_20260318_125330 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260318_125721 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260318_130116 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260318_135238 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7054; mean_ari=0.4407 |
| experiments_20260318_165702 | 2026-03-18 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=embed_dim and num_heads must be greater than 0, got embed_dim=0 and num_heads=8  |
| experiments_20260318_170147 | 2026-03-18 | Synthetic | Synthetic | Failed/Partial | No | successful_runs=0/1; error=name 'exp_dir' is not defined |
| experiments_20260318_170238 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.0000; mean_ari=0.0000 |
| experiments_20260318_170511 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7222; mean_ari=0.2658 |
| experiments_20260318_172610 | 2026-03-18 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7222; mean_ari=0.2658 |
| experiments_20260319_171221 | 2026-03-19 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.6576; mean_ari=0.2653 |
| experiments_20260322_115758 | 2026-03-22 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7222; mean_ari=0.2658 |
| experiments_20260322_120905 | 2026-03-22 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.0000; mean_ari=0.0000 |
| experiments_20260322_124556 | 2026-03-22 | Corpus matrix | Suite | Success | Yes | experiments=10; kernels=rbf,laplacian,rq,imq,matern; channels=logits,cls; corpora=real |
| experiments_20260322_131259 | 2026-03-22 | Corpus matrix | Suite | Success | Yes | experiments=40; kernels=rbf,laplacian,rq,imq,matern; channels=logits,cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260322_164026 | 2026-03-22 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf,laplacian,rq,imq,matern; channels=logits,cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260327_164553 | 2026-03-27 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260331_132547 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_132835 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_133145 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_133538 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_134827 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_140834 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_183315 | 2026-03-31 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_183617 | 2026-03-31 | Corpus matrix | Suite | Failed/Partial | No | experiments=1; failures=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_183758 | 2026-03-31 | Corpus matrix | Suite | Failed/Partial | No | experiments=1; failures=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260331_183945 | 2026-03-31 | Corpus matrix | Suite | Failed/Partial | No | experiments=1; failures=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260401_052605 | 2026-04-01 | Corpus matrix | Suite | Failed/Partial | No | experiments=1; failures=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260401_053144 | 2026-04-01 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260401_060613 | 2026-04-01 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260401_093406 | 2026-04-01 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260401_165142 | 2026-04-01 | Corpus matrix | Suite | Success | Yes | experiments=2; kernels=rbf; channels=logits,cls; corpora=real |
| experiments_20260401_165342 | 2026-04-01 | Corpus matrix | Suite | Success | Yes | experiments=2; kernels=rbf; channels=logits,cls; corpora=real |
| experiments_20260401_165738 | 2026-04-01 | Corpus matrix | Suite | Success | Yes | experiments=2; kernels=rbf; channels=logits,cls; corpora=real |
| experiments_20260402_055445 | 2026-04-02 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260402_055810 | 2026-04-02 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260402_060234 | 2026-04-02 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260402_060643 | 2026-04-02 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260402_072107 | 2026-04-02 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=rbf; channels=cls; corpora=real |
| experiments_20260402_185107 | 2026-04-02 | Corpus matrix | Suite | Failed/Partial | No | experiments=3; failures=2; kernels=matern; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260403_173250 | 2026-04-03 | Corpus matrix | Suite | Failed/Partial | No | experiments=2; failures=1; kernels=rbf; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260504_142311 | 2026-05-04 | Corpus matrix | Suite | Failed/Partial | No | experiments=4; failures=3; kernels=rbf,matern,imq; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260504_152434 | 2026-05-04 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260504_152436 | 2026-05-04 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260504_152438 | 2026-05-04 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260504_152509 | 2026-05-04 | Corpus matrix | Suite | Failed/Partial | No | experiments=8; failures=3; kernels=rbf,matern,imq; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260504_162500 | 2026-05-04 | Corpus matrix | Suite | Success | Yes | experiments=4; kernels=matern; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260504_171454 | 2026-05-04 | Corpus matrix | Suite | Success | Yes | experiments=4; kernels=matern; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260504_181207 | 2026-05-04 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=matern; channels=cls; corpora=real |
| experiments_20260504_183010 | 2026-05-04 | Corpus matrix | Suite | Success | Yes | experiments=1; kernels=matern; channels=cls; corpora=real |
| experiments_20260505_033149 | 2026-05-05 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260505_152950 | 2026-05-05 | Corpus matrix | Suite | Success | Yes | experiments=4; kernels=matern; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260505_161937 | 2026-05-05 | Corpus matrix | Suite | Success | Yes | experiments=4; kernels=matern; channels=cls; corpora=real,control_constant,control_shuffled,control_random |
| experiments_20260505_165356 | 2026-05-05 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260505_171521 | 2026-05-05 | Corpus matrix | Suite | Unknown/Empty | No | no experiments[] entries; config_keys=kernels,channels,corpora,seeds |
| experiments_20260505_171824 | 2026-05-05 | Synthetic | Synthetic | Success | Yes | successful_runs=1/1; mean_nmi=0.7226; mean_ari=0.3871 |

## Thesis Tables To Populate

### Table 0: Observer Recentring Ablation (Controlled Synthetic Labels)
- **Run under test**: `outputs/experiments/runs/experiments_20260314_184111/synthetic/rbf_seed42`
- **Label basis**: `perspective_tag` (8 balanced ideological synthetic labels, 10 articles each)
- **New trusted artifact**: `outputs/observer_recenter_meaning_probe/synthetic_local_track_recompute_multianchor_normalized_20260528/observer_recenter_meaning_probe.json`
- **Legacy baseline artifact**: `outputs/observer_recenter_meaning_probe/synthetic_artifact_view_normalized_20260528/observer_recenter_meaning_probe.json`
- **Interpretation**: Visual/artifact recentering alone does not create semantic deformation. Observer-centered manifolds become semantically meaningful only when Track 2 geometry, Track 1.5 shear/stress, and Track 3 density are locally recomputed from the selected observer payload.

| Recenter Mode | Local Track 2/1.5/3 Recompute | Observer Anchors | Label Coverage | Scale-Normalized Label-Gap Gain | All-Pairs Separation | Centroid Contraction | Silhouette | NMI/ARI | Permutation Test | Thesis-Safe For This Claim |
|---|---:|---:|---|---:|---|---|---|---|---|---|
| `artifact_view` | No | 1 | 1/8 labels | 0.0015 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | No |
| `local_track_recompute` | Yes | 8 | 8/8 labels | 0.3458 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | Yes |

Additional local-recompute aggregate metrics:
- Mean scale-normalized all-pairs separation gain: `0.4186`
- Mean silhouette gain: `0.1718`
- Mean NMI gain: `0.1241`
- Permutation observer p-value: `0.005` for each anchor in the controlled synthetic run
- Claim readiness: `mechanical_recenter_nontrivial=True`, `local_track_recompute_supported=True`, `ideological_validation_suite_pass=True`

Claim boundary:
- This supports the controlled synthetic claim that observer-local recomputation recovers planted ideological structure better than a translation-only or artifact-view recentering baseline.
- This does not, by itself, prove real-world ideological labels without independent labels or a defensible proxy-label protocol.
- The JSON field names `real_*` in this diagnostic mean "non-fixture evaluated run"; in this table the evaluated run is synthetic.

### Table 0B: Observer Recentring Robustness Grid (3 Kernels x 3 Seeds)
- **Run root**: `outputs/experiments/runs/experiments_20260506_192553/synthetic`
- **Robustness artifact**: `outputs/observer_recenter_robustness_suite/robustness_3x3_baselines_20260528/observer_recenter_robustness_suite.json`
- **CSV table**: `outputs/observer_recenter_robustness_suite/robustness_3x3_baselines_20260528/observer_recenter_robustness_suite.csv`
- **Variant sweep artifact**: `outputs/observer_recenter_robustness_suite/variant_sweep_20260528/observer_recenter_robustness_suite.json`
- **Variant sweep CSV**: `outputs/observer_recenter_robustness_suite/variant_sweep_20260528/observer_recenter_robustness_suite.csv`
- **Cells**: `rbf`, `matern`, `imq` x seeds `42`, `420`, `4200`
- **Label basis**: `perspective_tag`

| Baseline | Cells Passed | Pass Rate | Mean Primary Label Gain | Mean Scale-Normalized All-Pairs Gain | Mean Silhouette Gain | Mean NMI Gain |
|---|---:|---:|---:|---:|---:|---:|
| `translation_only` | 0/9 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `artifact_view` | 0/9 | 0.000 | n/a | n/a | n/a | n/a |
| `raw_track2_pca` | 6/9 | 0.667 | 0.2563 | 0.2193 | 0.4016 | 0.1653 |
| `cls_mean_pca` | 7/9 | 0.778 | 0.2839 | 0.2984 | 0.1657 | 0.0602 |
| `local_track_recompute` | 3/9 | 0.333 | 0.4174 | 0.2869 | 0.1796 | 0.0880 |
| `local_track_recompute:uniform_weighted_rks` | 7/9 | 0.778 | 0.2895 | 0.3045 | 0.1677 | 0.0671 |
| `local_track_recompute:local_tangent_pca` | 1/9 | 0.111 | 0.4184 | 0.2901 | 0.2024 | 0.0887 |
| `local_track_recompute:cls_mean_pca` | 7/9 | 0.778 | 0.2839 | 0.2984 | 0.1657 | 0.0602 |

Robustness interpretation:
- The earlier local-recompute proof is valid for its controlled cell, but the broader 3x3 grid does not support a universal superiority claim.
- `local_track_recompute` has the strongest mean primary label gain, but only passes the strict five-family gate in the three Matern cells.
- `raw_track2_pca` and `cls_mean_pca` are strong reviewer baselines and must be included in any publishable version.
- `local_track_recompute:uniform_weighted_rks` is the best current local-recompute repair candidate: it preserves the local recompute contract while improving the strict pass count from `3/9` to `7/9`.
- `local_track_recompute:local_tangent_pca` keeps high primary label gain but collapses under strict NMI/centroid/silhouette gates, suggesting the shared RKS projection is not the main instability.
- `translation_only` fails across all cells, supporting the claim that centering alone is insufficient.
- `artifact_view` is unavailable in the 3x3 matrix because those cells do not contain pre-rendered observer artifact directories; the materialized artifact-view baseline above remains the direct legacy-path comparison.

Paper-safe conclusion:
- Safe: observer-local recomputation is a promising mechanism; uniform weighting stabilizes it to `7/9` cells on the controlled synthetic grid, while translation-only and artifact-view controls fail.
- Unsafe: the original focus-weighted observer-local recomputation universally outperforms simpler baselines across kernels and seeds.

### Table A: Real vs Controls (Representative Snapshot)
- **Run ID**: `experiments_20260314_040950`
- **Kernels**: rbf, laplacian, rq, imq (Table shows RBF/Logits means)
- **Seeds**: 42, 420, 4200

| Corpus | Simple Variance (mean) | Procrustes (mean) | Consensus % | Residual % |
|---|---|---|---|---|
| Real | 0.0254 | 0.9173 | 33.82 | 66.18 |
| Constant | 0.0255 | 0.0021 | 33.03 | 66.97 |
| Shuffled | 0.0251 | 0.8776 | 34.35 | 65.65 |
| Random | 0.0253 | 0.9286 | 33.53 | 66.47 |

_Note: Procrustes measures geometric alignment to the article manifold. Higher values in controls (Shuffled/Random) indicate they preserve significant structural shadows of the real embedding space even when semantic labels are destroyed._

### Table B: Synthetic Ground-Truth Recovery
- Metrics: NMI, ARI
- Granularity: per kernel x seed and aggregate

### Table C: Stability / Variance
- Metrics: stage-wise or end-to-end stability metrics used in thesis claims
- Granularity: per run + aggregate confidence intervals

## Figure Inventory (To Fill)
- Figure 1: Core geometry/manifold visualization from canonical run set
- Figure 2: Real vs control separation summary
- Figure 3: Synthetic validation performance (NMI/ARI)
- Figure 4: Robustness/stability view across seeds and kernels

## Known Non-Canonical Evidence (Do Not Claim Directly)
- Historical logs with runtime failures (for example earlier synthetic empty-input traces).
- Archived pre-pivot and exploratory outputs under `archive/`.
