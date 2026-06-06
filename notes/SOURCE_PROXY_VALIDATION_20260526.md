# Source Proxy Validation - 2026-05-26

## Purpose

Test whether existing real-corpus manifold geometry can recover outlet/source
structure without using source labels during geometry construction.

This is the missing bridge between:

- "the manifold is non-random"
- and "the manifold corresponds to independently checkable journalistic
  structure"

## New Harness

Added:

`scripts/source_proxy_validation.py`

The harness consumes existing run leaves. It loads:

- `article_metadata.csv` or `MONOLITH_DATA.csv`
- `features.npy` or checkpoint feature arrays
- source-like columns: `source`, `publication`, `publisher`, `outlet`

It then filters to repeated source labels and tests:

- same-source mean distance
- different-source mean distance
- source distance effect: `different - same`
- nearest-neighbor source agreement
- shuffled-label permutation nulls
- claim-safe gates for effect size and p-values

This validates editorial/source coherence only. It is not a political-bias truth
label and not a source-quality judgment.

## Clean Unit Tests

Passed:

```powershell
python -m pytest test_source_proxy_validation.py -q
```

Result:

`4 passed`

## Current Focused Leaf Result

Command:

```powershell
python scripts\source_proxy_validation.py `
  --run-dir outputs\experiments\runs\experiments_20260506_183343\matern\cls\real `
  --run-dir outputs\experiments\runs\experiments_20260506_183343\imq\cls\real `
  --out outputs\source_proxy_validation\focused_20260506_real_sources\source_proxy_validation_summary.json
```

Result:

- `thesis_safe=false`
- `run_count=2`
- `ok_run_count=0`
- `insufficient_source_run_count=2`
- Each leaf has `30` real articles but `27` unique source domains.
- No source has the default minimum `3` repeated articles.

Interpretation:

The current focused 30-article evidence leaves are not source-proxy-testable.
This is not evidence against the method. It means the sample was selected for
focused control/kernel proof, not outlet-replication validation.

## Legacy 500-Article Leaf Result

Command:

```powershell
python scripts\source_proxy_validation.py `
  --run-dir outputs\experiments\runs\experiments_20260403_173250\rbf\cls\real `
  --run-dir outputs\experiments\runs\experiments_20260402_185107\matern\cls\real `
  --run-dir outputs\experiments\runs\experiments_20260327_164553\rbf\cls\real `
  --out outputs\source_proxy_validation\legacy_500_real_sources\source_proxy_validation_summary.json
```

Result:

- `thesis_safe=false` at the bundle level.
- `ok_run_count=3`
- `passing_run_count=1`
- The Matern legacy leaf passes source-proxy gates.
- The two RBF legacy leaves fail mostly because the source-distance effect does
  not beat the shuffled-label null and nearest-neighbor excess is slightly below
  the default threshold.

Interpretation:

This is useful cartography, not a claim. It suggests source/outlet coherence may
be visible in some kernel/basis conditions, but it is not robust across the
tested legacy leaves. The clean move is a deliberately source-balanced rerun
across `rbf`, `matern`, and `imq`, not cherry-picking the passing Matern leaf.

## Relaxed Sensitivity Result

Command:

```powershell
python scripts\source_proxy_validation.py `
  --run-dir outputs\experiments\runs\experiments_20260506_183343\matern\cls\real `
  --run-dir outputs\experiments\runs\experiments_20260506_183343\imq\cls\real `
  --min-source-count 2 `
  --min-sources 3 `
  --min-articles 6 `
  --permutations 200 `
  --out outputs\source_proxy_validation\focused_20260506_real_sources\source_proxy_validation_relaxed_min2_summary.json
```

Result:

- `thesis_safe=false`
- `ok_run_count=2`
- `passing_run_count=0`
- Repeated-source subset: `6` articles from `3` sources.
- Matern source distance effect: `19.25`
- IMQ source distance effect: `23.47`
- Nearest-neighbor source excess: `0.133` in both leaves.
- Permutation p-values: about `0.169` for distance effect and `0.303` for
  nearest-neighbor excess.

Interpretation:

There is a suggestive same-source closeness signal in the tiny repeated-source
subset, but the subset is far too small for a claim. The signal does not beat
the shuffled-label null at publication thresholds.

## What This Means

We still need a real source-proxy validation run.

The next run should deliberately sample repeated outlets, for example:

- at least `6` sources
- at least `8-10` articles per source
- same channel/kernels/seeds as the focused proof bundle
- same verifier and packet flow

Acceptance target:

- same-source distance lower than different-source distance
- nearest-neighbor source agreement above shuffled labels
- permutation p-value <= `0.05`
- effect replicated across `rbf`, `matern`, and `imq`

This would give the paper the missing real-world semantic validation layer.

## Reproducible Slice Builder

Added:

`scripts/build_source_balanced_corpus.py`

This helper creates a deterministic JSONL slice from existing corpora without
changing the pipeline. It normalizes `publisher` / `publication` / `source` /
`outlet` into a `source_proxy_label`, then writes a balanced corpus plus a
manifest.

Recommended first clean slice:

```powershell
python scripts\build_source_balanced_corpus.py `
  --input data\temporal_cleaned\cleaned_batch_0001_20241224-20241231.jsonl `
  --out outputs\source_proxy_validation\source_balanced_20241224_20241231_8x10\real_source_balanced.jsonl `
  --manifest-out outputs\source_proxy_validation\source_balanced_20241224_20241231_8x10\manifest.json `
  --n-sources 8 `
  --articles-per-source 10 `
  --selection spread
```

The resulting JSONL can be passed directly as a corpus path because
`run_experiments.py` already supports direct JSONL injection.

Generated slice:

`outputs/source_proxy_validation/source_balanced_20241224_20241231_8x10/real_source_balanced.jsonl`

Generated manifest:

`outputs/source_proxy_validation/source_balanced_20241224_20241231_8x10/manifest.json`

Manifest summary:

- input pool: `3,382` usable articles
- eligible repeated sources with at least `10` articles: `77`
- selected articles: `80`
- selected sources: `8`
- sources: `israelnationalnews.com`, `globalsecurity.org`,
  `middleeastmonitor.com`, `presstv.ir`, `ynetnews.com`, `menafn.com`,
  `aa.com.tr`, `thefrontierpost.com`

These generated JSONL outputs are intentionally under `outputs/` and should not
be treated as source-controlled paper text.
