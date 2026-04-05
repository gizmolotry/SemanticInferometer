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
- Kernels: `rbf laplacian rq imq`
- Seeds: `42 420 4200`
- Channels: `logits cls`
- Corpora: `real control_constant control_shuffled control_random`
- Article limit per corpus: `500` (or thesis-declared fixed value if changed)

## Canonical Commands

### A) Main corpus matrix
```powershell
python run_full_experiment_suite.py --mode enhanced --seeds 42 420 4200 --kernels rbf laplacian rq imq --channels logits cls --corpora real control_constant control_shuffled control_random --limit 500
```

### B) Synthetic validation (ground-truth check)
```powershell
python run_full_experiment_suite.py --synthetic --mode enhanced --seeds 42 420 4200 --kernels rbf laplacian rq imq --channels cls --limit 60
```

## Recorded Artifacts Per Canonical Run
- Run directory under `outputs/experiments/runs/experiments_YYYYMMDD_HHMMSS`
- `experiment_manifest.json` (or equivalent run metadata file)
- Metric outputs used for thesis tables/figures
- Probe results if enabled
- Any derived figures used in thesis text

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
