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

| Run ID | Date | Purpose | Command Class | Status | Notes |
|---|---|---|---|---|---|
| experiments_20260225_225508 | 2026-02-25 | Synthetic baseline | Synthetic | Success | Single `rbf_seed42`; expand to full matrix |

## Thesis Tables To Populate

### Table A: Real vs Controls by Kernel/Seed
- Metrics: primary and secondary thesis metrics (declare once fixed)
- Granularity: per corpus x kernel x seed, plus aggregated means/std

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
