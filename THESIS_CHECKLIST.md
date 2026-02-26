# Thesis Readiness Checklist

## 1) Reproducibility
- [ ] Top-level docs updated (`README.md`, `METHODS.md`, `RESULTS.md`)
- [ ] Canonical command set frozen
- [ ] Canonical seeds frozen
- [ ] Canonical kernel set frozen
- [ ] Corpus definitions frozen
- [ ] Run IDs recorded with timestamps and commands

## 2) Data and Run Integrity
- [ ] Each canonical run has a manifest/metadata file
- [ ] No empty-input or partial-failure runs in canonical set
- [ ] Missing-file checks completed for all required artifacts
- [ ] Superseded runs clearly labeled in `RESULTS.md`

## 3) Evaluation Quality
- [ ] Real vs control comparisons are complete
- [ ] Synthetic validation includes all planned kernels/seeds
- [ ] Aggregated statistics (mean/std and dispersion) are computed
- [ ] Any ablation claims are backed by canonical artifacts

## 4) Figures and Tables
- [ ] Every thesis figure maps to a canonical run ID
- [ ] Every thesis table maps to canonical artifact paths
- [ ] Figure captions include kernel/seed/corpus metadata
- [ ] Final figure set uses a consistent visual style and axis semantics

## 5) Narrative Coherence
- [ ] Method claims match actual code path used for canonical runs
- [ ] Terminology is consistent (observer types, channels, kernels, controls)
- [ ] Limitations and failure modes are explicitly documented
- [ ] Exploratory results are separated from final claims

## 6) Final Freeze Before Writing Submission Draft
- [ ] No code changes between final canonical rerun and exported thesis artifacts
- [ ] Final canonical run directory list is locked
- [ ] Final metrics exported and copied into thesis tables
- [ ] Final results summary in `RESULTS.md` is complete
