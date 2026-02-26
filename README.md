# Belief Transformer V3

Thesis-oriented experimental workspace for studying framing-sensitive geometry in news representations.

## Current Status (as of 2026-02-26)
- Project outputs and archives have been reorganized:
  - Generated artifacts live under `outputs/`
  - Historical material lives under `archive/`
- Core pipeline code is active and under ongoing iteration.
- Thesis packaging docs are now present:
  - `METHODS.md`
  - `RESULTS.md`
  - `THESIS_CHECKLIST.md`

## Repo Structure
- `core/`: pipeline, feature extraction, fusion, geometry, diagnostics
- `run_experiments.py`: single-run / mode-focused experimental runner
- `run_full_experiment_suite.py`: full matrix runner (real + controls, synthetic mode, probes, optional analyses)
- `analysis/`: visualization and analysis scripts
- `outputs/`: generated run artifacts, logs, reports, monolith snapshots
- `archive/`: legacy snapshots and historical outputs
- `data/`, `config/`, `controls/`, `diag/`: datasets/config/supporting materials

## Thesis-Facing Workflow
1. Lock canonical settings in `METHODS.md`.
2. Execute canonical runs and record run IDs.
3. Populate `RESULTS.md` with only canonical-run evidence.
4. Complete publication checks in `THESIS_CHECKLIST.md`.

## Canonical Run Entry Points
- Full suite:
```powershell
python run_full_experiment_suite.py --mode enhanced --seeds 42 420 4200 --kernels rbf laplacian rq imq --channels logits cls --corpora real control_constant control_shuffled control_random --limit 500
```

- Synthetic validation:
```powershell
python run_full_experiment_suite.py --synthetic --mode enhanced --seeds 42 420 4200 --kernels rbf laplacian rq imq --channels cls --limit 60
```

## Notes
- Keep exploratory runs out of thesis claims unless promoted to canonical.
- Keep the root clean; write generated files to `outputs/`.
