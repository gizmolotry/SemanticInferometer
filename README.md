# SemanticInferometer

Curated source dump of the current Semantic Interferometer stack: the active pipeline, Dash/Monolith analysis layer, verification flow, controls, and thesis-facing runner scripts, without heavyweight generated artifacts.

## Included
- `core/`: active pipeline, fusion, geometry, walker, and support modules
- `analysis/`: Dash shell, Monolith renderer, waterfall tooling, verification helpers
- `config/`: framing query configuration
- `controls/`: control corpus generator and small manifests / seed corpora
- `run_experiments.py`: single-run entry point
- `run_full_experiment_suite.py`: matrix runner for real/control/synthetic suites
- `METHODS.md`, `RESULTS.md`, `THESIS_CHECKLIST.md`: thesis-facing workflow docs
- `test_*.py`: regression and contract tests for the current stack

## Intentionally Excluded
- generated run artifacts under `outputs/`
- historical material under `archive/`
- heavyweight local datasets and scratch experiment directories
- cached HTML, tensors, screenshots, and other build/runtime byproducts

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
- Generated artifacts should live outside this source repo or remain gitignored.
