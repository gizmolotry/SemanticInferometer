# Contributing

## Workflow
1. Create a branch from `main`.
2. Keep changes scoped and documented.
3. Run local quality checks before opening a PR.

## Local Checks
```powershell
python -m pip install -r requirements-dev.txt
python -m compileall core run_experiments.py run_full_experiment_suite.py
pytest -q
ruff check .
black --check .
```

## Commit Guidelines
- Use clear commit subjects.
- Include rationale and risk notes in commit body for non-trivial changes.
- Avoid mixing refactors and behavior changes in one commit.

## PR Expectations
- Link related issue(s) or thesis task(s).
- Include before/after behavior summary.
- Note reproducibility impact (seeds, kernels, corpora, manifests).
