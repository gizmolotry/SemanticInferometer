# Outputs Folder Contract

This folder contains generated artifacts only.

## Layout
- `experiments/runs/`: timestamped run folders (for example `experiments_YYYYMMDD_HHMMSS`)
- `logs/`: runtime logs and diagnostic logs
- `reports/`: text/json/csv run summaries and reports
- `monolith/html/`: monolith HTML previews/snapshots
- `monolith/reports/`: monolith text reports and status notes

## Rules
- Do not place source code in `outputs/`.
- New generated files should be written inside one of the folders above.
- Prefer writing each run into a timestamped subfolder under `experiments/runs/`.
- Keep root repository clean by avoiding new output files at top-level.
