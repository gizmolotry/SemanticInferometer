$ErrorActionPreference = "Stop"

$env:MONOLITH_RENDER_ALL_MODES = "1"
$env:MONOLITH_CANONICAL_Z_TEMPLATE = "D:/belief-transformer/V3/monolith_cockpit_restored_exact.html"

python analysis/MONOLITH_VIZ.py experiments_20260217_162351/synthetic/rbf_seed42 `
  -o monolith_canonical_locked_now.html --strict --mode synthesis
