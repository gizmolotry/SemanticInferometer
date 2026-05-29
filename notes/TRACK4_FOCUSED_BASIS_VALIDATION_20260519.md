# Track 4 Focused Basis Validation - 2026-05-19

Output directory:

`outputs\track4_focused_basis_validation\seed42_kernel_slice_20260519_033359`

Command shape:

```powershell
python scripts\run_track4_focused_basis_validation.py `
  --output-root outputs\track4_focused_basis_validation\seed42_kernel_slice_20260519_033359 `
  --corpora real control_shuffled control_random synthetic_microprobe `
  --kernels rbf matern imq `
  --seeds 42 420 4200 `
  --bases track2 logits_flat `
  --limit 60 `
  --skip-existing
```

## Matrix

- Corpora: `real`, `control_shuffled`, `control_random`, `synthetic_microprobe`.
- Kernels: `rbf`, `matern`, `imq`.
- Seeds: `42`, `420`, `4200`.
- Bases: `track2`, `logits_flat`.
- Cells: 36.
- Basis rows: 72.
- Failed cells: 0.

## Claim Boundary

```json
{
  "instrumentation_supported": true,
  "terrain_validity_supported": false,
  "basis_superiority_supported": false,
  "basis_recommendation_counts": {
    "logits_flat": 18,
    "track2": 18
  },
  "mean_basis_score_margin": 0.10134348884629366,
  "minimum_basis_score_margin": 0.05,
  "real_terrain_safe_rate": 0.6666666666666666,
  "control_terrain_safe_rate": 0.5833333333333334
}
```

## Basis Summary

| Basis | Rows | Mean Score | Terrain-Safe Rate | Mean Closed Loop | Mean Zones | Mean Work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `track2` | 36 | 1.0070 | 0.8056 | 0.9185 | 3.3611 | 154.6486 |
| `logits_flat` | 36 | 0.9352 | 0.4444 | 0.8870 | 3.4167 | 180.0292 |

## Interpretation

This full focused matrix supports the Track 4 instrumentation claim, not the terrain-validity claim.

The earlier property/theft-only hint that `logits_flat` might outperform `track2` did not survive the broader focused matrix. Recommendations split evenly: 18 cells recommend `track2`, 18 recommend `logits_flat`. Aggregated scores favor `track2`, and `track2` has a much higher terrain-safe rate.

This does not kill the low-dimensional logit-geometry hypothesis, but it downgrades it from "likely better" to "basis sensitivity only." If Track 4 is kept in the paper, it should remain an instrumentation/diagnostic layer unless a later corpus or terrain setup clears the terrain-validity boundary.

## Verification

- Focused Track 4 tests: `29 passed`.
- Full test suite after changes: passed.
