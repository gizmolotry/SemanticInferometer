# Track 4 Soft Terrain Diagnostic - 2026-05-25

## What Changed

The Track 4 terrain diagnostics now include a continuous, path-touched terrain
probe in addition to the older hard Bridge / Swamp / Tightrope / Void bins.

The soft probe consumes existing run artifacts only:

- `cyclic_paths.npz`
- `track3_density_rho.npy`
- `d_spectral.npy`

It recomputes normalized density/stress fields and assigns every touched path
node fuzzy terrain membership:

- `Bridge = density * (1 - stress)`
- `Swamp = density * stress`
- `Tightrope = (1 - density) * (1 - stress)`
- `Void = (1 - density) * stress`

This is intentionally closer to the continuous mechanics that Track 4 actually
uses than the hard quadrant labels used for screenshots and anchor provenance.

## Fresh Matched500 Result

Command:

```powershell
python scripts\track4_terrain_semantics_diagnostics.py `
  --validation-summary outputs\track4_focused_basis_validation\matched500_repair3_20260522\track4_focused_basis_validation_summary.json `
  --output-dir outputs\track4_terrain_semantics_diagnostics\matched500_soft_terrain_20260525
```

Primary output:

`outputs/track4_terrain_semantics_diagnostics/matched500_soft_terrain_20260525/track4_terrain_semantics_diagnostics_summary.json`

Claim boundary:

```json
{
  "terrain_construct_distinctness": true,
  "walker_sensitivity_detected": true,
  "native_work_decomposition_available": false,
  "terrain_specificity_supported": true,
  "soft_terrain_work_coupling_supported": true,
  "pooled_soft_terrain_specificity_supported": false,
  "matched_soft_terrain_specificity_supported": true,
  "soft_terrain_specificity_supported": true,
  "targeted_event_pair_status": "NO_DATA",
  "targeted_event_pair_available": false,
  "targeted_event_pair_claim_ready": false
}
```

## Interpretation

This is a useful but bounded improvement.

The positive result is that continuous high-stress / low-density terrain mass
does couple to walker work in the real corpus:

- Real path count: `135`
- Real `corr_soft_barrier_mass_work`: `0.296`
- Global `corr_soft_barrier_mass_work`: `0.213`
- Soft work coupling support: `true`

The limiting result is that the same kind of coupling is also present in
controls when everything is pooled:

- Control random `corr_soft_barrier_mass_work`: `0.354`
- Control shuffled `corr_soft_barrier_mass_work`: `0.178`
- Mean control barrier/work correlation: `0.266`
- Real minus control barrier/work correlation: `0.030`
- Pooled soft terrain specificity support: `false`

Pooling was too blunt, though. A matched-cell calibration by
`basis | kernel | seed` gives a stronger and fairer result against stochastic
controls:

- Usable matched cells: `9`
- Supporting cells: `7`
- Matched cell pass rate: `0.778`
- Mean real-minus-control excess correlation: `0.399`
- Median real-minus-control excess correlation: `0.634`
- Matched soft terrain specificity support: `true`

Cell-level pattern:

- `imq`: passed all three seeds, excess correlations `0.634`, `0.846`, `0.906`
- `matern`: passed seeds `42` and `4200`, failed seed `420`
- `rbf`: passed seeds `42` and `420`, failed seed `4200`

So the field mechanics are not dead. Track 4 is measuring a real traversal
property of the constructed graph, and the matched null calibration suggests the
effect is stronger in real data than stochastic controls in most cells. The
remaining caution is kernel/seed fragility: this is now promising Track 4
evidence, but not yet a standalone semantic ontology.

## Current Claim Boundary

Thesis-safe:

- Track 4 instrumentation exists.
- Walkers respond to terrain/action parameters.
- Hard terrain summaries on matched500 show real-vs-control lift at the summary
  level.
- Continuous terrain mass predicts work in real runs.
- Matched real-vs-stochastic-control soft terrain specificity passes in `7/9`
  kernel/seed cells.

Not thesis-safe yet:

- Bridge / Swamp / Tightrope / Void as a hard semantic ontology.
- Pooled soft terrain specificity; controls also show generic barrier/work
  coupling.
- Track 4 as proof of perspectival traversability in the capstone sense.
- Native work decomposition, because exported components are still incomplete.

## Engineering Implication

The next repair should not be another label rename. The failure mode is that
Track 4's terrain mechanics are too generic: controls can also produce
stress/barrier-work coupling. A publishable Track 4 needs either:

1. a matched same-event / different-frame corpus where semantic traversal is
   externally constrained, or
2. a null-calibrated action/terrain field where excess terrain work is computed
   against shuffled controls at the path or edge level.

Until then, Track 4 should be framed as mechanical telemetry and exploratory
phenomenology, not as final semantic verdict machinery.

## Verification

Targeted tests:

```powershell
python -m pytest test_track4_terrain_semantics_diagnostics.py test_track4_traversal_validity.py test_track4_action_graph.py test_track4_focused_basis_validation.py test_track4_pipeline_basis_probe.py test_track15_gradient_action_probe.py -q
```

Result: `54 passed`.

Full suite:

```powershell
python -m pytest -q
```

Result: passed.
