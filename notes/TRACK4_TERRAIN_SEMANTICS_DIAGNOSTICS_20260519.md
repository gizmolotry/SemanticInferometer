# Track 4 Terrain Semantics Diagnostics - 2026-05-19

## Artifact

Primary diagnostic bundle:

`outputs/track4_focused_basis_validation/seed42_kernel_slice_20260519_033359/terrain_semantics_diagnostics/`

Generated from:

`outputs/track4_focused_basis_validation/seed42_kernel_slice_20260519_033359/track4_focused_basis_validation_summary.json`

Method-sweep side evidence:

`outputs/microprobes/property_theft/track4_entropy_compact_grid_20260515_093357/track4_grid_method_sweep_summary.json`

## What The Diagnostics Now Test

The harness decomposes the failed broad terrain claim into six smaller claims:

1. Terrain construct validity: do Bridge, Swamp, Tightrope, and Void occupy distinct traversal telemetry regimes?
2. Walker sensitivity: do proposal parameters/modes materially change Track 4 behavior?
3. Work decomposition readiness: do exports support component-level work claims?
4. Real-vs-control specificity: does real data outperform matched controls on terrain telemetry?
5. Terrain contrast search: is Bridge/Void actually the strongest contrast, or are other zone pairs stronger?
6. Targeted event-pair evidence: do we have same-event, different-frame tests yet?

## Current Result

Track 4 now has evidence for mechanical construct distinctness, but not semantic specificity.

Claim boundary from `track4_terrain_semantics_diagnostics_summary.json`:

```json
{
  "terrain_construct_distinctness": true,
  "walker_sensitivity_detected": true,
  "native_work_decomposition_available": false,
  "terrain_specificity_supported": false,
  "targeted_event_pair_status": "NO_DATA",
  "targeted_event_pair_available": false,
  "targeted_event_pair_claim_ready": false
}
```

Important numbers:

- Terrain zones are mechanically distinct in the focused matrix: all four zones appear, global work range is `143.24`, and real-corpus matched work range is `736.35`.
- Controls also show construct distinctness: control support rate is `1.0`, so construct distinctness alone is not a semantic proof.
- Real-vs-control terrain specificity fails overall: real safe rate is `0.667`, controls are `0.583`, and mean basis score is lower for real by `-0.047`.
- The failure is not uniform: `logits_flat`, `rbf`, and `imq` pass local specificity slices; `track2` and `matern` fail. This is basis/kernel sensitivity, not a publishable global claim yet.
- Work decomposition is still proxy-only. Current exports do not separate edge-distance, density, shear, retreat, and proposal-cost components.
- The strongest terrain contrast is `Bridge_vs_Swamp`, not `Bridge_vs_Void`, in the current matrix. That suggests the original Bridge/Void narrative may not be the best empirical contrast.

## Code Hardening Added

- Added `scripts/track4_terrain_semantics_diagnostics.py` as an artifact-consuming diagnostic harness.
- Added `test_track4_terrain_semantics_diagnostics.py` with synthetic tests for all six diagnostics.
- Hardened `analysis/verification/scientific_summaries.py` so `path_anchor_idx` prefers exact anchor article IDs before falling back to legacy ordinal interpretation.
- Added a regression test proving nonzero anchor article IDs are not misresolved as ordinal positions.
- Added claim-boundary protection so uninterpreted targeted-event files do not count as targeted event-pair evidence.

## Verification

Passed:

```powershell
python -m pytest -q test_track4_terrain_semantics_diagnostics.py test_track4_traversal_validity.py
python -m pytest -q test_track4_pipeline_basis_probe.py test_track4_focused_basis_validation.py test_track4_method_sweep.py test_track4_basis_comparison.py test_terrain_incremental_signal.py
python -m pytest -q test_track4_terrain_semantics_diagnostics.py test_track4_traversal_validity.py test_track4_pipeline_basis_probe.py test_track4_focused_basis_validation.py test_track4_method_sweep.py test_track4_basis_comparison.py test_terrain_incremental_signal.py test_thesis_claim_matrix.py
```

Final broad slice result: `53 passed`.

## Next Empirical Move

Do not claim Track 4 terrain semantics as publishable yet. The next strongest experiment is a targeted same-event pair corpus:

- Hold event/topic constant.
- Vary source/frame.
- Test whether Track 4 work and survival separate same-event/different-frame pairs better than Track 2 distance alone.
- Include controls and kernels, but avoid another full run until artifacts are compacted; the focused matrix already occupies about `2.19 GB` and the disk has about `2.4 GB` free.

## Targeted Event-Pair Contract

The diagnostics harness now accepts targeted event-pair JSON/CSV files through `--targeted-event-results`. A result is only claim-ready when it validates the stricter schema; merely providing a file is not enough.

Minimal derived pair fields:

```csv
pair_id,event_id,relation,left_article_id,right_article_id,left_source,right_source,left_work,right_work,work_gap,track2_distance
e1_a,e_land_001,same_event_same_frame,17,18,reuters,reuters,10.0,10.8,,0.22
e1_b,e_land_001,same_event_different_frame,17,24,reuters,aljazeera,,,42.0,0.24
```

Accepted `relation` values:

- `same_event_same_frame`
- `same_event_different_frame`

Claim-ready gates:

- At least `2` mixed-frame events.
- At least `3` same-event same-frame pairs.
- At least `3` same-event different-frame pairs.
- At least `3` cross-source different-frame pairs.
- Different-frame mean work gap exceeds same-frame mean work gap by `> 0.10`.
- Cliff's delta for different-frame gaps over same-frame gaps is `> 0.0`.
- Per-event direction pass rate is at least `0.67`.

This creates the next clean test of Track 4: if the same event stays close under Track 2 but becomes expensive under Track 4 when the frame changes, then Track 4 starts looking like a perspectival traversal assay rather than a decorative route simulator.

## Portable Packet

Built a compact external-review packet with exactly `10` files:

`outputs/review_packets/track4_diagnostic_packet_20260519/`

Zip archive:

`outputs/review_packets/track4_diagnostic_packet_20260519.zip`

The packet intentionally omits heavy binaries such as `observer_*.pt`, `cyclic_paths.npz`, `walker_paths.npz`, and NLI caches. It is a claim-boundary packet, not a replay bundle.
