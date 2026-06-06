# Track 2 Foundation Audit - 2026-05-20

## Question

Does the failure of `track4_basis=track2` mean the whole Semantic Interferometer is broken?

## Current Answer

No. The current evidence says:

`foundation_status = NOT_BUSTED_TRACK4_LOCAL_FAILURE`

This means Track 2 failed a narrower Track 4 walker-basis specificity test, but the current evidence does not show that the whole system foundation has failed.

## Evidence

Audit artifact:

`outputs/track2_foundation_audit/current_20260520/track2_foundation_audit.json`

Check summary:

- Integrated geometry signal: `SUPPORTED`
- Track 5 fusion signal: `SUPPORTED`
- Synthetic recoverability signal: `SUPPORTED`
- Track 4 using Track 2 as walker basis: `UNSUPPORTED`
- Direct Track 2 necessity/removal ablation: `UNTESTED`

Key numbers:

- Integrated/comprehensive variance separation mean abs log ratio: `0.2996`
- Synthetic recovery mean NMI: `0.7433`
- Track 2-as-walker real minus control terrain-safe rate: `-0.2778`
- Track 2-as-walker real minus control basis-score gap: `-0.1417`
- `logits_flat` as walker basis has a positive real-control terrain-safe gap: `0.4444`

## Interpretation

Track 2 should be treated as the spatial substrate / map, not as proof that Track 4 terrain semantics work by itself.

The current failure says:

> Track 2 alone is not a sufficient Track 4 walker basis for real/control terrain specificity.

It does not yet say:

> Track 2 is useless or the whole system is busted.

That stronger conclusion would require the integrated geometry, Track 5 fusion, synthetic recovery, and direct Track 2 ablations to fail.

## Remaining Risk

The direct Track 2 necessity test is still open. We do not yet have a clean matrix comparing:

- `Track2-only`
- `Track1.5-only`
- `Track2 + Track1.5`
- `Full fusion with Track3 density`
- `No-Track2 replacement/null geometry`

This is the next serious ablation required before making or rejecting a strong Track 2 foundation claim.

Auxiliary microtest:

- `outputs/track2_component_ablation_microtest/current_20260520/track2_component_ablation_microtest.json`
- `notes/TRACK2_COMPONENT_ABLATION_MICROTEST_20260520.md`

This microtest confirms that the Track 5 kernel path can preserve separable Track 2 topic geometry and Track 1.5 stance/shear geometry in a controlled fixture. It remains `foundation_audit_consumable=false` and does not close the real-corpus direct Track 2 ablation gap above.

## Verification

Passed:

```powershell
python -m pytest -q test_track2_foundation_audit.py
python -m pytest -q test_track2_foundation_audit.py test_track4_terrain_semantics_diagnostics.py test_track4_diagnostic_packet.py test_track4_traversal_validity.py test_track4_pipeline_basis_probe.py test_track4_focused_basis_validation.py test_track5_ablation_matrix.py test_track5_assembly_modes.py test_thesis_claim_matrix.py
```

Final broad slice result: `60 passed`.
