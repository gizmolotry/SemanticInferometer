# Track 2 Component-Ablation Microtest - 2026-05-20

## Purpose

This diagnostic tests whether the Track 5 kernel assembly can preserve two separable synthetic factors:

- Track 2 carries topic/base-geometry signal.
- Track 1.5 carries stance/shear signal.
- Strict Hadamard fusion should preserve the joint topic-plus-stance intersection.

It is a lightweight component sanity check. It is not a real-corpus ablation and is not thesis-safe by itself.

## Artifact

`outputs/track2_component_ablation_microtest/current_20260520/track2_component_ablation_microtest.json`

Key contract fields:

- `diagnostic_type`: `track2_component_ablation_microtest`
- `claim_scope`: `microtest_only_not_foundation_ablation`
- `metric_basis`: `kernel_similarity`
- `foundation_audit_consumable`: `false`
- `safe_for_thesis_claim`: `false`
- `microtest_effect_detected`: `true`

## Result

The controlled fixture behaved as expected:

- `track2_only` strongly separates topic labels.
- `track15_only` strongly separates stance labels.
- `no_track2_null` loses topic separation and collapses to the Track 1.5-only behavior.
- `hadamard_track2_track15` recovers the joint topic-plus-stance labels.

## Interpretation

This reduces one narrow engineering risk: the current Track 5 kernel path is capable of combining Track 2 and Track 1.5 in the intended multiplicative way under a controlled synthetic fixture.

It does not close the open scientific risk in the Track 2 foundation audit. The direct Track 2 necessity/removal ablation for real/focused corpora is still untested.

## Verification

Passed:

```powershell
python -m pytest -q test_track2_component_ablation_microtest.py
python scripts\run_track2_component_ablation_microtest.py --output-dir outputs\track2_component_ablation_microtest\current_20260520
```
