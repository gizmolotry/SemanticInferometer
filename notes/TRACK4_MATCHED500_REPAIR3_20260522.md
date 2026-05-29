# Track 4 matched-500 repair3 evidence, 2026-05-22

## Scope

This note records the matched 500-article Track 4 repair run using existing artifacts, not a new pipeline design.

Observer artifact root:
`outputs/track4_focused_basis_validation/matched500_repair3_20260522`

Canonical replay root:
`outputs/track4_action_graph/observer_state_repair3_matched500_20260522`

Auxiliary non-overwriting replay roots:

- `outputs/track4_action_graph/observer_state_repair3_matched500_20260522_parallel_null_track2`
- `outputs/track4_action_graph/observer_state_repair3_matched500_20260522_parallel_null_integrated`
- `outputs/track4_action_graph/observer_state_repair3_matched500_20260522_parallel_richer_track2`

The canonical root is archival and all-branch/all-basis. The auxiliary roots were run in separate output directories because the matrix runner does not skip existing action summaries and same-root parallelism would overwrite shared summaries.

## Completed observer grid

The focused pipeline completed all 36 observer cells:

- Corpora: `real`, `control_random`, `control_shuffled`, `control_constant`
- Kernels: `rbf`, `matern`, `imq`
- Seeds: `42`, `420`, `4200`
- Basis: `track2`
- Limit: `500`

The focused validator reported:

- `row_count=36`
- `cell_count=36`
- `failed_cell_count=0`

## Summarizer correction

The null-calibrated branch originally failed thesis safety because the observer-state summary gate still evaluated raw `mean_hysteresis_penalty`.

That was incorrect for `null_calibrated_hysteresis`, whose action term is `mean_calibrated_hysteresis_penalty`, equivalent here to positive excess hysteresis over the shuffled/null observer baseline.

Patch applied:

- `scripts/summarize_track4_action_graph_runs.py` now loads the null/excess/positive/calibrated hysteresis fields.
- The shuffled hysteresis gate is branch-aware.
- `null_calibrated_hysteresis` uses `mean_calibrated_hysteresis_penalty`.
- A zero destroyed baseline with positive full-real calibrated hysteresis is treated as separated rather than missing.

Verification:

- `python -m pytest -q test_track4_action_graph.py test_track4_observer_state_matrix.py test_track4_observer_state_ablation_summary.py`
- `28 passed`

Later broader targeted verification:

- `python -m pytest -q test_track4_action_graph.py test_track4_observer_state_matrix.py test_track4_observer_state_ablation_summary.py test_thesis_claim_matrix.py`
- `39 passed`

## Repaired branch results

### Null-calibrated hysteresis, Track 2

Root:
`outputs/track4_action_graph/observer_state_repair3_matched500_20260522_parallel_null_track2`

Summary:

- `safe_for_thesis_claim=true`
- `point_estimate=3.636586829817`
- `hysteresis_gate_metric=mean_calibrated_hysteresis_penalty`
- `failure_reasons=[]`
- Kernel ratios: `imq=3.9808`, `matern=3.5146`, `rbf=3.4964`
- Seed ratios: `42=3.8248`, `420=3.8892`, `4200=3.2035`
- Basis ratio: `track2=3.6366`

Stochastic-only control comparison, excluding constant:

- Real mean action: `24.3626`
- Random/shuffled mean action: `10.0489`
- Stochastic-only ratio: `2.4244`

Interpretation:

This is the cleanest current Track 4 repair. It preserves the existing article-graph walker but measures hysteresis as excess over a shuffled/null observer trajectory. It is thesis-safe in the current exploratory Track 4 evidence contract.

### Null-calibrated hysteresis, integrated basis

Root:
`outputs/track4_action_graph/observer_state_repair3_matched500_20260522_parallel_null_integrated`

Summary:

- `safe_for_thesis_claim=true`
- `point_estimate=3.521032661379`
- `hysteresis_gate_metric=mean_calibrated_hysteresis_penalty`
- `failure_reasons=[]`
- Kernel ratios: `imq=4.1997`, `matern=2.3810`, `rbf=4.3362`
- Seed ratios: `42=3.4342`, `420=3.6029`, `4200=3.5260`
- Basis ratio: `integrated=3.5210`

Stochastic-only control comparison, excluding constant:

- Real mean action: `41.5852`
- Random/shuffled mean action: `17.7113`
- Stochastic-only ratio: `2.3479`

Interpretation:

This passes the formal gate, but it is less clean than Track 2 because the Matern cell is weaker and the stochastic-only ratio is slightly lower. It still supports the claim that null-calibrated observer hysteresis separates real discourse from stochastic controls.

### Richer walker state, Track 2

Root:
`outputs/track4_action_graph/observer_state_repair3_matched500_20260522_parallel_richer_track2`

Summary:

- `safe_for_thesis_claim=false`
- `point_estimate=5.260147783185`
- `failure_reasons=[kernel_robustness_failed, seed_robustness_failed, basis_seed_robustness_failed, kernel_basis_robustness_failed, kernel_seed_basis_robustness_failed]`
- Kernel ratios: `imq=5.9449`, `matern=5.2797`, `rbf=4.6746`
- Seed ratios: `42=5.2429`, `420=5.5678`, `4200=4.9693`

Stochastic-only control comparison, excluding constant:

- Real mean action: `44.7566`
- Random/shuffled mean action: `12.7628`
- Stochastic-only ratio: `3.5068`

Virtual-state scale:

- Mean virtual node count across summaries: about `11897`
- Max virtual node count: `14835`

Interpretation:

This branch is powerful but not clean. It amplifies real/control separation, but some kernel/seed cells fail the raw shuffled-hysteresis robustness gate. It is better treated as an engineering lead for latent rhetorical intermediates, not the current thesis-safe Track 4 claim.

## Baseline comparison

Canonical baseline full-cell ratios completed before the canonical run moved into repaired branches:

- Baseline Track 2 stochastic-only ratio: `2.6059`
- Baseline integrated stochastic-only ratio: `2.5127`
- Baseline Track 2 all-control ratio: `3.9089`
- Baseline integrated all-control ratio: `3.7681`

The baseline separates too, but the repair adds a cleaner interpretation: not merely high action through the geometry, but excess hysteretic observer-state action relative to a null/shuffled observer trajectory.

## Current scientific interpretation

Track 4 should be framed as a stochastic metric-graph traversal diagnostic over observer-conditioned manifolds, not as a fully continuous MCMC sampler.

The new positive result is:

> Real articles require more null-calibrated observer-state action than matched stochastic controls across seeds and kernels. This remains true on Track 2 and integrated bases, with the Track 2 basis currently cleaner.

The strongest thesis-safe Track 4 branch is:

`null_calibrated_hysteresis + track2`

The strongest engineering research branch is:

`richer_walker_state + track2`

The latter is not thesis-safe yet, but it is the most interesting path toward virtual rhetorical transition states.

## Caveats

- Summary point estimates include `control_constant`; stochastic-only ratios should also be reported because constant controls are degenerate topology controls, not stochastic semantic nulls.
- The canonical all-branch run was launched before the branch-aware summarizer patch, so its final summaries should be regenerated after completion using the patched summarizer.
- Same-root replay is not safe to parallelize because existing action summaries are overwritten and summary files are shared.
- Future same-root replays can now use `--skip-existing` on `scripts\run_track4_observer_state_matrix.py` to resume without recomputing completed `track4_action_summary.json` cells.

## Evidence selection hardening

The thesis evidence loader now treats Track 4 observer-state action summaries as claim evidence, not simply as "latest file wins" artifacts.

- Added an explicit CLI pin: `--track4-observer-state-summary`.
- Default selection policy is now `thesis_safe_then_summary_all_then_mtime`.
- Explicit selection policy is `explicit_summary_path_then_thesis_safe_then_summary_all_then_mtime`.
- Safe scoped findings are surfaced across sibling roots, so `track2` and `integrated` null-calibrated evidence can coexist.
- Stale same-scope findings are deduplicated in favor of the selected source, then `summary_all`, then shallower/current paths.
- Scoped findings are limited to the selected action-graph family. For the focused bundle, this means only the `observer_state_repair3_matched500_20260522` sibling roots are surfaced, avoiding older 20260521 positives.

Focused paper bundle:

`outputs/thesis_validation/focused_paper_evidence_20260522`

Inputs:

- Suite: `experiments_20260506_172550` (`hadamard_strict`)
- Suite: `experiments_20260506_183343` (`riemannian_strict`)
- Synthetic: `experiments_20260506_192553`
- Track 4 action evidence: `observer_state_repair3_matched500_20260522_parallel_null_track2`

Focused claim status:

- Paper profile: `publication_ready=true`
- Core supported claims: `procrustes_control_separation`, `stochastic_control_variance_separation`, `observer_relativity`, `track5_ablation_coverage`, `verification_provenance`
- Track 4 observer-state action claim: `safe_for_thesis_claim=true`, point estimate `3.636586829817`
- Unsafe/non-core claims remain unsafe: `control_destruction`, `stochastic_control_variance_compression`, `terrain_incremental_signal`, `track4_traversal_validity`, `track4_work_barrier_signal`, `canonical_freeze`

Compact review packet:

`outputs/review_packets/focused_paper_evidence_20260522_MAX10`

This packet contains exactly 10 files, including a compact `review_digest.json`, for sharing with other agents or collaborators.

Regeneration command:

```powershell
python scripts\build_focused_paper_packet.py `
  --evidence-dir outputs\thesis_validation\focused_paper_evidence_20260522 `
  --out-dir outputs\review_packets\focused_paper_evidence_20260522_MAX10 `
  --track4-basis-summary outputs\track4_focused_basis_validation\matched500_repair3_20260522\track4_focused_basis_validation_summary.json `
  --max-files 10
```

The digest explicitly marks the packet scope as `focused_core_claim_profile`; it should not be read as a claim that every historical/full-canonical thesis claim is solved.

Focused Markdown brief:

`paper/focused_paper_claim_brief_20260522.md`

Regeneration command:

```powershell
python scripts\render_focused_paper_brief.py `
  --packet-dir outputs\review_packets\focused_paper_evidence_20260522_MAX10 `
  --out paper\focused_paper_claim_brief_20260522.md
```

This brief is meant for fast human review. The JSON packet remains the source of truth.

## 2026-05-22 hardening checkpoint

Live replay root:

`outputs/track4_action_graph/observer_state_repair3_matched500_20260522`

Current replay status at checkpoint:

- `completed_count`: 278 / 378
- `missing_count`: 100
- Active unfinished branch: `richer_walker_state`
- Error log: empty
- Process state: Python replay still running under the parent PowerShell launcher

Status command:

```powershell
python scripts\track4_replay_status.py `
  outputs\track4_action_graph\observer_state_repair3_matched500_20260522 `
  --out outputs\track4_action_graph\observer_state_repair3_matched500_20260522\track4_replay_status.json
```

New hardening added during the replay:

- `scripts\track4_replay_status.py` now prefers `observer_state_matrix_inventory.json` when present, supports custom run suffixes, groups custom branches correctly, and emits an `inventory_warning` when a dry-run inventory is paired with real completed summaries.
- `scripts\track4_replay_status.py --require-complete` now returns nonzero while expected replay cells are missing, so downstream proof scripts can gate on completed matrices.
- `scripts\track4_replay_status.py` also reports the freshest completed summary and rough completion throughput for long-running replay monitoring.
- `scripts\focused_proof_status.py --track4-replay-root <root>` now folds Track 4 replay progress into the existing focused proof status console.
- `scripts\focused_proof_status.py` now recognizes Track 4 validation/replay/action-graph scripts as active proof processes.
- `scripts\run_track4_observer_state_matrix.py` now writes dry-run manifests to `observer_state_matrix_inventory.dry_run.json` so dry runs cannot overwrite the real replay inventory again.
- `analysis\verification\thesis_evidence.py` records unreadable Track 4 observer-state summary candidates in `candidate_inventory` instead of silently dropping them.
- `scripts\build_thesis_evidence.py` now builds and validates focused evidence before writing the output bundle.
- `scripts\build_focused_paper_packet.py` fails by default when required packet files are missing and writes `review_digest.json` atomically.
- `scripts\build_focused_paper_packet.py` now records SHA-256 fingerprints and byte counts for packet source files.
- `scripts\verify_review_packet.py` verifies those packet fingerprints. Store its output beside the packet, not inside the MAX10 folder, to preserve the 10-file cap.
- `scripts\render_focused_paper_brief.py` now surfaces blocked core claims and publication warnings.

Verification:

```powershell
python -m pytest -q
```

Full suite passed after the checkpoint patches.

## 2026-05-24 completed matrix result

The matched500 repair3 matrix completed:

- `completed_count`: 378 / 378
- `missing_count`: 0
- Error log: empty
- Completed branches: `baseline_raw_action`, `null_calibrated_hysteresis`, `richer_walker_state`
- Completed bases: `track2`, `integrated`
- Completed kernels: `rbf`, `matern`, `imq`
- Completed seeds: `42`, `420`, `4200`

Final branch ranking:

1. `null_calibrated_hysteresis`: thesis-safe, point estimate `3.562855511014`
2. `richer_walker_state`: largest action ratio, point estimate `5.244286388508`, but not thesis-safe because fine-grained basis/kernel/seed robustness fails
3. `baseline_raw_action`: positive action separation, point estimate `3.819029513437`, but not thesis-safe because fine-grained robustness fails

Interpretation:

- Track 4 action/work separation is real in this matrix.
- The safe paper-facing Track 4 observer-state claim should use `null_calibrated_hysteresis`, not the richer virtual-state branch.
- `richer_walker_state` remains an engineering research lead: stronger effect size, weaker robustness.
- The old terrain traversal/work claims remain unsafe separately; do not conflate them with the observer-state action claim.

Focused evidence was refreshed to pin:

`outputs/track4_action_graph/observer_state_repair3_matched500_20260522/summary_null_calibrated_hysteresis/track4_observer_state_ablation_summary.json`

The focused packet was regenerated and verified:

`outputs/review_packets/focused_paper_evidence_20260522_MAX10`

Verification output:

`outputs/review_packets/focused_paper_evidence_20260522_MAX10_packet_verification.json`
