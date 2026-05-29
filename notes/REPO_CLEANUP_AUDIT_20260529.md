# Repository Cleanup Audit - 2026-05-29

## Scope

This audit reviewed the dirty worktree on branch `codex/dash-observer-atlas-migration-20260529` after the observer-recenter, observer-atlas, Track 4 action-graph, and focused-proof work. The goal was to separate real prior work from scratch debris, check that new modules are wired into the DAG/ledger/test surface, and close places where the system could appear claim-safe while using missing, placeholder, stale, or fallback artifacts.

## Current Inventory

- Tracked dirty entries: 85.
- Untracked entries: 84.
- Staged `.pytest_local_tmp` removals: 43 tracked temp artifacts are being de-tracked; `.pytest_local_tmp` is now ignored.
- Untracked core modules: 6, all currently wired into tests and/or runners.
- Untracked scripts: 20, all have direct test coverage or runtime references.
- Untracked notes/paper files: 26, all are evidence interpretation, claim-boundary, or publication scaffolding rather than runtime code.

## Keep And Integrate

- `core/ablation_dag.py`: used by `core/master_ablation.py`, focused proof scripts, observer recenter scripts, and `test_master_ablation_dag.py`.
- `core/observer_local_recompute.py`: used by recenter experiments and observer-recenter tests.
- `core/observer_manifold.py`, `core/observer_atlas_bundle.py`, `core/observer_slice_transport.py`: used by Dash precompute, observer atlas tests, and observer slice transport artifacts.
- `core/track4_action_graph.py`: used by Track 4 action graph scripts and tests.
- `scripts/run_track4_observer_state_matrix.py`, `scripts/run_track4_action_graph.py`, `scripts/summarize_track4_action_graph_runs.py`, `scripts/track4_replay_status.py`: Track 4 observer-state/action proof surface.
- `scripts/run_observer_recenter_meaning_probe.py`, `scripts/run_observer_recenter_robustness_suite.py`: observer-conditioned recentering proof surface.
- `scripts/build_capstone_review_packet.py`, `scripts/compile_track4_basis_comparison.py`, `scripts/focused_proof_status.py`, `scripts/run_focused_proof_bundle.py`: publication/review packet surface.
- `scripts/run_track15_gradient_probe.py`, `scripts/run_track15_gradient_action_probe.py`, `scripts/property_theft_deberta_probe.py`, `scripts/property_theft_microprobe.py`: exploratory/ablation probes with tests or downstream references. These should remain clearly labeled as probes, not canonical pipeline replacement.

## Guardrails Tightened In This Pass

- Required consumer JSON artifacts now fail the contract if present but malformed. Previously malformed required files could be swallowed as empty payloads.
- `validation.json` with failed/error status or `UNAVAILABLE`/`FAILED` trust level now blocks claim validity even if an NMI number can be inferred.
- Validation repair now preserves failed/unavailable status instead of promoting inferred NMI to measured success.
- Observer relativity cache backfill now requires every selected/all observer sidecar, not merely one existing sidecar.
- Corrupt or missing observer payloads no longer silently fall back to the global payload when emitting observer relativity artifacts.
- Relativity DAG outputs now declare all expected observer sidecars, not only `observer_0.pt`.
- Track 4 replay folders no longer hard-code stale suffix `20260521`; the runner now records a run suffix and summary/status tools are suffix-agnostic.
- Track 4 basis comparison output now writes both canonical and paper packet aliases.
- Observer atlas fallback article pairs are now explicitly labeled as `fallback_adjacent_row_pairs_no_edge_action_ledger`.
- MONOLITH visualization contract failures now fail fast by default. Legacy bypass requires `MONOLITH_ALLOW_CONTRACT_BYPASS=1`.
- Observer precompute relativity emission now reports `invalid` when the generated observer relativity scientific summary is invalid or thesis-unsafe.
- Consumer optional inventory now includes observer relativity, observer recentering, observer atlas, Track 4, Track 5, and leaf inventory artifacts.
- Focused observer precompute now hard-fails if observer-local Track 2/1.5/3 artifacts are not materialized at `relativity_cache/obs_<idx>/features.npy`.
- Focused MONOLITH renders can be forced to require observer-local recompute with `MONOLITH_REQUIRE_LOCAL_OBSERVER_RECOMPUTE=1`; precompute sets this during focused renders.
- Observer atlas construction now rejects rendered observer view states that declare `recenter_mode` as global/sidecar fallback unless an explicit accepted sidecar mode is recorded.

## Remaining Design Boundaries

- Dash base claim gating intentionally treats observer-atlas/recenter artifacts as optional panel readiness, not as global run validity. That is acceptable only if the UI keeps atlas/recenter readiness visible and does not present observer-slice claims as globally verified.
- Real-corpus runs remain geometry/control/provenance evidence unless labels are independently defended. Source/proxy labels are exploratory.
- Property/theft and gradient probes are useful engineering probes, not replacement evidence for full-corpus validation.
- Some legacy visualizer/regression files remain dirty because they are prior visualization work, not because this pass rewrote them from scratch.

## Verification Run

- `python -m py_compile` passed for patched core/ledger/Dash/precompute/script files.
- `python -m py_compile` passed for all 95 dirty/untracked Python files in the worktree.
- Focused guardrail suite passed: 33 tests.
- Broader recent-clean suite passed: 149 tests.
- Final patched surface passed: 202 tests.
- Focused fallback P1 patch surface passed: 18 tests.

Commands used:

```powershell
pytest -q test_suite_bundle_hooks.py test_dash_contract_schema.py test_dash_contract_gating_diagnostics.py test_track4_observer_state_matrix.py test_track4_replay_status.py test_track4_basis_comparison.py test_capstone_review_packet.py test_observer_atlas_bundle.py test_precompute_observer_artifacts.py test_monolith_viz_runtime_regressions.py
python -m py_compile analysis\MONOLITH_VIZ.py analysis\regression\precompute_observer_artifacts.py analysis\isolated_dash_prototype.py analysis\verification\contract.py run_full_experiment_suite.py run_experiments.py core\master_ablation.py core\observer_atlas_bundle.py scripts\run_track4_observer_state_matrix.py scripts\track4_replay_status.py scripts\summarize_track4_action_graph_runs.py scripts\compile_track4_basis_comparison.py scripts\build_capstone_review_packet.py
```

## Cleanup Recommendation

1. Commit the tested core/ledger/Dash/script/test changes together as the observer-atlas and claim-safety migration.
2. Commit notes/paper files either in the same research branch or a separate evidence-docs commit.
3. Keep exploratory probes, but label them as probes in docs and avoid routing them into claim-safe outputs unless their suites pass.
4. Do not delete prior dirty files blindly; the current audit found most of them are either tested runtime work or evidence scaffolding.
5. Archive only files with no tests, no imports, no runtime references, and no paper/evidence role after a second targeted pass.
