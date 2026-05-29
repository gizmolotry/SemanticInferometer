# Track 4 Least-Action Graph - 2026-05-21

## Purpose

The legacy Track 4 cyclic walker treated articles as stepping stones in a stochastic graph. The new action-graph path treats articles as landmarks in a metric field and asks a more mechanical question:

`What is the minimum action required to traverse from one perspectival landmark to another?`

This is implemented additively. It does not replace `physarum_walk.py`.

## Mechanics

The action integral decomposes into:

- `metric`: Track 2 / density-warped metric distance from the existing metric graph.
- `shear_penalty`: directional Track 1.5 mismatch along the edge.
- `density_penalty`: extra cost for low-rho / void-like regions.
- `stress_penalty`: scalar stress barrier from Track 1.5 magnitude.
- `curvature_penalty`: semantic momentum penalty for sharp turns.
- `observer_transport_penalty`: symmetric movement across the eight V-observer simplex.
- `hysteresis_penalty`: directed KL cost, `KL(next_observer_state || current_observer_state)`, so article-to-article traversal can carry path directionality rather than only undirected distance.

The solver uses Dijkstra over `(previous_node, current_node)` states so turn cost is path-dependent. Neighborhood support is undirected kNN: if either endpoint selects the other as a neighbor, the edge can carry action.

Existing observer artifacts do not always persist the full Track 3 annealing `weight_trajectory`, so the replay runner derives the article-level V-observer simplex from the strongest available existing artifact in this order:

- explicit Track 3/observer simplex weights if present
- Track 1.5 spectral probe magnitudes when available as `[N, 8]`
- per-bot CLS magnitudes from `cls_per_bot [N, 8, H]`

## Artifacts

Implementation:

- `core/track4_action_graph.py`
- `scripts/run_track4_action_graph.py`
- `scripts/run_track4_observer_state_matrix.py`
- `scripts/summarize_track4_action_graph_runs.py`

Tests:

- `test_track4_action_graph.py`
- `test_track4_action_graph_probe_summary.py`
- `test_track4_pipeline_basis_probe.py`
- `test_track4_observer_state_matrix.py`

Small replay artifacts:

- `outputs/track4_action_graph/real_rbf_seed42_track2_20260521/`
- `outputs/track4_action_graph/control_random_rbf_seed42_track2_20260521/`
- `outputs/track4_action_graph/control_shuffled_rbf_seed42_track2_20260521/`
- `outputs/track4_action_graph/probe_summary_track2_matched_20260521/track4_action_graph_probe_summary.json`
- `outputs/track4_action_graph/observer_simplex_track2_seed42_20260521/summary/track4_action_graph_probe_summary.json`
- `outputs/track4_action_graph/observer_state_matrix_3seed_full_20260521/summary_all/track4_observer_state_ablation_summary.json`

## Initial Small-Run Result

Matched Track2 replay on existing 60-article RBF/seed42 artifacts:

- Real mean action: `29.5833`
- Matched control mean action: `6.9335`
- Real/control action ratio: `4.2667`
- Reached paths: `3/3` for real, `3/3` for both matched controls

This is `engineering_probe_not_thesis_claim`. It is promising because it suggests deterministic least-action mechanics expose a real/control traversal-cost gap that the stochastic walker failed to stabilize, but it is not yet a publishable Track 4 claim.

## Observer-Simplex Replay

Matched Track2 replay on the same existing 60-article seed42 artifacts across `rbf`, `matern`, and `imq`, with V-observer simplex transport and hysteresis enabled:

- Rows: `9` (`real`, `control_random`, `control_shuffled` across three kernels)
- Real mean action: `54.8519`
- Matched control mean action: `10.4050`
- Real/control action ratio: `5.2717`
- Real mean observer-transport penalty: `4.4510`
- Control mean observer-transport penalty: `0.6933`
- Real mean hysteresis penalty: `8.9993`
- Control mean hysteresis penalty: `1.5021`
- Observer-simplex contract supported: `true` for all rows

This is still `engineering_probe_not_thesis_claim`. The important change is conceptual: the Track 4 graph is no longer only walking between article coordinates. It now carries a local state over the eight V-observers and charges work for observer-state transport and directed hysteresis.

## Three-Seed Observer-State Matrix

The replay was promoted from a hand-written one-off to a repeatable matrix runner over existing observer artifacts. It used the complete Track 4 focused artifact grid:

- Corpus set: `real`, `control_random`, `control_shuffled`
- Kernels: `rbf`, `matern`, `imq`
- Seeds: `42`, `420`, `4200`
- Traversal bases: `track2`, `integrated`
- Full replay rows plus destroyed-state baselines: `108`

Overall matrix result:

- Safe for exploratory Track 4 claim: `true`
- Real mean action: `49.6507`
- Control mean action: `14.4535`
- Real/control action ratio: `3.4352`
- Action gap: `35.1972`
- Observer-transport ratio: `4.6949`
- Hysteresis ratio: `3.9802`

By basis:

- `track2`: action ratio `5.2717`, gap `44.4469`, claim pass `true`
- `integrated`: action ratio `2.4024`, gap `25.9475`, claim pass `true`

The refreshed thesis evidence artifact is:

- `outputs/thesis_validation/observer_state_action_3seed_20260521/track4_observer_state_action_summary.json`

This remains an exploratory Track 4 claim, not a core publication gate. But it is now seed-spanning, kernel-spanning, ablated, and backed by a repeatable replay script.

## Verification

Passed:

```powershell
python -m pytest -q test_track4_action_graph.py test_track4_action_graph_probe_summary.py
python -m pytest -q test_track4_action_graph.py test_track4_action_graph_probe_summary.py test_track4_pipeline_basis_probe.py
python -m pytest -q test_track4_observer_state_matrix.py test_track4_observer_state_ablation_summary.py test_thesis_claim_matrix.py
python -m pytest -q test_track4_action_graph.py test_track4_action_graph_probe_summary.py test_track4_pipeline_basis_probe.py test_track4_traversal_validity.py test_track4_terrain_semantics_diagnostics.py test_track4_focused_basis_validation.py test_track4_basis_comparison.py test_track4_method_sweep.py
python -m pytest -q test_track4_action_graph.py test_track4_action_graph_probe_summary.py test_track4_traversal_validity.py test_track4_pipeline_basis_probe.py test_track4_focused_basis_validation.py test_track4_terrain_semantics_diagnostics.py test_track5_assembly_modes.py test_track2_foundation_audit.py test_thesis_claim_matrix.py
python -m pytest -q
```

Full suite collection after this patch: `381` tests.
