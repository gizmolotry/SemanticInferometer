# Visualizer Fidelity Roadmap

This note records a code-grounded visual roadmap for the MONOLITH/Dash stack.
The rule is evidence first: every visual effect should either be backed by a
fresh artifact field or explicitly mark itself as disabled/provisional.

## Current Stabilization Targets

- Keep ranked Track 4 walker paths visible by default, with caps to prevent path
  overdraw.
- Keep the article hitbox as the only heavy popup carrier; visual article traces
  should stay lightweight.
- Treat flat panels as disabled evidence, not decorative output. Empathy Gap now
  exposes active/disabled state through Plotly layout metadata.
- Keep camera presets scene-relative so screenshots do not fly away when terrain
  ranges or path spans are extreme.
- Cache contract verification by artifact fingerprint so click interactions do
  not repeatedly walk the same leaf.

## Near-Term Visual Upgrades

1. Evidence-first cockpit badges.
   - Show `verified`, `secondary_supported`, `unsafe`, `placeholder`, or
     `non_comparable` beside each panel.
   - Add source paths and field names to panel metadata.
   - Use `claim_matrix.json`, `verification_report.json`,
     `scientific_validation_summary.json`, and `baseline_meta.json`.

2. Observer displacement lens.
   - Render global points as a ghost layer and observer-conditioned points as the
     active layer.
   - Draw displacement tethers for the top changed articles.
   - Surface axis rotation, path flips, and null-observer equivalence directly in
     the side panel.
   - Required fields: baseline XYZ, observer XYZ, displacement rank, stable UID.

3. Track 4 traversal inspector.
   - Clicking an anchor should isolate its five walkers, show hot/cold class,
     closed loop state, work integral, terrain sequence, and failure mode.
   - Add path-touched zone summaries directly to the audit panel.
   - Required fields: `path_id`, `article_indices_touched`, `zone_sequence`,
     `work_by_step`, `closed_loop_reason`.

4. Screenshot presets.
   - Add named camera/layout presets for global evidence, Track 4 audit,
     observer lens, and diagnostics.
   - Persist selected preset in `MONOLITH.view_state.json`.

## Mid-Term Systems

1. Run comparison atlas.
   - Small multiples across kernels, corpora, Track 5 modes, and seeds.
   - Goal: see whether a claim survives across the focused proof bundle without
     opening runs one at a time.
   - Required fields: comparable group ID, run family ID, artifact freshness,
     primary baseline run.

2. Waterfall-to-MONOLITH continuity view.
   - Scrub from T0/T1/T1.5/T2/T3/T5 checkpoints into the final synthesis.
   - Goal: show when structure appears, degrades, or flips.
   - Required fields: per-article projection trace, checkpoint cluster labels,
     NMI/ARI delta by checkpoint.

3. Track 4 Markov overlay.
   - Overlay committor, MFPT, and reactive flux diagnostics beside walker paths.
   - Goal: separate decorative pathing from publishable transition-path evidence.
   - Required fields: committor-to-void, MFPT-to-bridge, MFPT-to-void, edge flux,
     dominant reactive paths.

## Ambitious Alternate Render Systems

1. WebGL/deck.gl graph layer.
   - Keep Dash for controls, but render points/paths through a dedicated WebGL
     layer with instanced segments and trace budgets.
   - Best for large `N`, heavy Track 4 path ensembles, and animation.

2. Three.js shader terrain.
   - Convert the terrain surface into a shader-driven mesh with stress/density
     textures, path ribbons, and hover picking.
   - Best for cinematic screenshots and observer-universe morphs.

3. Observable evidence notebook.
   - Export canonical JSON/NPZ summaries into an Observable-style interactive
     article, emphasizing reproducibility and reviewer inspection.
   - Best for publication supplements and external agent review.

4. PyVista/VTK offline renderer.
   - Generate high-resolution static plates with controlled lighting, path
     bundling, and consistent cameras.
   - Best for thesis figures where browser interaction is irrelevant.

## Guardrails

- Do not fabricate paths, density, observer displacement, or tooltip fields.
- Do not show uniform matrices as colorful evidence.
- Do not make Track 4 categorical verdicts in the visualizer; categorical
  interpretation belongs downstream.
- Prefer additive artifact fields over inference from trace names or colors.
- Preserve `MONOLITH.html` and Dash contracts while introducing alternate
  renderers as sidecars.
