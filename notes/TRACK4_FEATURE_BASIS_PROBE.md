# Track 4 Feature-Basis Probe

Track 4 now supports an explicit traversal feature basis in the full pipeline:

- `track2`: production default, Track 2 fused geometry.
- `logits_flat`: flattened NLI logits, useful for direct decision-geometry traversal probes.
- `bot_norms`: per-observer CLS norm magnitudes.
- `cls_stacked`: flattened per-observer CLS stack.
- `spectral_pc1`: scalar Track 1.5 spectral stress coordinate.

Run a focused probe with:

```powershell
python scripts\run_track4_pipeline_basis_probe.py `
  --corpus outputs\microprobes\property_theft\deberta_20260515\property_theft_corpus.jsonl `
  --output-root outputs\microprobes\property_theft\pipeline_basis_probe_YYYYMMDD_HHMMSS `
  --nli-cache-path outputs\microprobes\property_theft\deberta_20260515\pipeline_nli_cache.pt `
  --limit 11 `
  --bases track2 logits_flat `
  --track4-adaptive-tpt-connectivity `
  --walker-temperature 0.75 `
  --walker-k-neighbors 8
```

Current tiny property/theft smoke result:

- Summary: `outputs\microprobes\property_theft\pipeline_basis_probe_20260515_100314\track4_pipeline_basis_probe_summary.json`
- Recommended diagnostic basis: `logits_flat`, score `0.7636`.
- `track2` score: `0.7627`.
- Both tiny-corpus runs are still thesis-unsafe for terrain validity because path-touched terrain coverage is only two zones and Bridge/Void effect sizes are below threshold.

Interpretation: this is instrumentation evidence, not a publication claim. It shows the basis-switching contract works end-to-end and gives a cheap way to choose the next expensive Track 4 run.
