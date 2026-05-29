# Track 4 Claim Boundary

Track 4 currently has three different evidence levels. They should not be mixed.

## 1. Instrumentation Evidence

Supported by the current code and tests:

- Track 4 can run over multiple explicit feature bases.
- The production default remains `track2`.
- Opt-in bases include `logits_flat`, `bot_norms`, `cls_stacked`, and `spectral_pc1`.
- The pipeline records requested/effective basis, proposal mode, adaptive TPT settings, and per-path basis/proposal metadata.
- The probe runner emits comparison summaries and a conservative `claim_boundary` field.

Safe language:

> Track 4 now supports explicit basis/proposal ablations and records traversal telemetry end-to-end.

## 2. Basis Sensitivity Evidence

Tentatively suggested, not proven:

- On the tiny property/theft full-pipeline probe, `logits_flat` scored `0.7636` and `track2` scored `0.7627`.
- The margin is effectively zero.
- The standalone microprobe basis comparison ranked `logits_flat` highest, but this is still the same narrow semantic environment.

Safe language:

> The property/theft microprobe suggests NLI logit geometry is worth testing as a Track 4 traversal basis at larger scale.

Unsafe language:

> `logits_flat` is better than `track2`.

Why: the full-pipeline margin is too small, and neither tiny full-pipeline run is terrain-safe.

## 3. Terrain Validity Evidence

Not supported by the tiny property/theft full-pipeline probe.

Both full-pipeline runs fail for the same reasons:

- Path-touched terrain coverage reaches only two terrain zones.
- Bridge/Void closed-loop gap is below the minimum semantic effect threshold.
- Bridge/Void work-integral gap is below the minimum semantic effect threshold.

Safe language:

> The current microprobe does not validate the Bridge/Swamp/Tightrope/Void terrain interpretation.

Unsafe language:

> Track 4 proves Bridge/Void terrain semantics.

## Working Hypothesis

The interesting technical hypothesis is not "logits already wins." It is:

> Track 4 may behave better over lower-dimensional NLI decision geometry because the walker is less exposed to high-dimensional sparse-graph artifacts than when traversing the 2048-dimensional Track 2 expansion.

This is a testable hypothesis, not a result.

## Next Validation Requirement

Before Track 4 can be promoted beyond diagnostics, run a larger focused basis comparison:

- Corpora: real, shuffled control, random control, synthetic.
- Kernels: `rbf`, `matern`, `imq`.
- Bases: `track2`, `logits_flat`, and one compact Track 1.5-derived basis.
- Seeds: at least `42`, `420`, `4200`.
- Required pass conditions:
  - At least three path-touched zones in canonical real runs.
  - Nontrivial Bridge/Void work or survival gap.
  - Real/control separation in Track 4 work, survival, committor, MFPT, or reactive flux.
  - Basis preference margin clears the probe threshold, not just a decimal dust-up.
