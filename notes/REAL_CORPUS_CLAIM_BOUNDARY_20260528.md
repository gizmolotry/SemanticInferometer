# Real-Corpus Claim Boundary - 2026-05-28

This note defines paper-safe wording for real-corpus claims. It does not add new evidence or change the canonical protocol in `METHODS.md` / `RESULTS.md`.

## Scope

- Controlled synthetic runs have planted labels and can support label-recovery claims when the relevant validation gates pass.
- Real-corpus runs can support geometry, control-separation, observer-conditioning, and provenance claims.
- Real-corpus runs cannot support real-world ideological label accuracy unless an independent label target is attached and justified.
- Source labels, publisher labels, outlet ratings, heuristic frame labels, and other proxy labels are exploratory unless independently validated for the specific claim.

## Claim Tiers

### Controlled Synthetic Label Claims

Safe when backed by the synthetic validation artifacts:

- The system recovers planted synthetic perspective structure above control or baseline conditions.
- Observer-local recomputation improves recovery of known synthetic labels relative to artifact-view or translation-only baselines.
- NMI, ARI, silhouette, permutation, or related label metrics may be described as label-recovery evidence because the label basis is controlled.

Safe language:

> On controlled synthetic corpora with planted perspective labels, the pipeline recovers known label structure and distinguishes observer-local recomputation from visual recentering baselines.

Boundary:

> This validates the mechanism under known labels; it does not by itself prove that real articles have been assigned correct ideological labels.

### Real-Corpus Geometry And Control Claims

Safe when backed by verified real/control artifacts:

- Real articles produce measurable geometry under the stated kernel, seed, channel, and corpus protocol.
- Real and control corpora differ on registered geometric, variance, Procrustes, observer-conditioning, or traversal-telemetry metrics.
- The result is provenance-backed when every number maps to a run ID, command class, artifact path, and verification packet.

Safe language:

> On the real corpus, the method detects reproducible geometric and observer-conditioned structure that separates from matched controls under the reported protocol.

Boundary:

> These are geometry/control/provenance claims, not claims that the system has recovered true ideological classes for real-world articles.

### Source Or Proxy Label Claims

Treat as exploratory unless independently justified:

- Source-level labels may describe publisher metadata, not article-level semantic truth.
- Proxy labels can be used for stratification, stress tests, sanity checks, or hypothesis generation.
- Agreement with source/proxy labels should not be reported as ideological accuracy unless the proxy has an explicit validation argument.

Safe language:

> Source/proxy-label analyses are reported as exploratory alignment checks against external metadata, not as ground-truth validation.

Unsafe language:

> The real-corpus clusters recover the true ideology of articles.

Promotion requirement:

> Promote a source/proxy-label result only if the paper states the label source, unit of analysis, known limitations, and independent reason it is a valid target for the specific claim.

## Paper Wording

Use:

> Synthetic experiments test whether the mechanism can recover known planted perspective labels. Real-corpus experiments test whether geometry, controls, observer conditioning, and artifact provenance behave nontrivially on natural text. Source and proxy labels, where used, are exploratory metadata unless independently justified as a claim target.

Avoid:

> Because synthetic labels are recovered, the real-corpus labels are correct.

Corrected claim:

> Synthetic recovery supports the controlled-label mechanism; real-corpus results support geometry and control-separation claims under verified provenance. Real-world ideological labeling remains exploratory unless independently validated.

## Reporting Rule

For each real-corpus sentence in the paper, ask which claim type it makes:

- If it says "labels," "ideology," "bias class," or "frame class," require independent label justification.
- If it says "geometry," "separation from controls," "observer conditioning," "stability," or "provenance," tie it to the relevant run artifact and metric.
- If it uses source/proxy metadata, mark it exploratory unless the paper explicitly defends the proxy as a valid target.
