

ASSIGNMENT COVER SHEET
ANDREW, J. C.
Student Number: 22184813
Supervisor: Henry Silke
Date submitted: 3/4/2026
Word count including footnotes, excluding bibliography: 9,589
Module : JM4018 - INDIVIDUAL JOURNALISM PROJECT 2025/6 SEM2
I have read and accept the University's policy on plagiarism. I confirm that this is entirely my own
work and that it has not been submitted for assessment as part of any other programme.
Signed: Andrew Clair
Date: 1/12/2026
Generative AI Acknowledgement
Declaration I acknowledge the use of Gemini CLI, Codex CLI, and ChatGPT to generate

the equation LaTeX, generation of repository code, as well as serving as a sounding board
for the machine learning and mathematical operations as the system developed.
# Aligned Perspectival Kernel Projections and Bias Reification

## Lit Review and Introduction
This paper outlines a multistage, zero-shot NLI architectural approach towards large scale
data classification which relies on a Riemannian1 metric as a means of formalising semantic
geometry. In the literature review we will interrogate the current academic discourse on NLI
classification, the Overton Window (Mackinac Center for Public Policy, 2019), and the bedrock
of civility and pluralism which current news aggregation and anti bias platforms like Ground
News target as their main selling point. As the V-Dem Institute at the University of Gothenburg
identifies (2023) a quarter of countries as having increased polarization, half of which are
established democracies, we examine our methods of classification, especially as classification
serves as the cold start fuel driving all of the AI tools we rely upon. This paper suggests that,
through a lens critical of both pure relativism and unity focused agreement, we can create an
interperspectival system powered by complex vector embeddings to not only plot out a general
1 The method used in our system to generate its final map outputs which allows for localised curvature in
modeling semantic disagreement

shape of bias but to analyse its parts as physical properties. In our methodology we will outline
the different tracks by which we use a vertical approach to identify different physical dimensions
and properties of the semantic space, and how we can simulate particle movement across the
manifold to make salient inferences about the nature of the perspectival boundaries which define
discourse.
Ground News2 is a private news aggregation platform with the specific purpose of
reducing 'bias', understood as a negatively coded deviation from neutrality. Their platform is
meant to organise information in a way that contextualizes content in relation to their perceived
'middle' using left to right scales and applying it to groups of articles on various political
subjects. The business' advertisements emphasize this; the product is positioned as a tool that
improves contemporary knowledge and critical thinking by imposing politically pluralist
information streams. In practice, this pluralism is applied through source-based profiling and
left-right labeling which relies on various third parties. Ground News, what we describe as a
horizontal measurement system, employs an architecture that functions through extrinsic
build-up, in the sense that at each stage of processing in their information system they introduce
new information. For example, when a new article is processed in Ground News' aggregation,
they employ prior information (a preconceived understanding of bias surrounding the publisher,
which itself is an identifier determined completely outside the information stream) to classify the
article. On the other hand, using endogenous methods of classification hinge on features derived
from the input.
This type of system does not derive properties from the signal in question but rather
appends 'explanatory' exogenous metadata that exists topologically on a completely separate
2 Website link: https://ground.news/

vector of information. This is a horizontal expansion of the shape of bias that is contingent on
metadata rather than content. Because of this, Ground News does not actually perform any direct
semantic analysis itself on behalf of its readers. What it does is create a source profile,
connecting data points not through any presupposed inherent values but lines drawn by arbitrary
classification of the system on a global level. They derive ratings from third party outlets like
AllSides, Ad Fontes, and Media Bias Fact Check. Ground News is relevant to this literature
review not because it is uniquely flawed, but the opposite; it signals a broader methodological
tendency. This tendency concerns how systems that claim to reveal bias often rely on prior
classificatory frameworks that are external to the content they organize, raising questions about
whether such frameworks describe political reality or actively participate in it, particularly when
the system formalises a 'middle' perspective (Bowker and Star, 1999).
The journalistic discourse on the subject of objectivity and its relationship to semantic
classification over the past century incites direct philosophical implications of such an approach;
Thomas Kuhn in the 1960s comments about the theoretical limits of observation, arguing that it
is impossible to look at any given data without a framework and so, essentially, any critical
derivation of data is non-neutral (Kuhn, 1962). Nelson Goodman radicalises this, later asserting a
sort of mereological nihilism in that humans make sets of the universe as symbols, culture, art,
and more fundamentally measurement itself (Goodman, 1978). The stars themselves might exist
on a hard matter basis, but their supposed constellations certainly do not. For a given information
system, measurement frameworks don't merely reveal structure but actually constitute the objects
they describe. One 'remedy' for this is interperspectival frameworks which move the analytical
objective away from platonically correct features and towards probable distributions over

interpretations. This should not be confused with labeling by aggregations, which harmonizes
individual labels and suppresses minority perspectives.
We will now consider the academic discourse that exists on exploring disagreement as a
signal in and of itself rather than treating it as noise, particularly in relation to natural language
inference and labeling tasks. Frenda et al. (2024) explicitly frame perspectivist NLP as an
alternative to collapsing annotator identities and disagreement into one canonical target, arguing
instead that interpretive diversity can itself be methodologically informative for subjective
phenomena. Xu and Jurgens (2026) extend this in their work by treating disagreement as a
widespread feature of NLP tasks that require its own separate modeling and evaluation
framework.
At the same time, work on inherent disagreement in Natural Language Inference (eg.,
Pavlick and Kwiatkowski, 2019) shows that even when annotators are well trained and
instructions are tightly controlled, semantic judgements remain unstable across contexts. Other
research on human annotated dataset collection indicates that while more detailed guidelines can
raise agreement scores by reducing room biased interpretive assumptions this does not
necessarily increase annotation quality (Aroyo and Welty, 2015). The specification strategy
amounts to "perfuming the agreement scores," as workers find themselves forced into using
assumptions they might not believe in, which would inversely introduce more of the examiner's
bias. This instability has been interpreted to mean that linguistic meaning is underdetermined and
can yield systematically distinct but still internally coherent interpretations on the same inputs.
This is supported by the general field of semantic analysis, which over the past decade has grown
to reject a number of previously standardized intuitions that inevitably bleed into the field of
NLP.

The CrowdTruth method, for instance, constitutes the foundation for the system described
in this paper which functions on many of the same principles developed and discovered
regarding semantic analysis through groups of workers. Aroyo and Welty's (2015) claim,
however, that "outside mathematics" truth is entirely relative and is most closely related to
agreement and consensus, inhibits analysis. The system described in this paper would seem to
share this perspective, but attempts to begin where CrowdTruth ends. Natural Language
Inference, however, provides a modern viable route from which we might escape this
dichotomy, measuring semantic relations as artifacts of the relationship between the input,
structured hypotheses, and the weights of the model. While Aroyo and Welty formalized a
crucial refutation of 'golden middle' statistical reductions, they ultimately maintain a strict
epistemological dichotomy between objective physics and subjective sociology. CrowdTruth also
documents profiling different subtypes of disagreement statistically based on their perceived
usefulness. This has an important consequence; disagreement is preserved but only after it has
gone through some heuristic sorting method. Ambiguity inherent to the signal, such as perceived
poor task design, ontological mismatch of semantics, or low quality annotation, are all treated as
discrete sources of variance and consequently are classified as 'noise'. The disagreement upheld
by CrowdTruth is not merely projected but a curated statistical signal. In this way the
CrowdTruth framework is strongest as a critique of forced consensus, but can fall short of
modeling agreement once its existence is assumed.
By the same token, the traditional procedural journalistic ethics as a whole will warn
repeatedly that perspectives cannot slide into an academic "all opinions are equal" epistemology,
illustrated in the IFJ's Global Charter of Ethics for Journalists which encodes a "respect for the
facts" (International Federation of Journalists, 2019). Practically, however, we argue news

aggregation implementations like Ground News are fundamentally based in moralistic premises
which contradict the underlying principles of liberal ethics, as described by Alison Reiheld. She
points out that an "exceptionless civility" may not lead to pluralism but a "simplistic and
functional moral relativism". She brings up Isaiah Berlin's view (1998), which posits the
differences between humans is part of what it is to be human, to which "toleration and liberal
consequences follow". Lucian Pye's (1999) claim that "Pluralistic democracy, especially when it
involves rival moral concepts, requires an exceptionally high level of civility" also backs this
perspective, which, when analyzed under Reiheld's (2013) dichotomous Substantive Purposes of
Civility, reveals a critical tension. Projects like Pye's and Berlin's seem to typically focus on the
first type of objective of civility: "it makes society work". The second goal, however,
"embodying respect for persons", becomes increasingly complicated when formulations like
Berlin's, which sees empathy as a major facet of civility, offer respite for value systems which
impose a threat on the form of life of another moral agent. As Reiheld (2013) says, : "Ah, how
convenient it would be if the fact that someone's beliefs negated tolerable forms of life also
negated the need to respect it as a human pursuit." This reluctance to impose 'values' beyond the
absolute unitarian imperative is what Reiheld calls "Exceptionless Civility".
By enforcing a left-right spectrum it makes intuitive sense that Ground News might be
increasing reader awareness in some way which reduces polarisation. But beyond the obvious
critiques of the left-right frame (an introduced axial bias, methodological opacity), the kind of
pluralism which 'bias centered' news aggregation fosters a managed, civility oriented discourse
rather than a genuinely agnostic one. This is because the practical functioning of democracy
hinges on conflict. In other words, conflict is not necessarily corrosive to systems but a core part
of democracy that must have a controlled method (Swanstrom and Weissmann, 2005). When

consensus oriented frameworks neutralize this antagonism as a goal in and of itself, this can
weaken the conflictive process of democracy (Mouffe, 2000).
When working in high dimensional spaces, which complex semantics necessarily
requires, we must cast a 'light' on the thing itself only to observe the shadow. While one might
do this more literally with Principal Component Analysis, essentially projecting a high
dimensional object into its highest variance shadows, Ground News also does this in a sense by
using metadata to heuristically compress political semantics into one dimension. The issue only
arises from what happens afterward; the reification of one observer as a platonic vision which
describes the entire dimension globally. On a broader metaphysical level, philosopher Ludwig
Wittgenstein argued that the standpoint from which the world is described cannot itself be
straightforwardly placed inside that descriptive understanding as simply another component. As
he famously states, "the subject does not belong to the world but it is a limit of the world."
In reality, groups of participants in the Blind Bias Surveys by which AllSides provides
Ground News with their data all projected their own light and cast their own shadow, revealing
the approximate shape of bias over time. While Ground News borrows from the CrowdTruth
intuition, it stops just before the core principles that actually make it work in the first place. It
inherits the CrowdTruth discovery that multiple perspectives are better than one, but collapses
them at the level of upstream rating and metadata and further collapses them with a single one
dimensional approximation. When combining Wittgenstein's (1922) point about the limits of
representation. Bowker and Star's (1999) arguments on classificatory systems and their role in
social organizing, and Kuhn's (1962) total rejection of the neutrality of classification
frameworks, there exists a nuanced but concatenatable inference; middle positions are not neutral
points but the effects of a particular representational frame onto an object.

This presents a crucial mismatch of units of analysis for systems like Ground News. In
the following methodology we will discuss how our system attempts to account for this
ontological discrepancy as well as the perspectival opacity and concatenation that hide the values
which constitute discourse.
## Methodology
A vertical pipeline, like the one proposed in this paper, is endogenous as of its second
layer, in the sense that the 'first layer' is the raw text processed through a pre-trained natural
language inference model (DeBERTa-v2-xlarge-mnli) in which we introduce millions of tiny,
unavoidable pre-justified premises. Initially, this architecture was designed to help
mathematically formalize the previously mentioned relativism in an attempt to demonstrate the
myth of objective neutrality. By concentrating different perspectives with adversarial framing,
however, it became clear that the system ultimately functions as an attempt at measuring the
ideal of an Overton Window (what is perceived as the 'middle ground' or the range of politically
acceptable ideas) through the mapping of structural invariants and relativistic variations. By
treating bias as a physical shape with physical properties, we can perform various
transformations that help us make certain inferences about the structure of semantics in the
context of language models.
To understand how the system would calculate the Overton Window, we have to first
define what a structural invariant is and is not, and why their existence anchors semantic
meaning geometrically. Survival of structural invariants was not inevitable. High dimensional
embeddings are notoriously brittle and yielded no meaningful invariants from the noisy control

floors until we started treating bias as a geometric and observer conditioned warp. Initially the
architecture relied on averaging raw language model embeddings together to find a consensus
and then computing the result. Instead we now map complete independent manifolds from each
distinct bot's perspective. To serve as a control for agreement we establish a "Global Mean
Manifold" which acts as a collective stress map. The manifold is designed to preserve
disagreement by explicitly constructing the shape of bias based on the traces of perspectival
divergence. In this way, our rank-one bias approximation is not meant to act as an averaging field
but a concatenating one where disagreement is the main information vector. Articles that occupy
similar positions on a two dimensional baseline representation which reflect a coarser or more
topically informed semantic alignment are preserved as such, but become delineated through
sharper perspectival forces in the third dimension.
By generating a series of complex manifolds and comparing article inputs with Procrustes
we can make inferences about specific semantic structures by comparing their alignment and
residual divergence. To expand on the use of Procrustes in the architecture, imagine taking two
article feature maps of the same discourse geometry, but with one map, the center is a fiercely
pro-Israel standpoint, while the other from a pro-Palestine standpoint. In statistics Procrustes
analysis, which is named after a blacksmith in Greek mythology who violently physically altered
his guests to fit an iron bed, we compare the similarity of these two shapes by attempting to align
them as much as possible without warping the maps themselves and calculating the average
distance remainder.
Then we dynamically recalculate the literal shape of the manifold between concepts
based on the individual observer's standpoints before mixing occurs. Where an observer's
worldview creates ideological friction, the mathematical space between those ideas is

proportionally stretched apart. Only after these uniquely warped maps are generated do we mix
them together. By forcing the source to physically deform on the specific biases first, any nodes
that resist this relativistic shear are bound by some shared semantic gravity, and can be compared
against shuffled, random, and constant controls.
The proposed system takes one frozen NLI bot and gives them 8 different adversarial or
contrastive hypotheses and prompts them to analyze any given number of articles. In this paper
we refer to these different conditioned perspectives as 'V-Observers'.
In our implementation we used both real and synthetic articles relating to the Palestinian
genocide. First, our real articles were scraped using a mix of curated RSS feeds and GDELT in
parallel, filtering them for Palestinian related keywords, deduplicated by URL, extracted full
article text, and then wrote the results to a JSON ledger. For our synthetic articles we attempted
to generate a balanced corpus by shifting through 8 ideological clusters using a
Qwen2.5-Coder-14B bot. Using this bot we prompted the local model to generate a number of
articles constrained by perspectives, publication bias, and tone to the best of its ability.
The mechanism of adversarial framing allows our approach to dialectically process each
article under paired framing hypotheses rather than the raw average embeddings. Furthermore,
the system as a whole is based on the idea that bias is not a static property but the measure of
perspectival elasticity. For every article we query a series of hypothesis pairs designed to stress
test the narrative structure of reporting on Palestine rather than for a particular ideological
position.
Queries:
# Pair 1: Israeli actions framing

Israeli_defense_vs_aggression:
A: "This article portrays Israeli military actions as legitimate self-defense against
terrorism"
B: "This article portrays Israeli military actions as disproportionate aggression against
civilians"
description: "How Israeli military operations are framed"
# Pair 2: Palestinian framing
palestinian_victim_vs_aggressor:
A: "This article portrays Palestinians as innocent victims of occupation and violence"
B: "This article portrays Palestinians as violent aggressors and terrorists"
description: "How Palestinians are characterized"
# Pair 3: Proportionality framing
proportionate_vs_disproportionate:
A: "This article describes Israeli responses as proportionate and measured"
B: "This article describes Israeli responses as excessive collective punishment"
description: "Assessment of force proportionality"
# Pair 4: Resistance vs terrorism framing
resistance_vs_terrorism:
A: "This article frames Palestinian armed groups as legitimate resistance movements"
B: "This article frames Palestinian armed groups as terrorist organizations"

description: "Characterization of Palestinian militant groups"
# Pair 5: Occupation framing
occupation_vs_security:
A: "This article frames the situation as Israeli occupation of Palestinian territory"
B: "This article frames the situation as Israeli security measures against threats"
description: "Overall conflict framework"
# Pair 6: Civilian casualties framing
civilian_targeting_vs_collateral:
A: "This article suggests Israeli forces deliberately target civilians"
B: "This article suggests civilian casualties are unintended collateral damage"
description: "Intentionality of civilian harm"
# Pair 7: Historical context framing
colonialism_vs_indigenous:
A: "This article frames Israel as a colonial settler state"
B: "This article frames Israel as the indigenous homeland of the Jewish people"
description: "Historical legitimacy framing"
# Pair 8: International law framing
violations_vs_compliance:
A: "This article emphasizes Israeli violations of international law"

B: "This article emphasizes Israeli compliance with international law and right to
self-defense"
description: "Legal framework assessment"
By forcing the text to pass through these specific dialectical gates we can attempt to
measure friction generated when a narrative is stretched between two opposing coordinated
frameworks of reality. In the process of building this system, moreover, we came to acknowledge
a number of constraints to ensure meaningful results. Firstly, traditional Bi-Encoders, or simply a
method to independently encode two inputs into two separate vectors, are geometrically shallow.
This is unlike DeBERTa-v2 (a multi-genre natural language inference bot), which processes the
article and hypothesis simultaneously in the same attention layer. DeBERTa-v2 is also a
decoupled model, separating content to content meaning and content to relative meaning which
massively sharpens the model's syntactic clarity. Because the model no longer has to rely on
mixing words and their positions into the same embedding tokens are spread further apart in the
attention space. This makes models like DeBERTA's final geometric representation of the
model's attention actually more anisotropic, but it's a worthy tradeoff: the model is much better
at understanding the semantic logic of a text ('The IDF attacked Hamas' versus 'Hamas attacked
the IDF'). Standard classifiers, such as basic BERT-for-Sequence-Classification, also would
generally require training on this particular taxonomy, which would break our system into a
Ground News adjacent horizontal pipeline. For example, a standard BERT has a preconceived
spatial geometry of 'war' and 'peace' which it would impose onto the pipeline as a means of
classification. NLI models ask a completely different question, which allows them to be more
generalisable: does X logically follow from Yinfinity As a result, NLI avoids the need for training

entirely with zero-shot neutrality. We do not ever need to teach what 'bias' actually represents.
Second, LLM's like GPT-4 would be even worse for this specific use case. While they are more
generally capable, they are contaminated by RLHF (Reinforcement Learning from Human
Feedback) and are stochastic in the sense that they predict the next token, not a specific answer
to a question. This is only exacerbated by gradient drift, a problem unique to LLMs. Because
they optimize for the probability of the next token rather than resolving the logical entailment of
an entire sequence, the gradient vector of any given LLM will be highly unstable and therefore
unusable as a dynamic semantic terrain we can mathematically interrogate. You can't build a
map if the landscape is actively shifting under your feet. Any given input should output
something deterministic rather than merely probable. Finally, a vertical pipeline like the one
we've described is not designed to 'complete the pattern,' but serve as a critic. This is
exemplified in the way the two model types are trained. DeBERTa is optimized using a
Discriminative Objective (when you ask an NLI bot to predict a hidden token in context), only
learning to categorize relationships. One is built to minimize disagreement and mimic while the
other is trained to maximize logical separation while standing outside of the data it's measuring.
While any structure derived from this subtract necessarily reflects the ideological
regularities present in training, the system attempts to hold them to constraint to measure
differential interaction with the source. By treating the model weights not as truth but rather a
semantic substrate on which we can perform perspectival analysis we can employ the internal
attention dynamics of a collection of NLI modules without having to rely on individual invariant
assumptions of DeBERTa. In this way, the DeBERTa weights serve as the base level of prior
distribution from which all semantic derivations might be measured. In other words, the goal is

to measure the interaction between the model and source text mechanistically by determining the
magnitude of struggle between the source texts and its assumptions through variance lenses.
The models internal embeddings, especially in decoupled NLI models like the ones
employed are not built to approximate such a coordinate system due to anisotropy of the model's
learning mechanisms. Bi-Encoders like DeBERTa suffer from representation degeneration,
organising their self attention into a narrow cone due to Zipf's law3 (Gabaix, Xavier. 1999) and
training 'laziness'. A coordinate system used to map language in a geometrically meaningful
structure requires orthogonal axes which can be analyzed in isolation. Anisotropy means those
very axes have concentrated. Even if we fixed this engineering problem with various bandaids,
flattening the cone until distribution was properly flat, the system will still face a fundamental
tension between the linear algebraic forms which exist in attention locally and the wider
non-euclidean space.
Imagine a large room with a number of floating balloons. Rather than being spread out
across the room, imagine all the balloons coalesce into one band; this is our full discourse. If we
try to just blow a big gust so all the balloons spread apart more, the core shape of anisotropy, the
balloon's directional concentration, does not change. Though the model thinks in linear algebra, a
collection of straight lines, the emergent meanings of that network live in curved space. This is
why even after 'blowing up' the band, the underlying system remains attached to these clustered
directions in space.
Standard embedding models are optimized within Euclidean vector spaces even if the
semantic relations they model exhibit non-Euclidean behavior. For any two points A and B, there
exists a geodesic (a straight line) between them, and the entire space between is filled with
3 A given word's usage frequency decreases approximately inversely with its rank, producing a
heavy tailed distribution

potential legible meaning. Consider the topological relationship between the words "Property"
and "Theft." From an anarchist semantic framework, there exists a direct traversable path
connecting these two ideas. Since the ideology defines property as theft, the two concepts reside
in the same semantic valley. A model or human can slide from the concept of ownership to the
concept of exploitation smoothly without having to alter their point of view. The shortest path is
a straight line through text logic. But if you take a couple steps to the right, into a liberal
framework, the relative positions of the space change. You haven't just altered your thoughts on
the objects in vision themselves, but also the terrain relative to your position. Even if the words
maintain the same absolute positions in space, they might be divided by a barrier or curvature in
space that didn't exist from your initial perspective. Here the Raw Cosine Similarity4 can be
useful to measure orientation, but not access. It tells us that the two concepts are facing each
other and that they belong in the same general topical region, but it cannot tell us if the semantic
paths are perspectivally traversable.
To navigate through this embedding terrain without binding ourselves to the flatness of
local attention space, we cannot simply rely on the standard dimensionality reduction techniques
like UMAP to project article tokens (Leland McInnes, John Healy, James Melville, 2018). While
these algorithms represent industry standards for high dimensional visualization, they are
fundamentally unsuited for diagnosing endogenous semantic bias. Especially when dealing with
anisotropic NLI models, UMAP will attempt to patch together the homological breaks where the
meaning of disagreement is defined. It erases the very separations in semantic clusters that allow
us to tell apart 'aggressor' from 'victim'. Secondly, UMAP enforces a Geometric Normalization
on the input articles. The algorithm is designed to preserve local neighborhoods by pulling
4 A method of quickly comparing two vectors by the angle of the Cosine between them

similar things together though aggressively shrinking the vector space between opposing clusters
to make the visualization legible to a human. In this context that behavior is counterproductive.
Relying on UMAP means the system will compress distance between an article framed as a
terrorist event and an article reporting on glorious resistance. Ironically this means a locally
obsessed system will also create and preserve a unified semantic field blurring all 'bias shapes'
into amorphous topic-conditioned blobs. Borrowing from topology, the system attempts to
account for the complexity of discourse in combination with the NLI anisotropy problem by
treating embedding space as a multidimensional sheaf rather than relying entirely on flat
Euclidean comparisons of absolute variance across seeded initializations.
"In the days when Sussman was a novice, Minsky once came to him as he sat hacking at
the PDP-6. What are you doing?'' asked Minsky. I am training a randomly wired neural net to
play tic-tac-toe,'' Sussman replied. Why is the net wired randomly?'' asked Minsky. Sussman
replied, I do not want it to have any preconceptions of how to play.'' Minsky then shut his eyes.
Why do you close your eyes?'' Sussman asked his teacher. "So that the room will be empty,"
replied Minsky. At that moment, Sussman was enlightened." - Ali Rahimi, Benjamin Recht
Conceptually the system is based on a Vertical Waterfall architecture. In line with
aforementioned constraints, the system is organized into sequential tracks where the output of
one layer may be extracted and analyzed while also serving as the input to the next layer. We will
describe each layer as a 'track'.
### Track 1: Logits

The first step of the pipeline is ingestion. We take a target article and run it through the eight adversarial NLI gates. As the article passes through each gate, the model measures the friction between the text and the prompted hypothesis. It outputs a three-dimensional score: entailment, neutral, and contradiction. In implementation, this becomes a raw 24-dimensional verdict channel, corresponding to eight NLI views multiplied by three NLI labels. These logits function strictly as the baseline diagnostic of the model's forward-pass verdict behavior. They are not treated as the main geometric object of the system and are kept separate from the CLS-derived observer geometry.

This tensor of scores provides low-resolution information but represents the final classification product of the NLI forward pass. It is useful precisely because it is comparatively simple: it tells us how the model distributes entailment pressure across the hypothesis gates before any of the later geometric operations. Other than projecting logits for comparison and diagnostic visualization, we do not treat Track 1 as the primary manifold. Track 1 is therefore best understood as the constitutional floor of the system: the raw verdict substrate against which later geometric transformations can be compared.

### Track 1.5: Spectral Polarity Extraction

The logits, as mentioned, do not give us very many specifics as to why the model made a decision for that particular article and hypothesis. A score of high entailment, for example, could reflect meaningful semantic alignment, or it could reflect the model reacting to a small set of vocabulary cues. To get a better picture, Track 1.5 measures the article's weighted resistance to reinterpretation by extracting the semantic shift induced by contrastive observer frames.

Earlier versions of the architecture considered a more computationally expensive backward-pass gradient probe, in which the model weights were frozen and the loss was differentiated with respect to the input representation. The current architecture uses a forward-pass approximation instead. For each article, the system extracts the CLS or mean-pooled observer representations generated by the paired hypotheses and computes the A-minus-B polarity deltas across the eight contrastive pairs. These deltas form an eight-way aggregate field of local semantic shear.

More simply, the "Polarity Delta" measures how the article's semantic position shifts when it is contrasted against opposing observer frames. Where the field is steep, the article is resistant to reinterpretation; where it is flatter, the article has more leeway of interpretation. This gives us two pieces of information: the direction of semantic shift, and the magnitude of stress required to move the article through the observer field.

We already know why logits are not enough on their own, but even raw forward-pass geometry is misleading. Mapping the articles directly in vector space inherits the anisotropy of the transformer representation, causing data to cluster together in ways that can obscure the very disagreement the system is meant to inspect. Specifically, Track 1.5 performs a spectral decomposition over these discrete contrastive delta matrices. Instead of assuming a manually labeled binary conflict and subtracting magnitudes by hand, we treat the eight contrastive pairs as an unstructured point cloud of observer-induced shifts.

Since our system relies on contrastive hypotheses, our bots do not have static personas, such as a simple "Pro-Israel" or "Pro-Palestine" bot. Instead, the system creates an interferometric field from the composite polarity vector space. If we used static personas, that would introduce an unnecessary layer of exogenous information: we would be the ones orchestrating the poles. The current field represents the net geometric tension generated when different contrastive topical frames exert high-dimensional divergence on the embedding. We define this as an Observer Shear Field. It does not simulate a conversation. It performs a structural stress test within the positional geometry of the model's forward-pass representation.

By performing Singular Value Decomposition, equivalently PCA on the centered contrastive delta matrices, the system endogenously discovers the primary axes of semantic stress without human intuition dictating which side of the conflict must dominate. In this sense, Track 1.5 does not tell the model what the ideological poles are. It asks which direction of observer-conditioned variation carries the most stable stress signal. **Alternative phrasing: Track 1.5 is the system's compass, not its judge. It discovers the dominant direction of semantic shear but does not decide what that shear morally means.**

### Track 2: Geometric Expansion

As our baseline experiments indicated, computationally speaking, a flat surface is not the best container for semantic thought. Given that we have established the Spectral Polarity Delta as a superior measurement of ideological resistance, a natural question arises: why retain the geometric map at all? Why not simply apply the polarity delta to logits, using resistance to add more information to a confidence score and discarding embedding space entirely?

The answer is that the goal is not simply to score the narrative, but to traverse it. If we applied the deltas only to logits, we would reduce the article to a scalar or low-dimensional verdict profile. Such a score is a static judgment. It cannot tell us which other articles are nearby, whether a transition path exists, or how observer stress changes as the article moves through a field of related texts. A logit can have no neighbors and cannot occupy a position in a manifold. Track 2 supplies the coordinate container necessary to position Track 1.5 shear in relation to a latent topology.

This is the latent embedding that represents a specific observer-conditioned understanding of an article. More specifically, the magnitude and stability of these embeddings contribute to the semantic mass of an observation: high-confidence or structurally stable observations act as denser anchors in the manifold, while low-confidence or unstable observations act as lighter, more volatile points.

Unlike logits, which can be mathematically compared in a shared 24-dimensional verdict channel, embeddings have no automatic shared currency. The solution is to enforce a shared spatial geometry, making the observer views born-aligned in the same coordinate system. To resolve this, we employ Random Kitchen Sinks (Rahimi and Recht, 2008), not merely to expand dimensionality but to approximate a kernel feature space in which different observer-conditioned embeddings can be compared as objects in a common Hilbert space.

This approach offers two advantages. First, standard kernel methods can imply a theoretical manifold without producing a directly navigable shape for our downstream system. RKS provides an explicit finite-dimensional approximation of that kernel space. Second, by using a shared basis across observer views, the system prevents each observer from inventing a private coordinate system. All observer projections pass through the same mathematical lens.

We experimented with whitening methods such as ZCA, which attempt to inflate the anisotropic transformer cone while preserving the original rotation of the data. In practice, whitening was not always necessary and could be counterproductive, because it sometimes destroyed local disagreement texture. This is especially important because Track 1.5 already uses spectral analysis to identify semantically relevant directions of variation. Whitening can make the space cleaner while making the disagreement less legible.

The system supports several kernel families, including RBF, IMQ, and Matern. The Matern implementation samples from a Student-t spectral distribution, where the smoothness parameter controls the heaviness of the tails. Conceptually, this was motivated by the idea that heavy-tailed spectral sampling could preserve high-frequency semantic features and topological barriers that a Gaussian kernel might smooth over (Rasmussen and Williams, 2006). However, this should be treated as an ablation condition rather than a proven universal advantage. The current evidence supports testing kernels comparatively, not declaring one kernel as intrinsically superior.

We formally qualify the interaction between geometry and observer stress through the Observer Metric Tensor. While standard embeddings imply a Euclidean distance between concepts, the kernel-approximated manifold allows observer shear to warp the effective distance between articles. Conceptually, we can write this as:

$$g_{\mu\nu}^{total}(x) = \frac{1}{\rho(x)}\delta_{\mu\nu} + \nabla_\mu \Phi \nabla_\nu \Phi$$

Here, $\delta_{\mu\nu}$ represents the base content-to-content geometry supplied by Track 2, while $\nabla_\mu \Phi \nabla_\nu \Phi$ represents semantic shear from Track 1.5. Track 3 later supplies the conformal density term $\rho(x)$. Operationally, the production Track 5 path implements this through kernel fusion and conformal scaling, while the stricter Riemannian implementation exists as an ablation branch. This means the equation should be understood as the conceptual scaffold for the architecture rather than a claim that every visualization directly solves a continuous geodesic field.

The polarity delta is the force vector of observer shear stress. In regions of consensus, the delta may be flatter and the metric resembles a more ordinary Euclidean space. Near a structural singularity, the shear magnitude increases and the effective distance between neighboring articles can stretch. Though the V-Observers are born-aligned by way of RKS, the Observer Polarity Delta from Track 1.5 acts as a local warping force inside the shared geometric container.

This is where we unlock a defined physical manifestation of the aforementioned singularities. Consensus manifests as constructive wave interference, where different observer-conditioned waveforms align to create a higher-amplitude, lower-resistance region. The bridge is defined by the constructive alignment of article density in the geometric manifold and reduced observer shear. The topological vocabulary used throughout the system emerges from two orthogonal quantities:

The Landscape: this relates to existence. It asks whether a geometric path exists between articles in the shared manifold. Physically, it is derived from Track 2 content-to-content structure.

The Polarity Delta: this relates to resistance. It asks whether the path is actually traversable under observer stress. Physically, it is derived from Track 1.5 content-to-observer interference.

Together these produce four terrain modifiers:

"The Bridge" (High Density, Low Gradient): constructive interference where the system detects many similar articles occupying one region, and the V-Observers are in relative perspectival agreement.

"The Swamp" (High Density, High Gradient): phase incoherence. There is a high volume of related articles, but the V-Observers are out of phase. The base geometric path exists, but it is difficult to navigate cleanly because observer shear is high.

"The Tightrope" (Low Density, Low Gradient): brittle resonance. The V-Observers have comparatively high perspective rapport, but the source-text landscape is sparse. If the path deviates even slightly, it risks falling into the void.

"The Void" (Low Density, High Gradient): sparse geometry combined with incoherent observer stress. Here the coordinate system is not simply empty; it is weakly supported and high-resistance.

This completes the geometric manifold, Track 2. Defining a map, however, is not the same as making it navigable. We have the slope and the plane, but we still need an explicit density validator.

### Track 3: The Dirichlet Sweep

If Track 2 provides a map, Track 3 provides a lens. Having established a topological floor for the manifold, we face another limitation: we cannot possibly account for every rhetorical frame in existence. A system with 10,000 bots is computationally unfeasible and would be incoherent. Because of this, early versions of Track 3 were motivated by the need to blur the lens and estimate missing dialectic structure between frames. This was not the final role of Track 3.

The original intuition that we needed to smooth the gaps between observers rested on a dangerous Euclidean assumption. Inside the model, there is no guaranteed continuous plane where the distance between every two points is meaningful. If we rely only on ordinary Euclidean metrics, we demand an indestructible field that hallucinates continuity where the model may know nothing. As a result, Track 3 becomes a density scanner: gaps are not smoothed away, but treated as evidence about the substance or fragility of semantic support.

Consider the dialectic of International Law. One pole emphasizes violations, while the other emphasizes compliance. Scenario A: an article discusses a border skirmish using language that strongly activates both frames, increasing tension. Scenario B: an article offers diplomatic platitudes without citing specific legal frameworks or rhetorical devices, causing both poles to activate weakly. In both cases, a simple midpoint location may look similar. But the first midpoint is the result of strong opposing forces, while the second is the result of weak semantic substance. A snapshot of the tug-of-war cannot tell us which case we are seeing. Track 3 asks how stable the article is under pressure.

Mechanically, Track 3 uses a dynamically optimized weight simplex over RKHS-projected observer views. Instead of treating the observer outputs as one softmax verdict, we treat the eight observer views as a simplex of possible mixtures. With high alpha, the system is forced toward consensus: probability mass is spread broadly across observer views. With low alpha, the system is allowed to fracture: probability mass can collapse into sharper observer-specific spikes.

This is the algorithmic source of the tug-of-war metaphor. The article begins near a high-entropy barycenter. Over sequential annealing stages, the system relaxes the constraint and records whether the observer mixture holds together or tears apart. Functionally, this acts as KL-regularized simplex-weight annealing over RKHS-projected bot perspectives. The KL term binds each stage to the previous stage, giving the system inertia. Without this term, each alpha interval would be an independent snapshot. With it, Track 3 measures the integrity of a path through observer-mixture space.

The implementation is best described as hot-to-cold annealing. The system begins in a high-alpha consensus state with strong prior regularization and weak separation pressure. It then cools toward lower alpha values, where prior constraint relaxes and structured observer separation can emerge. The variance term encourages observer mixtures to pull apart, while KL divergence binds the process to the previous stage. This treats the eight-component simplex of attention allocated to the observer gates as a dynamic state variable rather than a static hyperparameter.

Concretely, the annealing objective can be summarized as a competition between separation and inertia:

$$L_t(w_t) = -\alpha_t^{-1}\,\mathrm{Var}\left(\sum_b w_{t,b}\phi_b(x)\right) + \lambda_{KL}\alpha_t\,KL(w_t||w_{t-1})$$

where $w_t \in \Delta^7$ is the simplex-constrained observer weight vector at annealing stage $t$, $\phi_b(x)$ is the RKHS-projected semantic embedding contributed by observer $b$, and $\alpha_t$ controls the annealing schedule. At high alpha, the KL term is strong and separation pressure is weak, keeping the system near the consensus barycenter. At low alpha, the constraint relaxes and observer separation can emerge.

From this trajectory, the system derives two important quantities. The first is adaptive work, measured as the accumulated distance between successive simplex states:

$$W_{anneal} = \sum_{t=0}^{T-1} ||w_{t+1} - w_t||$$

The second is the Track 3 conformal density:

$$\rho_i = \frac{1}{1 + \tau ||b_i||}$$

where $b_i$ is the Track 3 fused variance for article $i$, and $\tau$ is the temperature or willingness-to-agree parameter. Higher variance produces lower density, stretching the local manifold; lower variance produces higher density, compressing the local geometry. In the current implementation, this density is the capstone-facing Track 3 density. Local KNN density used by the visualizer is retained only as a geometry diagnostic or fallback and must not be confused with Track 3 rho.

Track 3 therefore does not add another raw coordinate block to the final Track 5 representation. It produces a conformal density field used to validate and scale the fused manifold. Articles that maintain their observer-mixture position under annealing produce denser, more stable local geometry. Articles that tear away from the consensus position produce lower-density, higher-uncertainty regions. **Alternative phrasing: Track 3 is not a third map; it is the pressure test that tells us whether the map is physically trustworthy.**

Earlier versions of Track 3 outputted Cracks and Bonds metrics, which allowed easier comparison across kernel geometries. These diagnostics remain useful, but they are no longer the primary Track 3 claim. The primary Track 3 output is the conformal density field used by the later synthesis and terrain logic.

### Track 4: Traversal Telemetry

In attempting to operationalize the information geometry of the fused manifold, Track 4 simulates particles across the article graph based on density, stress, and metric distance. Its purpose is to measure the work required to make semantic leaps between article nodes. Mechanically, Track 4 is best described as a thermodynamic stochastic walk over a metric-respecting article graph. It is Markovian because each transition depends on the current article, local neighbors, walker temperature, and metric resistance. It is MCMC-inspired, but it should not be described as a full posterior sampler, because the current implementation does not define a target posterior distribution and then prove asymptotic convergence through accept-reject sampling.

Take our dataset $E \in \mathbb{R}^{N \times H}$, where $N$ is the number of articles and $H$ is the embedding dimension of the fused representation. At step $t$, a walker is at article $i$. The system evaluates the local neighborhood of $k$ closest articles using the metric graph constructed from Track 2 geometry, Track 1.5 stress, and Track 3 density. For each possible neighbor $j$, it calculates a transition weight based on the walker temperature and the metric-respecting edge cost:

$$w_{ij} = \exp\left(-\frac{d_g(x_i,x_j)}{\tau}\right)$$

where $d_g$ is the discrete metric edge cost induced by the fused geometry. This is not a continuous geodesic integral over a smooth manifold; it is a graph approximation of metric-aware traversal.

We condition two types of walkers: Cold Walkers and Hot Walkers. Cold Walkers have lower temperature and are more rigidly attached to the boundaries of the manifold. Hot Walkers use a higher temperature multiplier, allowing them to tunnel through higher-stress regions. As the walker explores, it accumulates work:

$$W_{total} = \sum_{t=0}^{T-1} d_g(x_t,x_{t+1})$$

When accumulated work reaches the cognitive horizon, the walker flips into retreat mode and is penalized toward the anchor coordinate:

$$D_{j \to A} = d_g(x_j,x_A)$$

The resulting retreat transition weight is:

$$w_{ij} = \exp\left(-\frac{d_g(x_i,x_j)}{\tau}\right)\cdot\exp\left(-\gamma d_g(x_j,x_A)\right)$$

The key architectural boundary is that Track 4 does not assign final semantic verdicts. It exports mechanical telemetry: high-dimensional path coordinates, the work integral of each trajectory, a closed-loop survival boolean, and Markov diagnostics such as committor probability, mean first passage time, and reactive flux when boundary states are available. Whether a high-work or failed-loop path should be interpreted as semantically important belongs downstream in Track 5 or the evaluation layer. Track 4 tells us how the traversal behaved; it does not decide what the article is.

Current anchor selection is also terrain-aware. Rather than selecting only the highest-stress articles, the system selects a small set of polarity anchors using zone-constrained farthest-first logic, attempting to include Bridge and Void anchors when available and then a third distinct terrain region. For each anchor, the current swarm runs five walkers: two hot and three cold. This preserves the visual intuition of a Hero's Journey audit while keeping the underlying export contract mechanical.

### Track 5: Synthesis

Track 5 is the synthesis layer of the architecture. Earlier iterations of the Semantic Interferometer relied on concatenating all tracks, simply treating all raw feature vectors as another set of coordinates. This failed because Track 2 geometry, Track 1.5 shear, and Track 3 density do not represent the same kind of object. Concatenation forces coordinate features, directional stress, and density/uncertainty into one flat vector space, which collapses the mathematical distinction the architecture is designed to preserve.

The final architecture relies on kernel and metric fusion rather than raw concatenation. Rather than snowballing a giant vector, the system calculates article-to-article similarity in each relevant track. In the Hadamard branch, Track 2 and Track 1.5 are converted into similarity kernels and multiplied element-wise. This means a strong connection must be supported by both base geometry and observer-shear agreement:

$$K_{fused} = K_{T2} \odot K_{T1.5}$$

Track 3 is then used as a conformal density normalizer rather than a spatial coordinate block. Its role is to stretch or compress the fused manifold based on consensus and disagreement:

$$D_{conformal}(i,j)=\frac{D_{fused}(i,j)}{\sqrt{\rho_i\rho_j}}$$

where $\rho_i$ and $\rho_j$ are the Track 3 densities for the two articles. Low-density articles stretch local distances, while high-density articles compress them.

The system also includes a stricter Riemannian ablation branch. In that branch, Track 2 acts as the base geometry container, Track 1.5 enters as the shear or anisotropy term, and Track 3 enters as the conformal density term. A pairwise metric-respecting distance matrix is built first, then embedded back into a vector space for downstream compatibility. This branch is useful for testing whether the conceptual metric tensor improves evidence quality, but the production-default path remains the Hadamard/conformal assembly unless explicitly changed.

The crucial correction is that Track 3 is not multiplied onto final 2D or 3D visualization coordinates. Doing so would merely push points radially away from an arbitrary plotting origin and destroy local topology. Instead, Track 3 modifies pairwise distance or kernel structure before visualization. The visible manifold is therefore a rendered projection of a fused structure, not the mathematical object itself.

**Alternative framing: Track 5 is the interferometer proper. Track 2 provides where articles are, Track 1.5 provides how observer stress pulls them, and Track 3 provides how much local semantic substance exists. Track 5 is where those separate physical meanings become one testable manifold.**

## Discussion
Where guidelines constrain human annotation workers, hypotheses in NLI constrains bots
internal activation pattern and attention allocation, shaping which semantic relations are made
salient during inference. In this sense the NLI hypothesis is likely bound to similar
methodological realities; the hypothesis can serve to safeguard the model from its own bias but
would likely not improve the signal or enhance its 'neutrality'. Where disagreement among
workers can serve as a useful statistical signal, agreement in NLI amounts to consistent outputs
across hypotheses. This all suggests, however, that consensus, whether among human annotators
or across model outputs, is not an output with enough definition to indicate semantic clarity. In

both cases the apparent structure may be the results of the probing framework rather than the
structure of the text itself. Because of this, the critical variable here is not the presence or
absence of agreement, but the conditions under which agreement is produced and maintained.
This is where the unique advantage of an observer based NLI emerges. Beyond scale, the system
allows for controlled variation of interpretive frames within a shared representational space,
allowing for a form of latent space tomography where it is possible to interrogate the nature of
the represented signal of semantics itself rather than a labeled profile. The question then shifts,
then, from a matter of more complicated statistical analysis of labeled profiles to direct analysis
of the geometric of perspectival variation under controlled conditions.
The system in this paper attempts to begin precisely at this point, asking whether
disagreement can be evaluated beyond non-linear profiling statistics as a series of
transformations applied within a shared semantic space. In CrowdTruth, judgement happens
upstream, based on questions such as: is this disagreement an artifact of ambiguity? Is this bad
worker behavior? Task design? In Semantic Interferometer the statistical adjudication partially
occurs further downstream, as all NLI observer instances are allowed to generate all their
respective outputs without supervision. Only once those perspectives are injected into a shared
representational space does the system emergently discover perspectival shear, quantifying the
relevance of the different observer perspectives as mathematical derivations.
Where in CrowdTruth disagreement is portrayed as an artifact of the object itself, the
architecture proposed in this paper attempts to acknowledge the inherent limits of measuring
semantic shadows. Using the collection of several AI 'mindspace' projects collided into a single
object, the Semantic Interferometer attempts to represent disagreement as the repeated structure
of the shadows statistically as they build a complex shape. Disagreement can become a question

of coherence, stability, asymmetry, density, and path cost over a series of perspectival
perturbations, rather than noise. The Semantic Interferometer was designed to mathematically
explore relativism in journalistic semantic analysis and demonstrate the failures of objective
neutrality. Prolonged interaction with the foundational mechanism of natural language
processing, particularly in relation to high dimensional embeddings and their inherent anisotropic
properties, prompted a methodological and conceptual drift. With each dead end (from
pre-labeled signed axis to raw embedding concatenation to GPA procrustes sequencing) the
constraints of current NLP architectures repeatedly demonstrated that treating semantics as
perfectly sociologically fluid did not align with the statistical reality of the learned representation
space. Every attempt to use arbitration as a means to force the text into a certain framework
while transposing that frame heuristically failed on an empirical basis. Crucially, these failures
were not evidence of relativistic uncertainty but rather suggested structural resistance of the
vector space in relation to its pairwise perspective. An assortment of approaches in creating a
Semantic Interferometer failed largely because they relied on the infinite malleability of encoded
text when the topology of the specific model's latent space demands to be respected even in a
vacuum.
The issue is not that Ground News reduces the complexity of semantics into a one
dimensional line, which is a forgivable inevitability; the system described in this paper works
similarly, producing an annotated rank-1 projection, or a three dimensional object. Our system
attempts to indicate how the very concept of the Overton Window, the unsaid bedrock on which
Ground News sits, is massively underdefined and interrogated in such a way that inhibits a more
causal understanding of observer dependent semantics. Through this lens Ground News goes
beyond revealing bias and actively operationalizes a preconceived center by treating the axis as a

static object, when it is not. Having theoretically established how the Semantic Interferometer
attempts to account for an inherently 'shadowy' Overton Window, we will now assess the
specific results of our system when fed both real and synthetic corpora.
## Results
If bias is the embedded signal we were hunting for, there were a number of noisy red
herrings that attempted to corrupt the system and lead to erroneous results. First, the
aforementioned anisotropy trap; models group things together if they talk about the same things,
which we solve in Track 1.5. Second is the Euclidean hallucinations that accompany methods
like UMAP, which we solved using the RKS kernel trick. The third is barycentric consensus, or
the 'Ground News effect' where the classifier is geared towards finding a 'golden middle'. This
we address in Track 3 by ripping apart polite consensus, defining the middle path as the noisy
compromise.
Using our Procrustes analysis we can compare real, random, constant,
and shuffled versions of our inputs to estimate whether the resulting geometry separates from
matched controls. In the submitted capstone run, the shuffled control yielded a distance of 1.313
against the real manifold, while the random control yielded a distance of 1.302. These distances
should be interpreted as evidence of geometric separation under that run configuration rather
than as a universal proof that the system has isolated semantic bias. The stronger claim is that
the manifold contains a non-trivial structure that can be compared against stochastic and
matched controls, and that the direction and magnitude of separation must be reported by kernel,
control family, seed, and metric basis.
All this said, a fundamental sensitivity here is the contrastive NLI entailment and
hypotheses set. Unavoidably, the entire system is defined by these inputs. If they could somehow

be auto-formalised on inference, the system might serve as a much more reliable Semantic
Interferometer.
Figure 1: This is our Global Mean observer manifold snapshot of a group of 30 synthetically
generated articles, displaying the foundational coordinate prior to targeted observer
displacement.

Figure 2: This visualisation illustrates the system's ability to use the 16 contrastive probes to
discover relevant axes. Rather than enforcing manual labels, the architecture uses Singular Value
Decomposition on the resulting matrices to discover the principal axes of the discourse. Notice
here that the nature of our dialectal prompting likely inspired a generally two sided conflict

where the main axis leads one direction.
Figure 3: This rendering demonstrates the path mechanics and constraints of the Semantic
Interferometer. This is a sequence of walker trajectories generated from a selected polarity
anchor on the far right of the image. The path color indicates high traversal work and instability
in the path telemetry, not a final Track 4 semantic verdict.
5 Polarity anchors represent selected high-structure or high-shear articles used to probe traversal behavior.

The starting path requires disproportionately higher energy relative to its manifold placement. In
the language of the traversal layer, the walker's accumulated fatigue exceeded the thermal
threshold before it could close a stable loop.
Figure 4: Consider the following synthetic article labeled 'Liberal Zionist'. It sits on a ridge far
from the center on this set, indicating that both Track 2 kernel outputs and Track 1.5 Polarity
perceive the article as further outside the discourse, suggesting it has both perspectival shear and

a geometrically distinct topical embedding. Articles near the center on this given set are largely
consistent of Western Leftists and Islamic Resistance framings.
Figure 5: This demonstrates the system's observer-centered visualization mode. A selected
article can be treated as the local reference point for a centered manifold view, allowing us to
inspect how article neighborhoods, observer displacement, and local stress differ relative to that
anchor. The claim is not that the model literally adopts the article's opinions, but that the
geometry can be re-expressed around an article-conditioned reference frame. In this view, the
Mean Manifold and an Article Observer Manifold may cluster differently, revealing changes in
local alignment, displacement, and observer shear.
Figure 6: This snapshot records the system's waterfall architecture and how the NMI score of the
output fluctuates but remains stable throughout the inference despite the number of
transformations applied throughout the architecture.

Github Link: https://github.com/gizmolotry/SemanticInferometer

## Declaration of AI-assisted writing and development support

During preparation of this revised manuscript, the author used OpenAI ChatGPT/Codex to assist with manuscript restructuring, codebase auditing, wording refinement, and consistency checks between the written methods and the implemented software. The author reviewed, edited, and verified the final manuscript, code claims, citations, and reported results, and takes full responsibility for the content of the work.

## Works Cited
Aroyo, L. and Welty, C., 2015. Truth is a lie: Crowd truth and the seven myths of human
annotation. AI Magazine, 36(1), pp.15-24.

Berlin, I., 1998. The First and the Last. The New York Review of Books, 45(8), pp.47-50.
Bowker, G.C. and Star, S.L., 1999. Sorting things out: Classification and its consequences.
Cambridge, MA: MIT Press.
Frenda, S., Abercrombie, G., Basile, V., et al., 2024. Perspectivist approaches to natural language
processing: a survey. Language Resources and Evaluation, 58.
Goodfellow, I., Bengio, Y. and Courville, A., 2016. Deep learning. Cambridge, MA: MIT press.
Goodman, N., 1978. Ways of worldmaking. Indianapolis: Hackett Publishing.
International Federation of Journalists (IFJ), 2019. Global Charter of Ethics for Journalists.
[online] Available at: https://www.ifj.org.
Kuhn, T.S., 1962. The structure of scientific revolutions. Chicago: University of Chicago Press.
Mackinac Center for Public Policy (n.d.) The Overton Window. Available at:
https://www.mackinac.org/OvertonWindow
McInnes, L., Healy, J. and Melville, J. (2018) 'UMAP: Uniform Manifold Approximation and
Projection for Dimension Reduction', arXiv preprint arXiv:1802.03426 [online]. Available at:
https://arxiv.org/abs/1802.03426
Pavlick, E. and Kwiatkowski, T., 2019. Inherent disagreements in human textual inferences.
Transactions of the Association for Computational Linguistics, 7, pp.677-694.
Pye, L.W., 1999. Civility, Social Capital, and Civil Society: Three Powerful Concepts for
Explaining Asia. Journal of Interdisciplinary History, 29(4), pp.763-782.

Rahimi, A. and Recht, B. (2008) 'Weighted Sums of Random Kitchen Sinks: Replacing
minimization with randomization in learning', Advances in Neural Information Processing
Systems 21. Available at:
https://papers.nips.cc/paper_files/paper/2008/hash/0efe32849d230d7f53049ddc4a4b0c60-Abstra
ct.html
Reiheld, A., 2013. Asking too muchinfinity Civility vs. Pluralism. Journal of Social Philosophy, 44(2),
pp.144-164.
V-Dem Institute, 2023. Democracy Report 2023: Defiance in the Face of Autocratization.
Gothenburg: University of Gothenburg.
Wittgenstein, L., 1922. Tractatus logico-philosophicus. London: Kegan Paul, Trench, Trubner &
Co.
Xu, Y. and Jurgens, D., 2026. Beyond Consensus: Perspectivist Modeling and Evaluation of
Annotator Disagreement in NLP. arXiv preprint arXiv:2601.09065. Available at:
https://arxiv.org/abs/2601.09065.

## Additional implementation references for revised manuscript

Bochner, S., 1933. Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse. Mathematische Annalen, 108, pp.378-410.

Chodera, J.D. and Noe, F., 2014. Markov state models of biomolecular conformational dynamics. Current Opinion in Structural Biology, 25, pp.135-144.

E, W. and Vanden-Eijnden, E., 2010. Transition-path theory and path-finding algorithms for the study of rare events. Annual Review of Physical Chemistry, 61, pp.391-420.

Golub, G.H. and Van Loan, C.F., 2013. Matrix computations. 4th ed. Baltimore: Johns Hopkins University Press.

He, P., Liu, X., Gao, J. and Chen, W., 2021. DeBERTa: Decoding-enhanced BERT with disentangled attention. International Conference on Learning Representations. Available at: https://arxiv.org/abs/2006.03654

Rasmussen, C.E. and Williams, C.K.I., 2006. Gaussian processes for machine learning. Cambridge, MA: MIT Press.

Scholkopf, B., Smola, A. and Muller, K.-R., 1998. Nonlinear component analysis as a kernel eigenvalue problem. Neural Computation, 10(5), pp.1299-1319.

Williams, A., Nangia, N. and Bowman, S.R., 2018. A broad-coverage challenge corpus for sentence understanding through inference. Proceedings of NAACL-HLT 2018, pp.1112-1122.
