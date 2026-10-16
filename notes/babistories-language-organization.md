# Probing language organization in nanoMosaics

Research notes from the discussion on October 15, 2026.

Target run: `babistories-12L768d`, trained on BabiStories. These notes describe proposed experiments, not findings from the trained checkpoint.

The aim is to study how syntax, semantics, and lexical knowledge are organized within the model, using recordings, decoding, and selective interventions. Distinguish three questions throughout: where information is decodable, where it causally influences behavior, and which learned parameters support it. These locations need not coincide.

## Research questions

- Is syntactic structure more localized or distributed than semantic structure?
- In which layers, memory types, heads, slots, or representational subspaces does this organization occur?
- Where is lexical knowledge represented and used? Is it concentrated or distributed across memory mosaics?
- Do different linguistic constructions share components, or depend on distinct subsets?
- How do stable word knowledge and context-dependent meaning interact?

The model can serve as a computational system for studying language organization. Claims about human language networks would require additional comparisons with human behavioral or neural measurements using matched stimuli.

## Model anatomy

The repository [training configuration](../nanoMosaics/config/train_babistories.py) specifies 12 layers, 768 embedding dimensions, 12 heads, one persistent memory bank per layer, and 2,688 persistent slots per head. Confirm these settings against the checkpoint before running experiments.

In the [implementation](../nanoMosaics/mosaic_model.py), each block applies contextual memory and then persistent memory, each with a residual connection.

| Component | Candidate role to investigate |
| --- | --- |
| Contextual memory, 12 heads per layer | Retrieval and combination of information from the current text |
| Persistent memory, 12 heads per layer | Learned associations accessed through key–value slots |
| Residual stream | Information carried between computations |
| Tied token embedding and output matrix | Token representations and conversion to token predictions |

There are 144 contextual heads and 144 persistent heads. Start at the layer and head scales before examining individual slots. Record head outputs before the output projection mixes them.

Localization must specify a scale: a property could be concentrated in two layers but distributed across their heads and dimensions. Native memory slots and heads are architectural units; individual residual coordinates are a more basis-dependent unit of analysis.

## Controlled behavioral tasks

Use vocabulary and constructions supported by BabiStories. Generate controlled stimuli with explicit labels, and first establish above-baseline model performance. Score continuation probabilities rather than relying on instruction following.

| Target | Manipulation | Measurement |
| --- | --- | --- |
| Syntactic agreement | “The dog near the cats is/are…” | Correct verb preference and distractor-number effects |
| Syntactic structure | Same construction with varied nouns and verbs | Subject identity, dependency relations, constituent depth |
| Lexical semantics | Animals, foods, tools, and locations in varied contexts | Semantic category and word-specific associations |
| Semantic roles | “The rabbit chased the fox” versus “The fox chased the rabbit” | Who did what to whom |
| Meaning across syntax | Active and passive descriptions of the same event | Generalization of event representations across form |
| Discourse state | Characters moving or transferring objects | Current location, ownership, and reference |

For agreement, use the score `log P(correct verb | prefix) - log P(incorrect verb | prefix)`. For multi-token candidates, score the complete candidate consistently and control tokenization and length.

Counterbalance words, positions, frequency, and lengths. Hold out lexical combinations and sentence templates, and use held-out word types where testing abstraction beyond lexical identity. Separate task difficulty from localization by comparing tasks at similar baseline performance. Keep all labels and readout positions consistent with the information available in the causal prefix.

## Map decodable information

Record representations before and after each contextual and persistent memory module, individual head outputs, and persistent retrieval weights and values. Fit small regularized probes for syntax, lexical categories, semantic roles, and discourse state.

Compare single heads, head groups, and full representations with controlled probe capacity and training data. Include embedding-only, untrained-model, and randomized-label controls. A probe may learn shortcuts or memorize word identities; high decoding accuracy is not evidence that the model uses the decoded feature.

A structural probe can test whether transformed representation distances and norms reflect dependency-tree distances and depth. Earlier token states in this causal model cannot incorporate future words, so do not interpret them as full-sentence representations.

Deliverable: maps of decodability by layer, memory type, head, and readout position.

## Establish causal involvement

### Selective lesions

Replace a component output with a matched baseline and measure effects on the target task, other linguistic tasks, and ordinary next-token prediction. Compare matched resampling and zeroing to assess intervention sensitivity.

A component whose removal impairs everything is not necessarily specialized for syntax. Stronger evidence is a selective agreement impairment with relatively preserved lexical-semantic performance. Match comparison lesions for size, layer, and activation magnitude where feasible.

### Activation swaps

Use matched inputs such as “The dog near the cats…” and “The dogs near the cats…”. Transfer one component activation from the plural run into the singular run and test whether the prediction shifts toward “are”. Repeat across nouns, templates, distractors, and aligned token positions.

For semantic roles, swap activations between role-reversed events and test whether continuations change consistently with the transferred agent or patient. Use held-out examples to validate components selected on a discovery set.

Separately intervene on memory routing and retrieved content: query/key interventions test access, while value interventions test the contribution of retrieved content. These experiments identify causal influence under the intervention, not a unique storage location.

## Quantify localization and distribution

Rank candidate components on a discovery set, then evaluate cumulative lesions of the top 1, 2, 4, 8, and more components on held-out examples. Compare against repeated matched random sets. Also test whether restoring a small component set rescues behavior under a defined corruption.

| Pattern | Interpretation supported |
| --- | --- |
| Small set produces selective impairment and targeted rescue | Concentrated causal contribution |
| Many components produce small effects | Distributed contribution |
| Single lesions have little effect but joint lesions strongly impair behavior | Redundancy or interactions |
| Information is widely decodable but interventions matter at few sites | Distributed accessibility with concentrated causal use |

Report single and joint effects with uncertainty across stimuli. A necessary bottleneck can depend on distributed upstream computation. Avoid reducing localization to a single binary label or assuming effects add independently.

## Locate lexical knowledge

Separate three meanings of “lexicon”:

1. **Word form:** token-specific embedding rows provide an explicit interface, but tokens may be subword pieces. This is not proof that complete word meanings reside there.
2. **Stable lexical knowledge:** properties and associations that generalize across contexts, such as an animal category or characteristic action.
3. **Context-dependent meaning:** the referent, event role, location, or other state of a particular word occurrence in a story.

Compare lexical decoding and causal effects in embeddings, persistent-memory outputs, and contextual representations.

Construct word-by-slot retrieval maps across diverse contexts. Ask whether a word repeatedly recruits the same slots, whether related words recruit overlapping slots, and whether patterns span multiple heads and layers. Then intervene on candidate slots and measure selective effects across new contexts and tasks.

Attention or retrieval overlap is descriptive evidence. High retrieval weight alone does not establish that a slot stores a word or its meaning. Distinguish suppressing a slot’s retrieval from altering its value, since these interventions change different parts of the computation.

The input embedding and output matrix are tied. Editing a row changes both input processing and token prediction. Use activation interventions to distinguish these roles before interpreting parameter edits.

## Recommended first study

Start with agreement, agent–patient roles, and lexical semantic categories. These offer a tractable comparison of syntax, compositional meaning, and word knowledge.

1. Verify checkpoint configuration and establish behavioral baselines.
2. Build controlled stimulus sets and separate discovery and evaluation splits.
3. Map decodability across layers and memory types, then heads.
4. Test promising components with selective lesions and activation swaps.
5. Measure cumulative lesion and rescue curves against matched random controls.
6. Examine persistent slots only in heads with reproducible effects.

Produce three aligned results: decodability maps, causal-effect maps, and performance curves as components are removed or restored. Track unrelated-task performance alongside the target task.

Starting hypothesis: stable lexical knowledge involves embeddings and persistent memories, while compositional behavior involves interactions between persistent and contextual memories. This is a hypothesis to test, not an architectural conclusion.

An example of a possible result is: “Agreement relies disproportionately on a small subset of contextual heads, whereas lexical-category behavior is supported across many persistent heads.” The experiments must also allow the opposite result or no clear separation.

## Methodological references

- [Hewitt and Manning 2019, A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/): decoding syntactic tree geometry.
- [Hewitt and Liang 2019, Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/): controls for probe memorization and interpretation.
- [Meng et al. 2022, Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262): causal tracing as a methodological starting point; its GPT findings are not findings about this nanoMosaics run.
- [Hase et al. 2023, Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213): why causal localization and successful parameter editing need not identify the same locations.

## Lexicon pilot plan

The immediate priority is to test stable lexical knowledge, context-dependent interpretation, and their interaction before undertaking the broader syntax and semantics comparison above. This is a proposed experiment; no results have been collected.

### Definitions

A stable knowledge base means learned lexical associations recurring across diverse contexts. It need not be a separate dictionary or compact group of memory slots. Establish the model's own defaults rather than assuming human associations hold.

Context-dependent interpretation includes sense selection (a bat in a cave versus baseball), property updating (this rabbit dislikes carrots), reference and event roles (the rabbit chased the fox), and newly assigned meanings (a dax is a small animal that hops).

Start with property updating. This tests how lexical defaults interact with entity-specific story facts; it does not by itself establish a change in the noun's lexical meaning. Sense selection can follow after confirming the model knows both senses.

### Step 1 Establish lexical defaults

Test contrasting continuations across neutral frames, for example: “The rabbit was hungry. It wanted to eat ...”, with candidates “carrots” and “fish”. Define:

`D = log P(default candidate | prefix) - log P(alternative candidate | prefix)`

Score complete multi-token candidates consistently, controlling tokenization and length. Repeat across paraphrases, names, and irrelevant details. Counterbalance candidate words across nouns to distinguish lexical knowledge from a general preference for a completion.

Select reliable associations on a discovery set and confirm on held-out frames. Report screening results; later conclusions are conditional on associations the model demonstrates. Stability establishes a learned default, not its storage location.

### Step 2 Manipulate context

Use the same target noun and final continuation across four conditions:

| Condition | Illustrative story information | Purpose |
| --- | --- | --- |
| Neutral | A rabbit lived beside the woods | Measure the default |
| Consistent | This rabbit loved carrots and disliked fish | Reinforce the default |
| Conflicting | This rabbit loved fish and disliked carrots | Test contextual override |
| Other entity | The fox loved fish and disliked carrots | Test entity specificity |

Shared continuation: “When the rabbit chose its favorite food, it picked ...”

Examples are illustrative; retain vocabulary and associations passing baseline tests. Match entity introductions, candidate mentions, and sentence structures as closely as possible. Counterbalance entity identity and mention order. The other-entity condition tests whether the model binds information to the correct entity rather than merely responding to nearby words.

Evaluate held-out paraphrases and insert unrelated sentences between the property statement and prediction. Keep stimuli within the context window.

### Step 3 Measure the interaction

Look for a default in neutral contexts, a shift when the target receives a conflicting property, a smaller shift when another entity receives that property, and return of the default in a fresh independent context.

`U_target = D(conflicting) - D(neutral)`

`U_other = D(other entity) - D(neutral)`

`Entity specificity = U_target - U_other`

A negative target override indicates movement toward the alternative under this default-oriented score. Use matched baselines when entity introductions differ. Report raw scores alongside contrasts, uncertainty across nouns and templates, and effects of intervening context. Distinguish weakening a default from reversing the preferred candidate.

With frozen weights, story information changes activations rather than permanently rewriting parameters. Return of the default in fresh context is a consistency check, not independent proof of a storage mechanism.

### Step 4 Locate the interaction

Record aligned prediction positions before and after contextual and persistent memory in every layer. Probe separately for noun identity or category, the property currently assigned to the target entity, and conflict with the default. Decoding noun identity alone does not establish representation of its lexical association. Use held-out contexts and controlled probe capacity.

Persistent memory has fixed learned keys and values, but its retrieval query depends on context. Its output can change between ordinary and unusual rabbits. “Persistent” describes parameters, not context-invariant outputs.

| Hypothesis | Supporting evidence |
| --- | --- |
| Context changes access to persistent associations | Persistent routing changes systematically; swapping relevant routing changes predictions |
| Contextual memory supplies an override | Swapping contextual-head outputs transfers the contextual preference |
| Both pathways cooperate | Joint intervention effects differ from the sum of individual effects on the chosen score |

Start with module-output swaps, then head-output swaps, between matched neutral and conflicting runs. Patch neutral activations into the conflicting run and test whether the default returns; test the reverse direction too. Align token positions carefully. Select components on discovery examples and validate on held-out examples. Measure unrelated prediction performance alongside target effects.

After identifying reproducible head effects, inspect persistent slots and contextual retrieval positions. Separate routing interventions from value interventions. Retrieval weights alone do not establish storage, and non-additive behavioral effects alone do not uniquely identify a mechanism.

### Evidence criteria and deliverables

Evidence for stable lexical knowledge should combine:

1. Associations recurring across diverse neutral contexts.
2. Generalization across different predictions involving the same property.
3. Causal interventions weakening those associations across contexts with limited unrelated impairment.
4. Entity-specific contextual overrides, with defaults preserved in fresh contexts.

These establish functional stability and flexibility. Whether supporting parameters are concentrated or distributed remains a separate localization question.

Begin with approximately 20–40 candidate nouns, several candidate properties per noun, and 8–10 story templates. These are pilot targets, not a power calculation. Retain independently confirmed baseline associations for the override analysis.

First deliverable: a behavioral plot of default strength, contextual override, and entity specificity, including uncertainty and failure examples. If these effects are reliable, proceed to layer/head causal maps and cumulative intervention curves before examining thousands of memory slots.

### Executable pilot

The concrete plan and implementation are in [experiments/lexicon_pilot](../experiments/lexicon_pilot/README.md), with a [prespecified design](../experiments/lexicon_pilot/PLAN.md). The initial run uses 20 animal nouns, one candidate food association per noun, eight templates, four conditions, and two presentation variants. It tests 1,280 prompts against the 23,000-step checkpoint. This narrower behavioral pilot precedes internal interventions.

### Initial pilot completed

The [October 15 pilot results](../experiments/lexicon_pilot/RESULTS.md) report 1,280 evaluated prompts. Seven proposed noun–food associations passed discovery screening and six confirmed on held-out templates. Contextual preferences had a larger effect when assigned to the queried animal, but information about its companion also produced a substantial shift. Strong presentation-order effects warrant a more controlled behavioral follow-up before localizing memory mechanisms. The report includes exact stimuli, scores, figures, and reproduction instructions.

### Mechanism probing using successful examples

The [mechanism-probing plan](../experiments/lexicon_pilot/MECHANISM_PLAN.md) starts from a successful squirrel example. It proposes bidirectional activation swaps at 24 layer/module sites between stories assigning the same preference to the squirrel or monkey, followed by contextual-override tests, head-level interventions, and comparisons with matched failures. It includes controls and separate mechanistic discovery and validation examples. No activation interventions have been run.
