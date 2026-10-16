# Mechanism probing plan

Updated October 15, 2026. Proposed follow-up to [RESULTS.md](RESULTS.md). No activation interventions have been run. Keep the original checkpoint unchanged.

## Questions

Separate three questions: where the learned default association is supported, where context changes its retrieval or expression, and where the final answer is selected. These may involve different sites. Persistent parameters are fixed during inference, but persistent-memory outputs are context-dependent.

## Discovery cases

Use all six held-out four-condition successes in [scores.csv](results/2026-10-15/scores.csv). Frame and variant identifiers are zero-based. Define D = log P(default) - log P(alternative), scoring complete candidate continuations. Contradictory and irrelevant correspond to CSV conditions `conflicting` and `other`.

| Animal | Frame | Variant | Neutral D | Consistent D | Contradictory D | Irrelevant D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cat | 5 | 0 | 0.4230 | 0.0995 | -0.0548 | 1.0056 |
| squirrel | 6 | 1 | 2.0345 | 2.3746 | -1.9144 | 0.3759 |
| squirrel | 7 | 1 | 0.7304 | 2.6859 | -0.7204 | 1.6611 |
| lion | 6 | 0 | 3.0034 | 3.2740 | -0.1744 | 2.0208 |
| dog | 5 | 1 | 0.2494 | 7.0881 | -1.8146 | 0.5446 |
| elephant | 6 | 0 | 3.0989 | 2.7227 | -1.5482 | 2.0289 |

These are six examples spanning five animals, not six independent animal-level replications. Success means the intended candidate beats its paired alternative, not unrestricted generation accuracy or zero irrelevant-context effect. Several margins are small. These inspected examples now serve as mechanistic discovery data; validation requires new stimuli and matched failures.

## Preliminary deliverable: example-guided memory atlas

Before causal patching, inspect persistent weight geometry and contextual/persistent retrieval for all 24 prompts from the six cases. The [atlas experiment](../memory_atlas/README.md) provides linked key/value PCA maps, original-space neighbors, four-condition retrieval overlays, contextual source-position maps, and selected-slot vocabulary alignments. Complete candidate scores and reconstructed memory outputs are numerically validated. This descriptive stage nominates hypotheses; it does not establish causal function or stable storage. The causal deliverable below remains a subsequent experiment.

## First causal deliverable: maps of entity-sensitive context use

Question: at which layer and memory-module output does transferring activity between contradictory and irrelevant contexts change the food preference?

Start with this contrast because the preference statement changes only its subject while food words and the surrounding story remain fixed. For example, squirrel/frame 7/variant 1 introduces a monkey and a squirrel, then asks what the squirrel fills its bowl with. The contradictory statement says the squirrel disliked nuts and loved bananas; the irrelevant statement assigns that preference to the monkey. The preferred answer switches from bananas to nuts.

### Intervention protocol

1. Reproduce all 24 unpatched condition scores in frozen evaluation mode. Verify candidate boundaries, tokenization, complete-continuation likelihoods, and reference-forward agreement.
2. Cache final-prompt-token outputs of `mosaic.h[layer].attn` (contextual memory) and `mosaic.h[layer].mlp` (persistent memory) for contradictory and irrelevant runs at all 12 layers.
3. Replace one module output at a time with its donor-run output, before residual addition. Leave the recipient residual input intact and let downstream computation proceed normally. Test both directions.
4. Run 6 examples x 12 layers x 2 module types x 2 directions = 288 intervention configurations, each scored on both full candidate continuations. Baselines and controls are additional.
5. For multi-token candidates, patch the selected prompt position consistently in every required teacher-forced forward pass, using prompt-only donor activations. Never import donor activity conditioned on a different answer continuation.
6. Run identity patches at every tested site. Add unrelated-example donor controls matched by layer, module, and final-prompt role. Record their identities; interpret them as disruption controls rather than perfectly matched counterfactuals.

Report D_patched, delta D = D_patched - D_recipient, donor-recipient baseline gap, and whether the preferred candidate reverses. Irrelevant-to-contradictory transfer should increase D if it transfers the donor preference; the reverse should decrease D. Raw score changes are primary. Avoid unstable normalized recovery measures and do not rank sites solely by small-margin reversals.

### Required artifacts

- `baselines.csv`: exact case IDs, prompts, candidate strings/token IDs, candidate log probabilities, and all 24 baseline contrasts.
- `patch_scores.csv`: all 288 configurations, donor/recipient conditions, layer, module, patched position, candidate scores, baseline and patched D, delta D, and reversal flags.
- `controls.csv`: identity and unrelated-donor results, reference-scoring discrepancies, and numerical tolerances.
- `entity_binding_maps.png` and `.svg`: six rows of paired 12-layer by 2-module heatmaps, one column per direction. Use a shared delta-D scale and annotate baseline gaps and preference reversals. Aggregate maps are secondary.
- `metadata.json`: checkpoint identity, source hashes or code revision, runtime environment, precision, seeds, patch semantics, and validation status.
- `REPORT.md`: supported findings, case-to-case differences, control behavior, limitations, and a shortlist of sites for follow-up.

Completion requires all six cases, both directions, reference and identity checks, and the report. Set numerical tolerances from reference validation before interpreting effects; investigate discrepancies rather than silently relaxing tolerances. A weak or heterogeneous map is a valid completed result. This deliverable identifies causal contributions to entity-sensitive context use, not stable knowledge storage.

## Stage 2: distinguish routes across all four conditions

At reproducible sites, compare:

| Bidirectional comparison | Question |
| --- | --- |
| Neutral and contradictory | Which activity restores the default or introduces an override? |
| Consistent and contradictory | Which activity distinguishes the stated preference direction? |
| Neutral and consistent | Does supporting context use a different route despite the same answer? |
| Neutral and irrelevant | Does irrelevant information alter processing despite preserving the answer? |

Start at the final prompt position. If effects are weak, expand to subject, preference, and query spans aligned by linguistic role, not raw token index. Test motivated groups of sites when individual patches are weak. A single-site null does not rule out distributed computation.

Narrow reproducible module effects to head outputs before output projection, then to contributions from relevant statement positions. Retrieval plots generate hypotheses; selective interventions test them.

Add new prompts independently varying introduction order and clause order, affirmative-only preferences, and balanced food mentions. The pilot couples the two order changes, and neutral prompts lack explicit food mentions. Neutral-context comparisons alone cannot separate updating from mention, length, and wording effects.

## Stage 3: distinguish storage from retrieval routing

At effective persistent-memory sites, separate interventions on routing and content:

- Routing: insert neutral-run retrieval weights into a contradictory run while retaining the same stored values. Test whether the default returns, and test the reverse transfer.
- Content: hold routing fixed while removing or replacing candidate stored-value contributions. Test neutral defaults across paraphrases and unrelated animal/food contrasts. Use transient interventions without overwriting the checkpoint.

An effective output patch establishes causal use, not storage location. Stable associations may be distributed across values, keys, embeddings, projections, and other parameters. High retrieval weight alone does not identify a slot as a word or fact store. Require selective causal effects across prompts, with general prediction effects monitored as disruption controls.

## Stage 4: test interaction and mediation

For candidate contextual and downstream persistent sites, compare neither patched, contextual only, persistent only, and both patched in the contradictory recipient. Use a common neutral donor for default-versus-override tests; repeat with an irrelevant donor for the better-matched entity contrast.

Measure I = D_both - D_context_only - D_persistent_only + D_neither. This measures nonadditivity on the log-probability-contrast scale; it does not alone establish a causal route.

For mediation, patch the contextual output while clamping downstream persistent retrieval weights to the recipient's original weights. Compare with the same contextual patch allowing routing to change naturally. Attenuation supports mediation through that routing operation, subject to clamp controls and alternative routes. Clamping weights does not clamp every persistent-memory pathway.

| Hypothesis | Evidence sought |
| --- | --- |
| Context redirects persistent retrieval | Contextual patches change retrieval; fixing recipient routing attenuates their effect |
| Context adds a competing answer | A default-supporting causal pathway remains available while a contextual pathway promotes the alternative |
| Context suppresses default retrieval | Default contribution falls under contradiction; restoring retrieval restores default preference |
| Later computation selects between signals | Earlier pathways support both signals; a later selective intervention switches the answer |

Mechanisms may coexist or differ across examples. Do not force a single account.

## Stage 5: retained defaults and independent validation

After contradiction, separately query the individual's preference and the category's usual food, and test a fresh neutral prompt. Validate each question format independently because wording changes the task. Recovery supports context-dependent expression of retained knowledge, not a unique storage site. Combine recovery with interventions; decodability alone does not establish causal use.

Validate discovered components on new controlled examples and matched failures, preferably the same animal/candidates with one controlled wording difference. Test whether successful-run activity repairs failures selectively. Cross-template transfers need positional alignment and controls for unrelated donor information.

Keep same-animal examples grouped when assessing replication. The six selected successes support discovery, not prevalence estimates or broad generalization. Monitor neutral and consistent cases, unrelated contrasts, and general prediction effects for specificity.

## Execution and interpretation

Before running, follow Sherlock instructions, inspect available allocation resources, and benchmark a small representative patch batch before scaling. Save exact provenance and retain small reports and code in the repository; use the project's established scratch layout for large activation artifacts, which must not be the only copy of anything that must be retained.

The initial claim is narrow: a component contributes causally to using a preference as belonging to the queried animal under these conditions. Storage claims require content interventions and specificity tests. Interaction claims require routing and mediation tests. A general circuit or comparison to human language networks requires further evidence.
