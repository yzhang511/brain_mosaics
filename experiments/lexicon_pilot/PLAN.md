# Lexicon behavioral pilot

Run: babistories-12L768d. Plan recorded before checkpoint evaluation.

Test whether learned noun-related food associations generalize across neutral story frames, whether explicit preferences override them, and whether the update is specific to the named entity. This pilot concerns food associations and story facts, not all lexical meaning or a localized knowledge store.

Use 20 animal nouns arranged in 10 pairs. Within each pair reverse the same two candidate foods, controlling general candidate preference. Each noun has one candidate association, eight continuation templates (four discovery, four held out), four conditions (neutral, consistent, conflicting, other entity), and two presentation variants. Both animals appear in every introduction. Variants reverse introduction order and preference-clause order; this pilot does not independently disentangle those two order effects.

Score complete candidate continuations with GPT-2 tokenization and frozen evaluation-mode weights. D is log probability of the hypothesized default minus log probability of the alternative. Select nouns using discovery neutral scores only: positive mean and positive scores for each of the four templates after averaging variants. Report all nouns and independently report confirmation on held-out templates; do not further select on held-out behavior.

Primary evaluation uses held-out templates for discovery-selected nouns. Report four raw condition means, target override (conflicting minus neutral), other-entity shift (other minus neutral), and entity specificity (conflicting minus other). Negative specificity means a larger alternative shift for target-specific information. Distinguish score shifts from preference reversals. Report pair-balanced neutral scores for all ten pairs as a check against candidate bias.

Bootstrap noun pairs, retaining both members when present, to reflect shared candidate words; templates are a fixed pilot stimulus set. Confidence intervals do not establish generalization to all vocabulary or syntax. No formal power claim or significance threshold. Report failed associations and per-noun results.

Implementation validation: compare batched full-candidate scores against the model's ordinary forward method on representative examples; check batched versus single-example scoring and unchanged checkpoint metadata across loading. Benchmark a small batch before scaling. Use the existing CPU allocation if estimated runtime fits; otherwise request a scheduled GPU job. Expected storage is under 10 MB for stimuli, scores, metadata, and figures; the checkpoint remains read-only.

Deliverables: exact stimuli and scores, checkpoint provenance and settings, screening table, held-out condition plot with bootstrap intervals, and a concise interpretation. Only proceed to internal causal interventions after interpreting this behavioral pilot; those interventions are outside this first run.
