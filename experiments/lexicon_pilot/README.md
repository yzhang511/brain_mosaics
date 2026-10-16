# Lexicon pilot

This experiment evaluates food-related lexical defaults and entity-specific contextual updating in `babistories-12L768d`. See [PLAN.md](PLAN.md) for the design recorded before evaluation.

Run only inside an allocation with sufficient resources. The measured CPU pilot uses one thread, approximately 11 minutes of scoring time, and a 1.5 GB checkpoint. Allow additional loading and analysis time. No training or checkpoint modification occurs.

```bash
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/lexicon_pilot/run.py \
  --checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --out "$SCRATCH/memory_mosaics/lexicon_pilot/2026-10-15"

ml purge
ml load viz py-matplotlib/3.8.3_py312
python3 experiments/lexicon_pilot/analyze.py \
  --input "$SCRATCH/memory_mosaics/lexicon_pilot/2026-10-15" \
  --out "$SCRATCH/memory_mosaics/lexicon_pilot/2026-10-15"
```

Add `--benchmark` to `run.py` for a small scoring validation and runtime estimate. The scorer checks full candidate scores against the original model forward method and verifies single-example versus padded-batch agreement. It uses float32, evaluation mode, and GPT-2 BPE, matching the training preparation script.

The fixed design includes 20 nouns in ten pairs, one candidate food association per noun, eight templates, four conditions, and two presentation variants: 1,280 prompts and 2,560 scored continuations. Templates 0–3 are for discovery, 4–7 for evaluation. Discovery selection requires positive neutral scores on all four discovery template means, averaged over the two variants. Held-out confirmation requires a positive neutral mean and at least three positive template means; it is descriptive and never used for reselection.

Outputs include `scores.csv` with exact prompts and candidate token counts, `metadata.json` with checkpoint and code provenance, `screening.csv`, `summary.json`, and condition/per-noun figures. Intervals resample noun pairs, retaining their dependence; they condition on the selected noun set and fixed templates and do not include selection uncertainty. Pair-balanced results on all nouns help diagnose generic candidate preferences. Candidate lengths may differ; their orientation is reversed within each pair rather than normalizing log probabilities by length.

Limits: this is a hand-authored food-preference pilot, not a corpus-wide lexical benchmark. Story facts about individuals are not identical to lexical word senses. The animal characters and preference language may introduce anthropomorphism. Introduction order and preference-clause order vary together. Neutral prompts lack the explicit food mentions present in other conditions; conflicting versus other-entity is the cleaner entity-binding contrast. The experiment establishes behavioral effects, not parameter storage locations or causal mechanisms.

Completed run: [results and interpretation](RESULTS.md), with preserved artifacts in [results/2026-10-15](results/2026-10-15). The presentation-order diagnostic in the analysis was added after the primary results and is labeled post hoc in the report.

Planned follow-up: [mechanism probing using successful examples](MECHANISM_PLAN.md), starting with bidirectional swaps at 24 module sites for the squirrel-versus-monkey preference contrast. This plan has not been executed.
