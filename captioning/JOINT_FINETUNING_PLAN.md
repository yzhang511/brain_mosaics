# Joint ResNet and Memory Mosaics fine-tuning on Alessandro's captioning data

Recorded October 15, 2026. Status: agreed plan for later implementation. No dataset inspection, implementation changes, training jobs, or downloads are authorized by the request to save this document. Resume implementation when the user asks.

## Revised staging, agreed October 15, 2026

Stage 1's dataset audit (below) found that of Alessandro's t20 `image_captioning` sessions, only 6 run an actual image-captioning paradigm: `t20.2026.01.22` (fixed 76-image set, scripted declarative captions) and five sessions on the generic `t20_screens_ol.yaml` graph — `t20.2026.02.11`, `.02.18`, `.02.23`, `.03.02`, `.03.04` — that show real, unmodified COCO images (1,420 trials, 1,419 distinct COCO image ids; manifest in `captioning/alessandro_t20_manifest.json`, category `coco_free_recall`). The four later `*_constrained_image_captioning*` sessions (Sept-Oct 2026) are a different paradigm with no usable images or captions and are out of scope here.

Agreed three-phase path:

1. **Phase A (do first): COCO-10k warm-up fine-tune.** Fine-tune the current `babistories-12L768d` language model, with the same visual-prefix architecture described below, on a larger generic sample (~10,000 images) from the public COCO captions, reusing and scaling the existing `captioning/prepare.py` pilot pipeline. This is a sanity-checked, well-understood captioning baseline before specializing to Alessandro's much smaller, stimulus-specific set. See new Stage 0 below.
2. **Phase B (next): specialize on the real stimulus images.** Continue fine-tuning from the Phase A checkpoint using the 1,420 `coco_free_recall` trials' images and captions (the five t20 sessions above) — the actual images shown to participant t20. Captions for these come from COCO's public annotations (the trial `.mat` files carry no scripted caption), as these are standard COCO images.
3. **Phase C (deferred): brain distillation.** Once a captioning model exists for the exact images participant t20 saw, pair it with the neural recordings from those same five sessions to distill/relate model representations to brain activity. Explicitly future work — no neural data has been touched for this plan; see the Data risk note in Stage 1.

Phases B and C are not elaborated further until Phase A is done; the rest of this document (Stage 1 audit, starting architecture, Stages 2-5) describes the longer-run joint fine-tuning design these phases will eventually converge on.

## Stage 0: COCO-10k warm-up fine-tune (Phase A)

Goal: a solid, general-purpose captioning checkpoint, initialized from `babistories-12L768d`, before any specialization to Alessandro's small and idiosyncratic stimulus set.

- Scale `captioning/prepare.py`'s existing COCO pilot (previously 1,000 images) to roughly 10,000 train images plus a held-out validation slice, keeping its existing manifest/caching/sha256 provenance approach.
- **Exclude the 1,419 COCO image ids referenced by the t20 `coco_free_recall` sessions** (listed in `captioning/alessandro_t20_manifest.json`) from this sampling pool. These must stay unseen by Phase A so Phase B is a genuine specialization step, not a repeat exposure.
- Architecture and initialization follow the "Starting architecture" section below (ResNet-18 through layer2, 128-to-768 adapter, `babistories-12L768d` checkpoint, GPT-2 BPE). Whether ResNet is frozen or jointly trained for this warm-up stage, batch size, learning rates, and epoch budget should come from a short pilot on this 10k sample, not assumed from the 1k pilot's settings.
- Deliverable: a checkpoint plus held-out caption/generation quality results on generic COCO, used as the Phase B starting point. Preserve it separately from the original language-only checkpoint, consistent with the preservation requirement in the Objective section.

### Stage 0 progress / where to pick up (as of October 15, 2026)

**Done and validated:**

- `captioning/model.py`: `CaptionConfig.block_size` (defaults to the old `grid_size**2 + max_caption_tokens` formula; set explicitly to match a pretrained core) and `Captioner.load_pretrained_core(state_dict)` (loads into `self.core`, re-initializes only the BOS row of the tied `wte`/`lm_head` weight). Validated against the real checkpoint at `/scratch/users/yizzhang/memory_mosaics/babistories/out/ckpt.pt` (step 23,000, `n_layer=12, n_head=12, n_embd=768, block_size=512, pmem_size=2688, pmem_count=1, vocab_size=50304, bias=False`): `load_state_dict` reports all keys matched, non-BOS logits are bit-identical across two independent loads. New hermetic unit tests in `captioning/test_model.py` (`test_block_size_override_matches_pretrained_shapes`, `test_load_pretrained_core_reinitializes_bos_row_only`) use small synthetic configs, not the real checkpoint file.
- `captioning/prepare.py`: `load_t20_coco_exclusions()` + `--exclude-t20-manifest` (defaults to `captioning/alessandro_t20_manifest.json`) filters the 1,419 `coco_free_recall` COCO ids out of `choose_records`' sampling pool. Verified against real COCO annotations.
- **User decision (2026-10-15): the already-existing `reports/caption_pilot_10k/manifest.json` (10k/1k/2k train/val/test, from an earlier pre-exclusion run, job 47873053) is fine to keep using as-is**, even though it contains 161 of the 1,419 t20 stimulus images (87 train / 28 val / 46 reserved-test). The exclusion filter above only affects *future* `prepare.py` runs, not this existing manifest/feature cache.
- `captioning/train.py`: added `--n-layer/--n-embd/--n-head/--pmem-size/--block-size/--init-from-checkpoint` (validates checkpoint `model_args` against the requested architecture, ignoring `dropout` which isn't shape-relevant) and `--qualitative-panel-size` (caps the final report's expensive autoregressive-generation sweep to a fixed panel per split/condition instead of the full split; full-split NLL/token-accuracy in the same report is unaffected since `evaluate()` doesn't generate).
- **Shakedown job completed**: `scripts/train_coco10k_babistories_shakedown.sbatch`, Slurm job 48046325, henderj GPU partition, COMPLETED in 9m05s. Used the existing (unfiltered) `reports/caption_pilot_10k/manifest.json` + its cached features, `--init-from-checkpoint` the babistories ckpt, 20 steps, batch-size 16, `--qualitative-panel-size 8`. Results (`$SCRATCH/memory_mosaics/caption_pilot_10k/babistories_shakedown/results.json`): `median_step_seconds=0.411`, `peak_gpu_memory≈20.0GB`. NLL moved 6.36→4.59 over 20 steps; `train/correct` vs `train/shuffled` vs `train/zero_content` NLL were all ≈4.42 (no measurable image-dependence yet — expected at 20 steps with a freshly-initialized adapter, not a red flag).

**Open design issue, now resolved and implemented (2026-10-15):** the shakedown had trained the *entire* model — the brand-new random adapter and all 123M pretrained-core parameters — jointly, with one shared AdamW and `lr=0.001` (carried over from the small from-scratch pilot's default; babistories pretraining itself used `6e-4` with warmup for a model starting from scratch). Risk: an untrained adapter injects noisy gradients into a 23,000-step-converged core from step 0, with no warm-up and no rate separation. This matched the plan's own Stage 3 §1/§3 below, written for the longer-run joint fine-tune but applying equally here.

**Decisions agreed 2026-10-15:**

- **ResNet stays frozen for the entire Stage 0 / Phase A run**, not just during adapter warm-up. Unlike the plan's general Stage 3 design (which eventually unfreezes ResNet for Alessandro's joint fine-tune), the 10k-image COCO warm-up's only job is to produce a solid checkpoint for Phase B — 10k images is small next to ResNet's ~1.28M-image ImageNet pretraining, so unfreezing it here risks degrading pretrained visual features with too little signal to re-adapt them, for no benefit to this stage's purpose. No code change was needed for this: `captioning/vision.py`'s `FrozenEncoder` already calls `requires_grad_(False)` and `eval()` unconditionally, and `train.py` extracts features from it only once in `prepare.py`, before training — ResNet never enters `train.py`'s optimizer or backward pass at all. Only `adapter` and `core` are ever trainable in this pilot's architecture.
- **BOS row during warm-up: Option (a).** The caption-start (BOS) embedding row (inside `core.wte.weight`, the same tensor as the whole pretrained vocabulary) is frozen along with the rest of `core` during adapter warm-up, rather than carving out a per-row gradient mask. It keeps its one-shot random re-init from `load_pretrained_core` until `core` unfreezes, then trains normally with the rest of the core. Simpler than masking `wte.weight.grad` per-row; the deviation from the plan's literal wording ("adapter, image-type embedding, and caption-start embedding" all warm up together) was judged not worth the extra code.

**Implemented in `captioning/train.py`:**

1. `--core-freeze-steps` (default 0, backward compatible): `model.core.requires_grad_(not core_frozen)` is recomputed every training step from `core_frozen = step <= args.core_freeze_steps` (not `torch.no_grad()` around its forward, which would also block gradient flow back to the adapter). Recomputing every step rather than only at a transition means `--resume` derives the correct frozen/unfrozen state automatically from `step`, with no special-casing. `core_frozen` is recorded in each `history` eval entry.
2. `--adapter-lr` / `--core-lr` (both default to `--learning-rate` if unset, so existing invocations are unaffected). Two AdamW param groups: `{adapter.parameters() + [image_type], lr=adapter_lr}` and `{core.parameters(), lr=core_lr}`. No optimizer reconstruction needed when toggling freeze — AdamW already skips params whose `.grad` is `None`. Recorded in `config.json`'s `optimizer` block.

**Next step:** run one more short shakedown (~100-200 steps, same script + new flags, e.g. `--core-freeze-steps 50 --adapter-lr 1e-3 --core-lr 1e-4`) and check whether `train/correct` starts diverging from `train/zero_content` (first sign of real image-dependence) before picking final values for the real multi-thousand-step run and submitting it.

### Shakedown results and follow-up decisions (2026-10-15)

- **200-step shakedown** (job 48052659, `scripts/train_coco10k_core_freeze_shakedown.sbatch`): completed, stable. `train/zero_content` NLL (3.44) was clearly worse than `train/correct`/`train/shuffled` (both ≈3.21-3.22) — first sign of real image-dependence (the model learned "there is an image" but not yet "which image"), versus the original unfrozen-everything shakedown where all three conditions were ≈4.42 (no dependence at all).
- **90-minute shakedown** (job 48056531/48055656, `scripts/train_coco10k_core_freeze_shakedown_90min.sbatch`, `--core-freeze-steps 200 --adapter-lr 1e-3 --core-lr 1e-4 --steps 14000`, W&B run `coco10k-core-freeze-shakedown-90min`): revealed clear overfitting by epoch ~4-6 of the ~22 epochs this run does over the same 10k images — train NLL kept falling (2.72→2.19→1.82 at steps 1400/2800/4200) while val NLL plateaued and ticked up slightly (3.00→2.87→2.88), a widening train/val gap. Expected given 10k distinct images repeated many epochs against a 123M-parameter core.
- **Decision: scale Stage 0's training sample toward full COCO.** Standard captioning practice (Show-and-Tell through BLIP-2/LLaVA-style pipelines) trains on the full COCO train2017 set (~118k images), not a small subsample; this addresses overfitting at the root (sample diversity) rather than only via regularization. `captioning/prepare.py` already supported arbitrary `--train-size`/`--val-size`/`--test-size` with no code change needed. New `scripts/prepare_100k.sbatch`: `--train-size 100000 --val-size 2000 --test-size 2400`, output to a separate `caption_pilot_100k` dataset (keeps the 10k pilot's artifacts intact). Note: val2017 has only 4,878 images available after the t20 exclusion (118 of the 1,419 t20 stimulus images are themselves in val2017, not train2017) — an initial attempt requesting the full 5,000 (job 48063185, `--val-size 2500 --test-size 2500`) failed immediately with no margin for that; `choose_records`' error message in `prepare.py` was also clarified to mention id-based exclusions, not just over-length-caption skips, since that was the actual cause here.
- **Decision: switch the frozen feature tap from ResNet-18 layer2 to layer4** for the 100k dataset. layer2 (128-dim, texture/edge-level, native 28x28) was an unverified carryover assumption flagged in the "Information required before implementation" section below (Alessandro's earlier backbone/tap choice was never established); layer4 (512-dim, more object/semantic-level, native 7x7) matches standard captioning practice (Show-Attend-Tell and most ResNet-based captioners condition on the last conv stage) and may help the model learn to distinguish `correct` from `shuffled`, not just "image present vs. absent," which the 10k-image runs never achieved. Implemented via `captioning/vision.py`'s `FrozenEncoder`/`TAPS` registry (`layer='layer2'|'layer4'`, configurable `pool_size`; default unchanged, so existing 10k-pilot behavior and `test_model.py`'s `test_preprocess_and_frozen_encoder` are unaffected) and `captioning/prepare.py`'s new `--feature-layer`/`--feature-pool-size` flags. `captioning/train.py` gained matching `--grid-size`/`--feature-dim` flags (default 14/128, unchanged) so `CaptionConfig` can be pointed at layer4-shaped features (`--grid-size 7 --feature-dim 512`) for the eventual real run on `caption_pilot_100k`.
- Sequencing note: both the data-scale and feature-tap changes were made together before re-running the longer shakedown, so if the next long run's `correct`-vs-`shuffled` separation is still weak, it won't be possible to attribute that cleanly to one change or the other without an ablation.

## Objective and agreed scope

Use a pretrained ResNet to extract spatial visual features, connect them to the pretrained Memory Mosaics language model, and jointly fine-tune both models on Alessandro's image-caption pairs. The primary training condition updates ResNet, the visual adapter, and the language model from caption supervision. A frozen ResNet is an optional warm-up or comparison, not the final intended setup.

Preserve the original language checkpoint and record new checkpoints separately. The earlier COCO pilots provide reusable code and debugging experience; Alessandro's data, not the COCO subsets, is the intended training dataset.

## Starting architecture

```text
Image
  -> pretrained ResNet spatial feature map
  -> trainable visual adapter + spatial positions + image-type embedding
  -> visual prefix followed by caption input tokens
  -> pretrained Memory Mosaics
  -> next-caption-token predictions
```

Starting proposal, to confirm against Alessandro's previous setup:

- ResNet-18 with ImageNet pretrained weights, through layer2, retaining the existing 128 x 14 x 14 pooled feature map. If Alessandro's earlier experiment used another backbone or feature layer, establish that configuration before deciding whether to change it. Layers beyond the selected feature tap are not part of the active encoder and do not receive caption gradients.
- Flatten the spatial map into 196 vectors in raster order. Use a trainable 128-to-768 adapter, fixed 2D spatial encodings, and a trainable image-type embedding.
- Initialize the language core from the 23,000-step `babistories-12L768d` checkpoint: 12 layers, 768 dimensions, 12 heads, one persistent bank, and 2,688 slots per head. Verify the checkpoint's actual metadata when resuming.
- Preserve GPT-2 BPE tokenization and tied input/output embeddings. Explicitly initialize the caption-start embedding because its reserved BOS ID was not meaningfully trained as a caption boundary. Do not replace the pretrained vocabulary with a dataset-specific word vocabulary.
- Retain the pretrained 512-position context capacity. The proposed 196 visual tokens plus 64 caption positions fit; audit actual caption lengths before setting a final limit. Do not silently truncate captions.
- Begin with the established RGB, full-frame 224 x 224 resize and ImageNet normalization unless dataset provenance or the chosen pretrained weights require a different transformation. Document aspect-ratio distortion. Add augmentations only when they preserve caption truth; flips and crops can invalidate spatial or object descriptions.

Reuse `captioning/model.py`, `captioning/vision.py`, `captioning/data.py`, and `captioning/train.py` where appropriate, but do not assume their current defaults load the larger pretrained core. The current `FrozenEncoder` deliberately disables gradients and forces evaluation mode, so it needs an explicit trainable counterpart or mode rather than simply being placed in an optimizer. Check persistent-buffer shapes when transferring a 512-position checkpoint into the caption wrapper.

## Information required before implementation

No reference to Alessandro's dataset was found in the project files inspected for this plan. Obtain:

- Dataset location and the user's authorized access route.
- A non-sensitive description of the images and the approved risk classification. Sherlock allows Low or Moderate Risk data; establish classification before inspecting potentially sensitive images or annotations.
- Annotation format, image identifier mapping, and whether images have one or multiple reference captions.
- Existing train/validation/test splits, related-image or stimulus-family identifiers, and any previous evaluation exclusions.
- Alessandro's earlier ResNet backbone, feature tap, preprocessing, and training configuration if available.

Do not request a new approval for routine implementation steps already authorized when the user resumes. Dataset details above are missing information, not a request to change the agreed objective.

## Stage 1: dataset audit and reproducible splits

Status: initial audit completed October 15, 2026 for the t20 `image_captioning` sessions; see `captioning/alessandro_audit.py` (script, reads only non-neural `.mat` fields) and `captioning/alessandro_t20_manifest.json` (3,252 trial-level records) for the category breakdown summarized in "Revised staging" above. Splits, caption-length distributions and a qualitative panel for the `coco_free_recall` and `templated_declarative` subsets are not yet built; that is remaining Stage 1 work once Phase A/B sampling choices are finalized.

Validate image-caption pairing, missing or corrupt files, duplicate images, caption distributions and token lengths. Respect existing splits; if no split exists, define and record one before training. Keep near-duplicates, image variants and related stimulus families in the same split. Reserve a test split for final evaluation.

Preserve a manifest with image IDs, caption IDs, grouping rules, split assignments, preprocessing and provenance. Choose a fixed qualitative validation panel in advance. For multiple human captions, sample references during training and use a documented multi-reference evaluation protocol. Adapt the batching and epoch schedule to the actual dataset size rather than inheriting the old COCO settings blindly.

## Stage 2: pretrained integration and numerical checks

Load the language core with an explicit inventory of loaded, new and intentionally regenerated parameters/buffers. Initialize the visual interface and caption-start embedding, then check the scale of visual-prefix embeddings relative to pretrained text embeddings. If needed, introduce a documented normalization or learned scale; avoid an uncontrolled input-scale mismatch from the adapter and spatial encodings.

Use teacher forcing with loss only on caption targets, including EOS. Ignore padding and apply no target loss to visual positions. Keep the upstream contextual mask and shifted-value construction unless a separately justified change is required. Verify no future caption tokens leak through the visual-caption boundary.

Required checks:

- Original text-only predictions are reproduced before fine-tuning when bypassing the visual wrapper.
- Forward/backward passes, masked caption loss, full-versus-truncated prefix agreement, and generation stopping are correct.
- Caption gradients reach the adapter, active ResNet layers, and language parameters in joint mode.
- A tiny optimizer step changes intended trainable parameters while leaving intentionally frozen parameters and buffers unchanged.
- Validation and generation restore the intended training modes; saved checkpoints reload reproducibly.

## Stage 3: adapter warm-up followed by joint fine-tuning

Proposed initialization schedule:

1. Briefly freeze ResNet and the language core and train the new visual adapter, image-type embedding and caption-start embedding. Preserve gradient flow through the frozen language model to the adapter; do not wrap that forward pass in `no_grad`. If only the BOS embedding row is trainable, ensure optimizer weight decay cannot alter the other frozen rows.
2. Unfreeze both the active ResNet encoder and the language core, including persistent keys and values, for the main joint-training condition.
3. Use separate optimizer groups: the new adapter initially receives the highest learning rate, pretrained language parameters a lower rate, and ResNet initially the lowest. Select actual rates, batch size, warm-up duration and training budget from a short stability/runtime pilot, not the old scratch-trained captioner's defaults.
4. For small batches, initially freeze ResNet batch-normalization running statistics while training its convolutional weights. State explicitly whether normalization affine parameters are trainable. Other training/evaluation modes must remain deliberate.
5. Use gradient clipping, validation monitoring and recoverable checkpoints. Choose checkpoint-selection and stopping criteria before the main run. Log trainable parameter groups and actual learning rates.

Train from images during joint fine-tuning. Fixed cached ResNet features become stale as encoder weights update and cannot propagate gradients into ResNet. Existing feature caches are suitable only for a matching frozen-encoder stage or comparison.

Benchmark a representative batch inside a suitable compute allocation before estimating the full budget. Follow current Sherlock resource discovery and scheduling guidance; do not guess GPU hardware or queue limits. Preserve code, manifests and required checkpoints outside purgeable storage, using the established project storage layout.

## Stage 4: behavioral evaluation

Primary questions: does the model describe held-out images accurately, and does it actually use the image?

- Score complete held-out caption likelihoods with the shared tokenizer and inspect free generations under a fixed decoding protocol.
- Evaluate correct images, fixed shuffled-image assignments and zero-content visual controls. For zero-content tests, define the intervention explicitly (for example, zero the current feature map immediately before the adapter); a black image is a different intervention because ResNet can produce nonzero features from it.
- Evaluate object, attribute and relationship correctness as appropriate for Alessandro's stimuli. Track hallucinated content and whether captions change appropriately when images change.
- Use suitable multi-reference caption metrics if supported by the annotation structure, alongside a fixed reviewed sample. Exact-reference matching is a debugging measure, not the sole quality criterion.
- Report paired correct-versus-shuffled differences, with uncertainty respecting image/stimulus grouping when sample size permits. A correct-image advantage is evidence of image dependence, not by itself proof of semantic generalization.
- Keep reserved test images untouched during development.

The earlier 1,000-image COCO pilot showed little held-out advantage for correct over shuffled images. This motivates image-dependence checks but is not a direct performance baseline: historical vocabulary, initialization and data differ.

## Stage 5: memory preservation and adaptation

Save checkpoints before adaptation, after adapter warm-up, and during/after joint fine-tuning. At these checkpoints:

- Rerun the six lexical examples, the broader original lexical conditions, and a fixed held-out text evaluation. The six selected successes alone cannot establish language preservation.
- Compare text-only persistent retrieval on identical prompts to measure drift.
- Separately measure changes in persistent keys/values and changes in retrieval distributions. Similar slot indices do not guarantee unchanged function after training, so examine parameter and functional drift together.
- During captioning, compare correct versus shuffled images under the same teacher-forced caption prefix at selected prediction positions. Different freely generated prefixes would confound image-driven retrieval differences.
- Distinguish changes in visual features, adapter outputs, retrieval weights and retrieved output vectors. Keep descriptive comparisons separate from causal conclusions.

Optional subsequent controls include a frozen-ResNet branch and a frozen-persistent-bank branch, using common initialization and comparable data exposure. Freezing persistent keys/values does not freeze all knowledge: other projections and query extractors can change. Do not selectively train or freeze layer 7/head 8 merely because it stood out in six lexical examples.

## First implementation deliverable

A small, reproducible end-to-end pilot on Alessandro's dataset containing:

1. Dataset audit, manifest, split policy and recorded preprocessing.
2. Validated transfer of both pretrained models into the captioning architecture.
3. Adapter warm-up and a bounded joint-training run demonstrating updates to both the active ResNet encoder and Memory Mosaics.
4. Held-out caption results and correct/shuffled/zero-content comparisons, with a fixed qualitative panel.
5. Text-baseline checks, training curves, provenance, saved checkpoints and a report recommending whether to scale or revise the interface.

A tiny overfit check is a diagnostic within this pilot, not evidence of held-out success. The deliverable remains valid if it exposes weak grounding or forgetting; those findings determine the next experiment. No implementation or training is part of the present documentation-only task.
