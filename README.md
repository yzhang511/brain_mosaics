# Memory Mosaics


Memory Mosaics are networks of associative memories working in concert to achieve a prediction task of interest. Like transformers, memory mosaics possess compositional capabilities and in-context learning capabilities. Unlike transformers, memory mosaics achieve these capabilities in comparatively transparent ways. We demonstrate these capabilities on toy examples and we also show that memory mosaics perform as well or better than transformers on medium-scale language modeling tasks.

**Latest News**
- [2025.09] [Memory Mosaics at scale](https://proceedings.neurips.cc/paper_files/paper/2025/file/3767842e2ebee900236f534855aa2c36-Paper-Conference.pdf) scales Memory Mosaics to 10B parameters and 1T training tokens, accepted by NeurIPS 2025 **Oral (top 0.36%)**.
- [2025.01] [Memory Mosaics](https://proceedings.iclr.cc/paper_files/paper/2025/file/59c3ac496e6fe7be0c2c2b95014e2210-Paper-Conference.pdf) is accepted by ICLR 2025.



**License**: This code is released under the [Apache-2.0 License](LICENSE.txt).



The following folders contain dataset, code, and instructions for paper [Memory Mosaics](https://proceedings.iclr.cc/paper_files/paper/2025/file/59c3ac496e6fe7be0c2c2b95014e2210-Paper-Conference.pdf).

* [`BabiStories`](BabiStories) : the BabiStories dataset
* [`ThreeMoons`](ThreeMoons) : The [mlx](https://github.com/ml-explore/mlx) code for the three moon experiments
* [`ICLL`](ICLL): Memory Mosaics code for the In-context Language Learning experiments
* [`Library`](Library): A [pytorch](https://pytorch.org/) library of efficient Memory Mosaics implementation for language tasks. 
* [`nanoMosaics`](nanoMosaics): An alternate [pytorch](https://pytorch.org/) implementation derived from nanoGPT.



## Captioning project: understanding and intended scope

This project will build a compact image-captioning model as a first task-driven
model of language production using the official Memory Mosaics repository.
The immediate goal is a working, interpretable behavioral baseline that learns
representations useful for comprehending image content (objects, attributes, and
relationships) and expressing it in language. Memory Mosaics is intended to do
semantic processing as well as sentence production. Caption supervision does not
by itself enforce a separate meaning representation or a strict understand-then-
speak sequence; those are hypotheses to test. The longer-term goal is to test how
visual information becomes a linguistic description and whether internal model
states explain human neural activity.

**Status:** Initial captioner, reproducible pilot data pipeline, and a 32-image
overfit experiment are complete. Held-out captioning remains poor; scaling and
semantic generalization are untested. Upstream documentation is retained above.

### Implementation update: agreed preprocessing and first code

**Preprocessing decision:** Direct full-frame resize to 224 x 224, bilinear
interpolation with antialiasing, RGB conversion, and ImageNet mean/std
normalization. This retains all image content but distorts aspect ratios. Padding
is deferred to a controlled comparison. This decision supersedes the open
preprocessing review points below.

The first implementation is in `captioning/model.py` and `captioning/vision.py`.
It uses frozen ResNet-18 ImageNet V1 through layer2, pooled to 128 x 14 x 14,
a trainable adapter, fixed sinusoidal row/column positions, and an image-type
embedding. Defaults are 4 blocks, dimension 256, 4 heads, one persistent bank
with 256 slots per head per block, tied token/output weights, and 64 caption
input positions including BOS.

Inputs are cached features shaped `[batch, 128, 14, 14]` and caption token IDs
`[BOS, words]`; targets are `[words, EOS]`. Only caption positions receive loss.
Right-padding targets use PAD (default 0) or -1 and are ignored. PAD/BOS/EOS
(default 0/1/2) must be distinct; callers supply a training-only vocabulary.
The encoder stays in eval mode even when its parent enters training mode.

The upstream strict contextual mask and next-position value construction are
preserved. Visual tokens have causal raster-order context. The final visual
value may include BOS, and a caption query may retrieve values containing its
own input token, but cannot retrieve future caption tokens. Context is rebuilt
on every call; persistent banks are learned parameters shared across examples.
Greedy generation recomputes the full prefix and stops per example at EOS or
the configured length limit. Optional `return_states=True` returns final hidden
states for all visual and caption positions; detailed memory traces remain pending.

Run checks within an allocation:

```bash
ml load math py-torchvision/0.19.1_py312
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m unittest captioning.test_model -v
```

Checks cover future-token invariance, full/truncated agreement, caption-only
padding-masked loss, example independence, generation mode restoration, frozen
batch normalization, and a forward/backward pass at the default model size.
All five checks passed on the current CPU allocation.
Encoder tests use random weights to avoid downloading; they do not establish
pretrained-feature quality or captioning performance.

**Pilot completed:** The 32-image run stopped at step 500 with training NLL
0.01956, teacher-forced token accuracy 99.5%, and 30/32 exact free generations.
Correct/shuffled/zero-content training NLLs were 0.0196/0.7945/0.6886; exact
matches were 30/0/1 out of 32. This supports image dependence on memorized examples.

Held-out results are poor: correct-image validation NLL 7.1247, token accuracy
17%, and 0/8 exact generations. Qualitative review found unrelated scene content
in all eight outputs. The tiny training-only vocabulary maps 27.2% of validation
word/punctuation tokens to UNK, limiting interpretation. No reserved test images
were evaluated. Median CPU step time was 0.297 seconds; training took 156.5 seconds
excluding final generation/evaluation. No GPU measurement was made.

See the [W&B run](https://wandb.ai/yizzhang-stanford-university/memory-mosaics/runs/1crzqxsn),
[pilot report](reports/caption_pilot/run01/index.html),
[learning curve](reports/caption_pilot/run01/learning_curve.png),
[results](reports/caption_pilot/run01/results.json), and
[qualitative review](reports/caption_pilot/run01/qualitative_review.json).
The final checkpoint and exact run source snapshot are retained under
`reports/caption_pilot`, outside purgeable scratch. Reloading the retained
checkpoint reproduced the reported training NLL. Seven model/data tests passed.

**1k pilot completed:** The cache contains 1,000 train2017 images, 100 fixed
val2017 images, and 500 untouched reserved test IDs. The training-only vocabulary
has 3,403 entries; the longest selected caption uses 18 tokens. At 300 CPU steps,
training NLL was 2.772 with 45.5% token accuracy, while validation NLL was 4.144
with 33.6% token accuracy. Correct/shuffled/zero-content validation NLLs were
4.144/4.157/4.191, and all 100 free generations missed exact caption match.
This is a short sanity run, not a finished scaling experiment. Validation accuracy
is rising, but the nearly identical image interventions do not yet show clear
causal image use at this checkpoint. The run used about 0.30 CPU seconds per step
for the training loop and is logged as [coco1k-sanity in W&B](https://wandb.ai/yizzhang-stanford-university/memory-mosaics/runs/a4n6y41y).

See the [1k report](reports/caption_pilot_1k/run01/index.html), [results](reports/caption_pilot_1k/run01/results.json), and [manifest](reports/caption_pilot_1k/manifest.json). The feature tensor remains in scratch as a reproducible cache; the retained manifest records its provenance. Keep reserved test IDs untouched. Detailed memory traces and GPU benchmarks remain pending.

### Reproduce the 32-image COCO pilot

The pilot uses 32 train2017 images and 8 val2017 images, selected deterministically
with seed 20261014. A further 128 val2017 IDs are reserved without evaluation;
the four inspection IDs are excluded. The retained manifest records image and
weight hashes, fixed caption IDs, splits, preprocessing and software versions.
A lowercase word/punctuation vocabulary is fitted to all captions of the 32
training images only; this pilot vocabulary must be revisited before subset-scale
comparisons. Each image's lowest-ID caption is used for the overfit experiment.
Training samples images uniformly with replacement. No caption is truncated.

Within a CPU allocation, from the repository root:

```bash
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/caption_pilot/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/caption_pilot/wandb-config"
python3 -m captioning.prepare \
  --work-dir "$SCRATCH/memory_mosaics/caption_pilot" \
  --manifest-dir reports/caption_pilot
python3 -m captioning.train \
  --manifest reports/caption_pilot/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot/run01" \
  --steps 1600 --eval-every 100 --wandb-mode online
```

W&B project defaults to `memory-mosaics`; set `--wandb-entity` and
`--wandb-project` to select a destination. Use `--wandb-mode offline` for local
W&B recording or `disabled` to omit W&B. JSON results and checkpoints are saved
regardless. Each trainer invocation creates a new W&B run, including resumes;
`--resume PATH` restores model, optimizer and random state, while `--steps` is
the total target step count. CPU runs do not measure GPU timing or memory.

Training uses AdamW (learning rate 0.001, weight decay 0.01), batches of four,
gradient clipping at norm 1, and an early-stop training NLL target of 0.03.
Validation does not select checkpoints. Evaluation compares correct images,
a fixed split-wide feature rotation, and zero raw image features with the
same sequence layout. The latter keeps adapter bias and position/type embeddings.
A zero-content intervention is not a separately trained language-only baseline.

Source files, manifests, final checkpoints and small reports should be retained
outside scratch. Downloads and feature tensors are reproducible disposable caches.
A 32-image memorization result alone is not evidence of held-out semantic competence.

### Historical small implementation plan (superseded by results above)

**Starting configuration:** frozen ImageNet ResNet-18 through `layer2`, average
pooling from 128 x 28 x 28 to 128 x 14 x 14, then 196 visual prefix tokens.
Use a trainable 128-to-256 adapter, explicit 2D positions, and an image-type
embedding. Start from `nanoMosaics` with 4 blocks, dimension 256, 4 heads, one
persistent bank with 256 slots per head per block, and tied token/output weights.
Train the adapter and caption generator from scratch; keep ResNet in eval mode.

1. **Settle preprocessing before caching.** Review the existing
   [feature gallery](reports/feature_inspection/index.html). Choose how to preserve
   the full scene at 224 x 224: direct resizing distorts geometry; aspect-preserving
   padding introduces borders. Standard center cropping removed relevant objects
   in our examples. Record the choice and ImageNet normalization.
2. **Prepare a tiny COCO pilot.** No training dataset is downloaded yet; only four
   inspection images exist. Obtain captions and 32-128 training images for a fixed-
   caption overfit check. Record image IDs/seeds and keep validation/test images
   disjoint. Reserve inspection IDs 139, 39769, 252219, and 397133 for development,
   never the final held-out test set. Cache frozen features before the adapter only
   after preprocessing is settled.
3. **Implement the visual-prefix captioner.** Feed `[196 image tokens, BOS, words]`;
   predict the next caption token and EOS, with no image-position or padding loss.
   Keep the strict contextual retrieval mask paired with next-position value
   construction. Reset context per example; initially recompute the complete prefix
   during generation. Verify future-token invariance and full/truncated agreement.
4. **Run the tiny overfit experiment.** Save config, checkpoint, loss curve, free
   generations, actual step time, and peak GPU memory. Use an appropriate Slurm
   allocation for training; discover available resources before requesting them.
5. **Then test image use and scale.** Compare correct, shuffled, and zero-content
   image prefixes on held-out images. Move to nested 1k/10k subsets only after the
   pipeline works. Log selected visual-prefix and caption states for later probes.

**Scientific interpretation:** Memory Mosaics is a candidate model of semantic
processing and language production, not an established anatomical mapping. Do not
assume semantics is complete in the CNN or starts only in Memory Mosaics. Measure
object, attribute, and relationship information across stages, including states
before caption words are provided. First follow-up: `layer3` with the same 196-token
interface; a jointly trained CNN from scratch is a later comparison.

**Historical checklist:** The tiny pilot has now been completed. See the current
results and next steps above; the following sections preserve the original plan.

### Initial architecture

```text
Image
  -> frozen ImageNet-pretrained ResNet-18 layer2 spatial features
  -> spatial pooling to 14 x 14 = 196 visual prefix tokens
  -> explicit 2D spatial-position and image-type information
  -> trainable adapter to the Memory Mosaic hidden dimension
  -> Memory Mosaic caption generator
  -> autoregressive sentence
```

Inspect example features and choose the visual input with the user before bulk
feature caching. Then cache frozen features for efficient training. Train the visual adapter, Memory
Mosaic core, token embeddings, and vocabulary output head. Train the language
component from scratch. Begin with one captioning task on COCO Captions subsets;
neural supervision comes after the behavioral baseline.

The candidate `layer2` choice is an engineering starting point, not an anatomical
claim. Inspect `layer1`, `layer2`, and `layer3` now, before committing to the input.
A controlled training comparison is needed to establish how the boundary affects
learning; attractive activation plots cannot determine the best layer. ImageNet supervision already introduces category information, so we
cannot attribute all semantics to Memory Mosaics. Associative memories provide
candidate retrieval and specialization mechanisms; syntax, lexical, or semantic
specialization must be measured rather than assumed.

### Image conditioning and causality

**Decision:** Start with a visual prefix of 196 tokens, using `nanoMosaics`.
Separate visual memory is deferred. Start with `layer2`; image preprocessing
remains open pending review of the feature inspection. The options considered were:

| Option | Information flow | Tradeoff |
| --- | --- | --- |
| Visual prefix | Projected image tokens precede caption tokens. | Simple interface, with careful masking and memory handling. |
| Separate visual memory | Caption states retrieve image features through a dedicated interface. | Separates visual access from linguistic context but adds machinery. |

Document the selected information flow. Future caption tokens must never influence
current predictions, including through the paper's value-construction mechanism;
an attention mask alone is not proof of causality. Reset contextual caption memory
between independent examples and explain how the current image remains accessible.
Verify this behavior in teacher-forced training and free generation.

### Data and training objective

Create reproducible nested training subsets: initially 1,000 images, then 10,000,
then larger subsets up to the full training split. These are pilot sizes, not
claims about sufficient data. Split by image, keeping all captions of each image
together. Keep validation and test images fixed across subset experiments. Never
select test examples based on model performance.

Initially sample one human caption per image per epoch. Record subset IDs, seeds,
preprocessing, and caption sampling. Save tokenizer, vocabulary, special-token,
sequence-length, and decoding choices in experiment configurations.

Use teacher forcing to minimize autoregressive caption negative log-likelihood:

```text
L_caption = -sum_t log p_theta(w_t | w_<t, E(I))
```

Evaluate with free generation. The provisional decoder configuration and remaining decisions are recorded below;
optimization settings must also be saved before experiments.

### Validation and success criteria

1. Overfit a tiny training set.
2. Demonstrate improvement on held-out images.
3. Show appropriate prediction changes when the image changes.
4. Perform better with correct image features than shuffled or absent features.
5. Produce descriptions with correct objects and relationships.

Document consistent correct-image, shuffled-image, and image-free comparison
protocols. Report caption metrics alongside inspected examples and explicit error
categories: wrong objects, reversed roles, unsupported details, and grammatical
errors. Fluent generic captions alone are insufficient. Benchmark actual step
time and GPU memory use before estimating full training cost.

### Inspectable computation

Expose optional logging of memory keys, retrieved outputs, retrieval weights,
layer states at caption positions, and contributions of visual information versus
caption context. Document how those contributions are measured in the chosen
architecture. Use selected evaluation examples; do not store every activation
during routine training.

These interfaces will later support tests of object identity, relationships,
grammatical information, cross-example generalization, and memory ablations.
Comparisons with controlled human behavior and neural recordings follow the
behavioral baseline and are not part of the initial training objective.

### Expected implementation deliverables

- Documented architecture and image-conditioning implementation, including
  information flow, causal safeguards, and memory-reset behavior.
- Reproducible feature extraction, subset creation, training, and evaluation.
- Saved configurations, checkpoints, learning curves, and sample captions.
- Correct-image, shuffled-image, and image-free comparisons.
- Runtime and GPU-memory measurements.
- A short account of assumptions, limitations, and implementation changes.

**Phase-one success is a compact captioner that demonstrably uses image information
and exposes inspectable memory computations.** Claims about emergent specialization
and neural explanation require subsequent experiments.

### Visual feature inspection: available for review

The [feature inspection gallery](reports/feature_inspection/index.html) contains
four public COCO examples, comparing layer1/layer2/layer3, native versus 14 x 14
features, and full-frame resizing versus the pretrained model's center crop.
Start with [the preprocessing comparison](reports/feature_inspection/preprocessing.png),
then inspect the [cats example](reports/feature_inspection/000000039769_full_frame.png)
and [street example](reports/feature_inspection/000000252219_full_frame.png).
Individual-channel panels and center-crop versions are available in the gallery.

In the street example, the center crop removes peripheral pedestrians. In the cats
example, it removes parts of the scene near the image edges. Direct square resizing
preserves the frame but distorts geometry. These observations motivate an explicit
preprocessing decision; neither option has been selected for training. Layer3 is
already 14 x 14; layer1 and layer2 require pooling to reach 196 tokens. This does
not imply layer3 has the best semantic representation.

Interpretation: activation RMS measures magnitude, not importance. PCA colors show
three directions of feature variation, not semantic classes; a common PCA basis
is fitted across these examples and preprocessing modes separately for each layer.
The three displayed components explain approximately 28.8%, 18.7%, and 12.4% of
variance for layer1, layer2, and layer3 respectively on this inspection set;
indistinct PCA colors do not imply that semantic information is absent.
The similarity panel compares pooled vectors to a fixed marked token (row 7,
column 7), not to an annotated object. The marker denotes an approximate grid
location, not the extent of its receptive field. Channels are the four with the
largest mean activation across all four full-frame images, without manual concept
selection. Similar-looking regions may share texture or color rather than meaning.
No plot establishes recognition, relational understanding, or a complete semantic
representation. Compare encoder and downstream states with held-out semantic probes
and interventions later.

The [inspection script](scripts/inspect_resnet_features.py) saves plots and a
[metadata record](reports/feature_inspection/metadata.json) with image IDs, source
URLs, SHA-256 checksums, weight identity, versions, PCA basis, and selection rules.
These four val2017 images are development inspection examples and must be excluded
from the eventual held-out test set. No dataset-wide feature cache or captioner
has been implemented. Downloaded images and weights are disposable scratch inputs;
the small report and reproducible script are retained here.

To reproduce on Sherlock in an allocated job (one CPU is sufficient for this report):

```bash
ml load math py-torchvision/0.19.1_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$SCRATCH/memory_mosaics/inspection/matplotlib"
python3 scripts/inspect_resnet_features.py \
  --cache-dir "$SCRATCH/memory_mosaics/inspection/downloads" \
  --output-dir reports/feature_inspection
```

The script uses verified HTTPS downloads from the public COCO S3 bucket and
PyTorch's official weight host. It runs frozen ResNet in evaluation/inference mode
on CPU; recorded forward timings are inspection timings, not training benchmarks.
The next step is to settle preprocessing with the user, starting with `layer2`.

### Implementation plan and review point (2026-10-14)

1. **Inspect visual inputs first.** Download only a few public COCO examples,
   record IDs and sources, and designate them development/inspection images,
   excluded from the eventual held-out test set. Run frozen ImageNet ResNet-18
   on CPU in evaluation mode; do not build a dataset-wide feature cache yet.
   Compare layer1 (64 x 56 x 56), layer2 (128 x 28 x 28), and layer3
   (256 x 14 x 14) for 224 x 224 input. Compare native maps with spatial average
   pooling to 14 x 14. Show the original image and actual network input, including
   center-crop versus full-frame resize, so missing or distorted objects are visible.
2. **Produce interpretable inspection figures.** Show activation RMS, selected
   individual channels with a stated selection rule, and cosine similarity to
   marked spatial feature vectors. Use shared PCA projections across the inspection
   images within each layer as an optional view of feature variation. Explain all
   scaling and selection choices. These are feature diagnostics, not object masks,
   attention maps, semantic labels, or proof of understanding. Do not infer that a
   channel is a named concept from one example. The spatial locations have overlapping
   receptive fields, not independent 16 x 16 image patches.
3. **Review with the user.** The current starting layer is `layer2`; settle
   preprocessing after reviewing the figures. Keep 196 tokens fixed. Do not begin bulk caching or captioner
   implementation before this review. Later probes and controlled caption experiments
   must test object identity, attributes, relationships, and generalization; plots
   alone cannot establish whether the representation contains usable semantics.
4. **Implement the agreed prefix captioner.** Project each selected feature vector
   to dimension 256; add fixed 2D position encoding and a trainable image-type
   embedding. Concatenate the row-major image prefix with BOS and caption embeddings.
   Start with 4 decoder blocks, 4 heads, one persistent bank with 256 slots per head
   per block, tied token/output weights, and 64 caption positions. Retain existing
   contextual and persistent memory operations and causal key accumulation.
5. **Verify information flow before training.** Use caption-only next-token loss:
   `[image tokens, BOS, a, dog]` predicts `[ignored, a, dog, EOS]`. Mask padding loss.
   Preserve the strict contextual retrieval mask because values use the next
   position. The last visual value can include BOS; record this boundary behavior.
   Keep the image prefix causal, documenting its raster-order bias. Test future-token
   perturbations, agreement between full and truncated inputs in evaluation mode,
   loss masking, and independence of examples. Start generation by recomputing the
   full prefix plus caption; defer incremental caches.
6. **Build reproducible data/cache pipelines after input selection.** Freeze ResNet
   including batch-normalization statistics. Cache features before the trainable
   adapter; record weights/checksum, image IDs, normalization, resize/crop/pooling,
   versions, and seeds. Plan image-disjoint fixed validation/test sets and nested
   1k/10k/larger training subsets. Fit the tokenizer only on a declared training
   pool and keep it fixed across subset comparisons. Start with a roughly 4k
   vocabulary, measuring token lengths and truncation before fixing limits.
7. **Train and evaluate in stages.** Overfit 32-128 development training images
   using one fixed caption each, then sample one caption per image per epoch for
   subset experiments. Compare correct, shuffled, and zero-content visual prefixes
   at fixed sequence layout. The zero-content intervention retains position/type
   embeddings and removes only image features; distinguish it from a separately
   trained language-only baseline. Save learning curves, checkpoints, generated
   examples, error categories, step times, and peak GPU memory before scaling.
8. **Inspect semantic processing as well as output.** Log selected prefix and
   caption hidden states, memory keys/outputs/weights, and retrieval into visual
   versus caption positions. Use image/content interventions to test causal use.
   Later compare frozen encoder features with adapter and Memory Mosaic states on
   controlled semantic probes, with held-out images and linguistic confounds
   controlled. Neither fluent captions nor high retrieval weights prove a staged
   semantic representation. Human neural supervision remains a later phase.

## Reference

```
@inproceedings{zhang2025memory,
  title={Memory mosaics},
  author={Zhang, Jianyu and Nolte, Niklas and Sadhukhan, Ranajoy and Chen, Beidi and Bottou, L{\'e}on},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={36412--36433},
  year={2025}
}

@article{zhang2026memory,
  title={Memory Mosaics at scale},
  author={Zhang, Jianyu and Bottou, L{\'e}on},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={38929--38956},
  year={2026}
}
```
