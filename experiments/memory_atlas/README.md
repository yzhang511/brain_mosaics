# Example-guided memory atlas

Descriptive inspection of persistent-memory weights and final-prompt retrieval for the six successful lexical examples, across neutral, consistent, contradictory and irrelevant conditions. No causal interventions or checkpoint edits.

## Open the results

- [Standalone interactive atlas](results/2026-10-15/atlas.html)
- [Findings and methods](results/2026-10-15/REPORT.md)
- [Overview figure](results/2026-10-15/overview.png)
- [Per-example retrieval changes](results/2026-10-15/per_example.png)

The HTML embeds its data and needs no server or network connection. Open it in a browser. It includes all 387,072 slots' geometry and selected-slot overlays/profiles; complete retrieval arrays are saved separately. All identifiers are zero-based.

## Reproduce

Use a compute allocation. The measured extraction takes about one minute with one CPU thread; allow extra time for checkpoint loading, figures and serialization. The checkpoint is 1.5 GB; the existing 6.4 GB allocation sufficed. Use the existing Sherlock modules below; no installation is needed.

```bash
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/memory_atlas/build.py \
  --checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --out "$SCRATCH/memory_mosaics/memory_atlas/2026-10-15"

ml purge
ml load viz py-matplotlib/3.8.3_py312
python3 experiments/memory_atlas/summarize.py \
  --input "$SCRATCH/memory_mosaics/memory_atlas/2026-10-15"
```

`--benchmark` on `build.py` runs one prompt and one head for numerical validation and timing; its partial output is not a complete interactive atlas and must use a separate output directory. The full builder reproduces complete candidate scores, including multi-token candidates. Retrieval and first-token directional diagnostics use the final prompt position only.

`node experiments/memory_atlas/smoke_ui.js path/to/atlas.html` exercises selections, overview metrics and plot clicks with a minimal DOM/canvas harness. This is JavaScript smoke validation, not a real-browser rendering check.

The fixed slot-profile shortlist, PCA definitions, numerical thresholds, array schemas and limits are documented in the report. Raw arrays preserve full precision; plotted geometry is rounded to four decimal places to keep the HTML smaller. Small final artifacts are preserved here as well as in scratch. Scratch is temporary storage, not the only archival copy.
