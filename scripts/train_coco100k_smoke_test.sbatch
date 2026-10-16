#!/bin/bash
#SBATCH --job-name=coco100k-smoke-test
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_100k/smoke_test-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Smoke test for captioning/train.py against the new 100k/layer4 cache, before committing to a
# full-length shakedown run. --steps is tiny (20), but this is NOT a quick test: evaluate() runs
# unconditionally over the FULL train split at step 0 and again (x3 conditions) in the final
# report, regardless of --steps -- at the ~5.6ms/image rate measured in the 10k pilot's 90min
# shakedown (job 48056531), that alone is ~9.3min x 4 train-split passes = ~35-40min. Running the
# real cache (not a synthetic subset) deliberately, specifically to measure actual per-eval/report
# timing at 100k scale and size --time/--eval-every for the real shakedown run correctly, since
# the 10k-tuned timing budget does not transfer (see train_coco10k_core_freeze_shakedown_90min.sbatch's
# comment) and naively reusing it would very likely time out.
# --grid-size 7 --feature-dim 512: this cache uses the layer4 tap (512ch, native 7x7), unlike the
# 10k cache's default layer2 (128ch, pooled 14x14) -- grid-size/feature-dim must match or the
# model's forward pass will fail on a shape mismatch.
python3 -m captioning.train \
  --manifest reports/caption_pilot_100k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_100k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_100k/smoke_test" \
  --grid-size 7 --feature-dim 512 \
  --n-layer 12 --n-embd 768 --n-head 12 --pmem-size 2688 --block-size 512 \
  --init-from-checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --core-freeze-steps 10 --adapter-lr 1e-3 --core-lr 1e-4 \
  --steps 20 --batch-size 16 --eval-every 20 --qualitative-panel-size 4 \
  --caption-resampling per-epoch --device cuda \
  --wandb-mode disabled
