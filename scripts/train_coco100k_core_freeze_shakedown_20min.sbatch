#!/bin/bash
#SBATCH --job-name=coco100k-core-freeze-shakedown-20min
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_100k/core_freeze_shakedown_20min-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/caption_pilot_100k/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/caption_pilot_100k/wandb-config"

# First shakedown on the 100k/layer4 cache (captioning/JOINT_FINETUNING_PLAN.md Stage 0):
# confirms the pipeline runs end to end at this scale with the new layer4 (512ch, native 7x7)
# feature tap, before committing to a longer run. train.py's evaluate() always runs over the
# FULL split it's given -- at 10k images that's ~1min/pass (fine for the 90min shakedown this
# mirrors, train_coco10k_core_freeze_shakedown_90min.sbatch), but at 100k images it's ~9.3min/
# pass (measured rate from the 10k pilot), and it's called at step 0, every --eval-every, and
# 3x more (correct/shuffled/zero_content) in the final report -- >35min of mandatory eval alone,
# which cannot fit a 20min budget. New --eval-train-subset-size 2000 (captioning/train.py) fixes
# this: evaluates train on the same fixed random 2000-image subsample every time instead of the
# full 100k, making train eval as cheap as the existing 2000-image val eval. Training steps are
# unaffected (still drawn from the full 100k split).
# --grid-size 7 --feature-dim 512: matches this cache's layer4 tap (512ch, native 7x7), unlike
# the 10k cache's layer2 (128ch, pooled 14x14) -- must match or the forward pass shape-mismatches.
python3 -m captioning.train \
  --manifest reports/caption_pilot_100k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_100k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_100k/core_freeze_shakedown_20min" \
  --grid-size 7 --feature-dim 512 \
  --n-layer 12 --n-embd 768 --n-head 12 --pmem-size 2688 --block-size 512 \
  --init-from-checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --core-freeze-steps 200 --adapter-lr 1e-3 --core-lr 1e-4 \
  --steps 2000 --batch-size 16 --eval-every 500 --eval-train-subset-size 2000 \
  --qualitative-panel-size 8 \
  --caption-resampling per-epoch --device cuda \
  --wandb-mode online --wandb-project memory-mosaics \
  --wandb-name coco100k-core-freeze-shakedown-20min
