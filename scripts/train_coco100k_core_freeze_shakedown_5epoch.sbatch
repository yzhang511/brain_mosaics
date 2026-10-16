#!/bin/bash
#SBATCH --job-name=coco100k-core-freeze-shakedown-5epoch
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_100k/core_freeze_shakedown_5epoch-%j.out

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

# Real Stage 0 shakedown on the 100k/layer4 cache (captioning/JOINT_FINETUNING_PLAN.md), sized
# from the 20min smoke test (job 48131992): median_step_seconds=0.1668, and that run's ~177s of
# overhead beyond pure step compute (setup, evals, final report) came from checkpoint COUNT, not
# from --steps, so it stays roughly flat here. --steps 31250 = exactly 5 epochs over the 100k
# train set (100000/16 = 6250 steps/epoch) -- the same epoch range (4-6) where the 10k pilot
# (job 48056531) first showed train/val NLL diverging -- to see whether the move to 100k images
# actually closes that gap, not just whether the pipeline runs. --eval-every 6250: once per
# epoch, for a clean epoch-by-epoch train/val comparison like the 10k pilot's results.
# Estimated time: 31250*0.1668s =~ 87min step compute + ~2min eval overhead (5 epoch-boundary
# checkpoints) + ~2min setup/final-report/wandb-sync =~ 90min total, leaving ~30min buffer in
# the 2h allocation. --eval-train-subset-size 2000 (added to captioning/train.py for this):
# without it, every one of those evals would run over the full 100k train split (~9.3min/pass)
# instead of a fixed 2000-image subsample, which alone would blow well past 2 hours.
# --grid-size 7 --feature-dim 512: matches this cache's layer4 tap (512ch, native 7x7), unlike
# the 10k cache's layer2 (128ch, pooled 14x14) -- must match or the forward pass shape-mismatches.
python3 -m captioning.train \
  --manifest reports/caption_pilot_100k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_100k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_100k/core_freeze_shakedown_5epoch" \
  --grid-size 7 --feature-dim 512 \
  --n-layer 12 --n-embd 768 --n-head 12 --pmem-size 2688 --block-size 512 \
  --init-from-checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --core-freeze-steps 200 --adapter-lr 1e-3 --core-lr 1e-4 \
  --steps 31250 --batch-size 16 --eval-every 6250 --eval-train-subset-size 2000 \
  --qualitative-panel-size 8 \
  --caption-resampling per-epoch --device cuda \
  --wandb-mode online --wandb-project memory-mosaics \
  --wandb-name coco100k-core-freeze-shakedown-5epoch
