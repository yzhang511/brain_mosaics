#!/bin/bash
#SBATCH --job-name=coco10k-core-freeze-shakedown
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/core_freeze_shakedown-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Second Stage 0 shakedown (captioning/JOINT_FINETUNING_PLAN.md): exercises the new
# --core-freeze-steps/--adapter-lr/--core-lr flags (adapter-only warm-up for the first 50
# steps, then joint adapter+core fine-tuning for the remaining 150) on the same 10k-image
# manifest/feature cache and pretrained checkpoint as the first shakedown. Goal: confirm the
# run is stable end to end and check, from the final report's per-condition results, whether
# train/correct starts diverging from train/zero_content -- not to pick final hyperparameters
# for the real multi-thousand-step run yet.
python3 -m captioning.train \
  --manifest reports/caption_pilot_10k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_10k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_10k/core_freeze_shakedown" \
  --n-layer 12 --n-embd 768 --n-head 12 --pmem-size 2688 --block-size 512 \
  --init-from-checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --core-freeze-steps 50 --adapter-lr 1e-3 --core-lr 1e-4 \
  --steps 200 --batch-size 16 --eval-every 50 --qualitative-panel-size 8 \
  --caption-resampling per-epoch --device cuda
