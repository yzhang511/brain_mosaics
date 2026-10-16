#!/bin/bash
#SBATCH --job-name=coco10k-babistories-shakedown
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/babistories_shakedown-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Timing/sanity check only: measures seconds/step and confirms the pretrained-checkpoint
# path (Stage 0 of captioning/JOINT_FINETUNING_PLAN.md) runs end to end on a GPU, including
# the qualitative-panel-capped final report, before sizing the real run's --steps.
python3 -m captioning.train \
  --manifest reports/caption_pilot_10k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_10k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_10k/babistories_shakedown" \
  --n-layer 12 --n-embd 768 --n-head 12 --pmem-size 2688 --block-size 512 \
  --init-from-checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --steps 20 --batch-size 16 --eval-every 20 --qualitative-panel-size 8 \
  --caption-resampling per-epoch --device cuda
