#!/bin/bash
#SBATCH --job-name=memory-mosaics-10k
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/train-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/caption_pilot_10k/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/caption_pilot_10k/wandb-config"
mkdir -p "$SCRATCH/memory_mosaics/caption_pilot_10k/run10000"

python3 -m captioning.train \
  --manifest reports/caption_pilot_10k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_10k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_10k/run10000" \
  --steps 10000 --eval-every 1000 --batch-size 16 --device cuda \
  --caption-resampling per-epoch \
  --wandb-mode online --wandb-project memory-mosaics \
  --wandb-name coco10k-10000-b16-resample

mkdir -p reports/caption_pilot_10k/run10000
cp "$SCRATCH/memory_mosaics/caption_pilot_10k/run10000"/{config.json,history.json,results.json,wandb_run.json,checkpoint.pt} \
  reports/caption_pilot_10k/run10000/
