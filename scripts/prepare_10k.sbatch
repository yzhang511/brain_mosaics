#!/bin/bash
#SBATCH --job-name=memory-mosaics-prepare-10k
#SBATCH --partition=henderj
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/prepare-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 -m captioning.prepare \
  --work-dir "$SCRATCH/memory_mosaics/caption_pilot_10k" \
  --manifest-dir reports/caption_pilot_10k \
  --train-size 10000 --val-size 1000 --test-size 2000 \
  --device cpu --download-workers 16 --encode-batch-size 64
