#!/bin/bash
#SBATCH --job-name=openwebtext-smoke
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/openwebtext_smoke/train-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics/nanoMosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR="$SCRATCH/memory_mosaics/openwebtext_smoke"
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/openwebtext_smoke/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/openwebtext_smoke/wandb-config"
mkdir -p "$WANDB_DIR"

python3 train.py config/train_openwebtext_smoke.py
