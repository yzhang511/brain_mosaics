#!/bin/bash
#SBATCH --job-name=babistories-shakedown
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/babistories_shakedown/train-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics/nanoMosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR="$SCRATCH/memory_mosaics/babistories_shakedown"
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/babistories_shakedown/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/babistories_shakedown/wandb-config"
mkdir -p "$WANDB_DIR"

python3 train.py config/train_babistories.py \
  --out_dir="$SCRATCH/memory_mosaics/babistories_shakedown/out" \
  --wandb_run_name=babistories-shakedown \
  --max_iters=200 --lr_decay_iters=200 --warmup_iters=20 \
  --eval_interval=100 --eval_iters=10 --log_interval=10 \
  --compile=False
