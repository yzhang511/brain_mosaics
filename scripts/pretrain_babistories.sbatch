#!/bin/bash
#SBATCH --job-name=babistories-pretrain
#SBATCH --partition=henderj
#SBATCH --qos=high_p
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/babistories/train-%j.out

# max_iters/lr_decay_iters below are sized from a timed calibration run
# (job 47888952: batch_size=16/grad_accum=8, compile=False, ~2.12s/iter
# steady-state, ~12.2s eval+checkpoint overhead per eval_interval=500
# cycle) to fit a 15h wall-clock budget with buffer to spare.

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics/nanoMosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR="$SCRATCH/memory_mosaics/babistories"
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/babistories/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/babistories/wandb-config"
mkdir -p "$WANDB_DIR"

python3 train.py config/train_babistories.py \
  --max_iters=23000 --lr_decay_iters=23000 \
  --compile=False
