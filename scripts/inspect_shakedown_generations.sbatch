#!/bin/bash
#SBATCH --job-name=inspect-shakedown-gen
#SBATCH --partition=henderj
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/inspect-shakedown-gen-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="/home/users/yizzhang/memory_mosaics:${PYTHONPATH:-}"

python3 scripts/inspect_shakedown_generations.py
