#!/bin/bash
#SBATCH --job-name=download-openwebtext
#SBATCH --partition=henderj
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/openwebtext/download-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-datasets/2.18.0_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
export HF_TOKEN
HF_TOKEN="$(cat "$HOME/.hf_token")"
mkdir -p "$HF_HOME" "$SCRATCH/memory_mosaics/openwebtext"

python3 scripts/download_openwebtext.py
