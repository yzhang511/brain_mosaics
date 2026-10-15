#!/bin/bash
#SBATCH --job-name=babistories-prepare
#SBATCH --partition=henderj
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/babistories/prepare-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312

RAW_DIR="$SCRATCH/memory_mosaics/babistories_raw"
OUT_DIR="$SCRATCH/memory_mosaics/babistories"
mkdir -p "$RAW_DIR" "$OUT_DIR"

/share/software/user/open/p7zip/16.02/bin/7z x -y -o"$RAW_DIR" \
  BabiStories/data/babistories-dataset.7z.001

python3 scripts/prepare_babistories.py --raw-dir "$RAW_DIR" --out-dir "$OUT_DIR"
