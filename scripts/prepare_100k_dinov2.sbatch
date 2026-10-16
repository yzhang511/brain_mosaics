#!/bin/bash
#SBATCH --job-name=memory-mosaics-prepare-100k-dinov2
#SBATCH --partition=henderj
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_100k/prepare-dinov2-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-transformers/4.39.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="/home/users/yizzhang/memory_mosaics:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hf_cache"

# DINOv2 ViT-S/14 replaces the ResNet-18 layer4 tap used in the earlier 100k attempt (job
# 48063185/48064219, cancelled): a manual qualitative comparison of raw per-channel feature maps
# and PCA-RGB projections (ResNet-18 layer2, layer4, VGG16 conv5_3, CLIP ViT-B/16, DINOv2
# ViT-S/14) found DINOv2's PCA-projected patch features show cleaner object-level semantic
# clustering than any of the others, at native 16x16 resolution (vs. layer4's 7x7). Reuses
# caption_pilot_100k's work-dir/downloads (the ResNet run's raw jpgs and COCO annotations zip are
# still cached there; prepare.py's download() skips files that already exist) so this doesn't
# re-download from scratch. Train/val/test sizes (100000/2000/2400) match the fixed version of
# the earlier ResNet attempt (see prepare_100k.sbatch's comment for the val/test sizing fix).
python3 -m captioning.prepare \
  --work-dir "$SCRATCH/memory_mosaics/caption_pilot_100k" \
  --manifest-dir reports/caption_pilot_100k \
  --train-size 100000 --val-size 2000 --test-size 2400 \
  --feature-layer dinov2_vits14 \
  --device cpu --download-workers 16 --encode-batch-size 64
