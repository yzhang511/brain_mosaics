#!/bin/bash
#SBATCH --job-name=memory-mosaics-prepare-100k
#SBATCH --partition=henderj
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_100k/prepare-%j.out

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Scaled-up Stage 0 COCO warm-up sample (captioning/JOINT_FINETUNING_PLAN.md): the 10k-image
# pilot showed clear overfitting (widening train/val NLL gap by ~epoch 4-6 of the 90-min
# core-freeze shakedown, job 48056531) after repeating the same 10k images many epochs.
# Standard captioning practice trains on the full COCO train2017 set rather than a small
# subsample; this targets most of it (100,000 of 117,062 train2017 images available after
# excluding the 1,419 t20 coco_free_recall stimulus ids via --exclude-t20-manifest's default).
# val/test is 2000/2400 (not the full 5,000-image val2017 set: 118 of its images are also t20
# stimuli, leaving only 4,878 available -- job 48063185 failed requesting 2500/2500=5000 with
# no margin for that). Also switches the frozen feature tap from ResNet-18 layer2 (128-dim,
# texture/edge-level, native 28x28) to layer4 (512-dim, more object-level semantics, native
# 7x7) -- layer2 was an unverified carryover assumption (see "Information required before
# implementation" in the plan doc), and deeper features are standard practice for captioning.
# Kept as a separate caption_pilot_100k dataset rather than overwriting the existing 10k pilot.
python3 -m captioning.prepare \
  --work-dir "$SCRATCH/memory_mosaics/caption_pilot_100k" \
  --manifest-dir reports/caption_pilot_100k \
  --train-size 100000 --val-size 2000 --test-size 2400 \
  --feature-layer layer4 \
  --device cpu --download-workers 16 --encode-batch-size 64
