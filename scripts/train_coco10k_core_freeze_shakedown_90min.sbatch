#!/bin/bash
#SBATCH --job-name=coco10k-shakedown-90min
#SBATCH --partition=henderj
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/core_freeze_shakedown_90min-%j.out

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

# Third Stage 0 shakedown (captioning/JOINT_FINETUNING_PLAN.md): longer run (~90 min, ~22
# epochs over the 10k-image train manifest with per-epoch caption resampling) to see whether
# train/correct starts separating from train/shuffled with more training, not just from
# train/zero_content as the 200-step shakedown (job 48052659) already showed. Step/eval-every
# counts are sized from that run's timing (~0.307s/train step; ~67s per periodic full-split
# train+val eval; ~290s fixed setup+final-report overhead) to fit a ~90 minute budget with 10
# eval checkpoints. --time is padded to 2h; the job exits on its own once done.
python3 -m captioning.train \
  --manifest reports/caption_pilot_10k/manifest.json \
  --features "$SCRATCH/memory_mosaics/caption_pilot_10k/features.pt" \
  --output-dir "$SCRATCH/memory_mosaics/caption_pilot_10k/core_freeze_shakedown_90min" \
  --n-layer 12 --n-embd 768 --n-head 12 --pmem-size 2688 --block-size 512 \
  --init-from-checkpoint "$SCRATCH/memory_mosaics/babistories/out/ckpt.pt" \
  --core-freeze-steps 200 --adapter-lr 1e-3 --core-lr 1e-4 \
  --steps 14000 --batch-size 16 --eval-every 1400 --qualitative-panel-size 8 \
  --caption-resampling per-epoch --device cuda \
  --wandb-mode online --wandb-project memory-mosaics \
  --wandb-name coco10k-core-freeze-shakedown-90min
