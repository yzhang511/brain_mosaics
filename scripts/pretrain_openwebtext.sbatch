#!/bin/bash
#SBATCH --job-name=openwebtext-pretrain
#SBATCH --partition=henderj
#SBATCH --qos=high_p
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/yizzhang/memory_mosaics/openwebtext_pretrain/train-%j.out

# max_iters/lr_decay_iters sized from: the babistories-12L768d calibration (pretrain_babistories.sbatch,
# job 47888952: ~2.12s/iter steady-state at batch_size=16/grad_accum=8, ~12.2s eval+checkpoint overhead
# per eval_interval=500 cycle, compile=False) plus the openwebtext-12L768d-smoke run (job 48140608),
# which confirmed ~267ms/microbatch on this new dataset -- matching 2.12s/8 almost exactly, i.e. per-step
# cost is architecture-bound, not dataset-bound, so the babistories timing transfers directly. Targets
# ~21h of content time within the 24h budget (~3h buffer, same proportion babistories-pretrain left):
# 75600s / (2.12 + 12.2/500)s per iter =~ 34,935 -> 35000.
# 35000 iters * batch_size=16 * grad_accum=8 * block_size=512 =~ 2.29B tokens, ~25% of one pass over
# OpenWebText's ~9.0B train tokens -- a partial-epoch pretraining run, not a full epoch, which is normal
# at this scale/budget on a single GPU.
# config/train_openwebtext.py auto-resumes from out_dir/ckpt.pt if one already exists, so resubmitting
# this exact script (after a timeout/crash, or to continue past 35000 later) picks up where training
# left off instead of restarting from scratch.

set -euo pipefail
cd /home/users/yizzhang/memory_mosaics/nanoMosaics
ml purge
ml load math py-torchvision/0.19.1_py312 py-wandb/0.18.7_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR="$SCRATCH/memory_mosaics/openwebtext_pretrain"
export WANDB_CACHE_DIR="$SCRATCH/memory_mosaics/openwebtext_pretrain/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/memory_mosaics/openwebtext_pretrain/wandb-config"
mkdir -p "$WANDB_DIR"

python3 train.py config/train_openwebtext.py \
  --max_iters=35000 --lr_decay_iters=35000 \
  --compile=False
