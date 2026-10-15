# Real BabiStories pretraining run. Architecture matches the scale used by
# Library/scripts/train_babistories.sh (n_layer=12, n_embd=768, n_head=12,
# pmem_size=2688) but adapted for a single GPU here instead of 8 nodes x 8
# GPUs: smaller effective batch, and learning_rate/warmup/min_lr taken from
# nanoMosaics' own GPT2-scale defaults (config/train_gpt2.py uses the same
# n_layer/n_embd/n_head) rather than the production recipe's 5e-3, which was
# tuned for a ~512-sample effective batch we can't match on one GPU.
#
# max_iters/lr_decay_iters are placeholders -- set after a timed shakedown
# measures actual step time on the assigned GPU.
import os

out_dir = os.environ['SCRATCH'] + '/memory_mosaics/babistories/out'
dataset = os.environ['SCRATCH'] + '/memory_mosaics/babistories'

eval_interval = 500
eval_iters = 50
log_interval = 20
always_save_checkpoint = True
eval_only = False

wandb_log = True
wandb_project = 'memory-mosaics'
wandb_run_name = 'babistories-12L768d'

# this config already runs near the single-GPU memory ceiling at
# batch_size=32/block_size=512/pmem_size=2688, so keep qualitative sampling
# small -- the 100-token/3-sample defaults are sized for the smoke config
sample_num = 1
sample_max_new_tokens = 50

gradient_accumulation_steps = 8
batch_size = 16
block_size = 512

n_layer = 12
n_head = 12
n_embd = 768
pmem_count = 1
pmem_size = 2688
dropout = 0.05
bias = False

learning_rate = 6e-4
weight_decay = 1e-1
beta2 = 0.95
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 1000
min_lr = 6e-5

compile = True
