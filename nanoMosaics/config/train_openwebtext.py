# Real OpenWebText pretraining run, same babistories-12L768d architecture as
# config/train_babistories.py (n_layer=12, n_embd=768, n_head=12, pmem_size=2688),
# validated against this dataset by the openwebtext-12L768d-smoke run (job 48140608):
# 50 iters, loss 12.32->7.47, no NaNs, checkpoints saved cleanly.
#
# init_from auto-resumes from this run's own last checkpoint if one exists, so the
# exact same `sbatch scripts/pretrain_openwebtext.sbatch` command can be resubmitted
# after a timeout/crash, or deliberately to extend training further, and it will pick
# up from the saved iter_num/optimizer state rather than starting over from scratch.
#
# max_iters/lr_decay_iters are placeholders here (as in train_babistories.py) -- the
# real values are set via the sbatch script's CLI override, sized from this run's
# wall-clock budget.
import os

out_dir = os.environ['SCRATCH'] + '/memory_mosaics/openwebtext_pretrain/out'
dataset = os.environ['SCRATCH'] + '/memory_mosaics/openwebtext'
init_from = 'resume' if os.path.exists(os.path.join(out_dir, 'ckpt.pt')) else 'scratch'

eval_interval = 500
eval_iters = 50
log_interval = 20
always_save_checkpoint = True
eval_only = False

wandb_log = True
wandb_project = 'memory-mosaics'
wandb_run_name = 'openwebtext-12L768d-pretrain'

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
