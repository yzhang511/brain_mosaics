# Short smoke test: confirm the nanoGPT-style mosaic pretraining loop runs end
# to end (data loading, model, checkpointing, wandb) on the full OpenWebText
# train.bin/val.bin (downloaded via scripts/download_openwebtext.py), using the
# same babistories-12L768d architecture as config/train_babistories.py (the
# real BabiStories pretraining run, job 47889896) so this checks that exact
# config against a new, much larger dataset -- not a real pretraining run,
# max_iters is deliberately tiny. No meta.pkl here (tiktoken GPT-2 BPE, same
# as BabiStories), so train.py falls back to vocab_size=50304.
import os

out_dir = os.environ['SCRATCH'] + '/memory_mosaics/openwebtext_smoke/out'
dataset = os.environ['SCRATCH'] + '/memory_mosaics/openwebtext'  # absolute path overrides data/<dataset>

eval_interval = 25
eval_iters = 20
log_interval = 5
always_save_checkpoint = True
eval_only = False

wandb_log = True
wandb_project = 'memory-mosaics'
wandb_run_name = 'openwebtext-12L768d-smoke'

sample_num = 1
sample_max_new_tokens = 50

gradient_accumulation_steps = 1
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
max_iters = 50
lr_decay_iters = 50
warmup_iters = 10
min_lr = 6e-5

compile = False  # not worth the warmup cost for a 50-iter smoke test
