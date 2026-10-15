# Short smoke test: confirm the nanoGPT-style mosaic pretraining loop runs
# end to end (data loading, model, checkpointing, wandb) on BabiStories,
# using the same tiny dims as captioning.model.CaptionConfig's defaults
# (n_layer=4, n_head=4, n_embd=256, pmem_size=256, bias=False) so a later
# checkpoint could be loaded into Captioner.core without a config mismatch.
# Not a real pretraining run -- max_iters is deliberately tiny.
import os

out_dir = os.environ['SCRATCH'] + '/memory_mosaics/babistories_smoke/out'
dataset = os.environ['SCRATCH'] + '/memory_mosaics/babistories'  # absolute path overrides data/<dataset>

eval_interval = 25
eval_iters = 20
log_interval = 5
always_save_checkpoint = True
eval_only = False

wandb_log = True
wandb_project = 'memory-mosaics'
wandb_run_name = 'babistories-smoke'

gradient_accumulation_steps = 1
batch_size = 12
block_size = 256

n_layer = 4
n_head = 4
n_embd = 256
pmem_count = 1
pmem_size = 256
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 50
lr_decay_iters = 50
warmup_iters = 10
min_lr = 1e-4
beta2 = 0.99

compile = False  # not worth the warmup cost for a 50-iter smoke test
