"""Download and tokenize OpenWebText (GPT-2 BPE) into train.bin/val.bin on $SCRATCH.

Adapted from nanoMosaics/data/openwebtext/prepare.py, parameterized to write under
$SCRATCH instead of the script's own directory: train.bin is ~17GB and the raw HF
cache is ~54GB, both too large for the $HOME quota.
"""
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
# kept at 1 (unlike num_proc above): parallel metadata/file requests against the HF Hub API
# from Sherlock's shared outbound IP tripped a 429 rate limit (job 48074210)
num_proc_load_dataset = 1

OUT_DIR = Path(os.environ['SCRATCH']) / 'memory_mosaics' / 'openwebtext'
OUT_DIR.mkdir(parents=True, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # takes ~54GB in the HF cache dir (HF_HOME/HF_DATASETS_CACHE), about 8M documents
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # owt by default only contains the 'train' split, so create a held-out split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])  # ignores any special tokens
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = OUT_DIR / f'{split}.bin'
        dtype = np.uint16  # enc.max_token_value == 50256 fits in uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f'wrote {filename} ({arr_len} tokens)')

    print('done. data in', OUT_DIR)
