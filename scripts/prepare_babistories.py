"""Tokenize the extracted BabiStories train/val text files into nanoMosaics-style bin files.

Each line of <raw-dir>/{train,val}dataset.txt is a JSON-encoded story string
(matching Library/memory_mosaics/data/BabiStories/prepare.py's format). Uses
the same tiktoken gpt2 BPE as captioning/data.py (vocab_size=50304 default,
eot=50256), so a checkpoint pretrained on this data loads into the same
token-id space the captioner uses.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken


def tokenize(raw_path, out_path, enc):
    ids = []
    with open(raw_path, 'r') as stream:
        for index, line in enumerate(stream):
            ids.extend(enc.encode_ordinary(json.loads(line)))
            ids.append(enc.eot_token)
            if (index + 1) % 100000 == 0:
                print(f'{raw_path.name}: {index + 1} stories, {len(ids)} tokens so far', flush=True)
    array = np.array(ids, dtype=np.uint16)
    array.tofile(out_path)
    print(f'{raw_path.name}: wrote {out_path} ({len(array)} tokens)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True,
                        help='Directory containing extracted traindataset.txt / valdataset.txt')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding('gpt2')
    for split in ('val', 'train'):
        tokenize(args.raw_dir / f'{split}dataset.txt', args.out_dir / f'{split}.bin', enc)


if __name__ == '__main__':
    main()
