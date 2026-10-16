"""Frozen-checkpoint lexical defaults and entity-specific preference pilot."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nanoMosaics'))
from mosaic_model import MemoryMosaic, MemoryMosaicConfig

PAIRS = [
    ('rabbit', 'cat', 'carrots', 'fish'),
    ('monkey', 'squirrel', 'bananas', 'nuts'),
    ('cow', 'lion', 'grass', 'meat'),
    ('panda', 'penguin', 'bamboo', 'fish'),
    ('bee', 'frog', 'nectar', 'flies'),
    ('horse', 'bear', 'hay', 'honey'),
    ('mouse', 'owl', 'cheese', 'mice'),
    ('butterfly', 'dog', 'nectar', 'bones'),
    ('chicken', 'goat', 'seeds', 'leaves'),
    ('elephant', 'sheep', 'fruit', 'grass'),
]
FRAMES = [
    'When the {n} chose its favorite food, it picked',
    'The {n} was hungry. It wanted to eat',
    'At lunchtime, the {n} asked for a plate of',
    'The {n} looked for something tasty. It hoped to find',
    'For dinner, the {n} chose to have',
    'The {n} smiled when it saw its favorite food,',
    'When it was time to eat, the {n} reached for the',
    'The {n} decided to fill its bowl with',
]


def stimuli():
    rows = []
    for pair_id, (a, b, x, y) in enumerate(PAIRS):
        for noun, other, default, alternative in [(a, b, x, y), (b, a, y, x)]:
            for frame_id, frame in enumerate(FRAMES):
                for variant in range(2):
                    first, second = (noun, other) if variant == 0 else (other, noun)
                    intro = f'Once upon a time, a {first} and a {second} lived beside a quiet forest. '
                    for condition in ['neutral', 'consistent', 'conflicting', 'other']:
                        if condition == 'neutral':
                            statement = 'One morning, the two friends sat together beside a little table. '
                        else:
                            subject = other if condition == 'other' else noun
                            liked, disliked = (default, alternative) if condition == 'consistent' else (alternative, default)
                            if variant == 0:
                                statement = f'The {subject} loved {liked} and disliked {disliked}. '
                            else:
                                statement = f'The {subject} disliked {disliked} and loved {liked}. '
                        rows.append(dict(pair_id=pair_id, noun=noun, other=other, default=default,
                                         alternative=alternative, frame_id=frame_id, variant=variant,
                                         split='discovery' if frame_id < 4 else 'evaluation',
                                         condition=condition, prompt=intro + statement + frame.format(n=noun)))
    return rows


class Scorer:
    def __init__(self, model, enc, device):
        self.model, self.enc, self.device = model, enc, device

    def encode(self, prompt, candidate):
        prefix = self.enc.encode_ordinary(prompt)
        whole = self.enc.encode_ordinary(prompt + ' ' + candidate)
        assert whole[:len(prefix)] == prefix, 'Candidate changes prefix tokenization'
        assert len(whole) > len(prefix)
        assert len(whole) <= self.model.config.block_size
        return whole, len(prefix)

    @torch.inference_mode()
    def batch(self, examples):
        encoded = [self.encode(*example) for example in examples]
        max_length = max(len(ids) - 1 for ids, _ in encoded)
        idx = torch.full((len(encoded), max_length), self.enc.eot_token, dtype=torch.long, device=self.device)
        bs, positions, targets = [], [], []
        for b, (ids, prefix_length) in enumerate(encoded):
            idx[b, :len(ids)-1] = torch.tensor(ids[:-1], device=self.device)
            for pos in range(prefix_length, len(ids)):
                bs.append(b)
                positions.append(pos - 1)
                targets.append(ids[pos])
        # Same backbone as forward; evaluate the vocabulary only at candidate positions.
        x = self.model.mosaic.drop(self.model.mosaic.wte(idx))
        for block in self.model.mosaic.h:
            x = block(x)
        x = self.model.mosaic.ln_f(x)
        logits = self.model.lm_head(x[bs, positions]).float()
        token_scores = logits.log_softmax(-1).gather(1, torch.tensor(targets, device=self.device)[:, None]).squeeze(1)
        totals = torch.zeros(len(examples), device=self.device)
        totals.index_add_(0, torch.tensor(bs, device=self.device), token_scores)
        return totals.cpu().tolist()

    @torch.inference_mode()
    def reference(self, example):
        ids, prefix = self.encode(*example)
        total = 0.
        for pos in range(prefix, len(ids)):
            logits, _ = self.model(torch.tensor([ids[:pos]], device=self.device))
            total += logits[0, -1].float().log_softmax(-1)[ids[pos]].item()
        return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    torch.manual_seed(1729)
    start = time.monotonic()
    stat = args.checkpoint.stat()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = dict(checkpoint=str(args.checkpoint), checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                iteration=checkpoint.get('iter_num'), model_args=checkpoint['model_args'],
                training_run=checkpoint.get('config', {}).get('wandb_run_name'),
                torch_version=str(torch.__version__), device=args.device, dtype='float32',
                seed=1729, batch_size=args.batch_size, threads=torch.get_num_threads(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    model = MemoryMosaic(MemoryMosaicConfig(**checkpoint['model_args']))
    state = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    gc.collect()
    model = model.to(args.device).eval()
    assert args.checkpoint.stat().st_mtime_ns == stat.st_mtime_ns
    enc = tiktoken.get_encoding('gpt2')
    scorer = Scorer(model, enc, args.device)
    rows = stimuli()
    examples = list(dict.fromkeys((r['prompt'], r[c]) for r in rows for c in ['default', 'alternative']))
    # Include a multi-token candidate to validate teacher-forced accumulation.
    checks = examples[:2] + [(rows[35]['prompt'], 'fresh green vegetables')]
    batched = scorer.batch(checks)
    reference = [scorer.reference(e) for e in checks]
    singles = [scorer.batch([e])[0] for e in checks]
    error = max(abs(a-b) for a,b in zip(batched, reference))
    padding_error = max(abs(a-b) for a,b in zip(batched, singles))
    assert error < 2e-4 and padding_error < 2e-4, (error, padding_error)
    meta.update(reference_max_error=error, batch_max_error=padding_error, prompts=len(rows), unique_candidates=len(examples))
    bench_start = time.monotonic()
    count = min(32, len(examples))
    for i in range(0, count, args.batch_size):
        scorer.batch(examples[i:i+args.batch_size])
    elapsed = time.monotonic() - bench_start
    meta.update(benchmark_seconds=elapsed, estimated_scoring_seconds=elapsed / count * len(examples))
    print(json.dumps(meta, indent=2), flush=True)
    if args.benchmark:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i:i+args.batch_size]
        cache.update(zip(batch, scorer.batch(batch)))
        if i % (args.batch_size * 40) == 0:
            print(f'Scored {min(i+args.batch_size, len(examples))}/{len(examples)} candidates in {time.monotonic()-start:.1f}s', flush=True)
    for r in rows:
        for key in ['default', 'alternative']:
            r[key + '_logp'] = cache[(r['prompt'], r[key])]
            r[key + '_tokens'] = len(scorer.encode(r['prompt'], r[key])[0]) - len(enc.encode_ordinary(r['prompt']))
        r['D'] = r['default_logp'] - r['alternative_logp']
    with (args.out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta['total_seconds'] = time.monotonic() - start
    (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    main()
