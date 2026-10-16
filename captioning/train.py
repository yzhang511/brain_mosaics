"""Captioning trainer with reproducible image interventions.

Supports a fixed-caption overfit mode and a per-epoch caption resampling mode
for larger runs, where each training image's caption is redrawn uniformly
from its human references at the start of every epoch.
"""
import argparse
from dataclasses import asdict
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from captioning.data import PAD_ID, EOS_ID, caption_batch, decode, encode, select_captions
from captioning.model import CaptionConfig, Captioner
from captioning.prepare import sha256
from captioning.tracking import Tracker


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, features, inputs, targets, batch_size):
    model.eval()
    total_loss, count, correct = 0., 0, 0
    for start in range(0, len(features), batch_size):
        logits, _ = model(features[start:start + batch_size], inputs[start:start + batch_size])
        labels = targets[start:start + batch_size]
        valid = labels != PAD_ID
        total_loss += F.cross_entropy(logits[valid], labels[valid], reduction='sum').item()
        count += valid.sum().item()
        correct += ((logits.argmax(-1) == labels) & valid).sum().item()
    return dict(nll=total_loss / count, token_accuracy=correct / count, tokens=count)


@torch.no_grad()
def generations(model, features, records, batch_size):
    result = []
    for start in range(0, len(features), batch_size):
        generated = model.generate(features[start:start + batch_size]).tolist()
        for record, tokens in zip(records[start:start + batch_size], generated):
            expected = encode(record['caption'])[1:]
            predicted = tokens[1:]
            if EOS_ID in predicted:
                predicted = predicted[:predicted.index(EOS_ID) + 1]
            result.append(dict(image_id=record['image_id'], reference=record['caption'],
                generated=decode(tokens), generated_ids=predicted,
                exact_token_match=predicted == expected, emitted_eos=EOS_ID in predicted))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--core-freeze-steps', type=int, default=0,
                        help='Freeze model.core (language core, including the tied wte/lm_head '
                             'embedding) for the first N training steps, so only the adapter and '
                             'image_type warm up; core unfreezes for step N+1 onward. Freezing uses '
                             'requires_grad_(False), not no_grad, so caption gradients still flow '
                             'back through the frozen core into the adapter. 0 (default): never '
                             'freeze, matching prior behavior.')
    parser.add_argument('--adapter-lr', type=float, default=None,
                        help='Learning rate for adapter.parameters() + image_type. '
                             'Defaults to --learning-rate.')
    parser.add_argument('--core-lr', type=float, default=None,
                        help='Learning rate for core.parameters(). Defaults to --learning-rate.')
    parser.add_argument('--target-nll', type=float, default=0.03)
    parser.add_argument('--eval-train-subset-size', type=int, default=None,
                        help='If set, evaluate the train split (NLL/accuracy/generations, at step 0, '
                             'every --eval-every, and in the final report) on a fixed random subsample '
                             'of this many train images instead of the full split; the same subsample '
                             'is used every time, drawn deterministically from --seed. Training steps '
                             'still sample from the full train split. Intended for large train splits '
                             '(e.g. 100k) where full-split evaluation dominates runtime.')
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--caption-resampling', choices=['fixed', 'per-epoch'], default='fixed',
                        help='fixed: always train on each image\'s lowest-ID caption. '
                             'per-epoch: redraw a caption per training image, uniformly from its '
                             'references, at the start of every epoch (an epoch is ceil(train_size/batch_size) steps).')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--n-layer', type=int, default=4)
    parser.add_argument('--n-embd', type=int, default=256)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--pmem-size', type=int, default=256)
    parser.add_argument('--grid-size', type=int, default=14,
                        help='Must match the spatial grid size features were cached at '
                             '(captioning.prepare --feature-pool-size).')
    parser.add_argument('--feature-dim', type=int, default=128,
                        help='Must match the channel count of the cached features '
                             '(128 for ResNet-18 layer2, 512 for layer4).')
    parser.add_argument('--block-size', type=int, default=None,
                        help='Defaults to grid_size**2 + max_caption_tokens. Set explicitly to '
                             'match a pretrained core (512 for babistories-12L768d).')
    parser.add_argument('--init-from-checkpoint', type=Path,
                        help='nanoMosaics pretraining checkpoint (e.g. babistories-12L768d ckpt.pt). '
                             'Loaded into the language core only; adapter/image-type/positions stay '
                             'freshly initialized. Ignored if --resume is also given.')
    parser.add_argument('--qualitative-panel-size', type=int, default=32,
                        help='Autoregressive generation (for the final report) is capped to this '
                             'many images per split/condition; NLL/token-accuracy in the same report '
                             'still cover the full split. Generation is far more expensive than a '
                             'single forward pass, so running it over a full 10k-scale split is '
                             'infeasible; this is the fixed qualitative panel the plan calls for.')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='disabled')
    parser.add_argument('--wandb-project', default='memory-mosaics')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-name', default='coco32-fixed-caption-overfit')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        raise ValueError('Steps, batch size and evaluation interval must be positive')
    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(args.manifest.read_text())
    cache = torch.load(args.features, map_location='cpu', weights_only=True)
    if cache['manifest_sha256'] != sha256(args.manifest):
        raise ValueError('Feature cache does not match manifest')
    batches = {}
    for split in ('train', 'val'):
        records = manifest[split]
        if cache[f'{split}_ids'] != [record['image_id'] for record in records]:
            raise ValueError('Feature and caption image ordering differs')
        inputs, targets = caption_batch(records)
        batches[split] = (cache[split].to(device), inputs.to(device), targets.to(device))
    train_records = manifest['train']
    epoch_steps = -(-len(train_records) // args.batch_size)  # ceil division
    eval_train_indices = None
    if args.eval_train_subset_size is not None:
        subset_size = min(args.eval_train_subset_size, len(train_records))
        subset_rng = random.Random(f'{args.seed}:eval-train-subsample')
        eval_train_indices = torch.tensor(
            sorted(subset_rng.sample(range(len(train_records)), subset_size)), device=device)

    def eval_view(features, inputs, targets):
        if eval_train_indices is None:
            return features, inputs, targets
        return features[eval_train_indices], inputs[eval_train_indices], targets[eval_train_indices]

    def resample_train_batch(epoch_index):
        rng = random.Random(f'{args.seed}:caption-resample:{epoch_index}')
        captions = select_captions(train_records, rng)
        inputs, targets = caption_batch(train_records, captions=captions)
        return inputs.to(device), targets.to(device)

    config = CaptionConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                           pmem_size=args.pmem_size, block_size=args.block_size,
                           grid_size=args.grid_size, feature_dim=args.feature_dim)
    model = Captioner(config).to(device)
    pretrained_checkpoint_sha256 = None
    if args.init_from_checkpoint and not args.resume:
        pretrained = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        # dropout is a training hyperparameter, not an architectural/shape constraint, so a
        # fine-tuning run is free to pick its own value independent of what pretraining used.
        expected = dict(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
                        pmem_size=config.pmem_size, pmem_count=1, vocab_size=config.vocab_size,
                        bias=config.bias, block_size=model.core.config.block_size)
        actual = {key: pretrained['model_args'][key] for key in expected}
        if actual != expected:
            raise ValueError(f'Checkpoint architecture {actual} does not match '
                              f'the requested architecture {expected}')
        model.load_pretrained_core(pretrained['model'])
        pretrained_checkpoint_sha256 = sha256(args.init_from_checkpoint)
    adapter_lr = args.learning_rate if args.adapter_lr is None else args.adapter_lr
    core_lr = args.learning_rate if args.core_lr is None else args.core_lr
    optimizer = torch.optim.AdamW([
        {'params': list(model.adapter.parameters()) + [model.image_type], 'lr': adapter_lr},
        {'params': list(model.core.parameters()), 'lr': core_lr},
    ], weight_decay=0.01)
    history, timings, start_step = [], [], 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved['manifest_sha256'] != cache['manifest_sha256'] or saved['model_config'] != asdict(config):
            raise ValueError('Checkpoint configuration does not match this pilot')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_state'].cpu())
        if device.type == 'cuda' and saved.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        history, timings, start_step = saved['history'], saved['step_seconds'], saved['step']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256(Path(__file__).parent / name) for name in
                     ['model.py', 'vision.py', 'data.py', 'prepare.py', 'train.py', 'tracking.py']}
    experiment = dict(model=asdict(config), optimizer=dict(name='AdamW', adapter_lr=adapter_lr,
        core_lr=core_lr, core_freeze_steps=args.core_freeze_steps,
        weight_decay=0.01, betas=[0.9, 0.999], clip_grad_norm=1.0),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        manifest_sha256=cache['manifest_sha256'], feature_cache_sha256=sha256(args.features),
        pretrained_checkpoint_sha256=pretrained_checkpoint_sha256,
        source_sha256=source_hashes, torch_version=torch.__version__, device=str(device),
        sampling=('uniform random images with replacement, fixed lowest-ID caption' if args.caption_resampling == 'fixed'
                  else f'uniform random images with replacement; caption per training image resampled '
                       f'uniformly from its references every epoch ({epoch_steps} steps/epoch)'),
        decoding='greedy; suppress PAD/BOS; stop at EOS or 64 generated tokens',
        intervention='roll features by one across the entire split; zero raw cached features before adapter')
    (args.output_dir / 'config.json').write_text(json.dumps(experiment, indent=2) + '\n')
    tracker = Tracker(args.wandb_mode, args.wandb_project, args.wandb_entity,
                      args.wandb_name, args.output_dir, experiment)
    if tracker.run is not None:
        (args.output_dir / 'wandb_run.json').write_text(json.dumps(dict(
            id=tracker.run.id, url=tracker.run.url, mode=args.wandb_mode), indent=2) + '\n')
    features = batches['train'][0]
    if args.caption_resampling == 'per-epoch':
        current_epoch = start_step // epoch_steps
        inputs, targets = resample_train_batch(current_epoch)
    else:
        current_epoch = None
        inputs, targets = batches['train'][1], batches['train'][2]
    start_time = time.perf_counter()

    def checkpoint(step):
        path = args.output_dir / 'checkpoint.pt'
        temporary = path.with_suffix('.partial')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            model_config=asdict(config), manifest_sha256=cache['manifest_sha256'],
            step=step, history=history, step_seconds=timings,
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None), temporary)
        temporary.replace(path)
        (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')

    if not history:
        initial = evaluate(model, *eval_view(features, inputs, targets), args.batch_size)
        history.append(dict(step=0, train=initial))
        print(json.dumps(history[-1]), flush=True)
        tracker.log(0, {f'train/{key}': value for key, value in initial.items()})
    step = start_step
    for step in range(start_step + 1, args.steps + 1):
        if args.caption_resampling == 'per-epoch':
            epoch_index = (step - 1) // epoch_steps
            if epoch_index != current_epoch:
                current_epoch = epoch_index
                inputs, targets = resample_train_batch(epoch_index)
        model.train()
        core_frozen = step <= args.core_freeze_steps
        model.core.requires_grad_(not core_frozen)
        indices = torch.randint(len(features), (args.batch_size,), device=device)
        synchronize(device)
        before = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(features[indices], inputs[indices], targets[indices])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize(device)
        timings.append(time.perf_counter() - before)
        if step % 10 == 0:
            tracker.log(step, {'train/batch_nll': loss.item(), 'runtime/step_seconds': timings[-1]})
        if step % args.eval_every == 0 or step == args.steps:
            measured = evaluate(model, *eval_view(features, inputs, targets), args.batch_size)
            history.append(dict(step=step, train=measured, epoch=current_epoch, core_frozen=core_frozen,
                                elapsed_seconds=time.perf_counter() - start_time))
            print(json.dumps(history[-1]), flush=True)
            validation = evaluate(model, *batches['val'], args.batch_size)
            history[-1]['val'] = validation
            tracker.log(step, {**{f'train/{k}': v for k, v in measured.items()},
                               **{f'val/{k}': v for k, v in validation.items()}})
            checkpoint(step)
            if measured['nll'] <= args.target_nll:
                break
    checkpoint(step)
    report = dict(step=step, overfit_target_nll=args.target_nll,
        target_reached=history[-1]['train']['nll'] <= args.target_nll,
        median_step_seconds=statistics.median(timings) if timings else None,
        training_loop_seconds=time.perf_counter() - start_time,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        gpu_note='CPU run; GPU timing and peak GPU memory were not measured' if device.type == 'cpu' else None,
        results={})
    report_batches = dict(batches, train=eval_view(*batches['train']))
    for split, (features, inputs, targets) in report_batches.items():
        report['results'][split] = {}
        panel_size = min(args.qualitative_panel_size, len(features))
        for condition, modified in [('correct', features), ('shuffled', features.roll(1, 0)),
                                    ('zero_content', torch.zeros_like(features))]:
            metrics = evaluate(model, modified, inputs, targets, args.batch_size)
            samples = generations(model, modified[:panel_size], manifest[split][:panel_size], args.batch_size)
            metrics.update(generations=samples, qualitative_panel_size=panel_size,
                exact_match_rate=sum(s['exact_token_match'] for s in samples) / len(samples))
            if condition == 'shuffled':
                donor_ids = [r['image_id'] for r in manifest[split][:panel_size]]
                donor_ids = donor_ids[-1:] + donor_ids[:-1]
                for sample, donor in zip(samples, donor_ids):
                    sample['feature_image_id'] = donor
            report['results'][split][condition] = metrics
            tracker.log(step, {f'{split}/{condition}/{key}': value for key, value in metrics.items()
                               if key != 'generations'})
            tracker.samples(step, split, condition, samples)
            print(f'{split}/{condition}: nll={metrics["nll"]:.4f}, exact={metrics["exact_match_rate"]:.3f}', flush=True)
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    tracker.finish({key: value for key, value in report.items() if key != 'results'}, args.output_dir)


if __name__ == '__main__':
    main()
