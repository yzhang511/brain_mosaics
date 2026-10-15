"""Download a fixed COCO pilot and cache frozen pre-adapter features."""
import argparse
import hashlib
import json
import platform
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from captioning.data import VOCAB_SIZE, encoded_length
from captioning.vision import FrozenEncoder, preprocess_image

INSPECTION_IDS = {139, 39769, 252219, 397133}
ANNOTATIONS_URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=90) as response, open(temporary, 'wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(path)


def choose_records(archive, split, count, seed, excluded=(), max_caption_tokens=64):
    with archive.open(f'annotations/captions_{split}.json') as stream:
        data = json.load(stream)
    images = {item['id']: item for item in data['images'] if item['id'] not in excluded}
    captions = {image_id: [] for image_id in images}
    for item in data['annotations']:
        if item['image_id'] in captions:
            captions[item['image_id']].append(item)
    order = sorted(images)
    random.Random(seed).shuffle(order)
    records, excluded_long_caption_ids = [], []
    for image_id in order:
        if len(records) == count:
            break
        candidates = sorted(captions[image_id], key=lambda item: item['id'])
        caption = candidates[0]['caption']
        if encoded_length(caption) > max_caption_tokens:
            excluded_long_caption_ids.append(image_id)
            continue
        records.append(dict(image_id=image_id, split=split,
                            caption_id=candidates[0]['id'], caption=caption,
                            all_captions=[item['caption'] for item in candidates],
                            url=f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{images[image_id]["file_name"]}'))
    if len(records) != count:
        raise ValueError('Requested more images than available after excluding over-length captions')
    return records, excluded_long_caption_ids


def download_records(records, downloads, workers):
    def fetch(record):
        path = downloads / f'{record["image_id"]:012d}.jpg'
        download(record['url'], path)
        record['sha256'] = sha256(path)
        with Image.open(path) as image:
            record['original_size_wh'] = list(image.size)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, record): record for record in records}
        for index, future in enumerate(as_completed(futures)):
            future.result()
            print(f'downloaded {index + 1}/{len(records)}', flush=True)


def load_tensor(path):
    with Image.open(path) as image:
        return preprocess_image(image)


def encode_records(records, downloads, encoder, device, batch_size):
    encoded = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        paths = [downloads / f'{record["image_id"]:012d}.jpg' for record in chunk]
        inputs = torch.stack([load_tensor(path) for path in paths]).to(device)
        encoded.append(encoder(inputs).cpu())
        print(f'encoded {min(start + batch_size, len(records))}/{len(records)}', flush=True)
    return torch.cat(encoded, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--train-size', type=int, default=32)
    parser.add_argument('--val-size', type=int, default=8)
    parser.add_argument('--test-size', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device for frozen-encoder feature extraction.')
    parser.add_argument('--download-workers', type=int, default=8,
                        help='Concurrent image downloads; scale up for large caches.')
    parser.add_argument('--encode-batch-size', type=int, default=32,
                        help='Images per encoder forward pass.')
    args = parser.parse_args()
    if min(args.download_workers, args.encode_batch_size) < 1:
        raise ValueError('Download workers and encode batch size must be positive')
    torch.set_num_threads(1)
    downloads = args.work_dir / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    annotations = downloads / 'annotations_trainval2017.zip'
    download(ANNOTATIONS_URL, annotations)
    with zipfile.ZipFile(annotations) as archive:
        train, train_excluded = choose_records(archive, 'train2017', args.train_size, args.seed)
        held_out, held_out_excluded = choose_records(archive, 'val2017', args.val_size + args.test_size,
                                  args.seed + 1, INSPECTION_IDS)
    val, test = held_out[:args.val_size], held_out[args.val_size:]
    lengths = [encoded_length(record['caption']) for record in train + val]
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = downloads / Path(weights.url).name
    download(weights.url, checkpoint)
    weight_hash = sha256(checkpoint)
    if not weight_hash.startswith('f37072fd'):
        raise ValueError('Unexpected official ResNet-18 weights')
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    encoder = FrozenEncoder(weights=None).to(args.device)
    encoder.layers.load_state_dict(torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
        backbone.maxpool, backbone.layer1, backbone.layer2).state_dict())
    features = {}
    for name, records in [('train', train), ('val', val)]:
        download_records(records, downloads, args.download_workers)
        features[name] = encode_records(records, downloads, encoder, args.device, args.encode_batch_size)
        print(f'{name} features {len(records)}/{len(records)}', flush=True)
    # Test captions are never retained, tokenized, encoded, or evaluated.
    manifest = dict(seed=args.seed, train=train, val=val,
        reserved_test_ids=[record['image_id'] for record in test],
        excluded_inspection_ids=sorted(INSPECTION_IDS),
        excluded_long_caption_ids=dict(train=train_excluded, held_out=held_out_excluded),
        selection='shuffle sorted image IDs with random.Random(seed); val2017 seed+1; fixed caption is lowest annotation ID; '
                  'images whose lowest-ID caption exceeds max_caption_tokens are skipped during selection, not truncated',
        tokenizer='tiktoken gpt2 BPE; vocab_size=50304 (50257 real ids padded for efficiency); '
                  'bos=50257, eos=50256 (tiktoken eot), pad=50258 (reserved, unused by the base BPE)',
        vocab_size=VOCAB_SIZE,
        max_caption_tokens=64, max_observed_caption_tokens=max(lengths), truncated_captions=0,
        preprocessing=dict(resize=[224, 224], interpolation='bilinear', antialias=True,
                           color='RGB', exif_transpose=False, mean=[.485,.456,.406], std=[.229,.224,.225]),
        encoder=dict(weights='ResNet18_Weights.IMAGENET1K_V1', url=weights.url, sha256=weight_hash,
                     layer='layer2', pooling='adaptive_avg_pool2d(14,14)', frozen=True),
        annotations=dict(url=ANNOTATIONS_URL, sha256=sha256(annotations)),
        versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__))
    path = args.manifest_dir / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    features['manifest_sha256'] = sha256(path)
    features['train_ids'] = [record['image_id'] for record in train]
    features['val_ids'] = [record['image_id'] for record in val]
    torch.save(features, args.work_dir / 'features.pt')
    print(f'Saved {len(train)} training / {len(val)} validation images', flush=True)


if __name__ == '__main__':
    main()
