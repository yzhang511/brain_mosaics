"""Inspect frozen ResNet-18 on a few COCO images; never builds a training cache.

Run with --cache-dir on scratch and --output-dir for the small retained report.
Only the four declared images and the official model weights are downloaded.
"""
import argparse
import hashlib
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

IDS = [139, 39769, 252219, 397133]  # Fixed before inspecting activations.
LAYERS = ['layer1', 'layer2', 'layer3']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + '.partial')
    try:
        with urllib.request.urlopen(url, timeout=45) as response, open(tmp, 'wb') as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def savefig(fig, path):
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    start = time.perf_counter()
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = args.cache_dir / Path(weights.url).name
    print('Downloading/checking official weights', flush=True)
    download(weights.url, checkpoint)
    digest = sha256(checkpoint)
    assert digest.startswith('f37072fd'), 'Unexpected official ResNet-18 weight hash'
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.requires_grad_(False).eval()
    sources, originals, crops, resized = [], [], [], []
    for image_id in IDS:
        name = f'{image_id:012d}.jpg'
        url = f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}'
        path = args.cache_dir / name
        print(f'Downloading/checking COCO {image_id}', flush=True)
        download(url, path)
        with Image.open(path) as image:
            original = ImageOps.exif_transpose(image).convert('RGB')
        originals.append(original)
        crops.append(TF.center_crop(TF.resize(original, 256, InterpolationMode.BILINEAR), [224, 224]))
        resized.append(TF.resize(original, [224, 224], InterpolationMode.BILINEAR))
        sources.append({'coco_id': image_id, 'split': 'val2017', 'url': url,
                        'sha256': sha256(path), 'original_size_wh': list(original.size),
                        'role': 'development inspection; exclude from future held-out test'})
    fig, axes = plt.subplots(len(IDS), 3, figsize=(10, 11))
    for i in range(len(IDS)):
        for j, (im, label) in enumerate(zip([originals[i], crops[i], resized[i]],
                ['Original', 'Official resize + center crop', 'Full-frame resize (distorts aspect)'])):
            axes[i, j].imshow(im)
            axes[i, j].set_title(f'COCO {IDS[i]}: {label}', fontsize=9)
            axes[i, j].axis('off')
    fig.suptitle('Preprocessing changes which scene information reaches the encoder')
    fig.tight_layout()
    savefig(fig, out / 'preprocessing.png')

    # Both preprocessing choices are inspected; neither is silently selected for training.
    raw, pooled, timings = {}, {}, {}
    for mode, images in [('full_frame', resized), ('center_crop', crops)]:
        batch = torch.stack([TF.normalize(TF.to_tensor(im), MEAN, STD) for im in images])
        t0 = time.perf_counter()
        with torch.inference_mode():
            x = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
            for layer in LAYERS:
                x = getattr(model, layer)(x)
                raw[(mode, layer)] = x.clone()
                pooled[(mode, layer)] = F.adaptive_avg_pool2d(x, (14, 14))
        timings[mode] = time.perf_counter() - t0
    pca, basis_info = {}, {}
    for layer in LAYERS:
        # One centered PCA basis per layer, shared across images AND preprocessing modes.
        samples = torch.cat([pooled[(mode, layer)].permute(0, 2, 3, 1).reshape(-1, pooled[(mode, layer)].shape[1])
                             for mode in ['full_frame', 'center_crop']]).numpy().astype(np.float64)
        mean = samples.mean(0)
        centered = samples - mean
        eigvals, eigvecs = np.linalg.eigh(centered.T @ centered / (len(samples) - 1))
        basis = eigvecs[:, -3:][:, ::-1].copy()
        for c in range(3):
            if basis[np.argmax(abs(basis[:, c])), c] < 0:
                basis[:, c] *= -1
        projected = centered @ basis
        low, high = np.percentile(projected, [1, 99], axis=0)
        rgb = np.clip((projected - low) / np.maximum(high - low, 1e-8), 0, 1)
        rgb = rgb.reshape(2, len(IDS), 14, 14, 3)
        for m, mode in enumerate(['full_frame', 'center_crop']):
            pca[(mode, layer)] = rgb[m]
        basis_info[layer] = {'mean': mean.tolist(), 'components': basis.tolist(),
            'explained_variance_ratio': (eigvals[-3:][::-1] / eigvals.sum()).tolist(),
            'rgb_clip_percentiles': [1, 99], 'rgb_low': low.tolist(), 'rgb_high': high.tolist()}
    metrics = {}
    for mode, images in [('full_frame', resized), ('center_crop', crops)]:
        for i, image_id in enumerate(IDS):
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for row, layer in enumerate(LAYERS):
                feature = raw[(mode, layer)][i]
                pool = pooled[(mode, layer)][i]
                rms = feature.square().mean(0).sqrt().numpy()
                prms = pool.square().mean(0).sqrt().numpy()
                # Shared scale for every image and both resolutions within this layer/mode.
                maximum = float(raw[(mode, layer)].square().mean(1).sqrt().max())
                query = pool[:, 7, 7]
                similarity = F.cosine_similarity(pool, query[:, None, None], dim=0).numpy()
                axes[row, 0].imshow(images[i])
                axes[row, 0].plot(120, 120, 'x', color='red', markersize=10)
                axes[row, 0].set_title(f'{layer}: {tuple(feature.shape)}\nRed x: similarity query at token (7,7)', fontsize=9)
                artist = axes[row, 1].imshow(rms, cmap='magma', vmin=0, vmax=maximum, interpolation='nearest')
                axes[row, 1].set_title('Native activation RMS', fontsize=10)
                fig.colorbar(artist, ax=axes[row, 1], fraction=.046, pad=.04)
                axes[row, 2].imshow(prms, cmap='magma', vmin=0, vmax=maximum, interpolation='nearest')
                axes[row, 2].set_title('After pooling to 14 x 14\nSame intensity scale as native', fontsize=9)
                axes[row, 3].imshow(pca[(mode, layer)][i], interpolation='nearest')
                axes[row, 3].set_title('Three PCA components as RGB\nColors are NOT semantic labels', fontsize=9)
                artist = axes[row, 4].imshow(similarity, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
                axes[row, 4].plot(7, 7, 'rx', markersize=8)
                axes[row, 4].set_title('Cosine similarity to marked token\n0 = orthogonal; 1 = identical direction', fontsize=9)
                fig.colorbar(artist, ax=axes[row, 4], fraction=.046, pad=.04)
                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                reconstructed = F.interpolate(pool[None], size=feature.shape[-2:], mode='nearest')[0]
                relative_error = float((feature - reconstructed).square().sum() / feature.square().sum().clamp_min(1e-12))
                metrics[f'{mode}/{image_id}/{layer}'] = {
                    'native_shape': list(feature.shape), 'pooled_shape': list(pool.shape),
                    'pooling_relative_squared_reconstruction_error': relative_error,
                    'note': 'Within-feature spatial loss diagnostic, NOT semantic loss or cross-layer quality.'}
            fig.suptitle(f'COCO {image_id} | {mode} | Frozen ImageNet ResNet-18\n'
                         'Each row has its own activation scale; similarity and PCA are not attention or segmentation.', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .94))
            savefig(fig, out / f'{image_id:012d}_{mode}.png')
    selected = {}
    for layer in LAYERS:
        feature = raw[('full_frame', layer)]
        channels = feature.mean(dim=(0, 2, 3)).topk(4).indices.tolist()
        selected[layer] = channels
        fig, axes = plt.subplots(len(IDS), 5, figsize=(14, 10))
        for i, image_id in enumerate(IDS):
            axes[i, 0].imshow(resized[i])
            axes[i, 0].set_title(f'COCO {image_id}', fontsize=10)
            for j, channel in enumerate(channels, 1):
                v = feature[i, channel].numpy()
                axes[i, j].imshow(v, cmap='magma', vmin=0, vmax=float(feature[:, channel].max()), interpolation='nearest')
                axes[i, j].set_title(f'Channel {channel}', fontsize=10)
            for ax in axes[i]:
                ax.axis('off')
        fig.suptitle(f'{layer}: four channels with highest mean activation across ALL four images\n'
                     'Full-frame input. Same channel and intensity scale within each column; no object labels assigned.')
        fig.tight_layout(rect=(0, 0, 1, .94))
        savefig(fig, out / f'channels_{layer}.png')
    metadata = {'sources': sources, 'weights': str(weights), 'weights_url': weights.url,
        'weights_sha256': digest, 'device': 'cpu', 'threads': 1, 'seed': 0,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'torchvision': torchvision.__version__, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'preprocessing': {'full_frame': 'Bilinear direct resize to 224x224; aspect ratio changes',
                          'center_crop': 'Bilinear short-side resize to 256, center crop 224x224',
                          'normalization_mean': MEAN, 'normalization_std': STD},
        'pooling': 'adaptive average pool to 14x14 before trainable adapter (not implemented here)',
        'pca': basis_info, 'selected_channels': selected,
        'query_token_row_col': [7, 7], 'metrics': metrics,
        'forward_seconds_per_four_images': timings,
        'total_seconds_including_downloads_and_plots': time.perf_counter() - start,
        'limitations': ['Four development examples only; not a semantic evaluation.',
                        'PCA basis is shared within a layer, not across layers.',
                        'RMS measures activity, not semantic importance.',
                        'Spatial features have overlapping receptive fields.',
                        'No bulk features, caption model, or training created.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>ResNet-18 feature inspection</title>',
            '<style>body{font:17px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;line-height:1.5}img{width:100%;height:auto}details{margin:20px 0}summary{cursor:pointer;font-weight:bold}a{color:#1558a6}</style></head><body>',
            '<h1>Inspect the visual input before choosing it</h1>',
            '<p>Frozen ImageNet ResNet-18; four fixed COCO development examples. No captioner has been trained. '
            'Compare preprocessing, layers, and pooling. The eventual prefix length is 196 tokens; the layer remains undecided.</p>',
            '<p><b>Read the figures:</b> RMS shows activation magnitude, not semantic importance. '
            'PCA colors show three directions of variation, not object categories; colors are comparable within a layer only. '
            'Cosine similarity compares every pooled feature vector to the marked token; it is not attention or segmentation. '
            'Maps show spatial indices with overlapping receptive fields, not isolated image patches.</p>',
            '<h2>What does the network actually see?</h2><img src="preprocessing.png" alt="Original, cropped and full-frame inputs">']
    for image_id in IDS:
        html += [f'<h2>COCO {image_id}</h2>',
                 f'<img src="{image_id:012d}_full_frame.png" alt="Layer comparison for full-frame input">',
                 f'<details><summary>Compare official center-crop preprocessing</summary><img src="{image_id:012d}_center_crop.png" alt="Center-crop layer comparison"></details>']
    for layer in LAYERS:
        html += [f'<details><summary>{layer}: inspect individual channels</summary><img src="channels_{layer}.png" alt="Selected channel activations"></details>']
    html += ['<h2>Decision to discuss</h2><p>Which preprocessing preserves the objects and relationships of interest? '
             'Does pooling remove small spatial details? What remains distinguishable at each layer? '
             'These plots guide hypotheses; held-out semantic probes and controlled caption experiments must test them.</p>',
             '<p>Full-frame resizing distorts geometry; center cropping removes content. Neither has been selected for training. '
             'Aspect-preserving padding is another possible follow-up, with its own boundary effects.</p>',
             '<p><a href="metadata.json">IDs, sources, checksums, versions, PCA basis, selection rules and timing</a>. '
             'These inspection images must be excluded from the eventual held-out test set.</p></body></html>']
    (out / 'index.html').write_text('\n'.join(html))
    print(json.dumps({'report': str(out / 'index.html'), 'forward_seconds': timings,
                      'total_seconds': metadata['total_seconds_including_downloads_and_plots']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
