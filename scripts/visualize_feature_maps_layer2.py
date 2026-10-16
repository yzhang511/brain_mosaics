"""One-off: raw per-channel feature maps from ResNet-18 layer2 (native 28x28, 128 channels),
for the same 3 COCO train images, for comparison against layer4 and vgg16 conv5_3."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from captioning.vision import FrozenEncoder, preprocess_image

DOWNLOADS = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/downloads')
IMAGE_IDS = [197336, 311192, 89181]
NUM_CHANNELS = 8

encoder = FrozenEncoder(layer='layer2', pool_size=28)  # native resolution, no downsampling

fig, axes = plt.subplots(len(IMAGE_IDS), NUM_CHANNELS + 1,
                          figsize=(2 * (NUM_CHANNELS + 1), 2 * len(IMAGE_IDS)))
for row, image_id in enumerate(IMAGE_IDS):
    image = Image.open(DOWNLOADS / f'{image_id:012d}.jpg').convert('RGB')
    tensor = preprocess_image(image).unsqueeze(0)
    with torch.no_grad():
        features = encoder(tensor)[0]  # (128, 28, 28)
    axes[row, 0].imshow(image)
    axes[row, 0].set_ylabel(f'{image_id}', fontsize=8)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    for col in range(NUM_CHANNELS):
        ax = axes[row, col + 1]
        ax.imshow(features[col], cmap='viridis')
        if row == 0:
            ax.set_title(f'ch{col}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('resnet18_layer2: raw per-channel feature maps (28x28 native, 128 channels total, first 8 shown)')
fig.tight_layout()
out_path = Path('/home/users/yizzhang/memory_mosaics/reports/feature_maps_resnet18_layer2.png')
fig.savefig(out_path, dpi=150)
print(f'saved to {out_path}')
