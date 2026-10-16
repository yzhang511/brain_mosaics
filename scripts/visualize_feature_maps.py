"""One-off: visualize individual per-channel feature maps (not a channel-averaged activation
map) from ResNet-18 layer4 and VGG16 conv5_3, for the same 3 COCO train images."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

from captioning.vision import FrozenEncoder, preprocess_image

DOWNLOADS = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/downloads')
IMAGE_IDS = [197336, 311192, 89181]
NUM_CHANNELS = 8  # first 8 of 512 channels, shown raw (no overlay, no averaging)

resnet_encoder = FrozenEncoder(layer='layer4', pool_size=7)
vgg_encoder = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:30].eval()
vgg_encoder.requires_grad_(False)

images = {image_id: Image.open(DOWNLOADS / f'{image_id:012d}.jpg').convert('RGB') for image_id in IMAGE_IDS}
tensors = {image_id: preprocess_image(image).unsqueeze(0) for image_id, image in images.items()}

for name, encoder in [('resnet18_layer4', resnet_encoder), ('vgg16_conv5_3', vgg_encoder)]:
    fig, axes = plt.subplots(len(IMAGE_IDS), NUM_CHANNELS + 1,
                              figsize=(2 * (NUM_CHANNELS + 1), 2 * len(IMAGE_IDS)))
    for row, image_id in enumerate(IMAGE_IDS):
        with torch.no_grad():
            features = encoder(tensors[image_id])[0]  # (C, H, W), raw per-channel maps
        axes[row, 0].imshow(images[image_id])
        axes[row, 0].set_title(f'image_id={image_id}' if row == 0 else '')
        axes[row, 0].set_ylabel(f'{image_id}', fontsize=8)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for col in range(NUM_CHANNELS):
            ax = axes[row, col + 1]
            ax.imshow(features[col], cmap='viridis')
            if row == 0:
                ax.set_title(f'ch{col}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{name}: raw per-channel feature maps ({features.shape[1]}x{features.shape[2]}, '
                 f'{features.shape[0]} channels total, first {NUM_CHANNELS} shown)')
    fig.tight_layout()
    out_path = Path(f'/home/users/yizzhang/memory_mosaics/reports/feature_maps_{name}.png')
    fig.savefig(out_path, dpi=150)
    print(f'saved to {out_path}')
