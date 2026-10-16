"""One-off: visualize ResNet-18 layer4 activations for 3 COCO train images as heatmaps."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from captioning.vision import FrozenEncoder, preprocess_image

DOWNLOADS = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/downloads')
IMAGE_IDS = [197336, 311192, 89181]
OUT_PATH = Path('/home/users/yizzhang/memory_mosaics/reports/layer4_visualization.png')

encoder = FrozenEncoder(layer='layer4', pool_size=7)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for col, image_id in enumerate(IMAGE_IDS):
    path = DOWNLOADS / f'{image_id:012d}.jpg'
    image = Image.open(path).convert('RGB')
    tensor = preprocess_image(image).unsqueeze(0)
    with torch.no_grad():
        features = encoder(tensor)  # (1, 512, 7, 7)
    activation = features[0].mean(dim=0)  # mean over channels -> (7, 7)

    axes[0, col].imshow(image)
    axes[0, col].set_title(f'image_id={image_id}')
    axes[0, col].axis('off')

    axes[1, col].imshow(image)
    axes[1, col].imshow(activation, cmap='jet', alpha=0.5, extent=(0, image.width, image.height, 0))
    axes[1, col].set_title('layer4 mean activation')
    axes[1, col].axis('off')

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
print(f'saved to {OUT_PATH}')
