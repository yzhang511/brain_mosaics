"""One-off: visualize VGG16 last-conv-block activations (14x14x512, matching Show-Attend-Tell's
annotation vectors) for the same 3 COCO train images used in visualize_layer4.py, for comparison."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

from captioning.vision import preprocess_image

DOWNLOADS = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/downloads')
IMAGE_IDS = [197336, 311192, 89181]
OUT_PATH = Path('/home/users/yizzhang/memory_mosaics/reports/vgg16_visualization.png')

backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# features[:30] = conv1_1 ... conv5_3 + ReLU, stopping before the 5th maxpool: 14x14x512,
# matching Show-Attend-Tell's VGG19 conv5_4 annotation-vector resolution.
encoder = backbone.features[:30].eval()
encoder.requires_grad_(False)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for col, image_id in enumerate(IMAGE_IDS):
    path = DOWNLOADS / f'{image_id:012d}.jpg'
    image = Image.open(path).convert('RGB')
    tensor = preprocess_image(image).unsqueeze(0)
    with torch.no_grad():
        features = encoder(tensor)  # (1, 512, 14, 14)
    activation = features[0].mean(dim=0)  # mean over channels -> (14, 14)

    axes[0, col].imshow(image)
    axes[0, col].set_title(f'image_id={image_id}')
    axes[0, col].axis('off')

    axes[1, col].imshow(image)
    axes[1, col].imshow(activation, cmap='jet', alpha=0.5, extent=(0, image.width, image.height, 0))
    axes[1, col].set_title('vgg16 conv5_3 mean activation')
    axes[1, col].axis('off')

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
print(f'saved to {OUT_PATH}')
