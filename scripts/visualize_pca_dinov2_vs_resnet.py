"""One-off: PCA-RGB visualization of DINOv2 ViT-S/14 patch features (the standard way its dense
features are shown, per the DINOv2 paper) vs. the same for ResNet-18 layer4, side by side, for
the same 3 COCO train images."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from captioning.vision import FrozenEncoder, preprocess_image

DOWNLOADS = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/downloads')
IMAGE_IDS = [197336, 311192, 89181]

dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dino_model = AutoModel.from_pretrained('facebook/dinov2-small').eval()
dino_model.requires_grad_(False)
patch_size = dino_model.config.patch_size

resnet_encoder = FrozenEncoder(layer='layer4', pool_size=7)


def pca_rgb(features):
    # features: (C, H, W) -> top-3 principal components across channels, min-max normalized to RGB
    c, h, w = features.shape
    flat = features.reshape(c, h * w).T  # (H*W, C)
    flat = flat - flat.mean(dim=0, keepdim=True)
    u, s, v = torch.pca_lowrank(flat, q=3)
    projected = flat @ v[:, :3]  # (H*W, 3)
    projected -= projected.min(dim=0, keepdim=True).values
    projected /= projected.max(dim=0, keepdim=True).values.clamp(min=1e-6)
    return projected.T.reshape(3, h, w).permute(1, 2, 0).numpy()


fig, axes = plt.subplots(len(IMAGE_IDS), 3, figsize=(9, 3 * len(IMAGE_IDS)))
for row, image_id in enumerate(IMAGE_IDS):
    image = Image.open(DOWNLOADS / f'{image_id:012d}.jpg').convert('RGB')

    dino_inputs = dino_processor(images=image, return_tensors='pt')
    grid = dino_inputs['pixel_values'].shape[-1] // patch_size
    with torch.no_grad():
        dino_out = dino_model(**dino_inputs)
    dino_features = dino_out.last_hidden_state[0, 1:].T.reshape(-1, grid, grid)

    resnet_tensor = preprocess_image(image).unsqueeze(0)
    with torch.no_grad():
        resnet_features = resnet_encoder(resnet_tensor)[0]

    axes[row, 0].imshow(image)
    axes[row, 0].set_ylabel(f'{image_id}', fontsize=8)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    axes[row, 1].imshow(pca_rgb(dino_features))
    axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
    axes[row, 2].imshow(pca_rgb(resnet_features))
    axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
    if row == 0:
        axes[row, 0].set_title('image')
        axes[row, 1].set_title('dinov2_vits14 (PCA-RGB, top 3 of 384)')
        axes[row, 2].set_title('resnet18_layer4 (PCA-RGB, top 3 of 512)')

fig.tight_layout()
out_path = Path('/home/users/yizzhang/memory_mosaics/reports/pca_dinov2_vs_resnet.png')
fig.savefig(out_path, dpi=150)
print(f'saved to {out_path}')
