"""One-off: raw per-channel patch feature maps from CLIP ViT-B/16's frozen visual encoder
(the standard frozen-encoder choice in modern captioning/VLM adapters: BLIP-2, LLaVA, Flamingo),
for the same 3 COCO train images, for comparison against ResNet-18 layer2/layer4 and VGG16."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

DOWNLOADS = Path('/scratch/users/yizzhang/memory_mosaics/caption_pilot_10k/downloads')
IMAGE_IDS = [197336, 311192, 89181]
NUM_CHANNELS = 8

processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16')
model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16').eval()
model.requires_grad_(False)
grid = model.config.image_size // model.config.patch_size  # 224 / 16 = 14

fig, axes = plt.subplots(len(IMAGE_IDS), NUM_CHANNELS + 1,
                          figsize=(2 * (NUM_CHANNELS + 1), 2 * len(IMAGE_IDS)))
for row, image_id in enumerate(IMAGE_IDS):
    image = Image.open(DOWNLOADS / f'{image_id:012d}.jpg').convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
    patch_tokens = out.last_hidden_state[0, 1:]  # drop CLS token -> (196, 768)
    features = patch_tokens.T.reshape(-1, grid, grid)  # (768, 14, 14)
    axes[row, 0].imshow(image)
    axes[row, 0].set_ylabel(f'{image_id}', fontsize=8)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    for col in range(NUM_CHANNELS):
        ax = axes[row, col + 1]
        ax.imshow(features[col], cmap='viridis')
        if row == 0:
            ax.set_title(f'ch{col}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'clip_vit_b16: raw per-channel patch feature maps ({grid}x{grid}, '
             f'{features.shape[0]} channels total, first {NUM_CHANNELS} shown)')
fig.tight_layout()
out_path = Path('/home/users/yizzhang/memory_mosaics/reports/feature_maps_clip_vit_b16.png')
fig.savefig(out_path, dpi=150)
print(f'saved to {out_path}')
