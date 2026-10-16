"""Frozen visual encoders and the agreed full-frame preprocessing.

Two encoder families are supported as pre-adapter feature taps:
- ResNet-18 (ImageNet-supervised, via FrozenEncoder/TAPS): layer2 (mid-level, texture/edge) or
  layer4 (final conv stage, object/semantic-level).
- DINOv2 (self-supervised ViT, via Dinov2Encoder/DINO_MODELS): patch-token features. A manual
  qualitative comparison (see captioning/JOINT_FINETUNING_PLAN.md) found DINOv2's PCA-projected
  patch features show cleaner object-level semantic clustering than either ResNet tap, at native
  16x16 resolution for a 224x224 input (vs. layer4's 7x7), motivating its use for the 100k run.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

# Submodule names and output channel count for each supported ResNet-18 feature tap. layer2 is
# the original mid-level (texture/edge) tap; layer4 is ResNet-18's final conv stage, giving more
# object-level semantic features at native 7x7 resolution (for 224x224 input) at the cost of a
# smaller spatial grid than layer2's native 28x28.
TAPS = {
    'layer2': (('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2'), 128),
    'layer4': (('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'), 512),
}

# HuggingFace repo, output channel count, and ViT patch size for each supported DINOv2 variant.
DINO_MODELS = {
    'dinov2_vits14': dict(repo='facebook/dinov2-small', feature_dim=384, patch_size=14),
}

# Feature-tap name -> output channel count, across both encoder families, for callers (prepare.py,
# train.py) that need a single lookup regardless of which family a tap belongs to.
FEATURE_DIMS = {**{name: dim for name, (_, dim) in TAPS.items()},
                **{name: config['feature_dim'] for name, config in DINO_MODELS.items()}}


def preprocess_image(image):
    transform = Compose([
        Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image.convert('RGB'))


class FrozenEncoder(nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1, layer='layer2', pool_size=14):
        super().__init__()
        backbone = resnet18(weights=weights)
        names, self.feature_dim = TAPS[layer]
        self.layers = nn.Sequential(*(getattr(backbone, name) for name in names))
        self.pool_size = pool_size
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Preserve frozen batch-normalization statistics even under parent.train().
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        return F.adaptive_avg_pool2d(self.layers(images), (self.pool_size, self.pool_size))


class Dinov2Encoder(nn.Module):
    """Frozen DINOv2 ViT patch-token encoder. Matches FrozenEncoder's (B, C, H, W) output
    contract (dropping the CLS token and reshaping patch tokens to a spatial grid) so prepare.py
    and train.py can treat either encoder family interchangeably."""
    def __init__(self, name='dinov2_vits14', pool_size=None):
        super().__init__()
        from transformers import AutoModel  # lazy: only required when this encoder is used
        config = DINO_MODELS[name]
        self.model = AutoModel.from_pretrained(config['repo'])
        self.feature_dim = config['feature_dim']
        self.patch_size = config['patch_size']
        self.pool_size = pool_size
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        grid = images.shape[-1] // self.patch_size
        patch_tokens = self.model(pixel_values=images).last_hidden_state[:, 1:]  # drop CLS token
        features = patch_tokens.transpose(1, 2).reshape(images.shape[0], self.feature_dim, grid, grid)
        pool_size = grid if self.pool_size is None else self.pool_size
        return F.adaptive_avg_pool2d(features, (pool_size, pool_size))
