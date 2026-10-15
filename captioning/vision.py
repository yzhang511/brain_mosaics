"""Frozen ImageNet ResNet-18 layer2 and the agreed full-frame preprocessing."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode


def preprocess_image(image):
    transform = Compose([
        Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image.convert('RGB'))


class FrozenEncoder(nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        backbone = resnet18(weights=weights)
        self.layers = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                    backbone.maxpool, backbone.layer1, backbone.layer2)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Preserve frozen batch-normalization statistics even under parent.train().
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        return F.adaptive_avg_pool2d(self.layers(images), (14, 14))
