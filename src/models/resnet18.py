"""Small-image ResNet-18 classifier with the shared logits contract."""

import torch
from torch import nn
from torchvision.models import resnet18

from model import ClassificationOutput


class ResNet18Classifier(nn.Module):
    """Classify RGB or grayscale images without early spatial downsampling."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Preserve detail in 28x28 and 32x32 inputs with a small-image stem.
        self.network = resnet18(weights=None, num_classes=num_classes)
        self.network.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.network.maxpool = nn.Identity()

    def forward(self, images: torch.Tensor) -> ClassificationOutput:
        """Return one raw class logit per output category."""
        return ClassificationOutput(self.network(images))
