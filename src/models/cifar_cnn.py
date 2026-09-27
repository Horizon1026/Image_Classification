import torch
from torch import nn
from model import ClassificationOutput


# Define a compact convolutional classifier for CIFAR images.
class CifarCNN(nn.Module):
    """Small CIFAR-10 classifier producing one logit per class."""

    # Build image features and the final class projection.
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Extract spatial features with three convolutional stages.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # Map pooled features to class logits.
        self.classifier = nn.Linear(128, num_classes)

    # Wrap class logits in the shared classification output contract.
    def forward(self, images: torch.Tensor) -> ClassificationOutput:
        return ClassificationOutput(self.classifier(self.features(images).flatten(1)))
