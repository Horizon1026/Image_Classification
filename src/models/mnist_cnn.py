import torch
from torch import nn

from model import ClassificationOutput


# Classify single-channel digit images with a compact convolutional network.
class MnistCNN(nn.Module):
    """Classify 28x28 grayscale digit images."""

    # Build grayscale feature extraction and a spatially aware classifier.
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        # Preserve the pooled spatial layout for final class prediction.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    # Return differentiable class logits in the framework output structure.
    def forward(self, images: torch.Tensor) -> ClassificationOutput:
        return ClassificationOutput(self.classifier(self.features(images)))
