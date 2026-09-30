"""Verify ResNet-18 supports both image classification dataset contracts."""

import unittest

import torch

from model import ClassificationOutput
from models import ResNet18Classifier


class ResNet18ClassifierTest(unittest.TestCase):
    def test_rgb_and_grayscale_logits_support_backpropagation(self):
        for channels, size in ((3, 32), (1, 28)):
            with self.subTest(channels=channels):
                model = ResNet18Classifier(in_channels=channels, num_classes=10)
                images = torch.randn(2, channels, size, size)
                output = model(images)
                self.assertIsInstance(output, ClassificationOutput)
                self.assertEqual(tuple(output.logits.shape), (2, 10))
                output.logits.square().mean().backward()
                self.assertIsNotNone(model.network.conv1.weight.grad)
                self.assertEqual(model.network.conv1.stride, (1, 1))
                self.assertIsInstance(model.network.maxpool, torch.nn.Identity)


if __name__ == "__main__":
    unittest.main()
