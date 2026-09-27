# Export both dataset-specific classifier implementations.
from .cifar_cnn import CifarCNN
from .mnist_cnn import MnistCNN

__all__ = ["CifarCNN", "MnistCNN"]
