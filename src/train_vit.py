import torch
from mnist_vit.model import *
from train_test_on_mnist import *


if __name__ == '__main__':
    print('\033[93m' + '>> Test ViT model on MNIST dataset.' + '\033[0m')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = LoadDataset('./dataset/')
    train_loader, test_loader = GenerateDataLoader(batch_size=64, train_dataset=train_dataset, test_dataset=test_dataset)

    image_size = [1, 28, 28]
    patch_size = [28, 28]
    model = ViTNet(
        image_size = image_size,
        patch_size = patch_size,
        dim_token = image_size[0] * patch_size[0] * patch_size[1],
        dim_hidden_layer = 128,
        num_heads = 1,
        num_layers = 3,
        num_classes = 10,
        dropout = 0,
        use_class_token = True,
    )
    TrainModel(device, model, train_loader, max_epoch=1)
    TestModel(device, model, test_loader)
