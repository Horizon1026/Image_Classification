import torch
from mnist_resnet.model import *
from train_test_on_mnist import *


if __name__ == '__main__':
    print('\033[93m' + '>> Test ResNet model on MNIST dataset.' + '\033[0m')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = LoadDataset('./dataset/')
    train_loader, test_loader = GenerateDataLoader(batch_size=64, train_dataset=train_dataset, test_dataset=test_dataset)
    model = ResNet()
    TrainModel(device, model, train_loader, max_epoch=1)
    TestModel(device, model, test_loader)
