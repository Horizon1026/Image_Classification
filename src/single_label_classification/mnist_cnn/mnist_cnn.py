import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import numpy as np
from matplotlib import pyplot as plt

def LoadDataset(dataset_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))])
    train_dataset = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transform)
    return train_dataset, test_dataset

def GenerateDataLoader(batch_size, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define CNN model.
class CnnNet(torch.nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

def TrainModel(model, data_loader, max_epoch):
    learning_rate = 0.01
    momentum = 0
    # final_dx = - dx * lr + v * momemtum. v is previous (- dx * lr)
    criterion = torch.nn.CrossEntropyLoss()
    optimizor = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(data_loader, 0):
            inputs, target = data
            optimizor.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizor.step()

            if batch_idx % 200 == 0:
                print(">> epoch %d, batch idx %d, loss %.4f." % (epoch, batch_idx, loss.item()))

def TestModel(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
        accuracy = correct / total
        print(">> Accuracy on test dataset [%.2f %%]" % (accuracy * 100))

if __name__ == '__main__':
    print(torch.cuda.is_available())
    train_dataset, test_dataset = LoadDataset('/mnt/d/My_Github/Datasets/')
    train_loader, test_loader = GenerateDataLoader(batch_size=20, train_dataset=train_dataset, test_dataset=test_dataset)
    model = CnnNet()
    TrainModel(model, train_loader, max_epoch=5)
    TestModel(model, test_loader)
