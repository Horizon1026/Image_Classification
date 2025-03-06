import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

def LoadDataset(dataset_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))])
    train_dataset = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transform)
    return train_dataset, test_dataset

def GenerateDataLoader(batch_size, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define ResNet model.
class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else torch.nn.Identity()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = torch.nn.ReLU()(out)
        return out

def MakeLayer(in_channels, out_channels, num_blocks, stride):
    layers = []
    layers.append(ResNetBlock(in_channels, out_channels, stride))
    for _ in range(num_blocks):
        layers.append(ResNetBlock(out_channels, out_channels))
    return torch.nn.Sequential(*layers)

class ResNet(torch.nn.Module):
    def __init__(self, num_classes = 10, init_channels = 16):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=init_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(init_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = MakeLayer(init_channels, init_channels, 2, 1)
        self.layer3 = MakeLayer(init_channels, init_channels * 2, 2, 2)
        self.layer4 = MakeLayer(init_channels * 2, init_channels * 4, 2, 2)
        self.fc = torch.nn.Linear(init_channels * 4, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.nn.AvgPool2d(kernel_size=4)(x)
        x = x.reshape(B, -1)
        x = self.fc(x)
        x = torch.nn.LogSoftmax(dim=1)(x)
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

            if batch_idx % 100 == 0:
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
    print('>> Test ResNet model on MNIST dataset.')
    print(torch.cuda.is_available())
    train_dataset, test_dataset = LoadDataset('./dataset/')
    train_loader, test_loader = GenerateDataLoader(batch_size=64, train_dataset=train_dataset, test_dataset=test_dataset)
    model = ResNet()
    TrainModel(model, train_loader, max_epoch=1)
    TestModel(model, test_loader)
