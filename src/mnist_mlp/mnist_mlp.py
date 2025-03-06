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

# Define MLP model.
class MlpNet(torch.nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        batch_size = x.size(0)
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
    print('>> Test MLP model on MNIST dataset.')
    print(torch.cuda.is_available())
    train_dataset, test_dataset = LoadDataset('./dataset/')
    train_loader, test_loader = GenerateDataLoader(batch_size=64, train_dataset=train_dataset, test_dataset=test_dataset)
    model = MlpNet()
    TrainModel(model, train_loader, max_epoch=1)
    TestModel(model, test_loader)
