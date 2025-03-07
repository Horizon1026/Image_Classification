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

def TrainModel(device, model, data_loader, max_epoch):
    learning_rate = 0.01
    momentum = 0
    # final_dx = - dx * lr + v * momemtum. v is previous (- dx * lr)
    criterion = torch.nn.CrossEntropyLoss()
    optimizor = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model.train()
    model.to(device)
    for epoch in range(max_epoch):
        for batch_idx, data in enumerate(data_loader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            optimizor.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizor.step()

            if batch_idx % 100 == 0:
                print(">> epoch %d, batch idx %d, loss %.4f." % (epoch, batch_idx, loss.item()))

def TestModel(device, model, data_loader):
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in data_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
        accuracy = correct / total
        print(">> Accuracy on test dataset [%.2f %%]" % (accuracy * 100))
