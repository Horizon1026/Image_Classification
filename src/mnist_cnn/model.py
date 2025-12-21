import torch


# Define CNN model.
class CnnNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CnnNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=5),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(484, 128),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        batch_size = x.size(0)
        # self.conv1: Conv2d(1, 2, kernel_size=5) -> [batch_size, 2, 24, 24]
        x = self.conv1(x)
        # self.conv2:
        # Conv2d(2, 4, kernel_size=3) -> [batch_size, 4, 22, 22]
        # MaxPool2d(kernel_size=2) -> [batch_size, 4, 11, 11]
        x = self.conv2(x)
        # Flatten: [batch_size, 4 * 11 * 11] = [batch_size, 484]
        x = x.view(batch_size, -1)
        # self.fc: Linear(484, 128) -> Linear(128, num_classes) -> [batch_size, num_classes]
        x = self.fc(x)
        return x
