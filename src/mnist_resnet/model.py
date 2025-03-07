import torch


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
