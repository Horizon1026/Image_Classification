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
        # x: [batch_size, in_channels, H, W]
        out = self.conv1(x)
        # out: [batch_size, out_channels, H/stride, W/stride]
        out = self.bn1(out)
        out = torch.nn.ReLU()(out)
        out = self.conv2(out)
        # out: [batch_size, out_channels, H/stride, W/stride]
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
    def __init__(self, in_channels, num_classes, init_channels = 16):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(init_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = MakeLayer(init_channels, init_channels, 2, 1)
        self.layer3 = MakeLayer(init_channels, init_channels * 2, 2, 2)
        self.layer4 = MakeLayer(init_channels * 2, init_channels * 4, 2, 2)
        self.fc = torch.nn.Linear(init_channels * 4, num_classes)

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        B = x.shape[0]
        # self.layer1: Conv2d(1, 16, 3, 1, 1) -> [B, 16, 28, 28]
        # MaxPool2d(3, 2, 1) -> [B, 16, 14, 14]
        x = self.layer1(x)
        # self.layer2: MakeLayer(16, 16, 2, 1) -> [B, 16, 14, 14]
        x = self.layer2(x)
        # self.layer3: MakeLayer(16, 32, 2, 2) -> [B, 32, 7, 7]
        x = self.layer3(x)
        # self.layer4: MakeLayer(32, 64, 2, 2) -> [B, 64, 4, 4]
        x = self.layer4(x)
        # AvgPool2d(4) -> [B, 64, 1, 1]
        x = torch.nn.AvgPool2d(kernel_size=4)(x)
        # Flatten: [B, 64]
        x = x.reshape(B, -1)
        # self.fc: Linear(64, num_classes) -> [B, num_classes]
        x = self.fc(x)
        return x
