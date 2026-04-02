import torch

# Define CNN model.
class CnnNet(torch.nn.Module):
    def __init__(self, image_size, num_classes):
        """
        Args:
            image_size (list): Shape of input image [C, H, W]
            num_classes (int): Number of output classes
        """
        super(CnnNet, self).__init__()
        in_channels = image_size[0]
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=5),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        # Calculate flattened size dynamically.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            self.flattened_size = x.numel() // x.size(0)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.flattened_size, 128),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, C, H, W]
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
