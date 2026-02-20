import torch

# Define MLP model.
class MlpNet(torch.nn.Module):
    def __init__(self, image_size, dim_hidden_layer, num_classes):
        super(MlpNet, self).__init__()
        assert len(image_size) == 3, 'Image size must be list of [channels, rows, cols]'
        image_channels, image_rows, image_cols = image_size
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(image_channels * image_rows * image_cols, dim_hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_hidden_layer, num_classes),
        )

    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        batch_size = x.size(0)
        # Flatten: [batch_size, 1 * 28 * 28] = [batch_size, 784]
        x = x.view(batch_size, -1)
        # self.fc: Linear(784, dim_hidden_layer) -> Linear(dim_hidden_layer, num_classes) -> [batch_size, num_classes]
        x = self.fc(x)
        return x
