import torch


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
