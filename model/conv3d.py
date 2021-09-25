import torch.nn as nn


class Conv3D(nn.Module):

    def __init__(self):
        super(Conv3D, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(3, 64, (3, 3, 3)),
            nn.Conv3d(64, 64, (3, 3, 3)),
            nn.AvgPool3d(3),
            nn.Conv3d(64, 128, (3, 3, 3)),
            nn.Conv3d(128, 128, (3, 3, 3)),
            nn.AvgPool3d(3),
            nn.Flatten(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.Linear(100, 51),
        )

    def forward(self, x):
        return self.net(x)
