import torch.autograd
import torch.nn as nn


class Conv3D(nn.Module):

    def __init__(self):
        super(Conv3D, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(3, 16, (3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AvgPool3d(3),
            nn.Conv3d(16, 64, (3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AvgPool3d(3),
            nn.Flatten(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.Linear(100, 51),
        )

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        return self.net(x)
