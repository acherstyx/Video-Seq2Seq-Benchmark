import torch
import torch.nn as nn


class SlowFast(nn.Module):

    def __init__(self):
        super(SlowFast, self).__init__()
        self.slow_data = nn.Sequential(nn.Conv3d(3, 3, (1, 1, 1), stride=(8, 1, 1)),
                                       nn.Conv3d(3, 64, (1, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3)),
                                       nn.MaxPool3d((1, 3, 3), (1, 2, 2), padding=(0, 1, 1)))
        self.fast_data = nn.Sequential(nn.Conv3d(3, 3, (1, 1, 1), stride=(1, 1, 1)),
                                       nn.Conv3d(3, 8, (5, 7, 7), stride=(1, 1, 1), padding=(2, 3, 3)),
                                       nn.MaxPool3d((1, 3, 3), (1, 2, 2), padding=(0, 1, 1)))
        self.slow_res2 = [
            nn.Sequential(
                nn.Conv3d(64, 64, (1, 1, 1), stride=(1, 2, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv3d(64, 256, (1, 1, 1)),
                nn.BatchNorm2d(256))
        ]

        self.relu = nn.ReLU()

    def res_block(self, slow_layer, fast_layer):
        pass

    def forward(self, x):
        # (B,C,N,H,W)
        slow = self.slow_data(x)
        fast = self.fast_data(x)

        return slow, fast
