import torch
import torch.nn as nn


class Conv2DLSTM(nn.Module):

    def __init__(self):
        super(Conv2DLSTM, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(96, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, (3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, (3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(512, 32, (1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(10 * 10, 128//2, bidirectional=True, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(128, 51), nn.Softmax())

    def forward(self, x):
        # (B,C,N,H,W) -> (B,N,C,H,W)
        b, c, n, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)

        x = self.feature_extractor(x)
        x = x.reshape(b, n, -1)
        x, (hidden, cell) = self.rnn(x)
        x = self.mlp(x[:, -1, :])
        return x
