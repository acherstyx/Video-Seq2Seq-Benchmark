import torch
import torch.nn as nn


class ResBlock3d(nn.Module):
    def __init__(self, c_in, kernel, channel, downsample=False):
        super(ResBlock3d, self).__init__()

        padding = [[p // 2 for p in k] for k in kernel]
        padding = [tuple(p) for p in padding]

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(c_in, channel[0],
                               kernel_size=kernel[0],
                               stride=(1, 2, 2) if downsample else (1, 1, 1),
                               padding=padding[0])
        self.bn1 = nn.BatchNorm3d(channel[0])
        self.conv2 = nn.Conv3d(channel[0], channel[1],
                               kernel_size=kernel[1],
                               padding=padding[1])
        self.bn2 = nn.BatchNorm3d(channel[1])
        self.conv3 = nn.Conv3d(channel[1], channel[2],
                               kernel_size=kernel[2],
                               padding=padding[2])
        self.bn3 = nn.BatchNorm3d(channel[2])

        if downsample or c_in != channel[-1]:
            self.shortcut = nn.Conv3d(c_in, channel[-1],
                                      kernel_size=(1, 1, 1),
                                      stride=(1, 2, 2) if downsample else (1, 1, 1))
            self.bns = nn.BatchNorm3d(channel[-1])
        else:
            self.shortcut = self.bns = None

        self.bottle = nn.Sequential(
            self.conv1, self.bn1, self.relu,
            self.conv2, self.bn2, self.relu,
            self.conv3, self.bn3
        )

    def forward(self, x):
        bottleneck = self.bottle(x)
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
            shortcut = self.bns(shortcut)
        x = bottleneck + shortcut
        x = self.relu(x)
        return x


class SlowFast(nn.Module):

    def __init__(self):
        super(SlowFast, self).__init__()
        self.dropout = nn.Dropout3d()
        self.slow_data = nn.Sequential(nn.Conv3d(3, 3, (1, 1, 1), stride=(16, 1, 1)),
                                       nn.Conv3d(3, 64, (1, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3)),
                                       nn.MaxPool3d((1, 3, 3), (1, 2, 2), padding=(0, 1, 1)))
        self.fast_data = nn.Sequential(nn.Conv3d(3, 3, (1, 1, 1), stride=(2, 1, 1)),
                                       nn.Conv3d(3, 8, (5, 7, 7), stride=(1, 1, 1), padding=(2, 3, 3)),
                                       nn.MaxPool3d((1, 3, 3), (1, 2, 2), padding=(0, 1, 1)))
        self.slow_res = [{
            "kernel": ((1, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (64, 64, 256),
            "repeat": 3
        }, {
            "kernel": ((1, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (128, 128, 512),
            "repeat": 4
        }, {
            "kernel": ((3, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (256, 256, 1024),
            "repeat": 6
        }, {
            "kernel": ((3, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (512, 512, 2048),
            "repeat": 3
        }]
        self.fast_res = [{
            "kernel": ((3, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (8, 8, 32),
            "repeat": 3
        }, {
            "kernel": ((3, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (16, 16, 64),
            "repeat": 4
        }, {
            "kernel": ((3, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (32, 32, 128),
            "repeat": 6
        }, {
            "kernel": ((3, 1, 1), (1, 3, 3), (1, 1, 1)),
            "channel": (64, 64, 256),
            "repeat": 3
        }]

        self.relu = nn.ReLU()
        self.fast_pathway_stages = self.make_fast_pathway(8, self.fast_res)
        self.lateral_conv, self.slow_pathway_stages = self.make_slow_pathway(64, self.fast_res, self.slow_res)
        self.avg_pool_slow = nn.AvgPool3d((4, 7, 7))
        self.avg_pool_fast = nn.AvgPool3d((32, 7, 7))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.slow_res[-1]["channel"][-1] + self.fast_res[-1]["channel"][-1], 51)

    @staticmethod
    def make_layer(c_in, cfg):
        kernel = cfg["kernel"]
        channel = cfg["channel"]
        repeat = cfg["repeat"]
        c_in = [c_in, ] + [channel[-1], ] * (repeat - 1)

        block = []
        for i in range(repeat):
            block.append(ResBlock3d(c_in[i], kernel, channel, downsample=i == 0))

        return nn.Sequential(*block)

    @staticmethod
    def make_fast_pathway(c_in, fast_cfg):
        c_in = [c_in, ] + [stage_cfg["channel"][-1] for stage_cfg in fast_cfg][:-1]
        stage = []
        for c, stage_cfg in zip(c_in, fast_cfg):
            stage.append(SlowFast.make_layer(c, stage_cfg))
        return torch.nn.ModuleList(stage)

    @staticmethod
    def make_slow_pathway(c_in, fast_cfg, slow_cfg):
        lateral_conv = []
        stage = []

        c_lateral_in = [stage_cfg["channel"][-1] for stage_cfg in fast_cfg][:-1]
        c_lateral_out = [2 * c for c in c_lateral_in]
        c_slow_in = [stage_cfg["channel"][-1] for stage_cfg in slow_cfg][:-1]
        c_slow_in = [c_in, ] + [seq_in + lateral_in for seq_in, lateral_in in zip(c_slow_in, c_lateral_out)]

        for c_in, c_out in zip(c_lateral_in, c_lateral_out):
            lateral_conv.append(nn.Conv3d(c_in, c_out, (5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0)))

        for c, stage_cfg in zip(c_slow_in, slow_cfg):
            stage.append(SlowFast.make_layer(c, stage_cfg))
        return nn.ModuleList(lateral_conv), nn.ModuleList(stage)

    def forward(self, x):
        # (B,C,N,H,W)
        slow = self.slow_data(x)
        fast = self.fast_data(x)

        fast_stage = []
        for stage in self.fast_pathway_stages:
            fast = stage(fast)
            fast_stage.append(fast)

        for i, (stage, fast_lateral) in enumerate(zip(self.slow_pathway_stages, fast_stage)):
            slow = stage(slow)
            if i < len(self.lateral_conv):
                fast_lateral = self.lateral_conv[i](fast_lateral)
                slow = torch.cat([slow, fast_lateral], dim=1)

        slow = self.flat(self.avg_pool_slow(slow))
        fast = self.flat(self.avg_pool_fast(fast))
        x = torch.cat((slow, fast), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
