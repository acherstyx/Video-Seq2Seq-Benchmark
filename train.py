import os.path

import torch
import argparse
import tqdm
from datetime import datetime
from dataloader.hmdb51 import build_hmdb51_loader
from model import Conv3D
from torch.utils import data
from torchsummary import summary
from utils.train_utils import accuracy_metric, AvgMeter
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Performance test")

parser.add_argument("--batch_size", "--bs", default=4, type=int)
parser.add_argument("--log", "-l", default=None, type=str, required=True)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--model", type=str, choices=["conv3d", ], default="conv3d")
parser.add_argument("video", type=str, help="video dataset")
parser.add_argument("annotation", type=str, help="annotation")

model_zoo = {"conv3d": Conv3D}


def main():
    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.now())
    log_dir = os.path.join(args.log, timestamp)
    ckp_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(ckp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # data
    loader: data.DataLoader = build_hmdb51_loader(args.video, args.annotation, args.batch_size)
    # net
    net: torch.nn.Module = model_zoo[args.model]()
    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)

    summary(net, input_size=(3, 32, 112, 112), device="cpu")

    for epoch in range(args.epoch):
        train(loader, net, optimizer, criterion, accuracy_metric, epoch, writer=writer)

        if (epoch + 1) % 5 == 0:
            # save
            ckp_file = "checkpoint_{}.pt".format(epoch + 1)
            ckp_path = os.path.join(ckp_dir, ckp_file)
            state_dict = {
                "epoch": epoch + 1,
                "arch": args.model,
                "model_param": net.state_dict()
            }
            torch.save(state_dict, ckp_path)


def train(data_loader: data.DataLoader, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, acc_metric, epoch, writer=None):
    top1 = AvgMeter("Acc@1", ":6.2f")
    top5 = AvgMeter("Acc@5", ":6.2f")

    net.cuda()
    data_loader = tqdm.tqdm(data_loader)
    for step, (video, audio, label) in enumerate(data_loader):
        video = video.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        logits = net(video)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        acc1, acc5 = acc_metric(logits, label, topk=(1, 5))
        top1.update(acc1, video.size(0))
        top5.update(acc5, video.size(0))

        if writer is not None:
            total_step = step + epoch * len(data_loader)
            writer.add_scalars("train/acc", {"top1": top1.val, "top5": top5.val}, total_step)
            writer.add_scalar("train/loss", loss, total_step)


if __name__ == '__main__':
    main()
