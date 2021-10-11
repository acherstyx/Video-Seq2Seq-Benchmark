import os
import torch
import argparse
import tqdm
import logging
from collections import OrderedDict
from datetime import datetime
from dataloader import build_hmdb51_loader, build_kinetics_loader
from model import *
from torch.utils import data
from torchsummary import summary
from utils.train_utils import accuracy_metric, AvgMeter, summary_graph, ResetTimer, PreFetcher
from torch.utils.tensorboard import SummaryWriter

model_zoo = {
    "conv3d": Conv3D,
    "conv2d_lstm": Conv2DLSTM,
    "slowfast": SlowFast,
    "vivit": ViViT
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description="Performance test")

parser.add_argument("mod", choices=["train", "eval", "summary"])
# data loader
parser.add_argument("--fpc", type=int, default=64, help="frame per clip")
parser.add_argument("-W", "--width", type=int, default=112)
parser.add_argument("-H", "--height", type=int, default=112)
parser.add_argument("--skip", type=int, default=2,
                    help="drop some frame in the clip (default=2, 64fpc->32fpc)")
#   dataset
parser.add_argument("--workers", default=os.cpu_count(), type=int, help="dataloader num_worker")
dataset_parser = parser.add_subparsers(help="chose a dataset to load", title="dataset")
hmdb_parser = dataset_parser.add_parser("hmdb51")
hmdb_parser.set_defaults(dataset="hmdb51")
hmdb_parser.add_argument("--video", type=str, help="hmdb51 video file", required=True)
hmdb_parser.add_argument("--annotation", type=str, help="hmdb51 split file", required=True)
kinetics_parser = dataset_parser.add_parser("kinetics")
kinetics_parser.set_defaults(dataset="kinetics")
kinetics_parser.add_argument("--video", type=str, help="kinetics dataset video folder", required=True)
# train
parser.add_argument("--batch_size", "--bs", default=4, type=int)
parser.add_argument("--split_train", "--st", default=1, type=int,
                    help="(for low GPU memory) do optimize after [split_train] train step")
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--eval_freq", default=1, type=int, help="for every N epoch, run eval and save checkpoint")
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--fine_tune", default=False, action="store_true", help="use fine tune mode")
# log
parser.add_argument("--log", "-l", default=None, type=str, required=True,
                    help="directory to save tensorboard log and checkpoint")
parser.add_argument("--model", type=str, choices=[k for k, v in model_zoo.items()], default="conv3d")
parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
parser.add_argument("--name", type=str, default=None, help="naming log folder")
# lr schedule
parser.add_argument("--lr_schedule", type=tuple, default=None, help="decay at epoch")
parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="lr=lr*decay_rate")


def main():
    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.now())
    if args.name is None:
        args.name = input("experiment name: ")
    timestamp = os.path.join(args.name, timestamp) if args.name is not None else timestamp
    log_dir = os.path.join(args.log, timestamp)
    ckp_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(ckp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.info("log dir: %s", log_dir)

    # net
    logger.info("building model: %s", args.model)
    if args.dataset == "hmdb51":
        net: torch.nn.Module = model_zoo[args.model](num_classes=51)
    elif args.dataset == "kinetics":
        net: torch.nn.Module = model_zoo[args.model](num_classes=400)
    else:
        raise ValueError
    net.cuda()
    # train
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("param", str(args.__dict__), global_step=0)
    criterion = torch.nn.CrossEntropyLoss()

    if args.mod == "summary":
        print((3, args.fpc, args.height, args.width))
        summary(net, (3, args.fpc, args.height, args.width), device="cpu")
        # tensorboard graph
        summary_graph((1, 3, args.fpc, args.height, args.width), net, writer)
    else:
        # optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

        if args.resume is not None and not args.fine_tune:
            logger.info("resume from checkpoint... %s", args.resume)
            state_dict = torch.load(args.resume)
            assert state_dict["arch"] == args.model, "The model architecture is not match"
            epoch_start = state_dict["epoch"]
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])
            net.load_state_dict(state_dict["model_param"])
        elif args.resume is not None and args.fine_tune:
            logger.info("resume from checkpoint... %s", args.resume)
            state_dict = torch.load(args.resume)
            assert state_dict["arch"] == args.model, "The model architecture is not match"
            epoch_start = 0
            # match model parameters
            net_dict = net.state_dict()
            logger.info("fine %s layers", len(state_dict["model_param"].items()))
            state_dict["model_param"] = {k: v for k, v in state_dict["model_param"].items() if
                                         (k in net_dict and net_dict[k].shape == v.shape)}
            logger.info("resume %s layers from checkpoint", len(state_dict["model_param"].items()))
            net_dict.update(state_dict["model_param"])
            net.load_state_dict(OrderedDict(net_dict))
        else:
            epoch_start = 0

        # load train, eval and test data
        logger.info("creating data loader...")

        if args.dataset == "hmdb51":
            dataloader_train = build_hmdb51_loader(args.video, args.annotation,
                                                   num_workers=args.workers,
                                                   batch_size=args.batch_size,
                                                   skip=args.skip,
                                                   train=True,
                                                   size=(args.height, args.width),
                                                   frame_per_clip=args.fpc)
            dataloader_test = dataloader_val = build_hmdb51_loader(args.video, args.annotation,
                                                                   num_workers=args.workers,
                                                                   batch_size=args.batch_size,
                                                                   skip=args.skip,
                                                                   train=False,
                                                                   size=(args.height, args.width),
                                                                   frame_per_clip=args.fpc)
        elif args.dataset == "kinetics":
            dataloader_train = build_kinetics_loader(os.path.join(args.video, "train"),
                                                     num_workers=args.workers,
                                                     batch_size=args.batch_size,
                                                     skip=args.skip,
                                                     size=(args.height, args.width),
                                                     train=True,
                                                     frame_per_clip=args.fpc)
            dataloader_val = build_kinetics_loader(os.path.join(args.video, "val"),
                                                   num_workers=args.workers,
                                                   batch_size=args.batch_size,
                                                   skip=args.skip,
                                                   size=(args.height, args.width),
                                                   train=False,
                                                   frame_per_clip=args.fpc)
            dataloader_test = build_kinetics_loader(os.path.join(args.video, "test"),
                                                    num_workers=args.workers,
                                                    batch_size=args.batch_size,
                                                    skip=args.skip,
                                                    size=(args.height, args.width),
                                                    train=False,
                                                    frame_per_clip=args.fpc)
        else:
            raise ValueError
        dataloader_train: data.DataLoader
        dataloader_val: data.DataLoader
        dataloader_test: data.DataLoader

        if args.mod == "train":
            logger.info("Training...")
            for epoch in range(epoch_start, args.epoch):
                try:
                    logger.info("train epoch {}/{}:".format(epoch + 1, args.epoch))

                    train(dataloader_train, net, optimizer, criterion, accuracy_metric, epoch,
                          writer=writer, split=args.split_train)

                    if args.lr_schedule is not None and (epoch + 1) in args.lr_schedule:
                        optimizer.zero_grad()
                        optimizer.step()
                        scheduler.step()

                    if (epoch + 1) % args.eval_freq == 0:
                        # save
                        ckp_file = "checkpoint_{}.pt".format(epoch + 1)
                        ckp_path = os.path.join(ckp_dir, ckp_file)
                        state_dict = {
                            "epoch": epoch + 1,
                            "arch": args.model,
                            "model_param": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()
                        }
                        torch.save(state_dict, ckp_path)
                        logger.info("Checkpoint is saved to %s", ckp_path)
                        # eval
                        logger.info("evaluating...")
                        loss, top1, top5 = eval(dataloader_val, net, criterion, accuracy_metric)
                        writer.add_scalars("eval/acc", {"top1": top1, "top5": top5}, global_step=epoch + 1)
                        writer.add_scalar("eval/loss", loss, global_step=epoch + 1)
                except Exception as e:
                    # error exit save
                    ckp_file = "checkpoint_error_exit_epoch_{}.pt".format(epoch + 1)
                    ckp_path = os.path.join(ckp_dir, ckp_file)
                    state_dict = {
                        "epoch": epoch + 1,
                        "arch": args.model,
                        "model_param": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                    torch.save(state_dict, ckp_path)
                    logger.critical("Catch exception, checkpoint is saved to %s", ckp_path)
                    logger.critical("Exiting...")
                    raise e
        else:
            eval(dataloader_test, net, criterion, accuracy_metric)
    writer.close()


def train(data_loader: data.DataLoader, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, acc_metric, epoch, writer=None, split=1):
    net.cuda()
    net.train()
    optimizer.zero_grad()
    data_loader = tqdm.tqdm(data_loader)
    data_loader.set_description("Train")
    timer = ResetTimer()
    time_log = {}
    for step, (video, label) in enumerate(PreFetcher(data_loader, device="cuda:0")):
        time_log["load_data"] = timer()

        logits = net(video)
        loss = criterion(logits, label)
        time_log["forward"] = timer()
        loss.backward()
        time_log["backward"] = timer()

        if step % split == 0:
            optimizer.step()
            optimizer.zero_grad()
        time_log["optimize"] = timer()

        acc1, acc5 = acc_metric(logits, label, topk=(1, 5))
        time_log["acc"] = timer()

        if writer is not None:
            total_step = step + epoch * len(data_loader)
            writer.add_scalars("train/acc", {"top1": acc1, "top5": acc5}, total_step)
            writer.add_scalar("train/loss", loss, total_step)
            writer.add_scalars("train/time", time_log, total_step, )
            writer.add_scalar("train/lr", optimizer.state_dict()['param_groups'][0]['lr'], total_step)
            time_log["summary"] = timer()


def eval(data_loader: data.DataLoader, net: torch.nn.Module, criterion, acc_metric):
    top1 = AvgMeter("Acc@1", ":4.2f")
    top5 = AvgMeter("Acc@5", ":4.2f")
    avg_loss = AvgMeter("Loss", ":3.4f")

    net.cuda()
    net.eval()
    data_loader = tqdm.tqdm(data_loader)
    data_loader.set_description("Eval")
    with torch.no_grad():
        for step, (video, label) in enumerate(data_loader):
            video = video.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            logits = net(video)

            loss = criterion(logits, label)
            acc1, acc5 = acc_metric(logits, label, (1, 5))

            avg_loss.update(loss, video.size(0))
            top1.update(acc1, video.size(0))
            top5.update(acc5, video.size(0))
            data_loader.set_postfix_str(str(top1) + " " + str(top5))
    return avg_loss.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
