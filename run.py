import os
import torch
import argparse
import tqdm
import logging
from config import get_config, default_cfg
from collections import OrderedDict
from datetime import datetime
from data import build_loader
from model import build_model
from torch.utils import data
from torchsummary import summary
from utils.train_utils import *
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description="Performance test")

# parser.add_argument("mod", choices=["train", "eval", "summary"])
# # data loader
# parser.add_argument("--fpc", type=int, default=64, help="frame per clip")
# parser.add_argument("-W", "--width", type=int, default=112)
# parser.add_argument("-H", "--height", type=int, default=112)
# parser.add_argument("--skip", type=int, default=2,
#                     help="drop some frame in the clip (default=2, 64fpc->32fpc)")
# #   dataset
# parser.add_argument("--workers", default=os.cpu_count(), type=int, help="dataloader num_worker")
# dataset_parser = parser.add_subparsers(help="chose a dataset to load", title="dataset")
# hmdb_parser = dataset_parser.add_parser("hmdb51")
# hmdb_parser.set_defaults(dataset="hmdb51")
# hmdb_parser.add_argument("--video", type=str, help="hmdb51 video file", required=True)
# hmdb_parser.add_argument("--annotation", type=str, help="hmdb51 split file", required=True)
# kinetics_parser = dataset_parser.add_parser("kinetics")
# kinetics_parser.set_defaults(dataset="kinetics")
# kinetics_parser.add_argument("--video", type=str, help="kinetics dataset video folder", required=True)
# # train
# parser.add_argument("--batch_size", "--bs", default=4, type=int)
# parser.add_argument("--split_train", "--st", default=1, type=int,
#                     help="(for low GPU memory) do optimize after [split_train] train step")
# parser.add_argument("--epoch", default=10, type=int)
# parser.add_argument("--eval_freq", default=1, type=int, help="for every N epoch, run eval and save checkpoint")
# parser.add_argument("--lr", default=0.001, type=float)
# parser.add_argument("--fine_tune", default=False, action="store_true", help="use fine tune mode")
# # log
# parser.add_argument("--log", "-l", default=None, type=str, required=True,
#                     help="directory to save tensorboard log and checkpoint")
# parser.add_argument("--model", type=str, choices=[k for k, v in model_zoo.items()], default="conv3d")
# parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
# parser.add_argument("--name", type=str, default=None, help="naming log folder")
# # lr schedule
# parser.add_argument("--lr_schedule", type=str2tuple, default=None, help="decay at epoch")
# parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="lr=lr*decay_rate")

parser.add_argument("config", type=str, help="config file")


def main():
    args = parser.parse_args()

    config = get_config(args)

    log_dir = os.path.join(config.LOG.LOG_DIR, config.EXPERIMENT_NAME)
    ckpt_folder = os.path.join(log_dir, "checkpoint")
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.info("log dir: %s", log_dir)

    # net
    logger.info(f"building model {config.MODEL.ARCH}...")
    net = build_model(config)
    net.cuda()
    writer = SummaryWriter(log_dir=log_dir)
    criterion = torch.nn.CrossEntropyLoss()

    if config.MODE == "summary":
        summary(
            model=net,
            input_size=(3, config.DATA.FRAME_PER_CLIP // config.DATA.SKIP_FRAME,
                        config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
            device="cpu"
        )
        # tensorboard graph
        summary_graph(
            dummy_shape=(1, 3, config.DATA.FRAME_PER_CLIP // config.DATA.SKIP_FRAME,
                         config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]),
            net=net,
            summary_writer=writer
        )
    else:
        # optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=config.TRAIN.LR_BASE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCH,
                                                    gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

        if config.MODEL.RESUME:
            logger.info(f"resume from given checkpoint: {config.MODEL.RESUME}")
            epoch_start = load_checkpoint(ckpt_file=config.MODEL.RESUME,
                                          model=net,
                                          optimizer=optimizer,
                                          scheduler=scheduler)
        elif config.TRAIN.AUTO_RESUME:
            logger.info(f"auto resume...")
            ckpt_file = auto_resume(ckpt_folder)
            if ckpt_file is not None:
                epoch_start = load_checkpoint(ckpt_file=ckpt_file,
                                              model=net,
                                              optimizer=optimizer,
                                              scheduler=scheduler)
                logger.info(f"auto resume from checkpoint {ckpt_file}")
            else:
                logger.info(f"no checkpoint file found")
                epoch_start = 0
        else:
            epoch_start = 0

        # load train, eval and test data
        logger.info("creating data loader...")
        dataloader_train, dataloader_val, dataloader_test = build_loader(config)

        if config.MODE == "train":
            logger.info("Training...")
            for epoch in range(epoch_start, config.TRAIN.EPOCH):
                try:
                    logger.info("train epoch {}/{}:".format(epoch + 1, config.TRAIN.EPOCH))

                    train(dataloader_train, net, optimizer, criterion, accuracy_metric, epoch,
                          writer=writer, split=config.TRAIN.ACCUMULATION_STEP)
                    scheduler.step()

                    # save
                    if (epoch + 1) % config.TRAIN.SAVE_FREQ == 0:
                        ckpt_path = save_checkpoint(ckpt_folder=ckpt_folder,
                                                    epoch=epoch + 1,
                                                    model=net,
                                                    optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    config=config)
                        logger.info("Checkpoint is saved to %s", ckpt_path)
                    # eval
                    if config.TRAIN.EVAL_FREQ != -1 and (epoch + 1) % config.TRAIN.EVAL_FREQ == 0:
                        logger.info("evaluating...")
                        loss, top1, top5 = eval(dataloader_val, net, criterion, accuracy_metric)
                        writer.add_scalars("eval/acc", {"top1": top1, "top5": top5}, global_step=epoch + 1)
                        writer.add_scalar("eval/loss", loss, global_step=epoch + 1)
                except Exception as e:
                    # error exit save
                    ckpt_path = save_checkpoint(ckpt_folder=ckpt_folder,
                                                epoch="error_exit",
                                                model=net,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                config=config)
                    logger.critical("Catch exception, checkpoint is saved to %s", ckpt_path)
                    logger.critical("Exiting...")
                    raise e
        else:
            eval(dataloader_test, net, criterion, accuracy_metric)
    writer.close()


def train(data_loader: data.DataLoader, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, acc_metric, epoch, writer=None, split=1):
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
            time_log["total"] = sum([v if k != "total" else 0 for k, v in time_log.items()])


def eval(data_loader: data.DataLoader, net: torch.nn.Module, criterion, acc_metric):
    top1 = AvgMeter("Acc@1", ":4.2f")
    top5 = AvgMeter("Acc@5", ":4.2f")
    avg_loss = AvgMeter("Loss", ":3.4f")

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
