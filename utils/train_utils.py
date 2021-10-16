import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


def accuracy_metric(logits, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)[0])
        return res


class AvgMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.val = self.sum = self.count = 0
        self.fmt = fmt
        self.avg = 0

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        self.val = value
        self.count += n
        self.sum += n * value
        self.avg = self.sum / self.count

    def __str__(self):
        format_str = "{name} {val" + self.fmt + "}+({avg" + self.fmt + "})"
        return format_str.format(**self.__dict__)


def summary_graph(dummy_shape: tuple, net: torch.nn.Module, summary_writer: SummaryWriter):
    x = torch.randn(dummy_shape)
    summary_writer.add_graph(net, x)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Timer {}: {} s".format(self.name, time.time() - self.start))


class ResetTimer:
    def __init__(self):
        self.time = time.time()

    def __call__(self, reset=True):
        pre = self.time
        if reset:
            after = self.time = time.time()
        else:
            after = time.time()
        return (after - pre) * 1000


class PreFetcher:
    def __init__(self, data_loader, device):
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.batch = None
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = [sample.to(self.device, non_blocking=True) for sample in self.batch]

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        return self


def auto_resume(ckpt_folder):
    ckpt_files = [ckpt for ckpt in os.listdir(ckpt_folder) if ckpt.endswith(".pth")]
    if len(ckpt_files) > 0:
        return max([os.path.join(ckpt_folder, file) for file in ckpt_files], key=os.path.getmtime)
    else:
        return None


def save_checkpoint(ckpt_folder, epoch, model, optimizer, scheduler, config, prefix=""):
    stat_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config
    }
    ckpt_path = os.path.join(ckpt_folder, f"checkpoint{prefix}_{epoch}.pth")
    torch.save(stat_dict, ckpt_path)
    return ckpt_path


def load_checkpoint(ckpt_file, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                    restart_train=False):
    state_dict = torch.load(ckpt_file, map_location="cpu")
    try:
        missing = model.load_state_dict(state_dict["model"], strict=False)
        if missing:
            print(f"checkpoint key missing: {missing}")
    except RuntimeError:  # tensor shape mismatch (change num_classes)
        print("fail to directly recover from checkpoint, try to match each layers...")
        net_dict = model.state_dict()
        print("find %s layers", len(state_dict["model"].items()))
        state_dict["model"] = {k: v for k, v in state_dict["model"].items() if
                               (k in net_dict and net_dict[k].shape == v.shape)}
        print("resume %s layers from checkpoint", len(state_dict["model"].items()))
        net_dict.update(state_dict["model"])
        model.load_state_dict(OrderedDict(net_dict))

    if not restart_train:
        # remove optimizer state
        state_dict["optimizer"]["state"] = optimizer.state
        state_dict["optimizer"]["param_groups"][0]["params"] = optimizer.param_groups[0]["params"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        epoch = state_dict["epoch"]
    else:
        print("restart train, optimizer and scheduler will not be resumed")
        epoch = 0

    del state_dict
    torch.cuda.empty_cache()
    return epoch  # start epoch


class TrainErrorHelper:
    def __init__(self, ckpt_folder, model, optimizer, scheduler, config, epoch, logger=None):
        self.ckpt_folder = ckpt_folder
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.epoch = epoch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ckpt_path = save_checkpoint(ckpt_folder=self.ckpt_folder,
                                    epoch=self.epoch,
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    config=self.config,
                                    prefix="_error_exit")
        if exc_type is not None and self.logger is not None:
            self.logger.critical("catch exception, checkpoint is saved to %s", ckpt_path)
            self.logger.critical("exiting...")
        return False
