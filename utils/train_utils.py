import time

import torch
from torch.utils.tensorboard import SummaryWriter


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
        return int((after - pre) * 1000)


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
