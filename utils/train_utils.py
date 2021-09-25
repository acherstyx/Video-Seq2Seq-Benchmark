import torch


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
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AvgMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.val = self.sum = self.count = 0
        self.fmt = fmt
        self.avg = 0

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()[0]
        self.val = value
        self.count += n
        self.sum += n * value
        self.avg = self.sum / self.count

    def __str__(self):
        format_str = "{name} {val" + self.fmt + "}+({avg" + self.fmt + "})"
        return format_str.format(**self.__dict__)
