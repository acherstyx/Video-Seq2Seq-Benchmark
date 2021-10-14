import os
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from .hmdb51 import build_hmdb51_set
from .kinetics import build_kinetics_loader
import warnings

warnings.simplefilter("ignore", UserWarning)


def build_loader(config: CfgNode) -> (DataLoader, DataLoader, DataLoader):
    dataset = config.DATA.DATASET
    kwargs = {"num_workers": config.DATA.NUM_WORKER,
              "batch_size": config.DATA.BATCH_SIZE,
              "size": config.DATA.IMG_SIZE,
              "frame_per_clip": config.DATA.FRAME_PER_CLIP,
              "skip": config.DATA.SKIP_FRAME}

    if dataset == "hmdb51":
        args = [config.DATA.HMDB51.VIDEO_FOLDER, config.DATA.HMDB51.ANNOTATION]
        dataloader_train = build_hmdb51_set(*args, **kwargs, train=True)
        dataloader_test = dataloader_val = build_hmdb51_set(*args, **kwargs, train=False)
    elif dataset == "kinetics":
        root = config.DATA.KINETICS.VIDEO_FOLDER
        dataloader_train = build_kinetics_loader(os.path.join(root, "train"), **kwargs)
        dataloader_val = build_kinetics_loader(os.path.join(root, "val"), **kwargs)
        dataloader_test = build_kinetics_loader(os.path.join(root, "test"), **kwargs)
    else:
        raise ValueError

    return dataloader_train, dataloader_val, dataloader_test
