import os

import torch
from torch.utils import data
from torchvision.datasets import HMDB51
from torchvision.transforms._transforms_video import ToTensorVideo
from torchvision.transforms import Resize, Compose


def build_hmdb51_loader(root, annotation, batch_size, frame_per_clip=32, size=(112, 112), train=True):
    transforms = Compose([
        ToTensorVideo(),
        Resize(size)
    ])

    metadata_path = os.path.join(root, "metadata_cache.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    hmdb51 = HMDB51(root, annotation, frame_per_clip,
                    _precomputed_metadata=metadata, transform=transforms, train=train)
    if not os.path.exists(metadata_path):
        torch.save(hmdb51.metadata, metadata_path)

    return data.DataLoader(hmdb51, batch_size,
                           shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=64)
