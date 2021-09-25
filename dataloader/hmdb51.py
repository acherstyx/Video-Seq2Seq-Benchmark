import os

import torch
from torch.utils import data
from torchvision.datasets import HMDB51
from torchvision.transforms._transforms_video import ToTensorVideo
from torchvision.transforms import Resize, Compose


def build_hmdb51_loader(root, annotation, batch_size, train=True):
    frame_per_clip = 32
    resize = (112, 112)

    transforms = Compose([
        ToTensorVideo(),
        Resize(resize)
    ])

    metadata_path = os.path.join(root, "metadata_cache.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    hmdb51 = HMDB51(root, annotation, frame_per_clip,
                    _precomputed_metadata=metadata, num_workers=4, transform=transforms, train=train)
    if not os.path.exists(metadata_path):
        torch.save(hmdb51.metadata, metadata_path)

    return data.DataLoader(hmdb51, batch_size, shuffle=True, num_workers=4)
