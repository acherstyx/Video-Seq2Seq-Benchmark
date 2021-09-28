import os

import torch
from torch.utils import data
from torchvision.datasets import HMDB51
from torchvision.transforms._transforms_video import ToTensorVideo
from torchvision.transforms import *


def build_hmdb51_loader(root, annotation, batch_size, frame_per_clip=32, size=(112, 112), train=True):
    transforms = Compose([
        ToTensorVideo(),
        CenterCrop(size),
        # ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.3),
        GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        RandomPerspective(distortion_scale=0.6, p=1.0)
    ])

    metadata_path = os.path.join(root, "metadata_cache.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    hmdb51 = HMDB51(root, annotation, frame_per_clip, step_between_clips=frame_per_clip,
                    _precomputed_metadata=metadata, transform=transforms, train=train)
    if not os.path.exists(metadata_path):
        torch.save(hmdb51.metadata, metadata_path)

    return data.DataLoader(hmdb51, batch_size,
                           shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=64)
