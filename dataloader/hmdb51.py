import os

import torch
from torch.utils import data
from torchvision.datasets import HMDB51
from torchvision.transforms._transforms_video import ToTensorVideo
from torchvision.transforms import *
import warnings

warnings.simplefilter("ignore", UserWarning)


def build_hmdb51_loader(root, annotation, num_workers,
                        batch_size=1, frame_per_clip=64, skip=2, size=(224, 224), train=True):
    if train:
        transforms = Compose([
            ToTensorVideo(),
            Resize(max(*size)),
            CenterCrop(size),
            # ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.3),
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            RandomPerspective(distortion_scale=0.6, p=1.0)
        ])
    else:
        transforms = Compose([
            ToTensorVideo(),
            CenterCrop(size)
        ])

    metadata_path = os.path.join(root, "metadata_cache.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    hmdb51 = HMDB51(root, annotation, frame_per_clip, step_between_clips=frame_per_clip,
                    _precomputed_metadata=metadata, transform=transforms, train=train,
                    num_workers=os.cpu_count()
                    )
    if not os.path.exists(metadata_path):
        torch.save(hmdb51.metadata, metadata_path)

    return data.DataLoader(hmdb51, batch_size,
                           shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=1,
                           collate_fn=lambda batch: [
                               torch.cat(
                                   [batch[i][0][:, ::skip, :, :].unsqueeze(0) for i in range(len(batch))], 0
                               ),
                               torch.LongTensor([batch[i][2] for i in range(len(batch))]),
                           ], timeout=10)
