import os
from torch.utils import data
from torchvision.datasets import Kinetics400
from torchvision.transforms import *
from torchvision.transforms._transforms_video import ToTensorVideo
from .metadata import save_metadata, load_metadata
import warnings

warnings.simplefilter("ignore", UserWarning)


def build_kinetics_loader(video_root, num_workers,
                          batch_size=1, frame_per_clip=64, skip=2, size=(224, 224), train=True):
    if train:
        transforms = Compose([
            ToTensorVideo(),
            RandomResizedCrop(size, (0.5, 1))
        ])
    else:
        transforms = Compose([
            ToTensorVideo(),
            CenterCrop(size),
        ])

    metadata = load_metadata(video_root)
    kinetics = Kinetics400(
        video_root, frames_per_clip=frame_per_clip, step_between_clips=frame_per_clip,
        extensions=('mp4', 'avi'), transform=transforms,
        _precomputed_metadata=metadata,
        num_workers=os.cpu_count()  # for loading video metadata
    )
    save_metadata(video_root, metadata)

    return data.DataLoader(kinetics, batch_size,
                           shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True,
                           prefetch_factor=1,
                           collate_fn=lambda batch: [
                               torch.cat(
                                   [batch[i][0][:, ::skip, :, :].unsqueeze(0) for i in range(len(batch))], 0
                               ),
                               torch.LongTensor([batch[i][2] for i in range(len(batch))]),
                           ])
