import os
from torch.utils import data
from torchvision.datasets import Kinetics400
from torchvision.transforms import *
from torchvision.transforms._transforms_video import ToTensorVideo


def build_kinetics_loader(video_root, batch_size=1, frame_per_clip=64, size=(224, 224), train=True):
    if train:
        transforms = Compose([
            ToTensorVideo(),
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

    metadata_path = os.path.join(video_root, "metadata_cache.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    kinetics = Kinetics400(video_root, frames_per_clip=frame_per_clip, step_between_clips=frame_per_clip,
                           extensions=('mp4', 'avi'), transform=transforms,
                           _precomputed_metadata=metadata, num_workers=64)
    if not os.path.exists(metadata_path):
        torch.save(kinetics.metadata, metadata_path)

    return data.DataLoader(kinetics, batch_size,
                           shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=64)
