import os
from torch.utils import data
from torchvision.datasets import HMDB51
from torchvision.transforms import *
import warnings
from .metadata import save_metadata, load_metadata
import einops

warnings.simplefilter("ignore", UserWarning)


def build_hmdb51_set(root, annotation, num_workers,
                     batch_size=1, frame_per_clip=64, skip=2, size=(224, 224), train=True):
    if train:
        transforms = Compose([
            lambda x: einops.rearrange(x[::skip, :, :, :], "t h w c->c t h w"),
            RandomResizedCrop(size, (0.5, 1)),
        ])
    else:
        transforms = Compose([
            lambda x: einops.rearrange(x[::skip, :, :, :], "t h w c->c t h w"),
            Resize(size),
            CenterCrop(size),
        ])

    metadata = load_metadata(root)
    hmdb51 = HMDB51(root, annotation, frame_per_clip, step_between_clips=frame_per_clip,
                    _precomputed_metadata=metadata, transform=transforms, train=train,
                    num_workers=os.cpu_count()
                    )
    save_metadata(root, metadata)

    return data.DataLoader(hmdb51, batch_size,
                           shuffle=True, num_workers=num_workers, persistent_workers=True if num_workers > 0 else False,
                           pin_memory=True,
                           collate_fn=lambda batch: [
                               torch.stack([batch[i][0] for i in range(len(batch))], dim=0),
                               torch.LongTensor([batch[i][2] for i in range(len(batch))]),
                           ])
