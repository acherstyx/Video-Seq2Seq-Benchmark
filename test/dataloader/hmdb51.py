import torch.utils.data

from dataloader.hmdb51 import *
import unittest
import tqdm


class TestHMDB51(unittest.TestCase):

    def test_dataset(self):
        metadata = torch.load("/mnt/workshop/hmdb51/hmdb51_video/kinetics_metadata_hmdb51_video.pt")
        hmdb51_train = HMDB51(
            root="/mnt/workshop/hmdb51/hmdb51_video",
            annotation_path="/mnt/workshop/hmdb51/testTrainMulti_7030_splits",  # split
            frames_per_clip=32,
            transform=Compose(
                [ToTensorVideo(),
                 Resize((112, 112))]
            ),
            _precomputed_metadata=metadata,
        )
        hmdb51_test = HMDB51(
            root="/mnt/workshop/hmdb51/hmdb51_video",
            annotation_path="/mnt/workshop/hmdb51/testTrainMulti_7030_splits",  # split
            frames_per_clip=32,
            _precomputed_metadata=metadata,
            train=False
        )
        print("Train:", len(hmdb51_train))
        print("Test:", len(hmdb51_test))

        for v, a, l in tqdm.tqdm(hmdb51_train):
            print(v)
            print(v.shape)
            print(a.shape)
            print(l)
            break
        return hmdb51_train

    def test_loader(self):
        return torch.utils.data.DataLoader(self.test_dataset(), shuffle=True)

    def test_iter(self):
        for v, a, l in self.test_loader():
            print(l)

    def test_loader_build(self):
        hmdb51 = build_hmdb51_loader(root="/mnt/workshop/hmdb51/hmdb51_video",
                                     annotation="/mnt/workshop/hmdb51/testTrainMulti_7030_splits",
                                     batch_size=1)
        for v, a, l in hmdb51:
            print(l)
