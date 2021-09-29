from dataloader.kinetics import *
import unittest


class TestKinetics(unittest.TestCase):
    def test_build_dataset(self):
        import torch
        metadata = torch.load("/mnt/workshop/kinetics400/train/kinetics_metadata_train.pt")
        dataset = Kinetics400("/mnt/workshop/kinetics400/train",
                              frames_per_clip=64,
                              step_between_clips=64,
                              extensions=("mp4",),
                              num_workers=64,
                              _precomputed_metadata=metadata)
        return dataset

    def test_iter(self):
        for v, a, l in self.test_build_dataset():
            print(v.shape)
            print(a.shape)
            print(l)
            print("=====")
