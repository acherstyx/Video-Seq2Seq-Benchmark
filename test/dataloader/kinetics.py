from dataloader.kinetics import *
import unittest


class TestKinetics(unittest.TestCase):
    def test_build_dataset(self):
        import torch
        metadata = torch.load("/mnt/workshop/kinetics400/train/metadata_cache.pt")
        dataset = Kinetics400("/mnt/workshop/kinetics400/train",
                              frames_per_clip=64,
                              step_between_clips=64,
                              extensions=("mp4", "avi",),
                              num_workers=64,
                              _precomputed_metadata=metadata)
        return dataset

    def test_iter(self):
        for v, a, l in self.test_build_dataset():
            print(v.shape)
            print(a.shape)
            print(l)
            print("=====")

    def test_speed(self):
        def collate_fn(batch):
            return [
                torch.cat(
                    [batch[i][0][:, ::2, :, :].unsqueeze(0) for i in range(len(batch))], 0
                ),
                torch.LongTensor([batch[i][2] for i in range(len(batch))]),
            ]

        k400 = build_kinetics_loader("/mnt/workshop/kinetics400/train", os.cpu_count() * 2, 2, 64,
                                     collate_fn=collate_fn)
        import tqdm
        for v, l in tqdm.tqdm(k400):
            pass
