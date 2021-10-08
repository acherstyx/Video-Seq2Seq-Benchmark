from model.vivit import *
import unittest


class TestViViT(unittest.TestCase):
    def test_einops(self):
        import numpy as np
        import matplotlib.pyplot as plt

        sample = np.ones([9, 10, 10], dtype=float) / 9
        line = np.linspace(0, 9, 9).reshape((9, 1, 1))
        sample = sample * line
        # print(line)
        # print(sample)
        sample = einops.rearrange(sample, "(a b) h w -> (a h) (b w)", a=3)
        # sample = einops.rearrange(sample, "(h a) (w b) -> (h w a) b ", h=3, w=3)
        sample = einops.rearrange(sample, "(n_h h) (n_w w) -> (n_h n_w h) w", n_h=3, n_w=3)
        plt.imshow(sample)
        plt.show()
