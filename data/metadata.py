import os
import torch
from functools import wraps


def load_metadata(root):
    metadata_path = os.path.join(root, "metadata_cache.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    return metadata


def save_metadata(root, metadata):
    metadata_path = os.path.join(root, "metadata_cache.pt")
    if not os.path.exists(metadata_path):
        torch.save(metadata, metadata_path)
