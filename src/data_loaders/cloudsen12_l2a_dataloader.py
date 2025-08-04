import os
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset
import tacoreader


class Cloudsen12l2aDataloader(Dataset):
    def __init__(self, indices, selected_bands):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        taco_path = os.path.join(base_dir, "..", "..", "data", "CloudSen12+", "TACOs", "mini-cloudsen12-l2a-high-512.taco")
        taco_path = os.path.normpath(taco_path)

        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.dataset.read(self.indices[idx])
        s2_l1c_path = record.read(0)
        s2_label_path = record.read(1)

        with rio.open(s2_l1c_path) as src, rio.open(s2_label_path) as dst:
            img = src.read(indexes=self.selected_bands).astype(np.float32)
            label = dst.read(1).astype(np.uint8)

        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()

        return img, label


def get_cloudsen12_datasets(selected_bands, split_ratio=(0.7, 0.15, 0.15)):
    total_samples = 10000
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("split_ratio must sum to 1.0")

    indices = list(range(total_samples))

    train_end = int(split_ratio[0] * total_samples)
    val_end = train_end + int(split_ratio[1] * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_ds = Cloudsen12l2aDataloader(train_indices, selected_bands)
    val_ds = Cloudsen12l2aDataloader(val_indices, selected_bands)
    test_ds = Cloudsen12l2aDataloader(test_indices, selected_bands)

    return train_ds, val_ds, test_ds
