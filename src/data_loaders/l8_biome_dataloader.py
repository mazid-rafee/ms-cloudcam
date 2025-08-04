import os
import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio as rio
import random

class L8BiomeDataloader(Dataset):
    def __init__(self, scene_dirs, patch_size, stride, selected_bands):
        self.scene_dirs = scene_dirs
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        self.band_indices = selected_bands

        for scene_dir in self.scene_dirs:
            scene_id = os.path.basename(scene_dir)
            img_path = os.path.join(scene_dir, f"{scene_id}.TIF")
            mask_path = os.path.join(scene_dir, f"{scene_id}_fixedmask.TIF")

            with rio.open(img_path) as img:
                h, w = img.height, img.width

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_path, mask_path, x, y))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.patches[idx]

        with rio.open(img_path) as src:
            img = src.read(self.band_indices, window=rio.windows.Window(x, y, self.patch_size, self.patch_size)).astype(np.float32)

        with rio.open(mask_path) as dst:
            label_raw = dst.read(1, window=rio.windows.Window(x, y, self.patch_size, self.patch_size)).astype(np.uint8)

        mapping = {
            128: 0,  # Clear
            255: 1,  # Thick Cloud
            192: 2,  # Thin Cloud
            64:  3   # Shadow
        }

        label = np.full_like(label_raw, fill_value=255)
        for k, v in mapping.items():
            label[label_raw == k] = v

        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()

        return img, label


def find_valid_scene_dirs(root_dir):
    scene_dirs = []
    for cover_type in os.listdir(root_dir):
        cover_path = os.path.join(root_dir, cover_type)
        if not os.path.isdir(cover_path):
            continue
        for scene_id in os.listdir(cover_path):
            scene_path = os.path.join(cover_path, scene_id)
            if not os.path.isdir(scene_path):
                continue
            tif_file = os.path.join(scene_path, f"{scene_id}.TIF")
            mask_file = os.path.join(scene_path, f"{scene_id}_fixedmask.TIF")
            if os.path.isfile(tif_file) and os.path.isfile(mask_file):
                scene_dirs.append(scene_path)

    random.shuffle(scene_dirs, random=random.Random(42).random)

    return scene_dirs


def get_l8biome_datasets(
    band_indices,
    patch_size=512,
    stride=512,
    split_ratio=(0.6, 0.2, 0.2)
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(os.path.join(base_dir, "..", "..", "data", "l8_biome", "l8biome"))

    all_dirs = find_valid_scene_dirs(root_dir)
    total = len(all_dirs)

    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("Split ratio must sum to 1.0")

    train_end = int(split_ratio[0] * total)
    val_end = train_end + int(split_ratio[1] * total)

    scene_split = {
        'train': all_dirs[:train_end],
        'val': all_dirs[train_end:val_end],
        'test': all_dirs[val_end:]
    }

    train_ds = L8BiomeDataloader(scene_split['train'], patch_size, stride, band_indices)
    val_ds = L8BiomeDataloader(scene_split['val'], patch_size, stride, band_indices)
    test_ds = L8BiomeDataloader(scene_split['test'], patch_size, stride, band_indices)

    return train_ds, val_ds, test_ds
