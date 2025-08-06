import os
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset, ConcatDataset
from collections import Counter
from tqdm import tqdm
from data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader, l8_biome_dataloader


def show_class_distribution(train_ds, val_ds, test_ds, class_names, ignore_label=None):
    def compute_stats(dataset):
        counter = Counter()
        for _, label in tqdm(dataset, desc="Processing", leave=False):
            unique, counts = torch.unique(label, return_counts=True)
            for cls, cnt in zip(unique.tolist(), counts.tolist()):
                if ignore_label is not None and cls == ignore_label:
                    continue
                counter[cls] += cnt
        total = sum(counter.values())
        stats = []
        for label in range(len(class_names)):
            count = counter[label]
            ratio = 100 * count / total if total > 0 else 0
            stats.append((label, count, ratio))
        return stats, total

    def print_stats(title, stats, total):
        print(f"\n{title} — Total Pixels: {total:,}")
        print(f"{'Class Name':<16} | {'Label':<5} | {'Pixel Count':<15} | {'Ratio'}")
        print("-" * 60)
        for label, count, ratio in stats:
            name = class_names.get(label, f"class {label}")
            print(f"{name:<16} | {label:<5} | {count:<15,} | {ratio:.2f}%")

    print("Calculating class distributions...\n")

    train_stats, train_total = compute_stats(train_ds)
    print_stats("Train Set", train_stats, train_total)

    val_stats, val_total = compute_stats(val_ds)
    print_stats("Validation Set", val_stats, val_total)

    test_stats, test_total = compute_stats(test_ds)
    print_stats("Test Set", test_stats, test_total)

    full_ds = ConcatDataset([train_ds, val_ds, test_ds])
    full_stats, full_total = compute_stats(full_ds)
    print_stats("Full Dataset", full_stats, full_total)


if __name__ == "__main__":
    selected_bands = list(range(1, 14))
    train_ds, val_ds, test_ds = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
        selected_bands, split_ratio=(0.85, 0.05, 0.1)
    )

    show_class_distribution(
        train_ds, val_ds, test_ds,
        class_names={
            0: "clear",
            1: "thick cloud",
            2: "thin cloud",
            3: "cloud shadow"
        },
        ignore_label=None
    )

    selected_bands = list(range(1, 12))
    train_ds, val_ds, test_ds = l8_biome_dataloader.get_l8biome_datasets(
        selected_bands, 512, 512, split_ratio=(0.6, 0.2, 0.2)
    )

    show_class_distribution(
        train_ds, val_ds, test_ds,
        class_names={
            0: "clear",
            1: "thick cloud",
            2: "thin cloud",
            3: "cloud shadow"
        },
        ignore_label=255
    )
