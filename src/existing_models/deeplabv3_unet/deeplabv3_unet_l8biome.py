import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn.functional as F

# ---------------------
# Settings
# ---------------------
NUM_CLASSES = 4
L8_BANDS = list(range(1, 12))  # Landsat-8: bands 1..11
TILE_H, TILE_W = 512, 512
ENCODERS = [
    {"name": "resnet101", "weights": "imagenet"},
    {"name": "mobilenet_v2", "weights": "imagenet"},
]
ARCHS = ["DeepLabV3Plus", "UnetPlusPlus", "Unet"]
RESULTS_TXT = "results/Deeplabv3_Unet_L8Biome.txt"
EPOCHS = 100
BS = 8
LR = 1e-3
NUM_WORKERS = 4
IGNORE_INDEX = 255

# ---------------------
# Import L8Biome loader
# ---------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_loaders_path = os.path.join(parent_dir, 'data_loaders')
sys.path.append(data_loaders_path)
from l8_biome_dataloader import get_l8biome_datasets  # noqa: E402

# ---------------------
# Models
# ---------------------
def get_model(arch_name, encoder_name, encoder_weights, in_channels, classes):
    if arch_name == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif arch_name == "UnetPlusPlus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif arch_name == "Unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

# ---------------------
# Evaluation (with ignore_index)
# ---------------------
def evaluate_test_metrics(model, loader, device, num_classes=4, ignore_index=255):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)                 # [B, C, H, W]
            preds  = torch.argmax(logits, dim=1) # [B, H, W]
            preds  = torch.clamp(preds, 0, num_classes - 1)

            # Exclude ignore_index (255) and any out-of-range labels
            mask = (labels >= 0) & (labels < num_classes) & (labels != ignore_index)
            if mask.sum() == 0:
                continue

            p = preds[mask].reshape(-1).detach().cpu().numpy().astype(np.int64)
            y = labels[mask].reshape(-1).detach().cpu().numpy().astype(np.int64)

            flat = num_classes * y + p
            bincount = np.bincount(flat, minlength=num_classes ** 2)
            conf_mat += bincount.reshape(num_classes, num_classes)

            total_correct += (p == y).sum()
            total_pixels += y.size

    lines, ious, f1s, accs = [], [], [], []
    total_sum = conf_mat.sum()

    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = total_sum - (TP + FP + FN)

        iou  = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

        ious.append(iou); f1s.append(f1); accs.append(acc)
        lines.append(f"Class {i}: IoU={iou:.4f}, F1={f1:.4f}, Acc={acc:.4f}\n")

    mIoU = float(np.mean(ious)) if ious else 0.0
    mF1  = float(np.mean(f1s)) if f1s else 0.0
    mAcc = float(np.mean(accs)) if accs else 0.0
    aAcc = (total_correct / total_pixels) if total_pixels else 0.0

    lines.append(f"Mean IoU: {mIoU:.4f}\n")
    lines.append(f"Mean F1: {mF1:.4f}\n")
    lines.append(f"Mean Accuracy (mAcc): {mAcc:.4f}\n")
    lines.append(f"Overall Accuracy (aAcc): {aAcc:.4f}\n")

    print("".join(lines))
    return lines, conf_mat

# ---------------------
# Train/Val/Test Loop
# ---------------------
def train_and_test(arch_name, encoder_name, encoder_weights, tr_loader, va_loader, te_loader, in_channels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(arch_name, encoder_name, encoder_weights, in_channels=in_channels, classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    os.makedirs("results", exist_ok=True)
    best_val = float("inf")
    ckpt_best = f"results/best_{arch_name}_{encoder_name}_L8Biome.pth"
    ckpt_last = f"results/last_{arch_name}_{encoder_name}_L8Biome.pth"

    for epoch in range(EPOCHS):
        model.train()
        tr_loss_sum = 0.0
        tr_valid_px = 0
        for imgs, labels in tqdm(tr_loader, desc=f"{arch_name}-{encoder_name} | Train {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            logits = model(imgs)
            per_px = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="none")
            valid = (labels != IGNORE_INDEX) & (labels >= 0) & (labels < NUM_CLASSES)
            valid_count = valid.sum().item()
            if valid_count == 0:
                continue
            loss = per_px[valid].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss_sum += per_px[valid].sum().item()
            tr_valid_px += valid_count
        avg_tr = tr_loss_sum / max(1, tr_valid_px)

        model.eval()
        va_loss_sum = 0.0
        va_valid_px = 0
        with torch.no_grad():
            for imgs, labels in tqdm(va_loader, desc=f"{arch_name}-{encoder_name} | Val"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.long().to(device, non_blocking=True)
                logits = model(imgs)
                per_px = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="none")
                valid = (labels != IGNORE_INDEX) & (labels >= 0) & (labels < NUM_CLASSES)
                valid_count = valid.sum().item()
                if valid_count == 0:
                    continue
                va_loss_sum += per_px[valid].sum().item()
                va_valid_px += valid_count
        avg_va = va_loss_sum / max(1, va_valid_px)
        print(f"[{arch_name}-{encoder_name}] Train {avg_tr:.4f} | Val {avg_va:.4f}")

        if va_valid_px > 0 and avg_va < best_val:
            best_val = avg_va
            torch.save(model.state_dict(), ckpt_best)
            print(f"[{arch_name}-{encoder_name}] - Saved the best val model!")

    torch.save(model.state_dict(), ckpt_last)

    model.load_state_dict(torch.load(ckpt_last, map_location=device))
    lines_last, _ = evaluate_test_metrics(model, te_loader, device, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    model.load_state_dict(torch.load(ckpt_best, map_location=device))
    lines_best, _ = evaluate_test_metrics(model, te_loader, device, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    with open(RESULTS_TXT, "a") as f:
        f.write(f"{arch_name} | Encoder: {encoder_name} | Bands: {L8_BANDS} | Size: {TILE_H}x{TILE_W} | Checkpoint: LAST\n")
        f.writelines(lines_last)
        f.write(f"{arch_name} | Encoder: {encoder_name} | Bands: {L8_BANDS} | Size: {TILE_H}x{TILE_W} | Checkpoint: BEST\n")
        f.writelines(lines_best)


# ---------------------
# Run
# ---------------------
if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_l8biome_datasets(L8_BANDS, TILE_H, TILE_W, split_ratio=(0.6, 0.2, 0.2))

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    in_ch = len(L8_BANDS)
    for enc in ENCODERS:
        for arch in ARCHS:
            train_and_test(arch, enc["name"], enc["weights"], train_loader, val_loader, test_loader, in_ch)
