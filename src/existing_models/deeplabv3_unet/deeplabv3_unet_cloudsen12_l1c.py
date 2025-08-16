import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import tacoreader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ---------------------
# Settings
# ---------------------
NUM_CLASSES = 4
BANDS = [1,2,3,4,5,6,7,8,9,10,11,12,13]
ENCODERS = [
    {"name": "resnet101", "weights": "imagenet"},
    {"name": "mobilenet_v2", "weights": "imagenet"},
]
ARCHS = ["DeepLabV3Plus", "UnetPlusPlus", "Unet"]  # train & test in this order
TACO_PATH = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
RESULTS_TXT = "results/Deeplabv3_Unet_CloudSen12.txt"
EPOCHS = 100
BS = 8
LR = 1e-3
NUM_WORKERS = 4

# ---------------------
# Dataset
# ---------------------
class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rec = self.dataset.read(self.indices[idx])
        s2_l1c = rec.read(0)
        s2_label = rec.read(1)
        with rio.open(s2_l1c) as src, rio.open(s2_label) as dst:
            img = src.read(self.selected_bands, window=rio.windows.Window(0, 0, 512, 512)).astype(np.float32)
            label = dst.read(1, window=rio.windows.Window(0, 0, 512, 512)).astype(np.uint8)
        img = (img / 3000.0).transpose(1, 2, 0)  # HWC
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(label)  # CHW, H

def split_indices(n=10000):
    idx = list(range(n))
    return idx[:8500], idx[8500:9000], idx[9000:]

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
# Evaluation (Mask2Former-style metrics)
# ---------------------
def evaluate_test_metrics(model, loader, device, num_classes=4):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs  = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # SMP forward -> logits: [B, C, H, W]
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            preds = torch.clamp(preds, 0, num_classes - 1)

            # valid pixels only
            mask = (labels >= 0) & (labels < num_classes)
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
def train_and_test(arch_name, encoder_name, encoder_weights, bands):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_idx, va_idx, te_idx = split_indices()

    tr_set = CloudSegmentationDataset(TACO_PATH, tr_idx, bands)
    va_set = CloudSegmentationDataset(TACO_PATH, va_idx, bands)
    te_set = CloudSegmentationDataset(TACO_PATH, te_idx, bands)

    tr_loader = DataLoader(tr_set, batch_size=BS, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_loader = DataLoader(te_set, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = get_model(arch_name, encoder_name, encoder_weights, in_channels=len(bands), classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs("results", exist_ok=True)
    best_val = float("inf")
    ckpt_best = f"results/best_{arch_name}_{encoder_name}_cloudsen12.pth"
    ckpt_last = f"results/last_{arch_name}_{encoder_name}_cloudsen12.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(tr_loader, desc=f"{arch_name}-{encoder_name} | Train {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_tr = total_loss / max(1, len(tr_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(va_loader, desc=f"{arch_name}-{encoder_name} | Val {epoch+1}/{EPOCHS}"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.long().to(device, non_blocking=True)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item()
        avg_va = val_loss / max(1, len(va_loader))

        print(f"[{arch_name}-{encoder_name}] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_tr:.4f} | Val Loss: {avg_va:.4f}")

        if avg_va < best_val:
            best_val = avg_va
            torch.save(model.state_dict(), ckpt_best)
            print(f"[{arch_name}-{encoder_name}] - Saved the best val model!")

    # Save last model
    torch.save(model.state_dict(), ckpt_last)

    # ---- Test best checkpoint ----
    model.load_state_dict(torch.load(ckpt_best, map_location=device))
    lines_best, _ = evaluate_test_metrics(model, te_loader, device, num_classes=NUM_CLASSES)
    with open(RESULTS_TXT, "a") as f:
        f.write(f"{arch_name} | Encoder: {encoder_name} | Bands: {bands} | BEST MODEL\n")
        f.writelines(lines_best)

    # ---- Test last trained model ----
    model.load_state_dict(torch.load(ckpt_last, map_location=device))
    lines_last, _ = evaluate_test_metrics(model, te_loader, device, num_classes=NUM_CLASSES)
    with open(RESULTS_TXT, "a") as f:
        f.write(f"{arch_name} | Encoder: {encoder_name} | Bands: {bands} | LAST MODEL\n")
        f.writelines(lines_last)

# ---------------------
# Run (train & test one after another)
# ---------------------
if __name__ == "__main__":
    for enc in ENCODERS:
        for arch in ARCHS:
            train_and_test(arch, enc["name"], enc["weights"], BANDS)
