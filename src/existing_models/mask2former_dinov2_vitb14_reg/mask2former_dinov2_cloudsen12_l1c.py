# Added dinvov2.py in mmsegmentation/mmseg/models/backbones/
# Edited __init__.py in mmsegmentation/mmseg/models/backbones/ to include dinov2

import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/dinov2")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
import torchvision.transforms.functional as TF

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        record = self.dataset.read(self.indices[idx])
        s2_l1c_path = record.read(0)
        s2_label_path = record.read(1)
        with rio.open(s2_l1c_path) as src, rio.open(s2_label_path) as dst:
            img = src.read(indexes=self.selected_bands).astype(np.float32)
            label = dst.read(1).astype(np.uint8)
        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()
        img = TF.resize(img, [518,518])
        label = TF.resize(label.unsqueeze(0), [518,518], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return img, label

# ---------------------
# Config & Model
# ---------------------
cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py")
cfg.custom_imports = dict(imports=['mmseg.models.backbones.dinov2_backbone'], allow_failed_imports=False)
cfg.model.backbone = dict(type="DINOv2")
cfg.model.decode_head.in_channels = [768,768,768,768]
cfg.model.data_preprocessor = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)

# ---------------------
# Dataset Splits
# ---------------------
taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
indices = list(range(0, 10000))
train_indices = indices[:8500]
val_indices   = indices[8500:9000]
test_indices  = indices[9000:]

train_ds = CloudSegmentationDataset(taco_path, train_indices, [3, 4, 10])
val_ds   = CloudSegmentationDataset(taco_path, val_indices, [3, 4, 10])
test_ds  = CloudSegmentationDataset(taco_path, test_indices, [3, 4, 10])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

# ---------------------
# Train One Epoch
# ---------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        data_samples = []
        for label in labels:
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
            sample.set_metainfo(dict(img_shape=label.shape, ori_shape=label.shape))
            data_samples.append(sample)
        losses = model(imgs, data_samples, mode="loss")
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------
# Validate
# ---------------------
def validate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            data_samples = []
            for label in labels:
                sample = SegDataSample()
                sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
                sample.set_metainfo(dict(img_shape=label.shape, ori_shape=label.shape))
                data_samples.append(sample)
            losses = model(imgs, data_samples, mode="loss")
            loss = sum(v for v in losses.values())
            total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------
# Evaluate
# ---------------------
def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(518, 518), ori_shape=(518, 518))])
            preds = logits.argmax(dim=1)
            preds = torch.clamp(preds, 0, num_classes - 1)

            mask = (labels >= 0) & (labels < num_classes) & \
                   (preds >= 0) & (preds < num_classes)

            if mask.sum() == 0:
                continue

            p = preds[mask].cpu().numpy().ravel().astype(np.int64)
            y = labels[mask].cpu().numpy().ravel().astype(np.int64)

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

        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)
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
    return lines


# ---------------------
# Main Driver
# ---------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_mask2former_dinov2_cloudsen12_l1c.pth")
        
        torch.save(model.state_dict(), "results/last_model_mask2former_dinov2_cloudsen12_l1c.pth")

    print("\n=== Evaluating Best Model mask2former_dinov2_cloudsen12_l1c ===")
    model.load_state_dict(torch.load("results/best_model_mask2former_dinov2_cloudsen12_l1c.pth"))
    best_results = evaluate_test(model, test_loader)

    print("\n=== Evaluating Last Model mask2former_dinov2_cloudsen12_l1c ===")
    model.load_state_dict(torch.load("results/last_model_mask2former_dinov2_cloudsen12_l1c.pth"))
    last_results = evaluate_test(model, test_loader)