# Added dinvov2.py in mmsegmentation/mmseg/models/backbones/
# Edited __init__.py in mmsegmentation/mmseg/models/backbones/ to include dinov2


import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
data_loaders_path = os.path.join(parent_dir, 'data_loaders')

sys.path.append(data_loaders_path)
from l8_biome_dataloader import get_l8biome_datasets
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

# Landsat 8 cirrus band 9, Sentinel 2 cirrus band 10
train_ds, val_ds, test_ds = get_l8biome_datasets(
                [3, 4, 9], 518, 518, split_ratio=(0.6, 0.2, 0.2)
            )

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

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
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(518,518), ori_shape=(518,518))])
            preds = logits.argmax(dim=1)
            preds = torch.clamp(preds, 0, num_classes-1)
            mask = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
            flat = num_classes * labels[mask].cpu().numpy() + preds[mask].cpu().numpy()
            bincount = np.bincount(flat, minlength=num_classes**2)
            conf_mat += bincount.reshape(num_classes, num_classes)

    lines, ious, f1s = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i,i]
        FP = conf_mat[:,i].sum() - TP
        FN = conf_mat[i,:].sum() - TP
        iou = TP/(TP+FP+FN) if (TP+FP+FN) else 0
        prec = TP/(TP+FP) if (TP+FP) else 0
        rec = TP/(TP+FN) if (TP+FN) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"Class {i}: IoU={iou:.4f}, F1={f1:.4f}\n")
    lines.append(f"Mean IoU: {np.mean(ious):.4f}\nMean F1: {np.mean(f1s):.4f}\n")
    print("".join(lines))
    return lines

# ---------------------
# Main Driver
# ---------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_mask2former_dinov2_l8biome.pth")
        
        torch.save(model.state_dict(), "results/last_model_mask2former_dinov2_l8biome.pth")

    print("\n=== Evaluating Best Model mask2former_dinov2_l8biome ===")
    model.load_state_dict(torch.load("results/best_model_mask2former_dinov2_l8biome.pth"))
    best_results = evaluate_test(model, test_loader)

    print("\n=== Evaluating Last Model mask2former_dinov2_l8biome ===")
    model.load_state_dict(torch.load("results/last_model_mask2former_dinov2_l8biome.pth"))
    last_results = evaluate_test(model, test_loader)