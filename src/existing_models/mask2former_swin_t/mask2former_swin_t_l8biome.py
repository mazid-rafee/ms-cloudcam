import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
data_loaders_path = os.path.join(parent_dir, 'data_loaders')

sys.path.append(data_loaders_path)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from tqdm import tqdm
from l8_biome_dataloader import get_l8biome_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Landsat 8 cirrus band 9, Sentinel 2 cirrus band 10
train_ds, val_ds, test_ds = get_l8biome_datasets(
                [3, 4, 9], 512, 512, split_ratio=(0.6, 0.2, 0.2)
            )

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

# --------------------
# Config + Model
# --------------------
cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py")
cfg.model.data_preprocessor = None

model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

# --------------------
# Train/Val Functions
# --------------------
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
            sample.set_metainfo(dict(
                img_shape=label.shape,
                ori_shape=label.shape
            ))
            data_samples.append(sample)
        losses = model(imgs, data_samples, mode="loss")
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            data_samples = []
            for label in labels:
                sample = SegDataSample()
                sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
                sample.set_metainfo(dict(
                    img_shape=label.shape,
                    ori_shape=label.shape
                ))
                data_samples.append(sample)
            losses = model(imgs, data_samples, mode="loss")
            loss = sum(v for v in losses.values())
            total_loss += loss.item()
    return total_loss / len(loader)

# --------------------
# Evaluation
# --------------------
def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(512,512), ori_shape=(512,512))])
            preds = logits.argmax(dim=1)
            mask = (labels >= 0) & (labels < num_classes)
            conf_mat += np.bincount(
                num_classes * labels[mask].cpu().numpy() + preds[mask].cpu().numpy(),
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)
    # calculate metrics
    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i,i]
        FP = conf_mat[:,i].sum() - TP
        FN = conf_mat[i,:].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom else 0
        prec = TP / (TP+FP) if (TP+FP) else 0
        rec = TP / (TP+FN) if (TP+FN) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"Class {i}: IoU={iou:.4f}, F1={f1:.4f}")
    lines.append(f"Mean IoU: {np.mean(ious):.4f}")
    lines.append(f"Mean F1: {np.mean(f1s):.4f}")
    print("\n".join(lines))
    return lines

# --------------------
# Driver
# --------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_mask2former_swin_t_l8biome.pth")
        
        torch.save(model.state_dict(), "results/last_model_mask2former_swin_t_l8biome.pth")

    print("\n=== Evaluating Best Model mask2former_swin_t_l8biome ===")
    model.load_state_dict(torch.load("results/best_model_mask2former_swin_t_l8biome.pth"))
    evaluate_test(model, test_loader)

    print("\n=== Evaluating Last Model mask2former_swin_t_l8biome ===")
    model.load_state_dict(torch.load("results/last_model_mask2former_swin_t_l8biome.pth"))
    evaluate_test(model, test_loader)
