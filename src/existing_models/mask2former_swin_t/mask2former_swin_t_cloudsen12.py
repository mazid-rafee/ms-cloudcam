import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------
# Custom Dataset
# --------------------
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
        return img, torch.from_numpy(label).long()

# --------------------
# Dataset Setup
# --------------------
taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
indices = list(range(10000))

train_indices = indices[:8500]
val_indices   = indices[8500:9000]
test_indices  = indices[9000:]

train_ds = CloudSegmentationDataset(taco_path, train_indices, [3, 4, 10])
val_ds   = CloudSegmentationDataset(taco_path, val_indices, [3, 4, 10])
test_ds  = CloudSegmentationDataset(taco_path, test_indices, [3, 4, 10])

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

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_mask2former_swin_t_cloudsen12.pth")
        
        torch.save(model.state_dict(), "results/last_model_mask2former_swin_t_cloudsen12.pth")

    print("\n=== Evaluating Best Model mask2former_swin_t_cloudsen12 ===")
    model.load_state_dict(torch.load("results/best_model_mask2former_swin_t_cloudsen12.pth"))
    evaluate_test(model, test_loader)

    print("\n=== Evaluating Last Model mask2former_swin_t_cloudsen12 ===")
    model.load_state_dict(torch.load("results/last_model_mask2former_swin_t_cloudsen12.pth"))
    evaluate_test(model, test_loader)
