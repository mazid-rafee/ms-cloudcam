import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.dataset.read(self.indices[idx])
        s2_l2a_path = record.read(0)
        s2_label_path = record.read(1)

        with rio.open(s2_l2a_path) as src, rio.open(s2_label_path) as dst:
            img = src.read(indexes=self.selected_bands).astype(np.float32)
            label = dst.read(1).astype(np.uint8)

        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()

        return img, label

# Dataset setup
taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l2a-high-512.taco"
indices = list(range(0, 10000))
train_indices = indices[:8500]
val_indices = indices[8500:9000]
test_indices = indices[9000:]
selected_bands = list(range(1, 14))

train_ds = CloudSegmentationDataset(taco_path, train_indices, selected_bands)
val_ds   = CloudSegmentationDataset(taco_path, val_indices, selected_bands)
test_ds  = CloudSegmentationDataset(taco_path, test_indices, selected_bands)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

# Segformer model config
cfg = Config(dict(
    model=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=13,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 18, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            norm_cfg=dict(type='LN', requires_grad=True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
            )
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=512,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
))

model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        data_samples = []
        for i in range(labels.shape[0]):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=labels[i])
            data_samples.append(sample)
        losses = model(imgs, data_samples, mode='loss')
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
            for i in range(labels.shape[0]):
                sample = SegDataSample()
                sample.gt_sem_seg = PixelData(data=labels[i])
                data_samples.append(sample)
            losses = model(imgs, data_samples, mode='loss')
            loss = sum(v for v in losses.values())
            total_loss += loss.item()
    return total_loss / len(loader)

def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(
        num_classes * labels[mask].astype(int) + preds[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(512, 512), ori_shape=(512, 512))])
            preds = logits.argmax(dim=1)
            preds_np = preds.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            conf_mat += fast_confusion_matrix(preds_np, labels_np, num_classes)
            total_correct += (preds_np == labels_np).sum()
            total_pixels += labels_np.size

    ious, f1s, accs, lines = [], [], [], []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = conf_mat.sum() - (TP + FP + FN)

        denom = TP + FP + FN
        iou = TP / denom if denom else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)
        lines.append(f"  Class {i}: IoU={iou:.4f}, F1={f1:.4f}, Acc={acc:.4f}\n")

    mIoU = np.mean(ious)
    mF1 = np.mean(f1s)
    mAcc = np.mean(accs)
    aAcc = total_correct / total_pixels if total_pixels else 0.0

    lines.append(f"\n  Mean IoU (mIoU): {mIoU:.4f}")
    lines.append(f"\n  Mean F1 (mF1): {mF1:.4f}")
    lines.append(f"\n  Mean Accuracy (mAcc): {mAcc:.4f}")
    lines.append(f"\n  Overall Accuracy (aAcc): {aAcc:.4f}")

    print("".join(lines))
    return lines


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    best_val_loss = 0.0

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_segformer_mitb5_cloudsen12_l2a.pth")
            print("Saved best_model_segformer_mitb5_cloudsen12_l2a")

    torch.save(model.state_dict(), "results/last_model_segformer_mitb5_cloudsen12_l2a.pth")

    print("\n=== Evaluating best_model_segformer_mitb5_cloudsen12_l2a ===")
    model.load_state_dict(torch.load("results/best_model_segformer_mitb5_cloudsen12_l2a.pth"))
    best_results = evaluate_test(model, test_loader)
    print("".join(best_results))

    print("\n=== Evaluating last_model_segformer_mitb5_cloudsen12_l2a ===")
    model.load_state_dict(torch.load("results/last_model_segformer_mitb5_cloudsen12_l2a.pth"))
    last_results = evaluate_test(model, test_loader)
    print("".join(last_results))
