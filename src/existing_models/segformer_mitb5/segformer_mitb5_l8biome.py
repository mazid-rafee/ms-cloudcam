import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

# Load your custom dataloader
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_loaders_path = os.path.join(parent_dir, 'data_loaders')
sys.path.append(data_loaders_path)
from l8_biome_dataloader import get_l8biome_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
train_ds, val_ds, test_ds = get_l8biome_datasets(
    [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], 512, 512, split_ratio=(0.6, 0.2, 0.2)
)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

# Define SegFormer (MiT-B5) config
cfg = Config(dict(
    model=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=len(train_ds.band_indices),
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
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
))

# Build model
model = build_segmentor(cfg.model)
#model.decode_head.loss_decode.ignore_index = 255
model.init_weights()
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

# Training function
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

# Validation function
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

# Evaluation function
def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            data_samples = model(imgs, mode='predict')
            preds = torch.stack([x.pred_sem_seg.data for x in data_samples])
            preds = preds.squeeze(1)  # [B, H, W]
            mask = (labels >= 0) & (labels < num_classes)
            conf_mat += np.bincount(
                num_classes * labels[mask].cpu().numpy() + preds[mask].cpu().numpy(),
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)
    # Metrics
    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom else 0
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"Class {i}: IoU={iou:.4f}, F1={f1:.4f}")
    lines.append(f"Mean IoU: {np.mean(ious):.4f}")
    lines.append(f"Mean F1: {np.mean(f1s):.4f}")
    print("\n".join(lines))
    return lines

# Driver
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    best_val_loss = 0.0

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_segformer_mitb5_l8biome.pth")
            print("Saved best_model_segformer_mitb5_l8biome")

    torch.save(model.state_dict(), "results/last_model_segformer_mitb5_l8biome.pth")

    print("\n=== Evaluating Best Model segformer_mitb5_l8biome ===")
    model.load_state_dict(torch.load("results/best_model_segformer_mitb5_l8biome.pth"))
    evaluate_test(model, test_loader)

    print("\n=== Evaluating Last Model segformer_mitb5_l8biome ===")
    model.load_state_dict(torch.load("results/last_model_segformer_mitb5_l8biome.pth"))
    evaluate_test(model, test_loader)
