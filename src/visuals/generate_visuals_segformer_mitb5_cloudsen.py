import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from mmengine.config import Config
from mmseg.models import build_segmentor

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
        s2_l1c_path = record.read(0)
        s2_label_path = record.read(1)

        with rio.open(s2_l1c_path) as src, rio.open(s2_label_path) as dst:
            img = src.read(indexes=self.selected_bands).astype(np.float32)  # [C,H,W]
            label = dst.read(1).astype(np.uint8)                            # [H,W]

        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()
        return img, label


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


def _safe_load_state(model, model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded checkpoint: {model_path}")
    return model

def _minmax(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

def _tensor_to_rgb(img_t):
    x = img_t.detach().cpu().numpy()
    C, H, W = x.shape
    r, g, b = (3, 2, 1) if C >= 4 else (min(2, C-1), min(1, C-1), 0)
    return np.stack([_minmax(x[r]), _minmax(x[g]), _minmax(x[b])], axis=-1)

PALETTE = np.array([
    [135, 206, 235],
    [255, 255, 255],
    [200, 200, 200],
    [128, 128, 128],
], dtype=np.uint8)

def _colorize(mask_t, ignore_index=None):
    m = mask_t.detach().cpu().numpy().astype(np.int32)
    m2 = np.clip(m, 0, PALETTE.shape[0] - 1)
    out = PALETTE[m2]
    if ignore_index is not None:
        out[m == ignore_index] = [0, 0, 0]
    return out


taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l2a-high-512.taco"
indices = list(range(0, 10000))
train_indices = indices[:8500]
val_indices   = indices[8500:9000]
test_indices  = indices[9000:]
selected_bands = list(range(1, 14))

train_ds = CloudSegmentationDataset(taco_path, train_indices, selected_bands)
val_ds   = CloudSegmentationDataset(taco_path, val_indices, selected_bands)
test_ds  = CloudSegmentationDataset(taco_path, test_indices, selected_bands)


model = build_segmentor(cfg.model).to(device)

ckpt_path = "src/results/checkpoints/segformer_mitb5_cloudsen12_l2a.pth" 
model = _safe_load_state(model, ckpt_path)
model.eval()


idx = 745
img, label = test_ds[idx]
img_batch = img.unsqueeze(0).to(device)

with torch.no_grad():
    out = model.encode_decode(
        img_batch, [dict(img_shape=img_batch.shape[2:], ori_shape=img_batch.shape[2:])]
    )

    if isinstance(out, np.ndarray):
        out_t = torch.from_numpy(out)
    elif torch.is_tensor(out):
        out_t = out
    else:
        raise RuntimeError(f"Unsupported output type from encode_decode: {type(out)}")
    
    if out_t.ndim == 4:          # [N, C, H, W]
        pred_mask = out_t.argmax(dim=1)[0]
    elif out_t.ndim == 3:        # [C, H, W]
        pred_mask = out_t.argmax(dim=0)
    elif out_t.ndim == 2:        # [H, W]
        pred_mask = out_t
    else:
        raise RuntimeError(f"Unexpected output shape {out_t.shape}")



rgb       = _tensor_to_rgb(img)
gt_color  = _colorize(label)
pred_color= _colorize(pred_mask.cpu())

from PIL import Image
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

Image.fromarray(rgb).save(os.path.join(out_dir, "input_rgb.png"))
Image.fromarray(gt_color).save(os.path.join(out_dir, "ground_truth.png"))
Image.fromarray(pred_color).save(os.path.join(out_dir, "prediction.png"))


comparison = np.concatenate([rgb, gt_color, pred_color], axis=1)
Image.fromarray(comparison).save(os.path.join(out_dir, "comparison.png"))

print(f"[DONE] Saved outputs to {out_dir}")