import os
import sys
sys.path.append(os.path.abspath("DCNv4/segmentation"))
from mmseg_custom.models.backbones import FlashInternImage

import torch
import numpy as np
from PIL import Image
import rasterio as rio
import tacoreader
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from mmengine.config import Config
from mmseg.models import build_segmentor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Dataset
# -------------------------
class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        record = self.dataset.read(self.indices[idx])
        s2_l1c = record.read(0)
        s2_label = record.read(1)
        with rio.open(s2_l1c) as src, rio.open(s2_label) as dst:
            img = src.read(self.selected_bands).astype(np.float32)
            label = dst.read(1).astype(np.uint8)
        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()
        img = TF.resize(img, [512, 512])
        label = TF.resize(label.unsqueeze(0), [512, 512], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return img, label

# -------------------------
# Utilities
# -------------------------
def _minmax(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx <= mn: return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

def _tensor_to_rgb(img_t):
    x = img_t.detach().cpu().numpy()
    C, H, W = x.shape
    r, g, b = (2, 1, 0) if C >= 3 else (min(2, C-1), min(1, C-1), 0)
    return np.stack([_minmax(x[r]), _minmax(x[g]), _minmax(x[b])], axis=-1)

PALETTE = np.array([
    [135, 206, 235], # sky blue
    [255, 255, 255], # white (thick cloud)
    [200, 200, 200], # light gray (thin cloud)
    [128, 128, 128], # dark gray (shadow)
], dtype=np.uint8)

def _colorize(mask_t, ignore_index=None):
    m = mask_t.detach().cpu().numpy().astype(np.int32)
    m2 = np.clip(m, 0, PALETTE.shape[0] - 1)
    out = PALETTE[m2]
    if ignore_index is not None:
        out[m == ignore_index] = [0, 0, 0]
    return out

# -------------------------
# Visualization
# -------------------------
def visualize_sample(model, dataset, idx, ckpt_path, out_dir="outputs_upernet_internimage"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    img, label = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)

    H, W = img.shape[1:]
    img_metas = [dict(
        ori_shape=(H, W),
        img_shape=(H, W),
        pad_shape=(H, W),
        batch_input_shape=(H, W),
        scale_factor=1.0,
        flip=False
    )]

    with torch.no_grad():
        output = model.forward_test([img_batch], [img_metas])
        pred = output[0]  # numpy (H,W) or tensor
        if isinstance(pred, np.ndarray):
            pred_mask = torch.from_numpy(pred)
        else:
            pred_mask = pred.squeeze()

    rgb        = _tensor_to_rgb(img)
    gt_color   = _colorize(label)
    pred_color = _colorize(pred_mask.cpu())

    Image.fromarray(rgb).save(os.path.join(out_dir, f"sample{idx}_input_rgb.png"))
    Image.fromarray(gt_color).save(os.path.join(out_dir, f"sample{idx}_gt.png"))
    Image.fromarray(pred_color).save(os.path.join(out_dir, f"sample{idx}_pred.png"))
    print(f"[DONE] Visualization saved for sample {idx} -> {out_dir}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    cfg = Config.fromfile("DCNv4/segmentation/configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py")
    cfg.model.pretrained = None
    cfg.model.backbone.in_channels = 3
    cfg.model.decode_head.num_classes = 4
    if "auxiliary_head" in cfg.model:
        cfg.model.auxiliary_head.num_classes = 4

    model = build_segmentor(cfg.model).to(device)

    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l2a-high-512.taco"
    test_indices = list(range(9000, 10000))
    test_ds = CloudSegmentationDataset(taco_path, test_indices, [3, 4, 10])

    ckpt = "src/results/checkpoints/upernet_internimage_dcnv4_cloudsen12_l2a.pth"
    visualize_sample(model, test_ds, idx=745, ckpt_path=ckpt)
