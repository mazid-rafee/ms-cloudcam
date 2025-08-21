import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from torch.utils.data import Dataset
import rasterio as rio
import numpy as np
import tacoreader
from mmengine.config import Config
from mmseg.models import build_segmentor
from PIL import Image

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
            img = src.read(indexes=self.selected_bands).astype(np.float32)
            label = dst.read(1).astype(np.uint8)
        img = torch.from_numpy(img / 3000.0).float()
        return img, torch.from_numpy(label).long()

taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l2a-high-512.taco"
indices = list(range(10000))
test_indices = indices[9000:]
test_ds  = CloudSegmentationDataset(taco_path, test_indices, [3, 4, 10])

cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py")
cfg.model.data_preprocessor = None
model = build_segmentor(cfg.model).to(device)

def _safe_load_state(model, model_path):
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model

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

def visualize_sample(model, dataset, idx, out_dir="outputs", ckpt_path=None):
    os.makedirs(out_dir, exist_ok=True)
    if ckpt_path: model = _safe_load_state(model, ckpt_path)
    model.eval()
    img, label = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.encode_decode(img_batch, [dict(img_shape=img_batch.shape[2:], ori_shape=img_batch.shape[2:])])
        out_t = torch.from_numpy(out) if isinstance(out, np.ndarray) else out
        pred_mask = out_t.argmax(dim=1)[0] if out_t.ndim == 4 else out_t.argmax(dim=0) if out_t.ndim == 3 else out_t
    rgb        = _tensor_to_rgb(img)
    gt_color   = _colorize(label)
    pred_color = _colorize(pred_mask.cpu())
    Image.fromarray(rgb).save(os.path.join(out_dir, f"sample{idx}_input_rgb_l2a.png"))
    Image.fromarray(gt_color).save(os.path.join(out_dir, f"sample{idx}_ground_truth_l2a.png"))
    Image.fromarray(pred_color).save(os.path.join(out_dir, f"sample{idx}_prediction_l2a.png"))
    print(f"[DONE] Saved visualization for sample {idx} to {out_dir}")

if __name__ == "__main__":
    model = _safe_load_state(model, "src/results/checkpoints/mask2former_swin_t_cloudsen12_l2a.pth")
    visualize_sample(model, test_ds, idx=745, out_dir="outputs_mask2former")
