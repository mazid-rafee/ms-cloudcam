import sys
import os
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
data_loaders_path = os.path.join(parent_dir, 'data_loaders')
sys.path.append(data_loaders_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from torch.utils.data import DataLoader
import numpy as np
from mmengine.config import Config
from mmseg.models import build_segmentor
from l8_biome_dataloader import get_l8biome_datasets
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, test_ds = get_l8biome_datasets([3, 4, 9], 512, 512, split_ratio=(0.85, 0.05, 0.1))
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

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

def visualize_sample(model, dataset, idx, out_dir="outputs_l8biome", ckpt_path=None):
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
    Image.fromarray(rgb).save(os.path.join(out_dir, f"sample{idx}_input_rgb.png"))
    Image.fromarray(gt_color).save(os.path.join(out_dir, f"sample{idx}_ground_truth.png"))
    Image.fromarray(pred_color).save(os.path.join(out_dir, f"sample{idx}_prediction.png"))
    comparison = np.concatenate([rgb, gt_color, pred_color], axis=1)
    Image.fromarray(comparison).save(os.path.join(out_dir, f"sample{idx}_comparison.png"))
    print(f"[DONE] Saved visualization for sample {idx} to {out_dir}")

if __name__ == "__main__":
    model = _safe_load_state(model, "src/results/checkpoints/mask2former_swin_t_l8biome.pth")
    visualize_sample(model, test_ds, idx=869, out_dir="outputs_l8biome")
