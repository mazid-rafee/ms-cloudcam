import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/dinov2")

import os
import torch
import numpy as np
from PIL import Image
import rasterio as rio
import tacoreader
from torch.utils.data import Dataset
from mmengine.config import Config
from mmseg.models import build_segmentor
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # 🔑 resize to 518×518 for DINOv2 consistency
        img = TF.resize(img, [518, 518])
        label = TF.resize(label.unsqueeze(0), [518, 518], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return img, label

# ---------------------
# Utilities
# ---------------------
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
    [135, 206, 235], # class 0
    [255, 255, 255], # class 1
    [200, 200, 200], # class 2
    [128, 128, 128], # class 3
], dtype=np.uint8)

def _colorize(mask_t, ignore_index=None):
    m = mask_t.detach().cpu().numpy().astype(np.int32)
    m2 = np.clip(m, 0, PALETTE.shape[0] - 1)
    out = PALETTE[m2]
    if ignore_index is not None:
        out[m == ignore_index] = [0, 0, 0]
    return out

def _safe_load_state(model, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {model_path}")

    print(f"[INFO] Loading checkpoint: {model_path}")
    sd = torch.load(model_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            # Fix key prefix mismatch
            if not k.startswith("backbone.model."):
                new_sd["backbone.model." + k[len("backbone."):]] = v
            else:
                new_sd[k] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    if len(missing) == 0 and len(unexpected) == 0:
        print("[INFO] All checkpoint weights loaded successfully.")
    else:
        print("[WARNING] Some keys did not match after remapping:")
        if len(missing) > 0:
            print(f"  Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if len(unexpected) > 0:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    return model


# ---------------------
# Visualization
# ---------------------
def visualize_sample(model, dataset, idx, out_dir="outputs_dinov2", ckpt_path=None):
    os.makedirs(out_dir, exist_ok=True)
    if ckpt_path: model = _safe_load_state(model, ckpt_path)
    model.eval()
    img, label = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.encode_decode(img_batch, [dict(img_shape=(518,518), ori_shape=(518,518))])
        out_t = torch.from_numpy(out) if isinstance(out, np.ndarray) else out
        pred_mask = out_t.argmax(dim=1)[0]
    rgb        = _tensor_to_rgb(img)
    gt_color   = _colorize(label)
    pred_color = _colorize(pred_mask.cpu())
    Image.fromarray(rgb).save(os.path.join(out_dir, f"sample{idx}_input_rgb_dinov2.png"))
    Image.fromarray(gt_color).save(os.path.join(out_dir, f"sample{idx}_ground_truth_dinov2.png"))
    Image.fromarray(pred_color).save(os.path.join(out_dir, f"sample{idx}_prediction_dinov2.png"))
    print(f"[DONE] Saved visualization for sample {idx} to {out_dir}")

# ---------------------
# Main Driver
# ---------------------
if __name__ == "__main__":
    cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py")
    cfg.custom_imports = dict(imports=['mmseg.models.backbones.dinov2_backbone'], allow_failed_imports=False)
    cfg.model.backbone = dict(type="DINOv2")
    cfg.model.decode_head.in_channels = [768,768,768,768]
    cfg.model.data_preprocessor = None

    model = build_segmentor(cfg.model).to(device)
    model = _safe_load_state(model, "src/results/checkpoints/mask2former_dinov2_cloudsen12_l2a.pth")

    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l2a-high-512.taco"
    test_indices = list(range(9000, 10000))
    test_ds = CloudSegmentationDataset(taco_path, test_indices, [3, 4, 10])

    # 🔑 visualize sample 725
    visualize_sample(model, test_ds, idx=745, out_dir="outputs_mask2former_dinov2")
