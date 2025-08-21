import sys, os, torch
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/dinov2")

from mmengine.config import Config
from mmseg.models import build_segmentor
from PIL import Image
import numpy as np

# L8Biome loader
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(os.path.join(parent_dir, 'data_loaders'))
from l8_biome_dataloader import get_l8biome_datasets

# ---------------------
# Helpers
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
    print(f"[INFO] Loading checkpoint: {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            if not k.startswith("backbone.model."):
                new_sd["backbone.model." + k[len("backbone."):]] = v
            else:
                new_sd[k] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[INFO] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model

def visualize_sample(model, dataset, idx=0, out_dir="outputs_l8biome"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    img, label = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.encode_decode(img_batch, [dict(img_shape=(518,518), ori_shape=(518,518))])
        pred_mask = out.argmax(dim=1)[0].cpu()
    rgb        = _tensor_to_rgb(img)
    gt_color   = _colorize(label)
    pred_color = _colorize(pred_mask)
    Image.fromarray(rgb).save(os.path.join(out_dir, f"sample{idx}_input.png"))
    Image.fromarray(gt_color).save(os.path.join(out_dir, f"sample{idx}_gt.png"))
    Image.fromarray(pred_color).save(os.path.join(out_dir, f"sample{idx}_pred.png"))
    print(f"[DONE] Saved visualization for sample {idx}")

# ---------------------
# Config + Model
# ---------------------
cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py")
cfg.custom_imports = dict(imports=['mmseg.models.backbones.dinov2_backbone'], allow_failed_imports=False)
cfg.model.backbone = dict(type="DINOv2")
cfg.model.decode_head.in_channels = [768,768,768,768]
cfg.model.data_preprocessor = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_segmentor(cfg.model)
model.decode_head.loss_cls.ignore_index = 255
model = _safe_load_state(model, "src/results/checkpoints/mask2former_dinov2_l8biome.pth")
model = model.to(device)

# ---------------------
# Dataset + Run Viz
# ---------------------
_, _, test_ds = get_l8biome_datasets([3, 4, 9], 518, 518, split_ratio=(0.85, 0.05, 0.1))
visualize_sample(model, test_ds, idx=869, out_dir="outputs_l8biome")
