import sys
sys.path.insert(0, "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Segmentation/mmsegmentation")
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import numpy as np
from PIL import Image

from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_loaders_path = os.path.join(parent_dir, 'data_loaders')
sys.path.append(data_loaders_path)
from l8_biome_dataloader import get_l8biome_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BAND_INDICES = [1,2,3,4,5,6,7,9,10,11]
NUM_CLASSES  = 4
IGNORE_INDEX = 255
SCALE_DIVISOR = 10000.0
CKPT_PATH = "src/results/checkpoints/segformer_mitb5_l8biome.pth"
OUT_DIR   = "l8biome_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = np.array([
    [135,206,235],
    [255,255,255],
    [200,200,200],
    [128,128,128],
], dtype=np.uint8)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)

def colorize(mask_t, ignore_index=None):
    m = mask_t.detach().cpu().numpy().astype(np.int32)
    if ignore_index is not None:
        full_palette = np.vstack([PALETTE, IGNORE_COLOR[None, :]])
        m = np.where(m == ignore_index, NUM_CLASSES, m)
    else:
        full_palette = PALETTE
    m2 = np.clip(m, 0, full_palette.shape[0] - 1)
    return full_palette[m2]

def normalize_labels_if_1_based(label_t, ignore_index=IGNORE_INDEX):
    mn = int(label_t.min().item())
    mx = int(label_t.max().item())
    if mn >= 1 and mx <= NUM_CLASSES:
        return torch.where(label_t == ignore_index, label_t, label_t - 1)
    return label_t

def _minmax(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip((x - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

def tensor_to_rgb(img_t):
    x = img_t.detach().cpu().numpy()
    C, H, W = x.shape
    def band_to_idx(b):
        if b in BAND_INDICES:
            return BAND_INDICES.index(b)
        return min(max(b-1, 0), C-1)
    r = band_to_idx(4)
    g = band_to_idx(3)
    b = band_to_idx(2)
    return np.stack([_minmax(x[r]), _minmax(x[g]), _minmax(x[b])], axis=-1)

cfg = Config(dict(
    model=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=len(BAND_INDICES),
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
            num_classes=NUM_CLASSES,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
))

def safe_load_state(model, model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {model_path}")
    sd = torch.load(model_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded: {model_path}")
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    return model

_, _, test_ds = get_l8biome_datasets(BAND_INDICES, 512, 512, split_ratio=(0.85,0.05,0.1))

def run_single_sample(idx):
    model = build_segmentor(cfg.model).to(device)
    if hasattr(model.decode_head, 'loss_decode'):
        model.decode_head.loss_decode.ignore_index = IGNORE_INDEX
    try:
        model = safe_load_state(model, CKPT_PATH)
    except FileNotFoundError as e:
        print(e)
        print("[INFO] Proceeding with ImageNet-pretrained backbone (random head).")
    model.eval()

    img, label = test_ds[idx]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.max() > 1.5 and SCALE_DIVISOR is not None and SCALE_DIVISOR > 0:
        img = img / SCALE_DIVISOR
    img = img.float()

    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label).long()

    label = normalize_labels_if_1_based(label, ignore_index=IGNORE_INDEX)

    img_b = img.unsqueeze(0).to(device)
    H, W = label.shape[-2], label.shape[-1]
    data_sample = SegDataSample()
    data_sample.set_metainfo(dict(img_shape=(H,W), ori_shape=(H,W)))

    with torch.no_grad():
        out = model.encode_decode(img_b, [dict(img_shape=img_b.shape[2:], ori_shape=img_b.shape[2:])])
        if isinstance(out, np.ndarray):
            out_t = torch.from_numpy(out)
        elif torch.is_tensor(out):
            out_t = out
        else:
            raise RuntimeError(f"Unsupported output type: {type(out)}")
        if out_t.ndim == 4:
            pred_mask = out_t.argmax(dim=1)[0]
        elif out_t.ndim == 3:
            pred_mask = out_t.argmax(dim=0)
        elif out_t.ndim == 2:
            pred_mask = out_t
        else:
            raise RuntimeError(f"Unexpected output shape {out_t.shape}")

    rgb        = tensor_to_rgb(img)
    gt_color   = colorize(label, ignore_index=IGNORE_INDEX)
    pred_color = colorize(pred_mask.cpu(), ignore_index=None)

    Image.fromarray(rgb).save(os.path.join(OUT_DIR, "l8b_input_rgb.png"))
    Image.fromarray(gt_color).save(os.path.join(OUT_DIR, "l8b_ground_truth.png"))
    Image.fromarray(pred_color).save(os.path.join(OUT_DIR, "l8b_prediction.png"))
    comp = np.concatenate([rgb, gt_color, pred_color], axis=1)
    Image.fromarray(comp).save(os.path.join(OUT_DIR, "l8b_comparison.png"))
    print(f"[DONE] Saved outputs to {OUT_DIR}")

if __name__ == "__main__":
    run_single_sample(idx=869)
