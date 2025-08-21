import os, sys, importlib, argparse, torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from model.swin_crossattn_4w import Swin_CrossAttn_4W

def _safe_load_state(model, model_path):
    if model_path and os.path.isfile(model_path):
        sd = torch.load(model_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        sd = {k.replace("module.","").replace("model.",""): v for k,v in sd.items()}
        model.load_state_dict(sd, strict=False)
    return model

def _minmax(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx <= mn: return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn)/(mx-mn) * 255).clip(0,255).astype(np.uint8)

def _tensor_to_rgb(img_t):
    x = img_t.detach().cpu().numpy()
    C,H,W = x.shape
    r,g,b = (3,2,1) if C >= 4 else (min(2,C-1), min(1,C-1), 0)
    return np.stack([_minmax(x[r]), _minmax(x[g]), _minmax(x[b])], axis=-1)

PALETTE = np.array([
    [135,206,235],
    [255,255,255],
    [200,200,200],
    [128,128,128],
], dtype=np.uint8)

def _colorize(mask_t, ignore_index=None):
    m = mask_t.detach().cpu().numpy().astype(np.int32)
    m2 = np.clip(m, 0, PALETTE.shape[0]-1)
    out = PALETTE[m2]
    if ignore_index is not None:
        out[m == ignore_index] = [0,0,0]
    return out

DATASETS = {
    "cloudsen12_l1c": {
        "module": "data_loaders.cloudsen12_l1c_dataloader",
        "loader_fn": "get_cloudsen12_datasets",
        "bands": list(range(1,14)),
        "ckpt": "src/results/checkpoints/ms_cloudcam_2xdeepcross_attn_cloudsen12_l1c.pth",
        "ignore_index": None,
    },
    "cloudsen12_l2a": {
        "module": "data_loaders.cloudsen12_l2a_dataloader",
        "loader_fn": "get_cloudsen12_datasets",
        "bands": list(range(1,14)),
        "ckpt": "src/results/checkpoints/ms_cloudcam_2xdeepcross_attn_cloudsen12_l2a.pth",
        "ignore_index": None,
    },
    "l8biome": {
        "module": "data_loaders.l8_biome_dataloader",
        "loader_fn": "get_l8biome_datasets",
        "bands": list(range(1,12)),
        "ckpt": "src/results/checkpoints/ms_cloudcam_2xdeepcross_attn_l8biome.pth",
        "ignore_index": 255,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_out_dir = "src/results/visuals_one_separate"
os.makedirs(base_out_dir, exist_ok=True)

def _get_by_index(ds, idx):
    sample = ds[idx]
    if isinstance(sample,(list,tuple)):
        img, label = sample[0], sample[1]
    else:
        img, label = sample["image"], sample["label"]
    return img.unsqueeze(0), label.unsqueeze(0)

def run_one(name, idx=None):
    cfg = DATASETS[name]
    mdl = importlib.import_module(cfg["module"])
    get_fn = getattr(mdl, cfg["loader_fn"])
    _, _, test_ds = get_fn(cfg["bands"], split_ratio=(0.85,0.05,0.1))
    if idx is not None:
        if not (0 <= idx < len(test_ds)):
            raise IndexError(f"Requested idx {idx} out of range for {name} (len={len(test_ds)})")
        img, label = _get_by_index(test_ds, idx)
    else:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        batch = next(iter(test_loader))
        if isinstance(batch,(list,tuple)):
            img, label = batch[0], batch[1]
        else:
            img, label = batch["image"], batch["label"]
    img, label = img.to(device), label.to(device)
    in_channels = img.shape[1]
    model = Swin_CrossAttn_4W(in_channels=in_channels, num_classes=4).to(device)
    _safe_load_state(model, cfg["ckpt"])
    model.eval()
    with torch.no_grad():
        logits = model(img)
        if isinstance(logits,(list,tuple)): logits = logits[0]
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1)[0]
        rgb = _tensor_to_rgb(img[0])
        gt  = _colorize(label[0], ignore_index=cfg.get("ignore_index", None))
        pr  = _colorize(pred, ignore_index=None)
    out_dir = os.path.join(base_out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(rgb).save(os.path.join(out_dir, "input_rgb.png"))
    Image.fromarray(gt ).save(os.path.join(out_dir, "ground_truth.png"))
    Image.fromarray(pr ).save(os.path.join(out_dir, "prediction.png"))
    print(f"[ok] {name}: saved (overwritten)")

def _build_index_map(idx_args, dataset_names):
    m = {k: None for k in dataset_names}
    if not idx_args:
        return m
    ints = [int(t) for t in idx_args if t.isdigit()]
    kv = [t for t in idx_args if "=" in t]
    if kv:
        for tok in kv:
            k, v = tok.split("=", 1)
            if k in m:
                m[k] = int(v)
        if len(ints) == 1:
            d = ints[0]
            for k in m:
                if m[k] is None:
                    m[k] = d
        return m
    if len(ints) == 1:
        val = ints[0]
        for k in m:
            m[k] = val
        return m
    if len(ints) == len(dataset_names):
        for k, v in zip(dataset_names, ints):
            m[k] = v
        return m
    for i, k in enumerate(dataset_names):
        if i < len(ints):
            m[k] = ints[i]
        else:
            m[k] = ints[-1]
    return m

if __name__ == "__main__":
    idx_map = {
        "cloudsen12_l1c": 725,
        "cloudsen12_l2a": 745,
        "l8biome": 869,
    }

    for ds in ["cloudsen12_l1c", "cloudsen12_l2a", "l8biome"]:
        try:
            run_one(ds, idx=idx_map[ds])
        except Exception as e:
            print(f"[error] {ds}: {e}")

