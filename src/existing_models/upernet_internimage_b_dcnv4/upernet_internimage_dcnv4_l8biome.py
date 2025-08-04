import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
data_loaders_path = os.path.join(parent_dir, 'data_loaders')

sys.path.append(data_loaders_path)
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

sys.path.append(os.path.abspath("DCNv4/segmentation"))
from mmseg_custom.models.backbones import FlashInternImage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import tacoreader
from tqdm import tqdm
from mmengine.config import Config
from mmseg.models import build_segmentor
from l8_biome_dataloader import get_l8biome_datasets

band_combinations = [[3, 4, 9]]


def train_and_evaluate(bands, combo_index, total_combos, result_path):
    train_set, val_set, test_set = get_l8biome_datasets(
                [3, 4, 9], 512, 512, split_ratio=(0.6, 0.2, 0.2)
            )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    num_classes = 4

    cfg = Config.fromfile("DCNv4/segmentation/configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py")
    cfg.model.pretrained = None
    cfg.model.backbone.in_channels = len(bands)
    cfg.model.decode_head.num_classes = num_classes
    if "auxiliary_head" in cfg.model:
        cfg.model.auxiliary_head.num_classes = num_classes

    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    if hasattr(cfg.model.backbone, 'norm_cfg'):
        cfg.model.backbone.norm_cfg = cfg.norm_cfg
    if hasattr(cfg.model.decode_head, 'norm_cfg'):
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    if 'auxiliary_head' in cfg.model and hasattr(cfg.model.auxiliary_head, 'norm_cfg'):
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    cfg.model.test_cfg = dict(mode='whole')

    model = build_segmentor(cfg.model)
    model.init_weights()
    model = model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = f"checkpoints/best_model_upernet_internimage_dcnv4_l8biome.pth"
    last_model_path = f"checkpoints/last_model_upernet_internimage_dcnv4_l8biome.pth"

    for epoch in range(20):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Combo {combo_index+1}/{total_combos} | Epoch {epoch+1}/20"):
            imgs = imgs.to("cuda")
            labels = labels.long().unsqueeze(1).to("cuda")
            H, W = imgs.shape[2], imgs.shape[3]
            img_metas = [dict(ori_shape=(H, W), img_shape=(H, W), pad_shape=(H, W),
                              batch_input_shape=(H, W), scale_factor=1.0, flip=False)
                         for _ in range(imgs.size(0))]

            losses = model(imgs, img_metas=img_metas, gt_semantic_seg=labels)
            loss = sum(v for v in losses.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Avg Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval(); val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to("cuda")
                labels = labels.long().unsqueeze(1).to("cuda")
                H, W = imgs.shape[2], imgs.shape[3]
                img_metas = [dict(ori_shape=(H, W), img_shape=(H, W), pad_shape=(H, W),
                                  batch_input_shape=(H, W), scale_factor=1.0, flip=False)
                             for _ in range(imgs.size(0))]
                losses = model(imgs, img_metas=img_metas, gt_semantic_seg=labels)
                loss = sum(v for v in losses.values())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Avg Val Loss: {val_loss:.4f}")

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch+1}")

    torch.save(model.state_dict(), last_model_path)
    print("Saved last model.")

    def evaluate_model(model_path, label):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        total = 0
        total_iou = [0]*4
        total_f1 = [0]*4

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Evaluation ({label})"):
                imgs = imgs.to("cuda")
                labels = labels.to("cuda")
                batch_size, _, H, W = imgs.shape

                img_metas_batch = [
                    [dict(ori_shape=(H, W), img_shape=(H, W), pad_shape=(H, W),
                          batch_input_shape=(H, W), scale_factor=1.0, flip=False)]
                    for _ in range(batch_size)
                ]

                preds = []
                for i in range(batch_size):
                    output = model.forward_test([imgs[i].unsqueeze(0)], [img_metas_batch[i]])
                    pred_np = output[0]
                    pred_tensor = torch.from_numpy(pred_np) if isinstance(pred_np, np.ndarray) else pred_np
                    preds.append(pred_tensor)

                preds = torch.stack(preds)

                labels = labels.cpu()
                if labels.ndim == 4:
                    labels = labels.squeeze(1)

                assert preds.shape == labels.shape

                for c in range(4):
                    pred_c = preds == c
                    label_c = labels == c
                    inter = (pred_c & label_c).sum().float()
                    union = (pred_c | label_c).sum().float()
                    tp = inter
                    prec = tp / (pred_c.sum() + 1e-6)
                    rec = tp / (label_c.sum() + 1e-6)
                    total_iou[c] += (inter / (union + 1e-6)).item()
                    total_f1[c] += (2 * prec * rec / (prec + rec + 1e-6)).item()
                total += 1

        mean_iou = [x / total for x in total_iou]
        mean_f1 = [x / total for x in total_f1]
        class_metrics = "\n  " + "\n  ".join(
            [f"Class {c}: IoU={mean_iou[c]:.4f}, F1={mean_f1[c]:.4f}" for c in range(4)]
        )
        metrics = f"[{label} Model] {class_metrics}\n  Mean IoU: {np.mean(mean_iou):.4f}\n  Mean F1: {np.mean(mean_f1):.4f}"
        with open(result_path, "a") as f:
            f.write(f"Backbone: upernet_internimage_dcnv4_l8biome | Combo {combo_index+1}/{total_combos} | Bands: {bands} | {metrics}\n")
        print(metrics)

    evaluate_model(best_model_path, "Best upernet_internimage_dcnv4_l8biome")
    evaluate_model(last_model_path, "Last upernet_internimage_dcnv4_l8biome")

if __name__ == "__main__":
    result_file = f"results/UPerNet_InternImage.txt"
    for i, bands in enumerate(band_combinations):
        train_and_evaluate(bands, i, len(band_combinations), result_file)
