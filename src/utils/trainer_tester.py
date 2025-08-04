import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device, desc="Training", ignore_index=None):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    for imgs, labels in tqdm(loader, desc=desc):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)

        if isinstance(outputs, tuple) and len(outputs) == 3:
            main_out, aux1, aux2 = outputs

            target_size = labels.shape[-2:]
            aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

            loss_main = loss_fn(main_out, labels)
            loss_aux1 = loss_fn(aux1, labels)
            loss_aux2 = loss_fn(aux2, labels)

            loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2
        else:
            main_out = outputs
            loss = loss_fn(main_out, labels)

        if torch.isnan(loss):
            print("NaN loss detected!")
            print("Labels unique:", labels.unique())
            print("Loss input stats:", main_out.min().item(), main_out.max().item())
            raise ValueError("NaN loss. stopping")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(num_classes * labels[mask] + preds[mask], minlength=num_classes**2).reshape(num_classes, num_classes)


def evaluate_val(model, loader, device, desc="Validation", ignore_index=None):
    model.eval()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            if isinstance(outputs, tuple) and len(outputs) == 3:
                main_out, aux1, aux2 = outputs

                target_size = labels.shape[-2:]
                aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
                aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

                loss_main = loss_fn(main_out, labels)
                loss_aux1 = loss_fn(aux1, labels)
                loss_aux2 = loss_fn(aux2, labels)

                loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2
            else:
                main_out = outputs
                loss = loss_fn(main_out, labels)

            total_loss += loss.item()

    return total_loss / len(loader)



def evaluate_test(model, loader, device, num_classes=4, desc="Testing"):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(), labels.cpu().numpy().ravel(), num_classes)

    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"  Class {i}: IoU={iou:.4f}, F1={f1:.4f}\n")
    lines.append(f"  Mean IoU: {np.mean(ious):.4f}\n")
    lines.append(f"  Mean F1: {np.mean(f1s):.4f}\n")
    return lines
