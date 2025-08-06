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


def evaluate_val_iou(model, loader, device, num_classes=4, desc="Validation"):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            if isinstance(outputs, tuple):
                main_out = outputs[0]
            else:
                main_out = outputs

            preds = main_out.argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(),
                                              labels.cpu().numpy().ravel(),
                                              num_classes)

    ious = []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom > 0 else 0.0
        ious.append(iou)

    return np.mean(ious)

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


def evaluate_test_ext(model, loader, device, num_classes=4, ignore_index=None, desc="Testing"):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)

            if ignore_index is not None:
                valid_mask = labels != ignore_index
                preds = preds[valid_mask]
                labels = labels[valid_mask]

            preds_np = preds.cpu().numpy().ravel()
            labels_np = labels.cpu().numpy().ravel()

            if preds_np.size == 0:
                continue 

            conf_mat += fast_confusion_matrix(preds_np, labels_np, num_classes)
            total_correct += (preds_np == labels_np).sum()
            total_pixels += labels_np.size

    ious, f1s, accs, lines = [], [], [], []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = conf_mat.sum() - (TP + FP + FN)

        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)

        lines.append(f"  Class {i}: IoU={iou:.4f}, F1={f1:.4f}, Acc={acc:.4f}\n")

    mIoU = np.mean(ious)
    mF1 = np.mean(f1s)
    mAcc = np.mean(accs)
    aAcc = total_correct / total_pixels if total_pixels else 0.0

    lines.append(f"\n  Mean IoU (mIoU): {mIoU:.4f}")
    lines.append(f"\n  Mean Dice/F1 (mDice): {mF1:.4f}")
    lines.append(f"\n  Mean Accuracy (mAcc): {mAcc:.4f}")
    lines.append(f"\n  Overall Accuracy (aAcc): {aAcc:.4f}")
    
    return lines
