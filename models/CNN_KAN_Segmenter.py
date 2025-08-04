import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm
from fastkan import FastKAN as KAN
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ECABlock(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1) 
        y = self.sigmoid(y)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared(self.avg_pool(x))
        max_out = self.shared(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class CombinedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se = SEBlock(in_channels)
        self.cbam = CBAM(in_channels)
        self.eca = ECABlock(in_channels)

    def forward(self, x):
        x = self.se(x)
        x = self.cbam(x)
        x = self.eca(x)
        return x


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
        return img, label
    
class L8BiomePatchDataset(Dataset):
    def __init__(self, scene_dirs, patch_size, stride, selected_bands):
        self.scene_dirs = scene_dirs
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        self.band_indices = selected_bands

        for scene_dir in self.scene_dirs:
            scene_id = os.path.basename(scene_dir)
            img_path = os.path.join(scene_dir, f"{scene_id}.TIF")
            mask_path = os.path.join(scene_dir, f"{scene_id}_fixedmask.TIF")

            with rio.open(img_path) as img:
                h, w = img.height, img.width

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_path, mask_path, x, y))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.patches[idx]

        with rio.open(img_path) as src:
            img = src.read(self.band_indices, window=rio.windows.Window(x, y, self.patch_size, self.patch_size)).astype(np.float32)

        with rio.open(mask_path) as dst:
            label_raw = dst.read(1, window=rio.windows.Window(x, y, self.patch_size, self.patch_size)).astype(np.uint8)

        mapping = {
            128: 0,  # Clear
            255: 1,  # Thick Cloud
            192: 2,  # Thin Cloud
            64:  3   # Shadow
        }

        label = np.full_like(label_raw, fill_value=255)
        for k, v in mapping.items():
            label[label_raw == k] = v

        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()

        return img, label


class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        output = torch.cat(pyramids, dim=1)
        return self.bottleneck(output)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)

class CNN_CrossAttention(nn.Module):
    def __init__(self, in_dim, context_dim=None, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (in_dim // heads) ** 0.5

        context_dim = context_dim or in_dim

        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(context_dim, in_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(context_dim, in_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    def forward(self, x, context):
        B, C, H, W = x.shape

        # Linear projections
        Q = self.q_proj(x)   # [B, C, H, W]
        K = self.k_proj(context)
        V = self.v_proj(context)

        # Flatten spatial dims
        Q = Q.view(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)  # [B, heads, HW, C//heads]
        K = K.view(B, self.heads, C // self.heads, -1)                      # [B, heads, C//heads, HW]
        V = V.view(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)  # [B, heads, HW, C//heads]

        # Attention
        attn = torch.matmul(Q, K) / self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # [B, heads, HW, C//heads]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        return self.out_proj(out) + x  # Residual


class DeepCrossAttention(nn.Module):
    def __init__(self, in_dim, context_dim=None, heads=4, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            CNN_CrossAttention(in_dim, context_dim, heads)
            for _ in range(depth)
        ])

    def forward(self, x, context):
        for block in self.blocks:
            x = block(x, context)
        return x
    
class CNN_KAN_Segmenter(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.aspp = ASPP(512, 256)
        self.psp = PSPModule(512, [1, 2, 3, 6], 256)
        self.deep_attn = DeepCrossAttention(in_dim=512, context_dim=256, heads=4, depth=2)
        self.cbam = CombinedAttention(512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),  # Reduce channels
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Process
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),  # Restore channels
            nn.ReLU()
        )

        #self.kan = KAN([512, 256, 256, 512])

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.aux1 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.aux2 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.final_out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x_aspp = self.aspp(x)
        x_psp = self.psp(x)
        x = torch.cat([x_aspp, x_psp], dim=1)
        x = self.deep_attn(x, context=x_psp)
        
        x = self.cbam(x)
        x = self.bottleneck(x)

        # B, C, H, W = x.shape
        # x = x.permute(0, 2, 3, 1).reshape(-1, C)
        # x = self.kan(x)
        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.dec1(x)
        aux_out1 = self.aux1(F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False))

        x = self.dec2(x)
        aux_out2 = self.aux2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

        x = self.dec3(x)
        x = self.dec4(x)
        out = self.final_out(x)

        return out, aux_out1, aux_out2

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)


    for imgs, labels in tqdm(loader, desc='Training l8biome bottlekenck_no_kan_deepcrossattn_b8_seeded42'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        main_out, aux1, aux2 = model(imgs)

        # Ensure aux outputs match label spatial size
        target_size = labels.shape[-2:]
        aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
        aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

        loss_main = loss_fn(main_out, labels)
        loss_aux1 = loss_fn(aux1, labels)
        loss_aux2 = loss_fn(aux2, labels)

        loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(num_classes * labels[mask] + preds[mask], minlength=num_classes**2).reshape(num_classes, num_classes)

def evaluate_val(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(), labels.cpu().numpy().ravel(), num_classes)
    ious = []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        ious.append(iou)
    return np.mean(ious)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing l8biome bottlekenck_no_kan_deepcrossattn_b8_seeded42"):
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Helper functions
def find_valid_scene_dirs(root_dir):
    scene_dirs = []
    for cover_type in os.listdir(root_dir):
        cover_path = os.path.join(root_dir, cover_type)
        if not os.path.isdir(cover_path):
            continue
        for scene_id in os.listdir(cover_path):
            scene_path = os.path.join(cover_path, scene_id)
            if not os.path.isdir(scene_path):
                continue
            tif_file = os.path.join(scene_path, f"{scene_id}.TIF")
            mask_file = os.path.join(scene_path, f"{scene_id}_fixedmask.TIF")
            if os.path.isfile(tif_file) and os.path.isfile(mask_file):
                scene_dirs.append(scene_path)

    random.shuffle(scene_dirs, random=random.Random(42).random)

    return scene_dirs

# Main script
if __name__ == '__main__':
    root_dir = "data/l8_biome/l8biome"
    all_dirs = find_valid_scene_dirs(root_dir)

    # 60/20/20 split
    total = len(all_dirs)
    train_end = int(0.6 * total)
    val_end = int(0.8 * total)

    scene_split = {
        'train': all_dirs[:train_end],
        'val': all_dirs[train_end:val_end],
        'test': all_dirs[val_end:]
    }

    band_sets = {
        "Bands_All_1_to_12": list(range(1, 12))
    }

    os.makedirs("results", exist_ok=True)
    log_path = "results/CNN_KAN_Segmenter.txt"

    with open(log_path, "a") as log_file:
        for name, selected_bands in band_sets.items():
            # Optional: Patch-based datasets
            train_ds = L8BiomePatchDataset(scene_split['train'], patch_size=512, stride=512, selected_bands=selected_bands)
            val_ds = L8BiomePatchDataset(scene_split['val'], patch_size=512, stride=512, selected_bands=selected_bands)
            test_ds = L8BiomePatchDataset(scene_split['test'], patch_size=512, stride=512, selected_bands=selected_bands)

            g = torch.Generator().manual_seed(42)
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4,
                                    worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

            # Train your model here
            # train_model(train_loader, val_loader, test_loader, ...)

            model = CNN_KAN_Segmenter(in_channels=len(selected_bands), num_classes=4).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            best_miou = 0.0
            best_model_path = "results/CNN_KAN_Segmenter_l8biome_best_bottlekenck_no_kan_deepcrossattn_b8_seeded42.pth"
            last_model_path = "results/CNN_KAN_Segmenter_l8biome_last_bottlekenck_no_kan_deepcrossattn_b8_seeded42.pth"

            for epoch in range(100):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_miou = evaluate_val(model, val_loader)
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val mIoU = {val_miou:.4f}")

                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save(model.state_dict(), best_model_path)
                    print("Saved best model!")

            torch.save(model.state_dict(), last_model_path)

            print("\nEvaluating best model l8biome bottlekenck_no_kan_deepcrossattn_b8_seeded42:")
            model.load_state_dict(torch.load(best_model_path))
            results_best = evaluate_test(model, test_loader)
            print("".join(results_best))
            log_file.write("\nEvaluation of Best Model l8biome bottlekenck_no_kan_deepcrossattn_b8_seeded42:\n")
            log_file.writelines(results_best)
            log_file.write("\n")

            print("\nEvaluating last model l8biome bottlekenck_no_kan_deepcrossattn_b8_seeded42:")
            model.load_state_dict(torch.load(last_model_path))
            results_last = evaluate_test(model, test_loader)
            print("".join(results_last))
            log_file.write("\nEvaluation of Last Model l8biome bottlekenck_no_kan_deepcrossattn_b8_seeded42:\n")
            log_file.writelines(results_last)
            log_file.write("\n")
