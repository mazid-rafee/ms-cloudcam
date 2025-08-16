import os
import argparse
import torch
from torch.utils.data import DataLoader
from data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader, l8_biome_dataloader
from model.swin_crossattn_4w import Swin_CrossAttn_4W
from utils.helpers import seed_everything, seed_worker, evaluate_ext

parser = argparse.ArgumentParser(description="Evaluate MSCloudCAM on datasets")
parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

seed = 42
seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = os.path.join("src", "results")

# -------- CloudSEN12 L1C evaluation --------
selected_bands = list(range(1, 14))
train_ds, val_ds, test_ds = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
    selected_bands, split_ratio=(0.85, 0.05, 0.1)
)
model_name = "checkpoints/ms_cloudcam_cloudsen12_l1c.pth"
ignore_index = None

g = torch.Generator().manual_seed(seed)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model_path = os.path.join(results_dir, model_name)
evaluate_ext(model, model_path, test_loader, device, ignore_index=ignore_index, description="on cloudsen12_l1c dataset - MSCloudCAM")

# -------- CloudSEN12 L2A evaluation --------
selected_bands = list(range(1, 14))
train_ds, val_ds, test_ds = cloudsen12_l2a_dataloader.get_cloudsen12_datasets(
    selected_bands, split_ratio=(0.85, 0.05, 0.1)
)
model_name = "checkpoints/ms_cloudcam_cloudsen12_l2a.pth"
ignore_index = None

g = torch.Generator().manual_seed(seed)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model_path = os.path.join(results_dir, model_name)
evaluate_ext(model, model_path, test_loader, device, ignore_index=ignore_index, description="on cloudsen12_l2a dataset - MSCloudCAM")

# -------- L8Biome evaluation --------
selected_bands = list(range(1, 12))
train_ds, val_ds, test_ds = l8_biome_dataloader.get_l8biome_datasets(
    selected_bands, 512, 512, split_ratio=(0.6, 0.2, 0.2)
)
model_name = "checkpoints/ms_cloudcam_l8biome.pth"
ignore_index = 255

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model_path = os.path.join(results_dir, model_name)
evaluate_ext(model, model_path, test_loader, device, ignore_index=ignore_index, description="on l8biome dataset - MSCloudCAM")
