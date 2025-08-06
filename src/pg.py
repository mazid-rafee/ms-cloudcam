import os
import argparse
import torch
from torch.utils.data import DataLoader
from data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader, l8_biome_dataloader
from model.swin_crossattn_4w import Swin_CrossAttn_4W
from utils.helpers import seed_everything, seed_worker, evaluate_and_log, evaluate_ext


parser = argparse.ArgumentParser(description="Train Swin Cross Attention 4w")
parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
parser.add_argument('--dataset', type=str, default='l8biome', choices=['l8biome', 'cloudsen12'], help='Dataset to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

seed = 42
seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = os.path.join("src", "results")

if args.dataset == 'l8biome':
    selected_bands = list(range(1, 12))
    train_ds, val_ds, test_ds = l8_biome_dataloader.get_l8biome_datasets(
        selected_bands, 512, 512, split_ratio=(0.6, 0.2, 0.2)
    )
    model_name = "checkpoints/swin_cross_attn_4w_segmenter_l8biome_miou5830.pth"
    ignore_index = 255

elif args.dataset == 'cloudsen12':
    selected_bands = list(range(1, 14))
    train_ds, val_ds, test_ds = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
        selected_bands, split_ratio=(0.85, 0.05, 0.1)
    )
    model_name = "checkpoints/swin_cross_attn_4w_segmenter_cloudsen12_miou7549.pth"
    ignore_index = None

g = torch.Generator().manual_seed(seed)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)

model_path = os.path.join(results_dir, model_name)
evaluate_ext(model, model_path, test_loader, device, ignore_index=ignore_index, description=f"{args.dataset} evaluation")
