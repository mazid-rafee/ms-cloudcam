import os
import argparse
import torch
from torch.utils.data import DataLoader
from data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader, l8_biome_dataloader
from model.swin_crossattn_4w import Swin_CrossAttn_4W
from utils.trainer_tester import train_one_epoch, evaluate_val, evaluate_test, evaluate_val_iou
from utils.helpers import seed_everything, seed_worker, evaluate_and_log

def main(epochs, gpu_id, dataset_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    seed = 42
    seed_everything(seed)

    results_dir = os.path.join("src", "results")
    os.makedirs(results_dir, exist_ok=True)
    

    model_base_name = f"leanattention_with_deepcross_attn_ms_cloudcam_{dataset_name.lower()}"
    log_path = os.path.join(results_dir, f"Cross_Attn_Segmenter.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_path, "a") as log_file:
        if dataset_name.lower() == 'cloudsen12_l1c':
            selected_bands = list(range(1, 14))
            train_ds, val_ds, test_ds = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
                selected_bands, split_ratio=(0.85, 0.05, 0.1)
            )
            ignore_index = None

        elif dataset_name.lower() == 'cloudsen12_l2a':
            selected_bands = list(range(1, 14))
            train_ds, val_ds, test_ds = cloudsen12_l2a_dataloader.get_cloudsen12_datasets(
                selected_bands, split_ratio=(0.85, 0.05, 0.1)
            )
            ignore_index = None

        elif dataset_name.lower() == 'l8biome':
            selected_bands = list(range(1, 12))
            train_ds, val_ds, test_ds = l8_biome_dataloader.get_l8biome_datasets(
                selected_bands, 512, 512, split_ratio=(0.6, 0.2, 0.2)
            )
            ignore_index = 255

        else:
            raise ValueError("Invalid dataset name. Use 'cloudsen12_l1c', 'cloudsen12_l2a', or 'l8biome'.")


        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

        model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_iou = 0.0
        best_test_iou = 0.0
        best_train_loss = float('inf')

        best_model_path_val = os.path.join(results_dir, f"{model_base_name}_best_val.pth")
        best_model_path_train = os.path.join(results_dir, f"{model_base_name}_best_train.pth")
        best_model_path_test = os.path.join(results_dir, f"{model_base_name}_best_test.pth")
        last_model_path_train = os.path.join(results_dir, f"{model_base_name}_last_train.pth")

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, ignore_index=ignore_index)
            val_iou = evaluate_val_iou(model, val_loader, device, desc="Val Validation")
            test_iou = evaluate_val_iou(model, test_loader, device, desc="Test Validation")

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val IOU = {val_iou:.4f}, Test IOU = {test_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), best_model_path_val)
                print("Saved best val model!")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), best_model_path_train)
                print("Saved best train model!")

            if test_iou > best_test_iou:
                best_test_iou = test_iou
                torch.save(model.state_dict(), best_model_path_test)
                print("Saved best test model!")

        torch.save(model.state_dict(), last_model_path_train)

        evaluate_and_log(model, best_model_path_val, test_loader, device, log_file, f"{model_base_name}_best_val")
        evaluate_and_log(model, best_model_path_train, test_loader, device, log_file, f"{model_base_name}_best_train")
        evaluate_and_log(model, best_model_path_test, test_loader, device, log_file, f"{model_base_name}_best_test")
        evaluate_and_log(model, last_model_path_train, test_loader, device, log_file, f"{model_base_name}_last_train")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Swin Cross Attention 4w")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index (e.g., 0, 1, 2, 3)')
    parser.add_argument('--dataset', type=str, required=True, choices=['cloudsen12_l1c', 'cloudsen12_l2a', 'l8biome'], help='Dataset to use: cloudsen12_l1c, cloudsen12_l2a, or l8biome')
    args = parser.parse_args()

    main(args.epochs, args.gpu, args.dataset)
