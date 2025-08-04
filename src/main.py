import os
import argparse
import torch
from torch.utils.data import DataLoader
from data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader, l8_biome_dataloader
from model.cnn_crossattn import Cnn_CrossAttn
from model.swin_crossattn import Swin_CrossAttn
from model.swin_crossattn_4w import Swin_CrossAttn_4W
from utils.trainer_tester import train_one_epoch, evaluate_val, evaluate_test
from utils.helpers import seed_everything, seed_worker, evaluate_and_log


def main(epochs, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    seed = 42
    seed_everything(seed)

    band_sets = {
        #"Bands_All_1_to_11": list(range(1, 12))
        #"RGCirrus" : [3, 4, 10]
        "Bands_All_1_to_13": list(range(1, 14))
    }

    results_dir = os.path.join("src", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "Cross_Attn_Segmenter.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_path, "a") as log_file:
        for name, selected_bands in band_sets.items():
            # train_ds, val_ds, test_ds = l8_biome_dataloader.get_l8biome_datasets(selected_bands, 512, 512, split_ratio=(0.6, 0.2, 0.2))
            train_ds, val_ds, test_ds = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(selected_bands, split_ratio=(0.85, 0.05, 0.1))
        
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

            model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            best_val_loss = float('inf')
            best_model_path = os.path.join(results_dir, "Swin_Cross_Attn_4w_Segmenter_Cloudsen12_best.pth")
            last_model_path = os.path.join(results_dir, "Swin_Cross_Attn_4w_Segmenter_Cloudsen12_last.pth")

            for epoch in range(epochs):
                # train_loss = train_one_epoch(model, train_loader, optimizer, device, ignore_index=255)
                # val_loss = evaluate_val(model, val_loader, device, ignore_index=255)
                train_loss = train_one_epoch(model, train_loader, optimizer, device)
                val_loss = evaluate_val(model, val_loader, device)
                
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    print("Saved best model!")

            torch.save(model.state_dict(), last_model_path)


            evaluate_and_log(model, best_model_path, test_loader, device, log_file, "Best Swin Cross Attn 4w")
            evaluate_and_log(model, last_model_path, test_loader, device, log_file, "Last Swin Cross Attn 4w")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Swin Cross Attention 4w")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index (e.g., 0, 1, 2, 3)')
    args = parser.parse_args()

    main(args.epochs, args.gpu)
