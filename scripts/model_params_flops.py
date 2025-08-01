import torch
import sys
import os

from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.CNN_KAN_Segmenter import CNN_KAN_Segmenter

if __name__ == '__main__':
    in_channels = 13
    num_classes = 4
    input_size = (in_channels, 512, 512)

    model = CNN_KAN_Segmenter(in_channels=in_channels, num_classes=num_classes)
    model.eval()

    print("[INFO] Starting FLOPs and parameter count using ptflops...")

    macs, params = get_model_complexity_info(
        model,
        input_size,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=False
    )

    print(f"\n[ptflops]")
    print(f"MACs (1 MAC = 2 FLOPs): {macs}")
    print(f"Parameters: {params}")

    print("\n[INFO] Starting FLOPs and parameter count using fvcore...")

    dummy_input = torch.randn(1, *input_size)
    flops = FlopCountAnalysis(model, dummy_input)

    print(f"\n[fvcore]")
    print("FLOPs:", flops.total())  # in raw number of FLOPs
    print("FLOPs (Giga):", flops.total() / 1e9)
    print(parameter_count_table(model))
