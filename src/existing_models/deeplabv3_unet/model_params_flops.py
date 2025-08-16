import torch
import segmentation_models_pytorch as smp
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
import os
from contextlib import redirect_stdout

NUM_CLASSES = 4
BANDS = [1,2,3,4,5,6,7,8,9,10,11,12,13]
ENCODERS = [
    {"name": "resnet101", "weights": "imagenet"},
    {"name": "mobilenet_v2", "weights": "imagenet"},
]
ARCHS = ["DeepLabV3Plus", "UnetPlusPlus", "Unet"]
IMG_SIZE = (512, 512)
LOG_FILE = "model_stats.txt"

def get_model(arch_name, encoder_name, encoder_weights, in_channels, classes):
    if arch_name == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif arch_name == "UnetPlusPlus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif arch_name == "Unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

def model_stats_print(model, in_channels, img_size=(512, 512)):
    model.eval()
    dummy = torch.randn(1, in_channels, *img_size)
    try:
        flops = FlopCountAnalysis(model, dummy)
        print(f"FLOPs (1×{in_channels}×{img_size[0]}×{img_size[1]}): {flops.total()/1e9:.3f} GFLOPs")
        print(flop_count_table(flops, max_depth=2))
    except Exception as e:
        print("FLOP counting failed:", e)

if __name__ == "__main__":
    os.makedirs(".", exist_ok=True)
    with open(LOG_FILE, "w") as f:
        with redirect_stdout(f):  # write to file
            for enc in ENCODERS:
                for arch in ARCHS:
                    print(f"\n=== {arch} | Encoder: {enc['name']} | Bands: {BANDS} ===")
                    model = get_model(arch, enc["name"], enc["weights"], in_channels=len(BANDS), classes=NUM_CLASSES)
                    model_stats_print(model, in_channels=len(BANDS), img_size=IMG_SIZE)

    # Also print to console
    with open(LOG_FILE, "r") as f:
        print(f.read())
