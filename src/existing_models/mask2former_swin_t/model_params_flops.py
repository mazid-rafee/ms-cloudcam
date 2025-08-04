

import os
import torch
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dummy_input = torch.randn(1, 3, 512, 512).to(device)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        meta = [dict(img_shape=(512, 512), ori_shape=(512, 512))]
        return self.model.encode_decode(x, meta)

cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py")
cfg.model.data_preprocessor = None

model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)

wrapped_model = Wrapper(model).eval().to(device)

with torch.no_grad():
    try:
        flops = FlopCountAnalysis(wrapped_model, dummy_input)
        param_str = parameter_count_table(model)
        print(flop_count_table(flops, max_depth=5))

    except Exception as e:
        print("FLOP counting failed:", e)