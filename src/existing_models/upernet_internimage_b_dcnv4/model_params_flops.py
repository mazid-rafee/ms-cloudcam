import os
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
data_loaders_path = os.path.join(parent_dir, 'data_loaders')

sys.path.append(data_loaders_path)

sys.path.append(os.path.abspath("DCNv4/segmentation"))
from mmseg_custom.models.backbones import FlashInternImage



os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 3
dummy_input = torch.randn(1, in_channels, 512, 512).to(device)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        meta = [dict(img_shape=(512, 512), ori_shape=(512, 512), pad_shape=(512, 512), batch_input_shape=(512, 512), scale_factor=1.0, flip=False)]
        return self.model.encode_decode(x, meta)

cfg = Config.fromfile("DCNv4/segmentation/configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py")
cfg.model.pretrained = None
cfg.model.backbone.in_channels = in_channels
cfg.model.decode_head.num_classes = 4
if "auxiliary_head" in cfg.model:
    cfg.model.auxiliary_head.num_classes = 4
cfg.model.test_cfg = dict(mode='whole')
cfg.norm_cfg = dict(type='BN', requires_grad=True)
if hasattr(cfg.model.backbone, 'norm_cfg'):
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
if hasattr(cfg.model.decode_head, 'norm_cfg'):
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
if 'auxiliary_head' in cfg.model and hasattr(cfg.model.auxiliary_head, 'norm_cfg'):
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)
wrapped_model = Wrapper(model).eval().to(device)

with torch.no_grad():
    try:
        flops = FlopCountAnalysis(wrapped_model, dummy_input)
        print(parameter_count_table(model))
        # print(flop_count_table(flops, max_depth=5)) #Gets stuck for unknown reason
    except Exception as e:
        print("FLOP counting failed:", e)
