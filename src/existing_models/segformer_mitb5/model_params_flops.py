import torch
from mmengine.config import Config
from mmseg.models import build_segmentor
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

class ForwardHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.decode_head = model.decode_head

    def forward(self, x):
        feats = self.backbone(x)
        out = self.decode_head(feats)
        return out

cfg_b5 = Config(dict(
    model=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 18, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            norm_cfg=dict(type='LN', requires_grad=True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
            )
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=512,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
))

model = build_segmentor(cfg_b5.model)
model.init_weights()
wrapped_model = ForwardHead(model).eval()
dummy_input = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    try:
        flops = FlopCountAnalysis(wrapped_model, dummy_input)
        param_str = parameter_count_table(wrapped_model)
        print(flop_count_table(flops, max_depth=5))
    except Exception as e:
        print("FLOP counting failed:", e)
