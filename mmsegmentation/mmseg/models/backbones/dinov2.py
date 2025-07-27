from mmengine.model import BaseModule
from mmseg.registry import MODELS
import torch

@MODELS.register_module()
class DINOv2(BaseModule):
    def __init__(self, model_name="dinov2_vitb14_reg"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()

        # To freeze the backbone
        # for p in self.model.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        feats = self.model.get_intermediate_layers(x, n=4)
        outs = []
        for f in feats:
            outs = [f.permute(0, 2, 1).reshape(f.shape[0], -1, int(f.shape[1]**0.5), int(f.shape[1]**0.5)) for f in feats]
        return outs
