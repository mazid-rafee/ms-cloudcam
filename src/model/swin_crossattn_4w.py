import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from model.deep_crossattn import DeepCrossAttention
from model.psp import PSPModule
from model.aspp import ASPPModule
from model.combined_attn import CombinedAttention

class Swin_CrossAttn_4W(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            img_size=512,
            in_chans=in_channels
        )

        self.feat_channels = self.swin.feature_info.channels()
        swin_out = self.feat_channels[-1]

        self.aspp = ASPPModule(swin_out, 384)
        self.psp = PSPModule(self.feat_channels[2], [1, 2, 3, 6], 384)

        self.deep_attn = DeepCrossAttention(in_dim=768, context_dim=384, heads=4, depth=2)
        self.cbam = CombinedAttention(768)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(768, 256, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 1), nn.ReLU()
        )

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2), nn.BatchNorm2d(256), nn.ReLU())
        self.aux1 = nn.Conv2d(256, num_classes, 1)

        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), nn.BatchNorm2d(128), nn.ReLU())
        self.aux2 = nn.Conv2d(128, num_classes, 1)

        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU())
        self.final_out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        feats = self.swin(x)
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        f1, f2, f3, f4 = feats

        x_aspp = self.aspp(f4)
        x_psp = self.psp(f3)
        x_aspp = F.interpolate(x_aspp, size=x_psp.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_aspp, x_psp], dim=1)
        x = self.deep_attn(x_cat, context=x_psp)
        x = self.cbam(x)
        x = self.bottleneck(x)

        x = self.dec1(x)
        aux_out1 = self.aux1(F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False))

        x = self.dec2(x)
        aux_out2 = self.aux2(F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False))

        x = self.dec3(x)
        x = self.dec4(x)
        out = self.final_out(F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False))

        return out, aux_out1, aux_out2
