import torch
import torch.nn as nn
from fastkan import FastKAN as KAN
import torch.nn.functional as F
from model.deep_crossattn import DeepCrossAttention
from model.psp import PSPModule
from model.aspp import ASPPModule
from model.combined_attn import CombinedAttention
    
class Cnn_CrossAttn(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.aspp = ASPPModule(512, 256)
        self.psp = PSPModule(512, [1, 2, 3, 6], 256)
        self.deep_attn = DeepCrossAttention(in_dim=512, context_dim=256, heads=4, depth=2)
        self.cbam = CombinedAttention(512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),  # Reduce channels
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Process
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),  # Restore channels
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.aux1 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.aux2 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.final_out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x_aspp = self.aspp(x)
        x_psp = self.psp(x)
        x = torch.cat([x_aspp, x_psp], dim=1)
        x = self.deep_attn(x, context=x_psp)
        
        x = self.cbam(x)
        x = self.bottleneck(x)

        x = self.dec1(x)
        aux_out1 = self.aux1(F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False))

        x = self.dec2(x)
        aux_out2 = self.aux2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

        x = self.dec3(x)
        x = self.dec4(x)
        out = self.final_out(x)

        return out, aux_out1, aux_out2