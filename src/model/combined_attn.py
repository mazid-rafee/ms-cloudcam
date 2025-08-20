import math
import torch
import torch.nn as nn

class ECABlock(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(in_channels) / gamma) + b))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y

class CombinedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ECABlock(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out + x
