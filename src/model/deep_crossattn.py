import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_CrossAttention_Module(nn.Module):
    def __init__(self, in_dim, context_dim=None, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (in_dim // heads) ** 0.5

        context_dim = context_dim or in_dim

        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(context_dim, in_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(context_dim, in_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    def forward(self, x, context):
        B, C, H, W = x.shape

        # Linear projections
        Q = self.q_proj(x)   # [B, C, H, W]
        K = self.k_proj(context)
        V = self.v_proj(context)

        # Flatten spatial dims
        Q = Q.view(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)  # [B, heads, HW, C//heads]
        K = K.view(B, self.heads, C // self.heads, -1)                      # [B, heads, C//heads, HW]
        V = V.view(B, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)  # [B, heads, HW, C//heads]

        # Attention
        attn = torch.matmul(Q, K) / self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # [B, heads, HW, C//heads]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        return self.out_proj(out) + x  # Residual

class DeepCrossAttention(nn.Module):
    def __init__(self, in_dim, context_dim=None, heads=4, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            CNN_CrossAttention_Module(in_dim, context_dim, heads)
            for _ in range(depth)
        ])

    def forward(self, x, context):
        for block in self.blocks:
            x = block(x, context)
        return x