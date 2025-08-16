import os
import sys
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.swin_crossattn_4w import Swin_CrossAttn_4W

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

selected_bands = list(range(1, 14))
model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model.eval()

dummy_input = torch.randn(1, len(selected_bands), 512, 512).to(device)

with torch.no_grad():
    try:
        flops = FlopCountAnalysis(model, dummy_input)
        print("FLOP Count:\n", flop_count_table(flops, max_depth=5))
    except Exception as e:
        print("FLOP counting failed:", e)
