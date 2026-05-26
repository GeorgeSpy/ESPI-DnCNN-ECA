import torch
import torch.nn as nn
import time
import math

# --- Architectures (same as stress test) ---
def eca_kernel_for(C, gamma=2.0, b=1.0):
    return max(3, int(round(math.log2(max(C, 1)) / gamma + b / gamma)) | 1)

class ECA_v4(nn.Module):
    def __init__(self, ch, k_size=0, temp=0.75, gain=0.5, centered=True):
        super().__init__()
        k = k_size if (k_size and k_size%2==1) else eca_kernel_for(ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,k,padding=k//2,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.temp=temp; self.gain=gain; self.centered=centered
    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(1,2)
        y = self.conv(y)
        g = self.sigmoid(y / max(self.temp,1e-6)).transpose(1,2).unsqueeze(-1)
        return x * ((1.0 + self.gain*(g-0.5)*2.0) if self.centered else g)

class ECA_v5(nn.Module):
    def __init__(self, ch, k_size=0, temp=1.0, gain=0.25, centered=True, use_maxpool=True, learnable=True):
        super().__init__()
        k = k_size if (k_size and k_size%2==1) else eca_kernel_for(ch)
        self.gap, self.gmp = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.use_maxpool, self.centered = use_maxpool, centered
        self.conv_avg = nn.Conv1d(1,1,k,padding=k//2,bias=False)
        self.conv_max = nn.Conv1d(1,1,k,padding=k//2,bias=False) if use_maxpool else None
        self.sigmoid = nn.Sigmoid()
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.raw_gain = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        y = self.conv_avg(self.gap(x).squeeze(-1).transpose(1,2))
        if self.use_maxpool: y = 0.5*(y + self.conv_max(self.gmp(x).squeeze(-1).transpose(1,2)))
        g = self.sigmoid(y / 0.75).transpose(1,2).unsqueeze(-1)
        return x * ((1.0 + 0.5*(g-0.5)*2.0) if self.centered else g)

class DnCNNLite(nn.Module):
    def __init__(self, version="v4", use_eca=False, eca_positions=None):
        super().__init__()
        c = 32
        self.head = nn.Sequential(nn.Conv2d(1,c,3,1,1,bias=False), nn.ReLU(True))
        blocks, ecas = [], []
        pos = set(eca_positions or [])
        for i in range(15):
            blocks.append(nn.Sequential(nn.Conv2d(c,c,3,1,1,bias=False), nn.GroupNorm(8,c), nn.ReLU(True)))
            if use_eca and i in pos:
                ecas.append(ECA_v5(c) if version=="v5" else ECA_v4(c))
            else: ecas.append(nn.Identity())
        self.blocks, self.eca = nn.ModuleList(blocks), nn.ModuleList(ecas)
        self.tail = nn.Conv2d(c,1,3,1,1,bias=False)
    def forward(self, x):
        h = self.head(x)
        for b, e in zip(self.blocks, self.eca): h = e(b(h))
        return x - self.tail(h)

def benchmark(model, device="cuda"):
    model.to(device).eval()
    x = torch.randn(1, 1, 256, 256).to(device)
    params = sum(p.numel() for p in model.parameters())
    
    # Methodology: Warmup 50, Measure 500
    with torch.no_grad():
        for _ in range(50): _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(500): _ = model(x)
        torch.cuda.synchronize()
        latency = (time.time() - t0) / 500 * 1000 # ms
    return params, latency

device_name = torch.cuda.get_device_name(0)
print(f"Benchmarking on: {device_name}")
print(f"Methodology: Batch=1, 256x256, 50 warmup, 500 measured, cuda_sync, no_grad\n")

models = [
    ("DnCNN Base (Static)", DnCNNLite(use_eca=False)),
    ("DnCNN V4 ECA (3 Pos)", DnCNNLite(version="v4", use_eca=True, eca_positions=[0,1,2])),
    ("DnCNN V5 ECA (7 Pos)", DnCNNLite(version="v5", use_eca=True, eca_positions=[0,1,2,3,6,10,14])),
]

print(f"{'Model':<25} | {'Params':<10} | {'Latency (ms)':<12}")
print("-" * 52)
for name, m in models:
    p, l = benchmark(m)
    print(f"{name:<25} | {p:<10,} | {l:<12.3f}")
