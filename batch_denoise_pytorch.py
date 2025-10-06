#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch denoise PNGs with the trained DnCNN-Lite+ECA model (PyTorch).
- Loads training checkpoint (best.pth or last.pth)
- Rebuilds model from saved args
- Tiled inference with Hann blending
Usage (example):
python batch_denoise_pytorch.py ^
  --ckpt "C:\...\outputs_W01\checkpoints\best.pth" ^
  --input "C:\...\W01_ESPI_90db-Noisy" ^
  --output "C:\...\W01_ESPI_90db-Denoised" ^
  --tile 256 --overlap 32 --device cpu
"""
import argparse, os, numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_uint01(arr: np.ndarray, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    ensure_dir(path.parent)
    Image.fromarray((arr*255.0+0.5).astype(np.uint8), mode="L").save(path)

def hann2d(h: int, w: int, eps: float = 1e-6) -> torch.Tensor:
    wx = torch.hann_window(w).unsqueeze(0); wy = torch.hann_window(h).unsqueeze(1)
    return (wy @ wx) + eps

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3: x = x.unsqueeze(0)
    return x

@torch.no_grad()
def denoise_tiled(model: nn.Module, image: torch.Tensor, tile: int = 256, overlap: int = 32) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    x = _ensure_nchw(image).to(device)  # [1,1,H,W]
    _, _, H, W = x.shape
    out = torch.zeros_like(x); weight = torch.zeros_like(x)
    step = tile - overlap; win = hann2d(tile, tile).to(device).view(1,1,tile,tile)
    for top in range(0, H, step):
        for left in range(0, W, step):
            bottom = min(top + tile, H); right = min(left + tile, W)
            pad_h = tile - (bottom - top); pad_w = tile - (right - left)
            patch = torch.zeros((1,1,tile,tile), device=device)
            patch[:,:,:tile-pad_h,:tile-pad_w] = x[:,:,top:bottom,left:right]
            pred = model(patch) * win
            out[:,:,top:bottom,left:right] += pred[:,:,:tile-pad_h,:tile-pad_w]
            weight[:,:,top:bottom,left:right] += win[:,:,:tile-pad_h,:tile-pad_w]
    return out / (weight + 1e-6)

# ---------------- Model (same as training) ----------------
class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        k = k_size if (k_size % 2 == 1) else (k_size + 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(1,2)
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(1,2).unsqueeze(-1)
        return x * y

class SpatialLiteAttention(nn.Module):
    def __init__(self, k: int = 5):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mxx, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.cat([avg, mxx], dim=1)
        a = self.sigmoid(self.conv(m))
        return x * a

def make_norm(kind: str, ch: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "batch": return nn.BatchNorm2d(ch)
    if kind == "group":
        for g in [8,4,2,1]:
            if ch % g == 0: return nn.GroupNorm(g, ch)
        return nn.GroupNorm(1, ch)
    return nn.Identity()

class DnCNNLiteECA(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=32, depth=17, norm="group",
                 eca_interval=4, eca_k=3, spa_interval=0, spa_k=5, residual_weight=1.0):
        super().__init__()
        c = features
        self.residual_weight = residual_weight
        self.head = nn.Sequential(nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
                                  nn.ReLU(inplace=True))
        blocks, ecas, spas = [], [], []
        for i in range(depth - 2):
            blocks.append(nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                make_norm(norm, c),
                nn.ReLU(inplace=True)
            ))
            ecas.append(ECA(c, eca_k) if ((i+1) % eca_interval == 0) else nn.Identity())
            if spa_interval and ((i+1) % spa_interval == 0):
                spas.append(SpatialLiteAttention(spa_k))
            else:
                spas.append(nn.Identity())
        self.blocks = nn.ModuleList(blocks)
        self.eca = nn.ModuleList(ecas)
        self.spa = nn.ModuleList(spas)
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1, bias=False)
    def forward(self, x):
        idt = x; h = self.head(x)
        for b, ec, sp in zip(self.blocks, self.eca, self.spa):
            h = b(h); h = ec(h); h = sp(h)
        noise = self.tail(h)
        return idt - self.residual_weight * noise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("args", {})
    model = DnCNNLiteECA(
        in_ch=1, out_ch=1,
        features=cfg.get("features", 32),
        depth=cfg.get("depth", 17),
        norm=cfg.get("norm", "group"),
        eca_interval=cfg.get("eca_interval", 4),
        eca_k=3,
        spa_interval=cfg.get("spa_interval", 0),
        spa_k=cfg.get("spa_ks", 5),
        residual_weight=1.0
    )
    model.load_state_dict(ck["model"], strict=False)
    device = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device).eval()

    in_dir = Path(args.input); out_dir = Path(args.output)
    files = sorted(in_dir.rglob("*.png"))
    if not files:
        print("No PNGs found under", in_dir); return
    for f in files:
        arr = imread_uint01(f)
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        den = denoise_tiled(model, x, tile=args.tile, overlap=args.overlap)
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel
        imsave_uint01(den.squeeze(0).squeeze(0).cpu().numpy(), out_path)
        print("saved", out_path)

if __name__ == "__main__":
    main()
