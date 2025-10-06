#!/usr/bin/env python3
"""
Batch denoise ESPI frames with a DnCNN-Lite + ECA model.
- Tile-based inference with Hann blending
- Safe padding: reflect when valid, fallback to replicate
- Inference-only: model.eval() + torch.no_grad()
- Saves 16-bit PNG if input looks like 16-bit, otherwise 8-bit

Usage (CMD):
  python batch_denoise_pytorch_v3_REFLECTFIX_FULL.py ^
    --ckpt  "C:\path\to\best_finetune_MODEL.pth" ^
    --input "C:\path\to\INPUT_DIR" ^
    --output "C:\path\to\OUTPUT_DIR" ^
    --tile 256 --overlap 64 --device cpu --residual-weight 1.0
"""
import argparse, math, os, sys, glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utils
# ------------------------------
def load_png_float01(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
        bit16 = True
    else:
        # assume 8-bit or float-like
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
        bit16 = False
    # ensure shape HxW, grayscale
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr, bit16

def imsave_uint01(arr01, out_path, prefer_16bit=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(arr01, 0.0, 1.0)
    if prefer_16bit:
        x16 = (x * 65535.0 + 0.5).astype(np.uint16)
        Image.fromarray(x16, mode="I;16").save(out_path)
    else:
        x8 = (x * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(x8, mode="L").save(out_path)

def percentile_norm(x, p=99.5):
    if p <= 0:
        return x
    val = np.percentile(x, p)
    if val > 1e-8:
        return np.clip(x / val, 0.0, 1.0)
    return x

def prefilter_np(x, kind="none"):
    if kind == "median3":
        import cv2
        return cv2.medianBlur((x*65535).astype(np.uint16), 3).astype(np.float32)/65535.0
    return x

# ------------------------------
# Model: DnCNN-Lite + ECA
# ------------------------------
class ECALayer(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)                      # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)  # B,1,C
        y = self.conv(y)                     # B,1,C
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        return x * y

class DnCNNLiteECA(nn.Module):
    def __init__(self, in_ch=1, features=64, depth=17, eca_every=4, eca_ks=3, residual=True):
        super().__init__()
        self.residual = residual
        body = []
        body.append(nn.Conv2d(in_ch, features, 3, 1, 1, bias=True))
        body.append(nn.ReLU(inplace=True))

        for i in range(1, depth-1):
            body.append(nn.Conv2d(features, features, 3, 1, 1, bias=True))
            body.append(nn.ReLU(inplace=True))
            if eca_every > 0 and (i % eca_every) == 0:
                body.append(ECALayer(features, k_size=eca_ks))

        body.append(nn.Conv2d(features, in_ch, 3, 1, 1, bias=True))
        self.net = nn.Sequential(*body)

    def forward(self, x, residual_weight=1.0):
        noise = self.net(x)
        return x - residual_weight * noise

# ------------------------------
# Tiled inference with Hann blending
# ------------------------------
def reflect_pad_patch(x, top, left, tile):
    patch = x[..., top:top+tile, left:left+tile]
    ph = max(0, tile - patch.shape[-2])  # pad bottom
    pw = max(0, tile - patch.shape[-1])  # pad right
    if ph > 0 or pw > 0:
        mode = "reflect"
        # reflect requires pad < dim; if not possible, fallback to replicate
        if (patch.shape[-2] <= 1 or patch.shape[-1] <= 1 or
            ph >= patch.shape[-2] or pw >= patch.shape[-1]):
            mode = "replicate"
        patch = F.pad(patch, (0, pw, 0, ph), mode=mode)
    return patch

@torch.no_grad()
def denoise_tiled(model, x, tile=256, overlap=64, residual_weight=1.0, device="cpu"):
    """
    x: tensor [1,1,H,W] in [0,1]
    """
    model = model.to(device)
    _, _, H, W = x.shape
    x = x.to(device)

    # Hann window for blending
    win1 = torch.hann_window(tile, periodic=False, device=device)
    win2 = torch.hann_window(tile, periodic=False, device=device)
    win2d = torch.ger(win1, win2)  # tile x tile
    win2d = win2d.clamp_min(1e-6)  # avoid zeros at edges
    win2d = win2d.unsqueeze(0).unsqueeze(0)  # 1,1,t,t

    out = torch.zeros_like(x, device=device)
    wsum = torch.zeros_like(x, device=device)

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            patch = reflect_pad_patch(x, top, left, tile)
            den = model(patch, residual_weight=residual_weight)
            # crop back to fit
            t_h = min(tile, H - top)
            t_w = min(tile, W - left)
            den_c = den[..., :t_h, :t_w]
            win_c = win2d[..., :t_h, :t_w]
            out[..., top:top+t_h, left:left+t_w] += den_c * win_c
            wsum[..., top:top+t_h, left:left+t_w] += win_c

    out = out / wsum.clamp_min(1e-6)
    return out

# ------------------------------
# Robust checkpoint load
# ------------------------------
def load_model_from_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    # try several common keys
    sd = None
    for k in ("model", "model_state", "state_dict", "net", "network"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            sd = ck[k]; break
    if sd is None and isinstance(ck, dict):
        # maybe it's already a naked state_dict
        if all(isinstance(v, torch.Tensor) for v in ck.values()):
            sd = ck
    if sd is None:
        raise RuntimeError("Could not find model state_dict in checkpoint. Top-level keys: {}".format(list(ck.keys()) if isinstance(ck, dict) else type(ck)))

    # instantiate model (matches your training setting)
    model = DnCNNLiteECA(in_ch=1, features=64, depth=17, eca_every=4, eca_ks=3, residual=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict: missing:", missing, "unexpected:", unexpected)
    model.eval()
    return model

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pth (expects {'model': state_dict} or compatible)")
    ap.add_argument("--input", required=True, help="Input folder with PNGs")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--residual-weight", type=float, default=1.0)
    ap.add_argument("--percentile-norm", type=float, default=0.0, help="e.g., 99.5; 0 to disable")
    ap.add_argument("--prefilter", type=str, default="none", choices=["none","median3"])
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_ckpt(args.ckpt)
    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[INFO] Using device={device}")

    files = sorted(glob.glob(str(in_dir / "*.png")))
    if not files:
        print(f"[ERR] No PNGs found under {in_dir}")
        sys.exit(1)

    print(f"[INFO] Files: {len(files)} | tile={args.tile} overlap={args.overlap} device={device}")

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for fp in files:
            rel = os.path.basename(fp)
            x_np, is16 = load_png_float01(fp)
            # optional prefilter & percentile norm
            x_np = prefilter_np(x_np, args.prefilter)
            if args.percentile_norm > 0:
                x_np = percentile_norm(x_np, args.percentile_norm)

            x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).float()
            den = denoise_tiled(model, x, tile=args.tile, overlap=args.overlap,
                                residual_weight=args.residual_weight, device=device)
            den_np = den.detach().squeeze().cpu().numpy().astype(np.float32)
            imsave_uint01(den_np, out_dir / rel, prefer_16bit=is16)

    print("[DONE] Saved to", out_dir)

if __name__ == "__main__":
    main()
