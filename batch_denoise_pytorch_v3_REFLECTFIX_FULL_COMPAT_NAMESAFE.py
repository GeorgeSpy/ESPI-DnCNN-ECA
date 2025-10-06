#!/usr/bin/env python3
"""
Batch denoise ESPI frames with a DnCNN-Lite (+BN/ECA compatible) model.
- Builds layers that match checkpoint names EXACTLY (entry, mid.<index>, exit) using ModuleDict
- Supports Conv2d, BatchNorm2d, ECALayer; inserts no-param ReLU implicitly after Conv during forward
- Tile-based inference with Hann blending
- Safe padding: reflect when valid, fallback to replicate
- Inference-only: model.eval() + torch.no_grad()
- Saves 16-bit PNG if input looks like 16-bit, otherwise 8-bit
"""
import argparse, os, sys, glob, re
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
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
        bit16 = False
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
# Modules
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

class CompatNet(nn.Module):
    """
    Build a network that mirrors checkpoint keys exactly:
      - self.entry : Conv2d
      - self.mid   : ModuleDict with keys '0','1','2',... as in checkpoint
                     each value is one of: Conv2d / BatchNorm2d / ECALayer
      - self.exit  : Conv2d
    During forward we also apply ReLU **after Conv2d** implicitly, since ReLU has no params
    and is not present in the state_dict.
    """
    def __init__(self, sd):
        super().__init__()
        # infer in/out channels from entry
        if "entry.weight" not in sd:
            raise RuntimeError("Checkpoint missing 'entry.weight'")
        in_ch = sd["entry.weight"].shape[1]
        feat  = sd["entry.weight"].shape[0]
        self.entry = nn.Conv2d(in_ch, feat, 3, 1, 1, bias=True)

        # collect mid indices from keys like 'mid.<i>...'
        idxs = set()
        for k in sd.keys():
            m = re.match(r"mid\.(\d+)", k)
            if m:
                idxs.add(int(m.group(1)))
        self.mid_order = sorted(list(idxs))
        self.mid = nn.ModuleDict()
        for i in self.mid_order:
            # decide module type by inspecting tensors
            eca_key = f"mid.{i}.conv.weight"            # Conv1d from ECA
            w_key   = f"mid.{i}.weight"                 # could be Conv2d or BN
            mean_key= f"mid.{i}.running_mean"           # BN stats exist only for BN

            if eca_key in sd:
                # kernel length from weight shape [1,1,k]
                k = sd[eca_key].shape[-1] if hasattr(sd[eca_key], "shape") and sd[eca_key].ndim==3 else 3
                self.mid[str(i)] = ECALayer(feat, k_size=int(k))
            elif w_key in sd and hasattr(sd[w_key], "ndim") and sd[w_key].ndim == 4:
                Cout, Cin, kh, kw = sd[w_key].shape
                pad = kh // 2
                self.mid[str(i)] = nn.Conv2d(feat, feat, kh, 1, pad, bias=True)
            elif w_key in sd and mean_key in sd and hasattr(sd[w_key], "ndim") and sd[w_key].ndim == 1:
                self.mid[str(i)] = nn.BatchNorm2d(feat)
            else:
                # unknown/no-param layer in ckpt; insert identity
                self.mid[str(i)] = nn.Identity()

        # exit conv
        if "exit.weight" in sd:
            out_ch = sd["exit.weight"].shape[0]  # should be in_ch
        else:
            out_ch = in_ch
        self.exit = nn.Conv2d(feat, out_ch, 3, 1, 1, bias=True)

    def forward(self, x, residual_weight=1.0):
        y = self.entry(x)
        for i in self.mid_order:
            m = self.mid[str(i)]
            if isinstance(m, nn.Conv2d):
                y = m(y)
                y = F.relu(y, inplace=True)  # implicit ReLU after Conv2d
            else:
                y = m(y)
        y = self.exit(y)
        return x - residual_weight * y

# ------------------------------
# Tiled inference with Hann blending
# ------------------------------
def reflect_pad_patch(x, top, left, tile):
    patch = x[..., top:top+tile, left:left+tile]
    ph = max(0, tile - patch.shape[-2])  # pad bottom
    pw = max(0, tile - patch.shape[-1])  # pad right
    if ph > 0 or pw > 0:
        mode = "reflect"
        if (patch.shape[-2] <= 1 or patch.shape[-1] <= 1 or
            ph >= patch.shape[-2] or pw >= patch.shape[-1]):
            mode = "replicate"
        patch = F.pad(patch, (0, pw, 0, ph), mode=mode)
    return patch

@torch.no_grad()
def denoise_tiled(model, x, tile=256, overlap=64, residual_weight=1.0, device="cpu"):
    model = model.to(device)
    _, _, H, W = x.shape
    x = x.to(device)
    win1 = torch.hann_window(tile, periodic=False, device=device)
    win2 = torch.hann_window(tile, periodic=False, device=device)
    win2d = torch.ger(win1, win2).clamp_min(1e-6).unsqueeze(0).unsqueeze(0)
    out = torch.zeros_like(x, device=device)
    wsum = torch.zeros_like(x, device=device)
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            patch = reflect_pad_patch(x, top, left, tile)
            den = model(patch, residual_weight=residual_weight)
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
    sd = None
    for k in ("model", "model_state", "state_dict", "net", "network"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            sd = ck[k]; break
    if sd is None and isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        sd = ck
    if sd is None:
        raise RuntimeError("Could not find model state_dict in checkpoint.")

    model = CompatNet(sd)
    # now load by exact names; shapes will match
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[WARN] compat load_state_dict: missing:", missing, "unexpected:", unexpected)
    model.eval()
    return model

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pth")
    ap.add_argument("--input", required=True, help="Input folder with PNGs")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--residual-weight", type=float, default=1.0)
    ap.add_argument("--percentile-norm", type=float, default=0.0, help="e.g., 99.5; 0 to disable")
    ap.add_argument("--prefilter", type=str, default="none", choices=["none","median3"])
    args = ap.parse_args()

    in_dir = Path(args.input); out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_ckpt(args.ckpt)
    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[INFO] Using device={device}")

    files = sorted(glob.glob(str(in_dir / "*.png")))
    if not files:
        print(f"[ERR] No PNGs found under {in_dir}"); sys.exit(1)

    print(f"[INFO] Files: {len(files)} | tile={args.tile} overlap={args.overlap} device={device}")

    with torch.no_grad():
        for fp in files:
            rel = os.path.basename(fp)
            x_np, is16 = load_png_float01(fp)
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
