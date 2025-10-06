# -*- coding: utf-8 -*-
"""
Batch denoise for ESPI (compat ckpt) with robust percentile normalization
and residual->clean composition. Safe reflect tiling for any tile/overlap.
Saves both linear-output and an extra _viz stretch for γρήγορο έλεγχο.
"""

import argparse, os, math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------- Model (Compat) ----------------
class ConvBNReLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(c)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

class CompatNet(nn.Module):
    """
    Συμβατή αρχιτεκτονική τύπου DnCNN-lite (1→64→…→64→1).
    Φορτώνει state_dict με strict=False ώστε να δουλεύει με τα ckpt σου.
    """
    def __init__(self, depth=17, features=64):
        super().__init__()
        self.entry = nn.Conv2d(1, features, 3, 1, 1)
        blocks = []
        for _ in range(depth-2):
            blocks.append(ConvBNReLU(features))
        self.mid = nn.Sequential(*blocks)
        self.exit = nn.Conv2d(features, 1, 3, 1, 1)

    def forward(self, x):
        x = self.entry(x)
        x = self.mid(x)
        x = self.exit(x)
        return x

def load_model_from_ckpt(ckpt_path, device):
    sd_file = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Δοκίμασε να εντοπίσεις state_dict σε κοινά κλειδιά:
    if isinstance(sd_file, dict):
        if "model" in sd_file and isinstance(sd_file["model"], dict):
            sd = sd_file["model"]
        elif "model_state" in sd_file and isinstance(sd_file["model_state"], dict):
            sd = sd_file["model_state"]
        else:
            # ίσως είναι ήδη καθαρό state_dict
            sd = sd_file
    else:
        sd = sd_file

    model = CompatNet(depth=17, features=64).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

# ---------------- Utils ----------------
def imread_gray(path):
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def imsave_uint01(arr01, path):
    arr = np.clip(arr01, 0.0, 1.0)
    Image.fromarray((arr*255.0+0.5).astype(np.uint8)).save(path)

def imread_u16aware(path):
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)

def imsave_u16(arr01, path):
    arr = np.clip(arr01, 0.0, 1.0)
    Image.fromarray((arr*65535.0+0.5).astype(np.uint16), mode="I;16").save(path)

def reflect_pad_patch(x, top, left, tile):
    _, _, H, W = x.shape
    bottom = min(top+tile, H)
    right  = min(left+tile, W)
    patch = x[:, :, top:bottom, left:right]
    ph = max(0, tile - (bottom - top))
    pw = max(0, tile - (right  - left))
    if ph > 0 or pw > 0:
        # reflect pad αλλά ποτέ μεγαλύτερο από το μέγεθος του patch
        ph = min(ph, patch.shape[2]-1) if patch.shape[2] > 1 else 0
        pw = min(pw, patch.shape[3]-1) if patch.shape[3] > 1 else 0
        patch = F.pad(patch, (0, pw, 0, ph), mode="reflect")
    return patch

def denoise_tiled(model, x, tile, overlap, device):
    # x: [1,1,H,W] normalized
    _, _, H, W = x.shape
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")

    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    if ys[-1] != max(0, H - tile): ys.append(max(0, H - tile))
    if xs[-1] != max(0, W - tile): xs.append(max(0, W - tile))

    with torch.no_grad():
        for top in ys:
            for left in xs:
                patch = reflect_pad_patch(x, top, left, tile)
                den   = model(patch.to(device))
                h = min(tile, H-top)
                w = min(tile, W-left)
                out[:, :, top:top+h, left:left+w] += den[:, :, :h, :w].cpu()
                weight[:, :, top:top+h, left:left+w] += 1.0
    out = out / torch.clamp_min(weight, 1e-6)
    return out

def robust_norm(x, p_lo, p_hi):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    scale = max(hi - lo, 1e-6)
    x_n = np.clip((x - lo) / scale, 0.0, 1.0)
    return x_n, lo, scale

def invert_norm(y_n, lo, scale):
    return y_n * scale + lo

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile low for normalization")
    ap.add_argument("--p_hi", type=float, default=99.5, help="percentile high for normalization")
    ap.add_argument("--predicts-residual", action="store_true", help="if set, model output is residual")
    ap.add_argument("--viz-percentile", type=float, default=99.7, help="extra stretched preview")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--norm-mode", choices=["percentile","u16"], default="percentile", help="Input normalization: percentile per frame or u16 passthrough")
    ap.add_argument("--save-u16", action="store_true", help="Save linear outputs as 16-bit PNGs")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model_from_ckpt(args.ckpt, device)

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir.parent / (out_dir.name + "_viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.png")])
    if args.limit > 0:
        files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={device}")

    for p in tqdm(files):
        if args.norm_mode == "u16":
            x = imread_u16aware(p)       # [H,W] in 0..1 from 16-bit aware loader
            x_n = x
            lo, sc = 0.0, 1.0
        else:
            x = imread_gray(p)           # [H,W] in 0..1
            x_n, lo, sc = robust_norm(x, args.p_lo, args.p_hi)

        # to tensor
        xt = torch.from_numpy(x_n).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # run tiled
        den = denoise_tiled(model, xt, tile=args.tile, overlap=args.overlap, device=device)  # normalized domain

        if args.predicts_residual:
            clean_n = torch.clamp(xt - den, 0.0, 1.0)
        else:
            clean_n = torch.clamp(den, 0.0, 1.0)

        clean = clean_n.squeeze().cpu().numpy()
        if args.norm_mode == "u16":
            clean_lin = clean
        else:
            clean_lin = invert_norm(clean, lo, sc)           # back to raw domain [≈0..1]
        if args.save_u16:
            imsave_u16(clean_lin, out_dir / p.name)
        else:
            imsave_uint01(clean_lin, out_dir / p.name)

        # extra preview (contrast stretch για να φαίνεται)
        vhi = np.percentile(clean_lin, args.viz_percentile)
        viz = np.clip(clean_lin / max(vhi, 1e-6), 0.0, 1.0)
        imsave_uint01(viz, viz_dir / p.name)

    print(f"[DONE] Saved linear outputs to: {out_dir}")
    print(f"[DONE] Saved previews (stretched) to: {viz_dir}")

if __name__ == "__main__":
    main()
