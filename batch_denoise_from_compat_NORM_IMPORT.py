# -*- coding: utf-8 -*-
"""
Batch denoise using a 'compat' model defined in an external python file
(e.g. denoise_debug_dualmode_COMPAT.py), with robust percentile normalization
and residual->clean composition. Safe reflect-tiling and viz previews.
"""

import argparse, runpy, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------- IO utils ----------
def imread_gray(path):
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0

def imsave_uint01(arr01, path):
    arr = np.clip(arr01, 0.0, 1.0)
    Image.fromarray((arr*255.0+0.5).astype(np.uint8)).save(path)

# ---------- normalization ----------
def robust_norm(x, p_lo, p_hi):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    scale = max(hi - lo, 1e-6)
    x_n = np.clip((x - lo) / scale, 0.0, 1.0)
    return x_n, lo, scale

def invert_norm(y_n, lo, scale):
    return y_n * scale + lo

# ---------- tiling ----------
def reflect_pad_patch(x, top, left, tile):
    _, _, H, W = x.shape
    bottom = min(top+tile, H)
    right  = min(left+tile, W)
    patch = x[:, :, top:bottom, left:right]
    ph = max(0, tile - (bottom - top))
    pw = max(0, tile - (right  - left))
    # reflect padding δεν επιτρέπεται >= dim -> περιορισμός
    if ph > 0:
        ph = min(ph, max(patch.shape[2]-1, 0))
    if pw > 0:
        pw = min(pw, max(patch.shape[3]-1, 0))
    if ph > 0 or pw > 0:
        patch = F.pad(patch, (0, pw, 0, ph), mode="reflect")
    return patch

@torch.no_grad()
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
    last_y = max(0, H - tile)
    last_x = max(0, W - tile)
    if ys[-1] != last_y: ys.append(last_y)
    if xs[-1] != last_x: xs.append(last_x)

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

# ---------- compat loader ----------
def build_compat_model(compat_py_path, sd, device):
    """
    Φορτώνει το python αρχείο (runpy) και προσπαθεί με μια από τις παρακάτω επιλογές:
    - build_model_from_state_dict(sd)
    - CompatNet(sd)
    - CompatNet(); model.load_state_dict(sd, strict=False)
    """
    mod = runpy.run_path(str(compat_py_path))
    model = None
    if "build_model_from_state_dict" in mod:
        model = mod["build_model_from_state_dict"](sd).to(device)
    elif "CompatNet" in mod:
        try:
            model = mod["CompatNet"](sd).to(device)
        except TypeError:
            model = mod["CompatNet"]().to(device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[compat] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        raise RuntimeError("Compat file does not expose CompatNet or build_model_from_state_dict.")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--compat", required=True, help="Path to denoise_debug_dualmode_COMPAT.py")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.5)
    ap.add_argument("--predicts-residual", action="store_true")
    ap.add_argument("--viz-percentile", type=float, default=99.7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    sd_file = torch.load(args.ckpt, map_location="cpu")
    # βρες σωστό state_dict
    if isinstance(sd_file, dict) and "model" in sd_file and isinstance(sd_file["model"], dict):
        sd = sd_file["model"]
    elif isinstance(sd_file, dict) and "model_state" in sd_file and isinstance(sd_file["model_state"], dict):
        sd = sd_file["model_state"]
    else:
        sd = sd_file

    model = build_compat_model(Path(args.compat), sd, device)

    in_dir  = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir.parent / (out_dir.name + "_viz"); viz_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.png")])
    if args.limit > 0:
        files = files[:args.limit]
    print(f"[INFO] files={len(files)} tile={args.tile} overlap={args.overlap} device={device}")

    for p in tqdm(files):
        x = imread_gray(p)
        x_n, lo, sc = robust_norm(x, args.p_lo, args.p_hi)
        xt = torch.from_numpy(x_n).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        den = denoise_tiled(model, xt, args.tile, args.overlap, device)  # normalized space
        if args.predicts_residual:
            clean_n = torch.clamp(xt - den, 0.0, 1.0)
        else:
            clean_n = torch.clamp(den, 0.0, 1.0)

        clean = clean_n.squeeze().cpu().numpy()
        clean_lin = invert_norm(clean, lo, sc)
        imsave_uint01(clean_lin, out_dir / p.name)

        vhi = np.percentile(clean_lin, args.viz_percentile)
        viz = np.clip(clean_lin / max(vhi, 1e-6), 0.0, 1.0)
        imsave_uint01(viz, viz_dir / p.name)

    print(f"[DONE] Saved linear outputs to: {out_dir}")
    print(f"[DONE] Saved previews (stretched) to: {viz_dir}")

if __name__ == "__main__":
    main()
