
"""
Tile-based denoising for ESPI frames (CPU-safe) with two output modes:
  1) predicts-clean   -> y = net(x) (use when the checkpoint outputs CLEAN)
  2) residual (default) -> y = x - w * net(x) (use when the checkpoint outputs NOISE)

- Safe reflect/replicate padding for edge tiles
- Hann blending to suppress seams
- Preserves 8-bit / 16-bit bit-depth based on input
- Optional percentile normalization & median prefilter

Usage (CLEAN mode, likely your case):
  python batch_denoise_predicts_clean_or_residual.py ^
    --ckpt "...\best_finetune_MODEL.pth" ^
    --input "...\W01_ESPI_90db-PseudoNoisy_MATCH_v2" ^
    --output "...\W01_ESPI_90db-Denoised_MATCH_v2_CLEANMODE" ^
    --tile 256 --overlap 64 --device cpu --predicts-clean

Residual mode (if your checkpoint outputs NOISE):
  python batch_denoise_predicts_clean_or_residual.py ^
    --ckpt "...\best.pth" --input "...Noisy" --output "...Denoised" ^
    --tile 256 --overlap 64 --device cpu --residual-weight 1.0

Author: ChatGPT helper
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model definition (compat)
# -----------------------------
class CompatNet(nn.Module):
    """
    Minimal DnCNN-like net that matches checkpoints saved as:
      - entry (Conv2d)
      - mid   (Conv2d, BatchNorm2d, ReLU) * N groups (three layers per group)
      - exit  (Conv2d)

    The number of mid groups is inferred from the state_dict (conv count).
    """
    def __init__(self, groups: int, in_ch=1, feat=64, out_ch=1):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, feat, 3, padding=1, bias=True)

        layers = []
        for g in range(groups):
            layers += [
                nn.Conv2d(feat, feat, 3, padding=1, bias=True),
                nn.BatchNorm2d(feat),
                nn.ReLU(inplace=True),
            ]
        self.mid = nn.Sequential(*layers)
        self.exit = nn.Conv2d(feat, out_ch, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.entry(x)
        x = self.mid(x)
        x = self.exit(x)
        return x


def load_state_dict_any(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model", "model_state", "state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if isinstance(obj, dict):
        # Might already be a state_dict
        # Heuristic: at least one tensor value
        has_tensor = any(torch.is_tensor(v) for v in obj.values())
        if has_tensor:
            return obj
    raise RuntimeError("Could not extract a state_dict from the checkpoint.")


def build_model_from_state_dict(sd: dict) -> nn.Module:
    # Infer number of mid convs by counting keys like "mid.X.weight" with 4D tensors
    conv_keys = [k for k, v in sd.items()
                 if k.startswith("mid.") and k.endswith(".weight") and torch.is_tensor(v) and v.ndim == 4]
    # Three layers per group (conv, bn, relu) -> groups == number of convs in "mid"
    groups = len(conv_keys)
    if groups < 1:
        # Fallback: guess 15 groups
        groups = 15
    model = CompatNet(groups=groups, in_ch=1, feat=64, out_ch=1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing: {list(missing)} unexpected: {list(unexpected)}")
    return model


# -----------------------------
# Utilities
# -----------------------------
def read_image_gray(path: Path):
    im = Image.open(path)
    # Remember original bit-depth
    mode = im.mode
    arr = np.array(im)
    if arr.dtype == np.uint16:
        scale = 65535.0
    else:
        scale = 255.0
        # Convert to uint8 if not already
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    arr_f = arr.astype(np.float32) / scale
    # if the image has multiple channels, reduce to luminance
    if arr_f.ndim == 3:
        arr_f = arr_f.mean(axis=2)
    return arr_f, scale


def save_image_gray(path: Path, img01: np.ndarray, scale: float):
    img01 = np.clip(img01, 0.0, 1.0)
    if scale > 255:
        arr = (img01 * 65535.0 + 0.5).astype(np.uint16)
    else:
        arr = (img01 * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def hann2d(h, w, device):
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    w2 = torch.ger(wy, wx)  # outer product
    return w2


def safe_pad_reflect_or_replicate(patch, th, tw):
    """Pad to target size. Prefer 'reflect' but fall back to 'replicate' if reflect would fail."""
    _, _, ph, pw = patch.shape
    pad_bottom = max(0, th - ph)
    pad_right = max(0, tw - pw)
    if pad_bottom == 0 and pad_right == 0:
        return patch
    # reflect requires pad < size
    mode = "reflect"
    if pad_bottom >= ph or pad_right >= pw:
        mode = "replicate"
    return F.pad(patch, (0, pad_right, 0, pad_bottom), mode=mode)


def tile_denoise(net, x, tile=256, overlap=64, device="cpu", predicts_clean=False, residual_weight=1.0):
    """
    x: torch tensor [1,1,H,W] in [0,1]
    """
    _, _, H, W = x.shape
    tile = int(tile)
    overlap = int(overlap)
    step = max(1, tile - overlap)

    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x)

    win = hann2d(tile, tile, device=x.device).view(1, 1, tile, tile)

    for top in range(0, H, step):
        for left in range(0, W, step):
            h = min(tile, H - top)
            w = min(tile, W - left)

            # crop patch
            patch = x[:, :, top:top + h, left:left + w]

            # pad to full tile for network
            patch = safe_pad_reflect_or_replicate(patch, tile, tile)

            with torch.no_grad():
                y = net(patch)

            if not predicts_clean:
                # residual => clean = x - w * noise
                y = patch - residual_weight * y

            # remove padding
            y = y[:, :, :h, :w].contiguous()

            # weight for current patch
            win_hw = win[:, :, :h, :w]

            acc[:, :, top:top + h, left:left + w] += y * win_hw
            wsum[:, :, top:top + h, left:left + w] += win_hw

    out = acc / torch.clamp(wsum, min=1e-8)
    return torch.clamp(out, 0.0, 1.0)


def maybe_percentile_norm(img01: np.ndarray, pct: float | None):
    if pct is None:
        return img01
    p_lo = np.percentile(img01, 100 - pct)
    p_hi = np.percentile(img01, pct)
    if p_hi <= p_lo + 1e-6:
        return img01
    img01 = (img01 - p_lo) / (p_hi - p_lo)
    return np.clip(img01, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--residual-weight", type=float, default=1.0)
    p.add_argument("--predicts-clean", action="store_true", help="Assume model outputs CLEAN directly")
    p.add_argument("--limit", type=int, default=0, help="limit number of images (debug)")
    p.add_argument("--percentile-norm", type=float, default=None, help="e.g., 99.5 to stretch contrast")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load state dict & build model
    sd = load_state_dict_any(args.ckpt)
    model = build_model_from_state_dict(sd).to(args.device)
    model.eval()

    files = sorted([p for p in inp.glob("*.png")])
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    print(f"[INFO] Files: {len(files)} | tile={args.tile} overlap={args.overlap} device={args.device} "
          f"| mode={'CLEAN' if args.predicts_clean else 'RESIDUAL'}")

    for fp in files:
        img, scale = read_image_gray(fp)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(args.device)
        y = tile_denoise(model, x, tile=args.tile, overlap=args.overlap, device=args.device,
                         predicts_clean=args.predicts_clean, residual_weight=args.residual_weight)
        out_np = y.squeeze(0).squeeze(0).detach().cpu().numpy()

        # optional normalisation (for visualisation only)
        out_np = maybe_percentile_norm(out_np, args.percentile_norm)

        rel = fp.name  # keep original name
        save_image_gray(out / rel, out_np, scale)

    print(f"[DONE] Saved to {out}")
    print("Tip: If outputs are too dark/flat, try --predicts-clean or adjust --percentile-norm 99.5")

if __name__ == "__main__":
    main()
