#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fft_inspect.py
--------------
Inspect how an overlay region affects the FFT carrier detection.

Saves two spectra images (log-magnitude) and prints detected peak (py, px):
  - spectrum_full.png          : original
  - spectrum_masked.png        : with a rectangular overlay masked (filled by local median)

Usage:
python fft_inspect.py --image "C:\path\frame.png" --center-suppress 10 --mask-rect 0 0 420 120

Notes:
- Uses the same peak finder as phase_extract_fft.py (center suppression + zero axes).
- If peak changes a lot between full/masked, the overlay is contaminating the FFT.
"""

import argparse, numpy as np
from pathlib import Path
from PIL import Image, ImageOps

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ("I;16","I;16B","I","L"):
        im = ImageOps.grayscale(im)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_gray01(arr01: np.ndarray, path: Path):
    arr01 = np.clip(arr01, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr01*255.0+0.5).astype(np.uint8), mode="L").save(path)

def fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def find_dominant_peak(mag: np.ndarray, center_suppress: int = 10):
    H, W = mag.shape
    cy, cx = H//2, W//2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rr2 = (yy-cy)**2 + (xx-cx)**2
    m = mag.copy()
    m[rr2 < center_suppress**2] = 0.0
    m[cy,:] = 0.0; m[:,cx] = 0.0
    idx = np.argmax(m)
    py, px = np.unravel_index(idx, m.shape)
    return int(py), int(px), float(m[py,px])

def mask_rect_fill_median(arr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = arr.shape
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0: return arr.copy()
    pad = 10
    yb0 = max(0, y0-pad); yb1 = min(H, y1+pad)
    xb0 = max(0, x0-pad); xb1 = min(W, x1+pad)
    neigh = arr[yb0:yb1, xb0:xb1]
    med = float(np.median(neigh))
    out = arr.copy()
    out[y0:y1, x0:x1] = med
    return out

def log_spectrum(mag: np.ndarray) -> np.ndarray:
    mg = np.log1p(mag)
    lo, hi = np.percentile(mg, [1.0, 99.0])
    return np.clip((mg - lo) / (hi - lo + 1e-6), 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--center-suppress", type=int, default=10)
    ap.add_argument("--mask-rect", type=int, nargs=4, metavar=("X","Y","W","H"), default=None)
    ap.add_argument("--outdir", default="fft_inspect_out")
    args = ap.parse_args()

    img = imread_uint01(Path(args.image))
    x = img - img.mean()

    X = fft2c(x); mag = np.abs(X).astype(np.float32)
    py, px, val = find_dominant_peak(mag, center_suppress=args.center_suppress)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    imsave_gray01(log_spectrum(mag), outdir / "spectrum_full.png")

    if args.mask_rect is not None:
        xm, ym, wm, hm = args.mask_rect
        xmimg = mask_rect_fill_median(img, xm, ym, wm, hm)
        x2 = xmimg - xmimg.mean()
        X2 = fft2c(x2); mag2 = np.abs(X2).astype(np.float32)
        py2, px2, val2 = find_dominant_peak(mag2, center_suppress=args.center_suppress)
        imsave_gray01(log_spectrum(mag2), outdir / "spectrum_masked.png")
        print(f"[FULL ] peak at (py,px)=({py},{px}) mag={val:.3f}")
        print(f"[MASK ] peak at (py,px)=({py2},{px2}) mag={val2:.3f}")
        dy, dx = abs(py2-py), abs(px2-px)
        print(f"Delta peak: dy={dy}, dx={dx}")
        if dy+dx > 5:
            print(">>> Overlay likely contaminates the FFT carrier detection.")
        else:
            print(">>> Peak stable. Overlay likely not affecting carrier.")
    else:
        print(f"[FULL ] peak at (py,px)=({py},{px}) mag={val:.3f}")
        print("Run again with --mask-rect x y w h to compare.")

if __name__ == "__main__":
    main()
