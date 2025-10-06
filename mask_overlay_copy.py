#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_overlay_copy.py
--------------------
Create a masked copy of a folder tree by replacing a rectangular overlay area
with the local median (to neutralize labels/OSD) before phase extraction.

Usage:
python mask_overlay_copy.py ^
  --input  "C:\...\W01_ESPI_90db-Averaged" ^
  --output "C:\...\W01_ESPI_90db-Averaged_masked" ^
  --mask-rect 0 0 420 120
"""
import argparse, numpy as np
from pathlib import Path
from PIL import Image, ImageOps

VALID_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ("I;16","I;16B","I","L"):
        im = ImageOps.grayscale(im)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_uint01(arr: np.ndarray, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    ensure_dir(path.parent)
    Image.fromarray((arr*255.0+0.5).astype(np.uint8), mode="L").save(path)

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

def gather_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask-rect", type=int, nargs=4, metavar=("X","Y","W","H"), required=True)
    args = ap.parse_args()

    in_dir = Path(args.input); out_dir = Path(args.output)
    files = gather_files(in_dir)
    if not files:
        print(f"No images found under {in_dir}")
        return

    x,y,w,h = args.mask_rect
    for f in files:
        arr = imread_uint01(f)
        arr2 = mask_rect_fill_median(arr, x,y,w,h)
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".png")
        imsave_uint01(arr2, out_path)
    print(f"[DONE] Masked copy saved to {out_dir}")

if __name__ == "__main__":
    main()
