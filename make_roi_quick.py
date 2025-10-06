#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_roi_quick.py
-----------------
Create a simple rectangular ROI mask from a sample image.
White (=255) inside ROI, black (=0) outside.
Default: exclude a top strip (overlay) and keep the rest as ROI.

Usage:
python make_roi_quick.py ^
  --sample "C:\path\to\one\averaged_frame.png" ^
  --out "C:\Users\...\roi_mask.png" ^
  --exclude-top 140  --exclude-bottom 0 --exclude-left 0 --exclude-right 0
"""
import argparse, numpy as np
from PIL import Image, ImageOps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="path to one image to infer size (PNG/TIF/JPG)")
    ap.add_argument("--out", required=True, help="where to save the ROI PNG")
    ap.add_argument("--exclude-top", type=int, default=0)
    ap.add_argument("--exclude-bottom", type=int, default=0)
    ap.add_argument("--exclude-left", type=int, default=0)
    ap.add_argument("--exclude-right", type=int, default=0)
    args = ap.parse_args()

    im = Image.open(args.sample)
    if im.mode != "L":
        im = ImageOps.grayscale(im)
    W, H = im.size
    top = max(0, args.exclude_top); bottom = max(0, args.exclude_bottom)
    left = max(0, args.exclude_left); right = max(0, args.exclude_right)

    mask = np.zeros((H, W), dtype=np.uint8)
    y0 = top
    y1 = H - bottom
    x0 = left
    x1 = W - right
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 255

    out = Image.fromarray(mask, mode="L")
    out.save(args.out)
    print(f"[DONE] ROI saved to {args.out}  (size {W}x{H}, ROI rect: x=[{x0},{x1}), y=[{y0},{y1}))")

if __name__ == "__main__":
    main()
