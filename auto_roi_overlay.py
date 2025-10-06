#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
auto_roi_overlay.py
-------------------
Auto-detect overlay banner/text region on averaged ESPI frames and create an ROI mask
(white inside ROI, black outside). Optionally, produce a *masked copy* of the dataset
where the detected overlay area is replaced by local median (so FFT won't pick it up).

Detection heuristics (robust, no extra deps):
1) Try **top-left rectangle** search (up to max_w_ratio, max_h_ratio of image).
   We scan widths/heights on a grid and pick the largest area whose region is mostly dark
   (allows some bright text). Good for overlays like a dark box with white text at top-left.
2) If not found, fallback to **full-width top band**: find the tallest band from top
   whose mean intensity is very low.
We also ensure the rectangle touches the top edge (overlay is at the top).

Outputs:
- ROI PNG: white (=255) valid area, black (=0) excluded overlay.
- Optional masked copy of all images (PNG), preserving folder structure.
- Debug preview with rectangle drawn on the sample.

Usage example:
python auto_roi_overlay.py ^
  --input-dir "C:\Users\...\W01_ESPI_90db-Averaged" ^
  --roi-out   "C:\Users\...\roi_mask.png" ^
  --masked-out "C:\Users\...\W01_ESPI_90db-Averaged_masked" ^
  --prefer topleft --pad-x 4 --pad-y 4 --preview "C:\Users\...\overlay_preview.png"

Notes:
- Works with PNG/TIF/JPG/BMP (8-bit or 16-bit). Outputs masked images as PNG.
- You can also pass a single --sample path instead of --input-dir.
"""

import argparse, numpy as np, sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

VALID_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ("I;16","I;16B","I","L"):
        im = ImageOps.grayscale(im)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_uint01(arr01: np.ndarray, path: Path):
    arr01 = np.clip(arr01, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr01*255.0+0.5).astype(np.uint8), mode="L").save(path)

def gather_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p

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

def detect_topleft_rect(img: np.ndarray, max_h_ratio=0.35, max_w_ratio=0.75, step=8, dark_thr=None):
    """Scan top-left region for a large mostly-dark rectangle. Return (x,y,w,h) or None."""
    H, W = img.shape
    max_h = int(H * max_h_ratio)
    max_w = int(W * max_w_ratio)
    max_h = max(32, min(max_h, H))
    max_w = max(64, min(max_w, W))
    # dynamic threshold: very dark compared to global; cap to 0.25
    if dark_thr is None:
        mu = float(img.mean()); sd = float(img.std())
        dark_thr = min(0.25, mu - 0.5*sd)
        dark_thr = max(0.02, dark_thr)  # keep in [0.02, 0.25]
    best = None; best_area = 0
    # Precompute integral image for speed on dark mask
    dark = (img < dark_thr).astype(np.uint8)
    ii = dark.cumsum(0).cumsum(1)  # integral image
    def sum_rect(x, y, w, h):
        x1, y1 = x+w-1, y+h-1
        A = ii[y1, x1]
        B = ii[y-1, x1] if y > 0 else 0
        C = ii[y1, x-1] if x > 0 else 0
        D = ii[y-1, x-1] if (x>0 and y>0) else 0
        return int(A - B - C + D)
    for h in range(32, max_h+1, step):
        for w in range(64, max_w+1, step):
            x, y = 0, 0  # anchored top-left
            area = w*h
            dark_count = sum_rect(x, y, w, h)
            dark_ratio = dark_count / float(area)
            # allow some bright text; require high dark ratio
            if dark_ratio >= 0.75 and area > best_area:
                best_area = area
                best = (x, y, w, h)
    return best

def detect_topband(img: np.ndarray, max_h_ratio=0.35):
    """Detect a full-width top band that is dark on average. Return (0,0,W,h) or None."""
    H, W = img.shape
    max_h = int(H * max_h_ratio)
    # row mean profile
    row_mean = img[:max_h, :].mean(axis=1)
    mu = row_mean.mean(); sd = row_mean.std()
    thr = min(0.35, mu - 0.25*sd)  # relatively dark
    # find the largest contiguous run from the top whose means are below a relaxed threshold
    h = 0
    for y in range(len(row_mean)):
        if row_mean[y] < thr:
            h = y+1
        else:
            break
    if h >= 24:  # need a minimum thickness
        return (0, 0, W, int(h))
    return None

def choose_overlay(img: np.ndarray, prefer="topleft"):
    cand1 = detect_topleft_rect(img)
    cand2 = detect_topband(img)
    if prefer == "topleft":
        return cand1 if cand1 is not None else cand2
    else:
        return cand2 if cand2 is not None else cand1

def build_roi_mask(shape, rect, pad_x=0, pad_y=0):
    H, W = shape
    x, y, w, h = rect
    x = max(0, x - pad_x); y = max(0, y - pad_y)
    w = min(W - x, w + 2*pad_x); h = min(H - y, h + 2*pad_y)
    m = np.ones((H, W), dtype=np.uint8) * 255
    m[y:y+h, x:x+w] = 0
    return m, (x, y, w, h)

def draw_preview(img01: np.ndarray, rect, path: Path):
    x, y, w, h = rect
    H, W = img01.shape
    vis = Image.fromarray((img01*255.0+0.5).astype(np.uint8)).convert("RGB")
    drw = ImageDraw.Draw(vis)
    drw.rectangle([x, y, x+w-1, y+h-1], outline=(255,0,0), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="", help="root folder with images (any of PNG/TIF/JPG/BMP)")
    ap.add_argument("--sample", default="", help="use this single image to detect overlay (overrides input-dir)")
    ap.add_argument("--roi-out", required=True, help="where to save ROI mask PNG")
    ap.add_argument("--masked-out", default="", help="optional: save masked copy of the dataset here")
    ap.add_argument("--prefer", choices=["topleft","topband"], default="topleft", help="preferred overlay shape")
    ap.add_argument("--pad-x", type=int, default=0, help="inflate rectangle by this many pixels horizontally")
    ap.add_argument("--pad-y", type=int, default=0, help="inflate rectangle by this many pixels vertically")
    ap.add_argument("--preview", default="", help="optional preview PNG with rectangle drawn on sample")
    args = ap.parse_args()

    # Pick sample image
    sample_path = None
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.exists():
            print("[ERR] Sample not found:", sample_path); sys.exit(1)
    else:
        root = Path(args.input_dir)
        if not root.exists():
            print("[ERR] input-dir not found"); sys.exit(1)
        for f in gather_files(root):
            sample_path = f; break
        if sample_path is None:
            print("[ERR] No images found under", root); sys.exit(1)

    # Detect overlay on sample
    img = imread_uint01(sample_path)
    rect = choose_overlay(img, prefer=args.prefer)
    if rect is None:
        print("[WARN] Could not detect overlay automatically. Writing full-white ROI (no exclusion).")
        m = np.ones_like(img, dtype=np.uint8) * 255
        Image.fromarray(m, mode="L").save(args.roi_out)
        sys.exit(0)

    # Build/save ROI
    roi_mask, rect_padded = build_roi_mask(img.shape, rect, pad_x=args.pad_x, pad_y=args.pad_y)
    Image.fromarray(roi_mask, mode="L").save(args.roi_out)
    print(f"[ROI] Saved: {args.roi_out}  rect={rect_padded} (x,y,w,h)")

    # Optional preview
    if args.preview:
        draw_preview(img, rect_padded, Path(args.preview))
        print("[Preview] Saved:", args.preview)

    # Optional masked copy
    if args.masked_out:
        in_dir = Path(args.input_dir) if args.input_dir else sample_path.parent
        out_dir = Path(args.masked_out); out_dir.mkdir(parents=True, exist_ok=True)
        x,y,w,h = rect_padded
        n = 0
        for f in gather_files(in_dir):
            arr = imread_uint01(f)
            arr2 = mask_rect_fill_median(arr, x,y,w,h)
            rel = f.relative_to(in_dir)
            dst = out_dir / rel.with_suffix(".png")
            imsave_uint01(arr2, dst)
            n += 1
        print(f"[Masked] Wrote {n} images to {out_dir}")

if __name__ == "__main__":
    main()
