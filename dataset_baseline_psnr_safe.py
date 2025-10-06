#!/usr/bin/env python3
# dataset_baseline_psnr_safe.py
# Robust PSNR between two folders (handles 8/16-bit PNG, grayscale align).

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from math import log10
from tqdm import tqdm

def load_f32_gray(p: Path):
    im = Image.open(p).convert("L")
    arr = np.array(im)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def psnr(clean, noisy):
    mse = float(np.mean((clean - noisy)**2))
    if mse <= 1e-12: return 99.0, 0.0
    return 10.0 * log10(1.0 / mse), mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    clean_dir = Path(args.clean); noisy_dir = Path(args.noisy)
    files = sorted([p for p in clean_dir.glob("*.png")])
    if args.limit>0: files = files[:args.limit]

    n=0; mse_sum=0.0; ps_sum=0.0
    for c in tqdm(files, desc="Pairs"):
        n_path = noisy_dir / c.name
        if not n_path.exists():
            continue
        cimg = load_f32_gray(c); nimg = load_f32_gray(n_path)
        # align shapes
        h = min(cimg.shape[0], nimg.shape[0])
        w = min(cimg.shape[1], nimg.shape[1])
        cimg = cimg[:h,:w]; nimg = nimg[:h,:w]
        ps, mse = psnr(cimg, nimg)
        ps_sum += ps; mse_sum += mse; n += 1
    if n==0:
        print("No pairs compared."); return
    print(f"Pairs: {n}  Mean MSE: {mse_sum/n:.4f}  Mean PSNR(noisy->clean): {ps_sum/n:.2f} dB")

if __name__ == "__main__":
    main()
