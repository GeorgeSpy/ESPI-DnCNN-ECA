#!/usr/bin/env python3
# dataset_baseline_psnr.py — report PSNR between pseudo-noisy and clean (upper bound sanity).
# Example:
#   python dataset_baseline_psnr.py --noisy "...\PseudoNoisy" --clean "...\Averaged" --limit 200

import argparse, math
from pathlib import Path
import numpy as np
from PIL import Image

def imread_f32(p: Path):
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 255: arr = arr / 65535.0
    else:               arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)

def mse(a,b): return float(np.mean((a-b)**2))

def psnr_from_mse(m): 
    if m <= 1e-12: return 99.0
    return 10.0 * math.log10(1.0/m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    nroot = Path(args.noisy); croot = Path(args.clean)
    files = sorted([p for p in croot.glob(f"*{args.ext}") if (nroot/p.name).exists()])
    if args.limit>0: files = files[:args.limit]
    if not files:
        print("No pairs.")
        return
    mses=[]; psnrs=[]
    for c in files:
        n = nroot / c.name
        a = imread_f32(n); b = imread_f32(c)
        m = mse(a,b); mses.append(m); psnrs.append(psnr_from_mse(m))
    print(f"Pairs: {len(files)}  Mean MSE: {np.mean(mses):.4f}  Mean PSNR(noisy->clean): {np.mean(psnrs):.2f} dB")

if __name__ == "__main__":
    main()
