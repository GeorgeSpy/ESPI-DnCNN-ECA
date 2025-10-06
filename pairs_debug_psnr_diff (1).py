#!/usr/bin/env python3

# pairs_debug_psnr_diff.py
# Compute PSNR per (clean,noisy) pair and export the worst-K triplets
# Works with 8-bit or 16-bit PNGs (grayscale).

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from math import log10
from tqdm import tqdm

def load_gray_f32(p: Path):
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

def to_u8(x):
    return (np.clip(x,0,1)*255.0 + 0.5).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="Folder with clean PNGs")
    ap.add_argument("--noisy", required=True, help="Folder with noisy PNGs")
    ap.add_argument("--outdir", required=True, help="Output folder for CSV and worst triplets")
    ap.add_argument("--limit", type=int, default=0, help="Only first N files (after sorting)")
    ap.add_argument("--save-worst", type=int, default=5, help="How many worst triplets to export")
    args = ap.parse_args()

    clean_dir = Path(args.clean); noisy_dir = Path(args.noisy)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(clean_dir.glob("*.png"))
    if args.limit>0:
        files = files[:args.limit]
    if not files:
        print("No PNGs in clean dir."); return

    rows = []
    for c in tqdm(files, desc="Compare"):
        n = noisy_dir / c.name
        if not n.exists():
            continue
        cimg = load_gray_f32(c); nimg = load_gray_f32(n)
        h = min(cimg.shape[0], nimg.shape[0]); w = min(cimg.shape[1], nimg.shape[1])
        if h<=0 or w<=0:
            continue
        cimg = cimg[:h,:w]; nimg = nimg[:h,:w]
        p, mse = psnr(cimg, nimg)
        rows.append((c.name, p, mse, h, w))

    if not rows:
        print("No pairs compared."); return

    # Summary
    ps = [r[1] for r in rows]; ms = [r[2] for r in rows]
    print(f"Pairs: {len(rows)}  Mean MSE: {np.mean(ms):.4f}  Mean PSNR: {np.mean(ps):.2f} dB")
    print(f"Min/Max PSNR: {np.min(ps):.2f} / {np.max(ps):.2f} dB")

    # Save worst K
    rows_sorted = sorted(rows, key=lambda x: x[1])  # ascending by PSNR
    worst = rows_sorted[:max(0, args.save_worst)]
    (outdir / "worst").mkdir(exist_ok=True)
    for name, p, mse, h, w in worst:
        c = load_gray_f32(clean_dir/name); n = load_gray_f32(noisy_dir/name)
        c = c[:h,:w]; n = n[:h,:w]
        diff = np.abs(n - c)
        d_vis = diff / (diff.max()+1e-8)  # normalize for visualization
        trip = np.concatenate([c, n, d_vis], axis=1)
        Image.fromarray(to_u8(trip)).save(outdir/"worst"/f"{name}_PSNR{p:.2f}.png")
    print(f"[SAVED] Worst triplets -> {outdir/'worst'}")

    # CSV
    with open(outdir/"pairs_psnr.csv", "w", encoding="utf-8") as f:
        f.write("name,psnr_db,mse,height,width\n")
        for name, p, mse, h, w in rows_sorted:
            f.write(f"{name},{p:.4f},{mse:.6f},{h},{w}\n")
    print(f"[CSV] {outdir/'pairs_psnr.csv'}")

if __name__ == "__main__":
    main()
