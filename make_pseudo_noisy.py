#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pseudo_noisy.py
--------------------
Generate synthetic single-shot noisy images from pseudo-clean "Averaged" frames.
Noise model (applied in this order):
  1) Multiplicative speckle: Gamma(k, 1/k) with mean=1.0  (--speckle-k)
  2) Shot noise (Poisson):  y = Poisson(x * peak) / peak   (--poisson-peak)
  3) Additive Gaussian:     y = y + N(0, sigma)           (--gauss-sigma)

All intensities are in [0,1]. Input images can be PNG/TIF/JPG/BMP (16-bit or 8-bit).
Outputs are saved as PNG (8-bit) mirroring the input folder tree.
Also writes a CSV with per-image baseline PSNR/SSIM vs clean.

Usage example:
python make_pseudo_noisy.py ^
  --input "C:\...\W01_ESPI_90db-Averaged" ^
  --output "C:\...\W01_ESPI_90db-PseudoNoisy" ^
  --speckle-k 3.0 --poisson-peak 60 --gauss-sigma 0.012 --seed 42
"""

import argparse, csv, numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

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

def psnr(x, y, eps=1e-12):
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0: return 99.0
    import math
    return 20.0 * math.log10(1.0 / (mse**0.5 + eps))

def ssim_fallback(x, y):
    # Lightweight SSIM approximation (not full). Try to use skimage if available.
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(x, y, data_range=1.0))
    except Exception:
        mu_x = x.mean(); mu_y = y.mean()
        sig_x = x.var(); sig_y = y.var()
        sig_xy = ((x - mu_x) * (y - mu_y)).mean()
        C1 = 0.01**2; C2 = 0.03**2
        num = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
        den = (mu_x*mu_x + mu_y*mu_y + C1) * (sig_x + sig_y + C2)
        return float(num / (den + 1e-12))

def gather_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)

def add_noise(img01: np.ndarray, k_speckle: float, peak: float, sigma_g: float, rng: np.random.Generator) -> np.ndarray:
    # 1) Speckle multiplicative
    if k_speckle > 0:
        speckle = rng.gamma(shape=k_speckle, scale=1.0/k_speckle, size=img01.shape).astype(np.float32)  # mean 1
        y = img01 * speckle
    else:
        y = img01.copy()

    # 2) Poisson shot noise
    if peak > 0:
        lam = np.clip(y, 0.0, 1.0) * peak
        y = rng.poisson(lam).astype(np.float32) / float(peak)

    # 3) Additive Gaussian
    if sigma_g > 0:
        y = y + rng.normal(0.0, sigma_g, size=img01.shape).astype(np.float32)

    return np.clip(y, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--speckle-k", type=float, default=3.0, help="Gamma shape k (scale=1/k). 0=off")
    ap.add_argument("--poisson-peak", type=float, default=60.0, help="Shot noise peak (photons). 0=off")
    ap.add_argument("--gauss-sigma", type=float, default=0.012, help="Additive Gaussian std in [0,1]. 0=off")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    in_dir = Path(args.input); out_dir = Path(args.output)
    files = gather_files(in_dir)
    if not files:
        print(f"No images found under {in_dir}")
        return

    rows = [("rel_path","psnr_noisy_vs_clean","ssim_noisy_vs_clean","speckle_k","poisson_peak","gauss_sigma")]
    for f in tqdm(files, desc="Synthesizing pseudo-noisy"):
        clean = imread_uint01(f)
        noisy = add_noise(clean, args.speckle_k, args.poisson_peak, args.gauss_sigma, rng)
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".png")
        imsave_uint01(noisy, out_path)

        p = psnr(noisy, clean)
        s = ssim_fallback(noisy, clean)
        rows.append((str(rel), f"{p:.4f}", f"{s:.6f}", f"{args.speckle_k}", f"{args.poisson_peak}", f"{args.gauss_sigma}"))

    # CSV
    csv_path = out_dir / "pseudo_noisy_baseline.csv"
    ensure_dir(csv_path.parent)
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as fo:
        _csv.writer(fo).writerows(rows)

    print(f"[DONE] Saved pseudo-noisy images to: {out_dir}")
    print(f"Baseline CSV: {csv_path}")
    print("Tip: denoise them with your model and evaluate with eval_denoise_batch.py")

if __name__ == "__main__":
    main()
