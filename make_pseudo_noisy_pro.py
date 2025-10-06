#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pseudo_noisy_pro.py
------------------------
Advanced pseudo-noisy generator for ESPI-like single-shot frames.

Pipeline (in order):
1) Flat-field illumination (low-pass multiplicative shading)
2) Correlated speckle (multiplicative), selectable mode (gamma/lognormal/exponential)
3) Poisson shot noise (photon count limited)
4) Additive banding/readout (row/column offsets, sinusoidal banding) [optional]
5) Additive Gaussian noise
6) Salt & pepper outliers [optional]

Extras:
- Spatial correlation controlled via Gaussian blur sigma (px)
- All random parts can be seeded for reproducibility
- Option to save intermediate components (fields) for inspection

Dependencies: numpy, pillow
Optional: opencv-python OR scipy (for Gaussian blur). If neither present, a small numpy fallback is used.

Example (harder speckle + flat-field shading):
python make_pseudo_noisy_pro.py ^
  --input "C:\...\W01_ESPI_90db-Averaged" ^
  --output "C:\...\W01_ESPI_90db-PseudoNoisy_HARD" ^
  --flat-sigma 90 --flat-strength 0.35 ^
  --speckle-mode lognormal --speckle-contrast 0.45 --speckle-corr 7.0 ^
  --poisson-peak 50 --gauss-sigma 0.010 ^
  --row-sigma 0.002 --col-sigma 0.002 --band-amp 0.01 --band-period 240 --band-axis x ^
  --saltpepper-p 0.0005 --seed 123 --save-components
"""

import argparse, numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from math import ceil, pi, sin, cos
from tqdm import tqdm

VALID_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

# ---------- I/O ----------

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

def gather_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)

# ---------- Gaussian blur helper (cv2 -> scipy -> numpy fallback) ----------

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    try:
        import cv2
        k = int(2*ceil(3*sigma)+1)
        return cv2.GaussianBlur(img, (k,k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    except Exception:
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma=sigma, mode="reflect")
        except Exception:
            # simple separable conv fallback
            r = int(ceil(3*sigma))
            xs = np.arange(-r, r+1, dtype=np.float32)
            ker = np.exp(-(xs*xs)/(2.0*sigma*sigma + 1e-12))
            ker /= ker.sum()
            # pad reflect
            def pad_reflect(a, pad, axis):
                a0 = np.take(a, np.arange(pad,0,-1), axis=axis)
                a1 = np.take(a, np.arange(a.shape[axis]-2, a.shape[axis]-pad-2, -1), axis=axis)
                return np.concatenate([a0, a, a1], axis=axis)
            # conv along axis 0 then 1
            out = img.copy()
            # axis 0
            pad = len(ker)//2
            tmp = pad_reflect(out, pad, 0)
            out0 = np.apply_along_axis(lambda m: np.convolve(m, ker, mode='valid'), 0, tmp)
            # axis 1
            tmp = pad_reflect(out0, pad, 1)
            out1 = np.apply_along_axis(lambda m: np.convolve(m, ker, mode='valid'), 1, tmp)
            return out1

# ---------- Noise components ----------

def gen_flatfield(H: int, W: int, sigma: float, strength: float, rng: np.random.Generator) -> np.ndarray:
    """
    Low-frequency multiplicative shading field with mean 1.0
    strength ~ std of the log-field; typical 0.2–0.5
    """
    if sigma <= 0 or strength <= 0:
        return np.ones((H,W), dtype=np.float32)
    base = rng.normal(0.0, 1.0, size=(H,W)).astype(np.float32)
    low = gaussian_blur(base, sigma=sigma)
    field = np.exp(low * strength).astype(np.float32)
    field /= (field.mean() + 1e-6)
    return field

def gen_speckle_field(H: int, W: int, mode: str, k: float, contrast: float, corr_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Multiplicative speckle field with mean ~1.0
    - mode 'gamma': Gamma(k, 1/k) mean 1; contrast tuned by k and optional smoothing
    - mode 'lognormal': exp(N(0,1)*alpha); alpha tuned by 'contrast'
    - mode 'exponential': Exp(1.0) (mean 1), then smooth if corr_sigma>0
    """
    mode = (mode or "gamma").lower()
    if mode == "gamma":
        k = max(0.5, float(k))
        field = rng.gamma(shape=k, scale=1.0/k, size=(H,W)).astype(np.float32)  # mean 1
        if corr_sigma > 0: field = gaussian_blur(field, corr_sigma)
        field /= (field.mean() + 1e-6)
        # optional contrast adjustment via power-law towards 1
        if contrast > 0:
            mu = field.mean()
            field = mu + (field - mu) * (1.0 + contrast)
        return field.astype(np.float32)

    if mode == "exponential":
        field = rng.exponential(scale=1.0, size=(H,W)).astype(np.float32)  # mean 1
        if corr_sigma > 0: field = gaussian_blur(field, corr_sigma)
        field /= (field.mean() + 1e-6)
        return field

    # lognormal
    alpha = max(0.0, float(contrast))
    base = rng.normal(0.0, 1.0, size=(H,W)).astype(np.float32)
    if corr_sigma > 0: base = gaussian_blur(base, corr_sigma)
    field = np.exp(base * alpha).astype(np.float32)
    field /= (field.mean() + 1e-6)
    return field

def add_poisson(x: np.ndarray, peak: float, rng: np.random.Generator) -> np.ndarray:
    if peak <= 0: return x
    lam = np.clip(x, 0.0, 1.0) * peak
    y = rng.poisson(lam).astype(np.float32) / float(peak)
    return np.clip(y, 0.0, 1.0)

def add_gauss(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0: return x
    return np.clip(x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)

def add_row_col_offsets(x: np.ndarray, row_sigma: float, col_sigma: float, rng: np.random.Generator) -> np.ndarray:
    H, W = x.shape
    out = x.copy()
    if row_sigma > 0:
        row_off = rng.normal(0.0, row_sigma, size=(H,1)).astype(np.float32)
        out += row_off
    if col_sigma > 0:
        col_off = rng.normal(0.0, col_sigma, size=(1,W)).astype(np.float32)
        out += col_off
    return np.clip(out, 0.0, 1.0)

def add_banding(x: np.ndarray, amp: float, period: float, phase: float, axis: str = "x") -> np.ndarray:
    if amp <= 0 or period <= 1: return x
    H, W = x.shape
    if axis.lower() == "x":  # vertical bands (vary along x/columns)
        xs = np.arange(W, dtype=np.float32)
        band = 1.0 + amp * np.sin(2*pi*xs/period + phase)
        out = x * band[np.newaxis, :]
    else:  # horizontal bands
        ys = np.arange(H, dtype=np.float32)
        band = 1.0 + amp * np.sin(2*pi*ys/period + phase)
        out = x * band[:, np.newaxis]
    return np.clip(out, 0.0, 1.0)

def add_saltpepper(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    if p <= 0: return x
    out = x.copy()
    mask = rng.random(size=x.shape)
    out[mask < p*0.5] = 0.0
    out[(mask >= p*0.5) & (mask < p)] = 1.0
    return out

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    # Flat-field
    ap.add_argument("--flat-sigma", type=float, default=80.0, help="Gaussian sigma (px) for low-pass shading (0=off)")
    ap.add_argument("--flat-strength", type=float, default=0.30, help="strength of flat-field (0..~0.6 typical)")
    # Speckle
    ap.add_argument("--speckle-mode", type=str, default="lognormal", choices=["gamma","lognormal","exponential"])
    ap.add_argument("--speckle-k", type=float, default=3.0, help="Gamma shape k (for mode=gamma)")
    ap.add_argument("--speckle-contrast", type=float, default=0.40, help="contrast (lognormal alpha or gamma power)")
    ap.add_argument("--speckle-corr", type=float, default=6.0, help="Gaussian sigma (px) for speckle correlation (0=uncorrelated)")
    # Poisson & Gaussian
    ap.add_argument("--poisson-peak", type=float, default=60.0)
    ap.add_argument("--gauss-sigma", type=float, default=0.010)
    # Readout & banding
    ap.add_argument("--row-sigma", type=float, default=0.0, help="add per-row offsets std")
    ap.add_argument("--col-sigma", type=float, default=0.0, help="add per-column offsets std")
    ap.add_argument("--band-amp", type=float, default=0.0, help="multiplicative sinusoidal band amplitude")
    ap.add_argument("--band-period", type=float, default=240.0, help="band period (pixels)")
    ap.add_argument("--band-axis", type=str, default="x", choices=["x","y"], help="x: vertical bands; y: horizontal bands")
    # Outliers
    ap.add_argument("--saltpepper-p", type=float, default=0.0, help="probability per pixel for salt/pepper")
    # Generic
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-components", action="store_true", help="save fields (flat, speckle, etc.) next to outputs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    in_dir = Path(args.input); out_dir = Path(args.output)
    files = gather_files(in_dir)
    if not files:
        print(f"No images found under {in_dir}")
        return

    comp_dir = out_dir / "_components"
    if args.save_components:
        ensure_dir(comp_dir)

    for f in tqdm(files, desc="Synthesizing HARD pseudo-noisy"):
        clean = imread_uint01(f)
        H, W = clean.shape

        # 1) Flat-field shading
        flat = gen_flatfield(H, W, sigma=args.flat_sigma, strength=args.flat_strength, rng=rng)
        x = np.clip(clean * flat, 0.0, 1.0)

        # 2) Correlated speckle multiplicative
        sp = gen_speckle_field(H, W, mode=args.speckle-mode, k=args.speckle_k,
                               contrast=args.speckle_contrast, corr_sigma=args.speckle_corr, rng=rng)
        x = np.clip(x * sp, 0.0, 1.0)

        # 3) Poisson
        x = add_poisson(x, args.poisson_peak, rng)

        # 4) Readout & banding
        if args.row_sigma > 0 or args.col_sigma > 0:
            x = add_row_col_offsets(x, args.row_sigma, args.col_sigma, rng)
        if args.band_amp > 0:
            phase = rng.uniform(0.0, 2*pi)
            x = add_banding(x, args.band_amp, args.band_period, phase, axis=args.band_axis)

        # 5) Gaussian
        x = add_gauss(x, args.gauss_sigma, rng)

        # 6) Salt & pepper
        x = add_saltpepper(x, args.saltpepper_p, rng)

        # Save
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".png")
        imsave_uint01(x, out_path)

        if args.save_components:
            base = comp_dir / rel.with_suffix("")
            imsave_uint01((flat/flat.max()), base.with_name(base.name + "_flat.png"))
            imsave_uint01((sp/sp.max()),   base.with_name(base.name + "_speckle.png"))

    print(f"[DONE] Saved to {out_dir}")
    if args.save_components:
        print(f"Components saved under {comp_dir}")
