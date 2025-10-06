#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_extract_fft.py
--------------------
Takeda-style Fourier demodulation for ESPI fringe patterns with:
- optional flat-field correction (FFT Gaussian low-pass division)
- automatic carrier peak detection with center suppression
- Gaussian band-pass around dominant side-lobe
- frequency recentering (roll) and IFFT to complex analytic signal
- wrapped phase, optional unwrapped phase (via scikit-image if available)
- quality map (modulation depth = |analytic|)
- ROI mask support (png; auto-resized to match)
- displacement map if --kappa given
- CSV with per-image stats over ROI

Dependencies (minimal):
    pip install numpy pillow tqdm
Optional for unwrapping:
    pip install scikit-image
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- I/O ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_gray01(arr01: np.ndarray, path: Path) -> None:
    arr01 = np.clip(arr01, 0.0, 1.0)
    ensure_dir(path.parent)
    Image.fromarray((arr01 * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)

def load_roi_mask(mask_path: Optional[Path], target_hw: Tuple[int,int]) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    if not mask_path.exists():
        print(f"[ROI] Mask not found: {mask_path}. Ignoring.")
        return None
    m = Image.open(mask_path).convert("L")
    if m.size != (target_hw[1], target_hw[0]):
        m = m.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
    m = (np.array(m, dtype=np.uint8) > 0).astype(np.float32)
    return m

# ---------- FFT helpers ----------

def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT (like fftshift(fft2(ifftshift(x)))) on real input."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X: np.ndarray) -> np.ndarray:
    """Centered 2D inverse FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def gaussian_lp_spectrum(h: int, w: int, sigma: float) -> np.ndarray:
    """Centered 2D Gaussian LP filter in frequency domain."""
    cy, cx = h // 2, w // 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-(rr2) / (2.0 * (sigma ** 2) + 1e-12)).astype(np.float32)

def gaussian_bp_around(h: int, w: int, py: int, px: int, sigma: float) -> np.ndarray:
    """Centered 2D Gaussian mask around (py,px) in frequency domain."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr2 = (yy - py) ** 2 + (xx - px) ** 2
    return np.exp(-(rr2) / (2.0 * (sigma ** 2) + 1e-12)).astype(np.float32)

def find_dominant_peak(mag: np.ndarray, center_suppress: int = 10) -> Tuple[int,int]:
    """Find dominant side-lobe peak, ignoring a central disk of given radius."""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = rr2 >= (center_suppress ** 2)
    mag_s = mag.copy()
    mag_s[~mask] = 0.0
    # Zero-out exact axes to avoid DC/stripe artifacts bias
    mag_s[cy, :] = 0.0
    mag_s[:, cx] = 0.0
    # Find global maximum
    idx = np.argmax(mag_s)
    py, px = np.unravel_index(idx, mag_s.shape)
    return int(py), int(px)

def flatfield_correct(img01: np.ndarray, sigma_lp: float) -> np.ndarray:
    """Flat-field via FFT Gaussian low-pass division."""
    if sigma_lp <= 0:
        return img01
    H, W = img01.shape
    X = fft2c(img01)
    Hlp = gaussian_lp_spectrum(H, W, sigma_lp)
    # Low-pass in spatial via IFFT of (spectrum * Hlp)
    low = np.real(ifft2c(X * Hlp))
    low = np.clip(low, 1e-3, None)  # avoid division by zero
    corr = img01 / low
    # Renormalize to mean 0.5 for stability
    m = np.mean(corr)
    if m > 1e-6:
        corr = corr * (0.5 / m)
    return np.clip(corr, 0.0, 1.0)

# ---------- Phase extraction ----------

def takeda_phase(img01: np.ndarray, band_sigma: float, center_suppress: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (wrapped phase [-pi,pi], quality map = |analytic|).
    Steps:
    - subtract global mean
    - FFT (centered), find dominant side-lobe
    - Gaussian band-pass around that peak
    - recenter peak to origin via np.roll
    - IFFT to complex analytic signal
    """
    H, W = img01.shape
    # Subtract mean (DC)
    x = img01.astype(np.float32)
    x = x - np.mean(x)
    # FFT
    Xc = fft2c(x)
    mag = np.abs(Xc).astype(np.float32)
    py, px = find_dominant_peak(mag, center_suppress=center_suppress)
    # Band-pass around chosen peak
    G = gaussian_bp_around(H, W, py, px, band_sigma)
    Xc_bp = Xc * G
    # Recentering: roll so that (py,px) goes to center (cy,cx)
    cy, cx = H // 2, W // 2
    dy, dx = cy - py, cx - px
    Xc_base = np.roll(np.roll(Xc_bp, shift=dy, axis=0), shift=dx, axis=1)
    # IFFT -> analytic field
    z = ifft2c(Xc_base)
    phase_wrapped = np.angle(z).astype(np.float32)  # [-pi,pi]
    quality = np.abs(z).astype(np.float32)
    # normalize quality to [0,1] per-image for visualization
    qmin, qmax = np.percentile(quality, [1.0, 99.0])
    qviz = np.clip((quality - qmin) / (qmax - qmin + 1e-6), 0.0, 1.0)
    return phase_wrapped, qviz

def unwrap_if_available(phase_wrapped: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Try to unwrap using skimage if present."""
    try:
        from skimage.restoration import unwrap_phase
    except Exception:
        return None, "skimage not installed; skipping unwrap"
    try:
        un = unwrap_phase(phase_wrapped)
        return un.astype(np.float32), "ok"
    except Exception as e:
        return None, f"unwrap error: {e}"

# ---------- Stats ----------

def circular_std(ang: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Circular std of angles in radians ([-pi,pi])."""
    if mask is not None:
        a = ang[mask > 0]
    else:
        a = ang.ravel()
    if a.size == 0:
        return float("nan")
    C = np.mean(np.cos(a)); S = np.mean(np.sin(a))
    R = np.sqrt(C*C + S*S) + 1e-12
    return float(np.sqrt(-2.0 * np.log(R)))

def std_roi(arr: np.ndarray, mask: Optional[np.ndarray]) -> float:
    if mask is not None:
        v = arr[mask > 0]
    else:
        v = arr.ravel()
    if v.size == 0:
        return float("nan")
    return float(np.std(v - np.mean(v)))

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder with denoised PNGs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--roi-mask", default="", help="PNG mask (white=ROI); auto-resize to image size")
    ap.add_argument("--flatfield", type=float, default=0.0, help="sigma (px) for Gaussian LP (0=off)")
    ap.add_argument("--band", type=float, default=25.0, help="sigma (px) for Gaussian band-pass around carrier peak")
    ap.add_argument("--center-suppress", type=int, default=10, help="radius (px) to zero near DC when searching peak")
    ap.add_argument("--unwrap", default="auto", choices=["auto","none"])
    ap.add_argument("--kappa", type=float, default=0.0, help="rad per unit displacement (e.g., 2*pi if 1 fringe = 1 unit)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir); out_dir = Path(args.output_dir)
    files = sorted(in_dir.rglob("*.png"))
    if not files:
        print("No PNGs under", in_dir); return

    # Output subfolders
    out_wrapped_png = out_dir / "phase_wrapped_png"
    out_wrapped_npy = out_dir / "phase_wrapped_npy"
    out_unw_png    = out_dir / "phase_unwrapped_png"
    out_unw_npy    = out_dir / "phase_unwrapped_npy"
    out_quality    = out_dir / "quality_png"
    out_disp_png   = out_dir / "displacement_png"
    out_disp_npy   = out_dir / "displacement_npy"
    ensure_dir(out_wrapped_png); ensure_dir(out_wrapped_npy); ensure_dir(out_quality)
    if args.unwrap == "auto":
        ensure_dir(out_unw_png); ensure_dir(out_unw_npy)
    if args.kappa > 0:
        ensure_dir(out_disp_png); ensure_dir(out_disp_npy)

    stats_csv = out_dir / "phase_stats.csv"
    rows = [("rel_path","q_mean","q_std","phi_wrapped_circstd","phi_unwrapped_std","disp_std")]

    # Preload ROI if single global file; otherwise allow per-image ROI by same name
    global_roi = Path(args.roi_mask) if args.roi_mask else None

    for f in tqdm(files, desc="Phase extraction"):
        img = imread_uint01(f)
        H, W = img.shape

        # ROI: try image-specific mask with same relative path name first
        roi_mask = None
        if global_roi:
            # if folder, try to find matching filename; else use single mask
            if global_roi.is_dir():
                cand = global_roi / f.name
                if cand.exists():
                    roi_mask = load_roi_mask(cand, (H, W))
                else:
                    roi_mask = load_roi_mask(global_roi, (H, W))  # fallback try direct file (unlikely)
            else:
                roi_mask = load_roi_mask(global_roi, (H, W))

        # Flat-field
        x = img
        if args.flatfield > 0:
            x = flatfield_correct(x, sigma_lp=args.flatfield)

        # Takeda demod
        phi_wrapped, qviz = takeda_phase(x, band_sigma=args.band, center_suppress=args.center_suppress)

        # Save wrapped
        rel = f.relative_to(in_dir)
        p_wr_png = out_wrapped_png / rel
        p_wr_npy = out_wrapped_npy / rel.with_suffix(".npy")
        imsave_gray01((phi_wrapped + math.pi) / (2.0 * math.pi), p_wr_png)
        ensure_dir(p_wr_npy.parent); np.save(p_wr_npy, phi_wrapped.astype(np.float32))

        # Save quality
        p_q_png = out_quality / rel
        imsave_gray01(qviz, p_q_png)

        # Stats init
        q_mean = float(np.mean(qviz if roi_mask is None else qviz[roi_mask > 0]))
        q_std  = float(np.std(qviz if roi_mask is None else qviz[roi_mask > 0]))
        phi_cstd = circular_std(phi_wrapped, roi_mask)
        phi_unw_std = float("nan")
        disp_std = float("nan")

        # Optional unwrap
        if args.unwrap == "auto":
            unwrapped, msg = unwrap_if_available(phi_wrapped)
            if unwrapped is not None and np.all(np.isfinite(unwrapped)):
                # Normalize by removing mean over ROI
                if roi_mask is not None:
                    m = float(np.mean(unwrapped[roi_mask > 0]))
                    un = unwrapped - m
                else:
                    m = float(np.mean(unwrapped))
                    un = unwrapped - m
                # Save
                p_unw_png = out_unw_png / rel
                p_unw_npy = out_unw_npy / rel.with_suffix(".npy")
                # map for PNG viz: robust min/max
                lo, hi = np.percentile(un, [1.0, 99.0])
                un_viz = np.clip((un - lo) / (hi - lo + 1e-6), 0.0, 1.0)
                imsave_gray01(un_viz, p_unw_png)
                ensure_dir(p_unw_npy.parent); np.save(p_unw_npy, un.astype(np.float32))
                phi_unw_std = std_roi(un, roi_mask)

                # Displacement if kappa given
                if args.kappa > 0:
                    disp = un / args.kappa
                    p_dp_png = out_disp_png / rel
                    p_dp_npy = out_disp_npy / rel.with_suffix(".npy")
                    dlo, dhi = np.percentile(disp, [1.0, 99.0])
                    d_viz = np.clip((disp - dlo) / (dhi - dlo + 1e-6), 0.0, 1.0)
                    imsave_gray01(d_viz, p_dp_png)
                    ensure_dir(p_dp_npy.parent); np.save(p_dp_npy, disp.astype(np.float32))
                    disp_std = std_roi(disp, roi_mask)
            else:
                print(f"[unwrap] {f.name}: {msg}")

        rows.append((str(rel), f"{q_mean:.6f}", f"{q_std:.6f}",
                     f"{phi_cstd:.6f}", f"{phi_unw_std:.6f}", f"{disp_std:.6f}"))

    # Save CSV
    ensure_dir(stats_csv.parent)
    with open(stats_csv, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(rows)

    print(f"[DONE] Results saved under: {out_dir}")
    print(f" - Wrapped phase:   {out_wrapped_png}")
    if args.unwrap == 'auto':
        print(f" - Unwrapped phase: {out_unw_png}")
    print(f" - Quality maps:    {out_quality}")
    if args.kappa > 0:
        print(f" - Displacement:    {out_disp_png}")
    print(f" - Stats CSV:       {stats_csv}")

if __name__ == "__main__":
    main()
