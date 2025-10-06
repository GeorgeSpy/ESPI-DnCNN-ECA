# -*- coding: utf-8 -*-
"""
phase_extract_fft_STRICT_FIXED.py
---------------------------------
Strict FFT demodulation for ESPI fringe patterns (Takeda-style).

- Finds dominant side-lobe within an annulus (rmin..rmax), optionally prefer-right
- Builds a circular Hann band-pass around the peak (radius = --band)
- Shifts selected lobe to baseband and IFFT to get analytic signal
- Wrapped phase = angle(analytic); optional unwrapping (skimage if available, else 2-pass np.unwrap)
- Flat-field illumination correction via Gaussian (scipy.ndimage)
- ROI support (mask png, >0 = valid)
- Saves NPY (float32, radians) for wrapped & unwrapped, PNGs for previews only
- Saves a simple quality map from phase gradient magnitude (float32 NPY + PNG)

Usage (PowerShell/CMD):
python C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py ^
  --input-dir  "C:\path\to\frames" ^
  --output-dir "C:\path\to\PhaseOut" ^
  --band 18 ^
  --center-suppress 16 ^
  --annulus 6 320 ^
  --flatfield 60 ^
  --roi-mask "C:\path\to\roi_mask.png" ^
  --unwrap auto ^
  --prefer-right ^
  --debug 1
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Optional skimage unwrap
try:
    from skimage.restoration import unwrap_phase as sk_unwrap_phase
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


def imread_gray_float01(p: Path) -> np.ndarray:
    """Read PNG/TIF/etc as grayscale float32 in [0,1]."""
    img = Image.open(str(p)).convert("I")  # 32-bit int
    a = np.array(img, dtype=np.float32)
    # Heuristic normalisation
    mx = a.max()
    if mx <= 0:
        return np.zeros_like(a, dtype=np.float32)
    # If looks like 16-bit
    if mx > 255.0:
        a = a / 65535.0
    else:
        a = a / 255.0
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return a


def save_png_uint8(a01: np.ndarray, path: Path):
    """Save array in [0,1] to PNG uint8."""
    a = np.clip(a01, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(str(path))


def to_uint8_preview_phase(phase: np.ndarray) -> np.ndarray:
    """Map phase [-pi, pi] to [0,1] for preview."""
    z = (phase + math.pi) / (2.0 * math.pi)
    z = np.clip(z, 0.0, 1.0)
    return (z * 255.0).astype(np.uint8)


def fftshift2(a: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(a)


def ifftshift2(a: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(a)


def roll2(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def make_disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)


def make_annulus_mask(h, w, cy, cx, rmin, rmax):
    yy, xx = np.ogrid[:h, :w]
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (rr2 >= (rmin * rmin)) & (rr2 <= (rmax * rmax))


def circular_hann_band(h, w, cy, cx, py, px, radius):
    """
    Circular Hann mask centered at (py,px) in the shifted FFT grid
    with radius 'radius'.
    """
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - py) ** 2 + (xx - px) ** 2).astype(np.float32)
    R = float(radius)
    m = np.zeros((h, w), dtype=np.float32)
    inside = rr <= R
    # Hann taper from center to R:
    #  w(r) = 0.5 * (1 + cos(pi * r / R)) for r in [0, R]; 0 else
    m[inside] = 0.5 * (1.0 + np.cos(np.pi * rr[inside] / R)).astype(np.float32)
    return m


def find_peak_in_annulus(mag: np.ndarray, rmask: np.ndarray, prefer_right: bool) -> tuple[int, int]:
    """
    Find the (row,col) peak location inside rmask.
    If prefer_right=True, only search right half-plane (x > cx).
    """
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    cand = rmask.copy()
    if prefer_right:
        # keep only columns strictly to the right of center
        right_mask = np.zeros_like(cand, dtype=bool)
        right_mask[:, cx + 1 :] = True
        cand &= right_mask

    if not np.any(cand):
        # fallback to entire annulus
        cand = rmask

    masked = mag.copy()
    masked[~cand] = -np.inf
    flat_idx = np.argmax(masked)
    py, px = np.unravel_index(flat_idx, mag.shape)
    return int(py), int(px)


def unwrap_auto(phase_wrapped: np.ndarray) -> np.ndarray:
    """
    Try skimage.unwrap_phase; fall back to simple 2-pass np.unwrap.
    Returns float32.
    """
    if HAVE_SKIMAGE:
        try:
            out = sk_unwrap_phase(phase_wrapped.astype(np.float64))
            return out.astype(np.float32)
        except Exception:
            pass
    # Fallback: unwrap along x then y (simple, not perfect)
    p = np.unwrap(phase_wrapped, axis=1)
    p = np.unwrap(p, axis=0)
    return p.astype(np.float32)


def build_outputs(root: Path):
    (root / "phase_wrapped_npy").mkdir(parents=True, exist_ok=True)
    (root / "phase_wrapped_png").mkdir(parents=True, exist_ok=True)
    (root / "phase_unwrapped_npy").mkdir(parents=True, exist_ok=True)
    (root / "phase_unwrapped_png").mkdir(parents=True, exist_ok=True)
    (root / "quality_npy").mkdir(parents=True, exist_ok=True)
    (root / "quality_png").mkdir(parents=True, exist_ok=True)
    (root / "debug").mkdir(parents=True, exist_ok=True)


def save_debug_spectrum(Fsh: np.ndarray, mask: np.ndarray, out_debug: Path, stem: str):
    """Save log magnitude of spectrum and masked spectrum (grayscale)."""
    eps = 1e-12
    mag = np.log1p(np.abs(Fsh))
    mag = mag / (np.percentile(mag, 99.9) + eps)
    mag = np.clip(mag, 0.0, 1.0)
    save_png_uint8(mag, out_debug / f"{stem}_spectrum.png")

    mag_m = np.log1p(np.abs(Fsh * mask))
    mag_m = mag_m / (np.percentile(mag_m, 99.9) + eps)
    mag_m = np.clip(mag_m, 0.0, 1.0)
    save_png_uint8(mag_m, out_debug / f"{stem}_spectrum_masked.png")


def main():
    ap = argparse.ArgumentParser("FFT phase extraction (STRICT, FIXED save)")
    ap.add_argument("--input-dir", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=str)

    ap.add_argument("--band", type=int, default=18, help="Radius of circular Hann band around peak (px)")
    ap.add_argument("--center-suppress", type=int, default=16, help="Zero a disk at DC (px)")
    ap.add_argument("--annulus", type=int, nargs=2, default=[6, 320], help="Search ring [rmin rmax] (px)")
    ap.add_argument("--prefer-right", action="store_true", help="Search only right half-plane for the peak")

    ap.add_argument("--flatfield", type=float, default=0.0, help="Gaussian sigma for flat-field; 0=off")
    ap.add_argument("--roi-mask", type=str, default=None, help="PNG mask (>0 = valid). Must match image size")
    ap.add_argument("--unwrap", type=str, default="auto", choices=["auto", "none"], help="Unwrap strategy")
    ap.add_argument("--debug", type=int, default=0, help="Save spectrum debug images (0/1)")

    args = ap.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_wrapped_npy = out_dir / "phase_wrapped_npy"
    out_wrapped_png = out_dir / "phase_wrapped_png"
    out_unwrapped_npy = out_dir / "phase_unwrapped_npy"
    out_unwrapped_png = out_dir / "phase_unwrapped_png"
    quality_npy = out_dir / "quality_npy"
    quality_png = out_dir / "quality_png"
    out_debug = out_dir / "debug"

    build_outputs(out_dir)

    # Load ROI (if provided)
    roi = None
    if args.roi_mask:
        try:
            rimg = Image.open(args.roi_mask).convert("L")
            roi = (np.array(rimg) > 0)
        except Exception:
            roi = None

    files = sorted([p for p in in_dir.glob("*.png")])
    if not files:
        print(f"[ERR] No PNGs found under {in_dir}")
        return

    # Process each frame
    for p in tqdm(files, desc="Phase extraction"):
        stem = p.stem
        # 1) Load & pre-process
        x = imread_gray_float01(p)  # HxW float [0..1]
        h, w = x.shape
        cy, cx = h // 2, w // 2

        # Flat-field (illumination) if requested
        if args.flatfield and args.flatfield > 0:
            low = gaussian_filter(x, sigma=float(args.flatfield))
            low = np.clip(low, 1e-6, None)
            x = x / low
            # robust normalisation to [0,1]
            lo, hi = np.percentile(x, [0.5, 99.5])
            if hi > lo:
                x = (x - lo) / (hi - lo)
            x = np.clip(x, 0.0, 1.0).astype(np.float32)

        # DC removal
        x0 = x - float(x.mean())

        # 2) FFT
        F = np.fft.fft2(x0)
        Fsh = fftshift2(F)

        # 3) Center suppression
        if args.center_suppress > 0:
            cs = args.center_suppress
            mask_cs = ~make_disk_mask(h, w, cy, cx, cs)
            Fsh = Fsh * mask_cs

        # 4) Peak search within annulus
        rmin, rmax = args.annulus
        rmask = make_annulus_mask(h, w, cy, cx, rmin, rmax)
        mag = np.abs(Fsh)
        py, px = find_peak_in_annulus(mag, rmask, prefer_right=args.prefer_right)

        # 5) Build circular Hann band-pass around peak
        band = max(3, int(args.band))
        B = circular_hann_band(h, w, cy, cx, py, px, band).astype(np.float32)

        # 6) Shift masked lobe to baseband (center)
        S = Fsh * B
        dy, dx = (cy - py), (cx - px)
        S0 = roll2(S, dy, dx)  # now the selected lobe is centered
        S0 = ifftshift2(S0)
        analytic = np.fft.ifft2(S0)  # complex
        phase_wrapped = np.angle(analytic).astype(np.float32)

        # 7) Optional unwrap
        phi_unwrapped = None
        if args.unwrap == "auto":
            phi_unwrapped = unwrap_auto(phase_wrapped)

        # 8) Quality from gradient magnitude (robust normalisation)
        gx, gy = np.gradient(phase_wrapped)
        gradmag = np.hypot(gx, gy).astype(np.float32)
        scale = float(np.percentile(gradmag, 99.0)) + 1e-6
        q = np.clip(gradmag / scale, 0.0, 1.0).astype(np.float32)

        # Apply ROI to quality (zero outside)
        if roi is not None and roi.shape == q.shape:
            q = np.where(roi, q, 0.0).astype(np.float32)

        # 9) SAVE (NPY float32 + PNG preview)
        # Wrapped NPY (float32, radians)
        np.save(out_wrapped_npy / f"{stem}.npy", phase_wrapped.astype(np.float32))
        # Wrapped PNG preview
        Image.fromarray(to_uint8_preview_phase(phase_wrapped)).save(out_wrapped_png / f"{stem}.png")

        # Unwrapped (if available)
        if phi_unwrapped is not None:
            np.save(out_unwrapped_npy / f"{stem}.npy", phi_unwrapped.astype(np.float32))
            # preview: wrap back to [-pi,pi] just for visualisation
            prev = np.angle(np.exp(1j * phi_unwrapped)).astype(np.float32)
            Image.fromarray(to_uint8_preview_phase(prev)).save(out_unwrapped_png / f"{stem}.png")

        # Quality save
        np.save(quality_npy / f"{stem}.npy", q)
        save_png_uint8(q, quality_png / f"{stem}.png")

        # 10) Debug spectra if requested
        if args.debug and args.debug > 0:
            save_debug_spectrum(fftshift2(F), B, out_debug, stem)  # full spectrum (pre-suppression)
            # also masked spectrum around selected lobe
            save_debug_spectrum(fftshift2(F), rmask.astype(np.float32), out_debug, f"{stem}_annulus")

    print(f"[DONE] Results saved under: {out_dir}")
    print(f" - Wrapped phase:   {out_wrapped_png}")
    print(f" - Unwrapped phase: {out_unwrapped_png}")
    print(f" - Quality maps:    {quality_png}")
    print(f" - Stats CSV:       {out_dir / 'phase_stats.csv'} (not generated here; use your QC script)")
    # (Τα στατιστικά τα γράφει το phase_qc script σας)


if __name__ == "__main__":
    main()
