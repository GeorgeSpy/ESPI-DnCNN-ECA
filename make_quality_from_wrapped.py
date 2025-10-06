# -*- coding: utf-8 -*-
"""
Rebuild quality maps from wrapped phase using local phase coherence:
  Q = | mean_window( exp(i*phi) ) |
Pure NumPy implementation (no OpenCV/SciPy). Saves NPY + PNG.
"""
import os, glob
import numpy as np
from pathlib import Path
from PIL import Image

def robust_scale01(x, p_lo=1.0, p_hi=99.0):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi <= lo:
        return np.clip(x, 0, 1)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def movmean1d(a, k, axis):
    """Moving average 1D με reflected padding, καθαρό NumPy."""
    r = k // 2
    pad_width = [(0,0)] * a.ndim
    pad_width[axis] = (r, r)
    ap = np.pad(a, pad_width, mode='reflect')
    # cumsum με leading zero ώστε τα παράθυρα να είναι Lp-k+1 = L
    c = np.cumsum(ap, axis=axis, dtype=np.float64)
    shape0 = list(c.shape)
    shape0[axis] = 1
    z = np.zeros(shape0, dtype=c.dtype)
    c = np.concatenate([z, c], axis=axis)
    # window sums: c[i+k]-c[i]
    slicer_hi = [slice(None)] * a.ndim
    slicer_lo = [slice(None)] * a.ndim
    slicer_hi[axis] = slice(k, k + a.shape[axis])
    slicer_lo[axis] = slice(0, a.shape[axis])
    s = c[tuple(slicer_hi)] - c[tuple(slicer_lo)]
    return (s / float(k)).astype(np.float32)

def box_mean(img, k):
    """2D box mean: separable moving average (rows -> cols)."""
    out = movmean1d(img, k, axis=1)  # κατά πλάτος
    out = movmean1d(out, k, axis=0)  # κατά ύψος
    return out

def main():
    out_root = r"C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_PhaseOut_STRICT_DEN_CLEAN"
    wrap_dir = Path(out_root) / "phase_wrapped_npy"
    q_png    = Path(out_root) / "quality_png"
    q_npy    = Path(out_root) / "quality_npy"
    q_png.mkdir(parents=True, exist_ok=True)
    q_npy.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(wrap_dir / "*.npy")))
    if not files:
        print("[ERR] No wrapped npy in:", wrap_dir); return

    win = 11  # window size (odd, π.χ. 7/9/11)
    for fp in files:
        name = Path(fp).stem
        phi  = np.load(fp).astype(np.float32)  # radians
        # τοπική κυκλική συνοχή:
        z_re = np.cos(phi); z_im = np.sin(phi)
        mu_re = box_mean(z_re, win)
        mu_im = box_mean(z_im, win)
        Q = np.sqrt(mu_re*mu_re + mu_im*mu_im)  # [0..1]

        np.save(q_npy / f"{name}.npy", Q.astype(np.float32))
        vis = (robust_scale01(Q, 1.0, 99.0) * 255.0).astype(np.uint8)
        Image.fromarray(vis).save(q_png / f"{name}.png")

    print(f"[OK] Wrote {len(files)} quality maps to:\n - {q_npy}\n - {q_png}")

if __name__ == "__main__":
    main()
