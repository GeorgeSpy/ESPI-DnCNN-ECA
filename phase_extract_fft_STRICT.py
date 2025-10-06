#!/usr/bin/env python3
# ASCII-only, Windows-safe
# Robust FFT-based phase extractor for ESPI fringe patterns.
# - Locks to RIGHT side-lobe (kx > 0) when --prefer-right is set
# - Suppresses DC with a center disk
# - Finds peak in an annulus and applies a circular band-pass around it
# - Recenters the side-lobe to baseband (frequency roll) and IFFT -> analytic signal
# - Outputs wrapped phase, (optional) unwrapped (requires scikit-image), and quality map
# - Removes global phase offset (circular mean) inside ROI
# - Saves debug spectra & masks for the first N frames (if --debug > 0)

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def imread_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        im = im.convert("L")
        arr = np.array(im)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        m = float(arr.max())
        if m > 0:
            arr = arr / m
    return arr

def save_phase_png(phi: np.ndarray, path: Path):
    w = wrap_pi(phi)
    arr01 = (w + np.pi) / (2.0*np.pi)
    arr = (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)

def save_gray01_png(x01: np.ndarray, path: Path):
    arr = (np.clip(x01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0*np.pi) - np.pi

def hann2d(h, w):
    wx = 0.5 - 0.5*np.cos(2.0*np.pi*np.arange(w)/float(w))
    wy = 0.5 - 0.5*np.cos(2.0*np.pi*np.arange(h)/float(h))
    return np.outer(wy, wx).astype(np.float32)

def circular_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy)**2 + (xx - cx)**2) <= (r*r)

def annulus_mask(h, w, cy, cx, rmin, rmax):
    yy, xx = np.ogrid[:h, :w]
    rr2 = (yy - cy)**2 + (xx - cx)**2
    return (rr2 >= rmin*rmin) & (rr2 <= rmax*rmax)

def find_peak_right(mag, cy, cx, rmin, rmax):
    h, w = mag.shape
    m = annulus_mask(h, w, cy, cx, rmin, rmax)
    # right half-plane only
    cols = np.arange(w)[None, :]
    m &= (cols > cx)
    sel = mag * m
    idx = np.argmax(sel)
    py, px = np.unravel_index(idx, mag.shape)
    return int(py), int(px), float(mag[py, px])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--band", type=int, default=22, help="radius (px) of circular band-pass around side-lobe")
    ap.add_argument("--center-suppress", type=int, default=16, help="radius (px) to zero at the DC center")
    ap.add_argument("--annulus", type=int, nargs=2, default=[8, 300], help="min max radii to search the side-lobe")
    ap.add_argument("--prefer-right", action="store_true", help="enforce right side-lobe (kx>0)")
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--flatfield", type=int, default=0, help="radius (px) for low-pass background; 0=off")
    ap.add_argument("--unwrap", default="auto", choices=["auto","off"], help="use skimage.unwrap_phase if available")
    ap.add_argument("--debug", type=int, default=0, help="save debug spectra/masks for first N frames")
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    wpng = out / "phase_wrapped_png"; wpng.mkdir(exist_ok=True)
    wnpy = out / "phase_wrapped_npy"; wnpy.mkdir(exist_ok=True)
    upng = out / "phase_unwrapped_png"; upng.mkdir(exist_ok=True)
    unpy = out / "phase_unwrapped_npy"; unpy.mkdir(exist_ok=True)
    qpng = out / "quality_png"; qpng.mkdir(exist_ok=True)
    qnpy = out / "quality_npy"; qnpy.mkdir(exist_ok=True)
    dbg  = out / "debug"; dbg.mkdir(exist_ok=True)

    # ROI mask (white=valid)
    roi = None
    if args.roi_mask:
        try:
            roi = np.array(Image.open(args.roi_mask).convert("L"), dtype=np.uint8)
        except Exception as e:
            print("[WARN] failed to read ROI:", e)
            roi = None

    # unwrap availability
    do_unwrap = (args.unwrap == "auto")
    if do_unwrap:
        try:
            from skimage.restoration import unwrap_phase
        except Exception:
            print("[WARN] scikit-image not available, unwrap disabled")
            do_unwrap = False

    # list images
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    files = sorted([p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not files:
        print("[ERR] no images in", inp)
        return

    saved = 0
    for i, f in enumerate(tqdm(files, desc="Phase extraction")):
        x = imread_gray(f)
        h, w = x.shape
        cy, cx = h//2, w//2

        # optional flat-field (FFT low-pass background subtraction)
        if args.flatfield > 0:
            X = np.fft.fftshift(np.fft.fft2(x))
            low = np.zeros_like(X)
            cm = circular_mask(h, w, cy, cx, args.flatfield)
            low[cm] = X[cm]
            bg = np.fft.ifft2(np.fft.ifftshift(low))
            x = np.real(x - bg).astype(np.float32)
            lo = np.percentile(x, 1.0); hi = np.percentile(x, 99.0)
            if hi > lo:
                x = (x - lo) / (hi - lo)
            x = np.clip(x, 0.0, 1.0)

        # Hann apodization
        x = x * hann2d(h, w)

        # FFT
        F = np.fft.fftshift(np.fft.fft2(x))
        mag = np.abs(F)

        # suppress DC
        if args.center_suppress > 0:
            m0 = circular_mask(h, w, cy, cx, args.center_suppress)
            F[m0] = 0.0
            mag[m0] = 0.0

        # peak search
        rmin, rmax = args.annulus
        if args.prefer_right:
            py, px, _ = find_peak_right(mag, cy, cx, rmin, rmax)
        else:
            m = annulus_mask(h, w, cy, cx, rmin, rmax)
            sel = mag * m
            idx = np.argmax(sel)
            py, px = np.unravel_index(idx, mag.shape)
            if px <= cx:
                px = w - px  # mirror to right

        # circular band-pass around peak
        band = int(args.band)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - py)**2 + (xx - px)**2) <= (band*band)
        Fbp = np.zeros_like(F)
        Fbp[mask] = F[mask]

        # recenter side-lobe to origin (frequency roll)
        sh_y = cy - py
        sh_x = cx - px
        Fcen = np.roll(np.roll(Fbp, sh_y, axis=0), sh_x, axis=1)

        # analytic signal
        analytic = np.fft.ifft2(np.fft.ifftshift(Fcen))
        comp = analytic.astype(np.complex64)
        phase_wrapped = np.angle(comp).astype(np.float32)
        amp = np.abs(comp).astype(np.float32)

        # quality normalization (99th percentile), optionally inside ROI
        if roi is not None and roi.shape == (h, w):
            vv = amp[roi > 0]
            p99 = np.percentile(vv, 99.0) if vv.size else 1.0
        else:
            p99 = np.percentile(amp, 99.0)
        q = np.clip(amp / max(p99, 1e-6), 0.0, 1.0).astype(np.float32)

        # unwrap (optional)
if do_unwrap:
    from skimage.restoration import unwrap_phase
    # some skimage versions don't support seed=...
    try:
        phi_u = unwrap_phase(phase_wrapped, seed=None)  # newer versions
    except TypeError:
        phi_u = unwrap_phase(phase_wrapped)             # older versions
else:
    phi_u = phase_wrapped.copy()


        # remove global offset using circular mean inside ROI (or all)
        valid = (roi > 0) if (roi is not None and roi.shape == (h, w)) else np.ones_like(phase_wrapped, dtype=bool)
        diff = wrap_pi(phase_wrapped - wrap_pi(phi_u))
        z = np.exp(1j * diff[valid])
        c = float(np.angle(z.mean())) if z.size else 0.0
        phi_u = phi_u + c

        # save
        stem = f.stem
        np.save((wnpy / f"{stem}.npy"), phase_wrapped.astype(np.float32))
        save_phase_png(phase_wrapped, (wpng / f"{stem}.png"))
        np.save((unpy / f"{stem}.npy"), phi_u.astype(np.float32))
        save_phase_png(phi_u, (upng / f"{stem}.png"))
        np.save((qnpy / f"{stem}.npy"), q.astype(np.float32))
        save_gray01_png(q, (qpng / f"{stem}.png"))

        # debug spectra/mask for first N frames
        if args.debug > 0 and i < args.debug:
            sm = np.log1p(np.abs(F))
            sm = sm / (np.percentile(sm, 99.0) + 1e-6)
            save_gray01_png(sm, (dbg / f"{stem}_spectrum.png"))
            msk = np.zeros((h, w), np.float32); msk[mask] = 1.0
            save_gray01_png(msk, (dbg / f"{stem}_bandmask.png"))

        saved += 1

    meta = {
        "input_dir": str(inp),
        "output_dir": str(out),
        "band": int(args.band),
        "center_suppress": int(args.center_suppress),
        "annulus": [int(args.annulus[0]), int(args.annulus[1])],
        "prefer_right": bool(args.prefer_right),
        "flatfield": int(args.flatfield),
        "unwrap": args.unwrap,
        "frames": int(saved),
    }
    with open(out/"extract_meta.json", "w", encoding="utf-8") as fo:
        json.dump(meta, fo, indent=2)
    print(f"[DONE] Results saved under: {out}")
    print(" - Wrapped phase:  ", wpng)
    print(" - Unwrapped phase:", upng)
    print(" - Quality maps:   ", qpng)
    print(" - Stats JSON:     ", out/"extract_meta.json")

if __name__ == "__main__":
    main()
