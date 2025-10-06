#!/usr/bin/env python3
# ASCII-only, Windows-safe
# Extract nodal/phase features from unwrapped phase maps for mode classification.
# - Uses ROI + quality mask (q >= qmin)
# - Nodal proxy via |phi| < tau_zero and sign-change counts horizontally/vertically/diagonally
# - Gradient & Laplacian (NumPy-only; no SciPy/OpenCV required)
# Outputs a CSV with one row per frame.

import argparse, csv, re
from pathlib import Path
import numpy as np
from PIL import Image

def read_phase(path: Path):
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    im = Image.open(path).convert("L")
    arr01 = np.array(im, dtype=np.uint8).astype(np.float32) / 255.0
    return (arr01 * (2.0*np.pi) - np.pi).astype(np.float32)

def read_gray01(path: Path):
    if path.suffix.lower() == ".npy":
        x = np.load(path).astype(np.float32)
        return np.clip(x, 0.0, 1.0)
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8).astype(np.float32) / 255.0
    return arr

def roll_sum4(x: np.ndarray) -> np.ndarray:
    return (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) +
            np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1))

def laplacian_np(phi: np.ndarray) -> np.ndarray:
    return roll_sum4(phi) - 4.0 * phi

def grad_mag(phi: np.ndarray) -> np.ndarray:
    gx = 0.5 * (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))
    gy = 0.5 * (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))
    return np.sqrt(gx * gx + gy * gy)

def count_sign_changes(sign_map: np.ndarray, valid: np.ndarray):
    s = sign_map
    nz = (s != 0) & valid
    # Horizontal
    s1 = s[:, :-1]; s2 = s[:, 1:]
    v1 = nz[:, :-1] & nz[:, 1:]
    ch_h = int(np.sum((s1 * s2 == -1) & v1))
    # Vertical
    s1 = s[:-1, :]; s2 = s[1:, :]
    v1 = nz[:-1, :] & nz[1:, :]
    ch_v = int(np.sum((s1 * s2 == -1) & v1))
    # Main diagonal
    s1 = s[:-1, :-1]; s2 = s[1:, 1:]
    v1 = nz[:-1, :-1] & nz[1:, 1:]
    ch_d1 = int(np.sum((s1 * s2 == -1) & v1))
    # Anti-diagonal
    s1 = s[:-1, 1:]; s2 = s[1:, :-1]
    v1 = nz[:-1, 1:] & nz[1:, :-1]
    ch_d2 = int(np.sum((s1 * s2 == -1) & v1))
    return ch_h, ch_v, ch_d1, ch_d2

def extract_freq(stem: str) -> int:
    m = re.search(r'(\d+)\s*Hz', stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-root", required=True, help="Folder like .../W01_PhaseOut_STRICT/bandXX")
    ap.add_argument("--roi-mask", default="", help="Optional ROI PNG (white=valid)")
    ap.add_argument("--qmin", type=float, default=0.30, help="Quality threshold")
    ap.add_argument("--tau-zero", type=float, default=0.25, help="Nodal band threshold in radians")
    ap.add_argument("--csv-out", default="nodal_features.csv")
    args = ap.parse_args()

    root = Path(args.band_root)
    unp = root / "phase_unwrapped_npy"
    qnp = root / "quality_npy"
    if not unp.exists():
        print("[ERR] phase_unwrapped_npy not found:", unp); return
    if not qnp.exists():
        print("[ERR] quality_npy not found:", qnp); return

    # ROI (optional)
    roi = None
    if args.roi_mask:
        try:
            roi = np.array(Image.open(args.roi_mask).convert("L"), dtype=np.uint8)
        except Exception as e:
            print("[WARN] failed to read ROI:", e); roi = None

    files = sorted(unp.rglob("*.npy"))
    if not files:
        print("[ERR] no .npy under", unp); return

    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        wr = csv.writer(fo)
        wr.writerow(["id","freq_hz","valid_px","zero_frac","chg_h","chg_v","chg_d1","chg_d2",
                     "grad_mean","grad_std","lap_mad","phase_std"])

        for p in files:
            phi = read_phase(p)  # radians
            stem = p.stem
            freq = extract_freq(stem)

            # quality (match shape)
            q = read_gray01((qnp / p.name))
            H = min(phi.shape[0], q.shape[0]); W = min(phi.shape[1], q.shape[1])
            phi = phi[:H, :W]; q = q[:H, :W]

            # ROI/valid
            valid = np.ones((H, W), dtype=bool)
            if roi is not None:
                rr = roi
                if rr.shape != (H, W):
                    rr = np.array(Image.open(args.roi_mask).convert("L").resize((W, H), Image.NEAREST))
                valid &= (rr > 0)
            valid &= (q >= float(args.qmin))
            n_valid = int(valid.sum())
            if n_valid < 100:
                wr.writerow([stem, freq, n_valid, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                continue

            # zero-band and sign map
            tz = float(args.tau_zero)
            zero_band = (np.abs(phi) < tz) & valid
            zero_frac = float(np.mean(zero_band))

            sign_map = np.zeros_like(phi, dtype=np.int8)
            sign_map[phi >  tz] =  1
            sign_map[phi < -tz] = -1

            ch_h, ch_v, ch_d1, ch_d2 = count_sign_changes(sign_map, valid)

            # gradient & laplacian stats (valid-only)
            g = grad_mag(phi)
            gv = g[valid]; grad_mean = float(np.mean(gv)); grad_std = float(np.std(gv))

            L = laplacian_np(phi)
            Lv = np.abs(L[valid])
            med = float(np.median(Lv))
            lap_mad = float(np.median(np.abs(Lv - med)) + 1e-9)

            phase_std = float(np.std(phi[valid]))

            wr.writerow([stem, freq, n_valid, f"{zero_frac:.6f}",
                         ch_h, ch_v, ch_d1, ch_d2,
                         f"{grad_mean:.6f}", f"{grad_std:.6f}", f"{lap_mad:.6f}", f"{phase_std:.6f}"])

    print("[DONE] CSV written to:", out_csv)

if __name__ == "__main__":
    main()
