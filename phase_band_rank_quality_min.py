#!/usr/bin/env python3
# Rank bands by quality map + phase smoothness (optional) inside ROI.
# Looks under out-root for subfolders bandXX with quality_* and (optionally) phase_unwrapped_npy.

import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image

def read_gray01(path: Path):
    if path.suffix.lower()==".npy":
        return np.load(path).astype(np.float32)
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8).astype(np.float32)/255.0
    return arr

def read_phase(path: Path):
    if path.suffix.lower()==".npy":
        return np.load(path).astype(np.float32)
    im = Image.open(path).convert("L")
    arr01 = np.array(im, dtype=np.uint8).astype(np.float32)/255.0
    return arr01*(2*np.pi)-np.pi

def lap_mad(phi, valid):
    # 4-neighbour Laplacian
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    from scipy.signal import convolve2d
    L = convolve2d(phi, k, mode="same", boundary="symm")
    v = np.abs(L[valid])
    if v.size==0: return float("inf")
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    return float(mad)

def score_folder(band_dir: Path, roi_mask: Path, qmin: float):
    qdir_npy = band_dir/"quality_npy"
    qdir_png = band_dir/"quality_png"
    udir = band_dir/"phase_unwrapped_npy"

    # find files
    rels = []
    base_q = None
    if qdir_npy.exists():
        rels = [p.relative_to(qdir_npy) for p in qdir_npy.rglob("*.npy")]
        base_q = qdir_npy
    elif qdir_png.exists():
        rels = [p.relative_to(qdir_png) for p in qdir_png.rglob("*.png")]
        base_q = qdir_png
    else:
        return None

    if not rels: return None

    # ROI
    roi = None
    if roi_mask:
        try:
            roi = np.array(Image.open(roi_mask).convert("L"), dtype=np.uint8)
        except:
            roi = None

    med_qs = []
    pct_cov = []
    lap_mads = []

    for rel in rels:
        q = read_gray01(base_q/rel)
        H, W = q.shape
        valid = np.ones((H,W), dtype=bool)
        if roi is not None:
            r = roi
            if r.shape != (H,W):
                r = np.array(Image.open(roi_mask).convert("L").resize((W,H), Image.NEAREST))
            valid &= (r > 0)

        v = q[valid]
        if v.size==0: continue
        med_qs.append(float(np.median(v)))
        pct_cov.append(float(np.mean(v >= qmin)))

        # optional smoothness if unwrapped exists
        up = udir/rel.with_suffix(".npy")
        if up.exists():
            phi = read_phase(up)
            phi = phi[:H,:W]
            lap_mads.append(lap_mad(phi, valid))

    if not med_qs: return None

    m_med_q = float(np.mean(med_qs))
    m_cov   = float(np.mean(pct_cov))
    m_lap   = float(np.mean(lap_mads)) if lap_mads else float("inf")
    # final ranking key: higher quality & coverage, lower lap_mad
    return {"median_q": m_med_q, "coverage": m_cov, "lap_mad": m_lap}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="Folder that contains bandXX subfolders")
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--qmin", type=float, default=0.30)
    ap.add_argument("--csv", default="band_quality_rank.csv")
    args = ap.parse_args()

    root = Path(args.out_root)
    bands = [p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("band")]
    if not bands:
        print("[ERR] No bandXX folders found under", root); return

    rows = [("band","median_q","coverage_ge_qmin","lap_mad","rank_key")]
    scored = []
    for b in sorted(bands, key=lambda p: p.name):
        s = score_folder(b, args.roi_mask, args.qmin)
        if s is None:
            print("[WARN] skipping", b)
            continue
        # rank key: sort by (-median_q, -coverage, +lap_mad)
        key = (-s["median_q"], -s["coverage"], s["lap_mad"])
        scored.append((b.name, s["median_q"], s["coverage"], s["lap_mad"], key))

    if not scored:
        print("[ERR] No scored bands."); return

    scored.sort(key=lambda t: t[-1])
    for row in scored:
        rows.append((row[0], f"{row[1]:.4f}", f"{100*row[2]:.2f}%", f"{row[3]:.6f}", str(row[-1])))

    # write CSV + print best
    out_csv = root / args.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(rows)

    best = scored[0]
    print(f"[BEST by quality] {best[0]}  median_q={best[1]:.4f}  coverage>={args.qmin}={100*best[2]:.2f}%  lap_mad={best[3]:.6f}")
    print("CSV:", out_csv)

if __name__ == "__main__":
    main()
