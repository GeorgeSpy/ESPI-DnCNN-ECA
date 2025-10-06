#!/usr/bin/env python3
# Standalone QC: compares phase folders against a reference, saves CSV + edge overlays.
# Deps: numpy, pillow (PIL). No SciPy/OpenCV required.

import argparse, os, csv, math
from pathlib import Path
import numpy as np
from PIL import Image

def load_any(path: Path) -> np.ndarray:
    p = str(path).lower()
    if p.endswith(".npy"):
        arr = np.load(path)
        return arr.astype(np.float32)
    elif p.endswith(".png"):
        im = Image.open(path)
        a = np.array(im)
        if a.dtype == np.uint16:
            a = a.astype(np.float32) / 65535.0
        else:
            a = a.astype(np.float32)
            if a.max() > 0: a /= a.max()
        return a
    else:
        raise ValueError(f"Unsupported file: {path}")

def load_mask(mask_path: Path) -> np.ndarray:
    m = np.array(Image.open(mask_path).convert("L"))
    m = (m > 0).astype(np.float32)
    return m

def zscore(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = x[m > 0]
    if v.size < 10:
        return x*0.0
    mu = float(v.mean())
    sd = float(v.std()) if v.std() > 1e-9 else 1.0
    y = (x - mu) / sd
    return y

def cosine(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    av = a[m > 0].ravel()
    bv = b[m > 0].ravel()
    na = float(np.linalg.norm(av)) or 1.0
    nb = float(np.linalg.norm(bv)) or 1.0
    return float(np.dot(av, bv) / (na * nb))

def sobel_mag(x: np.ndarray) -> np.ndarray:
    # simple 3x3 Sobel without SciPy
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    Ky = Kx.T
    x = x.astype(np.float32)
    xpad = np.pad(x, ((1,1),(1,1)), mode="edge")
    gx = (
        Kx[0,0]*xpad[:-2,:-2] + Kx[0,1]*xpad[:-2,1:-1] + Kx[0,2]*xpad[:-2,2:] +
        Kx[1,0]*xpad[1:-1,:-2] + Kx[1,1]*xpad[1:-1,1:-1] + Kx[1,2]*xpad[1:-1,2:] +
        Kx[2,0]*xpad[2:,:-2] + Kx[2,1]*xpad[2:,1:-1] + Kx[2,2]*xpad[2:,2:]
    )
    gy = (
        Ky[0,0]*xpad[:-2,:-2] + Ky[0,1]*xpad[:-2,1:-1] + Ky[0,2]*xpad[:-2,2:] +
        Ky[1,0]*xpad[1:-1,:-2] + Ky[1,1]*xpad[1:-1,1:-1] + Ky[1,2]*xpad[1:-1,2:] +
        Ky[2,0]*xpad[2:,:-2] + Ky[2,1]*xpad[2:,1:-1] + Ky[2,2]*xpad[2:,2:]
    )
    mag = np.sqrt(gx*gx + gy*gy)
    if mag.max() > 0: mag = mag / mag.max()
    return mag

def pick_file(root: Path, stem: str):
    for ext in (".npy", ".png"):
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: search first match
    for p in root.glob(stem + ".*"):
        if p.suffix.lower() in (".npy",".png"):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--qmin", type=float, default=0.20)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ref_root = Path(args.ref_root)
    mask_path = Path(args.roi_mask)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mask = load_mask(mask_path)
    # build common stems
    out_stems = {p.stem for p in out_root.iterdir() if p.suffix.lower() in (".png",".npy")}
    ref_stems = {p.stem for p in ref_root.iterdir() if p.suffix.lower() in (".png",".npy")}
    stems = sorted(out_stems.intersection(ref_stems))

    csv_path = save_dir / "qc_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["name","q_score","sign","note"])

        passed = 0
        for stem in stems:
            pf = pick_file(out_root, stem)
            rf = pick_file(ref_root, stem)
            if pf is None or rf is None:
                wr.writerow([stem,"","",f"missing ({pf},{rf})"])
                continue

            a = load_any(pf)
            b = load_any(rf)

            # ensure same size
            if a.shape != b.shape:
                # try to center-crop to common min shape
                H = min(a.shape[0], b.shape[0])
                W = min(a.shape[1], b.shape[1])
                a = a[:H,:W]; b = b[:H,:W]; m = mask[:H,:W]
            else:
                m = mask

            az = zscore(a, m); bz = zscore(b, m)
            c1 = cosine(az, bz, m)
            c2 = cosine(az, -bz, m)
            q = max(c1, c2)
            sign = 1 if c1 >= c2 else -1

            # save edge overlay
            ea = sobel_mag(az) * m
            eb = sobel_mag(bz) * m
            rgb = np.zeros((ea.shape[0], ea.shape[1], 3), np.uint8)
            rgb[...,1] = (np.clip(ea,0,1)*255).astype(np.uint8)  # green: out
            rgb[...,0] = (np.clip(eb,0,1)*255).astype(np.uint8)  # red: ref
            Image.fromarray(rgb).save(save_dir / f"{stem}_edges.png")

            wr.writerow([stem, f"{q:.3f}", sign, ""])
            if q >= args.qmin: passed += 1

    print(f"[QC] Compared {len(stems)} common files. CSV: {csv_path}")
    print("Open the PNGs in the save-dir to visually verify nodal overlap (red=ref, green=out).")

if __name__ == "__main__":
    main()
