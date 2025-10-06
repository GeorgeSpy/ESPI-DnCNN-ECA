#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
phase_qc_roi_FIXED.py
---------------------
QC for FFT-based phase extraction with optional ROI mask and quality threshold.
- Re-wrap check: residual = wrap( phi_wrapped - wrap(phi_unwrapped) )
- Metrics on *valid* pixels only:
    * RMSE(rad)
    * %>|pi/4|
    * %>|pi/2|
- Options:
    --roi-mask <PNG>  : use white(=255) as valid, black(=0) ignored
    --qmin 0.0..1.0   : ignore pixels with quality < qmin (if quality maps exist)

Saves CSV and optional residual heatmaps (|residual|).
Assumes wrapped/unwrapped/quality are under <out-root>/phase_*_{npy|png}.
"""

import argparse, numpy as np, math, csv
from pathlib import Path
from PIL import Image

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi

def load_npy_or_png(path_npy: Path, path_png: Path, scale_png_to_rad: bool) -> np.ndarray:
    if path_npy.exists():
        return np.load(path_npy).astype(np.float32)
    if path_png.exists():
        arr01 = np.array(Image.open(path_png).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
        if scale_png_to_rad:
            return arr01 * 2*np.pi - np.pi  # assume encoding [-pi,pi] -> [0,1]
        else:
            return arr01
    raise FileNotFoundError(f"Missing both {path_npy} and {path_png}")

def list_relatives(root: Path, exts):
    return sorted([p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--csv", default="phase_qc_roi.csv")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--roi-mask", default="", help="PNG mask (255 valid, 0 ignore)")
    ap.add_argument("--qmin", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.out_root)
    wpng = root / "phase_wrapped_png"
    wnpy = root / "phase_wrapped_npy"
    upng = root / "phase_unwrapped_png"
    unpy = root / "phase_unwrapped_npy"
    qpng = root / "quality_png"
    qnpy = root / "quality_npy"

    # build reference list from wrapped dirs (prefer PNG)
    rels = []
    base_w = None
    if wpng.exists():
        rels = list_relatives(wpng, {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
        base_w = wpng
    elif wnpy.exists():
        rels = list_relatives(wnpy, {".npy"})
        base_w = wnpy
    if not rels:
        print("[ERR] No wrapped files found under", wpng, "or", wnpy)
        return

    # load ROI if provided
    roi = None
    if args.roi_mask:
        m = Image.open(args.roi_mask).convert("L")
        roi = np.array(m, dtype=np.uint8)

    rows = [("name","n_valid","rmse_rad","p_gt_pi4","p_gt_pi2")]
    acc = {"n":0, "rmse":0.0, "p_pi4":0.0, "p_pi2":0.0, "valid":0}
    maps_dir = root / "qc_residual_maps_roi"
    if args.save_maps: ensure_dir(maps_dir)

    for rel in rels:
        name = rel.stem
        try:
            phi_w = load_npy_or_png((unpy.parent.parent / "phase_wrapped_npy" / rel).with_suffix(".npy"),
                                    (upng.parent.parent / "phase_wrapped_png" / rel).with_suffix(".png"),
                                    True)
            phi_u = load_npy_or_png((unpy / rel).with_suffix(".npy"),
                                    (upng / rel).with_suffix(".png"),
                                    True)
        except FileNotFoundError:
            # skip if unwrapped missing
            continue

        H = min(phi_w.shape[0], phi_u.shape[0]); W = min(phi_w.shape[1], phi_u.shape[1])
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]
        # masks
        valid = np.ones((H,W), dtype=bool)
        if roi is not None:
            if (roi.shape[0], roi.shape[1]) != (H,W):
                roi_r = np.array(Image.fromarray(roi, mode="L").resize((W,H), resample=Image.NEAREST))
            else:
                roi_r = roi
            valid &= (roi_r > 0)
        # quality filter
        if qpng.exists() or qnpy.exists():
            try:
                q = load_npy_or_png((qnpy / rel).with_suffix(".npy"), (qpng / rel).with_suffix(".png"), False)
                q = q[:H,:W]
                if args.qmin > 0.0:
                    valid &= (q >= float(args.qmin))
            except Exception:
                pass

        if valid.sum() < 100:
            continue  # too few pixels

        phi_uw = wrap_pi(phi_u)
        resid = wrap_pi(phi_w - phi_uw)
        r = resid[valid]; a = np.abs(r)

        rmse = float(np.sqrt(np.mean(r**2)))
        p_pi4 = float(np.mean(a > (np.pi/4)))
        p_pi2 = float(np.mean(a > (np.pi/2)))

        rows.append((name, int(valid.sum()), f"{rmse:.6f}", f"{p_pi4:.6f}", f"{p_pi2:.6f}"))
        acc["n"] += 1; acc["rmse"] += rmse; acc["p_pi4"] += p_pi4; acc["p_pi2"] += p_pi2; acc["valid"] += int(valid.sum())

        if args.save_maps:
            m = np.zeros((H,W), dtype=np.float32)
            vals = np.clip(np.abs(resid)/np.pi, 0.0, 1.0)
            m[valid] = vals[valid]  # fixed: don't index the 1-D array
            Image.fromarray((m*255.0+0.5).astype(np.uint8), mode="L").save((maps_dir / name).with_suffix(".png"))

    out_csv = root / args.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(rows)

    if acc["n"] == 0:
        print("[WARN] No frames evaluated (check unwrapped presence / ROI size / qmin).")
        print("CSV written to:", out_csv)
        return

    n = acc["n"]
    print(f"[SUMMARY over {n} image(s)] (mean over valid pixels)")
    print(f"  RMSE(rad): {acc['rmse']/n:.4f}")
    print(f"  >pi/4:     {100.0*acc['p_pi4']/n:.2f}%")
    print(f"  >pi/2:     {100.0*acc['p_pi2']/n:.2f}%")
    print(f"  Mean valid px/frame: {acc['valid']/n:.0f}")
    print("CSV saved to:", out_csv)
    if args.save_maps:
        print("Residual maps saved to:", maps_dir)

if __name__ == "__main__":
    main()
