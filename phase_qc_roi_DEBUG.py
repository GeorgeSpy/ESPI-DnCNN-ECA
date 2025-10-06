#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
phase_qc_roi_DEBUG.py
---------------------
Verbose QC to diagnose why no frames are evaluated.
Prints per-frame info (first N frames):
  - found wrapped/unwrapped/quality?
  - shapes, ROI match/resized
  - valid pixels after ROI and after qmin

Usage:
python phase_qc_roi_DEBUG.py ^
  --out-root "C:\...\band28" ^
  --roi-mask "C:\...\roi_mask.png" ^
  --qmin 0.30 --max-print 15
"""
import argparse, numpy as np
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--qmin", type=float, default=0.0)
    ap.add_argument("--max-print", type=int, default=10)
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
    if wpng.exists():
        rels = list_relatives(wpng, {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
        base_w = wpng
    elif wnpy.exists():
        rels = list_relatives(wnpy, {".npy"})
        base_w = wnpy
    else:
        print("[ERR] No wrapped files found.")
        return

    if not rels:
        print("[ERR] Wrapped dir exists but contains no files.")
        return

    roi = None
    if args.roi_mask:
        try:
            roi = np.array(Image.open(args.roi_mask).convert("L"), dtype=np.uint8)
        except Exception as e:
            print("[WARN] Failed to read ROI:", e)

    printed = 0
    ok_count = 0
    for rel in rels:
        name = rel.stem
        # find unwrapped counterpart
        w_npy = (wnpy / rel).with_suffix(".npy")
        w_png = (wpng / rel).with_suffix(".png")
        u_npy = (unpy / rel).with_suffix(".npy")
        u_png = (upng / rel).with_suffix(".png")
        q_npy = (qnpy / rel).with_suffix(".npy")
        q_png = (qpng / rel).with_suffix(".png")

        # choose which wrapped exists
        w_path = w_npy if w_npy.exists() else w_png if w_png.exists() else None
        u_path = u_npy if u_npy.exists() else u_png if u_png.exists() else None
        q_path = q_npy if q_npy.exists() else q_png if q_png.exists() else None

        if printed < args.max_print:
            print(f"\n[FILE] {name}")
            print("  wrapped exists:", w_path is not None, "->", w_path)
            print("  unwrapped exists:", u_path is not None, "->", u_path)
            print("  quality exists:", q_path is not None, "->", q_path)

        if w_path is None or u_path is None:
            continue

        phi_w = load_npy_or_png(w_npy, w_png, True)
        phi_u = load_npy_or_png(u_npy, u_png, True)

        H = min(phi_w.shape[0], phi_u.shape[0]); W = min(phi_w.shape[1], phi_u.shape[1])
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]

        valid = np.ones((H,W), dtype=bool)
        if roi is not None:
            roi_r = roi
            if roi_r.shape != (H,W):
                roi_r = np.array(Image.fromarray(roi, mode="L").resize((W,H), resample=Image.NEAREST))
            valid &= (roi_r > 0)

        if q_path is not None:
            q = load_npy_or_png(q_npy, q_png, False)
            q = q[:H,:W]
            if args.qmin > 0.0:
                valid &= (q >= float(args.qmin))

        if printed < args.max_print:
            print(f"  shape: {H}x{W}, valid after ROI/qmin: {int(valid.sum())} px ({100.0*valid.mean():.2f}%)")

        if valid.sum() >= 100:
            ok_count += 1
        printed += 1

    print(f"\n[SUMMARY] Frames with >=100 valid px: {ok_count} / {len(rels)}")

if __name__ == "__main__":
    main()
