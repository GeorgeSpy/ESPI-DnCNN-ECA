# -*- coding: utf-8 -*-
import argparse, os, glob
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def robust_norm(img, p_lo=1.0, p_hi=99.5):
    v = np.percentile(img, [p_lo, p_hi])
    lo, hi = float(v[0]), float(v[1])
    if hi <= lo: hi = lo + 1e-6
    x = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return x

def read_gray_uint(path):
    im = Image.open(path)
    if im.mode not in ("L","I;16","I"):
        im = im.convert("L")
    a = np.array(im)
    if a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32) / 255.0
    return a

def save_u801(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(x,0,1)*255.0 + 0.5).astype(np.uint8))
    im.save(path)

def main():
    ap = argparse.ArgumentParser("Normalize ESPI PNGs with robust percentile scaling.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lo", type=float, default=1.0)
    ap.add_argument("--hi", type=float, default=99.5)
    args = ap.parse_args()

    inp = Path(args.input); out = Path(args.output)
    files = sorted(glob.glob(str(inp / "*.png")))
    if not files:
        print("[ERR] No PNGs found in", inp); return
    print(f"[INFO] files={len(files)}  p_lo={args.lo}  p_hi={args.hi}")

    for f in tqdm(files, desc="Normalize"):
        a = read_gray_uint(f)
        x = robust_norm(a, args.lo, args.hi)
        save_u801(x, out / Path(f).name)

    print("[DONE] Wrote normalized PNGs to:", out)

if __name__ == "__main__":
    main()
