# -*- coding: utf-8 -*-
import argparse, os, glob
import numpy as np
from pathlib import Path
from PIL import Image

def read_f32_gray(p):
    im = Image.open(p).convert('F')
    return np.array(im, dtype=np.float32)

def write_u8(dst, x):
    x8 = np.clip(np.round(x*255.0), 0, 255).astype(np.uint8)
    Image.fromarray(x8, mode='L').save(dst)

def percentile_norm(x, lo=0.5, hi=99.5):
    a = np.percentile(x, lo); b = np.percentile(x, hi)
    if b <= a: b = a + 1e-6
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0), a, b

def unpercentile(y, a, b):
    return y*(b - a) + a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noisy', required=True)
    ap.add_argument('--residual', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--p_lo', type=float, default=0.5)
    ap.add_argument('--p_hi', type=float, default=99.5)
    ap.add_argument('--mode', choices=['zero','pos'], default='zero',
                    help="Interpretation του RES PNG: 'zero' => (res-0.5)*2, 'pos' => res")
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    noisy_dir = Path(args.noisy)
    res_dir   = Path(args.residual)
    out_dir   = Path(args.out); (out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(noisy_dir/'*.png')))
    if args.limit>0:
        files = files[:args.limit]

    for i, p_noisy in enumerate(files, 1):
        name = Path(p_noisy).name
        p_res = res_dir/name
        if not p_res.exists():
            print(f"[SKIP] residual missing for {name}")
            continue

        x = read_f32_gray(p_noisy)     # NOISY σε [0..255] -> float
        r = read_f32_gray(p_res) / 255.0  # RES PNG -> [0..1] float

        # 1) normalize NOISY (ίδιο με inference)
        xn, a, b = percentile_norm(x, args.p_lo, args.p_hi)

        # 2) interpret residual map
        if args.mode == 'zero':
            # μηδενικό στο 0.5, εύρος [-1,1]
            r_signed = (r - 0.5) * 2.0
        else:
            # θετικό residual στο [0,1]
            r_signed = r

        # 3) compose στον normalized χώρο (DnCNN προβλέπει noise)
        clean_n = np.clip(xn - r_signed, 0.0, 1.0)

        # 4) unnormalize πίσω στην ένταση
        clean = unpercentile(clean_n, a, b)
        clean01 = np.clip(clean / 255.0, 0.0, 1.0)

        write_u8(out_dir/name, clean01)
        if i % 25 == 0:
            print(f"[{i}/{len(files)}] {name}")

    print(f"[DONE] Wrote CLEAN to: {out_dir}")

if __name__ == '__main__':
    main()
