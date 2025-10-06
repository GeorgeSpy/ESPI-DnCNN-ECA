# -*- coding: utf-8 -*-
import argparse, os, glob, numpy as np
from pathlib import Path
from PIL import Image

def imread_f32(p):
    im = Image.open(p).convert("L")
    return np.asarray(im).astype(np.float32)/255.0

def imsave_uint01(x, out):
    x = np.clip(x, 0.0, 1.0)
    Image.fromarray((x*255.0+0.5).astype(np.uint8)).save(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True)    # folder PNG [0,1]
    ap.add_argument("--resnp", required=True)    # folder NPY raw residuals
    ap.add_argument("--out",   required=True)    # output CLEAN PNG
    ap.add_argument("--mode", choices=["minus","plus"], default="minus")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.noisy,"*.png")))
    if args.limit>0: files = files[:args.limit]

    for i,fp in enumerate(files,1):
        base = os.path.basename(fp).replace(".png",".npy")
        rpath = os.path.join(args.resnp, base)
        if not os.path.exists(rpath): 
            print("[MISS]", base); 
            continue
        n = imread_f32(fp)
        r = np.load(rpath).astype(np.float32)  # raw same scale as net output
        if args.mode == "minus":
            c = n - r
        else:
            c = n + r
        imsave_uint01(c, os.path.join(args.out, os.path.basename(fp)))
        if i<=3:
            print(f"[{i}] {os.path.basename(fp)}  noisy({n.min():.3f},{n.max():.3f}) resid({r.min():.3f},{r.max():.3f})  -> clean({c.min():.3f},{c.max():.3f})")
    print("[DONE] wrote:", args.out)

if __name__ == "__main__":
    main()
