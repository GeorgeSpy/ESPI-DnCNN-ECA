# -*- coding: utf-8 -*-
"""
Compose denoised CLEAN = NOISY - RESIDUAL
- Διαβάζει ζευγάρια από:
    --noisy    : φάκελο με pseudo-noisy ή real noisy (PNG)
    --residual : φάκελο με residuals που έβγαλε το μοντέλο (PNG)
- Αποθηκεύει CLEAN PNGs στον --out
- Προεπιλογή: ΑΠΛΟ lineαρικό clip 0..1 χωρίς “άσπρισμα”.
  (Για οπτικοποίηση μόνο, μπορείς να δώσεις --p-lo/--p-hi για μικρό stretch.)
"""
import argparse, os, glob
from pathlib import Path
import numpy as np
from PIL import Image

def imread_float01(p):
    im = Image.open(p).convert("L")
    a = np.asarray(im)
    if a.dtype == np.uint8:
        a = a.astype(np.float32)/255.0
    elif a.dtype == np.uint16:
        a = (a.astype(np.float32)/65535.0)
    else:
        a = a.astype(np.float32)
        a = (a - a.min())/(a.max()-a.min()+1e-8)
    return a

def imsave_uint801(a, path):
    a = np.clip(a, 0.0, 1.0)
    im = Image.fromarray((a*255.0+0.5).astype(np.uint8), mode="L")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)

def percentile_stretch(a, lo, hi):
    lo_v = np.percentile(a, lo); hi_v = np.percentile(a, hi)
    if hi_v <= lo_v: return np.clip(a, 0, 1)
    b = (a - lo_v) / (hi_v - lo_v)
    return np.clip(b, 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy",   required=True)
    ap.add_argument("--residual",required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--p-lo", type=float, default=None)
    ap.add_argument("--p-hi", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    noisy_list = sorted(glob.glob(os.path.join(args.noisy, "*.png")))
    assert noisy_list, f"No PNGs in {args.noisy}"
    done = 0
    for npath in noisy_list:
        name = os.path.basename(npath)
        rpath = os.path.join(args.residual, name)
        if not os.path.exists(rpath):
            print(f"[skip] residual not found for {name}")
            continue
        n = imread_float01(npath)
        r = imread_float01(rpath)
        if r.shape != n.shape:
            print(f"[skip] shape mismatch for {name}: noisy{n.shape} residual{r.shape}")
            continue
        clean = n - r   # το μοντέλο έβγαλε residual → denoised intensity
        # χωρίς υπερ-άπλωμα. Προαιρετικά μικρό stretch για οπτικοποίηση:
        if args.p_lo is not None and args.p_hi is not None:
            clean = percentile_stretch(clean, args.p_lo, args.p_hi)
        else:
            clean = np.clip(clean, 0.0, 1.0)
        outp = os.path.join(args.out, name)
        imsave_uint801(clean, outp)
        done += 1
        if args.limit and done >= args.limit: break
    print(f"[DONE] wrote {done} PNGs to: {args.out}")

if __name__ == "__main__":
    main()
