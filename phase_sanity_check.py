# -*- coding: utf-8 -*-
import argparse, os, glob
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser("Quick sanity check for phase outputs")
    ap.add_argument("--root", required=True, help="PhaseOut folder (the parent that contains phase_*_npy)")
    ap.add_argument("--roi-mask", default=None, help="Optional ROI png (>0 valid)")
    ap.add_argument("--k", type=int, default=3, help="How many samples to print")
    args = ap.parse_args()

    root = Path(args.root)
    wdir = root / "phase_wrapped_npy"
    udir = root / "phase_unwrapped_npy"
    qdir = root / "quality_npy"

    wf = sorted(glob.glob(str(wdir/"*.npy")))
    uf = sorted(glob.glob(str(udir/"*.npy")))
    qf = sorted(glob.glob(str(qdir/"*.npy")))
    print(f"[FOUND] wrapped={len(wf)} unwrapped={len(uf)} quality={len(qf)}")

    roi = None
    if args.roi_mask and Path(args.roi_mask).exists():
        m = Image.open(args.roi_mask).convert("L")
        roi = (np.array(m) > 0)

    def stat(name, a, mask=None):
        if mask is not None and mask.shape == a.shape:
            a = a[mask]
        return (float(a.min()), float(a.max()), float(a.mean()), float(a.std()))

    if len(wf)==0:
        print("[ERR] No wrapped npy found"); return

    for f in wf[:args.k]:
        stem = Path(f).stem
        uw = str(udir / f"{stem}.npy")
        qw = str(qdir / f"{stem}.npy")
        a = np.load(f)                        # wrapped
        print(f"\n[{stem}] wrapped shape={a.shape}  min/max/mean/std={stat('w',a)}")
        if os.path.exists(uw):
            b = np.load(uw)                   # unwrapped
            print(f"[{stem}] unwrapped shape={b.shape}  min/max/mean/std={stat('u',b)}")
        else:
            print(f"[{stem}] unwrapped: MISSING")

        if os.path.exists(qw):
            q = np.load(qw).astype(np.float32)
            nz = float((q>0).mean())*100.0
            c10 = float((q>0.10).mean())*100.0
            c20 = float((q>0.20).mean())*100.0
            print(f"[{stem}] quality>0: {nz:.2f}% | >0.10: {c10:.2f}% | >0.20: {c20:.2f}%")
            if roi is not None and roi.shape==q.shape:
                rq = q[roi]
                rnz = float((rq>0).mean())*100.0
                rc20 = float((rq>0.20).mean())*100.0
                print(f"[{stem}] (ROI) quality>0: {rnz:.2f}% | >0.20: {rc20:.2f}%")
        else:
            print(f"[{stem}] quality: MISSING")

if __name__ == "__main__":
    main()
