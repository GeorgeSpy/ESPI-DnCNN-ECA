#!/usr/bin/env python3
# ASCII-only
# Check consistency: rewrap(unwrapped) should match wrapped if they come from same extraction.
# Prints per-frame MAE and RMSE; saves a small PNG of |residual| (scaled) for a few frames.

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def wrap_pi(x):
    return (x + np.pi) % (2.0*np.pi) - np.pi

def load(path):
    if path.suffix.lower()==".npy":
        return np.load(path).astype(np.float32)
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8).astype(np.float32)/255.0
    return arr * (2.0*np.pi) - np.pi

def save_resid_png(resid, out_png):
    a = np.clip(np.abs(resid)/np.pi,0,1)
    Image.fromarray((a*255.0+0.5).astype(np.uint8), mode="L").save(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="bandXX folder (contains phase_wrapped_*/phase_unwrapped_*)")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.root)
    w_npy = root/"phase_wrapped_npy"
    u_npy = root/"phase_unwrapped_npy"
    w_png = root/"phase_wrapped_png"
    u_png = root/"phase_unwrapped_png"

    # build list from wrapped (prefer npy else png)
    rels = []
    base_w=None; mode=""
    if w_npy.exists():
        rels = [p.relative_to(w_npy) for p in w_npy.rglob("*.npy")]
        base_w=w_npy; mode="npy"
    elif w_png.exists():
        rels = [p.relative_to(w_png) for p in w_png.rglob("*.png")]
        base_w=w_png; mode="png"
    else:
        print("[ERR] No wrapped found"); return
    if not rels:
        print("[ERR] Empty wrapped folder"); return

    outd = root/"qc_consistency_maps"; outd.mkdir(parents=True, exist_ok=True)

    count=0
    for rel in sorted(rels)[:args.limit]:
        w_path = base_w/rel
        # find matching unwrapped
        if (u_npy/rel.with_suffix(".npy")).exists():
            u_path = u_npy/rel.with_suffix(".npy")
        elif (u_png/rel.with_suffix(".png")).exists():
            u_path = u_png/rel.with_suffix(".png")
        else:
            print("[SKIP] no unwrapped for", rel); continue

        phi_w = load(w_path); phi_u = load(u_path)
        H = min(phi_w.shape[0], phi_u.shape[0]); W=min(phi_w.shape[1], phi_u.shape[1])
        phi_w=phi_w[:H,:W]; phi_u=phi_u[:H,:W]
        resid = wrap_pi(phi_w - wrap_pi(phi_u))
        mae = float(np.mean(np.abs(resid))); rmse = float(np.sqrt(np.mean(resid**2)))
        print(f"[{rel.stem}] MAE={mae:.4f} RMSE={rmse:.4f}")
        save_resid_png(resid, (outd/rel.stem).with_suffix(".png"))
        count+=1
    print(f"[DONE] wrote {count} maps to {outd}")

if __name__ == "__main__":
    main()
