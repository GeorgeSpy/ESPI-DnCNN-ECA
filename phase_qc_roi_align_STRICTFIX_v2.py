# -*- coding: utf-8 -*-
import argparse, glob, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_roi(png):
    if png and Path(png).exists():
        return (np.array(Image.open(png).convert("L"))>0)
    return None

def load_npy_list(root, sub):
    ff = sorted(glob.glob(str(Path(root)/sub/"*.npy")))
    return ff

def rmse(a):
    return float(np.sqrt(np.mean(a*a)))

def choose_sign_and_offset(cur, ref, mask):
    # cur ~ s*ref + o  (s in {+1,-1}), find s, o to minimize RMSE
    best = (1.0, 0.0, 1e9)  # (s,o,rmse)
    for s in (1.0, -1.0):
        d = s*cur[mask] - ref[mask]
        o = float(np.median(d))
        r = d - o
        e = rmse(r)
        if e < best[2]:
            best = (s, o, e)
    return best  # s, o, rmse

def save_residual_png(res, out_png):
    # map residual to [-pi,pi] then to [0,255] just για προεπισκόπηση
    rwrap = np.angle(np.exp(1j*res.astype(np.float32)))
    z = (rwrap + np.pi)/(2*np.pi)
    z = np.clip(z,0,1)
    Image.fromarray((z*255+0.5).astype(np.uint8)).save(out_png)

def main():
    ap = argparse.ArgumentParser("QC vs reference (sign+offset align)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--roi-mask", default=None)
    ap.add_argument("--qmin", type=float, default=0.20)
    ap.add_argument("--save-maps", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    udir = root / "phase_unwrapped_npy"
    qdir = root / "quality_npy"

    uf = load_npy_list(root, "phase_unwrapped_npy")
    if len(uf)==0:
        print("[ERR] No unwrapped npy found"); return
    qf = load_npy_list(root, "quality_npy")

    # build quality dict
    qmap = {}
    for qp in qf:
        qmap[Path(qp).stem] = np.load(qp).astype(np.float32)

    # ROI
    roi = load_roi(args.roi_mask)

    # reference = middle frame
    ref_path = uf[len(uf)//2]
    ref = np.load(ref_path).astype(np.float32)
    h,w = ref.shape

    # residual folder
    if args.save_maps:
        (root / "qc_residual_maps_refalign").mkdir(parents=True, exist_ok=True)

    n_ok = 0
    sum_rmse = 0.0
    sum_pi4 = 0.0
    sum_pi2 = 0.0
    sum_valid = 0

    for up in tqdm(uf, desc="QC"):
        stem = Path(up).stem
        cur = np.load(up).astype(np.float32)

        # valid mask: quality > qmin (και ROI αν υπάρχει)
        q = qmap.get(stem, None)
        if q is None:
            valid = np.ones_like(cur, dtype=bool)
        else:
            valid = (q > float(args.qmin))
        if roi is not None and roi.shape==valid.shape:
            valid &= roi

        n_valid = int(valid.sum())
        if n_valid < 100:
            continue

        s,o,e = choose_sign_and_offset(cur, ref, valid)
        res = s*cur - ref - o
        # Μετρήσεις μόνο στα valid
        rv = res[valid]
        e = rmse(rv)
        p4 = float(np.mean(np.abs(rv) > (np.pi/4.0)))*100.0
        p2 = float(np.mean(np.abs(rv) > (np.pi/2.0)))*100.0

        n_ok += 1
        sum_rmse += e
        sum_pi4  += p4
        sum_pi2  += p2
        sum_valid += n_valid

        if args.save_maps:
            png = root / "qc_residual_maps_refalign" / f"{stem}.png"
            save_residual_png(res, png)

    if n_ok==0:
        print("[WARN] No frames evaluated (check unwrapped presence / ROI / qmin).")
        return

    print(f"[SUMMARY over {n_ok} image(s)] (valid px, sign+offset corrected vs reference)")
    print(f"  RMSE(rad): {sum_rmse/n_ok:.4f}")
    print(f"  >pi/4:     {sum_pi4/n_ok:.2f}%")
    print(f"  >pi/2:     {sum_pi2/n_ok:.2f}%")
    print(f"  Mean valid px/frame: {int(sum_valid/n_ok)}")

if __name__ == "__main__":
    main()
