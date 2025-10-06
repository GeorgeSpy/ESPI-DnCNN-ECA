# -*- coding: utf-8 -*-
import argparse, os, glob
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_quality_map(qpath_png, qpath_npy, shape):
    if os.path.isfile(qpath_npy):
        q = np.load(qpath_npy).astype(np.float32)
        return q
    if os.path.isfile(qpath_png):
        q = np.array(Image.open(qpath_png).convert("L"), dtype=np.float32)/255.0
        return q
    # fallback: uniform ones (όλα έγκυρα αν δεν βρεθεί χάρτης)
    return np.ones(shape, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser(description="QC: compare out-root vs ref-root (sign+offset corrected)")
    ap.add_argument("--out-root", required=True, help="denoised/NEW results root (has phase_unwrapped_npy & quality)")
    ap.add_argument("--ref-root", required=True, help="baseline/REFERENCE results root (has phase_unwrapped_npy & quality)")
    ap.add_argument("--roi-mask", default=None, help="optional ROI mask PNG")
    ap.add_argument("--qmin", type=float, default=0.20, help="quality threshold [0..1]")
    ap.add_argument("--save-maps", action="store_true", help="save residual maps")
    args = ap.parse_args()

    out_u_dir = os.path.join(args.out_root, "phase_unwrapped_npy")
    ref_u_dir = os.path.join(args.ref_root, "phase_unwrapped_npy")
    out_q_png = os.path.join(args.out_root, "quality_png")
    ref_q_png = os.path.join(args.ref_root, "quality_png")
    out_q_npy = os.path.join(args.out_root, "quality_npy")
    ref_q_npy = os.path.join(args.ref_root, "quality_npy")

    os.makedirs(out_q_npy, exist_ok=True)  # δεν πειράζει, χρησιμοποιείται μόνο αν υπάρχουν npy
    os.makedirs(ref_q_npy, exist_ok=True)

    if args.save_maps:
        res_dir = os.path.join(args.out_root, "qc_residual_maps_vs_ref")
        os.makedirs(res_dir, exist_ok=True)
    else:
        res_dir = None

    roi = None
    if args.roi_mask and os.path.isfile(args.roi_mask):
        roi = np.array(Image.open(args.roi_mask).convert("L")) > 0

    files = sorted(glob.glob(os.path.join(out_u_dir, "*.npy")))
    n_ok = 0
    rmse_list = []
    gt_pi4 = 0
    gt_pi2 = 0
    valid_px_total = 0

    for f in tqdm(files, desc="QC"):
        name = os.path.basename(f)
        f_ref = os.path.join(ref_u_dir, name)
        if not os.path.isfile(f_ref):
            continue

        out_u = np.load(f).astype(np.float32)
        ref_u = np.load(f_ref).astype(np.float32)
        H, W = out_u.shape
        if ref_u.shape != (H, W):
            continue

        # quality maps
        q_out = load_quality_map(os.path.join(out_q_png, name.replace(".npy",".png")),
                                 os.path.join(out_q_npy, name), out_u.shape)
        q_ref = load_quality_map(os.path.join(ref_q_png, name.replace(".npy",".png")),
                                 os.path.join(ref_q_npy, name), ref_u.shape)

        valid = (q_out >= args.qmin) & (q_ref >= args.qmin)
        if roi is not None and roi.shape == (H, W):
            valid &= roi

        vcount = int(valid.sum())
        if vcount < 100:
            continue

        # sign + offset alignment
        # δοκιμή s=+1 και s=-1, με αφαίρεση μέσου offset στον έγκυρο χώρο
        def rmse_with_sign(s):
            diff = (out_u - s*ref_u)
            mu = float(diff[valid].mean())
            diffc = diff - mu
            r = float(np.sqrt(np.mean(diffc[valid]**2)))
            return r, diffc

        r_pos, diff_pos = rmse_with_sign(+1.0)
        r_neg, diff_neg = rmse_with_sign(-1.0)

        if r_pos <= r_neg:
            rmse = r_pos
            diffc = diff_pos
        else:
            rmse = r_neg
            diffc = diff_neg

        rmse_list.append(rmse)
        n_ok += 1
        valid_px_total += vcount
        a = np.abs(diffc)[valid]
        gt_pi4 += int((a > (np.pi/4)).sum())
        gt_pi2 += int((a > (np.pi/2)).sum())

        if res_dir is not None:
            # αποθήκευση residual PNG για οπτικό έλεγχο (linear scale)
            dshow = np.clip((diffc - (-np.pi)) / (2*np.pi), 0.0, 1.0)  # map [-pi,pi] -> [0,1]
            im = Image.fromarray((dshow*255.0).astype(np.uint8))
            im.save(os.path.join(res_dir, name.replace(".npy", ".png")))

    if n_ok == 0:
        print("[WARN] No frames evaluated (check folders/quality/roi/qmin).")
        return

    rmse_mean = float(np.mean(rmse_list))
    mean_valid_px = int(valid_px_total / n_ok)
    # ποσοστά με βάση το σύνολο των valid pixels που αξιολογήθηκαν
    # (για απλότητα τα ορίζουμε ανά frame-σύνολο)
    p_pi4 = 100.0 * gt_pi4 / max(valid_px_total, 1)
    p_pi2 = 100.0 * gt_pi2 / max(valid_px_total, 1)

    print(f"[SUMMARY over {n_ok} image(s)] (valid px, sign+offset corrected vs reference)")
    print(f"  RMSE(rad): {rmse_mean:.4f}")
    print(f"  >pi/4:     {p_pi4:.2f}%")
    print(f"  >pi/2:     {p_pi2:.2f}%")
    print(f"  Mean valid px/frame: {mean_valid_px}")

if __name__ == "__main__":
    main()
