#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
phase_qc_roi_SIGNFIX.py
-----------------------
QC for FFT-based phase extraction with optional ROI mask and quality threshold,
**robust to global phase sign flips** (common when selecting left vs right side-lobe).
For each frame, it computes residual metrics twice:
  A) resid = wrap( phi_w - wrap(+phi_u) )
  B) resid = wrap( phi_w - wrap(-phi_u) )
and keeps the variant with lower RMSE.

Other features:
- Recursively scans outputs (PNG/NPY).
- Metrics on valid pixels only (ROI white=keep, q>=qmin keep).
- Saves CSV and optional residual heatmaps (from best-sign choice).

Usage:
python phase_qc_roi_SIGNFIX.py ^
  --out-root "C:\...\band28" ^
  --roi-mask "C:\...\roi_mask.png" ^
  --qmin 0.30 --save-maps
"""
import argparse, numpy as np, csv
from pathlib import Path
from PIL import Image

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi

def read_phase_any(path: Path, is_phase: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    im = Image.open(path).convert("L")
    arr01 = np.array(im, dtype=np.uint8).astype(np.float32) / 255.0
    if is_phase:
        return arr01 * 2*np.pi - np.pi
    return arr01

def iter_rel_files(root: Path, exts):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p.relative_to(root)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--csv", default="phase_qc_roi_signfix.csv")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--qmin", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.out_root)
    w_png_root = root / "phase_wrapped_png"
    w_npy_root = root / "phase_wrapped_npy"
    u_png_root = root / "phase_unwrapped_png"
    u_npy_root = root / "phase_unwrapped_npy"
    q_png_root = root / "quality_png"
    q_npy_root = root / "quality_npy"

    # Build reference list from wrapped roots (prefer PNG; else NPY)
    if w_png_root.exists():
        rels = list(iter_rel_files(w_png_root, {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}))
        w_base = w_png_root
        w_is_png = True
    elif w_npy_root.exists():
        rels = list(iter_rel_files(w_npy_root, {".npy"}))
        w_base = w_npy_root
        w_is_png = False
    else:
        print("[ERR] No wrapped phase found.")
        return
    if not rels:
        print("[ERR] Wrapped folder exists but empty.")
        return

    # ROI
    roi = None
    if args.roi_mask:
        try:
            roi = np.array(Image.open(args.roi_mask).convert("L"), dtype=np.uint8)
        except Exception as e:
            print("[WARN] Failed to read ROI mask:", e)
            roi = None

    # Outputs
    rows = [("name","n_valid","rmse_rad","p_gt_pi4","p_gt_pi2","sign")]
    maps_dir = root / "qc_residual_maps_roi_sign"
    if args.save_m
:
        maps_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    sum_rmse = 0.0; sum_p4 = 0.0; sum_p2 = 0.0; sum_valid = 0

    for rel in rels:
        name = rel.stem
        w_path = (w_base / rel)
        # choose unwrapped counterpart
        candidates = []
        if w_is_png:
            candidates.extend([(u_png_root / rel.with_suffix(".png")), (u_npy_root / rel.with_suffix(".npy"))])
        else:
            candidates.extend([(u_npy_root / rel.with_suffix(".npy")), (u_png_root / rel.with_suffix(".png"))])
        u_path = None
        for c in candidates:
            if c.exists():
                u_path = c; break
        if u_path is None:
            continue  # skip if no unwrapped

        # quality (optional)
        q_path = None
        for q in [(q_png_root / rel.with_suffix(".png")), (q_npy_root / rel.with_suffix(".npy"))]:
            if q.exists():
                q_path = q; break

        # read arrays
        try:
            phi_w = read_phase_any(w_path, is_phase=True)
            phi_u = read_phase_any(u_path, is_phase=True)
        except Exception:
            continue

        H = min(phi_w.shape[0], phi_u.shape[0]); W = min(phi_w.shape[1], phi_u.shape[1])
        if H <= 0 or W <= 0: continue
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]

        valid = np.ones((H,W), dtype=bool)
        if roi is not None:
            if roi.shape != (H,W):
                roi_r = np.array(Image.fromarray(roi, mode="L").resize((W,H), resample=Image.NEAREST))
            else:
                roi_r = roi
            valid &= (roi_r > 0)

        if q_path is not None:
            try:
                q = read_phase_any(q_path, is_phase=False)
                q = q[:H,:W]
                if args.qmin > 0.0:
                    valid &= (q >= float(args.qmin))
            except Exception:
                pass

        n_valid = int(valid.sum())
        if n_valid < 100:
            continue

        # Try both signs
        resid_pos = wrap_pi(phi_w - wrap_pi(+phi_u))
        resid_neg = wrap_pi(phi_w - wrap_pi(-phi_u))

        a_pos = np.abs(resid_pos[valid]); a_neg = np.abs(resid_neg[valid])
        rmse_pos = float(np.sqrt(np.mean(resid_pos[valid]**2)))
        rmse_neg = float(np.sqrt(np.mean(resid_neg[valid]**2)))

        if rmse_neg < rmse_pos:
            resid = resid_neg; sign = -1; a = a_neg; rmse = rmse_neg
        else:
            resid = resid_pos; sign = +1; a = a_pos; rmse = rmse_pos

        p_pi4 = float(np.mean(a > (np.pi/4)))
        p_pi2 = float(np.mean(a > (np.pi/2)))

        rows.append((name, n_valid, f"{rmse:.6f}", f"{p_pi4:.6f}", f"{p_pi2:.6f}", str(sign)))
        n_frames += 1; sum_rmse += rmse; sum_p4 += p_pi4; sum_p2 += p_pi2; sum_valid += n_valid

        if args.save_maps:
            m = np.zeros((H,W), dtype=np.float32)
            vals = np.clip(np.abs(resid)/np.pi, 0.0, 1.0)
            m[valid] = vals[valid]
            Image.fromarray((m*255.0+0.5).astype(np.uint8), mode="L").save((maps_dir / name).with_suffix(".png"))

    # write CSV
    out_csv = Path(args.out_root) / args.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(rows)

    if n_frames == 0:
        print("[WARN] No frames evaluated (check unwrapped presence / ROI size / qmin).")
        print("CSV written to:", out_csv)
        return

    print(f"[SUMMARY over {n_frames} image(s)] (valid pixels only, sign-corrected)")
    print(f"  RMSE(rad): {sum_rmse/n_frames:.4f}")
    print(f"  >pi/4:     {100.0*sum_p4/n_frames:.2f}%")
    print(f"  >pi/2:     {100.0*sum_p2/n_frames:.2f}%")
    print(f"  Mean valid px/frame: {sum_valid//n_frames}")
    print("CSV saved to:", out_csv)
    if args.save_maps:
        print("Residual maps saved to:", maps_dir)

if __name__ == "__main__":
    main()
