#!/usr/bin/env python3
# ASCII-only, Windows-safe
# Apply sign+offset alignment so that rewrap(unwrapped_aligned) ~= wrapped, per frame.
# Writes phase_unwrapped_aligned_npy/phase_unwrapped_aligned_png
# Optional: also writes phase_wrapped_rewrap_from_aligned_png for sanity-check.
#
# Usage:
# python phase_align_apply_min.py ^
#   --out-root "C:\...\band28" ^
#   --roi-mask "C:\...\roi_mask.png" ^
#   --qmin 0.30 --save-rewrap
#
# After running, you can re-run QC against the same band folder to confirm low residuals.

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0*np.pi) - np.pi

def read_phase_any(path: Path, is_phase: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    im = Image.open(path).convert("L")
    arr01 = (np.array(im, dtype=np.uint8).astype(np.float32)) / 255.0
    if is_phase:
        return arr01 * (2.0*np.pi) - np.pi
    return arr01

def save_phase_png_rad(phi: np.ndarray, path_png: Path):
    # map [-pi,pi] -> [0,1] for saving
    arr01 = (wrap_pi(phi) + np.pi) / (2.0*np.pi)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr01,0,1)*255.0+0.5).astype(np.uint8), mode="L").save(path_png)

def best_offset(phi_w: np.ndarray, phi_u_wrapped: np.ndarray, valid: np.ndarray) -> float:
    diff = wrap_pi(phi_w - phi_u_wrapped)
    z = np.exp(1j * diff[valid])
    if z.size == 0:
        return 0.0
    return float(np.angle(z.mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="bandXX folder with phase_* subfolders")
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--qmin", type=float, default=0.0)
    ap.add_argument("--save-rewrap", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    w_png_root = root / "phase_wrapped_png"
    w_npy_root = root / "phase_wrapped_npy"
    u_png_root = root / "phase_unwrapped_png"
    u_npy_root = root / "phase_unwrapped_npy"

    # Discover list of files from wrapped (prefer NPY else PNG)
    rels = []
    base_w = None
    if w_npy_root.exists():
        rels = [p.relative_to(w_npy_root) for p in w_npy_root.rglob("*.npy")]
        base_w = w_npy_root
    elif w_png_root.exists():
        exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
        rels = [p.relative_to(w_png_root) for p in w_png_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        base_w = w_png_root
    else:
        print("[ERR] No wrapped folder found."); return
    if not rels:
        print("[ERR] Wrapped folder exists but empty."); return

    # ROI
    roi = None
    if args.roi_mask:
        try:
            roi = np.array(Image.open(args.roi_mask).convert("L"), dtype=np.uint8)
        except Exception as e:
            print("[WARN] Failed to read ROI mask:", e)
            roi = None

    out_npy = root / "phase_unwrapped_aligned_npy"
    out_png = root / "phase_unwrapped_aligned_png"
    out_rew = root / "phase_wrapped_rewrap_from_aligned_png"  # optional

    saved = 0; skipped = 0
    for rel in rels:
        # paths
        w_path = base_w / rel
        # matching unwrapped
        u_path = None
        if (u_npy_root / rel.with_suffix(".npy")).exists():
            u_path = u_npy_root / rel.with_suffix(".npy")
        elif (u_png_root / rel.with_suffix(".png")).exists():
            u_path = u_png_root / rel.with_suffix(".png")
        else:
            skipped += 1
            continue

        # read arrays
        try:
            phi_w = read_phase_any(w_path, is_phase=True)
            phi_u = read_phase_any(u_path, is_phase=True)
        except Exception:
            skipped += 1
            continue

        H = min(phi_w.shape[0], phi_u.shape[0])
        W = min(phi_w.shape[1], phi_u.shape[1])
        if H <= 0 or W <= 0:
            skipped += 1
            continue
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]

        # valid mask
        valid = np.ones((H,W), dtype=bool)
        if roi is not None:
            if roi.shape != (H,W):
                roi_r = np.array(Image.fromarray(roi, mode="L").resize((W,H), resample=Image.NEAREST))
            else:
                roi_r = roi
            valid &= (roi_r > 0)

        # quality optional (if exists)
        q = None
        q_png_root = root / "quality_png"
        q_npy_root = root / "quality_npy"
        q_path = None
        if (q_npy_root / rel.with_suffix(".npy")).exists():
            q_path = q_npy_root / rel.with_suffix(".npy")
        elif (q_png_root / rel.with_suffix(".png")).exists():
            q_path = q_png_root / rel.with_suffix(".png")
        if q_path is not None:
            try:
                q = read_phase_any(q_path, is_phase=False)[:H,:W]
                if args.qmin > 0.0:
                    valid &= (q >= float(args.qmin))
            except Exception:
                pass

        if int(valid.sum()) < 100:
            skipped += 1
            continue

        # sign + offset search
        best_phi_u = None
        best_rmse = None
        for sgn in (+1.0, -1.0):
            phi_u_s = sgn * phi_u
            c = best_offset(phi_w, wrap_pi(phi_u_s), valid)
            resid = wrap_pi(phi_w - wrap_pi(phi_u_s + c))
            r = resid[valid]
            rmse = float(np.sqrt(np.mean(r**2)))
            if (best_rmse is None) or (rmse < best_rmse):
                best_rmse = rmse
                best_phi_u = phi_u_s + c  # aligned unwrapped

        # save aligned unwrapped
        dst_npy = (out_npy / rel).with_suffix(".npy"); dst_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_npy, best_phi_u.astype(np.float32))
        dst_png = (out_png / rel).with_suffix(".png"); save_phase_png_rad(best_phi_u, dst_png)

        # optional: save rewrap for sanity visual
        if args.save_rewrap:
            rew = wrap_pi(best_phi_u)
            dst_rew = (out_rew / rel).with_suffix(".png"); save_phase_png_rad(rew, dst_rew)

        saved += 1

    print(f"[DONE] Saved aligned unwrapped for {saved} frame(s). Skipped {skipped}.")
    print("Folders written:")
    print(" -", out_npy)
    print(" -", out_png)
    if args.save_rewrap:
        print(" -", out_rew)

if __name__ == "__main__":
    main()
