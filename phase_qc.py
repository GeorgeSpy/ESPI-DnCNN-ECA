#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_qc.py
-----------
Quick quality-control for FFT-based phase extraction results.

Given an output root (from phase_extract_fft.py), it:
- Loads wrapped phase (NPY preferred) and unwrapped phase (NPY preferred)
- Re-wraps the unwrapped phase and computes residual wrt wrapped
- Reports per-image metrics:
    * RMSE of residual [rad]
    * % pixels with |residual| > pi/4 and > pi/2
    * Correlation between quality map (if available) and residual magnitude (expect negative)
- Saves residual heatmaps (|residual| scaled to [0,1] with pi mapped to 1)

Usage:
python phase_qc.py --out-root "C:\\...\\W01_PhaseOut_masked" --save-maps
"""

import argparse, numpy as np, math, csv
from pathlib import Path
from PIL import Image

def wrap_pi(x: np.ndarray) -> np.ndarray:
    # wrap radians to (-pi, pi]
    return (x + np.pi) % (2*np.pi) - np.pi

def load_npy_or_png(path_npy: Path, path_png: Path, scale_png_to_rad: bool) -> np.ndarray:
    if path_npy.exists():
        return np.load(path_npy).astype(np.float32)
    if path_png.exists():
        arr01 = np.array(Image.open(path_png).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
        if scale_png_to_rad:
            return arr01 * 2*np.pi - np.pi  # assume encoding [-pi,pi] -> [0,1]
        else:
            return arr01  # quality map [0,1]
    raise FileNotFoundError(f"Missing both {path_npy} and {path_png}")

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="phase output root (contains phase_wrapped_*, phase_unwrapped_*, quality_*)")
    ap.add_argument("--csv", default="phase_qc.csv")
    ap.add_argument("--save-maps", action="store_true", help="save residual heatmaps")
    args = ap.parse_args()

    root = Path(args.out_root)
    w_dir_npy = root / "phase_wrapped_npy"
    w_dir_png = root / "phase_wrapped_png"
    u_dir_npy = root / "phase_unwrapped_npy"
    u_dir_png = root / "phase_unwrapped_png"
    q_dir_npy = root / "quality_npy"
    q_dir_png = root / "quality_png"

    # List files by using wrapped_npy/png as reference
    cand = []
    if w_dir_npy.exists():
        cand = sorted(w_dir_npy.glob("*.npy"))
        key = "npy"
    elif w_dir_png.exists():
        cand = sorted(w_dir_png.glob("*.png"))
        key = "png"
    else:
        print("No wrapped phase found under", w_dir_npy, "or", w_dir_png)
        return

    rows = [("name","rmse_rad","p_gt_pi4","p_gt_pi2","corr_quality_vs_err")]
    acc = {"n":0, "rmse":0.0, "p_pi4":0.0, "p_pi2":0.0, "corr":0.0}
    maps_dir = root / "qc_residual_maps"
    if args.save_maps: ensure_dir(maps_dir)

    for wf in cand:
        name = wf.stem
        if key == "npy":
            w_path_npy = wf
            w_path_png = (w_dir_png / name).with_suffix(".png")
        else:
            w_path_npy = (w_dir_npy / name).with_suffix(".npy")
            w_path_png = wf

        u_path_npy = (u_dir_npy / name).with_suffix(".npy")
        u_path_png = (u_dir_png / name).with_suffix(".png")
        q_path_npy = (q_dir_npy / name).with_suffix(".npy")
        q_path_png = (q_dir_png / name).with_suffix(".png")

        try:
            phi_w = load_npy_or_png(w_path_npy, w_path_png, scale_png_to_rad=True)
            phi_u = load_npy_or_png(u_path_npy, u_path_png, scale_png_to_rad=True)
        except FileNotFoundError as e:
            print("Skipping", name, ":", e)
            continue

        H = min(phi_w.shape[0], phi_u.shape[0]); W = min(phi_w.shape[1], phi_u.shape[1])
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]

        # Re-wrap unwrapped and compute residual
        phi_u_wrapped = wrap_pi(phi_u)
        residual = wrap_pi(phi_w - phi_u_wrapped)  # residual in (-pi,pi]

        rmse = float(np.sqrt(np.mean(residual**2)))
        abs_r = np.abs(residual)
        p_pi4 = float(np.mean(abs_r > (np.pi/4)))
        p_pi2 = float(np.mean(abs_r > (np.pi/2)))

        # Correlation with quality (if available)
        corr = float('nan')
        try:
            q = load_npy_or_png(q_path_npy, q_path_png, scale_png_to_rad=False)
            q = q[:H,:W]
            r = abs_r.reshape(-1); qv = q.reshape(-1)
            if np.std(r) > 1e-9 and np.std(qv) > 1e-9:
                r_norm = (r - r.mean()) / (r.std() + 1e-9)
                q_norm = (qv - qv.mean()) / (qv.std() + 1e-9)
                corr = float(np.mean(r_norm * q_norm))
        except Exception:
            pass

        rows.append((name, f"{rmse:.6f}", f"{p_pi4:.6f}", f"{p_pi2:.6f}", "" if math.isnan(corr) else f"{corr:.6f}"))
        acc["n"] += 1; acc["rmse"] += rmse; acc["p_pi4"] += p_pi4; acc["p_pi2"] += p_pi2
        if not math.isnan(corr): acc["corr"] += corr

        if args.save_maps:
            # save |residual| heatmap (0..pi) -> 0..255
            m = np.clip(abs_r / np.pi, 0.0, 1.0)
            im = Image.fromarray((m*255.0+0.5).astype(np.uint8), mode="L")
            im.save((maps_dir / name).with_suffix(".png"))

    out_csv = root / args.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as fo:
        csv.writer(fo).writerows(rows)

    n = acc["n"]
    if n == 0:
        print("No images evaluated.")
        return
    print(f"[SUMMARY over {n} image(s)]")
    print(f"  RMSE(rad): {acc['rmse']/n:.4f}")
    print(f"  >pi/4:     {100.0*acc['p_pi4']/n:.2f}%")
    print(f"  >pi/2:     {100.0*acc['p_pi2']/n:.2f}%")
    if "corr" in acc and not math.isnan(acc.get("corr", float('nan'))):
        print(f"  Corr(q,|res|): {acc['corr']/n:.3f}  (negative is good; high quality => low error)")
    print("CSV saved to:", out_csv)
    if args.save_maps:
        print("Residual maps saved to:", maps_dir)

if __name__ == "__main__":
    main()
