#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_auto_compare.py
---------------------
Runs phase_extract_fft.py multiple times (e.g., band=22 vs 28) with tuned defaults,
evaluates each run (RMSE of rewrap residual, %>|pi/4|, %>|pi/2|), and picks the best.
Copies the best run under "<out-root>\\best" and writes a best_choice.txt summary.

Usage example:
python phase_auto_compare.py ^
  --phase-script "C:\ESPI_DnCNN\phase_extract_fft.py" ^
  --input-dir    "C:\Users\...\W01_ESPI_90db-Averaged_masked" ^
  --out-root     "C:\Users\...\W01_PhaseOut_masked_comp" ^
  --bands 22 28 ^
  --center-suppress 14 --flatfield 60 --unwrap auto ^
  --roi-mask "C:\Users\...\roi_mask.png"

Notes:
- You can pass >2 bands; the script will evaluate all.
- Decision rule: lowest mean RMSE first; tie-break by lower %>|pi/2| then %>|pi/4|.
"""

import argparse, subprocess, sys, shutil, math, numpy as np
from pathlib import Path
from PIL import Image

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi

def imread_phase(path_npy: Path, path_png: Path, is_phase: bool) -> np.ndarray:
    if path_npy.exists():
        return np.load(path_npy).astype(np.float32)
    if path_png.exists():
        arr01 = np.array(Image.open(path_png).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
        if is_phase:
            return arr01 * 2*np.pi - np.pi
        return arr01
    raise FileNotFoundError(f"Missing both {path_npy} and {path_png}")

def list_names(dir_npy: Path, dir_png: Path):
    names = set()
    if dir_npy.exists():
        for f in dir_npy.glob("*.npy"): names.add(f.stem)
    if dir_png.exists():
        for f in dir_png.glob("*.png"): names.add(f.stem)
    return sorted(names)

def qc_summary(out_root: Path):
    w_npy = out_root / "phase_wrapped_npy"
    w_png = out_root / "phase_wrapped_png"
    u_npy = out_root / "phase_unwrapped_npy"
    u_png = out_root / "phase_unwrapped_png"
    names = list_names(w_npy, w_png)
    if not names:
        return None
    rmses = []; p4 = []; p2 = []
    for name in names:
        pw_npy = (w_npy / name).with_suffix(".npy")
        pw_png = (w_png / name).with_suffix(".png")
        pu_npy = (u_npy / name).with_suffix(".npy")
        pu_png = (u_png / name).with_suffix(".png")
        try:
            phi_w = imread_phase(pw_npy, pw_png, True)
            phi_u = imread_phase(pu_npy, pu_png, True)
        except FileNotFoundError:
            continue
        H = min(phi_w.shape[0], phi_u.shape[0]); W = min(phi_w.shape[1], phi_u.shape[1])
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]
        phi_uw = wrap_pi(phi_u)
        resid = wrap_pi(phi_w - phi_uw)
        abs_r = np.abs(resid)
        rmses.append(float(np.sqrt(np.mean(resid**2))))
        p4.append(float(np.mean(abs_r > (np.pi/4))))
        p2.append(float(np.mean(abs_r > (np.pi/2))))
    if not rmses:
        return None
    return {
        "n": len(rmses),
        "rmse": float(np.mean(rmses)),
        "p_pi4": float(np.mean(p4)),
        "p_pi2": float(np.mean(p2)),
    }

def run_one(phase_script: Path, input_dir: Path, out_dir: Path, roi_mask: str,
            flatfield: int, band: int, center: int, unwrap: str):
    cmd = [sys.executable, str(phase_script),
           "--input-dir", str(input_dir),
           "--output-dir", str(out_dir),
           "--flatfield", str(flatfield),
           "--band", str(band),
           "--center-suppress", str(center),
           "--unwrap", unwrap]
    if roi_mask:
        cmd += ["--roi-mask", roi_mask]
    print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Extractor failed (band={band}) with code {rc}")

def copy_best(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-script", default="phase_extract_fft.py")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--bands", type=int, nargs="+", default=[22,28])
    ap.add_argument("--center-suppress", type=int, default=14)
    ap.add_argument("--flatfield", type=int, default=60)
    ap.add_argument("--unwrap", default="auto")
    args = ap.parse_args()

    phase_script = Path(args.phase_script)
    inp = Path(args.input_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for b in args.bands:
        out_dir = out_root / f"band{b:02d}"
        run_one(phase_script, inp, out_dir, args.roi_mask, args.flatfield, b, args.center_suppress, args.unwrap)
        qc = qc_summary(out_dir)
        if qc is None:
            print(f"[WARN] No QC metrics computed for band={b} (missing outputs?)")
            continue
        print(f"[QC] band={b}  n={qc['n']}  RMSE={qc['rmse']:.4f}  >pi/4={qc['p_pi4']*100:.2f}%  >pi/2={qc['p_pi2']*100:.2f}%")
        results.append((b, qc))

    if not results:
        print("No successful runs to compare.")
        sys.exit(1)

    # Select best: lowest RMSE, then lowest p_pi2, then lowest p_pi4
    def keyfun(item):
        b, qc = item
        return (qc["rmse"], qc["p_pi2"], qc["p_pi4"], b)
    best_b, best_qc = sorted(results, key=keyfun)[0]

    best_src = out_root / f"band{best_b:02d}"
    best_dst = out_root / "best"
    copy_best(best_src, best_dst)

    summary = out_root / "best_choice.txt"
    with open(summary, "w", encoding="utf-8") as fo:
        fo.write(f"BEST band = {best_b}\n")
        fo.write(f"RMSE = {best_qc['rmse']:.6f}\n")
        fo.write(f">%pi/4 = {best_qc['p_pi4']*100:.3f}%\n")
        fo.write(f">%pi/2 = {best_qc['p_pi2']*100:.3f}%\n")
        fo.write(f"Source: {best_src}\n")

    print(f"[BEST] band={best_b}  RMSE={best_qc['rmse']:.4f}  >pi/4={best_qc['p_pi4']*100:.2f}%  >pi/2={best_qc['p_pi2']*100:.2f}%")
    print("Best outputs copied to:", best_dst)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
