#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
phase_auto_compare_WINFIX_v2.py
-------------------------------
Improved (Windows-safe) auto-compare for FFT phase extraction:
- Runs phase_extract_fft.py for multiple band values
- RECURSIVELY reads outputs (handles nested subfolders)
- Computes QC (RMSE rewrap residual, %>|pi/4|, %>|pi/2|)
- Picks the best band and copies its folder to "<out-root>\best"

Usage:
python phase_auto_compare_WINFIX_v2.py ^
  --phase-script "C:\ESPI_DnCNN\phase_extract_fft.py" ^
  --input-dir    "C:\...\W01_ESPI_90db-Averaged_masked" ^
  --out-root     "C:\...\W01_PhaseOut_masked_comp" ^
  --bands 22 28 ^
  --center-suppress 14 --flatfield 60 --unwrap auto ^
  --roi-mask "C:\...\roi_mask.png"

Notes:
- Works whether outputs are flat or nested folders under phase_* dirs.
- If unwrapped is missing for some frames, those frames are skipped in QC.
"""

import argparse, subprocess, sys, shutil, math, numpy as np
from pathlib import Path
from PIL import Image

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi

def read_phase(path: Path, is_phase: bool) -> np.ndarray:
    """Reads .npy or image; converts to float32; if phase image, map [0..1]->[-pi..pi]."""
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    # image route
    im = Image.open(path).convert("L")
    arr01 = np.array(im, dtype=np.uint8).astype(np.float32) / 255.0
    if is_phase:
        return arr01 * 2*np.pi - np.pi
    return arr01

def iter_rel_files(root: Path, exts):
    """Yield relative file paths (Posix style) for files under root with given extensions (set)."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p.relative_to(root)

def qc_summary_recursive(out_root: Path):
    w_root_png = out_root / "phase_wrapped_png"
    w_root_npy = out_root / "phase_wrapped_npy"
    u_root_png = out_root / "phase_unwrapped_png"
    u_root_npy = out_root / "phase_unwrapped_npy"

    # Build candidate list from wrapped roots (png preferred; else npy)
    rel_list = []
    if w_root_png.exists():
        rel_list = list(iter_rel_files(w_root_png, {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}))
        w_mode = "png"
        w_base = w_root_png
    elif w_root_npy.exists():
        rel_list = list(iter_rel_files(w_root_npy, {".npy"}))
        w_mode = "npy"
        w_base = w_root_npy
    else:
        return None

    if not rel_list:
        return None

    rmses = []; p4 = []; p2 = []; n_used = 0
    for rel in rel_list:
        # Wrapped paths
        w_path = w_base / rel
        # Match unwrapped path (prefer same extension family as available)
        candidates = []
        if u_root_npy.exists():
            candidates.append(u_root_npy / rel.with_suffix(".npy"))
        if u_root_png.exists():
            candidates.append(u_root_png / rel.with_suffix(".png"))
        u_path = None
        for c in candidates:
            if c.exists():
                u_path = c; break
        if u_path is None:
            continue  # skip if no unwrapped counterpart

        try:
            phi_w = read_phase(w_path, is_phase=True)
            phi_u = read_phase(u_path, is_phase=True)
        except Exception:
            continue

        H = min(phi_w.shape[0], phi_u.shape[0]); W = min(phi_w.shape[1], phi_u.shape[1])
        if H <= 0 or W <= 0: 
            continue
        phi_w = phi_w[:H,:W]; phi_u = phi_u[:H,:W]
        phi_uw = wrap_pi(phi_u)
        resid = wrap_pi(phi_w - phi_uw)
        abs_r = np.abs(resid)

        rmses.append(float(np.sqrt(np.mean(resid**2))))
        p4.append(float(np.mean(abs_r > (np.pi/4))))
        p2.append(float(np.mean(abs_r > (np.pi/2))))
        n_used += 1

    if n_used == 0:
        return None
    return {
        "n": n_used,
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
        qc = qc_summary_recursive(out_dir)
        if qc is None:
            print(f"[WARN] No QC metrics computed for band={b} (missing outputs or no unwrapped?)")
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
