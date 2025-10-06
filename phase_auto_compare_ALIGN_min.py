#!/usr/bin/env python3
# ASCII-only
# Run phase_extract_fft.py for a list of bands, then evaluate with sign+offset-robust QC (with ROI & qmin)
# and pick the best result. Requires phase_qc_roi_align_min_FIXED.py in the same folder or PATH.

import argparse, subprocess, sys, shutil, csv
from pathlib import Path

def run_extract(phase_script, input_dir, out_dir, band, center_suppress, flatfield, roi_mask, unwrap):
    cmd = [sys.executable, str(phase_script),
           "--input-dir", str(input_dir),
           "--output-dir", str(out_dir),
           "--band", str(band),
           "--center-suppress", str(center_suppress),
           "--flatfield", str(flatfield),
           "--unwrap", unwrap]
    if roi_mask:
        cmd += ["--roi-mask", str(roi_mask)]
    print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"Extractor failed for band={band} (code {rc})")

def run_qc(qc_script, out_dir, roi_mask, qmin, save_maps):
    cmd = [sys.executable, str(qc_script),
           "--out-root", str(out_dir)]
    if roi_mask:
        cmd += ["--roi-mask", str(roi_mask)]
    if qmin is not None:
        cmd += ["--qmin", str(qmin)]
    if save_maps:
        cmd += ["--save-maps"]
    print("[QC ]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"QC failed for {out_dir}")

def parse_qc_csv(csv_path: Path):
    # Reads last printed summary from file by re-parsing rows; returns (mean_rmse, mean_p4, mean_p2)
    # Our QC writes per-frame rows after header, so aggregate here to be safe.
    import csv, numpy as np
    rows=[]
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            return None
        # detect columns
        col_rmse = header.index("rmse_rad") if "rmse_rad" in header else None
        col_p4   = header.index("p_gt_pi4") if "p_gt_pi4" in header else None
        col_p2   = header.index("p_gt_pi2") if "p_gt_pi2" in header else None
        for r in rdr:
            if len(r) < 5: continue
            try:
                rows.append((float(r[col_rmse]), float(r[col_p4]), float(r[col_p2])))
            except:
                continue
    if not rows:
        return None
    import numpy as np
    arr = np.array(rows, dtype=float)
    return float(arr[:,0].mean()), float(arr[:,1].mean()), float(arr[:,2].mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-script", default="phase_extract_fft.py")
    ap.add_argument("--qc-script", default="phase_qc_roi_align_min_FIXED.py")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bands", type=int, nargs="+", default=[18,22,26,30,34,38,42])
    ap.add_argument("--center-suppress", type=int, default=16)
    ap.add_argument("--flatfield", type=int, default=80)
    ap.add_argument("--roi-mask", default="")
    ap.add_argument("--qmin", type=float, default=0.30)
    ap.add_argument("--unwrap", default="auto")
    ap.add_argument("--save-maps", action="store_true")
    args = ap.parse_args()

    phase_script = Path(args.phase_script)
    qc_script = Path(args.qc_script)
    inp = Path(args.input_dir)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for b in args.bands:
        out_dir = out_root / f"band{b:02d}"
        run_extract(phase_script, inp, out_dir, b, args.center_suppress, args.flatfield, args.roi_mask, args.unwrap)
        run_qc(qc_script, out_dir, args.roi_mask, args.qmin, args.save_maps)
        qc_csv = out_dir / "phase_qc_roi_align.csv"
        metrics = parse_qc_csv(qc_csv)
        if metrics is None:
            print(f"[WARN] No QC rows for band={b}")
            continue
        mean_rmse, mean_p4, mean_p2 = metrics
        print(f"[QC ] band={b}  RMSE={mean_rmse:.4f}  >pi/4={100*mean_p4:.2f}%  >pi/2={100*mean_p2:.2f}%")
        results.append((b, mean_rmse, mean_p2, mean_p4))

    if not results:
        print("[ERR] All runs failed or had no QC.")
        raise SystemExit(1)

    # choose best by (RMSE, p_pi2, p_pi4)
    best = sorted(results, key=lambda t: (t[1], t[2], t[3]))[0]
    best_b = best[0]
    print(f"[BEST] band={best_b}   (RMSE={best[1]:.4f}, >pi/2={100*best[2]:.2f}%, >pi/4={100*best[3]:.2f}%)")

    # copy best to out_root/best
    import shutil
    src = out_root / f"band{best_b:02d}"
    dst = out_root / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    with open(out_root/"best_choice.txt", "w", encoding="utf-8") as f:
        f.write(f"BEST band = {best_b}\nRMSE = {best[1]:.6f}\n>%pi/2 = {100*best[2]:.3f}%\n>%pi/4 = {100*best[3]:.3f}%\n")

if __name__ == "__main__":
    main()
