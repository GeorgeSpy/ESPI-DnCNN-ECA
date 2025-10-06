#!/usr/bin/env python3
import os, csv, json, numpy as np
from pathlib import Path
from PIL import Image

def main():
    PHASE = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseOut_b18_cs16_ff100")
    REF = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseRef_b18_cs16_ff100")
    ROI = Path("C:/ESPI_TEMP/roi_mask.png")
    
    OUT = PHASE / "qc_align_B"
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Load ROI mask
    m = np.array(Image.open(ROI))
    m = m[..., 0] if m.ndim == 3 else m
    m = m > 0
    pi = np.pi
    
    # Get available files
    phase_files = sorted([p.stem for p in (PHASE / "phase_wrapped_npy").glob("*.npy")])
    ref_files = sorted([p.stem for p in (REF / "phase_wrapped_npy").glob("*.npy")])
    
    print(f"Phase files: {len(phase_files)}")
    print(f"Ref files: {len(ref_files)}")
    
    # Find matching files (remove suffixes from phase files)
    rows = []
    for phase_file in phase_files:
        # Remove numbered suffix if present
        base_name = phase_file
        if '_' in phase_file and phase_file.split('_')[-1].isdigit():
            base_name = '_'.join(phase_file.split('_')[:-1])
        
        if base_name in ref_files:
            try:
                # Load phase data
                wt = np.load(PHASE / "phase_wrapped_npy" / f"{phase_file}.npy").astype(np.float32)
                wr = np.load(REF / "phase_wrapped_npy" / f"{base_name}.npy").astype(np.float32)
                
                # Compute wrapped difference
                d = np.angle(np.exp(1j * (wt - wr))).astype(np.float32)
                
                # Unwrap
                d = np.unwrap(np.unwrap(d, axis=1), axis=0).astype(np.float32)
                
                # Fit plane and remove
                H, W = d.shape
                yy, xx = np.mgrid[0:H, 0:W]
                X = np.stack([xx[m], yy[m], np.ones(np.count_nonzero(m), np.float32)], 1)
                y = d[m][:, None]
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                a, b, c = coef.ravel().astype(np.float32)
                d2 = d - (a * xx + b * yy + c)
                
                # Compute metrics
                dd = d2[m]
                rmse = float(np.sqrt(np.mean(dd * dd)))
                pct2 = float(100 * np.mean(np.abs(dd) > (pi / 2)))
                pct4 = float(100 * np.mean(np.abs(dd) > (pi / 4)))
                
                rows.append((phase_file, rmse, pct2, pct4))
                print(f"Processed: {phase_file} -> {base_name} (RMSE: {rmse:.3f})")
                
            except Exception as e:
                print(f"Error processing {phase_file}: {e}")
                continue
    
    # Save results
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "rmse", "pct_gt_pi2", "pct_gt_pi4"])
        w.writerows(rows)
    
    if rows:
        vals = np.array([(r[1], r[2], r[3]) for r in rows], np.float32)
        q = lambda x, p: float(np.percentile(x, p))
        summary = {
            "n": len(rows),
            "rmse_median": float(np.median(vals[:, 0])),
            "rmse_p95": q(vals[:, 0], 95),
            "pct_pi2_median": float(np.median(vals[:, 1])),
            "pct_pi2_p95": q(vals[:, 1], 95),
            "pct_pi4_median": float(np.median(vals[:, 2])),
            "pct_pi4_p95": q(vals[:, 2], 95)
        }
        
        with open(OUT / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("[OK] summary:", summary)
    else:
        print("No valid files processed")

if __name__ == "__main__":
    main()

