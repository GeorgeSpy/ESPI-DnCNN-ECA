#!/usr/bin/env python3
import os, csv, json, re, numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_erosion

def main():
    PH = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseOut_b18_cs16_ff100")
    RF = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseRef_b18_cs16_ff100")
    ROI = Path("C:/ESPI_TEMP/roi_mask.png")
    
    OUT = PH / "qc_align_B"
    OUT.mkdir(parents=True, exist_ok=True)
    pi = np.pi
    
    # --- ROI (με μικρή διάβρωση για να αποφύγουμε boundary unwrap artifacts) ---
    m = np.array(Image.open(ROI))
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0)
    m = binary_erosion(m, iterations=3)
    
    # --- QC gate: αν υπάρχει metrics από wrapped QC, γκάραρέ το ---
    qc_path = PH / "qc_wrapped_metrics.csv"
    good_names = None
    if qc_path.exists():
        import pandas as pd
        q = pd.read_csv(qc_path)  # columns: basename, rmse, pct_gt_pi2, ...
        good = q[(q["rmse"] <= 1.2) & (q["pct_gt_pi2"] <= 30.0)]["basename"].tolist()
        good_names = set(good)
        print(f"QC gate: {len(good_names)} frames passed")
    
    # --- helper: ταιριάζει GPU names με _\d\d στο averaged ref (χωρίς suffix) ---
    def base_no_suffix(stem):
        return re.sub(r"_(\d+)$", "", stem)
    
    def load(root, sub, nm):
        return np.load(root / sub / (nm + ".npy")).astype(np.float32)
    
    rows = []
    ph_stems = sorted([p.stem for p in (PH / "phase_wrapped_npy").glob("*.npy")])
    
    print(f"Processing {len(ph_stems)} phase files...")
    
    for nm in ph_stems:
        base = base_no_suffix(nm)
        # QC gate per-frame (αν διαθέσιμο)
        if good_names is not None and base not in good_names:
            continue
        # φορτώνουμε test frame και ref με το base
        if not (RF / "phase_wrapped_npy" / (base + ".npy")).exists():
            continue
            
        try:
            wt = load(PH, "phase_wrapped_npy", nm)
            wr = load(RF, "phase_wrapped_npy", base)
            
            d = np.angle(np.exp(1j * (wt - wr))).astype(np.float32)
            # unwrap-unwrap + επίπεδο (plane) στο ROI
            d = np.unwrap(np.unwrap(d, axis=1), axis=0).astype(np.float32)
            H, W = d.shape
            yy, xx = np.mgrid[0:H, 0:W]
            X = np.stack([xx[m], yy[m], np.ones(np.count_nonzero(m), np.float32)], 1)
            y = d[m][:, None]
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b, c = coef.ravel().astype(np.float32)
            d2 = d - (a * xx + b * yy + c)
            dd = d2[m]
            
            rmse = float(np.sqrt(np.mean(dd * dd)))
            pct2 = float(100 * np.mean(np.abs(dd) > (pi / 2)))
            pct4 = float(100 * np.mean(np.abs(dd) > (pi / 4)))
            rows.append((nm, rmse, pct2, pct4))
            
            if len(rows) % 10 == 0:
                print(f"Processed {len(rows)} files...")
                
        except Exception as e:
            print(f"Error processing {nm}: {e}")
            continue
    
    # write
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "rmse", "pct_gt_pi2", "pct_gt_pi4"])
        w.writerows(rows)
    
    if rows:
        vals = np.array([(r[1], r[2], r[3]) for r in rows], np.float32)
        
        def q(x, p):
            return float(np.percentile(x, p)) if x.size else float("nan")
        
        summary = {
            "n": int(len(rows)),
            "rmse_median": float(np.nanmedian(vals[:, 0])) if len(rows) else float("nan"),
            "rmse_p95": q(vals[:, 0], 95) if len(rows) else float("nan"),
            "pct_pi2_median": float(np.nanmedian(vals[:, 1])) if len(rows) else float("nan"),
            "pct_pi2_p95": q(vals[:, 1], 95) if len(rows) else float("nan"),
            "pct_pi4_median": float(np.nanmedian(vals[:, 2])) if len(rows) else float("nan"),
            "pct_pi4_p95": q(vals[:, 2], 95) if len(rows) else float("nan")
        }
        
        with open(OUT / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("[OK] Robust-B summary:", summary)
    else:
        print("No valid files processed")

if __name__ == "__main__":
    main()

