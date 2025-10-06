#!/usr/bin/env python3
import os, csv, json, re, numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_erosion

def main():
    PH = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseOut_b18_cs16_ff100")
    RF = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseRef_b18_cs16_ff100")
    ROI = Path("C:/ESPI_TEMP/roi_mask.png")
    
    OUT = PH / "qc_align_B_IRLS"
    OUT.mkdir(parents=True, exist_ok=True)
    pi = np.pi
    
    # ROI (erosion μικρό για τα άκρα)
    m = np.array(Image.open(ROI))
    m = m[..., 0] if m.ndim == 3 else m
    m = (m > 0)
    m = binary_erosion(m, iterations=3)
    
    # Subsample του ROI για IRLS (κρατά μνήμη χαμηλά)
    idx = np.flatnonzero(m.ravel())
    np.random.seed(42)
    keep = 200_000 if idx.size > 200_000 else idx.size
    sub = np.random.choice(idx, size=keep, replace=False)
    H, W = m.shape
    yy, xx = np.divmod(sub, W)
    Xfull = np.stack([xx, yy, np.ones_like(xx)], 1).astype(np.float32)
    
    def base_no_suffix(stem):
        return re.sub(r"_(\d+)$", "", stem)
    
    load = lambda r, s, n: np.load(r / s / (n + ".npy")).astype(np.float32)
    
    rows = []
    stems = sorted([p.stem for p in (PH / "phase_wrapped_npy").glob("*.npy")])
    
    print(f"Processing {len(stems)} phase files with memory-safe IRLS...")
    
    for nm in stems:
        base = base_no_suffix(nm)
        pr = RF / "phase_wrapped_npy" / (base + ".npy")
        if not pr.exists():
            continue
            
        try:
            wt = load(PH, "phase_wrapped_npy", nm)
            wr = load(RF, "phase_wrapped_npy", base)
            d = np.angle(np.exp(1j * (wt - wr))).astype(np.float32)
            d = np.unwrap(np.unwrap(d, axis=1), axis=0).astype(np.float32)
            
            yfull = d.ravel()[sub][:, None].astype(np.float32)
            
            # IRLS (Huber) χωρίς τεράστια W: χρησιμοποιούμε βάρη w (vector)
            w = np.ones((Xfull.shape[0], 1), np.float32)
            for _ in range(5):
                # weighted normal equations: (X^T W X) a = X^T W y
                Xw = Xfull * w  # broadcast σε κάθε στήλη
                A = Xfull.T @ Xw  # 3x3
                b = Xw.T @ yfull  # 3x1
                coef = np.linalg.solve(A, b).ravel().astype(np.float32)
                a, b0, c = map(float, coef)
                # υπολογισμός residuals πάνω στο SUB
                r = (yfull - (Xfull @ coef.reshape(-1, 1)))
                s = max(1e-6, 1.4826 * np.median(np.abs(r)))  # robust scale (MAD)
                t = r / (1.345 * s)
                w = 1 / np.maximum(1, np.abs(t))  # βάρος [0,1]
            
            # τελικό plane σε FULL grid
            Y, X = np.mgrid[0:H, 0:W]
            plane = (a * X + b0 * Y + c).astype(np.float32)
            dd = (d - plane)[m]
            rmse = float(np.sqrt(np.mean(dd * dd)))
            pct2 = float(100 * np.mean(np.abs(dd) > (pi / 2)))
            pct4 = float(100 * np.mean(np.abs(dd) > (pi / 4)))
            rows.append((nm, rmse, pct2, pct4))
            
            if len(rows) % 10 == 0:
                print(f"Processed {len(rows)} files...")
                
        except Exception as e:
            print(f"Error processing {nm}: {e}")
            continue
    
    with open(OUT / "metrics.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["name", "rmse", "pct_gt_pi2", "pct_gt_pi4"])
        wcsv.writerows(rows)
    
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
        
        print("[IRLS] summary:", summary)
    else:
        print("No valid files processed")

if __name__ == "__main__":
    main()

