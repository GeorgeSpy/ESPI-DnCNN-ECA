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
    
    m = np.array(Image.open(ROI))
    m = m[..., 0] if m.ndim == 3 else m
    m = (m > 0)
    m = binary_erosion(m, iterations=3)
    
    def base_no_suffix(stem):
        return re.sub(r"_(\d+)$", "", stem)
    
    load = lambda r, s, n: np.load(r / s / (n + ".npy")).astype(np.float32)
    
    rows = []
    stems = sorted([p.stem for p in (PH / "phase_wrapped_npy").glob("*.npy")])
    
    print(f"Processing {len(stems)} phase files with IRLS...")
    
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
            
            H, W = d.shape
            yy, xx = np.mgrid[0:H, 0:W]
            X = np.stack([xx[m], yy[m], np.ones(np.count_nonzero(m), np.float32)], 1)
            y = d[m][:, None]
            
            # --- IRLS (Huber) για επίπεδο ---
            a = b = c = 0.0
            w = np.ones((X.shape[0], 1), np.float32)
            for _ in range(5):
                # weighted least squares
                W = np.diagflat(w)
                coef = np.linalg.lstsq((W @ X), (W @ y), rcond=None)[0].ravel()
                a, b, c = map(float, coef)
                plane = (a * xx + b * yy + c).astype(np.float32)
                r = (d - plane)[m][:, None]
                s = max(1e-6, 1.4826 * np.median(np.abs(r)))  # robust scale (MAD)
                t = r / (1.345 * s)  # Huber threshold
                w = 1 / np.maximum(1, np.abs(t))  # weights in [0,1]
            
            d2 = d - plane
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
        
        print("[IRLS] summary:", summary)
    else:
        print("No valid files processed")

if __name__ == "__main__":
    main()

