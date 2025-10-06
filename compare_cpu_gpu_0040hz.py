#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    # Paths
    GPU_PHASE = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseOut_b18_cs16_ff100")
    CPU_PHASE = Path("C:/ESPI_TEMP/CPU_AUX/W01_0040_PhaseOut_b18_cs16_ff100")
    REF = Path("C:/ESPI_TEMP/GPU_FULL/W01_PhaseRef_b18_cs16_ff100")
    ROI = Path("C:/ESPI_TEMP/roi_mask.png")
    
    pi = np.pi
    
    # Load ROI mask
    m = np.array(Image.open(ROI))
    m = m[..., 0] if m.ndim == 3 else m
    m = (m > 0)
    
    def load(root, sub, nm):
        return np.load(root / sub / (nm + ".npy")).astype(np.float32)
    
    # Compare a few 0040Hz files
    gpu_files = [f for f in (GPU_PHASE / "phase_wrapped_npy").glob("0040Hz_91.0db_*.npy")]
    cpu_files = [f for f in (CPU_PHASE / "phase_wrapped_npy").glob("0040Hz_91.0db_*.npy")]
    
    print(f"GPU files: {len(gpu_files)}")
    print(f"CPU files: {len(cpu_files)}")
    
    # Compare first few files
    results = []
    for i, (gpu_file, cpu_file) in enumerate(zip(gpu_files[:5], cpu_files[:5])):
        try:
            # Load data
            wt_gpu = load(GPU_PHASE, "phase_wrapped_npy", gpu_file.stem)
            wt_cpu = load(CPU_PHASE, "phase_wrapped_npy", cpu_file.stem)
            wr = load(REF, "phase_wrapped_npy", "0040Hz_91.0db")
            
            # Compute differences
            d_gpu = np.angle(np.exp(1j * (wt_gpu - wr))).astype(np.float32)
            d_cpu = np.angle(np.exp(1j * (wt_cpu - wr))).astype(np.float32)
            
            # Unwrap
            d_gpu = np.unwrap(np.unwrap(d_gpu, axis=1), axis=0).astype(np.float32)
            d_cpu = np.unwrap(np.unwrap(d_cpu, axis=1), axis=0).astype(np.float32)
            
            # Simple plane removal (no IRLS to avoid memory issues)
            H, W = d_gpu.shape
            yy, xx = np.mgrid[0:H, 0:W]
            X = np.stack([xx[m], yy[m], np.ones(np.count_nonzero(m), np.float32)], 1)
            
            # GPU
            y_gpu = d_gpu[m][:, None]
            coef_gpu = np.linalg.lstsq(X, y_gpu, rcond=None)[0].ravel()
            a_gpu, b_gpu, c_gpu = coef_gpu
            d2_gpu = d_gpu - (a_gpu * xx + b_gpu * yy + c_gpu)
            dd_gpu = d2_gpu[m]
            
            # CPU
            y_cpu = d_cpu[m][:, None]
            coef_cpu = np.linalg.lstsq(X, y_cpu, rcond=None)[0].ravel()
            a_cpu, b_cpu, c_cpu = coef_cpu
            d2_cpu = d_cpu - (a_cpu * xx + b_cpu * yy + c_cpu)
            dd_cpu = d2_cpu[m]
            
            # Metrics
            rmse_gpu = float(np.sqrt(np.mean(dd_gpu * dd_gpu)))
            rmse_cpu = float(np.sqrt(np.mean(dd_cpu * dd_cpu)))
            pct2_gpu = float(100 * np.mean(np.abs(dd_gpu) > (pi / 2)))
            pct2_cpu = float(100 * np.mean(np.abs(dd_cpu) > (pi / 2)))
            
            results.append({
                'file': gpu_file.stem,
                'gpu_rmse': rmse_gpu,
                'cpu_rmse': rmse_cpu,
                'gpu_pct2': pct2_gpu,
                'cpu_pct2': pct2_cpu
            })
            
            print(f"File {i+1}: {gpu_file.stem}")
            print(f"  GPU: RMSE={rmse_gpu:.3f}, %>|π/2|={pct2_gpu:.1f}%")
            print(f"  CPU: RMSE={rmse_cpu:.3f}, %>|π/2|={pct2_cpu:.1f}%")
            print()
            
        except Exception as e:
            print(f"Error processing {gpu_file.stem}: {e}")
            continue
    
    if results:
        # Summary
        gpu_rmse = [r['gpu_rmse'] for r in results]
        cpu_rmse = [r['cpu_rmse'] for r in results]
        gpu_pct2 = [r['gpu_pct2'] for r in results]
        cpu_pct2 = [r['cpu_pct2'] for r in results]
        
        print("SUMMARY (0040Hz):")
        print(f"GPU RMSE: {np.mean(gpu_rmse):.3f} ± {np.std(gpu_rmse):.3f}")
        print(f"CPU RMSE: {np.mean(cpu_rmse):.3f} ± {np.std(cpu_rmse):.3f}")
        print(f"GPU %>|π/2|: {np.mean(gpu_pct2):.1f}% ± {np.std(gpu_pct2):.1f}%")
        print(f"CPU %>|π/2|: {np.mean(cpu_pct2):.1f}% ± {np.std(cpu_pct2):.1f}%")
        
        # Improvement
        rmse_improvement = (np.mean(gpu_rmse) - np.mean(cpu_rmse)) / np.mean(gpu_rmse) * 100
        pct2_improvement = (np.mean(gpu_pct2) - np.mean(cpu_pct2)) / np.mean(gpu_pct2) * 100
        
        print(f"\nCPU vs GPU improvement:")
        print(f"RMSE: {rmse_improvement:.1f}% better")
        print(f"%>|π/2|: {pct2_improvement:.1f}% better")

if __name__ == "__main__":
    main()

