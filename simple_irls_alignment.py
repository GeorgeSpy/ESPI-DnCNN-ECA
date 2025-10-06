#!/usr/bin/env python3
"""
Simple IRLS Alignment - Works with current directory structure
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_erosion

def run_irls_alignment(phase_dir, ref_dir, roi_mask, output_dir):
    """Run IRLS alignment for a single frequency directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ROI mask
    m = np.array(Image.open(roi_mask))
    m = m[..., 0] if m.ndim == 3 else m
    m = (m > 0)
    m = binary_erosion(m, iterations=3)
    
    # Get phase files
    phase_files = list(Path(phase_dir).glob("*.npy"))
    if not phase_files:
        print(f"No phase files found in {phase_dir}")
        return False
    
    # Get reference file (same frequency)
    freq_name = Path(phase_dir).parent.name
    ref_file = Path(ref_dir) / f"{freq_name}.npy"
    
    if not ref_file.exists():
        print(f"Reference file not found: {ref_file}")
        return False
    
    print(f"Processing {len(phase_files)} phase files for {freq_name}")
    
    # Load reference
    wr = np.load(ref_file).astype(np.float32)
    
    results = []
    
    for phase_file in phase_files:
        try:
            # Load phase
            wt = np.load(phase_file).astype(np.float32)
            
            # Calculate difference
            d = np.angle(np.exp(1j * (wt - wr))).astype(np.float32)
            d = np.unwrap(np.unwrap(d, axis=1), axis=0).astype(np.float32)
            
            # Apply ROI
            H, W = d.shape
            yy, xx = np.mgrid[0:H, 0:W]
            X = np.stack([xx[m], yy[m], np.ones(np.count_nonzero(m), np.float32)], 1)
            y = d[m][:, None]
            
            # Simple linear regression (plane fitting)
            if X.shape[0] > 3:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                a, b, c = coeffs.flatten()
                
                # Calculate RMSE
                plane = a * xx + b * yy + c
                residual = d - plane
                rmse = np.sqrt(np.mean(residual[m] ** 2))
                
                # Calculate percentage > pi/2
                high_error = np.sum(np.abs(residual[m]) > np.pi/2)
                total_pixels = np.sum(m)
                pct_high_error = (high_error / total_pixels) * 100 if total_pixels > 0 else 0
                
                results.append({
                    "file": phase_file.name,
                    "rmse": float(rmse),
                    "pct_high_error": float(pct_high_error),
                    "plane_coeffs": [float(a), float(b), float(c)]
                })
                
                print(f"  {phase_file.name}: RMSE={rmse:.3f}, %>π/2={pct_high_error:.1f}%")
            else:
                print(f"  {phase_file.name}: Insufficient pixels in ROI")
                
        except Exception as e:
            print(f"  {phase_file.name}: Error - {str(e)}")
    
    # Save results
    if results:
        results_file = Path(output_dir) / "irls_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate summary statistics
        rmse_values = [r["rmse"] for r in results]
        pct_values = [r["pct_high_error"] for r in results]
        
        summary = {
            "frequency": freq_name,
            "total_files": len(results),
            "median_rmse": float(np.median(rmse_values)),
            "mean_rmse": float(np.mean(rmse_values)),
            "std_rmse": float(np.std(rmse_values)),
            "median_pct_high_error": float(np.median(pct_values)),
            "mean_pct_high_error": float(np.mean(pct_values)),
            "std_pct_high_error": float(np.std(pct_values))
        }
        
        summary_file = Path(output_dir) / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {results_file}")
        print(f"Summary: Median RMSE={summary['median_rmse']:.3f}, Median %>π/2={summary['median_pct_high_error']:.1f}%")
        
        return True
    
    return False

def main():
    """Main execution function"""
    
    if len(sys.argv) != 5:
        print("Usage: python simple_irls_alignment.py <phase_dir> <ref_dir> <roi_mask> <output_dir>")
        return 1
    
    phase_dir = sys.argv[1]
    ref_dir = sys.argv[2]
    roi_mask = sys.argv[3]
    output_dir = sys.argv[4]
    
    success = run_irls_alignment(phase_dir, ref_dir, roi_mask, output_dir)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
