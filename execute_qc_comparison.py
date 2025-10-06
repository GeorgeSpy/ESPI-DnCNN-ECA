#!/usr/bin/env python3
"""
Execute QC Comparison for all datasets
Compares denoised phase with reference phase
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_qc_comparison(out_root, ref_root, roi_mask, dataset_name):
    """Run QC comparison for a dataset"""
    
    if not os.path.exists(out_root):
        print(f"❌ Output root not found: {out_root}")
        return False
    
    if not os.path.exists(ref_root):
        print(f"❌ Reference root not found: {ref_root}")
        return False
    
    print(f"🔍 Starting QC comparison for {dataset_name}...")
    
    cmd = [
        sys.executable,
        r"C:\ESPI_DnCNN\phase_qc_compare_two_roots.py",
        "--out-root", out_root,
        "--ref-root", ref_root,
        "--roi-mask", roi_mask,
        "--qmin", "0.10",
        "--save-maps"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"✅ QC comparison successful for {dataset_name}")
            print(f"   Output: {result.stdout[-200:]}...")
            return True
        else:
            print(f"❌ QC comparison failed for {dataset_name}")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exception during QC comparison for {dataset_name}: {str(e)}")
        return False

def main():
    """Main execution function"""
    
    roi_mask = r"C:\ESPI_TEMP\roi_mask.png"
    
    # Dataset configurations
    datasets = [
        {
            "name": "W01",
            "out_root": r"C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100",
            "ref_root": r"C:\ESPI_TEMP\GPU_FULL2\W01_PhaseRef_b18_cs16_ff100"
        },
        {
            "name": "W02", 
            "out_root": r"C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100",
            "ref_root": r"C:\ESPI_TEMP\GPU_FULL2\W02_PhaseRef_b18_cs16_ff100"
        },
        {
            "name": "W03",
            "out_root": r"C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100", 
            "ref_root": r"C:\ESPI_TEMP\GPU_FULL2\W03_PhaseRef_b18_cs16_ff100"
        }
    ]
    
    print("🔍 QC COMPARISON STARTING")
    print("=" * 40)
    
    start_time = time.time()
    successful = 0
    
    for dataset in datasets:
        success = run_qc_comparison(
            dataset["out_root"], 
            dataset["ref_root"], 
            roi_mask, 
            dataset["name"]
        )
        if success:
            successful += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 40)
    print(f"🎯 QC COMPARISON COMPLETE")
    print(f"⏱️  Total time: {duration:.1f} seconds")
    print(f"📊 Successful: {successful}/{len(datasets)} datasets")
    
    if successful == len(datasets):
        print("🎉 All QC comparisons successful!")
        return 0
    else:
        print("⚠️  Some QC comparisons failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
