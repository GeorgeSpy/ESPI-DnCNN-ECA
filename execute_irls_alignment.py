#!/usr/bin/env python3
"""
Execute IRLS Robust Alignment for all datasets
Memory-safe IRLS processing with per-set ROI
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_irls_alignment(dataset_name, phase_root, roi_mask):
    """Run IRLS alignment for a dataset"""
    
    if not os.path.exists(phase_root):
        print(f"❌ Phase root not found: {phase_root}")
        return False
    
    print(f"🔄 Starting IRLS alignment for {dataset_name}...")
    
    # Use the appropriate IRLS script based on dataset
    if dataset_name == "W01":
        irls_script = r"C:\ESPI_DnCNN\robust_align_irls.py"
    elif dataset_name == "W02":
        irls_script = r"C:\ESPI_DnCNN\w02_irls_fixed.py"
    elif dataset_name == "W03":
        irls_script = r"C:\ESPI_DnCNN\w03_irls.py"
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    if not os.path.exists(irls_script):
        print(f"❌ IRLS script not found: {irls_script}")
        return False
    
    cmd = [
        sys.executable,
        irls_script,
        "--input-dir", phase_root,
        "--roi-mask", roi_mask,
        "--output-dir", os.path.join(phase_root, "qc_align_B_IRLS")
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"✅ IRLS alignment successful for {dataset_name}")
            return True
        else:
            print(f"❌ IRLS alignment failed for {dataset_name}")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exception during IRLS alignment for {dataset_name}: {str(e)}")
        return False

def main():
    """Main execution function"""
    
    # Dataset configurations with appropriate ROI masks
    datasets = [
        {
            "name": "W01",
            "phase_root": r"C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100",
            "roi_mask": r"C:\ESPI_TEMP\roi_mask.png"
        },
        {
            "name": "W02",
            "phase_root": r"C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100", 
            "roi_mask": r"C:\ESPI_TEMP\roi_mask_W02.png"
        },
        {
            "name": "W03",
            "phase_root": r"C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100",
            "roi_mask": r"C:\ESPI_TEMP\roi_mask_W03.png"
        }
    ]
    
    print("🔄 IRLS ALIGNMENT STARTING")
    print("=" * 40)
    
    start_time = time.time()
    successful = 0
    
    for dataset in datasets:
        # Check if ROI mask exists, fallback to default if not
        if not os.path.exists(dataset["roi_mask"]):
            print(f"⚠️  ROI mask not found: {dataset['roi_mask']}")
            print(f"   Using default ROI mask")
            dataset["roi_mask"] = r"C:\ESPI_TEMP\roi_mask.png"
        
        success = run_irls_alignment(
            dataset["name"],
            dataset["phase_root"],
            dataset["roi_mask"]
        )
        if success:
            successful += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 40)
    print(f"🎯 IRLS ALIGNMENT COMPLETE")
    print(f"⏱️  Total time: {duration/60:.1f} minutes")
    print(f"📊 Successful: {successful}/{len(datasets)} datasets")
    
    if successful == len(datasets):
        print("🎉 All IRLS alignments successful!")
        return 0
    else:
        print("⚠️  Some IRLS alignments failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
