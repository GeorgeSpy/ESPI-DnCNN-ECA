#!/usr/bin/env python3
"""
Complete Phase Extraction for W02 and W03
Processes all frequency subdirectories efficiently
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_phase_extraction(input_dir, output_dir, roi_mask, dataset_name):
    """Run phase extraction for all subdirectories in a dataset"""
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Starting {dataset_name} phase extraction...")
    print(f"Found {len(subdirs)} frequency subdirectories")
    
    completed = 0
    failed = 0
    
    for i, subdir in enumerate(subdirs):
        input_path = os.path.join(input_dir, subdir)
        output_path = os.path.join(output_dir, subdir)
        
        # Skip if already processed
        if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "phase_wrapped_npy")):
            print(f"⏭️  Skipping {subdir} (already processed)")
            completed += 1
            continue
        
        print(f"🔄 Processing {i+1}/{len(subdirs)}: {subdir}")
        
        cmd = [
            sys.executable, 
            r"C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", input_path,
            "--output-dir", output_path,
            "--band", "18",
            "--center-suppress", "16", 
            "--flatfield", "100",
            "--annulus", "8", "300",
            "--roi-mask", roi_mask,
            "--unwrap", "auto"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print(f"  ✅ Success: {subdir}")
                completed += 1
            else:
                print(f"  ❌ Error: {subdir}")
                print(f"     {result.stderr[:200]}...")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Exception: {subdir} - {str(e)}")
            failed += 1
        
        # Progress update every 10 directories
        if (i + 1) % 10 == 0:
            print(f"📊 Progress: {i+1}/{len(subdirs)} ({completed} completed, {failed} failed)")
    
    print(f"🎯 {dataset_name} Complete: {completed} successful, {failed} failed")
    return failed == 0

def main():
    """Main execution function"""
    
    roi_mask = r"C:\ESPI_TEMP\roi_mask.png"
    
    # W02 Phase Extraction
    w02_input = r"C:\ESPI_TEMP\GPU_FULL2\W02_CLEAN_u16"
    w02_output = r"C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100"
    
    # W03 Phase Extraction  
    w03_input = r"C:\ESPI_TEMP\GPU_FULL2\W03_CLEAN_u16"
    w03_output = r"C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"
    
    print("🚀 COMPLETE PHASE EXTRACTION STARTING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Process W02
    w02_success = run_phase_extraction(w02_input, w02_output, roi_mask, "W02")
    
    # Process W03
    w03_success = run_phase_extraction(w03_input, w03_output, roi_mask, "W03")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"🎉 PHASE EXTRACTION COMPLETE")
    print(f"⏱️  Total time: {duration/60:.1f} minutes")
    print(f"📊 W02: {'✅ Success' if w02_success else '❌ Failed'}")
    print(f"📊 W03: {'✅ Success' if w03_success else '❌ Failed'}")
    
    if w02_success and w03_success:
        print("🎯 All datasets processed successfully!")
        return 0
    else:
        print("⚠️  Some datasets had issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
