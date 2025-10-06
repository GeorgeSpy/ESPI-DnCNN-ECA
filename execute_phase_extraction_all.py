#!/usr/bin/env python3
"""
Execute phase extraction for all completed datasets.
"""
import subprocess
import os
from pathlib import Path

def run_phase_extraction_subdirectory(py_exe, roi_mask, input_dir, output_base, dataset_name):
    """Run phase extraction for a single subdirectory."""
    subdir_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_base, subdir_name)
    
    cmd = [
        py_exe, "C:/ESPI_DnCNN/phase_extract_fft_STRICT_FIXED.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--band", "18", "--center-suppress", "16", "--flatfield", "100",
        "--annulus", "8", "300", "--roi-mask", roi_mask, "--unwrap", "auto"
    ]
    
    print(f"🔧 Phase extraction {dataset_name}: {subdir_name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print(f"✅ {dataset_name}: {subdir_name} completed")
            return True
        else:
            print(f"❌ {dataset_name}: {subdir_name} failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name}: {subdir_name} timeout")
        return False
    except Exception as e:
        print(f"💥 {dataset_name}: {subdir_name} error: {e}")
        return False

def execute_phase_extraction():
    """Execute phase extraction for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    roi_mask = "C:/ESPI_TEMP/roi_mask.png"
    
    datasets = {
        "W01": {
            "input_base": "C:/ESPI_TEMP/GPU_FULL2/W01_CLEAN_u16",
            "output_base": "C:/ESPI_TEMP/GPU_FULL2/W01_PhaseOut_b18_cs16_ff100"
        },
        "W02": {
            "input_base": "C:/ESPI_TEMP/GPU_FULL2/W02_CLEAN_u16",
            "output_base": "C:/ESPI_TEMP/GPU_FULL2/W02_PhaseOut_b18_cs16_ff100"
        },
        "W03": {
            "input_base": "C:/ESPI_TEMP/GPU_FULL2/W03_CLEAN_u16",
            "output_base": "C:/ESPI_TEMP/GPU_FULL2/W03_PhaseOut_b18_cs16_ff100"
        }
    }
    
    print("ESPI Phase Extraction - All Datasets")
    print("=" * 60)
    
    for dataset_name, paths in datasets.items():
        input_base = paths["input_base"]
        output_base = paths["output_base"]
        
        print(f"\n📋 Processing {dataset_name}")
        print(f"Input: {input_base}")
        print(f"Output: {output_base}")
        
        # Create output directory
        os.makedirs(output_base, exist_ok=True)
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(input_base) 
                  if os.path.isdir(os.path.join(input_base, d))]
        
        print(f"Found {len(subdirs)} subdirectories")
        
        success_count = 0
        for subdir in subdirs:
            input_dir = os.path.join(input_base, subdir)
            if run_phase_extraction_subdirectory(py_exe, roi_mask, input_dir, output_base, dataset_name):
                success_count += 1
        
        print(f"✅ {dataset_name}: {success_count}/{len(subdirs)} subdirectories completed")
    
    print("\n🎉 Phase extraction completed for all datasets!")

if __name__ == "__main__":
    execute_phase_extraction()
