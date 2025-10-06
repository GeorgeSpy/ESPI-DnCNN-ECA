#!/usr/bin/env python3
"""
Execute reference phase extraction from averaged data.
"""
import subprocess
import os

def execute_reference_phase():
    """Execute reference phase extraction for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    roi_mask = "C:/ESPI_TEMP/roi_mask.png"
    
    datasets = {
        "W01": {
            "input": "C:/ESPI/data/wood_Averaged/W01_ESPI_90db-Averaged",
            "output": "C:/ESPI_TEMP/GPU_FULL2/W01_PhaseRef_b18_cs16_ff100"
        },
        "W02": {
            "input": "C:/ESPI/data/wood_Averaged/W02_ESPI_90db-Averaged",
            "output": "C:/ESPI_TEMP/GPU_FULL2/W02_PhaseRef_b18_cs16_ff100"
        },
        "W03": {
            "input": "C:/ESPI/data/wood_Averaged/W03_ESPI_90db-Averaged",
            "output": "C:/ESPI_TEMP/GPU_FULL2/W03_PhaseRef_b18_cs16_ff100"
        }
    }
    
    print("🚀 ESPI Reference Phase Extraction")
    print("=" * 50)
    
    for dataset_name, paths in datasets.items():
        print(f"\n🔧 Reference phase extraction - {dataset_name}")
        
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", paths["input"],
            "--output-dir", paths["output"],
            "--band", "18", "--center-suppress", "16", "--flatfield", "100",
            "--annulus", "8", "300", "--roi-mask", roi_mask, "--unwrap", "auto"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print(f"✅ {dataset_name} reference phase extraction completed")
            else:
                print(f"❌ {dataset_name} reference phase extraction failed: {result.stderr}")
        except Exception as e:
            print(f"💥 {dataset_name} reference phase extraction error: {e}")
    
    print("\n🎉 Reference phase extraction completed!")

if __name__ == "__main__":
    execute_reference_phase()
