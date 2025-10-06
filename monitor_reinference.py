#!/usr/bin/env python3
"""
Monitor re-inference progress and execute next phases automatically.
"""
import os
import time
import subprocess
from pathlib import Path

def count_png_files(directory):
    """Count PNG files in directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith('.png')])

def monitor_progress():
    """Monitor re-inference progress."""
    expected_counts = {
        "W01": 3923,
        "W02": 4029, 
        "W03": 2989
    }
    
    directories = {
        "W01": "C:/ESPI_TEMP/GPU_FULL2/W01_CLEAN_u16",
        "W02": "C:/ESPI_TEMP/GPU_FULL2/W02_CLEAN_u16",
        "W03": "C:/ESPI_TEMP/GPU_FULL2/W03_CLEAN_u16"
    }
    
    print("🚀 ESPI Re-Inference Monitor")
    print("=" * 50)
    
    while True:
        print(f"\n📊 Progress - {time.strftime('%H:%M:%S')}")
        
        all_complete = True
        for dataset, directory in directories.items():
            current_count = count_png_files(directory)
            expected = expected_counts[dataset]
            progress = (current_count / expected * 100) if expected > 0 else 0
            
            status = "✅ COMPLETE" if current_count >= expected else f"🔄 {progress:.1f}%"
            print(f"  {dataset}: {current_count}/{expected} files {status}")
            
            if current_count < expected:
                all_complete = False
        
        if all_complete:
            print("\n🎉 All re-inference processes completed!")
            print("🚀 Starting next phases...")
            return True
            
        print("\n⏳ Waiting 60 seconds...")
        time.sleep(60)

def execute_next_phases():
    """Execute the next phases of the pipeline."""
    print("\n📋 Executing Phase Extraction...")
    
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    roi_mask = "C:/ESPI_TEMP/roi_mask.png"
    
    datasets = ["W01", "W02", "W03"]
    
    # Phase extraction for denoised data
    for dataset in datasets:
        print(f"\n🔧 Phase extraction - {dataset}")
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_CLEAN_u16",
            "--output-dir", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseOut_b18_cs16_ff100",
            "--band", "18", "--center-suppress", "16", "--flatfield", "100",
            "--annulus", "8", "300", "--roi-mask", roi_mask, "--unwrap", "auto"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print(f"✅ {dataset} phase extraction completed")
            else:
                print(f"❌ {dataset} phase extraction failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"💥 {dataset} phase extraction error: {e}")
            return False
    
    # Phase extraction for reference data
    for dataset in datasets:
        print(f"\n🔧 Reference phase extraction - {dataset}")
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", f"C:/ESPI/data/wood_Averaged/{dataset}_ESPI_90db-Averaged",
            "--output-dir", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseRef_b18_cs16_ff100",
            "--band", "18", "--center-suppress", "16", "--flatfield", "100",
            "--annulus", "8", "300", "--roi-mask", roi_mask, "--unwrap", "auto"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print(f"✅ {dataset} reference phase extraction completed")
            else:
                print(f"❌ {dataset} reference phase extraction failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"💥 {dataset} reference phase extraction error: {e}")
            return False
    
    print("\n✅ All phase extractions completed!")
    return True

def main():
    """Main monitoring and execution function."""
    if monitor_progress():
        if execute_next_phases():
            print("\n🎉 Phase extraction completed successfully!")
            print("📋 Next: QC comparison, IRLS alignment, Features, RF training")
        else:
            print("\n❌ Phase extraction failed")
    else:
        print("\n⏳ Re-inference still in progress")

if __name__ == "__main__":
    main()
