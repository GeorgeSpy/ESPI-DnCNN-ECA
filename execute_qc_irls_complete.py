#!/usr/bin/env python3
"""
Execute QC comparison and IRLS robust alignment for all datasets.
"""
import subprocess
import os

def execute_qc_comparison():
    """Execute QC comparison for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    roi_mask = "C:/ESPI_TEMP/roi_mask.png"
    
    datasets = {
        "W01": {
            "out_root": "C:/ESPI_TEMP/GPU_FULL2/W01_PhaseOut_b18_cs16_ff100",
            "ref_root": "C:/ESPI_TEMP/GPU_FULL2/W01_PhaseRef_b18_cs16_ff100"
        },
        "W02": {
            "out_root": "C:/ESPI_TEMP/GPU_FULL2/W02_PhaseOut_b18_cs16_ff100",
            "ref_root": "C:/ESPI_TEMP/GPU_FULL2/W02_PhaseRef_b18_cs16_ff100"
        },
        "W03": {
            "out_root": "C:/ESPI_TEMP/GPU_FULL2/W03_PhaseOut_b18_cs16_ff100",
            "ref_root": "C:/ESPI_TEMP/GPU_FULL2/W03_PhaseRef_b18_cs16_ff100"
        }
    }
    
    print("🚀 ESPI QC Comparison")
    print("=" * 40)
    
    for dataset_name, paths in datasets.items():
        print(f"\n🔧 QC comparison - {dataset_name}")
        
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_qc_compare_two_roots.py",
            "--out-root", paths["out_root"],
            "--ref-root", paths["ref_root"],
            "--roi-mask", roi_mask,
            "--qmin", "0.10", "--save-maps"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print(f"✅ {dataset_name} QC comparison completed")
            else:
                print(f"❌ {dataset_name} QC comparison failed: {result.stderr}")
        except Exception as e:
            print(f"💥 {dataset_name} QC comparison error: {e}")

def execute_irls_alignment():
    """Execute IRLS robust alignment for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    
    # IRLS scripts for each dataset
    irls_scripts = {
        "W01": "C:/ESPI_DnCNN/robust_align_memory_safe.py",  # Already exists
        "W02": "C:/ESPI_DnCNN/w02_irls_fixed.py",
        "W03": "C:/ESPI_DnCNN/w03_irls.py"
    }
    
    print("\n🚀 ESPI IRLS Robust Alignment")
    print("=" * 40)
    
    for dataset_name, script_path in irls_scripts.items():
        print(f"\n🔧 IRLS alignment - {dataset_name}")
        
        cmd = [py_exe, script_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print(f"✅ {dataset_name} IRLS alignment completed")
            else:
                print(f"❌ {dataset_name} IRLS alignment failed: {result.stderr}")
        except Exception as e:
            print(f"💥 {dataset_name} IRLS alignment error: {e}")

def main():
    """Execute QC and IRLS for all datasets."""
    print("🚀 ESPI QC & IRLS Complete Pipeline")
    print("=" * 60)
    
    # Execute QC comparison
    execute_qc_comparison()
    
    # Execute IRLS alignment
    execute_irls_alignment()
    
    print("\n🎉 QC & IRLS pipeline completed!")

if __name__ == "__main__":
    main()
