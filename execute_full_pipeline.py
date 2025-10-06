#!/usr/bin/env python3
"""
Execute the complete ESPI pipeline: Phase → QC → IRLS → Features → RF
"""
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout (30 min)")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def execute_phase_extraction():
    """Execute phase extraction for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    
    datasets = ["W01", "W02", "W03"]
    
    for dataset in datasets:
        # Phase extraction for denoised data
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_CLEAN_u16",
            "--output-dir", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseOut_b18_cs16_ff100",
            "--band", "18", "--center-suppress", "16", "--flatfield", "100",
            "--annulus", "8", "300", "--roi-mask", "C:/ESPI_TEMP/roi_mask.png",
            "--unwrap", "auto"
        ]
        
        if not run_command(cmd, f"Phase extraction - {dataset}"):
            return False
        
        # Phase extraction for reference (averaged) data
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_extract_fft_STRICT_FIXED.py",
            "--input-dir", f"C:/ESPI/data/wood_Averaged/{dataset}_ESPI_90db-Averaged",
            "--output-dir", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseRef_b18_cs16_ff100",
            "--band", "18", "--center-suppress", "16", "--flatfield", "100",
            "--annulus", "8", "300", "--roi-mask", "C:/ESPI_TEMP/roi_mask.png",
            "--unwrap", "auto"
        ]
        
        if not run_command(cmd, f"Reference phase extraction - {dataset}"):
            return False
    
    return True

def execute_qc_comparison():
    """Execute QC comparison for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    datasets = ["W01", "W02", "W03"]
    
    for dataset in datasets:
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_qc_compare_two_roots.py",
            "--out-root", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseOut_b18_cs16_ff100",
            "--ref-root", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseRef_b18_cs16_ff100",
            "--roi-mask", "C:/ESPI_TEMP/roi_mask.png",
            "--qmin", "0.10", "--save-maps"
        ]
        
        if not run_command(cmd, f"QC comparison - {dataset}"):
            return False
    
    return True

def execute_irls_alignment():
    """Execute IRLS robust alignment for W02 and W03."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    
    # W02 IRLS
    cmd = [py_exe, "C:/ESPI_DnCNN/w02_irls_fixed.py"]
    if not run_command(cmd, "IRLS robust alignment - W02"):
        return False
    
    # W03 IRLS  
    cmd = [py_exe, "C:/ESPI_DnCNN/w03_irls.py"]
    if not run_command(cmd, "IRLS robust alignment - W03"):
        return False
    
    return True

def execute_features_extraction():
    """Execute features extraction and processing."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    
    # Features extraction for each dataset
    datasets = ["W01", "W02", "W03"]
    for dataset in datasets:
        cmd = [
            py_exe, "C:/ESPI_DnCNN/phase_nodal_features_min.py",
            "--band-root", f"C:/ESPI_TEMP/GPU_FULL2/{dataset}_PhaseOut_b18_cs16_ff100"
        ]
        
        if not run_command(cmd, f"Features extraction - {dataset}"):
            return False
    
    # Merge features
    cmd = [py_exe, "C:/ESPI_DnCNN/merge_all_features.py"]
    if not run_command(cmd, "Merge all features"):
        return False
    
    # Deduplication
    cmd = [
        py_exe, "C:/ESPI_DnCNN/simple_dedup.py",
        "--inp", "C:/ESPI_TEMP/features/all_features_QCpass.csv",
        "--out", "C:/ESPI_TEMP/features/all_features_QCpass_dedup.csv"
    ]
    if not run_command(cmd, "Deduplication"):
        return False
    
    # Numeric features
    cmd = [
        py_exe, "C:/ESPI_DnCNN/create_numeric_features.py",
        "--inp", "C:/ESPI_TEMP/features/all_features_QCpass_dedup.csv",
        "--out", "C:/ESPI_TEMP/features/features_numeric_only.csv"
    ]
    if not run_command(cmd, "Numeric features extraction"):
        return False
    
    return True

def execute_final_rf():
    """Execute final RF training."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    
    # Create labels
    cmd = [py_exe, "C:/ESPI_DnCNN/create_labels_corrected.py"]
    if not run_command(cmd, "Create labels"):
        return False
    
    # Hierarchical RF training
    cmd = [py_exe, "C:/ESPI_DnCNN/hierarchical_rf_classifier.py"]
    if not run_command(cmd, "Hierarchical RF training"):
        return False
    
    return True

def main():
    """Execute the complete pipeline."""
    print("🚀 ESPI Complete Pipeline Execution")
    print("=" * 60)
    
    steps = [
        ("Phase Extraction", execute_phase_extraction),
        ("QC Comparison", execute_qc_comparison),
        ("IRLS Alignment", execute_irls_alignment),
        ("Features Extraction", execute_features_extraction),
        ("Final RF Training", execute_final_rf)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Executing: {step_name}")
        print("-" * 40)
        
        if step_func():
            print(f"✅ {step_name} completed successfully")
        else:
            print(f"❌ {step_name} failed")
            return False
    
    print("\n🎉 Complete pipeline executed successfully!")
    return True

if __name__ == "__main__":
    main()
