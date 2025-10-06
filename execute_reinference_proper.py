#!/usr/bin/env python3
"""
Execute proper re-inference for all datasets with correct directory structure.
"""
import subprocess
import os
from pathlib import Path

def run_denoise_subdirectory(py_exe, ckpt, input_dir, output_base, dataset_name):
    """Run denoising for a single subdirectory."""
    subdir_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_base, subdir_name)
    
    cmd = [
        py_exe, "C:/ESPI_DnCNN/batch_denoise_from_compat_NORM.py",
        "--ckpt", ckpt,
        "--input", input_dir,
        "--output", output_dir,
        "--tile", "1400", "--overlap", "0", "--device", "cuda",
        "--predicts-residual", "--norm-mode", "u16", "--save-u16"
    ]
    
    print(f"🔧 Processing {dataset_name}: {subdir_name}")
    
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

def execute_reinference():
    """Execute re-inference for all datasets."""
    py_exe = "C:/ESPI_VENV2/Scripts/python.exe"
    ckpt = "C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/checkpoints/best.pth"
    
    datasets = {
        "W01": {
            "input_base": "C:/ESPI/data/wood_real_A/W01_ESPI_90db",
            "output_base": "C:/ESPI_TEMP/GPU_FULL2/W01_CLEAN_u16"
        },
        "W02": {
            "input_base": "C:/ESPI/data/wood_real_B/W02_ESPI_90db", 
            "output_base": "C:/ESPI_TEMP/GPU_FULL2/W02_CLEAN_u16"
        },
        "W03": {
            "input_base": "C:/ESPI/data/wood_real_C/W03_ESPI_90db",
            "output_base": "C:/ESPI_TEMP/GPU_FULL2/W03_CLEAN_u16"
        }
    }
    
    print("🚀 ESPI Re-Inference Execution")
    print("=" * 50)
    
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
            if run_denoise_subdirectory(py_exe, ckpt, input_dir, output_base, dataset_name):
                success_count += 1
        
        print(f"✅ {dataset_name}: {success_count}/{len(subdirs)} subdirectories completed")
    
    print("\n🎉 Re-inference completed!")

if __name__ == "__main__":
    execute_reinference()
