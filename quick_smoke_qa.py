#!/usr/bin/env python3
"""
Quick smoke QA test on the current best checkpoint.
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def main():
    # Check if checkpoint exists
    checkpoint = Path("C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/checkpoints/best.pth")
    if not checkpoint.exists():
        print("❌ Best checkpoint not found!")
        return
    
    print("🚀 Quick Smoke QA Test")
    print("=" * 40)
    print(f"Checkpoint: {checkpoint}")
    
    # Test case: W01 0050Hz
    test_input = "C:/ESPI/data/wood_real_A/W01_ESPI_90db/0050Hz_90.0db"
    test_output = "C:/ESPI_TEMP/SMOKE_QUICK/W01_CLEAN_0050"
    test_reference = "C:/ESPI/data/wood_Averaged/W01_ESPI_90db-Averaged/0050Hz_90.0db.png"
    
    # Create output directory
    Path(test_output).mkdir(parents=True, exist_ok=True)
    
    # 1. Denoise
    denoise_cmd = [
        "C:/ESPI_VENV2/Scripts/python.exe",
        "C:/ESPI_DnCNN/batch_denoise_from_compat_NORM.py",
        "--ckpt", str(checkpoint),
        "--input", test_input,
        "--output", test_output,
        "--tile", "1400", "--overlap", "0", "--device", "cuda",
        "--predicts-residual", "--norm-mode", "u16", "--save-u16"
    ]
    
    if not run_command(denoise_cmd, "Denoising W01 0050Hz"):
        return
    
    # 2. FFT Probe - Reference
    fft_ref_cmd = [
        "C:/ESPI_VENV2/Scripts/python.exe",
        "C:/ESPI_DnCNN/fft_peak_probe.py",
        "--img", test_reference,
        "--cs", "16", "--rmin", "8", "--rmax", "300"
    ]
    
    if not run_command(fft_ref_cmd, "FFT Probe - Reference"):
        return
    
    # 3. FFT Probe - Denoised
    denoised_file = Path(test_output) / "0050Hz_90.0db_00.png"
    if denoised_file.exists():
        fft_denoised_cmd = [
            "C:/ESPI_VENV2/Scripts/python.exe",
            "C:/ESPI_DnCNN/fft_peak_probe.py",
            "--img", str(denoised_file),
            "--cs", "16", "--rmin", "8", "--rmax", "300"
        ]
        
        if not run_command(fft_denoised_cmd, "FFT Probe - Denoised"):
            return
    else:
        print(f"❌ Denoised file not found: {denoised_file}")
        return
    
    print("\n🎉 Quick Smoke QA Completed!")
    print("Check the FFT outputs above for Δr≈0, Δθ≈0")

if __name__ == "__main__":
    main()

