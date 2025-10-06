#!/usr/bin/env python3
"""
Prepare smoke QA commands for when new best checkpoint is available.
"""
from pathlib import Path

def create_smoke_qa_commands():
    """Create smoke QA commands for testing the new denoiser."""
    
    # Base paths
    py_exe = "C:\\ESPI_VENV2\\Scripts\\python.exe"
    ckpt = "C:\\ESPI_TEMP\\denoise_finetune_GPU_FULLSET\\checkpoints\\best_finetune.pth"
    roi_mask = "C:\\ESPI_TEMP\\roi_mask.png"
    
    # Test frequencies and datasets
    test_cases = [
        ("W01", "0050Hz_90.0db", "C:\\ESPI\\data\\wood_real_A\\W01_ESPI_90db\\0050Hz_90.0db"),
        ("W01", "0040Hz_90.0db", "C:\\ESPI\\data\\wood_real_A\\W01_ESPI_90db\\0040Hz_90.0db"),
        ("W02", "0050Hz_90.0db", "C:\\ESPI\\data\\wood_real_B\\W02_ESPI_90db\\0050Hz_90.0db"),
        ("W03", "0050Hz_90.0db", "C:\\ESPI\\data\\wood_real_C\\W03_ESPI_90db\\0050Hz_90.0db"),
    ]
    
    commands = []
    
    for dataset, freq, input_dir in test_cases:
        output_dir = f"C:\\ESPI_TEMP\\SMOKE_GPUFULL\\{dataset}_CLEAN_{freq.replace('Hz_', 'Hz_')}"
        phase_dir = f"C:\\ESPI_TEMP\\SMOKE_GPUFULL\\{dataset}_PhaseOut_b18_cs16_ff100_{freq.replace('Hz_', 'Hz_')}"
        ref_dir = f"C:\\ESPI_TEMP\\SMOKE\\{dataset}_PhaseRef{freq.replace('Hz_', 'Hz_')}_b18_cs16_ff100"
        
        # Denoise command
        denoise_cmd = f"""
# Denoise {dataset} {freq}
$PY="{py_exe}"
$CK="{ckpt}"
$IN="{input_dir}"
$OUT="{output_dir}"

& $PY C:\\ESPI_DnCNN\\batch_denoise_from_compat_NORM.py `
  --ckpt $CK --input $IN --output $OUT `
  --tile 1400 --overlap 0 --device cuda `
  --predicts-residual --norm-mode u16 --save-u16
"""
        
        # FFT probe commands
        avg_file = f"C:\\ESPI\\data\\wood_Averaged\\{dataset}_ESPI_90db-Averaged\\{freq}.png"
        fft_cmd = f"""
# FFT probe {dataset} {freq}
& $PY C:\\ESPI_DnCNN\\fft_peak_probe.py --img "{avg_file}" --cs 16 --rmin 8 --rmax 300
& $PY C:\\ESPI_DnCNN\\fft_peak_probe.py --img "$OUT\\{freq}_00.png" --cs 16 --rmin 8 --rmax 300
"""
        
        # Phase extraction command
        phase_cmd = f"""
# Phase extraction {dataset} {freq}
& $PY C:\\ESPI_DnCNN\\phase_extract_fft_STRICT_FIXED.py `
  --input-dir  "$OUT" `
  --output-dir "{phase_dir}" `
  --band 18 --center-suppress 16 --flatfield 100 --annulus 8 300 `
  --roi-mask "{roi_mask}" --unwrap auto
"""
        
        # QC comparison command
        qc_cmd = f"""
# QC comparison {dataset} {freq}
& $PY C:\\ESPI_DnCNN\\phase_qc_compare_two_roots.py `
  --out-root "{phase_dir}" `
  --ref-root "{ref_dir}" `
  --roi-mask "{roi_mask}" --qmin 0.10 --save-maps
"""
        
        commands.append({
            'dataset': dataset,
            'frequency': freq,
            'denoise': denoise_cmd,
            'fft': fft_cmd,
            'phase': phase_cmd,
            'qc': qc_cmd
        })
    
    return commands

def save_commands(commands):
    """Save commands to files for easy execution."""
    output_dir = Path("C:/ESPI_TEMP/smoke_qa_commands")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cmd_set in commands:
        dataset = cmd_set['dataset']
        freq = cmd_set['frequency'].replace('Hz_', 'Hz_')
        
        # Save individual commands
        for cmd_type, cmd_content in cmd_set.items():
            if cmd_type not in ['dataset', 'frequency']:
                cmd_file = output_dir / f"{dataset}_{freq}_{cmd_type}.ps1"
                with open(cmd_file, 'w', encoding='utf-8') as f:
                    f.write(cmd_content)
        
        # Save combined command
        combined_cmd = "\n".join([
            cmd_set['denoise'],
            cmd_set['fft'],
            cmd_set['phase'],
            cmd_set['qc']
        ])
        
        combined_file = output_dir / f"{dataset}_{freq}_complete.ps1"
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write(combined_cmd)
    
    print(f"Smoke QA commands saved to: {output_dir}")
    print("Execute these commands once the new best checkpoint is available.")

if __name__ == "__main__":
    commands = create_smoke_qa_commands()
    save_commands(commands)
    
    print("\nSmoke QA Commands Ready!")
    print("=" * 50)
    for cmd_set in commands:
        print(f"{cmd_set['dataset']} {cmd_set['frequency']}: Ready for testing")

