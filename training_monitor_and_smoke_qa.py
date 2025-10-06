#!/usr/bin/env python3
"""
Comprehensive training monitoring and smoke QA framework.
Based on the go/no-go criteria and green-light table.
"""
import pandas as pd
import time
import subprocess
from pathlib import Path
import json

class TrainingMonitor:
    def __init__(self):
        self.log_file = Path("C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/train_log.csv")
        self.checkpoint_dir = Path("C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/checkpoints")
        self.baseline_metrics = {
            'psnr': 22.3,  # Baseline PSNR
            'wrapped_rmse': 1.2,  # Baseline wrapped RMSE
            'irls_rmse_w01': 3.47,  # Baseline IRLS RMSE W01
            'pct_pi2_w01': 26.0  # Baseline %>|π/2| W01
        }
        
    def check_training_progress(self):
        """Check current training progress against go/no-go criteria."""
        if not self.log_file.exists():
            return {"status": "no_log", "message": "Training log not found"}
        
        try:
            df = pd.read_csv(self.log_file)
            if len(df) == 0:
                return {"status": "no_data", "message": "No training data yet"}
            
            # Get latest valid epoch (non-nan validation)
            valid_epochs = df.dropna(subset=['val_loss'])
            if len(valid_epochs) == 0:
                return {"status": "validation_issue", "message": "Validation metrics are nan"}
            
            latest = valid_epochs.iloc[-1]
            
            # Go/No-go criteria
            criteria = {
                'epoch': latest['epoch'],
                'train_loss': latest['train_loss'],
                'val_loss': latest['val_loss'],
                'psnr': latest['val_psnr'],
                'ssim': latest['val_ssim'],
                'edge_f1': latest['val_edgeF1'],
                'lr': latest['lr']
            }
            
            # Check improvements
            improvements = {
                'psnr_improvement': latest['val_psnr'] - self.baseline_metrics['psnr'],
                'psnr_target_met': latest['val_psnr'] >= (self.baseline_metrics['psnr'] + 0.2),
                'val_loss_decreasing': len(valid_epochs) > 1 and latest['val_loss'] < valid_epochs.iloc[-2]['val_loss']
            }
            
            # Decision
            if improvements['psnr_target_met'] and improvements['val_loss_decreasing']:
                decision = "GO"
            elif latest['epoch'] >= 6:
                decision = "LR_REDUCTION_NEEDED"
            else:
                decision = "CONTINUE"
            
            return {
                "status": "success",
                "criteria": criteria,
                "improvements": improvements,
                "decision": decision
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def check_checkpoints(self):
        """Check if new checkpoints are available."""
        if not self.checkpoint_dir.exists():
            return {"status": "no_checkpoints", "message": "Checkpoint directory not found"}
        
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            return {"status": "no_checkpoints", "message": "No checkpoints found"}
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return {
            "status": "success",
            "checkpoints": [cp.name for cp in checkpoints],
            "latest": latest_checkpoint.name,
            "path": str(latest_checkpoint)
        }

class SmokeQARunner:
    def __init__(self):
        self.py_exe = "C:\\ESPI_VENV2\\Scripts\\python.exe"
        self.test_cases = [
            {
                "name": "W01_0050Hz",
                "input": "C:\\ESPI\\data\\wood_real_A\\W01_ESPI_90db\\0050Hz_90.0db",
                "output": "C:\\ESPI_TEMP\\SMOKE_GPUFULL\\W01_CLEAN_0050",
                "reference": "C:\\ESPI\\data\\wood_Averaged\\W01_ESPI_90db-Averaged\\0050Hz_90.0db.png"
            },
            {
                "name": "W02_0125Hz", 
                "input": "C:\\ESPI\\data\\wood_real_B\\W02_ESPI_90db\\0125Hz_90.0db",
                "output": "C:\\ESPI_TEMP\\SMOKE_GPUFULL\\W02_CLEAN_0125",
                "reference": "C:\\ESPI\\data\\wood_Averaged\\W02_ESPI_90db-Averaged\\0125Hz_90.0db.png"
            },
            {
                "name": "W03_0045Hz",
                "input": "C:\\ESPI\\data\\wood_real_C\\W03_ESPI_90db\\0045Hz_90.0db", 
                "output": "C:\\ESPI_TEMP\\SMOKE_GPUFULL\\W03_CLEAN_0045",
                "reference": "C:\\ESPI\\data\\wood_Averaged\\W03_ESPI_90db-Averaged\\0045Hz_90.0db.png"
            }
        ]
    
    def run_smoke_qa(self, checkpoint_path):
        """Run smoke QA on the new checkpoint."""
        results = {}
        
        for test_case in self.test_cases:
            print(f"\nRunning smoke QA for {test_case['name']}...")
            
            # 1. Denoise
            denoise_cmd = [
                self.py_exe, "C:\\ESPI_DnCNN\\batch_denoise_from_compat_NORM.py",
                "--ckpt", checkpoint_path,
                "--input", test_case['input'],
                "--output", test_case['output'],
                "--tile", "1400", "--overlap", "0", "--device", "cuda",
                "--predicts-residual", "--norm-mode", "u16", "--save-u16"
            ]
            
            try:
                result = subprocess.run(denoise_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    results[test_case['name']] = {"status": "denoise_failed", "error": result.stderr}
                    continue
            except subprocess.TimeoutExpired:
                results[test_case['name']] = {"status": "denoise_timeout"}
                continue
            
            # 2. FFT Probe
            fft_cmd = [
                self.py_exe, "C:\\ESPI_DnCNN\\fft_peak_probe.py",
                "--img", test_case['reference'], "--cs", "16", "--rmin", "8", "--rmax", "300"
            ]
            
            try:
                result = subprocess.run(fft_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    # Parse FFT results (simplified)
                    fft_output = result.stdout
                    results[test_case['name']] = {
                        "status": "success",
                        "fft_reference": fft_output.strip()
                    }
                else:
                    results[test_case['name']] = {"status": "fft_failed", "error": result.stderr}
            except subprocess.TimeoutExpired:
                results[test_case['name']] = {"status": "fft_timeout"}
        
        return results

def main():
    monitor = TrainingMonitor()
    smoke_qa = SmokeQARunner()
    
    print("🚀 ESPI Training Monitor & Smoke QA")
    print("=" * 50)
    
    while True:
        # Check training progress
        progress = monitor.check_training_progress()
        print(f"\n📊 Training Status: {progress['status']}")
        
        if progress['status'] == 'success':
            criteria = progress['criteria']
            improvements = progress['improvements']
            decision = progress['decision']
            
            print(f"Epoch {criteria['epoch']:03d} | "
                  f"Train: {criteria['train_loss']:.4f} | "
                  f"Val: {criteria['val_loss']:.4f} | "
                  f"PSNR: {criteria['psnr']:.2f} | "
                  f"SSIM: {criteria['ssim']:.4f} | "
                  f"EdgeF1: {criteria['edge_f1']:.4f}")
            
            print(f"PSNR Improvement: {improvements['psnr_improvement']:+.2f} dB")
            print(f"Decision: {decision}")
            
            if decision == "LR_REDUCTION_NEEDED":
                print("⚠️  Consider reducing LR to 7e-5 for polish epochs")
        
        # Check checkpoints
        checkpoints = monitor.check_checkpoints()
        if checkpoints['status'] == 'success':
            print(f"📁 Checkpoints: {checkpoints['checkpoints']}")
            print(f"Latest: {checkpoints['latest']}")
            
            # Run smoke QA if we have a new best checkpoint
            if 'best.pth' in checkpoints['checkpoints']:
                print("\n🧪 Running Smoke QA...")
                results = smoke_qa.run_smoke_qa(checkpoints['path'])
                
                print("\n📋 Smoke QA Results:")
                for name, result in results.items():
                    print(f"  {name}: {result['status']}")
        
        print("\n" + "="*50)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

