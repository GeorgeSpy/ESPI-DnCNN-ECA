#!/usr/bin/env python3
"""
Monitor training progress and prepare for next steps.
"""
import pandas as pd
import time
from pathlib import Path

def monitor_training():
    log_file = Path("C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/train_log.csv")
    checkpoint_dir = Path("C:/ESPI_TEMP/denoise_finetune_GPU_FULLSET/checkpoints")
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            if log_file.exists():
                try:
                    df = pd.read_csv(log_file)
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        print(f"\rEpoch {latest['epoch']:03d} | "
                              f"Train Loss: {latest['train_loss']:.4f} | "
                              f"Val Loss: {latest['val_loss']:.4f} | "
                              f"PSNR: {latest['val_psnr']:.2f} | "
                              f"SSIM: {latest['val_ssim']:.4f} | "
                              f"EdgeF1: {latest['val_edgeF1']:.4f} | "
                              f"LR: {latest['lr']:.2e}", end="")
                except Exception as e:
                    print(f"\nError reading log: {e}")
            
            # Check for new checkpoints
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                print(f"\nLatest checkpoint: {latest_checkpoint.name}")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_training()

