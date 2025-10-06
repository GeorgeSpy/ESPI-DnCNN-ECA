#!/usr/bin/env python3
"""
Quick diagnosis of training data and validation issues.
"""
import pandas as pd
import os
import json

def check_usable_samples():
    """Check how many samples are actually usable."""
    CSV = r"C:\ESPI_TEMP\pairs\pairs_mix_80_20_FULL.csv"
    df = pd.read_csv(CSV)
    
    def ok(p): 
        try: 
            return os.path.exists(p)
        except: 
            return False
    
    df["noisy_ok"] = df["noisy"].map(ok)
    df["clean_ok"] = df["clean"].map(ok)
    usable = df[df["noisy_ok"] & df["clean_ok"]]
    
    result = {
        "csv_rows": len(df),
        "usable_rows": len(usable),
        "missing_noisy": int((~df["noisy_ok"]).sum()),
        "missing_clean": int((~df["clean_ok"]).sum())
    }
    
    print("=== USABLE SAMPLES CHECK ===")
    print(json.dumps(result, indent=2))
    return result

def check_expected_steps():
    """Calculate expected training steps per epoch."""
    N = len(pd.read_csv(r"C:\ESPI_TEMP\pairs\pairs_mix_80_20_FULL.csv"))
    train = int(N * 0.85)  # 85% train, 15% val
    batch = 12
    
    print("\n=== EXPECTED STEPS ===")
    print(f"Total samples: {N}")
    print(f"Train samples: {train}")
    print(f"Val samples: {N - train}")
    print(f"Expected train steps (drop_last=True): {train // batch}")
    print(f"Expected val steps (drop_last=True): {(N - train) // batch}")

def check_val_split():
    """Check validation split specifically."""
    csv = r"C:\ESPI_TEMP\pairs\pairs_mix_80_20_FULL.csv"
    df = pd.read_csv(csv)
    N = len(df)
    tr = int(N * 0.85)
    val = df.iloc[tr:]
    
    print("\n=== VALIDATION SPLIT CHECK ===")
    print(f"Val rows: {len(val)}")
    print(f"Val sample paths (first 3):")
    for i, row in val.head(3).iterrows():
        print(f"  {i}: {row['noisy']} -> {row['clean']}")

def check_training_log():
    """Check current training log status."""
    log_file = r"C:\ESPI_TEMP\denoise_finetune_GPU_FULLSET\train_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        print("\n=== TRAINING LOG STATUS ===")
        print(f"Total epochs logged: {len(df)}")
        print("Last few epochs:")
        print(df.tail(3).to_string())
        
        # Check for NaN patterns
        nan_epochs = df[df['val_loss'].isna()]
        print(f"\nEpochs with NaN validation: {len(nan_epochs)}")
        if len(nan_epochs) > 0:
            print("NaN epochs:", nan_epochs['epoch'].tolist())

if __name__ == "__main__":
    check_usable_samples()
    check_expected_steps()
    check_val_split()
    check_training_log()

