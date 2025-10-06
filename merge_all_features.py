#!/usr/bin/env python3
import pandas as pd, re, pathlib

base = pathlib.Path(r"C:\ESPI_TEMP\features")

def load(tag):
    if tag == "W01":
        df = pd.read_csv(base / "W01.csv")
    elif tag == "W02":
        df = pd.read_csv(base / "W02.csv")
    else:  # W03
        df = pd.read_csv(base / "W03_features.csv")
    
    # Use 'id' column as basename
    df["basename"] = df["id"]
    df["set"] = tag
    return df

# Load all features
F = pd.concat([load("W01"), load("W02"), load("W03")], ignore_index=True)

def qc(tag):
    q = pd.read_csv(fr"C:\ESPI_TEMP\GPU_FULL\{tag}_PhaseOut_b18_cs16_ff100\qc_align_B_IRLS\metrics.csv")
    q = q.rename(columns={"name": "basename"})
    q["set"] = tag
    return q

# Load all QC data
Q = pd.concat([qc("W01"), qc("W02"), qc("W03")], ignore_index=True)

# Merge features with QC
DF = F.merge(Q[["set", "basename", "rmse", "pct_gt_pi2"]], on=["set", "basename"], how="left")

# Apply QC gate (same thresholds as W01)
gate = (DF["rmse"] <= 4.0) & (DF["pct_gt_pi2"] <= 50.0)
good = DF[gate].copy()

# Save QC-passed features
good.to_csv(base / "all_features_QCpass.csv", index=False)

print(f"[OK] QC-pass: {good.shape[0]} files")
print(f"QC retention by set:")
print(good.groupby("set").size())
