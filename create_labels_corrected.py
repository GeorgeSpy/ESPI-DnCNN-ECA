#!/usr/bin/env python3
import re, pandas as pd, numpy as np
from pathlib import Path

SRC = Path(r"C:\ESPI_TEMP\features\all_features_QCpass.csv")
df = pd.read_csv(SRC)

# Use freq_hz column directly
df["freq"] = df["freq_hz"]

# Get nodal columns
COL_H = "chg_h" if "chg_h" in df.columns else None
COL_V = "chg_v" if "chg_v" in df.columns else None

h = df[COL_H].astype(float) if COL_H else pd.Series(np.nan, index=df.index)
v = df[COL_V].astype(float) if COL_V else pd.Series(np.nan, index=df.index)

print(f"Using nodal columns: H={COL_H}, V={COL_V}")
print(f"Frequency range: {df['freq'].min()} - {df['freq'].max()} Hz")

def label_fine(f, h, v):
    if np.isnan(f): 
        return "unlabeled"
    
    # Adjusted frequency ranges for our actual data (40-295 Hz)
    if 40 <= f <= 50: 
        return "low_freq"  # 40-50 Hz range
    if 50 < f <= 60: 
        return "mid_freq"  # 50-60 Hz range  
    if 60 < f <= 100: 
        return "high_freq"  # 60-100 Hz range
    if 100 < f <= 200: 
        return "very_high_freq"  # 100-200 Hz range
    if 200 < f <= 300: 
        return "ultra_high_freq"  # 200-300 Hz range
    
    return "unlabeled"

def label_grouped(f, h, v):
    if np.isnan(f): 
        return "UNLABELED"
    
    # Create 5 classes based on our actual frequency distribution
    if 40 <= f <= 50: 
        return "LOW"      # 40-50 Hz
    if 50 < f <= 60: 
        return "MID"      # 50-60 Hz  
    if 60 < f <= 100: 
        return "HIGH"     # 60-100 Hz
    if 100 < f <= 200: 
        return "VHIGH"    # 100-200 Hz
    if 200 < f <= 300: 
        return "UHIGH"    # 200-300 Hz
    
    return "UNLABELED"

fine = [label_fine(f,hv,vv) for f,hv,vv in zip(df["freq"], h, v)]
grouped = [label_grouped(f,hv,vv) for f,hv,vv in zip(df["freq"], h, v)]

df["label_fine"] = fine
df["label_grouped"] = grouped

# exports
out_dir = Path(r"C:\ESPI_TEMP\features")
fine_out = out_dir/"labels_modes_fine.csv"
grp_out = out_dir/"labels_modes_grouped.csv"

cols = ["set","basename","freq","label_fine"]
df[cols].to_csv(fine_out, index=False)
df[["set","basename","freq","label_grouped"]].rename(columns={"label_grouped":"label"}).to_csv(grp_out, index=False)

print(f"[OK] wrote: {fine_out}, rows: {len(df)}")
print(f"[OK] wrote: {grp_out}, rows: {len(df)}")

# Show label distribution
print("\nLabel distribution (fine):")
print(df["label_fine"].value_counts())
print("\nLabel distribution (grouped):")
print(df["label_grouped"].value_counts())

