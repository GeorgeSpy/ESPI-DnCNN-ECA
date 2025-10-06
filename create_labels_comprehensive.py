#!/usr/bin/env python3
import re, pandas as pd, numpy as np
from pathlib import Path

SRC = Path(r"C:\ESPI_TEMP\features\all_features_QCpass.csv")
df = pd.read_csv(SRC)

# --- helpers ---
def get_freq(s):
    # πιάνει 3-4 ψηφία πριν από 'Hz'
    m = re.search(r'(\d{3,4})Hz', s)
    return float(m.group(1)) if m else np.nan

df["freq"] = df["basename"].astype(str).map(get_freq)

# προσπάθεια να εντοπίσουμε στήλες με κομβικές γραμμές (δεν είναι υποχρεωτικό)
def pick_col(cands):
    for c in cands:
        if c in df.columns: 
            return c
    return None

COL_H = pick_col(["n_lines_h","nodal_h","n_nodal_h","count_h","nodal_lines_h","chg_h"])
COL_V = pick_col(["n_lines_v","nodal_v","n_nodal_v","count_v","nodal_lines_v","chg_v"])

h = df[COL_H].astype(float) if COL_H else pd.Series(np.nan, index=df.index)
v = df[COL_V].astype(float) if COL_V else pd.Series(np.nan, index=df.index)

print(f"Using nodal columns: H={COL_H}, V={COL_V}")

def label_fine(f, h, v):
    if np.isnan(f): 
        return "unlabeled"
    # windows
    if 150<=f<=180: 
        return "(1,1)H"
    if 270<=f<=310: 
        return "(1,1)T"
    if 360<=f<=420: 
        # potential overlap with 2,1 near 420–430 handled below
        return "(1,2)"
    if 430<=f<=470: 
        return "(2,1)"
    if 680<=f<=705: 
        # (1,3) zone; tie-break vs (3,1) only in 701–704
        if 701<=f<=704 and not np.isnan(h) and not np.isnan(v):
            if (h+1e-6)/(v+1e-6) >= 1.3: 
                return "(1,3)"
            if (v+1e-6)/(h+1e-6) >= 1.3: 
                return "(3,1)"  # σπάνιο εδώ
            return "ambiguous_7xx"
        return "(1,3)"
    if 705<=f<=740:
        # (3,1); tie-break προς (1,3) αν 705–708 και features δείχνουν οριζόντιες
        if 705<=f<=708 and not np.isnan(h) and not np.isnan(v):
            if (h+1e-6)/(v+1e-6) >= 1.3: 
                return "(1,3)"
        return "(3,1)"
    if 750<=f<=775: 
        # (3,3) αδύναμος
        # αν υπάρχουν 2x2 γραμμές, κράτα (3,3), αλλιώς άφησέ το ως TRI_7xx downstream
        if not np.isnan(h) and not np.isnan(v) and h>=2 and v>=2:
            return "(3,3)"
        return "(3,3)"
    if 800<=f<=2000:
        return "HiModes"
    # overlap (1,2) vs (2,1) στην "γέφυρα" 420–430
    if 420<f<430:
        if not np.isnan(h) and not np.isnan(v):
            if (h+1e-6)/(v+1e-6) >= 1.3: 
                return "(1,2)"
            if (v+1e-6)/(h+1e-6) >= 1.3: 
                return "(2,1)"
        return "ambiguous_4xx"
    return "unlabeled"

fine = [label_fine(f,hv,vv) for f,hv,vv in zip(df["freq"], h, v)]
df["label_fine"] = fine

# grouped (6 classes by default)
def to_group(lbl):
    if lbl=="(1,1)H": 
        return "H"
    if lbl=="(1,1)T": 
        return "T"
    if lbl=="(1,2)":  
        return "DIP_X"
    if lbl=="(2,1)":  
        return "DIP_Y"
    if lbl in {"(1,3)","(3,1)","(3,3)","ambiguous_7xx"}: 
        return "TRI_7xx"
    if lbl=="HiModes": 
        return "HIMODES"
    return "UNLABELED"

df["label_group6"] = df["label_fine"].map(to_group)

# exports
out_dir = Path(r"C:\ESPI_TEMP\features")
fine_out = out_dir/"labels_modes_fine.csv"
grp_out = out_dir/"labels_modes_grouped.csv"
cols = ["set","basename","freq","label_fine"]
df[cols].to_csv(fine_out, index=False)
df[["set","basename","freq","label_group6"]].rename(columns={"label_group6":"label"}).to_csv(grp_out, index=False)

print(f"[OK] wrote: {fine_out}, rows: {len(df)}")
print(f"[OK] wrote: {grp_out}, rows: {len(df)}")

# Show label distribution
print("\nLabel distribution (fine):")
print(df["label_fine"].value_counts())
print("\nLabel distribution (grouped):")
print(df["label_group6"].value_counts())

