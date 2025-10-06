#!/usr/bin/env python3
import re, glob, numpy as np
from pathlib import Path
from PIL import Image

PH = Path(r"C:\ESPI_TEMP\GPU_FULL\W02_PhaseOut_b18_cs16_ff100")
RF = Path(r"C:\ESPI_TEMP\GPU_FULL\W02_PhaseRef_b18_cs16_ff100")
ROI = Path(r"C:\ESPI_TEMP\roi_mask.png")

def base_no_suffix(s): 
    return re.sub(r"_(\d+)$", "", s)

outs = sorted(PH.joinpath("phase_wrapped_npy").glob("*.npy"))
refs = {p.stem: p for p in RF.joinpath("phase_wrapped_npy").glob("*.npy")}

print(f"Found {len(outs)} output files")
print(f"Found {len(refs)} reference files")

picked = None
for p in outs:
    stem = p.stem
    base = base_no_suffix(stem)
    if base in refs:
        picked = (p, refs[base], stem, base)
        break

if picked is None:
    print("No matching name between OUT and REF. Check filenames.")
    print("Sample OUT names:", [p.stem for p in outs[:3]])
    print("Sample REF names:", list(refs.keys())[:3])
    exit(1)

p_out, p_ref, stem, base = picked
a = np.load(p_out)
b = np.load(p_ref)
roi = np.array(Image.open(ROI))
if roi.ndim == 3: 
    roi = roi[..., 0]

print("OUT:", p_out.name, a.shape)
print("REF:", p_ref.name, b.shape)
print("ROI:", roi.shape, "unique:", sorted(np.unique(roi))[:4])

print("OK_MATCH" if a.shape == b.shape else "SHAPE_MISMATCH")

