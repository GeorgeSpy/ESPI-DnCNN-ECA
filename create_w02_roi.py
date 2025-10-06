#!/usr/bin/env python3
import glob, numpy as np
from PIL import Image

# Get sample from W02 phase files
sample = sorted(glob.glob(r"C:\ESPI_TEMP\GPU_FULL\W02_PhaseOut_b18_cs16_ff100\phase_wrapped_npy\*.npy"))[0]
H, W = np.load(sample).shape

# Resize ROI to match W02 dimensions
roi = Image.open(r"C:\ESPI_TEMP\roi_mask.png").convert("L").resize((W, H), Image.NEAREST)
roi.save(r"C:\ESPI_TEMP\roi_mask_W02.png")

print(f"[OK] roi_mask_W02.png created with dimensions {H}x{W}")

