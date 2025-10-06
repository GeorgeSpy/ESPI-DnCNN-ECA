# -*- coding: utf-8 -*-
import os, glob
import numpy as np
from pathlib import Path
from PIL import Image

out_root = r"C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_PhaseOut_STRICT_DEN_CLEAN"
png_dir  = Path(out_root) / "phase_unwrapped_png"
npy_dir  = Path(out_root) / "phase_unwrapped_npy"
npy_dir.mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(str(png_dir / "*.png")))
if not files:
    print("[ERR] No PNGs found in:", png_dir)
    raise SystemExit(1)

for fp in files:
    name = Path(fp).stem  # π.χ. 0100Hz_90.0db
    img  = Image.open(fp).convert("F")  # float32
    arr  = np.array(img, dtype=np.float32)
    # προσοχή: εδώ υποθέτουμε ότι τα PNG είχαν ήδη κλίμακα σε radians
    # (αν ήταν οπτικοποίηση, δεν είναι σωστό — αλλά στα προηγούμενα scripts ήταν πραγματικές φάσεις)
    np.save(npy_dir / f"{name}.npy", arr)

print(f"[OK] Wrote {len(files)} NPY to", npy_dir)
