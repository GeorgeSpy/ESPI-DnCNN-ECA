import os, re, sys
from pathlib import Path
import numpy as np
from PIL import Image

tiles_dir = Path(r"C:\ESPI_TEMP\SMOKE\W01_CLEAN_0050")
ref_path  = Path(r"C:\ESPI\data\wood_Averaged\W01_ESPI_90db-Averaged\0050Hz_90.0db.png")
out_path  = Path(r"C:\ESPI_TEMP\SMOKE\W01_CLEAN_0050_STITCH\0050Hz_90.0db.png")
TILE      = 320
OVL       = 80
STRIDE    = TILE - OVL

# --- helpers ---
def load_u16(p):
    im = Image.open(p)
    arr = np.array(im, dtype=np.uint16)
    return arr

def hann2d(n):
    w = np.hanning(n)
    W = np.outer(w, w).astype(np.float32)
    return W

# --- read reference for target size ---
ref = load_u16(ref_path)
H, W = ref.shape
canvas = np.zeros((H, W), dtype=np.float64)
weight = np.zeros((H, W), dtype=np.float64)
Wmask  = hann2d(TILE)

# patterns: pixel coords or tile indices
pat_pix = re.compile(r'.*[_-]x(\d+)[_-]y(\d+)\.png$', re.IGNORECASE)
pat_idx = re.compile(r'.*[_-]tx(\d+)[_-]ty(\d+)\.png$', re.IGNORECASE)

tiles = sorted([p for p in tiles_dir.glob("*.png")])
if not tiles:
    raise SystemExit(f"No PNG tiles found in {tiles_dir}")

for p in tiles:
    name = p.name
    m = pat_pix.match(name)
    mode = "pix" if m else None
    if not m:
        m2 = pat_idx.match(name)
        if m2:
            mode = "idx"
            m = m2
    arr = load_u16(p).astype(np.float32)

    if arr.shape[0]!=TILE or arr.shape[1]!=TILE:
        # allow border tiles cropped by the writer
        th, tw = arr.shape
        wmask = hann2d(th) if th==tw else np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)
    else:
        th, tw = TILE, TILE
        wmask  = Wmask

    if mode=="pix":
        x0 = int(m.group(1))
        y0 = int(m.group(2))
    elif mode=="idx":
        tx = int(m.group(1))
        ty = int(m.group(2))
        x0 = tx * STRIDE
        y0 = ty * STRIDE
    else:
        # fallback: infer from sorted order row-major
        idx = tiles.index(p)
        tiles_per_row = int(np.ceil((W - TILE)/STRIDE)) + 1
        ty = idx // tiles_per_row
        tx = idx %  tiles_per_row
        x0 = tx * STRIDE
        y0 = ty * STRIDE

    x1 = min(x0 + tw, W)
    y1 = min(y0 + th, H)

    # crop if border exceeds canvas
    sub = arr[:(y1-y0), :(x1-x0)]
    wm  = wmask[:(y1-y0), :(x1-x0)]

    canvas[y0:y1, x0:x1] += (sub * wm)
    weight[y0:y1, x0:x1] += wm

# avoid div-by-zero στα κενά
mask = weight > 1e-8
canvas[mask] = canvas[mask] / weight[mask]
# ό,τι έμεινε κενό → από το ref (δεν θα έπρεπε, αλλά just in case)
canvas[~mask] = ref[~mask]

# κλιπάρισμα σε 16-bit
canvas = np.clip(np.rint(canvas), 0, 65535).astype(np.uint16)
Image.fromarray(canvas, mode="I;16").save(out_path)
print(f"[OK] Stitched: {out_path}  | size={canvas.shape[::-1]}")


