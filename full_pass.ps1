# --- Paths & Preset ---
$PY = "C:\ESPI_VENV2\Scripts\python.exe"
$COMPAT = "C:\ESPI_DnCNN\denoise_debug_dualmode_COMPAT.py"
$CKPT   = "C:\ESPI_TEMP\denoise_finetune_REAL+PSEUDO_SMALL\checkpoints\best_finetune.pth"
$CKPT2  = "C:\ESPI_TEMP\denoise_finetune_REAL+PSEUDO_SMALL\checkpoints\best.pth"
$ROI    = "C:\ESPI_TEMP\roi_mask.png"
$ANN    = "8 300"
$BAND   = 18; $CS=16; $FF=100
$TILE=320; $OVL=80

# Instruments & roots
$SETS = @(
  @{ name="W01"; real="C:\ESPI\data\wood_real_A\W01_ESPI_90db"; avg="C:\ESPI\data\wood_Averaged\W01_ESPI_90db-Averaged" },
  @{ name="W02"; real="C:\ESPI\data\wood_real_B\W02_ESPI_90db"; avg="C:\ESPI\data\wood_Averaged\W02_ESPI_90db-Averaged" },
  @{ name="W03"; real="C:\ESPI\data\wood_real_C\W03_ESPI_90db"; avg="C:\ESPI\data\wood_Averaged\W03_ESPI_90db-Averaged" }
)

# 0) 16-bit FFT helper: Βεβαιώσου ότι το fft_peak_probe.py ΔΕΝ κάνει convert("L")

foreach ($S in $SETS) {
  $N = $S.name; $REAL = $S.real; $AVG = $S.avg
  Write-Host "`n=== [$N] ===" -ForegroundColor Cyan

  # 1) Denoise όλων των frequency folders
  $CLEAN = "C:\ESPI_TEMP\FULL\$N`_CLEAN_u16"
  $RESNP = "C:\ESPI_TEMP\FULL\$N`_RES"
  New-Item -ItemType Directory -Force -Path $CLEAN,$RESNP | Out-Null

  Get-ChildItem -Path $REAL -Directory | ForEach-Object {
    if (Test-Path $CKPT) { $USE_CKPT = $CKPT } else { $USE_CKPT = $CKPT2 }
    & $PY C:\ESPI_DnCNN\batch_denoise_from_compat_NORM.py `
      --ckpt $USE_CKPT `
      --input  $_.FullName `
      --output $CLEAN `
      --tile $TILE --overlap $OVL --device cpu `
      --predicts-residual --norm-mode u16 --save-u16
  }

  # 2) Phase (Preset A)
  $PHASE = "C:\ESPI_TEMP\FULL\$N`_PhaseOut_b$BAND`_cs$CS`_ff$FF"
  & $PY C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py `
    --input-dir  $CLEAN `
    --output-dir $PHASE `
    --band $BAND --center-suppress $CS --flatfield $FF --annulus $ANN `
    --roi-mask $ROI --unwrap auto

  # 3) Reference Phase (αν δεν υπάρχει)
  $PHASE_REF = "C:\ESPI_TEMP\FULL\$N`_PhaseRef_b$BAND`_cs$CS`_ff$FF"
  if (!(Test-Path $PHASE_REF)) {
    & $PY C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py `
      --input-dir  $AVG `
      --output-dir $PHASE_REF `
      --band $BAND --center-suppress $CS --flatfield $FF --annulus $ANN `
      --roi-mask $ROI --unwrap auto
  }

  # 4) QC (wrapped) για όλες τις εικόνες
  & $PY C:\ESPI_DnCNN\phase_qc_compare_two_roots.py `
    --out-root $PHASE `
    --ref-root $PHASE_REF `
    --roi-mask $ROI --qmin 0.10 --save-maps

  # 5) Robust unwrapped alignment (Strategy B) batch → metrics.csv
  $ALIGN_OUT = "$PHASE\qc_align_B"
  New-Item -ItemType Directory -Force -Path $ALIGN_OUT | Out-Null

  $code = @'
import os, json, numpy as np
from pathlib import Path

PHASE   = Path(r"__PHASE__")
PHASE_R = Path(r"__PHASE_REF__")
ALIGN_O = Path(r"__ALIGN_OUT__")
ROI_P   = Path(r"__ROI__")

import csv
from PIL import Image
pi = np.pi

def load_npy(root, sub, name): return np.load(root/sub/(name+".npy")).astype(np.float32)
def roi_mask(p):
    m = np.array(Image.open(p)); 
    if m.ndim==3: m=m[...,0]; 
    return m>0

roi = roi_mask(ROI_P)
names = [p.stem for p in (PHASE/"phase_wrapped_npy").glob("*.npy")]
out_csv = ALIGN_O/"metrics.csv"
ALIGN_O.mkdir(parents=True, exist_ok=True)

def unwrap2d(a):
    a1 = np.unwrap(a, axis=1); a2 = np.unwrap(a1, axis=0); return a2.astype(np.float32)

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name","rmse","pct_gt_pi2","pct_gt_pi4"])
    for nm in names:
        w_test = load_npy(PHASE, "phase_wrapped_npy", nm)
        w_ref  = load_npy(PHASE_R, "phase_wrapped_npy", nm)

        d_wrapped = np.angle(np.exp(1j*(w_test - w_ref))).astype(np.float32)
        d_unw = unwrap2d(d_wrapped)
        # plane removal
        H,W = d_unw.shape
        yy,xx = np.mgrid[0:H,0:W]
        X = np.stack([xx[roi], yy[roi], np.ones(np.count_nonzero(roi), np.float32)], axis=1)
        y = d_unw[roi][:,None]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a,b,c = coef.ravel().astype(np.float32)
        plane = (a*xx + b*yy + c).astype(np.float32)
        d_unw2 = d_unw - plane

        dd = d_unw2[roi]
        rmse = float(np.sqrt(np.mean(dd*dd)))
        pct2 = float(100.0*np.mean(np.abs(dd) > (pi/2)))
        pct4 = float(100.0*np.mean(np.abs(dd) > (pi/4)))
        w.writerow([nm, f"{rmse:.4f}", f"{pct2:.2f}", f"{pct4:.2f}"])
print("[OK] wrote", out_csv)
'@
  $code = $code.Replace("__PHASE__", $PHASE).Replace("__PHASE_REF__", $PHASE_REF).Replace("__ALIGN_OUT__", $ALIGN_OUT).Replace("__ROI__", $ROI)
  & $PY -c $code
}
