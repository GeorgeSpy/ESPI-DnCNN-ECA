# watch_smoke.ps1
$ckpt = "C:\ESPI_TEMP\denoise_finetune_REAL+PSEUDO_SMALL\checkpoints\best_finetune.pth"
while (!(Test-Path $ckpt)) {
  Write-Host "[watch] Περιμένω checkpoint..." -ForegroundColor Yellow
  Start-Sleep -Seconds 20
}
Write-Host "[watch] Βρέθηκε checkpoint! Τρέχω SMOKE QA..." -ForegroundColor Green

# 1) Denoise (0050 Hz)
& C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\denoise_folder_with_compat_ckpt.py `
  --compat "C:\ESPI_DnCNN\denoise_debug_dualmode_COMPAT.py" `
  --ckpt   $ckpt `
  --input  "C:\ESPI\data\wood_real_A\W01_ESPI_90db\0050Hz_90.0db" `
  --out-clean "C:\ESPI_TEMP\SMOKE\W01_CLEAN" `
  --out-resnp "C:\ESPI_TEMP\SMOKE\W01_RES" `
  --tile 320 --overlap 80 --device cpu

# 2) FFT-probe (avg vs clean)
& C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\fft_peak_probe.py `
  --img "C:\ESPI\data\wood_Averaged\W01_ESPI_90db-Averaged\0050Hz_90.0db.png" --cs 16 --rmin 8 --rmax 300
& C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\fft_peak_probe.py `
  --img "C:\ESPI_TEMP\SMOKE\W01_CLEAN\0050Hz_90.0db.png" --cs 16 --rmin 8 --rmax 300

# 3) Phase + QC (Preset A)
& C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py `
  --input-dir  "C:\ESPI_TEMP\SMOKE\W01_CLEAN" `
  --output-dir "C:\ESPI_TEMP\SMOKE\W01_PhaseOut_DENFT_b18_cs16_ff100" `
  --band 18 --center-suppress 16 --flatfield 100 --annulus 8 300 `
  --roi-mask "C:\ESPI_TEMP\roi_mask.png" --unwrap auto

& C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\phase_qc_compare_two_roots.py `
  --out-root "C:\ESPI_TEMP\SMOKE\W01_PhaseOut_DENFT_b18_cs16_ff100" `
  --ref-root "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_PhaseOut_masked_grid\band18" `
  --roi-mask "C:\ESPI_TEMP\roi_mask.png" --qmin 0.20 --save-maps

Write-Host "[watch] Ολοκληρώθηκε το SMOKE QA. Έτοιμος για νούμερα." -ForegroundColor Cyan


