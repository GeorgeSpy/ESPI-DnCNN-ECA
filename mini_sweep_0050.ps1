$in  = "C:\ESPI_TEMP\SMOKE\W01_CLEAN_0050_FULL"
$out = "C:\ESPI_TEMP\SMOKE\W01_PhaseOut_SWEEP_0050"
$roi = "C:\ESPI_TEMP\roi_mask.png"
$ref = "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_PhaseOut_masked_grid\band18"
$bands = 17,18,19
$ffs   = 80,100,120
$css   = 14,16,18

foreach ($b in $bands) {
  foreach ($ff in $ffs) {
    foreach ($cs in $css) {
      $od = Join-Path $out ("b{0}_cs{1}_ff{2}" -f $b,$cs,$ff)
      Write-Host "Processing: band=$b, center_suppress=$cs, flatfield=$ff" -ForegroundColor Yellow
      
      & C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py `
        --input-dir $in --output-dir $od `
        --band $b --center-suppress $cs --flatfield $ff --annulus 8 300 `
        --roi-mask $roi --unwrap auto

      & C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\phase_qc_compare_two_roots.py `
        --out-root $od --ref-root $ref --roi-mask $roi --qmin 0.10
    }
  }
}

Write-Host "Sweep completed!" -ForegroundColor Green


