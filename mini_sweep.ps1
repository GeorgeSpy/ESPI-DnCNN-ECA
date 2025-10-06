# mini_sweep.ps1 - Mini-sweep για βελτίωση QC
$in  = "C:\ESPI_TEMP\SMOKE\W01_CLEAN"
$out = "C:\ESPI_TEMP\SMOKE\W01_PhaseOut_SWEEP"
$roi = "C:\ESPI_TEMP\roi_mask.png"
$ref = "C:\Users\Geiorge HTPC\Desktop\Mpouzoukia\W01_PhaseOut_masked_grid\band18"

$cs = 16; $ann_rmin=8; $ann_rmax=300
$bands = 18
$flatfields = 80,100,120
$centers = 14,16,18

foreach ($ff in $flatfields) {
  foreach ($c in $centers) {
    $od = Join-Path $out ("b{0}_cs{1}_ff{2}_c{3}" -f $bands,$cs,$ff,$c)
    & C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py `
      --input-dir $in --output-dir $od `
      --band $bands --center-suppress $c --flatfield $ff --annulus $ann_rmin $ann_rmax `
      --roi-mask $roi --unwrap auto

    & C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\phase_qc_compare_two_roots.py `
      --out-root $od --ref-root $ref --roi-mask $roi --qmin 0.20
  }
}


