# Harvester for QC→IRLS→Features as phases complete
$PY="C:\ESPI_VENV2\Scripts\python.exe"
$R01="C:\ESPI_TEMP\GPU_FULL2\W01_PhaseRef_b18_cs16_ff100"
$R02="C:\ESPI_TEMP\GPU_FULL2\W02_PhaseRef_b18_cs16_ff100"
$R03="C:\ESPI_TEMP\GPU_FULL2\W03_PhaseRef_b18_cs16_ff100"
$O01="C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100"
$O02="C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100"
$O03="C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"
$ROI_W01="C:\ESPI_TEMP\roi_mask_W01.png"
$ROI_W02="C:\ESPI_TEMP\roi_mask_W02.png"
$ROI_W03="C:\ESPI_TEMP\roi_mask_W03.png"
$ROI_DEFAULT="C:\ESPI_TEMP\roi_mask.png"

function Harvest-Set($OUT,$REF,$ROI){
  if(!(Test-Path $ROI)){ $ROI = $ROI_DEFAULT }
  
  $freqs = Get-ChildItem -Directory $OUT | Where-Object {
    (Test-Path (Join-Path $_.FullName "phase_unwrapped_npy")) -and `
    -not (Test-Path (Join-Path $_.FullName ".qc.done"))
  }
  
  Write-Host "Found $($freqs.Count) frequencies ready for QC/IRLS/Features"
  
  foreach($f in $freqs){
    $refFreq = $f.Name
    $r = Join-Path $REF "phase_wrapped_npy"
    $refFile = Join-Path $r "$refFreq.npy"
    
    if(Test-Path $refFile){
      Write-Host "[QC/IRLS/FEAT] Processing $($f.Name)"
      
      # QC (wrapped) - per frequency
      $wrappedDir = Join-Path $f.FullName "phase_wrapped_npy"
      if(Test-Path $wrappedDir){
        try {
          & $PY C:\ESPI_DnCNN\phase_qc_compare_two_roots.py `
            --out-root $f.FullName `
            --ref-root $REF `
            --roi-mask $ROI --qmin 0.10 --save-maps
          Write-Host "  QC: OK"
        } catch {
          Write-Host "  QC: SKIP (error)"
        }
      }

      # IRLS (memory-safe)
      try {
        & $PY C:\ESPI_DnCNN\simple_irls_alignment.py `
          $wrappedDir `
          $r `
          $ROI `
          (Join-Path $f.FullName "qc_align_B_IRLS")
        Write-Host "  IRLS: OK"
      } catch {
        Write-Host "  IRLS: SKIP (error)"
      }

      # Features
      try {
        & $PY C:\ESPI_DnCNN\phase_nodal_features_min.py --band-root $f.FullName
        Write-Host "  Features: OK"
      } catch {
        Write-Host "  Features: SKIP (error)"
      }

      Set-Content -Path (Join-Path $f.FullName ".qc.done") -Value (Get-Date -Format s)
      Write-Host "  [DONE] $($f.Name)"
    } else {
      Write-Host "  SKIP $($f.Name) (no reference)"
    }
  }
}

Write-Host "=== HARVESTER START ==="
Harvest-Set $O01 $R01 $ROI_W01
Harvest-Set $O02 $R02 $ROI_W02
Harvest-Set $O03 $R03 $ROI_W03
Write-Host "=== HARVESTER DONE ==="

