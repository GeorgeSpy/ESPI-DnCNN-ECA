# Bulletproof Phase Extraction with lock/stamp/resume
$PY  = "C:\ESPI_VENV2\Scripts\python.exe"
$ROI = "C:\ESPI_TEMP\roi_mask.png"
$ENV = @{ OMP_NUM_THREADS="1"; MKL_NUM_THREADS="1"; OPENBLAS_NUM_THREADS="1" }

$SETS = @(
  @{IN="C:\ESPI_TEMP\GPU_FULL2\W02_CLEAN_u16"; OUT="C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100"},
  @{IN="C:\ESPI_TEMP\GPU_FULL2\W03_CLEAN_u16"; OUT="C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"}
)

function Is-PhaseDone($dst){
  (Test-Path (Join-Path $dst "phase_unwrapped_npy")) -and `
  -not (Test-Path (Join-Path $dst ".phase.lock"))
}

foreach($S in $SETS){
  $in=$S.IN; $out=$S.OUT
  if(!(Test-Path $out)){ New-Item -ItemType Directory -Path $out | Out-Null }

  # input dirs: no *_viz
  $freqs = Get-ChildItem -Directory $in | Where-Object { $_.Name -notlike "*_viz" }

  foreach($f in $freqs){
    $dst = Join-Path $out $f.Name
    if(!(Test-Path $dst)){ New-Item -ItemType Directory -Path $dst | Out-Null }

    # output dirs: skip qc_*, *_viz, debug*
    if($f.Name -like "qc_*" -or $f.Name -like "*_viz" -or $f.Name -like "debug*"){ continue }

    if(Is-PhaseDone $dst){ continue }

    # lock so it won't be picked again
    New-Item -ItemType File -Path (Join-Path $dst ".phase.lock") -Force | Out-Null

    Write-Host "[PHASE] $($f.Name)"
    & $PY C:\ESPI_DnCNN\phase_extract_fft_STRICT_FIXED.py `
      --input-dir  $f.FullName `
      --output-dir $dst `
      --band 18 --center-suppress 16 --flatfield 100 --annulus 8 300 `
      --roi-mask $ROI --unwrap auto

    if($LASTEXITCODE -eq 0){
      # success stamp
      Set-Content -Path (Join-Path $dst ".phase.done") -Value (Get-Date -Format s)
    }
    Remove-Item (Join-Path $dst ".phase.lock") -ErrorAction SilentlyContinue
  }
}

Write-Host "[DONE] Phase extraction complete"

