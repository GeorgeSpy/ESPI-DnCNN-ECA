# Extract features for ALL frequencies with valid phase data
# No frequency filtering - process everything

$PY = "C:\ESPI_VENV2\Scripts\python.exe"
$FE = "C:\ESPI_DnCNN\phase_nodal_features_min.py"
$ROI = "C:\ESPI_TEMP\roi_mask.png"
$ROOTS = @(
  "C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100",
  "C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100",
  "C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"
)

$processed = 0
$skipped = 0
$errors = 0

Write-Host "=== EXTRACTING FEATURES FOR ALL FREQUENCIES ==="

foreach ($R in $ROOTS) {
  if (!(Test-Path $R)) { continue }
  
  $setName = Split-Path $R -Leaf
  Write-Host "`nProcessing $setName..."
  
  Get-ChildItem -Path $R -Directory | Where-Object { $_.Name -notlike "*_viz" } | ForEach-Object {
    $outFile = Join-Path $_.FullName "nodal_features.csv"
    
    # Skip if already exists
    if (Test-Path $outFile) {
      $skipped++
      return
    }
    
    # Check if phase_unwrapped_npy directory exists with valid files
    $phaseDir = Join-Path $_.FullName "phase_unwrapped_npy"
    if (!(Test-Path $phaseDir)) {
      return
    }
    
    $phaseFiles = Get-ChildItem $phaseDir\*.npy 2>$null
    if (!$phaseFiles -or $phaseFiles.Count -eq 0) {
      return
    }
    
    # Check if at least one file is valid (>1KB)
    $validFile = $phaseFiles | Where-Object { $_.Length -gt 1000 } | Select-Object -First 1
    if (!$validFile) {
      return
    }
    
    Write-Host "  PROC $($_.Name)"
    try {
      & $PY $FE `
        --band-root $_.FullName `
        --roi-mask $ROI `
        --csv-out $outFile 2>&1 | Out-Null
      
      if (Test-Path $outFile) {
        $processed++
      } else {
        $errors++
        Write-Host "    ERROR: Output file not created"
      }
    } catch {
      $errors++
      Write-Host "    ERROR: $($_.Exception.Message)"
    }
  }
}

Write-Host "`n=== SUMMARY ==="
Write-Host "Processed: $processed"
Write-Host "Skipped: $skipped"
Write-Host "Errors: $errors"
Write-Host "Total: $($processed + $skipped + $errors)"

