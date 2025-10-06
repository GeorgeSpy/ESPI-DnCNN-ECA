# Extract features for all frequencies in tetrachord bins
# Filters out _viz directories and processes only relevant frequencies

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

Write-Host "=== EXTRACTING FEATURES FOR TETRACHORD BINS ==="

foreach ($R in $ROOTS) {
  $setName = Split-Path $R -Leaf
  Write-Host "`nProcessing $setName..."
  
  Get-ChildItem -Path $R -Directory | Where-Object { $_.Name -notlike "*_viz" } | ForEach-Object {
    # Parse frequency from folder name e.g. 0155Hz_90.0db
    if ($_.Name -match '^0*(\d+)Hz') {
      $F = [int]$matches[1]
      
      # Keep ONLY tetrachord bins (wood) as defined:
      # mode_(1,1)H: 155-175Hz
      # mode_(1,1)T: 320-345Hz
      # mode_(1,2): 500-525Hz
      # mode_(2,1): 540-570Hz
      # mode_higher: 680+Hz
      $inBin = ($F -ge 155 -and $F -le 175) -or `
               ($F -ge 320 -and $F -le 345) -or `
               ($F -ge 500 -and $F -le 525) -or `
               ($F -ge 540 -and $F -le 570) -or `
               ($F -ge 680 -and $F -le 1500)
      
      if ($inBin) {
        $outFile = Join-Path $_.FullName "nodal_features.csv"
        
        # Skip if already exists
        if (Test-Path $outFile) {
          $skipped++
          Write-Host "  SKIP $($_.Name) (already exists)"
        } else {
          Write-Host "  PROC $($_.Name) ($F Hz)"
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
    }
  }
}

Write-Host "`n=== SUMMARY ==="
Write-Host "Processed: $processed"
Write-Host "Skipped: $skipped"
Write-Host "Errors: $errors"
Write-Host "Total: $($processed + $skipped + $errors)"

