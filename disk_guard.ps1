# Disk guard (safe cleanup & compression)
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 2)

Write-Host "=== DISK GUARD ==="
Write-Host "C: Free: $freeGB GB"

if($freeGB -lt 8){
  Write-Host "LOW DISK SPACE - Running cleanup..." -ForegroundColor Yellow
  
  # 1) Delete SMOKE directories
  Write-Host "Cleaning SMOKE directories..."
  Get-ChildItem "C:\ESPI_TEMP\SMOKE*" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Removing $($_.Name)"
    Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
  }
  
  # 2) Keep only best.pth and last.pth in checkpoints
  Write-Host "Cleaning checkpoint directories..."
  Get-ChildItem "C:\ESPI_TEMP\*\checkpoints" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $ckptDir = $_.FullName
    Get-ChildItem $ckptDir -Filter "*.pth" | Where-Object {
      $_.Name -ne "best.pth" -and $_.Name -ne "last.pth"
    } | ForEach-Object {
      Write-Host "  Removing $($_.Name)"
      Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    }
  }
  
  # 3) Delete visualization directories (_viz)
  Write-Host "Cleaning visualization directories..."
  Get-ChildItem "C:\ESPI_TEMP\GPU_FULL2\*_viz" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Removing $($_.Name)"
    Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
  }
  
  # 4) Delete debug directories
  Write-Host "Cleaning debug directories..."
  Get-ChildItem "C:\ESPI_TEMP\GPU_FULL2\*\debug" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Removing $($_.FullName)"
    Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
  }
  
  # 5) Delete QC residual maps (keep only summary files)
  Write-Host "Cleaning QC residual maps..."
  Get-ChildItem "C:\ESPI_TEMP\GPU_FULL2\*\qc_residual_maps_vs_ref" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $qcDir = $_.FullName
    # Keep only CSV and JSON files
    Get-ChildItem $qcDir -Recurse -File | Where-Object {
      $_.Extension -ne ".csv" -and $_.Extension -ne ".json"
    } | ForEach-Object {
      Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    }
  }
  
  $newFreeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
  $freedGB = [math]::Round($newFreeGB - $freeGB, 2)
  Write-Host "Cleanup complete. Freed: $freedGB GB. New free space: $newFreeGB GB" -ForegroundColor Green
  
  if($newFreeGB -lt 8){
    Write-Host "WARNING: Still low on disk space. Consider:" -ForegroundColor Yellow
    Write-Host "  1) Moving CLEAN_u16 directories to D:"
    Write-Host "  2) Compressing NPY files with NTFS compression"
    Write-Host "  3) Archiving old training runs"
  }
} else {
  Write-Host "Disk space OK ($freeGB GB free)" -ForegroundColor Green
}

# Optional: NTFS compression for NPY files (run if needed)
# compact /c /s:"C:\ESPI_TEMP\GPU_FULL2\W0*_PhaseOut_b18_cs16_ff100\*\phase_*_npy" /i
# compact /c /s:"C:\ESPI_TEMP\GPU_FULL2\W0*_PhaseOut_b18_cs16_ff100\*\qc_align_B_IRLS" /i

